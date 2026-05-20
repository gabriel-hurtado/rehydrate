#!/usr/bin/env python3
"""Recover and judge one compaction moment from an authorized rollout."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent

import baseline_helpers as bh
import rehydrate_compaction as rc


DEFAULT_PRIVATE_SNAPSHOT = (
    Path.home()
    / ".codex/sessions/2026/05/20/"
    / "rollout-2026-05-20T16-01-40-019e45b1-2dc0-7ea3-b7b5-694ac2f586e9.jsonl"
)
DEFAULT_COMPACTION_LINE = 912


class JudgedFact(BaseModel):
    fact_id: str
    fact: str
    importance: Literal["high", "medium", "low"]
    source_refs: list[str] = Field(default_factory=list)
    raw_line_ids: list[int] = Field(default_factory=list)
    survival: Literal["preserved", "partial", "lost"]
    evidence: str


class MomentJudgeResult(BaseModel):
    score_0_to_10: int = Field(ge=0, le=10)
    verdict: Literal["good", "mixed", "poor"]
    preserved_count: int = Field(ge=0)
    partial_count: int = Field(ge=0)
    missed_count: int = Field(ge=0)
    facts: list[JudgedFact] = Field(default_factory=list)
    missed_facts: list[JudgedFact] = Field(default_factory=list)
    rationale: str


def parse_timestamp(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def case_timestamp(case: dict[str, Any]) -> datetime | None:
    value = case.get("timestamp")
    if not isinstance(value, str):
        return None
    try:
        return parse_timestamp(value)
    except ValueError:
        return None


def select_case(
    cases: list[dict[str, Any]],
    *,
    line: int | None,
    at: str | None,
) -> dict[str, Any]:
    if not cases:
        raise RuntimeError("no compaction cases found in snapshot")

    if line is not None:
        for case in cases:
            if int(case["compaction_line"]) == line:
                return case
        raise RuntimeError(f"no compaction case found at line {line}")

    if at is not None:
        target = parse_timestamp(at)
        dated = [
            (timestamp, case)
            for case in cases
            if (timestamp := case_timestamp(case)) is not None
        ]
        before = [(timestamp, case) for timestamp, case in dated if timestamp <= target]
        if before:
            return max(before, key=lambda item: item[0])[1]
        if dated:
            return min(dated, key=lambda item: abs((item[0] - target).total_seconds()))[1]
        raise RuntimeError("snapshot compaction cases do not have parseable timestamps")

    return cases[-1]


def find_compaction_item_for_case(
    profile: dict[str, Any],
    case: dict[str, Any],
) -> dict[str, Any]:
    compaction_line = int(case["compaction_line"])
    candidates = [
        recovery_case
        for recovery_case in bh.compaction_server_recovery_cases(profile)
        if int(recovery_case["compaction_line"]) == compaction_line
        and isinstance(recovery_case.get("compaction_item"), dict)
    ]
    if not candidates:
        raise RuntimeError(
            f"compaction line {compaction_line} has no full encrypted_content artifact"
        )
    selected = candidates[-1]
    return {
        "source": selected["source_ref"],
        "source_kind": "snapshot",
        "compaction_line": selected["compaction_line"],
        "replacement_history_index": selected["replacement_history_index"],
        "item": selected["compaction_item"],
        "metadata": rc.compaction_item_metadata(selected["compaction_item"]),
    }


def recover_case(
    profile: dict[str, Any],
    case: dict[str, Any],
    *,
    model: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate = find_compaction_item_for_case(profile, case)
    recovery = rc.recover_compacted_message(item=candidate["item"], model=model)
    recovery = {
        **recovery,
        "candidate_source": candidate["source"],
        "candidate_source_kind": candidate["source_kind"],
        "candidate_compaction_line": candidate["compaction_line"],
        "candidate_replacement_history_index": candidate["replacement_history_index"],
        "artifact": candidate["metadata"],
    }
    if recovery.get("status") != "ok":
        raise RuntimeError(f"compaction recovery failed: {recovery.get('error')}")
    recovered = recovery.get("result") or {}
    if recovered.get("recovered") is not True:
        raise RuntimeError(
            "compaction artifact was accepted but no compacted message was recovered"
        )
    return bh.compaction_case_with_recovered_message(case, recovery), recovery


def pydantic_ai_model_name(model: str) -> str:
    stripped = model.strip()
    if ":" in stripped:
        return stripped
    return f"openai-responses:{stripped}"


def usage_payload(result: Any) -> dict[str, Any] | None:
    usage = getattr(result, "usage", None)
    if usage is None:
        return None
    if callable(usage):
        usage = usage()
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if dataclasses.is_dataclass(usage):
        return dataclasses.asdict(usage)
    return {"repr": repr(usage)}


def judge_case_with_pydantic_ai(
    case: dict[str, Any],
    *,
    model: str,
) -> dict[str, Any]:
    bh.load_project_env()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")

    judge_input = bh.compaction_survival_judge_input(case)
    judge_input["output_contract"] = {
        "score_0_to_10": "0 means the compacted message lost all important source facts; 10 means the important source facts survived.",
        "missed_facts": "Only include source facts judged as lost.",
    }
    agent = Agent(
        pydantic_ai_model_name(model),
        system_prompt=(
            "You are a strict compaction-quality judge. Compare one source "
            "replacement_history against one recovered compacted message. Extract "
            "only explicit source facts. Report missed facts when important facts "
            "are absent from the compacted message."
        ),
        output_type=MomentJudgeResult,
    )
    result = agent.run_sync(json.dumps(judge_input, indent=2, sort_keys=True))
    return {
        "status": "ok",
        "model": pydantic_ai_model_name(model),
        "result": result.output.model_dump(),
        "usage": usage_payload(result),
        "openai_call_count": 1,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    snapshot = args.snapshot.expanduser().resolve()
    if not snapshot.exists():
        raise RuntimeError(f"snapshot does not exist: {snapshot}")

    profile = bh.profile_rollout(snapshot)
    cases = bh.compaction_survival_cases(profile)
    case = select_case(cases, line=args.line, at=args.at)
    recovered_case, recovery = recover_case(profile, case, model=args.model)
    judge = judge_case_with_pydantic_ai(recovered_case, model=args.model)
    return {
        "status": "ok",
        "snapshot": str(snapshot),
        "case": {
            "case_id": recovered_case["case_id"],
            "compaction_line": recovered_case["compaction_line"],
            "timestamp": recovered_case.get("timestamp"),
            "replacement_history_items": len(recovered_case["source"]["items"]),
        },
        "recovery": {
            "status": recovery.get("status"),
            "response_id": recovery.get("response_id"),
            "model": recovery.get("model"),
            "confidence": (recovery.get("result") or {}).get("confidence"),
            "artifact": recovery.get("artifact"),
            "openai_call_count": recovery.get("openai_call_count", 0),
        },
        "judge": judge,
        "openai_call_count": recovery.get("openai_call_count", 0)
        + judge.get("openai_call_count", 0),
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "snapshot",
        nargs="?",
        type=Path,
        default=DEFAULT_PRIVATE_SNAPSHOT,
        help="Authorized rollout JSONL. Defaults to the local private demo rollout.",
    )
    target = parser.add_mutually_exclusive_group()
    target.add_argument(
        "--line",
        type=int,
        help="Compaction line to judge. Defaults to the known local demo line.",
    )
    target.add_argument(
        "--at",
        help="Timestamp; selects the latest compaction at or before this time.",
    )
    parser.add_argument(
        "--model",
        default=bh.default_judge_model(),
        help="OpenAI model. Plain names use Pydantic AI's openai-responses provider.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path. Use private/ for sensitive local results.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.line is None and args.at is None:
        args.line = DEFAULT_COMPACTION_LINE
    try:
        result = run(args)
    except Exception as exc:
        result = {"status": "error", "error": str(exc), "openai_call_count": 0}
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    else:
        print(payload, end="")
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
