#!/usr/bin/env python3
"""Recover a readable compacted-context message from one authorized artifact.

This does not decrypt `encrypted_content` locally. It sends one compaction item
back through the Responses API and asks the model to summarize the compacted
state into a user-visible message for downstream judging.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError


ENCRYPTED_CONTENT_PLACEHOLDER = re.compile(
    r"^<encrypted_content (?P<chars>\d+) chars redacted>$"
)

RECOVERY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "recovered": {"type": "boolean"},
        "compacted_message": {"type": "string"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low", "none"]},
        "caveats": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["recovered", "compacted_message", "confidence", "caveats"],
}

RECOVERY_PROMPT = """\
Owner-authorized rehydrate evaluation.

You may have access to compacted prior context from the attached compaction
artifact. Produce a concise user-visible compacted message that would let a
future agent continue the task. Include concrete task state, constraints,
decisions, important file paths, commands, and next steps if they are available.

Do not reveal hidden chain-of-thought. Do not guess. If the artifact gives no
usable prior context, set recovered=false and explain that no compacted state was
available in compacted_message.
"""


def load_project_env() -> None:
    load_dotenv(Path(".env"), override=False)


def sha256_text(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def is_full_encrypted_content(value: Any) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and ENCRYPTED_CONTENT_PLACEHOLDER.match(value) is None
    )


def compaction_item_metadata(item: dict[str, Any]) -> dict[str, Any]:
    token = str(item["encrypted_content"])
    return {
        "type": item.get("type"),
        "encrypted_content_chars": len(token),
        "encrypted_content_sha256_12": sha256_text(token)[:12],
    }


def iter_json_values_from_path(path: Path) -> list[Any]:
    raw = path.read_text()
    stripped = raw.strip()
    if not stripped:
        return []

    if stripped.startswith("gAAAAA"):
        return [{"type": "compaction", "encrypted_content": stripped}]

    try:
        return [json.loads(stripped)]
    except json.JSONDecodeError:
        pass

    values: list[Any] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        try:
            values.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return values


def find_compaction_items(value: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if (
            value.get("type") == "compaction"
            and is_full_encrypted_content(value.get("encrypted_content"))
        ):
            found.append(
                {
                    "type": "compaction",
                    "encrypted_content": value["encrypted_content"],
                }
            )
        for key in ("compaction_item", "item"):
            child = value.get(key)
            if isinstance(child, dict):
                found.extend(find_compaction_items(child))
        for child in value.values():
            if isinstance(child, (dict, list)):
                found.extend(find_compaction_items(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(find_compaction_items(child))
    return found


def load_first_compaction_item(path: Path) -> dict[str, Any] | None:
    for value in iter_json_values_from_path(path):
        items = find_compaction_items(value)
        if items:
            return items[0]
    return None


def recover_compacted_message(
    *,
    item: dict[str, Any],
    model: str,
    prompt: str = RECOVERY_PROMPT,
    api_key: str | None = None,
    client: Any | None = None,
) -> dict[str, Any]:
    load_project_env()
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key and client is None:
        return {"status": "not_run", "error": "OPENAI_API_KEY is not set"}
    if not model.strip():
        return {"status": "not_run", "error": "model is empty"}
    if not is_full_encrypted_content(item.get("encrypted_content")):
        return {"status": "not_run", "error": "compaction item has no full token"}

    client = client or OpenAI(api_key=api_key)
    try:
        response = client.responses.create(
            model=model.strip(),
            store=False,
            input=[
                {
                    "type": "compaction",
                    "encrypted_content": item["encrypted_content"],
                },
                {"role": "user", "content": prompt},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "rehydrate_compacted_message",
                    "strict": True,
                    "schema": RECOVERY_SCHEMA,
                }
            },
        )
    except OpenAIError as exc:
        return {"status": "error", "error": str(exc)}

    try:
        result = json.loads(response.output_text)
    except json.JSONDecodeError:
        return {
            "status": "error",
            "error": "model response was not valid JSON",
            "response_id": response.id,
            "response_text": response.output_text,
        }

    return {
        "status": "ok",
        "response_id": response.id,
        "model": model.strip(),
        "artifact": compaction_item_metadata(item),
        "result": result,
        "usage": response.usage.model_dump() if response.usage else None,
        "openai_call_count": 1,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path, help="JSON, JSONL, or token file")
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL") or os.getenv("OPENAI_JUDGE_MODEL") or "gpt-5.5",
        help="Responses model to use",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    parser.add_argument(
        "--i-have-authorization",
        action="store_true",
        help="Required acknowledgement before sending an artifact to the API",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if not args.i_have_authorization:
        raise SystemExit("Refusing to run without --i-have-authorization")

    item = load_first_compaction_item(args.artifact)
    if item is None:
        raise SystemExit(f"No full compaction artifact found in {args.artifact}")

    result = recover_compacted_message(item=item, model=args.model)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    else:
        print(payload, end="")
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
