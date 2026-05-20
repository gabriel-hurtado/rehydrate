"""Analysis helpers for the baseline notebooks.

The helpers keep raw JSONL lines as the authority. Parsed fields are shallow
indexes for profiling, retrieval, and benchmark-case construction.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

import rehydrate_compaction as rc


COMPACTION_EVENT_TYPES = {"compacted"}
COMPACTION_PAYLOAD_TYPES = {
    "compaction",
    "context_compacted",
    "context_compaction",
}

DISPLAY_COLUMNS = [
    "line_number",
    "timestamp",
    "event_type",
    "payload_type",
    "payload_role",
    "payload_name",
    "payload_keyset",
    "raw_hash",
]

SECRET_PATTERNS = [
    (re.compile(r"sk-[A-Za-z0-9_-]{10,}"), "<REDACTED_OPENAI_KEY>"),
]
ENCRYPTED_CONTENT_PLACEHOLDER = re.compile(
    r"^<encrypted_content (?P<chars>\d+) chars redacted>$"
)

JUDGE_RESULT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "pass": {"type": "boolean"},
        "score_0_to_2": {"type": "integer", "minimum": 0, "maximum": 2},
        "supported_claim_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
        "unsupported_claim_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
        "missing_citation_claim_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
        "rationale": {"type": "string"},
    },
    "required": [
        "pass",
        "score_0_to_2",
        "supported_claim_ids",
        "unsupported_claim_ids",
        "missing_citation_claim_ids",
        "rationale",
    ],
}

COMPACTION_SURVIVAL_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "cases": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "case_id": {"type": "string"},
                    "compaction_line": {"type": "integer"},
                    "compacted_artifact_available": {"type": "boolean"},
                    "facts": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "fact_id": {"type": "string"},
                                "fact": {"type": "string"},
                                "importance": {"type": "string"},
                                "source_refs": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "raw_line_ids": {
                                    "type": "array",
                                    "items": {"type": "integer"},
                                },
                                "survival": {
                                    "type": "string",
                                    "enum": ["preserved", "partial", "lost"],
                                },
                                "evidence": {"type": "string"},
                            },
                            "required": [
                                "fact_id",
                                "fact",
                                "importance",
                                "source_refs",
                                "raw_line_ids",
                                "survival",
                                "evidence",
                            ],
                        },
                    },
                    "preserved_count": {"type": "integer"},
                    "partial_count": {"type": "integer"},
                    "lost_count": {"type": "integer"},
                    "survival_score_0_to_2": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 2,
                    },
                    "rationale": {"type": "string"},
                },
                "required": [
                    "case_id",
                    "compaction_line",
                    "compacted_artifact_available",
                    "facts",
                    "preserved_count",
                    "partial_count",
                    "lost_count",
                    "survival_score_0_to_2",
                    "rationale",
                ],
            },
        },
        "overall_rationale": {"type": "string"},
    },
    "required": ["cases", "overall_rationale"],
}

COMPACTION_SURVIVAL_CASE_SCHEMA = COMPACTION_SURVIVAL_SCHEMA["properties"]["cases"][
    "items"
]

SERVER_COMPACTION_RECOVERY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "exact_available": {"type": "boolean"},
        "exact_compaction_output": {
            "type": ["string", "null"],
        },
        "best_effort_reconstruction": {"type": "string"},
        "caveats": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": [
        "exact_available",
        "exact_compaction_output",
        "best_effort_reconstruction",
        "caveats",
    ],
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_project_env(env_path: Path | None = None) -> None:
    path = env_path or Path(__file__).resolve().parent / ".env"
    load_dotenv(path, override=False)


def has_openai_api_key() -> bool:
    load_project_env()
    return bool(os.getenv("OPENAI_API_KEY"))


def default_judge_model() -> str:
    load_project_env()
    return os.getenv("OPENAI_JUDGE_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-5.5"


def split_line_ending(raw_line: bytes) -> tuple[bytes, bytes]:
    if raw_line.endswith(b"\r\n"):
        return raw_line[:-2], b"\r\n"
    if raw_line.endswith(b"\n"):
        return raw_line[:-1], b"\n"
    if raw_line.endswith(b"\r"):
        return raw_line[:-1], b"\r"
    return raw_line, b""


def redact_sensitive_text(text: str) -> str:
    redacted = text
    for pattern, replacement in SECRET_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


def redact_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: (
                f"<encrypted_content {len(child)} chars redacted>"
                if key == "encrypted_content" and isinstance(child, str)
                else redact_json_value(child)
            )
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [redact_json_value(item) for item in value]
    if isinstance(value, str):
        return redact_sensitive_text(value)
    return value


def encrypted_content_chars(value: Any) -> int:
    if not isinstance(value, str):
        return 0
    match = ENCRYPTED_CONTENT_PLACEHOLDER.match(value)
    if match:
        return int(match.group("chars"))
    return len(value)


def collect_key_paths(value: Any, target: str, prefix: str = "") -> Counter[str]:
    paths: Counter[str] = Counter()
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else key
            if key == target:
                paths[path] += 1
            paths.update(collect_key_paths(child, target, path))
    elif isinstance(value, list):
        for child in value:
            path = f"{prefix}[]" if prefix else "[]"
            paths.update(collect_key_paths(child, target, path))
    return paths


def parse_rollout_rows(path: Path) -> dict[str, Any]:
    raw_bytes = path.read_bytes()
    raw_lines = raw_bytes.splitlines(keepends=True)
    rows: list[dict[str, Any]] = []

    for line_number, raw_line_with_ending in enumerate(raw_lines, start=1):
        raw_line, line_ending = split_line_ending(raw_line_with_ending)
        raw_text = raw_line.decode("utf-8", errors="replace")
        parsed: dict[str, Any] | None = None
        json_error = None
        try:
            loaded = json.loads(raw_line)
            if isinstance(loaded, dict):
                parsed = loaded
            else:
                json_error = "JSON value is not an object"
        except json.JSONDecodeError as exc:
            json_error = str(exc)

        payload = parsed.get("payload") if isinstance(parsed, dict) else None
        payload_obj = payload if isinstance(payload, dict) else {}
        event_type = parsed.get("type") if isinstance(parsed, dict) else None
        payload_type = payload_obj.get("type")
        top_level_keys = sorted(str(key) for key in parsed.keys()) if parsed else []
        payload_keys = sorted(str(key) for key in payload_obj.keys())
        is_compaction = (
            isinstance(event_type, str)
            and event_type in COMPACTION_EVENT_TYPES
            or isinstance(payload_type, str)
            and payload_type in COMPACTION_PAYLOAD_TYPES
        )

        rows.append(
            {
                "source_file": str(path),
                "line_number": line_number,
                "raw_json": raw_text,
                "line_ending": line_ending.decode("ascii", errors="replace"),
                "raw_hash": sha256_bytes(raw_line),
                "raw_bytes": len(raw_line),
                "valid_json": parsed is not None,
                "json_error": json_error,
                "timestamp": parsed.get("timestamp") if parsed else None,
                "event_type": str(event_type) if event_type is not None else None,
                "payload_type": str(payload_type) if payload_type is not None else None,
                "payload_role": (
                    str(payload_obj["role"]) if payload_obj.get("role") is not None else None
                ),
                "payload_name": (
                    str(payload_obj["name"]) if payload_obj.get("name") is not None else None
                ),
                "payload_keys": payload_keys,
                "payload_keyset": ",".join(payload_keys) if payload_keys else None,
                "top_level_keys": top_level_keys,
                "top_level_keyset": ",".join(top_level_keys) if top_level_keys else None,
                "turn_id": (
                    str(payload_obj["turn_id"])
                    if payload_obj.get("turn_id") is not None
                    else None
                ),
                "session_meta_id": (
                    str(payload_obj["id"])
                    if event_type == "session_meta" and payload_obj.get("id") is not None
                    else None
                ),
                "is_compaction": is_compaction,
                "_parsed": parsed,
            }
        )

    return {
        "path": str(path),
        "sha256": sha256_bytes(raw_bytes),
        "rows": rows,
    }


def counter_records(counter: Counter[str], total: int) -> list[dict[str, Any]]:
    return [
        {
            "value": value,
            "count": count,
            "percent": round((count / total) * 100, 2) if total else None,
        }
        for value, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]


def count_field(rows: list[dict[str, Any]], field: str, missing_label: str) -> Counter[str]:
    return Counter(str(row.get(field) or missing_label) for row in rows)


def key_path_records(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    counter: Counter[str] = Counter()
    for row in rows:
        parsed = row.get("_parsed")
        if parsed is not None:
            counter.update(collect_key_paths(parsed, key))
    return counter_records(counter, sum(counter.values()))


def display_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{column: row.get(column) for column in DISPLAY_COLUMNS} for row in rows]


def profile_rollout(path: Path) -> dict[str, Any]:
    parsed = parse_rollout_rows(path)
    rows = parsed["rows"]
    total = len(rows)
    valid_rows = [row for row in rows if row["valid_json"]]
    payload_object_rows = [row for row in rows if row["payload_keyset"] is not None]
    compaction_rows = [row for row in rows if row["is_compaction"]]
    turn_context_rows = [row for row in rows if row["event_type"] == "turn_context"]
    session_ids = sorted({row["session_meta_id"] for row in rows if row["session_meta_id"]})

    event_counts = count_field(valid_rows, "event_type", "<missing>")
    payload_counts = count_field(payload_object_rows, "payload_type", "<missing>")
    role_counts = count_field(payload_object_rows, "payload_role", "<missing>")
    name_counts = count_field(payload_object_rows, "payload_name", "<missing>")
    top_level_keysets = count_field(valid_rows, "top_level_keyset", "<missing>")
    payload_keysets = count_field(payload_object_rows, "payload_keyset", "<empty>")

    schema_rows = []
    for event_type, count in event_counts.items():
        matching = [row for row in rows if row["event_type"] == event_type]
        first = matching[0]
        schema_rows.append(
            {
                "event_type": event_type,
                "count": count,
                "first_line": first["line_number"],
                "top_level_keyset": first["top_level_keyset"],
                "payload_keysets_observed": len(
                    {row["payload_keyset"] for row in matching}
                ),
                "common_payload_keyset": Counter(
                    row["payload_keyset"] or "<empty>" for row in matching
                ).most_common(1)[0][0],
            }
        )
    schema_rows.sort(key=lambda row: (-row["count"], row["event_type"]))

    summary_rows = [
        {"metric": "source file", "value": path.name},
        {"metric": "snapshot sha256", "value": parsed["sha256"]},
        {"metric": "source lines", "value": total},
        {"metric": "valid JSON object lines", "value": len(valid_rows)},
        {"metric": "invalid JSON lines", "value": total - len(valid_rows)},
        {"metric": "event types observed", "value": len(event_counts)},
        {"metric": "payload types observed", "value": len(payload_counts)},
        {"metric": "turn_context events", "value": len(turn_context_rows)},
        {"metric": "compaction-related events", "value": len(compaction_rows)},
        {
            "metric": "discoverable session ids",
            "value": ", ".join(session_ids) if session_ids else None,
        },
    ]

    return {
        **parsed,
        "total_lines": total,
        "valid_json_lines": len(valid_rows),
        "invalid_json_lines": total - len(valid_rows),
        "summary_rows": summary_rows,
        "schema_rows": schema_rows,
        "event_type_rows": counter_records(event_counts, len(valid_rows)),
        "payload_type_rows": counter_records(payload_counts, len(payload_object_rows)),
        "payload_role_rows": counter_records(role_counts, len(payload_object_rows)),
        "payload_name_rows": counter_records(name_counts, len(payload_object_rows)),
        "top_level_keyset_rows": counter_records(top_level_keysets, len(valid_rows)),
        "payload_keyset_rows": counter_records(payload_keysets, len(payload_object_rows)),
        "timestamp_path_rows": key_path_records(rows, "timestamp"),
        "id_path_rows": key_path_records(rows, "id"),
        "turn_id_path_rows": key_path_records(rows, "turn_id"),
        "session_id_path_rows": key_path_records(rows, "session_id"),
        "compaction_rows": display_rows(compaction_rows),
        "compaction_line_numbers": [row["line_number"] for row in compaction_rows],
        "turn_context_rows": display_rows(turn_context_rows),
        "session_ids": session_ids,
    }


def source_window(
    rows: list[dict[str, Any]],
    target_line: int,
    before: int,
    after: int,
) -> list[dict[str, Any]]:
    start = max(1, target_line - before)
    end = min(len(rows), target_line + after)
    return display_rows(
        [row for row in rows if start <= row["line_number"] <= end]
    )


def raw_source_window(
    rows: list[dict[str, Any]],
    target_line: int,
    before: int,
    after: int,
) -> list[dict[str, Any]]:
    start = max(1, target_line - before)
    end = min(len(rows), target_line + after)
    return [
        {
            "line_number": row["line_number"],
            "timestamp": row["timestamp"],
            "event_type": row["event_type"],
            "payload_type": row["payload_type"],
            "raw_json": row["raw_json"],
        }
        for row in rows
        if start <= row["line_number"] <= end
    ]


def json_safe_source(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, child in value.items():
            if key == "encrypted_content" and isinstance(child, str):
                out[key] = (
                    child
                    if ENCRYPTED_CONTENT_PLACEHOLDER.match(child)
                    else f"<encrypted_content {len(child)} chars omitted>"
                )
            else:
                out[key] = json_safe_source(child)
        return out
    if isinstance(value, list):
        return [json_safe_source(item) for item in value]
    if isinstance(value, str):
        return redact_sensitive_text(value)
    return value


def compaction_survival_cases(profile: dict[str, Any]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    rows = profile["rows"]
    for row in rows:
        if row["event_type"] != "compacted":
            continue
        parsed = row.get("_parsed") or {}
        payload = parsed.get("payload") if isinstance(parsed.get("payload"), dict) else {}
        replacement_history = payload.get("replacement_history")
        if not isinstance(replacement_history, list):
            continue

        compaction_line = int(row["line_number"])
        message = payload.get("message")
        message_text = redact_sensitive_text(message) if isinstance(message, str) else ""
        opaque_compaction_items = [
            {
                "source_ref": (
                    f"line-{compaction_line}:payload.replacement_history[{index}]"
                ),
                "raw_line_ids": [compaction_line],
                "replacement_history_index": index,
                "encrypted_content_chars": encrypted_content_chars(
                    item.get("encrypted_content")
                ),
            }
            for index, item in enumerate(replacement_history)
            if isinstance(item, dict)
            and item.get("type") == "compaction"
            and isinstance(item.get("encrypted_content"), str)
        ]
        post_signals = [
            {
                "line_number": candidate["line_number"],
                "event_type": candidate["event_type"],
                "payload_type": candidate["payload_type"],
            }
            for candidate in rows[compaction_line : min(len(rows), compaction_line + 4)]
        ]

        cases.append(
            {
                "case_id": f"compaction-survival:line-{compaction_line}",
                "compaction_line": compaction_line,
                "timestamp": row["timestamp"],
                "context": {
                    "scope": "full payload.replacement_history for this compacted event",
                    "truncated": False,
                    "source_file": row["source_file"],
                    "raw_line_ids": [compaction_line],
                },
                "source": {
                    "raw_line_ids": [compaction_line],
                    "field": "payload.replacement_history",
                    "items": [
                        {
                            "source_ref": (
                                f"line-{compaction_line}:"
                                f"payload.replacement_history[{index}]"
                            ),
                            "raw_line_ids": [compaction_line],
                            "replacement_history_index": index,
                            "item": json_safe_source(item),
                        }
                        for index, item in enumerate(replacement_history)
                    ],
                },
                "compacted_artifact": {
                    "raw_line_ids": [compaction_line],
                    "field": "payload.message",
                    "message": message_text,
                    "message_length": len(message_text),
                    "plaintext_message_available": len(message_text) > 0,
                    "opaque_compaction_items": opaque_compaction_items,
                    "opaque_compaction_item_count": len(opaque_compaction_items),
                    "opaque_compaction_chars": sum(
                        item["encrypted_content_chars"]
                        for item in opaque_compaction_items
                    ),
                    "post_compaction_signals": post_signals,
                },
            }
        )
    return cases


def compaction_survival_case_rows(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "case_id": case["case_id"],
            "compaction_line": case["compaction_line"],
            "replacement_history_items": len(case["source"]["items"]),
            "plaintext_message_available": case["compacted_artifact"][
                "plaintext_message_available"
            ],
            "plaintext_message_chars": case["compacted_artifact"]["message_length"],
            "opaque_compaction_items": case["compacted_artifact"][
                "opaque_compaction_item_count"
            ],
            "opaque_compaction_chars": case["compacted_artifact"][
                "opaque_compaction_chars"
            ],
            "auditable_survival_target": case["compacted_artifact"][
                "plaintext_message_available"
            ],
        }
        for case in cases
    ]


def compaction_server_recovery_cases(profile: dict[str, Any]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for row in profile["rows"]:
        if row["event_type"] != "compacted":
            continue
        parsed = row.get("_parsed") or {}
        payload = parsed.get("payload") if isinstance(parsed.get("payload"), dict) else {}
        replacement_history = payload.get("replacement_history")
        if not isinstance(replacement_history, list):
            continue

        compaction_line = int(row["line_number"])
        for index, item in enumerate(replacement_history):
            if not isinstance(item, dict) or item.get("type") != "compaction":
                continue
            encrypted_content = item.get("encrypted_content")
            if not isinstance(encrypted_content, str):
                continue
            token_available = (
                bool(encrypted_content)
                and ENCRYPTED_CONTENT_PLACEHOLDER.match(encrypted_content) is None
            )
            cases.append(
                {
                    "case_id": f"server-compaction-recovery:line-{compaction_line}",
                    "compaction_line": compaction_line,
                    "timestamp": row["timestamp"],
                    "source_file": row["source_file"],
                    "raw_line_ids": [compaction_line],
                    "source_ref": (
                        f"line-{compaction_line}:"
                        f"payload.replacement_history[{index}]"
                    ),
                    "replacement_history_index": index,
                    "encrypted_content_chars": encrypted_content_chars(encrypted_content),
                    "encrypted_content_available": token_available,
                    "compaction_item": (
                        {"type": "compaction", "encrypted_content": encrypted_content}
                        if token_available
                        else None
                    ),
                }
            )
    return cases


def compaction_server_recovery_case_rows(
    cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "case_id": case["case_id"],
            "compaction_line": case["compaction_line"],
            "replacement_history_index": case["replacement_history_index"],
            "encrypted_content_chars": case["encrypted_content_chars"],
            "encrypted_content_available": case["encrypted_content_available"],
            "source_ref": case["source_ref"],
        }
        for case in cases
    ]


def first_available_compaction_item(
    profile: dict[str, Any],
    *,
    artifact_path: Path | None = None,
) -> dict[str, Any] | None:
    if artifact_path is not None:
        item = rc.load_first_compaction_item(artifact_path)
        if item is None:
            return None
        return {
            "source": str(artifact_path),
            "source_kind": "artifact_file",
            "item": item,
            "metadata": rc.compaction_item_metadata(item),
        }

    for case in compaction_server_recovery_cases(profile):
        item = case.get("compaction_item")
        if isinstance(item, dict):
            return {
                "source": case["source_ref"],
                "source_kind": "snapshot",
                "case": case,
                "compaction_line": case["compaction_line"],
                "replacement_history_index": case["replacement_history_index"],
                "item": item,
                "metadata": rc.compaction_item_metadata(item),
            }
    return None


def last_available_compaction_item(profile: dict[str, Any]) -> dict[str, Any] | None:
    candidate = None
    for case in compaction_server_recovery_cases(profile):
        item = case.get("compaction_item")
        if isinstance(item, dict):
            candidate = {
                "source": case["source_ref"],
                "source_kind": "snapshot",
                "case": case,
                "compaction_line": case["compaction_line"],
                "replacement_history_index": case["replacement_history_index"],
                "item": item,
                "metadata": rc.compaction_item_metadata(item),
            }
    return candidate


def recover_first_compacted_message(
    profile: dict[str, Any],
    *,
    model: str,
    artifact_path: Path | None = None,
) -> dict[str, Any]:
    candidate = first_available_compaction_item(profile, artifact_path=artifact_path)
    if candidate is None:
        return {
            "status": "not_run",
            "error": "no full compaction artifact available",
            "openai_call_count": 0,
        }

    result = rc.recover_compacted_message(item=candidate["item"], model=model)
    return {
        **result,
        "candidate_source": candidate["source"],
        "candidate_source_kind": candidate["source_kind"],
        "candidate_compaction_line": candidate.get("compaction_line"),
        "candidate_replacement_history_index": candidate.get(
            "replacement_history_index"
        ),
        "artifact": candidate["metadata"],
    }


def recover_last_compacted_message(
    profile: dict[str, Any],
    *,
    model: str,
) -> dict[str, Any]:
    candidate = last_available_compaction_item(profile)
    if candidate is None:
        return {
            "status": "not_run",
            "error": "no full compaction artifact available",
            "openai_call_count": 0,
        }

    result = rc.recover_compacted_message(item=candidate["item"], model=model)
    return {
        **result,
        "candidate_source": candidate["source"],
        "candidate_source_kind": candidate["source_kind"],
        "candidate_compaction_line": candidate.get("compaction_line"),
        "candidate_replacement_history_index": candidate.get(
            "replacement_history_index"
        ),
        "artifact": candidate["metadata"],
    }


def compaction_case_with_recovered_message(
    case: dict[str, Any],
    recovery_result: dict[str, Any],
) -> dict[str, Any]:
    recovered = recovery_result.get("result") or {}
    compacted_message = str(recovered.get("compacted_message") or "")
    case_copy = json.loads(json.dumps(case))
    artifact = case_copy["compacted_artifact"]
    artifact["field"] = "server_recovered_compacted_message"
    artifact["message"] = compacted_message
    artifact["message_length"] = len(compacted_message)
    artifact["plaintext_message_available"] = (
        recovered.get("recovered") is True and bool(compacted_message.strip())
    )
    artifact["server_recovery"] = {
        "status": recovery_result.get("status"),
        "response_id": recovery_result.get("response_id"),
        "model": recovery_result.get("model"),
        "candidate_source": recovery_result.get("candidate_source"),
        "candidate_source_kind": recovery_result.get("candidate_source_kind"),
        "confidence": recovered.get("confidence"),
        "caveats": recovered.get("caveats") or [],
    }
    case_copy["case_id"] = f"{case_copy['case_id']}:server-recovered"
    return case_copy


def server_recovery_status_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    recovered = result.get("result") or {}
    usage = result.get("usage") or {}
    artifact = result.get("artifact") or {}
    return [
        {
            "status": result.get("status"),
            "recovered": recovered.get("recovered"),
            "confidence": recovered.get("confidence"),
            "response_id": result.get("response_id"),
            "model": result.get("model"),
            "candidate_source_kind": result.get("candidate_source_kind"),
            "candidate_source": result.get("candidate_source"),
            "candidate_compaction_line": result.get("candidate_compaction_line"),
            "candidate_replacement_history_index": result.get(
                "candidate_replacement_history_index"
            ),
            "artifact_chars": artifact.get("encrypted_content_chars"),
            "artifact_sha256_12": artifact.get("encrypted_content_sha256_12"),
            "openai_call_count": result.get("openai_call_count", 0),
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "error": result.get("error"),
        }
    ]


def judge_cases(
    profile: dict[str, Any],
    *,
    radius: int = 6,
    limit: int | None = 20,
) -> list[dict[str, Any]]:
    rows = profile["rows"]
    anchors = [row for row in rows if row["is_compaction"]]
    mode = "compaction_recovery"
    if not anchors:
        anchors = [row for row in rows if row["event_type"] == "turn_context"]
        mode = "turn_boundary_retrieval"
    if not anchors and rows:
        anchors = [rows[0]]
        mode = "file_boundary_retrieval"

    cases = []
    selected_anchors = anchors if limit is None else anchors[:limit]
    for anchor in selected_anchors:
        start = max(1, anchor["line_number"] - radius)
        end = min(profile["total_lines"], anchor["line_number"] + radius)
        cases.append(
            {
                "case_id": f"{mode}:line-{anchor['line_number']}",
                "judge_mode": mode,
                "target_line": anchor["line_number"],
                "anchor_event_type": anchor["event_type"],
                "anchor_payload_type": anchor["payload_type"],
                "source_window_start": start,
                "source_window_end": end,
                "required_raw_line_ids": ",".join(str(line) for line in range(start, end + 1)),
            }
        )
    return cases


def judge_source_windows(
    profile: dict[str, Any],
    cases: list[dict[str, Any]],
    *,
    radius: int = 6,
) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    for case in cases:
        for row in source_window(
            profile["rows"],
            int(case["target_line"]),
            radius,
            radius,
        ):
            windows.append({"case_id": case["case_id"], **row})
    return windows


def judge_case_inputs(
    profile: dict[str, Any],
    cases: list[dict[str, Any]],
    *,
    radius: int = 6,
) -> list[dict[str, Any]]:
    inputs: list[dict[str, Any]] = []
    for case in cases:
        inputs.append(
            {
                "case": case,
                "raw_window": raw_source_window(
                    profile["rows"],
                    int(case["target_line"]),
                    radius,
                    radius,
                ),
                "candidate_claims": normalise_candidate_claims(
                    default_candidate_claims(case)
                ),
            }
        )
    return inputs


def judge_candidate_claim_rows(case_inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in case_inputs:
        case = item["case"]
        for claim in item["candidate_claims"]:
            rows.append(
                {
                    "case_id": case["case_id"],
                    "target_line": case["target_line"],
                    **claim,
                }
            )
    return rows


def default_candidate_claims(case: dict[str, Any] | None) -> str:
    if case is None:
        return "[]"
    return json.dumps(
        [
            {
                "claim_id": "claim-1",
                "claim": (
                    f"Line {case['target_line']} is the anchor event for "
                    f"`{case['judge_mode']}`."
                ),
                "raw_line_ids": [case["target_line"]],
            }
        ],
        indent=2,
    )


def normalise_candidate_claims(text: str) -> list[dict[str, Any]]:
    stripped = text.strip()
    if not stripped:
        return []
    try:
        loaded = json.loads(stripped)
    except json.JSONDecodeError:
        loaded = [{"claim": stripped, "raw_line_ids": []}]
    if isinstance(loaded, dict) and isinstance(loaded.get("claims"), list):
        claims = loaded["claims"]
    elif isinstance(loaded, list):
        claims = loaded
    else:
        claims = [{"claim": str(loaded), "raw_line_ids": []}]

    normalised = []
    for index, claim in enumerate(claims, start=1):
        if isinstance(claim, dict):
            raw_line_ids = claim.get("raw_line_ids") or []
            if not isinstance(raw_line_ids, list):
                raw_line_ids = [raw_line_ids]
            normalised.append(
                {
                    "claim_id": str(claim.get("claim_id") or f"claim-{index}"),
                    "claim": str(claim.get("claim") or ""),
                    "raw_line_ids": raw_line_ids,
                }
            )
        else:
            normalised.append(
                {
                    "claim_id": f"claim-{index}",
                    "claim": str(claim),
                    "raw_line_ids": [],
                }
            )
    return normalised


def run_openai_judge(
    *,
    case: dict[str, Any],
    raw_window: list[dict[str, Any]],
    candidate_claims: list[dict[str, Any]],
    model: str,
    api_key: str | None = None,
) -> dict[str, Any]:
    load_project_env()
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"status": "not_run", "error": "OPENAI_API_KEY is not set"}
    if not model.strip():
        return {"status": "not_run", "error": "judge model is empty"}
    if not candidate_claims:
        return {"status": "not_run", "error": "no candidate claims to judge"}

    judge_input = {
        "case": case,
        "raw_source_window": raw_window,
        "candidate_claims": candidate_claims,
        "rules": [
            "Only judge whether each claim is supported by the supplied raw source window.",
            "A claim without raw_line_ids is missing citation.",
            "Do not infer decisions, failed attempts, todos, or open tasks from plausible text.",
            "Return pass=true only if all claims are cited and supported.",
        ],
    }
    client = OpenAI(api_key=api_key)
    try:
        response = client.responses.create(
            model=model.strip(),
            store=False,
            instructions=(
                "You are a strict provenance judge for a source-backed agent-context "
                "recovery system. Evaluate candidate claims against raw JSONL rollout "
                "events. Return only the requested structured JSON."
            ),
            input=json.dumps(judge_input, indent=2, sort_keys=True),
            text={
                "format": {
                    "type": "json_schema",
                    "name": "rehydrate_judge_result",
                    "strict": True,
                    "schema": JUDGE_RESULT_SCHEMA,
                }
            },
        )
    except OpenAIError as exc:
        return {"status": "error", "error": str(exc)}

    text = response.output_text
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        return {
            "status": "error",
            "error": "model response was not valid JSON",
            "response_text": text,
            "response_id": response.id,
        }

    return {
        "status": "ok",
        "response_id": response.id,
        "model": model.strip(),
        "result": result,
        "usage": response.usage.model_dump() if response.usage else None,
    }


def run_openai_judge_cases(
    profile: dict[str, Any],
    cases: list[dict[str, Any]],
    *,
    radius: int = 6,
    model: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in judge_case_inputs(profile, cases, radius=radius):
        case = item["case"]
        result = run_openai_judge(
            case=case,
            raw_window=item["raw_window"],
            candidate_claims=item["candidate_claims"],
            model=model,
        )
        judged = result.get("result") or {}
        usage = result.get("usage") or {}
        rows.append(
            {
                "case_id": case["case_id"],
                "target_line": case["target_line"],
                "status": result.get("status"),
                "pass": judged.get("pass"),
                "score_0_to_2": judged.get("score_0_to_2"),
                "supported_claim_ids": ",".join(judged.get("supported_claim_ids") or []),
                "unsupported_claim_ids": ",".join(
                    judged.get("unsupported_claim_ids") or []
                ),
                "missing_citation_claim_ids": ",".join(
                    judged.get("missing_citation_claim_ids") or []
                ),
                "rationale": judged.get("rationale"),
                "response_id": result.get("response_id"),
                "model": result.get("model") or model.strip(),
                "error": result.get("error"),
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
                "total_tokens": usage.get("total_tokens"),
            }
        )
    return rows


def compaction_survival_judge_input(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "case": case,
        "task": (
            "For this single compaction case, first identify important explicit "
            "facts from payload.replacement_history, then judge whether each fact "
            "survives in the compacted artifact payload.message. This is a "
            "compression-loss benchmark, not an anchor-citation benchmark."
        ),
        "rules": [
            "This request handles exactly one compaction case.",
            (
                "Treat case.source.items as the full context being compacted: "
                "the complete payload.replacement_history for this compacted event."
            ),
            "Do not invent missing context outside this case.",
            "Extract only facts explicitly supported by replacement_history items.",
            "Each fact must cite source_refs and raw_line_ids from the source object.",
            "Judge survival only against compacted_artifact.message.",
            "A post_compaction_signal is not preserved content.",
            (
                "If compacted_artifact.plaintext_message_available is false, "
                "the persisted JSONL does not expose a plaintext compaction "
                "summary. Do not treat encrypted artifacts as readable evidence."
            ),
            (
                "If compacted_artifact.message is empty, score only the persisted "
                "plaintext artifact as absent and explain that the real in-context "
                "summary is not auditable from this source."
            ),
            "Choose up to six important facts worth preserving.",
            "Do not infer decisions, failed attempts, todos, or open tasks from plausible text.",
            "Score 2 when important facts mostly survive, 1 when only partial gist survives, 0 when important facts are absent.",
        ],
    }


def run_openai_compaction_survival_judge(
    case: dict[str, Any],
    *,
    model: str,
    api_key: str | None = None,
    client: Any | None = None,
) -> dict[str, Any]:
    load_project_env()
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key and client is None:
        return {"status": "not_run", "error": "OPENAI_API_KEY is not set"}
    if not model.strip():
        return {"status": "not_run", "error": "judge model is empty"}
    if not case:
        return {"status": "not_run", "error": "no compaction survival case to judge"}

    judge_input = compaction_survival_judge_input(case)
    client = client or OpenAI(api_key=api_key)
    try:
        response = client.responses.create(
            model=model.strip(),
            store=False,
            instructions=(
                "You are a strict compression-survival judge for source-backed "
                "agent context recovery. Extract important explicit facts from "
                "one raw compaction source history and evaluate whether that "
                "compacted artifact preserves them. Return only the "
                "requested structured JSON."
            ),
            input=json.dumps(judge_input, indent=2, sort_keys=True),
            text={
                "format": {
                    "type": "json_schema",
                    "name": "rehydrate_compaction_survival_case",
                    "strict": True,
                    "schema": COMPACTION_SURVIVAL_CASE_SCHEMA,
                }
            },
        )
    except OpenAIError as exc:
        return {"status": "error", "error": str(exc)}

    text = response.output_text
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        return {
            "status": "error",
            "error": "model response was not valid JSON",
            "response_text": text,
            "response_id": response.id,
        }

    return {
        "status": "ok",
        "case_id": case.get("case_id"),
        "response_id": response.id,
        "model": model.strip(),
        "result": result,
        "usage": response.usage.model_dump() if response.usage else None,
        "openai_call_count": 1,
    }


def run_openai_compaction_survival_judges(
    cases: list[dict[str, Any]],
    *,
    model: str,
    api_key: str | None = None,
) -> dict[str, Any]:
    load_project_env()
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {
            "status": "not_run",
            "error": "OPENAI_API_KEY is not set",
            "results": [],
            "openai_call_count": 0,
        }
    if not model.strip():
        return {
            "status": "not_run",
            "error": "judge model is empty",
            "results": [],
            "openai_call_count": 0,
        }
    if not cases:
        return {
            "status": "not_run",
            "error": "no compaction survival cases to judge",
            "results": [],
            "openai_call_count": 0,
        }

    client = OpenAI(api_key=api_key)
    results = [
        run_openai_compaction_survival_judge(
            case,
            model=model,
            api_key=api_key,
            client=client,
        )
        for case in cases
    ]
    return {
        "status": "ok" if all(result.get("status") == "ok" for result in results) else "error",
        "results": results,
        "openai_call_count": len(results),
    }


def combine_compaction_survival_results(run_result: dict[str, Any]) -> dict[str, Any]:
    cases = [
        result["result"]
        for result in run_result.get("results") or []
        if result.get("status") == "ok" and isinstance(result.get("result"), dict)
    ]
    return {
        "cases": cases,
        "overall_rationale": (
            f"{len(cases)} compaction survival case(s) judged with "
            f"{run_result.get('openai_call_count', 0)} OpenAI call(s)."
        ),
    }


def compaction_survival_call_rows(run_result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for result in run_result.get("results") or []:
        usage = result.get("usage") or {}
        rows.append(
            {
                "case_id": result.get("case_id"),
                "status": result.get("status"),
                "response_id": result.get("response_id"),
                "model": result.get("model"),
                "error": result.get("error"),
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
                "total_tokens": usage.get("total_tokens"),
            }
        )
    return rows


def compaction_survival_result_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in result.get("cases") or []:
        for fact in case.get("facts") or []:
            rows.append(
                {
                    "case_id": case["case_id"],
                    "compaction_line": case["compaction_line"],
                    "survival_score_0_to_2": case["survival_score_0_to_2"],
                    "fact_id": fact["fact_id"],
                    "survival": fact["survival"],
                    "fact": fact["fact"],
                    "source_refs": ", ".join(fact["source_refs"]),
                    "raw_line_ids": ", ".join(str(line) for line in fact["raw_line_ids"]),
                    "evidence": fact["evidence"],
                }
            )
    return rows


def compaction_survival_summary_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "case_id": case["case_id"],
            "compaction_line": case["compaction_line"],
            "compacted_artifact_available": case["compacted_artifact_available"],
            "facts": len(case.get("facts") or []),
            "preserved": case["preserved_count"],
            "partial": case["partial_count"],
            "lost": case["lost_count"],
            "survival_score_0_to_2": case["survival_score_0_to_2"],
            "rationale": case["rationale"],
        }
        for case in result.get("cases") or []
    ]


def v0_task_rows() -> list[dict[str, str]]:
    return [
        {
            "task": "importer",
            "why": "store every raw line exactly once with hashes and shallow indexes",
        },
        {
            "task": "event explorer",
            "why": "surface observed event/payload distributions without semantic extraction",
        },
        {
            "task": "source slice",
            "why": "retrieve cited raw context around an anchor line",
        },
        {
            "task": "benchmark runner",
            "why": "turn notebook checks into repeatable package tests",
        },
        {
            "task": "judge-case compiler",
            "why": "produce one full-context judge input per compacted event",
        },
    ]
