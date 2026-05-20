# rehydrate

Rehydrate evaluates whether compacted agent state preserves the facts needed to
continue a task.

Given an authorized agent rollout JSONL, it can select one compaction moment,
recover a user-visible compacted message from the compaction artifact, and ask
an LLM judge whether important source facts survived.

## Project Structure

- `baseline_analysis.ipynb`: notebook walkthrough for one compaction moment.
- `rehydrate_moment.py`: CLI for recovery plus structured judging.
- `baseline_helpers.py`: JSONL parsing, case construction, and judge helpers.
- `rehydrate_compaction.py`: compaction-artifact recovery helper.
- `pyproject.toml` / `uv.lock`: reproducible Python environment.

Private rollouts, full encrypted artifacts, and judge outputs are intentionally
not committed. Keep local inputs and outputs under `private/`; that path is
ignored by git.

## Setup

```bash
uv sync
```

Create a local `.env` file:

```bash
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-5.5
```

`.env` is ignored by git.

## CLI

Run the tool against an authorized rollout JSONL:

```bash
uv run python rehydrate_moment.py path/to/rollout.jsonl \
  --line 1234 \
  --output private/moment.private.json
```

Or select the latest compaction at or before a timestamp:

```bash
uv run python rehydrate_moment.py path/to/rollout.jsonl \
  --at 2026-05-20T14:42:00Z \
  --output private/moment.private.json
```

The CLI performs two OpenAI calls:

1. Recover a compacted message from the selected `type="compaction"` artifact.
2. Use Pydantic AI structured output to judge how well the recovered message
   preserved facts from `payload.replacement_history`.

The JSON output includes the selected case metadata, recovery metadata,
`score_0_to_10`, `verdict`, extracted facts, and `missed_facts`.

For local development on this workstation, the CLI also has a private default
snapshot and compaction line wired in, so this command exercises the demo case
without passing a path:

```bash
uv run python rehydrate_moment.py --output private/moment-912.private.json
```

## Notebook

Open the notebook:

```bash
uv run jupyter lab baseline_analysis.ipynb
```

Execute it non-interactively:

```bash
uv run jupyter nbconvert --execute --to notebook \
  --output /tmp/rehydrate-baseline.ipynb baseline_analysis.ipynb
```

The committed notebook stores no outputs. Live recovery requires an authorized
private rollout with full `encrypted_content`; those rollout files are not
committed.
