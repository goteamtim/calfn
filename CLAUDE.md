# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Fine-tune FunctionGemma 270M to turn text into `create_event` tool calls. Small on purpose: computed labels, a frozen eval, a baseline from day one. Portfolio piece; sibling of `../function-gemma-fine-tune`.

## Key files

- `calfn/schema.py`: two schemas. CALL (`TOOL`: what the model writes, date as an expression) and EVENT (`EVENT_TOOL`: the resolved, absolute event that is scored and converted); validation, normalization, prompts. Single source of truth.
- `calfn/dates.py` (the date expression language + calendar arithmetic) and `calfn/resolve.py` (call + `now` -> event): the only place dates are computed (D9). The model never does date math and never sees the date.
- `calfn/task.py`: `extract` (default) vs `direct` (the v1 approach, kept for baselines and v1 checkpoints).
- `gen/resolver.py` + `gen/vocab.py` + `gen/generate.py`: the labeler and data generator; every date phrase carries its expression. `tests/refparse.py` is an independent parser the tests compare it with.
- `data/eval/`: `eval_v1` (dev), `test_v1` (score once), `natural_v1` (hand-drafted, unreviewed). Frozen: don't edit; make a v2. Their answers are resolved events; `train.jsonl` answers are calls (`meta.event` is the event).
- `scripts/inference.py` (score a model, also `--oracle`), `scripts/finetune.py`, `scripts/baseline_claude.py`, `scripts/summarize.py`.
- `docs/decisions.md` (dated D1..), `docs/experiments.md` (one row per run).

## Commands

```bash
.venv/bin/python -m pytest                                  # all tests (pytest.ini: pythonpath=., testpaths=tests)
.venv/bin/python -m pytest tests/test_resolver.py -k name   # one file / one test
python -m gen.generate --out data                           # regenerate train; never overwrites frozen eval unless --force (only for an eval v2)
python scripts/inference.py --oracle --eval data/eval/eval_v1.jsonl   # verify the scoring path with gold answers, no model
python scripts/inference.py --model output/calfn-v2 --eval data/eval/eval_v1.jsonl --name calfn-v2 --out-dir results/dev   # --task extract is the default; a v1 checkpoint needs --task direct
python scripts/inference.py --model output/calfn-v2 --prompt "lunch with Sam next Friday at noon" --now 2026-10-14T09:00
python scripts/baseline_claude.py --eval data/eval/eval_v1.jsonl --model claude-opus-5-5 --out-dir results/dev
python scripts/baseline_ollama.py --eval data/eval/eval_v1.jsonl --model gemma4:12b --name ollama-gemma4-12b --out-dir results/dev   # local Ollama baseline
python scripts/summarize.py results/dev                     # comparison table
python scripts/finetune.py --data data/train.jsonl --output output/calfn-v2   # pinned train env, GPU
```

## Architecture

Data flows: `gen/` (values -> phrases, labels computed) -> `data/*.jsonl` -> `scripts/finetune.py` -> model. At inference, the model's raw output goes through `calfn/parse.py` (output -> arguments), then `calfn/task.py` -> `calfn/resolve.py` (arguments + the example's `meta.now` -> an event; malformed or impossible calls fail as `invalid`), then `calfn/scoring.py` (strict exact-event metric, per-field accuracy with `start_date`/`start_time` separate, Wilson CIs; a correct rejection, i.e. no call, also passes). `calfn/convert.py` is plain code that turns an event into `.ics` / Google / Outlook links and a share token.

The model-facing prompt (short developer message, tool declaration, request; no date) comes from `calfn/task.py`, so training, inference and the baselines share it. The generator asserts for every row it writes that the call resolves to the expected event, and `gen.generate` verifies the frozen eval/test still reproduce byte for byte, so a change to `gen/` or `calfn/dates.py` that alters old labels fails loudly.

## Conventions

- Labels are computed, never written by hand or by an LLM (D1). Change a date convention in `calfn/dates.py`, `gen/resolver.py` (the phrases and their expressions), `DIRECT_SYSTEM_PROMPT` and `data/eval/README.md` together, then regenerate train and make an eval v2 (the old files will refuse to regenerate if a convention shifted).
- The model extracts and code computes (D9): don't ask the model for arithmetic. New phrasings need a new expression in `calfn/dates.py` plus a test in `tests/test_dates_resolve.py`.
- Never use `eval_v1` or `test_v1` to choose hyperparameters or checkpoints beyond reading results. `test_v1` is scored once (D4).
- Keep `docs/decisions.md` and `docs/experiments.md` current in the same session as the change. Numbers must come from files in `results/`.
- Tests: `.venv/Scripts/python.exe -m pytest` (Windows) / `.venv/bin/python -m pytest`. `tests/test_prompt_format.py` needs transformers and the tokenizer and is skipped without them.
- Training happens in a separate pinned env (`requirements-train.txt`); it runs on the RTX 3060 machine, not necessarily this one. Unload Ollama models first.
