# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Fine-tune FunctionGemma 270M to turn text into `create_event` tool calls. Small on purpose: computed labels, a frozen eval, a baseline from day one. Portfolio piece; sibling of `../function-gemma-fine-tune`.

## Key files

- `calfn/schema.py`: the tool, `SYSTEM_PROMPT` (it states the labeling conventions), validation, normalization. Single source of truth.
- `gen/resolver.py` + `gen/vocab.py` + `gen/generate.py`: the labeler and data generator. `tests/refparse.py` is an independent parser the tests compare it with.
- `data/eval/`: `eval_v1` (dev), `test_v1` (score once), `natural_v1` (hand-drafted, unreviewed). Frozen: don't edit; make a v2.
- `scripts/inference.py` (score a model, also `--oracle`), `scripts/finetune.py`, `scripts/baseline_claude.py`, `scripts/summarize.py`.
- `docs/decisions.md` (dated D1..), `docs/experiments.md` (one row per run).

## Commands

```bash
.venv/bin/python -m pytest                                  # all tests (pytest.ini: pythonpath=., testpaths=tests)
.venv/bin/python -m pytest tests/test_resolver.py -k name   # one file / one test
python -m gen.generate --out data                           # regenerate train; never overwrites frozen eval unless --force (only for an eval v2)
python scripts/inference.py --oracle --eval data/eval/eval_v1.jsonl   # verify the scoring path with gold answers, no model
python scripts/inference.py --model output/calfn-v1 --eval data/eval/eval_v1.jsonl --name calfn-v1 --out-dir results/dev
python scripts/inference.py --model output/calfn-v1 --prompt "lunch with Sam next Friday at noon" --now 2026-10-14T09:00
python scripts/baseline_claude.py --eval data/eval/eval_v1.jsonl --model claude-opus-5-5 --out-dir results/dev
python scripts/baseline_ollama.py --eval data/eval/eval_v1.jsonl --model gemma4:12b --name ollama-gemma4-12b --out-dir results/dev   # local Ollama baseline
python scripts/summarize.py results/dev                     # comparison table
python scripts/finetune.py --data data/train.jsonl --output output/calfn-v1   # pinned train env, GPU
```

## Architecture

Data flows: `gen/` (values -> phrases, labels computed) -> `data/*.jsonl` -> `scripts/finetune.py` -> model. At inference, the model's raw output goes through `calfn/parse.py` (output -> arguments), then `calfn/scoring.py` (strict exact-event metric, per-field accuracy, Wilson CIs; a correct rejection, i.e. no call, also passes). `calfn/convert.py` is plain code that turns a call into `.ics` / Google / Outlook links and a share token. The prompt the model sees (developer message with `Now: ...`, tool declaration, request) is built from `calfn/schema.py`, so training, inference and the Claude baseline all share it.

## Conventions

- Labels are computed, never written by hand or by an LLM (D1). Change a convention in `gen/resolver.py`, `calfn/schema.py` `SYSTEM_PROMPT` and `data/eval/README.md` together, then regenerate train and make an eval v2.
- Never use `eval_v1` or `test_v1` to choose hyperparameters or checkpoints beyond reading results. `test_v1` is scored once (D4).
- Keep `docs/decisions.md` and `docs/experiments.md` current in the same session as the change. Numbers must come from files in `results/`.
- Tests: `.venv/Scripts/python.exe -m pytest` (Windows) / `.venv/bin/python -m pytest`. `tests/test_prompt_format.py` needs transformers and the tokenizer and is skipped without them.
- Training happens in a separate pinned env (`requirements-train.txt`); it runs on the RTX 3060 machine, not necessarily this one. Unload Ollama models first.
