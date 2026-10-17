# CLAUDE.md

Fine-tune FunctionGemma 270M to turn text into `create_event` tool calls. Small on purpose: computed labels, a frozen eval, a baseline from day one. Portfolio piece; sibling of `../function-gemma-fine-tune`.

## Key files

- `calfn/schema.py`: the tool, `SYSTEM_PROMPT` (it states the labeling conventions), validation, normalization. Single source of truth.
- `gen/resolver.py` + `gen/vocab.py` + `gen/generate.py`: the labeler and data generator. `tests/refparse.py` is an independent parser the tests compare it with.
- `data/eval/`: `eval_v1` (dev), `test_v1` (score once), `natural_v1` (hand-drafted, unreviewed). Frozen: don't edit; make a v2.
- `scripts/inference.py` (score a model, also `--oracle`), `scripts/finetune.py`, `scripts/baseline_claude.py`, `scripts/summarize.py`.
- `docs/decisions.md` (dated D1..), `docs/experiments.md` (one row per run).

## Conventions

- Labels are computed, never written by hand or by an LLM (D1). Change a convention in `gen/resolver.py`, `calfn/schema.py` `SYSTEM_PROMPT` and `data/eval/README.md` together, then regenerate train and make an eval v2.
- Never use `eval_v1` or `test_v1` to choose hyperparameters or checkpoints beyond reading results. `test_v1` is scored once (D4).
- Keep `docs/decisions.md` and `docs/experiments.md` current in the same session as the change. Numbers must come from files in `results/`.
- Tests: `.venv/Scripts/python.exe -m pytest` (Windows) / `.venv/bin/python -m pytest`. `tests/test_prompt_format.py` needs transformers and the tokenizer and is skipped without them.
- Training happens in a separate pinned env (`requirements-train.txt`); it runs on the RTX 3060 machine, not necessarily this one. Unload Ollama models first.
