# Experiments

One row per run, added when the run happens. Numbers come from `results/` JSON files; the command that
produced each is in the notes. Nothing below is filled in with a guess.

## Status

| step | state |
|---|---|
| Schema, generator, scorer, converter, tests | done (55 tests pass; 4 more need transformers, see `tests/test_prompt_format.py`) |
| Harness check with oracle answers | done: 300/300 on eval_v1 (`--oracle`), confirms parse + score path |
| Model path smoke test (CPU) | done: base model on the first 8 eval rows loads, generates, stops at `<start_function_response>`, parses and scores (2/8, not a result: 8 rows, CPU, 24 s per row). It ignored the `Now:` line (answered 2024-03-30) and mostly refused, which fine-tuning should fix |
| Base FunctionGemma 270M on eval_v1 | **to run** (needs the GPU machine) |
| Claude baseline on eval_v1 | **to run** (needs `ANTHROPIC_API_KEY`) |
| Fine-tune v1 (D2 recipe) on train.jsonl | **to run** (needs the GPU machine) |
| test_v1, once | after the dev number is settled |

## Data

| file | rows | rejections | notes |
|---|---|---|---|
| `data/train.jsonl` | 3000 | 465 | seed 101, training vocabulary only |
| `data/eval/eval_v1.jsonl` | 300 | 41 | seed 202, 215 `ood` / 85 `ind` |
| `data/eval/test_v1.jsonl` | 200 | 28 | seed 303, 136 `ood` / 64 `ind` |
| `data/eval/natural_v1.jsonl` | 32 | 9 | hand-drafted, unreviewed |

Longest example is 701 tokens with the chat template and tool declaration, so `--max_seq_len 1024`
never truncates a tool call (`tests/test_prompt_format.py`).

## Runs

| run | model | data | config | eval | pass | 95% CI | notes |
|---|---|---|---|---|---|---|---|
| oracle | none (gold answers) | | | eval_v1 | 100% (300/300) | 98.7%-100% | `inference.py --oracle`; harness check only |

Add runs here as they happen:

```
# base model
python scripts/inference.py --model unsloth/functiongemma-270m-it --eval data/eval/eval_v1.jsonl --name base --out-dir results/dev
# Claude baselines
python scripts/baseline_claude.py --eval data/eval/eval_v1.jsonl --model claude-opus-5-5 --name claude-opus-5-5 --out-dir results/dev
python scripts/baseline_claude.py --eval data/eval/eval_v1.jsonl --model claude-haiku-5-5 --name claude-haiku-5-5 --out-dir results/dev
# train (pinned env), then score
python scripts/finetune.py --data data/train.jsonl --output output/calfn-v1
python scripts/inference.py --model output/calfn-v1 --eval data/eval/eval_v1.jsonl --name calfn-v1 --out-dir results/dev
python scripts/summarize.py results/dev
```

## What to look at in each result

- `ood` vs `ind` tags: a big gap means template memorization.
- `reject` and `reject_*` tags, false positives: calling the tool on a vague request is the failure that
  produces a valid but wrong event. (This was the weak spot on the Gridfinity project.)
- `per_field` accuracy for `start` by `date_*` tag: relative-date reasoning is the point of the project.
- `spurious` fields: a model that invents durations or locations.
