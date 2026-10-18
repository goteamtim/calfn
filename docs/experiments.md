# Experiments

One row per run, added when the run happens. Numbers come from `results/` JSON files; the command that
produced each is in the notes. Nothing below is filled in with a guess.

## Status

| step | state |
|---|---|
| Schema, generator, scorer, converter, tests | done (55 tests pass; 4 more need transformers, see `tests/test_prompt_format.py`) |
| Harness check with oracle answers | done: 300/300 on eval_v1 (`--oracle`), confirms parse + score path |
| Model path smoke test (CPU) | done: base model on the first 8 eval rows loads, generates, stops at `<start_function_response>`, parses and scores (2/8, not a result: 8 rows, CPU, 24 s per row). It ignored the `Now:` line (answered 2024-03-30) and mostly refused, which fine-tuning should fix |
| Base FunctionGemma 270M on eval_v1 | done: 13.3% (40/300), see Runs |
| Claude baseline on eval_v1 | **to run** (needs `ANTHROPIC_API_KEY`) |
| Fine-tune v1 (D2 recipe) on train.jsonl | done: 48.3% on eval_v1, see Runs. Task `direct`: the model wrote the event itself |
| Switch to extract-then-compute (D9) | code, data and tests done 2026-10-18; oracle 100% in both tasks; frozen eval/test verified byte-identical; **model not trained yet** |
| Fine-tune v2 (`--task extract`) on the regenerated train.jsonl | **to run** (needs the GPU machine). Prediction in D9: date >= 90%, overall 70-85% |
| Base FunctionGemma 270M, `--task extract` | **to run** (v1's base row was `direct`; this is the matching start line) |
| Claude / gemma4:12b with `--task extract` | optional: shows whether the expression language is learnable from a prompt alone |
| test_v1, once | after the v2 dev number is settled |

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
| base | unsloth/functiongemma-270m-it | none | greedy | eval_v1 | 13.3% (40/300) | 10.0%-17.6% | RTX 3060, `results/dev/base.json`. 0 exact; 191/259 real requests missed (no call), 68 invalid, 40/41 rejections correct, 1 false positive; 1088 ms mean. ood 12% / ind 16% |
| calfn-v1 | functiongemma-270m-it, full FT | train.jsonl (3000) | defaults: lr 5e-5, 6 ep, bs 4 x accum 4, seed 42; final train loss 0.0577 | eval_v1 | 48.3% (145/300) | 42.7%-54.0% | RTX 3060, `results/dev/calfn-v1.json`, `runs/train_20261018_093640.json`. 107 exact, 137 wrong_args, 5 invalid, 10 missed; 38/41 rejections ok, 3 false positives. start 64.8%, title 79.1%, duration 84.8%. ood 43% / ind 62%. Weakest: date_weekday 14%, date_rel 29%, time_range 21% |
| ollama-gemma4-12b | gemma4:12b via Ollama, prompted | none | temp 0, thinking off, num_ctx 4096 | eval_v1 | 62.7% (188/300) | 57.1%-68.0% | RTX 3060, `scripts/baseline_ollama.py`, `results/dev/ollama-gemma4-12b.json`. 174 exact, 70 wrong_args, 15 invalid, 0 missed; only 14/41 rejections ok (27 false positives). start 84.8%, title 96.3%. ood 62% / ind 64%. 2928 ms mean |

**Task column.** Every row above is `task=direct` (D9): the v1 approach, where the model is told the current
time and writes the absolute event. Runs from D9 on are `task=extract` and must say so in the model column.
The scores are comparable: same eval prompts, same labels, same metric. The v1 result files predate the
`task` key (and report `start` as one field instead of `start_date` / `start_time`).

**calfn-v1 error breakdown** (`results/dev/calfn-v1.json`; the evidence for D9): of the 137 calls that made the
wrong event, 86 had a wrong start date (79 wrong in nothing else), 34 were wrong only in title or location wording
(30 title, 4 location), and 22 had a wrong duration (all 22 derived from a range or "until"; 7 also had a wrong date).
Time of day was right in 245 of 249 calls, a stated duration in 101 of 101, a derived duration in 24 of 46. By tag, v1 passed `date_weekday` 14%, `date_rel` 29%, `time_range` 21%, `distractor` 20% (digits in
titles) and rejections 93%; gemma4:12b prompted passed 52%, 89%, 67%, 85% and 34%.

Add runs here as they happen:

```
# base model
python scripts/inference.py --model unsloth/functiongemma-270m-it --eval data/eval/eval_v1.jsonl --name base --out-dir results/dev
# Claude baselines
python scripts/baseline_claude.py --eval data/eval/eval_v1.jsonl --model claude-opus-5-5 --name claude-opus-5-5 --out-dir results/dev
python scripts/baseline_claude.py --eval data/eval/eval_v1.jsonl --model claude-haiku-5-5 --name claude-haiku-5-5 --out-dir results/dev
# v2: train on the regenerated (call-labelled) train.jsonl, then score (the default task is extract)
python scripts/finetune.py --data data/train.jsonl --output output/calfn-v2
python scripts/inference.py --model output/calfn-v2 --eval data/eval/eval_v1.jsonl --name calfn-v2 --out-dir results/dev
python scripts/inference.py --model unsloth/functiongemma-270m-it --eval data/eval/eval_v1.jsonl --name base-extract --out-dir results/dev
python scripts/summarize.py results/dev
# v1 (kept for reference): the checkpoint wrote the event itself
python scripts/inference.py --model output/calfn-v1 --task direct --eval data/eval/eval_v1.jsonl --name calfn-v1 --out-dir results/dev
```

## What to look at in each result

- `ood` vs `ind` tags: a big gap means template memorization.
- `reject` and `reject_*` tags, false positives: calling the tool on a vague request is the failure that
  produces a valid but wrong event. (This was the weak spot on the Gridfinity project.)
- `per_field` accuracy for `start_date` by `date_*` tag: this is what D9 changed. Check it first against the
  prediction (>= 90%). If a `date_*` tag is still low, look at the raw output: a wrong *kind* (`wd:` vs `nw:`) is a
  language/data problem, a malformed expression is a training problem.
- `invalid` in the breakdown: calls the resolver refuses (malformed expression, impossible date). New with D9.
- `spurious` fields: a model that invents durations or locations.
