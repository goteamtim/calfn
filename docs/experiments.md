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
| Switch to extract-then-compute (D9) | code, data and tests done 2026-10-18; oracle 100% in both tasks; frozen eval/test verified byte-identical |
| Fine-tune v2 (`--task extract`) on the regenerated train.jsonl | done: 74.3% on eval_v1, see Runs. D9 prediction (date >= 90%, overall 70-85%, rejections > 90%) held |
| Base FunctionGemma 270M, `--task extract` | done: 11.7% (35/300), see Runs. The matching start line for v2 |
| Fine-tune v3 (D10: wider train vocabulary, same recipe) | done: 84.7% on eval_v1, see Runs. D10 prediction (title >= 85%, ood >= 78%, overall 80-90%) held |
| Claude / gemma4:12b with `--task extract` | optional: shows whether the expression language is learnable from a prompt alone |
| test_v1, once | done 2026-10-18 for calfn-v3 only: 85.0% (170/200), see Runs. Not to be scored again for this model or used to choose anything (D4) |

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
| calfn-v2 (extract) | functiongemma-270m-it, full FT | train.jsonl (3000, call labels) | defaults: lr 5e-5, 6 ep, bs 4 x accum 4, seed 42; final train loss 0.0309 | eval_v1 | 74.3% (223/300) | 69.1%-79.0% | RTX 3060, `results/dev/calfn-v2.json`, `runs/train_20261018_130551.json`. 182 exact, 71 wrong_args, 2 invalid, 4 missed; 41/41 rejections ok, 0 false positives. start_date 100% (253/253), start_time 98.8%, title 72.7%, location 98.3%, duration 99.3%. ood 65% / ind 98%. 1041 ms mean |
| calfn-v3 (extract) | functiongemma-270m-it, full FT | train.jsonl (3000, D10 vocabulary: 346 titles, 58/23/20 frames) | defaults, same as v2; final train loss 0.0332 | eval_v1 | 84.7% (254/300) | 80.2%-88.3% | RTX 3060, `results/dev/calfn-v3.json`, `runs/train_20261018_150939.json`. 213 exact, 40 wrong_args, 2 invalid, 4 missed; 41/41 rejections ok, 0 false positives. start_date 100%, start_time 97.6%, title 88.5%, location 95.1%, duration 98.0%. ood 80% (v2 65%) / ind 96% (v2 98%). 1044 ms mean |
| calfn-v3 (extract), **test_v1, scored once** | same model as the calfn-v3 row | | | test_v1 | 85.0% (170/200) | 79.4%-89.3% | `results/test/calfn-v3-test.json`. 144 exact, 22 wrong_args, 0 invalid, 6 missed; 26/28 rejections ok, 2 false positives. start_date 100%, start_time 100%, title 87.9%, location 97.9%, duration 100%. ood 80% (109/136) / ind 95% (61/64). 1050 ms mean |
| base-extract | unsloth/functiongemma-270m-it, `extract` prompt | none | greedy | eval_v1 | 11.7% (35/300) | 8.5%-15.8% | RTX 3060, `results/dev/base-extract.json`. 0 exact; 157/259 real requests missed, 102 invalid, 35/41 rejections ok, 6 false positives; 1209 ms mean. ood 10% / ind 16%. All 35 passes are rejections |

**Task column.** Every row above is `task=direct` (D9): the v1 approach, where the model is told the current
time and writes the absolute event. Runs from D9 on are `task=extract` and must say so in the model column.
The scores are comparable: same eval prompts, same labels, same metric. The v1 result files predate the
`task` key (and report `start` as one field instead of `start_date` / `start_time`).

**calfn-v1 error breakdown** (`results/dev/calfn-v1.json`; the evidence for D9): of the 137 calls that made the
wrong event, 86 had a wrong start date (79 wrong in nothing else), 34 were wrong only in title or location wording
(30 title, 4 location), and 22 had a wrong duration (all 22 derived from a range or "until"; 7 also had a wrong date).
Time of day was right in 245 of 249 calls, a stated duration in 101 of 101, a derived duration in 24 of 46. By tag, v1 passed `date_weekday` 14%, `date_rel` 29%, `time_range` 21%, `distractor` 20% (digits in
titles) and rejections 93%; gemma4:12b prompted passed 52%, 89%, 67%, 85% and 34%.

**calfn-v2 error breakdown** (`results/dev/calfn-v2.json`): of the 71 calls that made the wrong event, 65 were wrong only
in the title, 2 in title and location, 2 in title and start, 1 only in location, 1 in start and duration. Every date
the model wrote resolved correctly. Title failures are the held-out sentence frame leaking into the title ("Throw
buy 3 gift cards", "Make change", "Pencil in breakfast with Alex") and invented titles ("Plum in the Room"). The big
`ood` / `ind` gap (65% vs 98%) is this: on training vocabulary it is near perfect, so titles and locations
for unseen frames and words are the remaining problem. `loc` 67% and `distractor` 50% are the same issue.

**calfn-v3 error breakdown** (`results/dev/calfn-v3.json`): 40 wrong calls, down from 71. 26 are wrong only in the title
(v2: 65), 4 in title plus one other field. Every date the model wrote still resolved correctly. The title errors that
are left are mostly the held-out lead-ins that D10 deliberately kept out of training: "Pencil in brunch", "Throw
supper with the parkers", "Throw a trivia night". Lead-ins the model has seen no longer leak. The rest are
invented or reworded titles ("Plum meeting", "Refresh my prescription"), cut-off titles ("Standing desk") and case
or spelling slips ("Call aunt rosa", "Tommas"). New non-title errors are few and small: "quarter to N" resolved an
hour late in 4 rows (it wrote N:45 instead of N-1:45; one more "quarter past" row and one "2:30-6:30pm" range also got the time wrong), a `recurrence_until` year/day slip in 2, location dropped or respelled in 4, 2 malformed calls
(`invalid`) and 4 rows with no call. By tag, v2 -> v3: `loc` 67% -> 80%, `distractor` 50% -> 80%, `time_range`
54% -> 83%, `date_weekday` 62% -> 72%, `recurring` 73% -> 87%. `ind` fell 98% -> 96% (2 rows).

**natural_v1** (32 hand-drafted rows, labels unreviewed; `results/natural/calfn-v2-natural.json`, `calfn-v3-natural.json`):
calfn-v2 71.9% (23/32, CI 54.6%-84.4%), calfn-v3 68.8% (22/32, CI 51.4%-82.0%). Dates and times are 20/20 for both. Every
wrong call is a title, 9/9 rejections are right for both, and both miss the same 3 requests (empty output on "Can you put
lunch with Sam on my calendar for Friday at noon?" and "Doctor's appointment Thursday Oct 22 at 11am", a refusal on
"pick up the kids at 3:15 today"). v2 and v3 differ on 5 rows that cancel out (v2 alone exact on 3, v3 alone on 2). With
n=32 this can neither confirm nor rule out the +10 points seen on eval_v1. Several "errors" are label-convention
questions in an unreviewed set: "dentist" vs "Dentist appointment", "Lunch w/ Dad" vs "Lunch with Dad", "Daily standup"
vs "Standup", "Meeting with Priya" vs "A meeting with Priya". Reviewing `natural_v1` (or an eval v2 with fresh held-out
vocabulary) is needed before it can say anything about title wording.

**calfn-v3 on test_v1** (read after scoring, not used for any decision): 22 wrong calls, 20 of them wrong only in the title,
1 in location, 1 in recurrence days and frequency. Every date and time resolved correctly (166/166 each). The 6 misses
are empty outputs on plain requests ("Set up dentist appointment on 10/1 at 8pm", "Need mortgage signing on Sunday next
week at 14:00"). The 2 false positives are "I have design review on Feb 30 at 2pm" (a date that does not exist) and "supper
with the Parkers at 09:45" (no date). This is close to the eval_v1 result (84.7%), so the dev number was not badly
optimistic, but `eval_v1` and `test_v1` come from the same generator and vocabulary, so neither measures people's own phrasing.

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
