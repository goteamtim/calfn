# calfn

Fine-tune [FunctionGemma 270M](https://huggingface.co/google/functiongemma-270m-it) to turn a sentence into a calendar event. The model **extracts** how the date and time were said; plain code **computes** the real date:

> "yoga every other Tuesday at 6:30pm until Dec 15"
> → model writes `create_event(title="Yoga", time="18:30", recurrence_freq="weekly", recurrence_interval=2, recurrence_days="TU", recurrence_until="12-15")`
> → `calfn/resolve.py`, given the current time, makes the event: first Tuesday after today, 18:30, every 2 weeks until the next Dec 15

`calfn/convert.py` turns the event into an `.ics` file and Google/Outlook add-to-calendar links. The point of the project is a small, measurable fine-tune: **every label is computed, not written**, the eval is frozen, and there is a baseline from day one.

## Why the model doesn't do the date math

The first fine-tune (v1) wrote the absolute date itself and got 48.3% on `eval_v1`. Reading its errors showed it was fine at everything except arithmetic: time of day 98%, stated durations 100%, but dates 65% (usually off by a few days) and durations derived from a range 52%. So v2 moves exactly those two jobs into code. The reasoning, the alternatives rejected, and a prediction written before the v2 run are in `docs/decisions.md` D9.

## Results

All on `eval_v1` (300 prompts, 215 with held-out wording). Same prompts, same labels and same strict metric for every row; `task` is how the model was asked (see D9). The v2 run is not done yet.

| model | task | dev pass | 95% CI | held-out wording (`ood`) | test (once) |
|---|---|---|---|---|---|
| base FunctionGemma 270M | direct | 13.3% (40/300) | 10.0%-17.6% | 12% | |
| base FunctionGemma 270M | extract | 11.7% (35/300) | 8.5%-15.8% | 10% | |
| gemma4:12b (prompted, local) | direct | 62.7% (188/300) | 57.1%-68.0% | 62% | |
| calfn v1 (fine-tuned) | direct | 48.3% (145/300) | 42.7%-54.0% | 43% | |
| Claude (prompted) | direct | | | | |
| calfn v2 (fine-tuned) | extract | 74.3% (223/300) | 69.1%-79.0% | 65% | |
| **calfn v3 (wider vocabulary)** | **extract** | **84.7% (254/300)** | **80.2%-88.3%** | **80%** | |

The base model's 40 passes are all correct rejections: it made no correct tool call. The harness is verified with oracle answers (300/300 on `eval_v1`, both tasks). Commands for the remaining rows are in `docs/experiments.md`.

## How it works

```
calfn/dates.py      the date expression language (wd:TH, in:30d, md:10-21, ...) and calendar arithmetic
calfn/resolve.py    model's call + current time -> the event (the only place dates are computed)
calfn/schema.py     CALL (what the model writes) and EVENT (what is scored/converted), prompts, validation
calfn/task.py       extract (the model writes expressions) vs direct (v1: told "Now", writes the event)
calfn/parse.py      FunctionGemma output -> arguments
calfn/scoring.py    exact-event metric, per-field accuracy, Wilson intervals
calfn/convert.py    event -> .ics, Google / Outlook links, share token
gen/                values -> phrases ("next Monday", "quarter to 5pm") with their labels; seeded, no LLM
scripts/            finetune.py, inference.py (score a model), baseline_claude.py, baseline_ollama.py, summarize.py
data/               train.jsonl, eval/{eval_v1,test_v1,natural_v1}.jsonl  (see data/eval/README.md)
tests/              labeler checked by an independent parser, resolver, scorer, converter, prompt format
docs/               decisions.md, experiments.md
```

The model sees a short developer message, the tool declaration and the request. It is never told the date. It outputs one `create_event` call, or no call when the date or time can't be determined (vague, missing, ambiguous am/pm) or the request isn't about creating an event.

## Setup

```bash
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt   # Windows: .venv/Scripts/python.exe
.venv/bin/python -m pytest                                                       # 137 tests, ~10 s (142 with transformers installed)
```

Scoring a model also needs `transformers`, `accelerate`, `peft` and a CUDA `torch`. Training uses its own pinned environment, copied from the Gridfinity project (newer unsloth/peft/torch diverged there):

```bash
uv venv .venv-train --python 3.12 && uv pip install --python .venv-train/bin/python -r requirements-train.txt
```

## Use

```bash
# train v2 (pinned env; unload other GPU users first)
python scripts/finetune.py --data data/train.jsonl --output output/calfn-v2

# score the dev eval (greedy decoding, writes results/dev/<name>.json)
python scripts/inference.py --model output/calfn-v2 --eval data/eval/eval_v1.jsonl --name calfn-v2 --out-dir results/dev

# a v1 checkpoint wrote the event itself; score it with --task direct
python scripts/inference.py --model output/calfn-v1 --task direct --eval data/eval/eval_v1.jsonl --name calfn-v1 --out-dir results/dev

# try one request: shows the call the model wrote and the event it resolves to
python scripts/inference.py --model output/calfn-v2 --prompt "lunch with Sam next Friday at noon" --now 2026-10-14T09:00

# baselines: prompted Claude / a local Ollama model (--task direct by default; --task extract gives them our prompt)
python scripts/baseline_claude.py --eval data/eval/eval_v1.jsonl --model claude-opus-5-5 --out-dir results/dev
python scripts/baseline_ollama.py --eval data/eval/eval_v1.jsonl --model gemma4:12b --name ollama-gemma4-12b --out-dir results/dev

# convert an event
python -m calfn.convert --args '{"title":"Yoga","start":"2026-10-20T18:30","recurrence_freq":"weekly","recurrence_days":"TU"}' --ics yoga.ics --links
```

`python scripts/summarize.py results/dev` builds the comparison table. `python scripts/inference.py --oracle --eval data/eval/eval_v1.jsonl` checks the scoring path with correct answers (no model needed).

## Rules the project holds itself to

- `eval_v1` is for iterating, `test_v1` is scored once, and nothing is chosen using either (`docs/decisions.md` D4). The v2 design was read off v1's errors on `eval_v1`, so its dev score is optimistic; the test score is the one to report (D9).
- Eval and test are 70% held-out sentence frames, titles and locations; the `ood` vs `ind` gap shows memorization.
- The metric is strict: the same event, or a correct rejection. Extra keys, impossible dates and out-of-range values fail.
- Regenerating data never overwrites the frozen eval files, and the tests check they still reproduce byte for byte (`gen.generate --force` is for an `eval_v2`).

## What the model writes (the call)

| param | type | required | notes |
|---|---|---|---|
| title | string | yes | |
| time | string | yes | start time as said, 24-hour `HH:MM` |
| date | string | usually | a date expression (`wd:TH`, `in:30d`, `md:10-21`, ...; table in `data/eval/README.md`). Omitted only for a daily/weekly series with no start date |
| end | string | no | `HH:MM`, only for a stated end time or range; code computes the duration |
| duration_minutes | int 5-1440 | no | only for a stated length (converter default 60) |
| location | string | no | |
| recurrence_freq | daily / weekly / monthly / yearly | no | |
| recurrence_interval | int 1-12 | no | omit for 1 |
| recurrence_days | string | no | weekly: `MO,WE,FR` in week order |
| recurrence_count | int 1-60 | no | total occurrences; not with `until` |
| recurrence_until | string | no | `MM-DD` (next occurrence) or `YYYY-MM-DD`, inclusive |

The resolved event (what is scored, converted and shared) has an absolute `start` (`YYYY-MM-DDTHH:MM`) instead of `date` and `time`/`end`.

## Next

- Run v2 and record it (`docs/experiments.md`), then score `test_v1` once.
- The remaining v1 errors that D9 doesn't touch: title wording (the sentence frame leaking into the title, invented titles) and digits in titles. That needs more training sentence frames and titles.
- A browser demo: export the fine-tuned model to ONNX (the base model has an [ONNX build](https://huggingface.co/onnx-community/functiongemma-270m-it-ONNX)), run it in a static page with Transformers.js, resolve the call in JS (port of `calfn/dates.py`), show the event card with the add-to-calendar buttons, and read `#e=<token>` (`calfn.convert.encode_share`) so an event can be shared as a link. Not started.
