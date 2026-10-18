# calfn

Fine-tune [FunctionGemma 270M](https://huggingface.co/google/functiongemma-270m-it) to turn a sentence into a calendar event:

> "yoga every other Tuesday at 6:30pm until Dec 15" → `create_event(title="Yoga", start="2026-10-20T18:30", recurrence_freq="weekly", recurrence_interval=2, recurrence_days="TU", recurrence_until="2026-12-15")`

Plain code (`calfn/convert.py`) turns that call into an `.ics` file and Google/Outlook add-to-calendar links. The point of the project is a small, measurable fine-tune: **every label is computed, not written**, the eval is frozen, and there is a baseline from day one.

## Results

Base model and calfn v1 scored on `eval_v1`; the Claude baseline is not run yet (needs an API key). The harness is verified with oracle answers (300/300 on `eval_v1`). The base model's 40 passes are all correct rejections: it made no correct tool call. See `docs/experiments.md` for the commands and the table to fill in.

| model | dev pass (eval_v1) | 95% CI | held-out vocab (`ood`) | test (once) |
|---|---|---|---|---|
| base FunctionGemma 270M | 13.3% (40/300) | 10.0%-17.6% | 12% | |
| Claude (prompted) | | | | |
| gemma4:12b (prompted, local) | 62.7% (188/300) | 57.1%-68.0% | 62% | |
| calfn v1 (fine-tuned) | 48.3% (145/300) | 42.7%-54.0% | 43% | |

## How it works

```
gen/resolver.py     values -> phrases ("next Monday", "quarter to 5pm", "every other Tuesday")
gen/generate.py     seeded generator for train / eval / test (no LLM)
calfn/schema.py     the create_event tool, system prompt, validation, normalization
calfn/parse.py      FunctionGemma output -> arguments
calfn/scoring.py    exact-event metric, per-field accuracy, Wilson intervals
calfn/convert.py    call -> .ics, Google / Outlook links, share token
scripts/            finetune.py, inference.py (score a model), baseline_claude.py, baseline_ollama.py, summarize.py
data/               train.jsonl, eval/{eval_v1,test_v1,natural_v1}.jsonl  (see data/eval/README.md)
tests/              labeler checked by an independent parser, scorer, converter, prompt format
docs/               decisions.md, experiments.md
```

The model sees a developer message (conventions + `Now: 2026-10-14 Wed 09:00`), the tool declaration, and the request. It outputs one `create_event` call, or no call when the date or time can't be determined (vague, missing, ambiguous am/pm) or the request isn't about creating an event.

## Setup

```bash
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt   # Windows: .venv/Scripts/python.exe
.venv/bin/python -m pytest                                                       # 55 tests, a few seconds (59 with transformers installed)
```

Scoring a model also needs `transformers`, `accelerate`, `peft` and a CUDA `torch`. Training uses its own pinned environment, copied from the Gridfinity project (newer unsloth/peft/torch diverged there):

```bash
uv venv .venv-train --python 3.12 && uv pip install --python .venv-train/bin/python -r requirements-train.txt
```

## Use

```bash
# score the dev eval (greedy decoding, writes results/dev/<name>.json)
python scripts/inference.py --model output/calfn-v1 --eval data/eval/eval_v1.jsonl --name calfn-v1 --out-dir results/dev

# try one request
python scripts/inference.py --model output/calfn-v1 --prompt "lunch with Sam next Friday at noon" --now 2026-10-14T09:00

# baseline: prompted Claude, same prompt, same strict scoring
python scripts/baseline_claude.py --eval data/eval/eval_v1.jsonl --model claude-opus-5-5 --out-dir results/dev

# baseline: a local Ollama model, same prompt and scoring (thinking off unless --think)
python scripts/baseline_ollama.py --eval data/eval/eval_v1.jsonl --model gemma4:12b --name ollama-gemma4-12b --out-dir results/dev

# train (pinned env; unload other GPU users first)
python scripts/finetune.py --data data/train.jsonl --output output/calfn-v1

# convert a call
python -m calfn.convert --args '{"title":"Yoga","start":"2026-10-20T18:30","recurrence_freq":"weekly","recurrence_days":"TU"}' --ics yoga.ics --links
```

`python scripts/summarize.py results/dev` builds the comparison table. `python scripts/inference.py --oracle --eval data/eval/eval_v1.jsonl` checks the scoring path with gold answers (no model needed).

## Rules the project holds itself to

- `eval_v1` is for iterating, `test_v1` is scored once, and nothing is chosen using either (`docs/decisions.md` D4).
- Eval and test are 70% held-out sentence frames, titles and locations; the `ood` vs `ind` gap shows memorization.
- The metric is strict: the same event, or a correct rejection. Extra keys and out-of-range values fail.
- Regenerating data never overwrites the frozen eval files (`gen.generate --force` is for an `eval_v2`).

## Tool

| param | type | required | notes |
|---|---|---|---|
| title | string | yes | |
| start | string | yes | local `YYYY-MM-DDTHH:MM` |
| duration_minutes | int 5-1440 | no | only if the request gives a length or end time (converter default 60) |
| location | string | no | |
| recurrence_freq | daily / weekly / monthly / yearly | no | |
| recurrence_interval | int 1-12 | no | omit for 1 |
| recurrence_days | string | no | weekly: `MO,WE,FR` in week order |
| recurrence_count | int 1-60 | no | total occurrences; not with `until` |
| recurrence_until | string | no | `YYYY-MM-DD`, inclusive |

## Next

Spike a browser demo: export the fine-tuned model to ONNX (the base model has an
[ONNX build](https://huggingface.co/onnx-community/functiongemma-270m-it-ONNX)), run it in a static page with Transformers.js, show the event card with the add-to-calendar buttons, and read `#e=<token>` (`calfn.convert.encode_share`) so a parsed event can be shared as a link. Not started.
