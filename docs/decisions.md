# Decisions

Dated, numbered, never deleted; a reversal is a new entry. Written at the start so the reasoning
is on record before any result can bias it.

## D1: Labels are computed, not written (2026-10-17)

**Decision.** The generator samples a value (a date, a time, a recurrence) and renders a phrase that
means it, instead of writing a phrase and labeling it. An independent parser (`tests/refparse.py`)
re-parses every rendered phrase and the tests compare it with the generator.

**Why.** The Gridfinity project's eval labels were drafted by an LLM and needed a review that never
happened, and its training-data generator silently dropped or mislabeled whole categories (D12 there).
Here a label can be wrong only if the generator and the reference parser are wrong in the same way.

**Checked.** Two deliberate bugs in the labeler (off-by-one on "strictly after today", a wrong
"quarter to" hour) were both caught by the tests.

**Not covered.** The conventions themselves (`data/eval/README.md`) are choices, not facts. "next
Thursday" has no single right answer; the generator avoids the ambiguous cases.

## D2: One recipe, taken from Gridfinity (2026-10-17)

**Decision.** Full fine-tune in bf16, lr 5e-5, 6 epochs, final epoch kept, effective batch 16
(`scripts/finetune.py` defaults), on the pinned stack in `requirements-train.txt`.

**Why.** On the Gridfinity project, full fine-tuning at 5e-5 beat LoRA r=16 and tied LoRA r=64 on the
bigger dataset, and 1e-5 barely trained. Reusing that recipe avoids a hyperparameter search on a
200-row dev set.

**Untested here.** This is the first run on this task. If it underfits, the Gridfinity evidence says
try `--lora_rank 64 --lr 1e-4` before changing anything else.

## D3: Flat tool schema (2026-10-17)

**Decision.** Recurrence is five flat fields (`recurrence_freq`, `_interval`, `_days`, `_count`,
`_until`), not one nested object, and `recurrence_days` is a comma-separated string, not an array.

**Why.** FunctionGemma's call format (`key:<escape>str<escape>,key:123`) and the parser were only
exercised on scalars in Gridfinity. Nested values would add an untested failure mode to a project
meant to stay small. The converter turns the flat fields into an `RRULE` in code.

## D4: Three evaluation sets with different rules (2026-10-17)

**Decision.** `eval_v1` (300) is for iterating, `test_v1` (200) is scored once at the end, and
`natural_v1` (32, hand-written) is reported separately until a human has reviewed it. No validation
set: the final epoch is kept, so nothing is chosen using eval or test data.

**Why.** On Gridfinity, hyperparameters were chosen on the same 200 rows that produced the headline
number, which makes that number optimistic. Here the headline is the test score, and it is
produced once.

**Held-out vocabulary.** 70% of eval and test use sentence frames, titles, locations and off-topic
prompts that training never sees, with the other 30% from training vocabulary. A result is only
trusted if the `ood` and `ind` slices are close.

## D5: Strict metric (2026-10-17)

**Decision.** A prediction passes only if it describes the same event after normalization (defaults
equal omitted values, titles/locations compare ignoring case and punctuation), or the model
correctly makes no call. Unknown keys, wrong types and out-of-range values fail.

**Why.** Gridfinity's scorer accepted divisions of 20 against a maximum of 8 and truncated `2.7` to
`2`. A calendar event that is 90 minutes instead of 60 is a wrong event. There is no partial credit
in the headline; per-field accuracy is reported alongside it.

## D6: One chat-template bos (2026-10-17)

**Decision.** `format_example` and `build_prompt` both strip the template's leading `<bos>`; the
tokenizer adds one.

**Why.** Checked against the real tokenizer: tokenizing the template output gives `[2, 2, ...]`
(two BOS tokens). The Gridfinity scripts did this in both training and inference, so it was
consistent there, but one BOS is the input the model was pretrained on.
`tests/test_prompt_format.py` verifies training text and inference prompt match.

## D7: Times are floating in the model and the converter (2026-10-17)

**Decision.** The model outputs local wall-clock time with no timezone, and the "Now" line is local
too. `calfn.convert` produces floating `.ics` times by default; with `--tz` it produces UTC for
one-offs and a `TZID` parameter for repeating events.

**Why.** A 270M model should not do timezone arithmetic, and a personal calendar wants "3pm" to
stay 3pm. `TZID` without a `VTIMEZONE` block is accepted by common clients but not guaranteed by
the standard; floating is the safe default.

## D8: Claude baseline uses the same prompt and strict scoring (2026-10-17)

**Decision.** `scripts/baseline_claude.py` sends each example's developer message (conventions plus
the Now line) as the system prompt with the same tool, `tool_choice: auto` (forced tool use is
unavailable on current models), and scores the result with the same strict rules.

**Why.** A baseline that isn't told the conventions would be graded against rules it couldn't know.
The prompt is deliberately given to both.

## D9: The model extracts, code computes (2026-10-18)

**Decision.** The model no longer writes the event. It copies how the date and time were said into a
short **date expression** (`wd:TH`, `in:30d`, `md:10-21`, `nth:L:FR:1`, ...; the full language is in
`calfn/dates.py`), plus the clock time (`15:00`) and either a stated length (`duration_minutes`) or an
end time (`end`). `calfn/resolve.py` turns that call into the real event given the current time. The
model is not told what day it is. The v1 approach (told "Now: ...", writes the absolute event) survives
as `--task direct`, used by the prompted baselines.

**Why.** The first fine-tune (calfn-v1, direct, 48.3% on eval_v1) failed almost entirely at arithmetic,
not understanding. Measured from `results/dev/calfn-v1.json`:

| piece | accuracy | where it lives now |
|---|---|---|
| time of day | 98.4% (245/249 calls) | model (unchanged) |
| stated duration ("for 90 minutes") | 100% (101/101) | model (unchanged) |
| date | 64.7% (161/249) | **code** |
| duration derived from a range or "until" | 52% (24/46) | **code** (model writes the end time) |

86 of the 137 wrong calls had a wrong start date (79 were wrong in nothing else), mostly by a few days
(-1d x16, -4d x13, +2d x11): "in 30 days" landed on day 28. Weekday-name prompts passed 14% of the time. A 270M model approximates
calendar arithmetic and misses by small amounts; a resolver does it exactly. A prompted 12B model got
85% on dates, so the skill is learnable at scale, but this project is about what a small model can do
reliably, and the production-grade design is to not ask it for arithmetic at all.

**What changed.**
- `CALL` (what the model writes) and `EVENT` (what gets scored, converted and shared) are separate
  schemas in `calfn/schema.py`. Scoring resolves the call with the example's `meta.now`, then compares
  events exactly as before, so v1 and v2 numbers are comparable.
- Generator: every date phrase carries the expression for it, and the generator **asserts for every row
  that the call resolves to the expected event**. An independent parser checks phrase -> date and the
  resolver checks expression -> date, so a phrase and its expression can't disagree
  (`tests/test_dates_resolve.py`).
- `data/train.jsonl` is regenerated with call labels (same 3000 prompts and events as v1, checked).
  `eval_v1` / `test_v1` are untouched: still byte-identical to what the generator makes, and still hold
  the resolved event as the answer.
- A daily or weekly series with no stated start date has no `date` at all; the resolver finds its first
  occurrence. Monthly and yearly series carry their day in `date` ("every month on the 15th" is `dom:15`).
- `recurrence_until` is `MM-DD` or `YYYY-MM-DD`; `MM-DD` means its next occurrence, resolved in code.

**Rejected.**
- *More data, epochs or LoRA rank for the arithmetic.* Cheap to try, but it keeps asking a small model
  for exact arithmetic and would still leave rare dates wrong. Better data diversity is still wanted
  for the title errors (below); that is a separate change.
- *Copy the date phrase verbatim and parse it in code (dateparser / parsedatetime).* Open-ended parsers
  have their own conventions ("next Thursday") that don't match ours, and a verbatim copy hands the
  hard part to a library we can't control. The expression language is small, closed, and ours.
- *Let the model reason step by step before answering.* Slower, and the format (a bare tool call)
  stays the same.

**Honesty about the dev set.** The design came from reading calfn-v1's errors on `eval_v1`, so any gain
on `eval_v1` is optimistic. `test_v1` has never been scored or analyzed; it is the number to report.
(Comparing v2 with v1 on `eval_v1` is still apples to apples: same prompts, same labels, same metric.)

**Prediction, written before the first v2 run.** Date accuracy >= 90% and overall pass 70-85% on
`eval_v1`; rejection handling stays above 90%. The remaining misses should be mostly title wording
(30 of v1's 137 wrong calls: the held-out sentence frame leaking into the title, invented titles) and
choosing the wrong date *kind* (`wd:` vs `nw:`). If date accuracy is not close to 90%, the problem is
the expression language or the data, not arithmetic, and this decision needs revisiting.

**Costs.** The tool declaration is longer (826 tokens for the longest training example, up from 701),
and the language has to be learned. It can only express what the language can: a phrase outside it
("the Tuesday after Thanksgiving") has no expression. Extending the language means a new `eval` version.

**Revisit if** dates stay under 90% after the first run, or the language turns out too small for the
phrasings people use.

## Ideas, not decisions

- More training diversity (sentence frames, event titles). calfn-v1 mixed the held-out frame into the title
  ("Throw buy 3 gift cards") and invented titles ("Plummettruf"); 20 frames and 80 titles is thin. Adding
  vocabulary changes the generator's random draws, so eval/test must be generated from separate vocab
  streams first (or versioned as v2).
- Response-only loss (don't train on the developer message and tool declaration). Untested; would be
  one ablation.
- A browser demo via ONNX export (`onnx-community/functiongemma-270m-it-ONNX` exists for the base
  model). Spike the export of the fine-tuned model before committing to it.
- Event-in-the-URL sharing (`calfn.convert.encode_share` / `decode_share` already round-trip a call
  into a short base64url token for a future `page.html#e=...`).
