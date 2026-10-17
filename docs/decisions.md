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

## Ideas, not decisions

- Response-only loss (don't train on the developer message and tool declaration). Untested; would be
  one ablation.
- A browser demo via ONNX export (`onnx-community/functiongemma-270m-it-ONNX` exists for the base
  model). Spike the export of the fine-tuned model before committing to it.
- Event-in-the-URL sharing (`calfn.convert.encode_share` / `decode_share` already round-trip a call
  into a short base64url token for a future `page.html#e=...`).
