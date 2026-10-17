# Eval sets

| file | rows | how it's made | use |
|---|---|---|---|
| `eval_v1.jsonl` | 300 (259 calls, 41 rejections) | generated, labels computed | **dev**: iterate against this |
| `test_v1.jsonl` | 200 (172 calls, 28 rejections) | generated, labels computed | **test**: score once, at the end |
| `natural_v1.jsonl` | 32 (23 calls, 9 rejections) | hand-drafted | extra slice, **not yet reviewed** by a human; see `natural_v1_review.md` |

`eval_v1` and `test_v1` are frozen. `python -m gen.generate` refuses to overwrite them. If a
labeling mistake turns up, fix it in an `eval_v2` / `test_v2` and re-score everything; don't edit v1.

## How they are built

- Every label is **computed**: the generator picks a date, time, duration, recurrence, title and
  location as values, then writes a sentence that means exactly that. Nothing is parsed, and no LLM
  writes labels. `tests/refparse.py` is a second, independent implementation that parses every
  generated date, time, duration and recurrence phrase back to its value; `tests/test_resolver.py`
  compares the two.
- **70% held-out vocabulary** (tag `ood`): sentence frames, event titles, locations and off-topic prompts
  that never appear in `train.jsonl`. **30% training vocabulary** (tag `ind`): the same words with new
  dates and combinations. The gap between `ood` and `ind` shows how much a model memorized templates.
- No user text in either file appears in `train.jsonl`, and no `test_v1` text appears in `eval_v1`
  (`tests/test_generate.py` checks this).
- Every example carries `meta.tags` for slicing results: date style (`date_rel`, `date_weekday`,
  `date_abs`, `date_dom`, `date_monthrel`, `date_today`), time style (`time_12h`, `time_24h`,
  `time_noon`, `time_words`, `time_period`, `time_range`), `dur`, `dur_until`, `dur_range`, `loc`,
  `distractor`, `recurring` + `rec_*`, and `reject` + `reject_*`.

## Labeling conventions

These are the rules every expected answer follows. They are also in `calfn/schema.py` `SYSTEM_PROMPT`,
so a prompted baseline is told the same things the fine-tuned model learns from data.

**Dates**
- A weekday name means the first such day **strictly after today**: `Thursday`, `this Thursday` and
  `next Thursday` all mean the same day. `this W` is only generated when that day is in the current
  Mon-Sun week, and `next W` only when it is in the next one, so the two readings never disagree.
- `W next week` is that weekday in the next Mon-Sun week. `a week from W` is the first `W` plus 7 days.
- A month and day (`Oct 21`, `10/21`) or a day of the month (`the 21st`) is its **next occurrence on or
  after today**. Numeric dates are month/day (US). A year is only given when needed or explicit.
- `end of the month`, `the last Friday of next month`, `the 15th of next month`, `in 3 days`,
  `in 2 weeks`, `tomorrow`, `the day after tomorrow` mean what they say.

**Times**
- Local wall-clock time, 24-hour `YYYY-MM-DDTHH:MM`, no timezone. `noon` is 12:00.
- A bare hour with no am/pm and no 24-hour cue (`at 3`, `at 10:30`) is **ambiguous, so reject**.
- `3 in the afternoon`, `quarter to 5pm`, `half past 2pm`, `15:00`, `3-4:30pm` are all explicit.

**Duration**
- `duration_minutes` only when the request gives a length (`for 90 minutes`), an end time
  (`until 4pm`) or a range (`from 3pm to 4:30pm`). Otherwise omit it (the converter assumes 60).

**Repeating events**
- `recurrence_freq` is one of daily, weekly, monthly, yearly. `recurrence_interval` only when it isn't 1
  (`every other Tuesday` is weekly, interval 2). `recurrence_days` is always set for weekly events
  (`MO,WE`, in week order; weekdays = `MO,TU,WE,TH,FR`).
- No start date given: the series starts at its first occurrence **strictly after today** (daily:
  tomorrow). Monthly and yearly use the month/day rule above.
- `recurrence_count` is the total number of occurrences (`for 6 weeks` on a single-day weekly event is 6;
  `10 times` is 10). `recurrence_until` is the last date, inclusive. Never both.

**Titles and locations**
- The title is what the user called the event, first letter capitalized. The location is the noun phrase
  after the preposition, as written (`the library`). Both are compared ignoring case and punctuation.
- Numbers inside a title or location (`dinner for 4`, `Room 204`, `1:1 with Jordan`) are not times
  (tag `distractor`).

**Rejections** (no tool call; the reply text isn't scored)
- No time (`dentist Friday`), no date (`dentist at 3pm`), a vague time (`sometime next week`,
  `tomorrow morning`, `tonight`), a bare hour (`tomorrow at 3`), a time in the past (`yesterday`,
  `last Tuesday`), a date that doesn't exist (`February 30`), or a request that isn't about creating an
  event (`what's on my calendar tomorrow?`, `cancel my 3pm`).

## Scoring

`scripts/inference.py` reports **pass rate** = the prediction describes the same event (after
`normalize()`: an explicit default equals an omitted one), or the model correctly makes no call. It
also reports per-field accuracy for fields the request states, the count of *spurious* extra
fields, rejection precision/recall, and a 95% Wilson interval. An invalid call (unknown key, wrong
type, out of range) fails.
