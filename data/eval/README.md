# Eval sets

| file | rows | how it's made | use |
|---|---|---|---|
| `eval_v1.jsonl` | 300 (259 calls, 41 rejections) | generated, labels computed | **dev**: iterate against this |
| `test_v1.jsonl` | 200 (172 calls, 28 rejections) | generated, labels computed | **test**: score once, at the end |
| `natural_v1.jsonl` | 32 (23 calls, 9 rejections) | hand-drafted | extra slice, **not yet reviewed** by a human; see `natural_v1_review.md` |

`eval_v1` and `test_v1` are frozen. `python -m gen.generate` refuses to overwrite them. If a
labeling mistake turns up, fix it in an `eval_v2` / `test_v2` and re-score everything; don't edit v1.

## What a row holds

The answer in every eval row is the **resolved event** (absolute start, duration, recurrence), whichever
task a model is solving. A model's output is turned into an event with the row's `meta.now` before it
is compared (`calfn/resolve.py`), so a model that writes date expressions (`--task extract`, the
default) and one that writes the event itself (`--task direct`, the v1 approach and the prompted
baselines) are scored by the same rule against the same labels. The developer message stored in these
rows is the direct task's ("Now: ..."); the scripts build the right prompt per task and ignore it.
`train.jsonl` is different: its answers are the model's training target (the call), and `meta.event`
holds the resolved event.

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

These are the rules every expected answer follows. `calfn/dates.py` implements them (the model never
does this arithmetic, see `docs/decisions.md` D9) and `calfn/schema.py` `DIRECT_SYSTEM_PROMPT` states
them for the `direct` baselines.

The date language the model writes, with what each means on a Wednesday, 2026-10-14:

| expression | means | example phrases |
|---|---|---|
| `in:0d` `in:1d` `in:30d` `in:3w` | N days or weeks from today | today (2026-10-14), tomorrow, in 30 days, in three weeks |
| `wd:TH` | first Thursday strictly after today (2026-10-15) | Thursday, this Thursday, next Thursday |
| `wd:TH:2` | the one after that (2026-10-22) | a week from Thursday |
| `nw:FR` | Friday in the next Mon-Sun week (2026-10-23) | Friday next week, Friday of next week |
| `md:10-21` | next 10/21 on or after today (2026-10-21) | Oct 21, 10/21, the 21st of October |
| `md:2027-10-21` | that exact date | October 21, 2027, 10/21/2027 |
| `dom:21` | next 21st on or after today (2026-10-21) | the 21st |
| `eom:0` `eom:1` | last day of this / next month (2026-10-31 / 2026-11-30) | the end of the month, end of next month |
| `nth:2:TU:1` `nth:L:FR:0` | 2nd Tuesday of next month / last Friday of this month | the second Tuesday of next month |
| `dnm:15` | the 15th of next month (2026-11-15) | the 15th of next month |

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
  The model writes a stated length as `duration_minutes` and an end time or range as `end`
  (`"end": "16:30"`); code computes the duration, so the event is the same either way.

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

`scripts/inference.py` reports **pass rate** = the call resolves to the same event as the expected one
(after `normalize()`: an explicit default equals an omitted one), or the model correctly makes no
call. It also reports per-field accuracy for fields the request states (`start_date` and `start_time`
separately, so a wrong date can't hide behind a right time), the count of *spurious* extra fields,
rejection precision/recall, and a 95% Wilson interval. A call that is malformed, impossible (Feb 30),
or resolves to an invalid event (a series ending before it starts) fails as `invalid`.
