"""
The labeler. Dates, times, durations and recurrence are sampled as VALUES first, then
rendered into phrases that mean exactly that value. No phrase is ever parsed, so labels are
right by construction (tests/date_parser.py re-parses every phrase independently as a check).

Conventions (also in data/eval/README.md and calfn/schema.py SYSTEM_PROMPT):
  - A weekday name always means the first such day STRICTLY after today ("Thursday",
    "this Thursday", "next Thursday"). "this W" is only generated when that day falls in the
    current Mon-Sun week and "next W" only when it falls in the next one, so the two readings
    of "next Thursday" never disagree.
  - "W next week" is that weekday in the next Mon-Sun week. "a week from W" is the first W + 7.
  - A month and day ("Oct 21", "10/21") or a day of the month ("the 21st") is its next
    occurrence on or after today. Numeric dates are month/day (US).
  - A repeating event with no start date starts on its first occurrence strictly after today
    (daily: tomorrow); monthly and yearly use the day-of-month rule above.
"""

import calendar
import random
from dataclasses import dataclass
from datetime import date, time, timedelta

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
WEEKDAY_ABBR = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
DAY_CODES = ["MO", "TU", "WE", "TH", "FR", "SA", "SU"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]
MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
NUM_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
             8: "eight", 9: "nine", 10: "ten"}
ORD_WORDS = {1: "first", 2: "second", 3: "third", 4: "fourth"}


# ---------------------------------------------------------------- date arithmetic

def ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def week_start(d: date) -> date:
    return d - timedelta(days=d.weekday())


def last_day(year: int, month: int) -> int:
    return calendar.monthrange(year, month)[1]


def add_months(year: int, month: int, k: int) -> tuple[int, int]:
    idx = year * 12 + (month - 1) + k
    return idx // 12, idx % 12 + 1


def next_weekday(today: date, wd: int) -> date:
    """First date strictly after today that falls on weekday wd (0=Monday)."""
    return today + timedelta(days=(wd - today.weekday()) % 7 or 7)


def next_month_day(today: date, month: int, day: int) -> date:
    """Next occurrence of month/day on or after today."""
    for year in range(today.year, today.year + 9):
        try:
            cand = date(year, month, day)
        except ValueError:
            continue
        if cand >= today:
            return cand
    raise ValueError(f"no occurrence of {month}/{day}")


def next_day_of_month(today: date, day: int) -> date:
    """Next date on or after today whose day of month is `day`."""
    year, month = today.year, today.month
    for _ in range(14):
        if day <= last_day(year, month):
            cand = date(year, month, day)
            if cand >= today:
                return cand
        year, month = add_months(year, month, 1)
    raise ValueError(f"no day {day}")


def num_text(n: int, rng: random.Random) -> str:
    """Digits, or a word for 1-10 about half the time."""
    if n in NUM_WORDS and rng.random() < 0.5:
        return NUM_WORDS[n]
    return str(n)


# ---------------------------------------------------------------- date phrases

@dataclass
class DatePhrase:
    kind: str   # family: today, rel, weekday, abs, dom, monthrel
    text: str   # "next Thursday"
    on: str     # same, with "on " where English wants it: "on Thursday"


FAMILY_WEIGHT = {"today": 1.0, "rel": 3.0, "weekday": 4.0, "abs": 3.0, "dom": 1.0, "monthrel": 1.5}


def date_candidates(target: date, today: date, rng: random.Random) -> dict[str, list[DatePhrase]]:
    """Every phrasing (grouped by family) that resolves to `target` when said on `today`."""
    delta = (target - today).days
    if delta < 0:
        raise ValueError("target is in the past")
    out: dict[str, list[DatePhrase]] = {}

    def add(kind, text, on=None):
        out.setdefault(kind, []).append(DatePhrase(kind, text, on if on is not None else text))

    wd = target.weekday()
    wname = WEEKDAYS[wd]

    if delta == 0:
        add("today", "today")
    if delta == 1:
        add("rel", "tomorrow")
    if delta == 2:
        add("rel", "the day after tomorrow")
    if 2 <= delta <= 30:
        add("rel", f"in {num_text(delta, rng)} days")
    if delta in (7, 14, 21, 28, 35, 42, 49, 56):
        weeks = delta // 7
        add("rel", "in a week" if weeks == 1 else f"in {num_text(weeks, rng)} weeks")

    if delta >= 1:
        first = next_weekday(today, wd)
        this_week = week_start(today)
        if target == first:
            add("weekday", wname, f"on {wname}")
            if week_start(first) == this_week:
                add("weekday", f"this {wname}")
            if week_start(first) == this_week + timedelta(days=7):
                add("weekday", f"next {wname}")
        if target == this_week + timedelta(days=7 + wd):
            add("weekday", f"{wname} next week", f"on {wname} next week")
            add("weekday", f"{wname} of next week", f"on {wname} of next week")
        if target == first + timedelta(days=7):
            add("weekday", f"a week from {wname}")

    # month + day
    m, d = target.month, target.day
    needs_year = next_month_day(today, m, d) != target
    mname, mabbr = MONTHS[m - 1], MONTH_ABBR[m - 1]
    if needs_year or rng.random() < 0.1:
        y = target.year
        for text in (f"{mname} {d}, {y}", f"{mabbr} {d}, {y}", f"{mname} {ordinal(d)}, {y}",
                     f"{m}/{d}/{y}", f"{d} {mname} {y}"):
            add("abs", text, f"on {text}")
    if not needs_year:
        for text in (f"{mname} {d}", f"{mabbr} {d}", f"{mname} {ordinal(d)}", f"{d} {mname}",
                     f"the {ordinal(d)} of {mname}", f"{m}/{d}"):
            add("abs", text, f"on {text}")

    # day of the month
    if delta <= 31 and next_day_of_month(today, d) == target:
        add("dom", f"the {ordinal(d)}", f"on the {ordinal(d)}")

    # month-relative
    eom = date(today.year, today.month, last_day(today.year, today.month))
    ny, nm = add_months(today.year, today.month, 1)
    eonm = date(ny, nm, last_day(ny, nm))
    if target == eom:
        for text in ("the end of the month", "end of the month", "the last day of the month"):
            add("monthrel", text, text)
    if target == eonm:
        for text in ("the end of next month", "the last day of next month"):
            add("monthrel", text, text)
    scope = None
    if (target.year, target.month) == (today.year, today.month) and target > today:
        scope = "this"
    elif (target.year, target.month) == (ny, nm):
        scope = "next"
    if scope:
        ords = []
        n = (d - 1) // 7 + 1
        if n <= 4:
            ords.append(ORD_WORDS[n])
        if d + 7 > last_day(target.year, target.month):
            ords.append("last")
        for o in ords:
            add("monthrel", f"the {o} {wname} of {scope} month")
        if scope == "next":
            add("monthrel", f"the {ordinal(d)} of next month")
    return out


def pick_date(target: date, today: date, rng: random.Random) -> DatePhrase:
    cands = date_candidates(target, today, rng)
    families = list(cands)
    weights = [FAMILY_WEIGHT[f] for f in families]
    family = rng.choices(families, weights)[0]
    return rng.choice(cands[family])


def abs_only(target: date, today: date, rng: random.Random) -> DatePhrase:
    """A month-and-day phrasing for target (used for 'until ...' dates)."""
    return rng.choice(date_candidates(target, today, rng)["abs"])


# ---------------------------------------------------------------- times

@dataclass
class TimePhrase:
    kind: str    # 12h, 24h, noon, words, period, range
    bare: str    # "3pm"
    text: str    # "at 3pm"


TIME_WEIGHT = {"12h": 6.0, "24h": 1.5, "noon": 3.0, "words": 1.5, "period": 1.2}


def h12(h: int) -> int:
    return h % 12 or 12


def meridiem(h: int, style: int) -> str:
    pm = h >= 12
    return [("am", "pm"), ("AM", "PM"), ("a.m.", "p.m.")][style][pm]


def fmt12(t: time, style: int, space: bool, force_minutes: bool = False) -> str:
    sfx = meridiem(t.hour, style)
    clock = f"{h12(t.hour)}:{t.minute:02d}" if (t.minute or force_minutes) else str(h12(t.hour))
    return f"{clock}{' ' if space else ''}{sfx}"


def time_candidates(t: time, rng: random.Random) -> dict[str, list[TimePhrase]]:
    out: dict[str, list[TimePhrase]] = {}

    def add(kind, bare):
        out.setdefault(kind, []).append(TimePhrase(kind, bare, f"at {bare}"))

    style = rng.choices([0, 1, 2], [0.7, 0.2, 0.1])[0]
    space = rng.random() < 0.25
    add("12h", fmt12(t, style, space, force_minutes=t.minute == 0 and rng.random() < 0.1))

    if t.hour >= 13 or 6 <= t.hour <= 9:
        add("24h", f"{t.hour:02d}:{t.minute:02d}")
    if (t.hour, t.minute) == (12, 0):
        for bare in ("noon", "12 noon"):
            add("noon", bare)

    sfx = meridiem(t.hour, style)
    if t.minute == 30:
        add("words", f"half past {h12(t.hour)}{sfx}")
    if t.minute == 15:
        add("words", f"quarter past {h12(t.hour)}{sfx}")
    if t.minute == 45 and (t.hour + 1) % 12 != 0 and (t.hour + 1) // 12 == t.hour // 12:
        add("words", f"quarter to {h12(t.hour + 1)}{meridiem(t.hour + 1, style)}")

    if t.minute in (0, 30):
        clock = f"{h12(t.hour)}:{t.minute:02d}" if t.minute else str(h12(t.hour))
        if 5 <= t.hour <= 11:
            add("period", f"{clock} in the morning")
        elif 13 <= t.hour <= 17:
            add("period", f"{clock} in the afternoon")
        elif 18 <= t.hour <= 21:
            add("period", f"{clock} in the evening")
    return out


def pick_time(t: time, rng: random.Random) -> TimePhrase:
    cands = time_candidates(t, rng)
    families = list(cands)
    family = rng.choices(families, [TIME_WEIGHT[f] for f in families])[0]
    return rng.choice(cands[family])


def range_phrase(t1: time, t2: time, rng: random.Random) -> TimePhrase:
    """'from 3pm to 4:30pm' or the compact '3-4:30pm' when both share a meridiem."""
    style = rng.choices([0, 1, 2], [0.7, 0.2, 0.1])[0]
    a, b = fmt12(t1, style, False), fmt12(t2, style, False)
    same = (t1.hour >= 12) == (t2.hour >= 12)
    if same and rng.random() < 0.5:
        sfx = meridiem(t2.hour, style)
        a_bare = a[: -len(sfx)]
        text = f"{a_bare}-{b}"
    else:
        text = f"from {a} to {b}"
    return TimePhrase("range", text, text)


def until_phrase(t_end: time, rng: random.Random) -> str:
    style = rng.choices([0, 1, 2], [0.7, 0.2, 0.1])[0]
    return f"{rng.choice(['until', 'till'])} {fmt12(t_end, style, False)}"


def ambiguous_time_phrase(hour: int, minute: int) -> str:
    """A bare clock time with no am/pm and no 24-hour cue: 'at 3', 'at 3:30'."""
    return f"at {hour}:{minute:02d}" if minute else f"at {hour}"


# ---------------------------------------------------------------- durations

def duration_phrase(minutes: int, rng: random.Random) -> str:
    h, m = divmod(minutes, 60)
    if minutes == 30:
        return rng.choice(["for half an hour", "for 30 minutes", "for 30 min"])
    if minutes == 60:
        return rng.choice(["for an hour", "for 1 hour", "for 60 minutes"])
    if minutes == 90:
        return rng.choice(["for an hour and a half", "for 1.5 hours", "for 90 minutes", "for 90 min"])
    if m == 0:
        return f"for {num_text(h, rng)} hours"
    if m == 30:
        return f"for {h}.5 hours"
    unit = rng.choice(["minutes", "min"])
    return f"for {minutes} {unit}"


# ---------------------------------------------------------------- recurrence

@dataclass
class RecurrenceSpec:
    freq: str                     # daily, weekly, monthly, yearly
    interval: int = 1
    days: tuple[int, ...] = ()    # weekday numbers (weekly)
    month: int = 0                # yearly
    day: int = 0                  # monthly (day of month) and yearly (day of month)


def join_names(names: list[str], rng: random.Random) -> str:
    if len(names) <= 2:
        return " and ".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


def recurrence_phrase(spec: RecurrenceSpec, rng: random.Random) -> str:
    n = spec.interval
    if spec.freq == "daily":
        if n == 1:
            return rng.choice(["every day", "daily", "each day"])
        if n == 2:
            return rng.choice(["every other day", "every 2 days", "every two days"])
        return f"every {num_text(n, rng)} days"

    if spec.freq == "weekly":
        days = list(spec.days)
        if days == [0, 1, 2, 3, 4] and n == 1:
            return rng.choice(["every weekday", "weekdays", "Monday through Friday", "Monday to Friday"])
        if days == [5, 6] and n == 1:
            return rng.choice(["every weekend", "on weekends"])
        abbr = rng.random() < 0.2
        names = [(WEEKDAY_ABBR if abbr else WEEKDAYS)[d] for d in days]
        plural = names if abbr else [x + "s" for x in names]
        if n == 1:
            if len(days) == 1:
                return rng.choice([f"every {names[0]}", f"each {names[0]}", f"on {plural[0]}",
                                   f"weekly on {names[0]}", f"every {names[0]}"])
            return rng.choice([f"every {join_names(names, rng)}", f"on {join_names(plural, rng)}"])
        # single-day intervals
        if n == 2:
            return rng.choice([f"every other {names[0]}", f"every two weeks on {names[0]}",
                               f"biweekly on {names[0]}"])
        return f"every {num_text(n, rng)} weeks on {names[0]}"

    if spec.freq == "monthly":
        o = ordinal(spec.day)
        if n == 1:
            return rng.choice([f"every month on the {o}", f"monthly on the {o}",
                               f"on the {o} of every month"])
        if n == 2:
            return f"every other month on the {o}"
        return f"every {num_text(n, rng)} months on the {o}"

    mname, mabbr = MONTHS[spec.month - 1], MONTH_ABBR[spec.month - 1]
    return rng.choice([f"every year on {mname} {spec.day}", f"annually on {mname} {spec.day}",
                       f"every {mname} {ordinal(spec.day)}", f"yearly on {mabbr} {spec.day}"])


def recurrence_first_date(spec: RecurrenceSpec, today: date) -> date:
    """Start date of an undated repeating event: its first occurrence under the conventions."""
    if spec.freq == "daily":
        return today + timedelta(days=1)
    if spec.freq == "weekly":
        return min(next_weekday(today, d) for d in spec.days)
    if spec.freq == "monthly":
        return next_day_of_month(today, spec.day)
    return next_month_day(today, spec.month, spec.day)


def count_phrase(spec: RecurrenceSpec, count: int, rng: random.Random) -> str:
    """'for 6 weeks' where the unit matches one-per-period, else 'N times'."""
    unit = {"daily": "days", "weekly": "weeks", "monthly": "months", "yearly": "years"}[spec.freq]
    one_per_period = spec.interval == 1 and (spec.freq != "weekly" or len(spec.days) == 1)
    if one_per_period and rng.random() < 0.6:
        return f"for {num_text(count, rng)} {unit}"
    return rng.choice([f"{count} times", f"for {count} occurrences", f"{num_text(count, rng)} times"])


def until_date_phrase(until: date, today: date, rng: random.Random) -> str:
    return f"{rng.choice(['until', 'through', 'ending'])} {abs_only(until, today, rng).text}"
