"""
Independent reference parser for the phrases gen/resolver.py renders.

The generator goes value -> phrase. This goes phrase -> value, written separately (it walks
calendars day by day instead of using arithmetic), so a bug in either direction shows up as
a disagreement in tests/test_resolver.py.
"""

import re
from datetime import date, time, timedelta

WD = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
MONTH_NAMES = ["january", "february", "march", "april", "may", "june", "july", "august",
               "september", "october", "november", "december"]
WORD_NUM = {"a": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
            "eight": 8, "nine": 9, "ten": 10}
ORD = {"first": 1, "second": 2, "third": 3, "fourth": 4}


def num(s: str) -> int:
    return WORD_NUM[s] if s in WORD_NUM else int(s)


def month_of(name: str) -> int:
    name = name.lower()
    for i, full in enumerate(MONTH_NAMES, 1):
        if full == name or full[:3] == name[:3] and len(name) <= 4 and full.startswith(name[:3]):
            return i
    raise ValueError(name)


def days_forward(start: date, limit: int = 800):
    d = start
    for _ in range(limit):
        yield d
        d += timedelta(days=1)


def monday_of(d: date) -> date:
    while d.weekday() != 0:
        d -= timedelta(days=1)
    return d


def first_weekday_after(today: date, wd: int) -> date:
    return next(d for d in days_forward(today + timedelta(days=1), 8) if d.weekday() == wd)


def parse_date(text: str, today: date) -> date:
    t = text.lower().strip()
    t = t[3:] if t.startswith("on ") else t

    if t == "today":
        return today
    if t == "tomorrow":
        return today + timedelta(days=1)
    if t == "the day after tomorrow":
        return today + timedelta(days=2)
    m = re.fullmatch(r"in (\w+) days", t)
    if m:
        return today + timedelta(days=num(m.group(1)))
    m = re.fullmatch(r"in (\w+) weeks?", t)
    if m:
        return today + timedelta(days=7 * num(m.group(1)))

    names = "|".join(WD)
    m = re.fullmatch(rf"(?:(this|next) )?({names})", t)
    if m:
        return first_weekday_after(today, WD.index(m.group(2)))
    m = re.fullmatch(rf"({names}) (?:of )?next week", t)
    if m:
        return monday_of(today) + timedelta(days=7 + WD.index(m.group(1)))
    m = re.fullmatch(rf"a week from ({names})", t)
    if m:
        return first_weekday_after(today, WD.index(m.group(1))) + timedelta(days=7)

    if t in ("the end of the month", "end of the month", "the last day of the month"):
        return max(d for d in _month_days(today.year, today.month))
    if t in ("the end of next month", "the last day of next month"):
        y, mo = (today.year + 1, 1) if today.month == 12 else (today.year, today.month + 1)
        return max(_month_days(y, mo))
    m = re.fullmatch(rf"the (first|second|third|fourth|last) ({names}) of (this|next) month", t)
    if m:
        y, mo = today.year, today.month
        if m.group(3) == "next":
            y, mo = (y + 1, 1) if mo == 12 else (y, mo + 1)
        hits = [d for d in _month_days(y, mo) if d.weekday() == WD.index(m.group(2))]
        return hits[-1] if m.group(1) == "last" else hits[ORD[m.group(1)] - 1]
    m = re.fullmatch(r"the (\d+)(?:st|nd|rd|th) of next month", t)
    if m:
        y, mo = (today.year + 1, 1) if today.month == 12 else (today.year, today.month + 1)
        return date(y, mo, int(m.group(1)))

    m = re.fullmatch(r"the (\d+)(?:st|nd|rd|th)", t)
    if m:
        return next(d for d in days_forward(today, 70) if d.day == int(m.group(1)))

    # month and day, with or without a year
    month_re = "|".join(n[:3] + r"[a-z]*" for n in MONTH_NAMES)
    forms = [
        (rf"({month_re}) (\d+)(?:st|nd|rd|th)?, (\d{{4}})", lambda g: (g[0], g[1], g[2])),
        (rf"(\d+) ({month_re}) (\d{{4}})", lambda g: (g[1], g[0], g[2])),
        (rf"({month_re}) (\d+)(?:st|nd|rd|th)?", lambda g: (g[0], g[1], None)),
        (rf"(\d+) ({month_re})", lambda g: (g[1], g[0], None)),
        (rf"the (\d+)(?:st|nd|rd|th) of ({month_re})", lambda g: (g[1], g[0], None)),
    ]
    for pattern, pick in forms:
        m = re.fullmatch(pattern, t)
        if m:
            mon, day, year = pick(m.groups())
            return _resolve_md(today, month_of(mon), int(day), int(year) if year else None)
    m = re.fullmatch(r"(\d+)/(\d+)/(\d{4})", t)
    if m:
        return date(int(m.group(3)), int(m.group(1)), int(m.group(2)))
    m = re.fullmatch(r"(\d+)/(\d+)", t)
    if m:
        return _resolve_md(today, int(m.group(1)), int(m.group(2)), None)
    raise ValueError(f"cannot parse date phrase: {text!r}")


def _month_days(year: int, month: int):
    d = date(year, month, 1)
    while d.month == month:
        yield d
        d += timedelta(days=1)


def _resolve_md(today: date, month: int, day: int, year: int | None) -> date:
    if year:
        return date(year, month, day)
    for d in days_forward(today, 366 * 9):
        if d.month == month and d.day == day:
            return d
    raise ValueError("no such day")


# ------------------------------------------------------------------ times

def parse_clock(s: str) -> time:
    s = s.lower().strip()
    if s in ("noon", "12 noon"):
        return time(12, 0)
    m = re.fullmatch(r"half past (\d+)(am|pm|a\.m\.|p\.m\.)", s)
    if m:
        return time(_h24(int(m.group(1)), m.group(2)), 30)
    m = re.fullmatch(r"quarter past (\d+)(am|pm|a\.m\.|p\.m\.)", s)
    if m:
        return time(_h24(int(m.group(1)), m.group(2)), 15)
    m = re.fullmatch(r"quarter to (\d+)(am|pm|a\.m\.|p\.m\.)", s)
    if m:
        h = _h24(int(m.group(1)), m.group(2))
        return time(h - 1, 45)
    m = re.fullmatch(r"(\d+)(?::(\d\d))? in the (morning|afternoon|evening)", s)
    if m:
        h = int(m.group(1)) + (12 if m.group(3) != "morning" else 0)
        return time(h, int(m.group(2) or 0))
    m = re.fullmatch(r"(\d+)(?::(\d\d))? ?(am|pm|a\.m\.|p\.m\.)", s)
    if m:
        return time(_h24(int(m.group(1)), m.group(3)), int(m.group(2) or 0))
    m = re.fullmatch(r"(\d\d):(\d\d)", s)
    if m:
        return time(int(m.group(1)), int(m.group(2)))
    raise ValueError(f"cannot parse time: {s!r}")


def _h24(h12: int, suffix: str) -> int:
    pm = suffix.startswith("p")
    return (h12 % 12) + (12 if pm else 0)


def parse_range(s: str) -> tuple[time, time]:
    s = s.lower().strip()
    m = re.fullmatch(r"from (.+) to (.+)", s)
    if m:
        return parse_clock(m.group(1)), parse_clock(m.group(2))
    m = re.fullmatch(r"(\d+(?::\d\d)?)-(.+)", s)
    if m:
        end = parse_clock(m.group(2))
        suffix = re.search(r"(am|pm|a\.m\.|p\.m\.)$", m.group(2)).group(1)
        return parse_clock(m.group(1) + suffix), end
    raise ValueError(s)


def parse_duration(s: str) -> int:
    s = s.lower().strip()
    if s == "for half an hour":
        return 30
    if s == "for an hour":
        return 60
    if s == "for an hour and a half":
        return 90
    m = re.fullmatch(r"for ([\d.]+|\w+) hours?", s)
    if m:
        v = m.group(1)
        return round(float(v) * 60) if v.replace(".", "").isdigit() else num(v) * 60
    m = re.fullmatch(r"for (\d+) (?:minutes|min)", s)
    if m:
        return int(m.group(1))
    raise ValueError(s)


# ------------------------------------------------------------------ recurrence

def parse_recurrence(s: str):
    """Returns (freq, interval, days tuple, month, day-of-month)."""
    t = s.lower()
    abbr = {d[:3]: i for i, d in enumerate(WD)}
    if "weekday" in t or "monday through friday" in t or "monday to friday" in t:
        return "weekly", 1, (0, 1, 2, 3, 4), 0, 0
    if "weekend" in t:
        return "weekly", 1, (5, 6), 0, 0
    if "year" in t or "annual" in t or any(re.search(rf"\b{m}\b", t) for m in MONTH_NAMES + [x[:3] for x in MONTH_NAMES]):
        mon = next(i + 1 for i, name in enumerate(MONTH_NAMES) if re.search(rf"\b{name}\b|\b{name[:3]}\b", t))
        day = int(re.search(r"(\d+)", t).group(1))
        return "yearly", 1, (), mon, day
    if "month" in t:
        day = int(re.search(r"(\d+)(?:st|nd|rd|th)", t).group(1))
        if "every other" in t:
            iv = 2
        else:
            m = re.search(r"every (\w+) months", t)
            iv = num(m.group(1)) if m else 1
        return "monthly", iv, (), 0, day
    days = tuple(sorted({abbr[w[:3]] for w in re.findall(r"[a-z]+", t) if w.rstrip("s") in WD or w.rstrip("s") in abbr}))
    if days:
        if "every other" in t or "every two weeks" in t or "biweekly" in t:
            iv = 2
        else:
            m = re.search(r"every (\w+) weeks", t)
            iv = num(m.group(1)) if m else 1
        return "weekly", iv, days, 0, 0
    if re.search(r"\bdaily\b|every day|each day", t):
        return "daily", 1, (), 0, 0
    if "every other day" in t:
        return "daily", 2, (), 0, 0
    m = re.search(r"every (\w+) days", t)
    return "daily", num(m.group(1)), (), 0, 0
