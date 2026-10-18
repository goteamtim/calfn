"""
Date arithmetic and the date expression language the model writes.

The model never does calendar math. It copies how a date was said into a short expression,
and `resolve_date(expr, today)` turns that into a real date, deterministically:

    in:0d  in:1d  in:30d  in:3w     days / weeks from today (today = in:0d, tomorrow = in:1d)
    wd:TH                           the first Thursday strictly after today
    wd:TH:2                         the second one ("a week from Thursday")
    nw:TH                           Thursday in the next Mon-Sun week
    md:10-21  md:2027-10-21         month and day: next occurrence on or after today, or that year
    dom:21                          day of the month: next occurrence on or after today
    eom:0  eom:1                    last day of this month / next month
    nth:2:TU:1  nth:L:FR:0          2nd Tuesday of next month / last Friday of this month
    dnm:15                          the 15th of next month

Weekday codes are MO TU WE TH FR SA SU. Conventions are documented in data/eval/README.md.
"""

import calendar
import re
from datetime import date, timedelta

DAY_CODES = ["MO", "TU", "WE", "TH", "FR", "SA", "SU"]  # index == date.weekday()

_DAY = "(MO|TU|WE|TH|FR|SA|SU)"
_PATTERNS = [
    ("in", re.compile(r"^in:(\d{1,3})(d|w)$")),
    ("wd", re.compile(rf"^wd:{_DAY}(?::([2-5]))?$")),
    ("nw", re.compile(rf"^nw:{_DAY}$")),
    ("md", re.compile(r"^md:(?:(\d{4})-)?(\d{2})-(\d{2})$")),
    ("dom", re.compile(r"^dom:(\d{1,2})$")),
    ("eom", re.compile(r"^eom:([01])$")),
    ("nth", re.compile(rf"^nth:([1-4]|L):{_DAY}:([01])$")),
    ("dnm", re.compile(r"^dnm:(\d{1,2})$")),
]


class DateError(ValueError):
    """The expression is malformed, or names a date that does not exist."""


# ---------------------------------------------------------------- arithmetic

def ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def week_start(d: date) -> date:
    return d - timedelta(days=d.weekday())


def last_day(year: int, month: int) -> int:
    return calendar.monthrange(year, month)[1]


def add_months(year: int, month: int, k: int) -> tuple[int, int]:
    idx = year * 12 + (month - 1) + k
    return idx // 12, idx % 12 + 1


def next_weekday(today: date, wd: int, n: int = 1) -> date:
    """The n-th date strictly after today that falls on weekday wd (0 = Monday)."""
    first = today + timedelta(days=(wd - today.weekday()) % 7 or 7)
    return first + timedelta(days=7 * (n - 1))


def next_month_day(today: date, month: int, day: int) -> date:
    """Next occurrence of month/day on or after today."""
    for year in range(today.year, today.year + 9):
        try:
            cand = date(year, month, day)
        except ValueError:
            continue
        if cand >= today:
            return cand
    raise DateError(f"no occurrence of {month}/{day}")


def next_day_of_month(today: date, day: int) -> date:
    """Next date on or after today whose day of the month is `day`."""
    year, month = today.year, today.month
    for _ in range(14):
        if day <= last_day(year, month):
            cand = date(year, month, day)
            if cand >= today:
                return cand
        year, month = add_months(year, month, 1)
    raise DateError(f"no day {day}")


def nth_weekday(year: int, month: int, wd: int, n: int | str) -> date:
    """The n-th (1-4) or last ('L') given weekday of a month."""
    days = [date(year, month, d) for d in range(1, last_day(year, month) + 1)
            if date(year, month, d).weekday() == wd]
    return days[-1] if n == "L" else days[int(n) - 1]


# ---------------------------------------------------------------- expressions

def parse_expr(expr: str):
    """(kind, groups) for a well-formed expression; raises DateError otherwise."""
    if not isinstance(expr, str):
        raise DateError(f"date must be a string: {expr!r}")
    for kind, pat in _PATTERNS:
        m = pat.match(expr.strip())
        if m:
            return kind, m.groups()
    raise DateError(f"not a date expression: {expr!r}")


def resolve_date(expr: str, today: date) -> date:
    kind, g = parse_expr(expr)
    try:
        if kind == "in":
            n = int(g[0])
            return today + timedelta(days=n * (7 if g[1] == "w" else 1))
        if kind == "wd":
            return next_weekday(today, DAY_CODES.index(g[0]), int(g[1] or 1))
        if kind == "nw":
            return week_start(today) + timedelta(days=7 + DAY_CODES.index(g[0]))
        if kind == "md":
            year, month, day = (int(g[0]) if g[0] else None), int(g[1]), int(g[2])
            return date(year, month, day) if year else next_month_day(today, month, day)
        if kind == "dom":
            return next_day_of_month(today, int(g[0]))
        if kind == "eom":
            y, m = add_months(today.year, today.month, int(g[0]))
            return date(y, m, last_day(y, m))
        if kind == "nth":
            y, m = add_months(today.year, today.month, int(g[2]))
            return nth_weekday(y, m, DAY_CODES.index(g[1]), g[0])
        y, m = add_months(today.year, today.month, 1)           # dnm
        return date(y, m, int(g[0]))
    except (ValueError, IndexError) as e:   # impossible dates: Feb 30, 5th Monday that doesn't exist, ...
        raise DateError(f"{expr!r} is not a real date: {e}") from e


def resolve_until(value: str, today: date) -> date:
    """recurrence_until: 'MM-DD' (next occurrence on or after today) or 'YYYY-MM-DD'."""
    m = re.fullmatch(r"(?:(\d{4})-)?(\d{2})-(\d{2})", value.strip())
    if not m:
        raise DateError(f"recurrence_until must be MM-DD or YYYY-MM-DD: {value!r}")
    year, month, day = (int(m.group(1)) if m.group(1) else None), int(m.group(2)), int(m.group(3))
    try:
        return date(year, month, day) if year else next_month_day(today, month, day)
    except ValueError as e:
        raise DateError(f"{value!r} is not a real date") from e


def first_occurrence(freq: str, days: tuple[int, ...], today: date) -> date:
    """Start date of a daily/weekly series with no stated start: its first date strictly after today."""
    if freq == "daily":
        return today + timedelta(days=1)
    if freq == "weekly" and days:
        return min(next_weekday(today, d) for d in days)
    raise DateError("a start date is required")
