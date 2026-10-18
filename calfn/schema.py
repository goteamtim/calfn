"""
Two schemas, one task.

  EVENT  the resolved event: an absolute start time, a duration, recurrence. This is what gets
         scored, converted to .ics, and shared. It never touches the model directly.
  CALL   what the model writes (TOOL): the date and time as the user said them, as a small
         expression, plus a stated length or end time. calfn/resolve.py turns a CALL into an
         EVENT, given the current time. See docs/decisions.md D9 for why the model doesn't do
         the date arithmetic itself.

The v1 approach (the model writes the EVENT directly and is told "Now: ...") is kept as
EVENT_TOOL + direct_developer_message(), used by the prompted baselines in --task direct.

Everything is a flat scalar on purpose. FunctionGemma's custom key:value call format
(and the parser in parse.py) is only exercised on scalars.
"""

import re
from datetime import date, datetime

from . import dates

DAY_CODES = dates.DAY_CODES  # index == date.weekday()

DEFAULT_DURATION = 60
DEFAULT_INTERVAL = 1

_RECURRENCE_PROPS = {
    "recurrence_freq": {
        "type": "string",
        "enum": ["daily", "weekly", "monthly", "yearly"],
        "description": "How often the event repeats. Omit for a one-off event.",
    },
    "recurrence_interval": {
        "type": "integer",
        "description": "Repeat every N periods (2 = every other). Default 1.",
        "minimum": 1,
        "maximum": 12,
    },
    "recurrence_days": {
        "type": "string",
        "description": "Weekly only: comma-separated weekday codes in week order, e.g. MO,WE,FR.",
    },
    "recurrence_count": {
        "type": "integer",
        "description": "Total number of occurrences. Not with recurrence_until.",
        "minimum": 1,
        "maximum": 60,
    },
}

# ---------------------------------------------------------------------- EVENT (resolved)

EVENT_TOOL = {
    "name": "create_event",
    "description": "Create a calendar event from a natural-language request.",
    "parameters": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Short event title, as the user named it."},
            "start": {
                "type": "string",
                "description": "Local start time, YYYY-MM-DDTHH:MM (24-hour, no timezone).",
            },
            "duration_minutes": {
                "type": "integer",
                "description": "Length in minutes. Only if the user gave a duration or end time.",
                "minimum": 5,
                "maximum": 1440,
            },
            "location": {"type": "string", "description": "Where the event is, only if the user said."},
            **_RECURRENCE_PROPS,
            "recurrence_until": {
                "type": "string",
                "description": "Last date of the series, YYYY-MM-DD, inclusive. Not with recurrence_count.",
            },
        },
        "required": ["title", "start"],
    },
}

EVENT_PROPS = EVENT_TOOL["parameters"]["properties"]
EVENT_REQUIRED = set(EVENT_TOOL["parameters"]["required"])
EVENT_ORDER = list(EVENT_PROPS)
RECURRENCE_PARAMS = {k for k in EVENT_PROPS if k.startswith("recurrence_")}
_TEXT_FIELDS = ("title", "location")

_START_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$")
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_DAYS_RE = re.compile(r"^(MO|TU|WE|TH|FR|SA|SU)(,(MO|TU|WE|TH|FR|SA|SU))*$")


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def parse_start(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M")


def _check_common(args: dict, props: dict) -> str | None:
    """Checks shared by events and calls: title, location, numbers, recurrence rules."""
    title = args["title"]
    if not isinstance(title, str) or not title.strip() or len(title) > 200:
        return "title must be a non-empty string"
    for key in ("duration_minutes", "recurrence_interval", "recurrence_count"):
        if key in args:
            v = args[key]
            if not _is_int(v):
                return f"{key} must be an integer: {v!r}"
            lo, hi = props[key]["minimum"], props[key]["maximum"]
            if not lo <= v <= hi:
                return f"{key} out of range [{lo},{hi}]: {v}"
    if "location" in args and (not isinstance(args["location"], str) or not args["location"].strip()):
        return "location must be a non-empty string"

    freq = args.get("recurrence_freq")
    if "recurrence_freq" in args and freq not in props["recurrence_freq"]["enum"]:
        return f"invalid recurrence_freq: {freq!r}"
    other = (RECURRENCE_PARAMS - {"recurrence_freq"}) & set(args)
    if other and freq is None:
        return f"{sorted(other)} given without recurrence_freq"
    if "recurrence_days" in args:
        days = args["recurrence_days"]
        if not isinstance(days, str) or not _DAYS_RE.match(days):
            return f"recurrence_days must look like MO,WE: {days!r}"
        codes = days.split(",")
        if codes != sorted(set(codes), key=DAY_CODES.index):
            return f"recurrence_days must be unique and in week order: {days!r}"
        if freq != "weekly":
            return "recurrence_days only applies to weekly"
    if "recurrence_count" in args and "recurrence_until" in args:
        return "recurrence_count and recurrence_until are mutually exclusive"
    return None


def validate_event(args) -> tuple[bool, str]:
    """Strict check of a resolved event. Returns (ok, message)."""
    if not isinstance(args, dict):
        return False, "event is not an object"
    unknown = set(args) - set(EVENT_PROPS)
    if unknown:
        return False, f"unknown params: {sorted(unknown)}"
    missing = EVENT_REQUIRED - set(args)
    if missing:
        return False, f"missing required params: {sorted(missing)}"
    start = args["start"]
    if not isinstance(start, str) or not _START_RE.match(start):
        return False, f"start must be YYYY-MM-DDTHH:MM: {start!r}"
    try:
        start_dt = parse_start(start)
    except ValueError:
        return False, f"start is not a real date and time: {start!r}"
    err = _check_common(args, EVENT_PROPS)
    if err:
        return False, err
    if "recurrence_until" in args:
        until = args["recurrence_until"]
        if not isinstance(until, str) or not _ISO_DATE_RE.match(until):
            return False, f"recurrence_until must be YYYY-MM-DD: {until!r}"
        try:
            until_d = datetime.strptime(until, "%Y-%m-%d").date()
        except ValueError:
            return False, f"recurrence_until is not a real date: {until!r}"
        if until_d < start_dt.date():
            return False, "recurrence_until is before start"
    return True, "OK"


def normalize(event: dict) -> dict:
    """
    Canonical form for comparing two events. Explicit defaults equal omitted ones
    (duration 60, interval 1), whitespace in text is collapsed, and keys are ordered.
    Titles and locations keep their case; use same_text() to compare them.
    """
    out = {}
    for k in EVENT_ORDER:
        if k in event and event[k] is not None:
            v = event[k]
            if isinstance(v, str):
                v = " ".join(v.split())
            out[k] = v
    out.setdefault("duration_minutes", DEFAULT_DURATION)
    if "recurrence_freq" in out:
        out.setdefault("recurrence_interval", DEFAULT_INTERVAL)
    return out


def _text_key(s: str) -> str:
    return re.sub(r"[^\w\s]", "", s).casefold().strip()


def same_text(a: str, b: str) -> bool:
    """Titles and locations compare ignoring case and punctuation."""
    return _text_key(a) == _text_key(b)


def same_event(a: dict, b: dict) -> bool:
    """True if two events are the same event."""
    na, nb = normalize(a), normalize(b)
    if na.keys() != nb.keys():
        return False
    for k in na:
        if k in _TEXT_FIELDS:
            if not same_text(na[k], nb[k]):
                return False
        elif na[k] != nb[k]:
            return False
    return True


def _flat(ev: dict) -> dict:
    """Normalized event with start split into start_date and start_time, for per-field scoring."""
    n = normalize(ev)
    start = n.pop("start", None)
    if isinstance(start, str) and "T" in start:
        n = {"start_date": start.split("T")[0], "start_time": start.split("T")[1], **n}
    return n


def field_matches(pred: dict, gold: dict) -> dict:
    """Per-field correctness for the fields the gold event states (defaults nobody asked for don't count)."""
    np_, ng = _flat(pred), _flat(gold)
    stated = {name for k in gold for name in (("start_date", "start_time") if k == "start" else (k,))}
    out = {}
    for k, gv in ng.items():
        if k in stated:
            pv = np_.get(k)
            out[k] = (isinstance(pv, str) and same_text(pv, gv)) if k in _TEXT_FIELDS else pv == gv
    return out


def spurious_fields(pred: dict, gold: dict) -> list[str]:
    """Fields the prediction sets to a non-default value that the gold event does not state."""
    np_, ng = normalize(pred), normalize(gold)
    defaults = {"duration_minutes": DEFAULT_DURATION, "recurrence_interval": DEFAULT_INTERVAL}
    return [k for k, v in np_.items()
            if k not in gold and v != defaults.get(k) and not (k in ng and ng[k] == v)]


# ---------------------------------------------------------------------- CALL (what the model writes)

TOOL = {
    "name": "create_event",
    "description": "Create a calendar event from a natural-language request.",
    "parameters": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Short event title, as the user named it."},
            "date": {
                "type": "string",
                "description": (
                    "The date as said, as kind:value. in:0d today, in:1d tomorrow, in:30d or in:3w from today; "
                    "wd:TH the next Thursday (wd:TH:2 the one after); nw:TH Thursday of next week; "
                    "md:10-21 or md:2027-10-21 a month and day; dom:21 the 21st; "
                    "eom:0 or eom:1 end of this or next month; nth:2:TU:1 the 2nd Tuesday of next month "
                    "(L = last, month 0 = this); dnm:15 the 15th of next month. "
                    "Omit only for a daily or weekly series with no start date."
                ),
            },
            "time": {"type": "string", "description": "Start time as said, 24-hour HH:MM."},
            "end": {
                "type": "string",
                "description": "End time HH:MM, only if the request gives an end time or a range. Not with duration_minutes.",
            },
            "duration_minutes": {
                "type": "integer",
                "description": "Length in minutes, only if the request gives a length. Not with end.",
                "minimum": 5,
                "maximum": 1440,
            },
            "location": {"type": "string", "description": "Where the event is, only if the user said."},
            **_RECURRENCE_PROPS,
            "recurrence_until": {
                "type": "string",
                "description": "Last date of the series, MM-DD or YYYY-MM-DD. Not with recurrence_count.",
            },
        },
        "required": ["title", "time"],
    },
}

CALL_PROPS = TOOL["parameters"]["properties"]
CALL_REQUIRED = set(TOOL["parameters"]["required"])
CALL_ORDER = list(CALL_PROPS)

_CLOCK_RE = re.compile(r"^\d{2}:\d{2}$")
_UNTIL_RE = re.compile(r"^(\d{4}-)?\d{2}-\d{2}$")


def _clock_minutes(s) -> int | None:
    if not isinstance(s, str) or not _CLOCK_RE.match(s):
        return None
    h, m = int(s[:2]), int(s[3:])
    return h * 60 + m if h < 24 and m < 60 else None


def validate_call(args) -> tuple[bool, str]:
    """Strict check of what the model wrote, before resolving it. Returns (ok, message)."""
    if not isinstance(args, dict):
        return False, "arguments are not an object"
    unknown = set(args) - set(CALL_PROPS)
    if unknown:
        return False, f"unknown params: {sorted(unknown)}"
    missing = CALL_REQUIRED - set(args)
    if missing:
        return False, f"missing required params: {sorted(missing)}"
    err = _check_common(args, CALL_PROPS)
    if err:
        return False, err

    start = _clock_minutes(args["time"])
    if start is None:
        return False, f"time must be HH:MM: {args['time']!r}"
    if "end" in args:
        end = _clock_minutes(args["end"])
        if end is None:
            return False, f"end must be HH:MM: {args['end']!r}"
        if end <= start:
            return False, "end must be later than time"
        if "duration_minutes" in args:
            return False, "end and duration_minutes are mutually exclusive"

    freq = args.get("recurrence_freq")
    if "date" in args:
        try:
            kind, g = dates.parse_expr(args["date"])
        except dates.DateError as e:
            return False, str(e)
        if kind == "md":                       # month/day must exist in some year (Feb 29 counts)
            try:
                date(int(g[0]) if g[0] else 2024, int(g[1]), int(g[2]))
            except ValueError:
                return False, f"not a real date: {args['date']!r}"
    else:
        if freq not in ("daily", "weekly"):
            return False, "date is required unless the event is a daily or weekly series"
        if freq == "weekly" and "recurrence_days" not in args:
            return False, "a weekly series with no date needs recurrence_days"
    if freq in ("monthly", "yearly") and "date" not in args:
        return False, f"a {freq} series needs a date"

    if "recurrence_until" in args:
        until = args["recurrence_until"]
        if not isinstance(until, str) or not _UNTIL_RE.match(until):
            return False, f"recurrence_until must be MM-DD or YYYY-MM-DD: {until!r}"
        try:
            dates.resolve_until(until, date(2024, 1, 15))
        except dates.DateError as e:
            return False, str(e)
    return True, "OK"


# ---------------------------------------------------------------------- prompts

SYSTEM_PROMPT = (
    "You are a calendar assistant. Turn the user's request into one create_event call. "
    "Write the date and time the way the user said them; the app works out the real date. "
    "Give duration_minutes only for a stated length, and end only for a stated end time or range. "
    "For repeating events set recurrence_freq, plus interval, days, count or until only if stated. "
    "If the date or time cannot be determined (no time, a vague time like 'sometime' or 'morning', "
    "or a bare hour with no am/pm), or the request is not about creating an event, do not call the tool."
)


def developer_message() -> str:
    return SYSTEM_PROMPT


# The v1 "direct" prompt: the model writes the resolved event itself and is told the current time.
_WEEKDAY_ABBR = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

DIRECT_SYSTEM_PROMPT = (
    "You are a calendar assistant. Turn the user's request into one create_event call. "
    "Times are local wall-clock time. Resolve relative dates from the Now line. "
    "A weekday name (Thursday, this Thursday, next Thursday) means the first such day after today. "
    "A month and day (Oct 21) or a day of the month (the 21st) means its next occurrence, today included. "
    "Give duration_minutes only if the user gave a length or an end time. "
    "For repeating events set recurrence_freq, plus interval, days, count or until only if stated; "
    "a repeating event with no start date starts on its first occurrence after today. "
    "If the date or time cannot be determined (no time, a vague time like 'sometime' or 'morning', "
    "or a bare hour with no am/pm), or the request is not about creating an event, do not call the tool."
)


def now_line(now: datetime) -> str:
    """The 'Now: ...' line of the direct prompt. English weekday, locale-proof."""
    return f"Now: {now:%Y-%m-%d} {_WEEKDAY_ABBR[now.weekday()]} {now:%H:%M}"


def direct_developer_message(now: datetime) -> str:
    return f"{DIRECT_SYSTEM_PROMPT}\n{now_line(now)}"
