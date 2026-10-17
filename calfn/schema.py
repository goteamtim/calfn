"""
The create_event tool: schema, system prompt, validation and normalization.

Single source of truth for training, eval, baselines and the converter.

Everything is a flat scalar on purpose. FunctionGemma's custom key:value call format
(and the parser in parse.py) is only exercised on scalars, so recurrence is five flat
fields instead of one nested object.
"""

import re
from datetime import date, datetime

TOOL = {
    "name": "create_event",
    "description": "Create a calendar event from a natural-language request.",
    "parameters": {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Short event title, as the user named it.",
            },
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
            "location": {
                "type": "string",
                "description": "Where the event is, only if the user said.",
            },
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
            "recurrence_until": {
                "type": "string",
                "description": "Last date of the series, YYYY-MM-DD, inclusive. Not with recurrence_count.",
            },
        },
        "required": ["title", "start"],
    },
}

PROPS = TOOL["parameters"]["properties"]
REQUIRED_PARAMS = set(TOOL["parameters"]["required"])
ALL_PARAMS = set(PROPS)
RECURRENCE_PARAMS = {k for k in PROPS if k.startswith("recurrence_")}

# Canonical key order for labels and output
PARAM_ORDER = list(PROPS)

DAY_CODES = ["MO", "TU", "WE", "TH", "FR", "SA", "SU"]  # index == date.weekday()

DEFAULT_DURATION = 60
DEFAULT_INTERVAL = 1

# Conventions are spelled out here because the model sees them in every example and a
# prompted baseline needs them too. data/eval/README.md has the full list.
SYSTEM_PROMPT = (
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
    """The 'Now: ...' line appended to the developer message. English weekday, locale-proof."""
    return f"Now: {now:%Y-%m-%d} {_WEEKDAY_ABBR[now.weekday()]} {now:%H:%M}"


_WEEKDAY_ABBR = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def developer_message(now: datetime) -> str:
    return f"{SYSTEM_PROMPT}\n{now_line(now)}"


_START_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_DAYS_RE = re.compile(r"^(MO|TU|WE|TH|FR|SA|SU)(,(MO|TU|WE|TH|FR|SA|SU))*$")


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def parse_start(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M")


def validate(args) -> tuple[bool, str]:
    """Strict check of a tool call's arguments against the schema. Returns (ok, message)."""
    if not isinstance(args, dict):
        return False, "arguments are not an object"
    unknown = set(args) - ALL_PARAMS
    if unknown:
        return False, f"unknown params: {sorted(unknown)}"
    missing = REQUIRED_PARAMS - set(args)
    if missing:
        return False, f"missing required params: {sorted(missing)}"

    title = args["title"]
    if not isinstance(title, str) or not title.strip() or len(title) > 200:
        return False, "title must be a non-empty string"

    start = args["start"]
    if not isinstance(start, str) or not _START_RE.match(start):
        return False, f"start must be YYYY-MM-DDTHH:MM: {start!r}"
    try:
        start_dt = parse_start(start)
    except ValueError:
        return False, f"start is not a real date and time: {start!r}"

    for key in ("duration_minutes", "recurrence_interval", "recurrence_count"):
        if key in args:
            v = args[key]
            if not _is_int(v):
                return False, f"{key} must be an integer: {v!r}"
            lo, hi = PROPS[key]["minimum"], PROPS[key]["maximum"]
            if not lo <= v <= hi:
                return False, f"{key} out of range [{lo},{hi}]: {v}"

    if "location" in args and (not isinstance(args["location"], str) or not args["location"].strip()):
        return False, "location must be a non-empty string"

    freq = args.get("recurrence_freq")
    if "recurrence_freq" in args and freq not in PROPS["recurrence_freq"]["enum"]:
        return False, f"invalid recurrence_freq: {freq!r}"
    other = (RECURRENCE_PARAMS - {"recurrence_freq"}) & set(args)
    if other and freq is None:
        return False, f"{sorted(other)} given without recurrence_freq"

    if "recurrence_days" in args:
        days = args["recurrence_days"]
        if not isinstance(days, str) or not _DAYS_RE.match(days):
            return False, f"recurrence_days must look like MO,WE: {days!r}"
        codes = days.split(",")
        if codes != sorted(set(codes), key=DAY_CODES.index):
            return False, f"recurrence_days must be unique and in week order: {days!r}"
        if freq != "weekly":
            return False, "recurrence_days only applies to weekly"

    if "recurrence_count" in args and "recurrence_until" in args:
        return False, "recurrence_count and recurrence_until are mutually exclusive"
    if "recurrence_until" in args:
        until = args["recurrence_until"]
        if not isinstance(until, str) or not _DATE_RE.match(until):
            return False, f"recurrence_until must be YYYY-MM-DD: {until!r}"
        try:
            until_d = date.fromisoformat(until)
        except ValueError:
            return False, f"recurrence_until is not a real date: {until!r}"
        if until_d < start_dt.date():
            return False, "recurrence_until is before start"
    return True, "OK"


def normalize(args: dict) -> dict:
    """
    Canonical form for comparing two calls. Explicit defaults equal omitted ones
    (duration 60, interval 1), whitespace in text is collapsed, and keys are ordered.
    Titles and locations keep their case; use same_text() to compare them.
    """
    out = {}
    for k in PARAM_ORDER:
        if k in args and args[k] is not None:
            v = args[k]
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


_TEXT_FIELDS = ("title", "location")


def same_event(a: dict, b: dict) -> bool:
    """True if two calls describe the same event."""
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


def field_matches(pred: dict, gold: dict) -> dict:
    """Per-field correctness for the fields the gold call states (defaults the user never asked for don't count)."""
    np_, ng = normalize(pred), normalize(gold)
    out = {}
    for k in gold:
        if k not in ng:
            continue
        gv, pv = ng[k], np_.get(k)
        if k in _TEXT_FIELDS:
            out[k] = isinstance(pv, str) and same_text(pv, gv)
        else:
            out[k] = pv == gv
    return out


def spurious_fields(pred: dict, gold: dict) -> list[str]:
    """Fields the prediction sets to a non-default value that the gold call does not state."""
    np_, ng = normalize(pred), normalize(gold)
    defaults = {"duration_minutes": DEFAULT_DURATION, "recurrence_interval": DEFAULT_INTERVAL}
    return [k for k, v in np_.items()
            if k not in gold and v != defaults.get(k) and not (k in ng and ng[k] == v)]
