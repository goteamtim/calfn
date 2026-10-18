"""
Turn what the model wrote (a CALL, see calfn/schema.py) into a real event, given the current time.

This is the only place calendar arithmetic happens. The model copies how the date and time were
said; this module applies the conventions in data/eval/README.md.
"""

from datetime import datetime, timedelta

from . import dates
from .schema import CALL_ORDER, DAY_CODES, EVENT_ORDER, validate_call, validate_event


class CallError(ValueError):
    """The call is malformed or cannot be resolved into a valid event."""


def resolve(call: dict, now: datetime) -> dict:
    """The event a call describes, as of `now`. Raises CallError if it can't."""
    ok, msg = validate_call(call)
    if not ok:
        raise CallError(msg)
    today = now.date()
    freq = call.get("recurrence_freq")
    try:
        if "date" in call:
            day = dates.resolve_date(call["date"], today)
        else:
            days = tuple(DAY_CODES.index(c) for c in call.get("recurrence_days", "").split(",") if c)
            day = dates.first_occurrence(freq, days, today)
        until = dates.resolve_until(call["recurrence_until"], today) if "recurrence_until" in call else None
    except dates.DateError as e:
        raise CallError(str(e)) from e

    hh, mm = int(call["time"][:2]), int(call["time"][3:])
    event = {"title": call["title"], "start": f"{day.isoformat()}T{call['time']}"}
    if "duration_minutes" in call:
        event["duration_minutes"] = call["duration_minutes"]
    elif "end" in call:
        event["duration_minutes"] = int(call["end"][:2]) * 60 + int(call["end"][3:]) - (hh * 60 + mm)
    for key in ("location", "recurrence_freq", "recurrence_interval", "recurrence_days", "recurrence_count"):
        if key in call:
            event[key] = call[key]
    if until is not None:
        event["recurrence_until"] = until.isoformat()

    event = {k: event[k] for k in EVENT_ORDER if k in event}
    ok, msg = validate_event(event)
    if not ok:
        raise CallError(f"resolved event is invalid: {msg}")
    return event


def event_to_call(event: dict) -> dict:
    """
    A call that resolves to `event` from any `now`: the date is written out in full (md:YYYY-MM-DD).
    Not what the model learns to write; used as the oracle in harness checks and in round-trip tests.
    """
    date_part, time_part = event["start"].split("T")
    call = {"title": event["title"], "date": f"md:{date_part}", "time": time_part}
    for key in ("duration_minutes", "location", "recurrence_freq", "recurrence_interval",
                "recurrence_days", "recurrence_count", "recurrence_until"):
        if key in event:
            call[key] = event[key]
    return {k: call[k] for k in CALL_ORDER if k in call}


def is_future(event: dict, now: datetime) -> bool:
    """False if the event starts at or before `now` (resolve() doesn't refuse these; the app can warn)."""
    return datetime.strptime(event["start"], "%Y-%m-%dT%H:%M") > now
