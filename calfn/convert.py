"""
Turn a resolved event into things a calendar understands. Pure functions, no model needed.

    python -m calfn.convert --args '{"title":"Yoga","start":"2026-10-20T18:00","recurrence_freq":"weekly","recurrence_days":"TU"}' --ics yoga.ics --links

Times are floating (wall-clock, no timezone) unless tz is given:
  - one-off events become UTC (DTSTART:...Z)
  - repeating events keep the zone as a TZID parameter so DST is respected. There is no
    VTIMEZONE block; Google, Apple and Thunderbird accept well-known zone names, but a strict
    importer may not. Floating times are the safest default for a personal calendar.

The add-to-calendar URL formats are unofficial. Google's `recur` parameter and Outlook's
deep link have changed over time and are not documented by the vendors, so test them.
"""

import argparse
import base64
import hashlib
import json
import sys
from datetime import date, datetime, timedelta, timezone
from urllib.parse import quote

from .schema import DEFAULT_DURATION, normalize, parse_start, validate_event

_FREQ = {"daily": "DAILY", "weekly": "WEEKLY", "monthly": "MONTHLY", "yearly": "YEARLY"}


def _zone(tz: str):
    from zoneinfo import ZoneInfo
    return ZoneInfo(tz)


def _end(start: datetime, args: dict) -> datetime:
    return start + timedelta(minutes=args.get("duration_minutes", DEFAULT_DURATION))


def _utc(dt: datetime, tz: str) -> datetime:
    return dt.replace(tzinfo=_zone(tz)).astimezone(timezone.utc).replace(tzinfo=None)


def _stamp(dt: datetime) -> str:
    return dt.strftime("%Y%m%dT%H%M%S")


def rrule(args: dict, tz: str | None = None) -> str | None:
    """The RRULE value (without the 'RRULE:' prefix) for a call, or None for a one-off."""
    freq = args.get("recurrence_freq")
    if freq is None:
        return None
    parts = [f"FREQ={_FREQ[freq]}"]
    if args.get("recurrence_interval", 1) > 1:
        parts.append(f"INTERVAL={args['recurrence_interval']}")
    if args.get("recurrence_days"):
        parts.append(f"BYDAY={args['recurrence_days']}")
    if "recurrence_count" in args:
        parts.append(f"COUNT={args['recurrence_count']}")
    elif "recurrence_until" in args:
        end_of_day = datetime.combine(date.fromisoformat(args["recurrence_until"]),
                                      datetime.max.time()).replace(microsecond=0)
        # With a TZID on DTSTART, RFC 5545 wants UNTIL in UTC
        parts.append("UNTIL=" + (_stamp(_utc(end_of_day, tz)) + "Z" if tz else _stamp(end_of_day)))
    return ";".join(parts)


def _escape(text: str) -> str:
    return (text.replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,")
            .replace("\r\n", "\\n").replace("\n", "\\n"))


def _fold(line: str) -> list[str]:
    """Fold a content line at 75 octets, continuing with a leading space (RFC 5545 3.1)."""
    raw = line.encode("utf-8")
    if len(raw) <= 75:
        return [line]
    out, cur, cur_len = [], "", 0
    limit = 75
    for ch in line:
        n = len(ch.encode("utf-8"))
        if cur_len + n > limit:
            out.append(cur)
            cur, cur_len, limit = " ", 1, 75
        cur += ch
        cur_len += n
    out.append(cur)
    return out


def to_ics(args: dict, tz: str | None = None, stamp: datetime | None = None) -> str:
    """A complete .ics document (CRLF line endings) with one VEVENT."""
    ok, msg = validate_event(args)
    if not ok:
        raise ValueError(msg)
    a = normalize(args)
    start = parse_start(a["start"])
    end = _end(start, a)
    rule = rrule(a, tz)

    if tz is None:
        dtstart, dtend = f"DTSTART:{_stamp(start)}", f"DTEND:{_stamp(end)}"
    elif rule:
        dtstart, dtend = f"DTSTART;TZID={tz}:{_stamp(start)}", f"DTEND;TZID={tz}:{_stamp(end)}"
    else:
        dtstart = f"DTSTART:{_stamp(_utc(start, tz))}Z"
        dtend = f"DTEND:{_stamp(_utc(end, tz))}Z"

    uid = hashlib.sha1(json.dumps(a, sort_keys=True).encode()).hexdigest()[:20]
    stamp = stamp or datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0)

    lines = [
        "BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//calfn//create_event//EN", "CALSCALE:GREGORIAN",
        "BEGIN:VEVENT",
        f"UID:{uid}@calfn",
        f"DTSTAMP:{_stamp(stamp)}Z",
        dtstart, dtend,
    ]
    if rule:
        lines.append(f"RRULE:{rule}")
    lines.append(f"SUMMARY:{_escape(a['title'])}")
    if a.get("location"):
        lines.append(f"LOCATION:{_escape(a['location'])}")
    lines += ["END:VEVENT", "END:VCALENDAR"]

    folded = [piece for line in lines for piece in _fold(line)]
    return "\r\n".join(folded) + "\r\n"


def to_google_url(args: dict, tz: str | None = None) -> str:
    """Google Calendar 'create event' link. Local times; add ctz when a zone is known."""
    a = normalize(args)
    start = parse_start(a["start"])
    q = [("action", "TEMPLATE"), ("text", a["title"]),
         ("dates", f"{_stamp(start)}/{_stamp(_end(start, a))}")]
    if tz:
        q.append(("ctz", tz))
    if a.get("location"):
        q.append(("location", a["location"]))
    rule = rrule(a, tz)
    if rule:
        q.append(("recur", f"RRULE:{rule}"))
    return "https://calendar.google.com/calendar/render?" + "&".join(
        f"{k}={quote(v, safe='/:')}" for k, v in q)


def to_outlook_url(args: dict, kind: str = "live") -> str:
    """
    Outlook on the web compose link ('live' = outlook.live.com, 'office' = outlook.office.com).
    The deep link has no recurrence parameter, so repeating events open as a single event.
    """
    a = normalize(args)
    start = parse_start(a["start"])
    host = {"live": "outlook.live.com", "office": "outlook.office.com"}[kind]
    q = [("path", "/calendar/action/compose"), ("rru", "addevent"), ("subject", a["title"]),
         ("startdt", start.strftime("%Y-%m-%dT%H:%M:%S")),
         ("enddt", _end(start, a).strftime("%Y-%m-%dT%H:%M:%S"))]
    if a.get("location"):
        q.append(("location", a["location"]))
    return f"https://{host}/calendar/0/deeplink/compose?" + "&".join(
        f"{k}={quote(v, safe='')}" for k, v in q)


def encode_share(args: dict) -> str:
    """Compact, URL-safe encoding of a call, for a future page.html#e=<this> share link."""
    ok, msg = validate_event(args)
    if not ok:
        raise ValueError(msg)
    clean = {k: v for k, v in args.items() if v is not None}
    raw = json.dumps(clean, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_share(token: str) -> dict:
    """Inverse of encode_share. Raises ValueError for anything that isn't a valid call."""
    try:
        raw = base64.urlsafe_b64decode(token + "=" * (-len(token) % 4))
        args = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as e:
        raise ValueError(f"not a share token: {e}") from e
    ok, msg = validate_event(args)
    if not ok:
        raise ValueError(msg)
    return args


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Convert a create_event call to .ics and add-to-calendar links.")
    p.add_argument("--args", required=True, help="JSON string of the create_event arguments")
    p.add_argument("--tz", help="IANA zone name, e.g. America/Los_Angeles (default: floating times)")
    p.add_argument("--ics", help="write an .ics file here")
    p.add_argument("--links", action="store_true", help="print Google/Outlook links and the share token")
    ns = p.parse_args(argv)

    args = json.loads(ns.args)
    ok, msg = validate_event(args)
    if not ok:
        print(f"invalid call: {msg}", file=sys.stderr)
        return 1
    if ns.ics:
        with open(ns.ics, "w", encoding="utf-8", newline="") as f:
            f.write(to_ics(args, ns.tz))
        print(f"wrote {ns.ics}")
    if ns.links or not ns.ics:
        print("google :", to_google_url(args, ns.tz))
        print("outlook:", to_outlook_url(args, "live"))
        if "recurrence_freq" in args:
            print("(the Outlook link cannot carry a recurrence)")
        print("share  :", encode_share(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
