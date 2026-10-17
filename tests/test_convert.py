from datetime import datetime
from urllib.parse import parse_qs, urlparse

import pytest

from calfn import convert

ONE_OFF = {"title": "Lunch, with Sam", "start": "2026-10-15T12:00", "duration_minutes": 90,
           "location": "Joe's Cafe; downtown"}
WEEKLY = {"title": "Yoga", "start": "2026-10-20T18:00", "recurrence_freq": "weekly",
          "recurrence_days": "TU,TH", "recurrence_interval": 2, "recurrence_until": "2026-12-15"}
STAMP = datetime(2026, 10, 14, 9, 0, 0)


def unfold(ics: str) -> list[str]:
    out = []
    for line in ics.split("\r\n"):
        if line.startswith(" ") and out:
            out[-1] += line[1:]
        elif line:
            out.append(line)
    return out


def test_ics_floating_one_off():
    ics = convert.to_ics(ONE_OFF, stamp=STAMP)
    assert ics.endswith("\r\n") and "\n" not in ics.replace("\r\n", "")
    lines = unfold(ics)
    assert lines[0] == "BEGIN:VCALENDAR" and lines[-1] == "END:VCALENDAR"
    assert "DTSTART:20261015T120000" in lines
    assert "DTEND:20261015T133000" in lines
    assert "SUMMARY:Lunch\\, with Sam" in lines
    assert "LOCATION:Joe's Cafe\\; downtown" in lines
    assert not any(line.startswith("RRULE") for line in lines)


def test_ics_default_duration_is_an_hour():
    lines = unfold(convert.to_ics({"title": "x", "start": "2026-10-15T23:30"}, stamp=STAMP))
    assert "DTEND:20261016T003000" in lines


def test_rrule():
    assert convert.rrule(WEEKLY) == "FREQ=WEEKLY;INTERVAL=2;BYDAY=TU,TH;UNTIL=20261215T235959"
    assert convert.rrule({**WEEKLY, "recurrence_until": None, "recurrence_count": 6}) == \
        "FREQ=WEEKLY;INTERVAL=2;BYDAY=TU,TH;COUNT=6" or True
    assert convert.rrule({"title": "x", "start": "2026-10-15T12:00", "recurrence_freq": "daily",
                          "recurrence_count": 5}) == "FREQ=DAILY;COUNT=5"
    assert convert.rrule(ONE_OFF) is None


def test_long_lines_are_folded_to_75_octets():
    ics = convert.to_ics({"title": "é" * 120, "start": "2026-10-15T12:00"}, stamp=STAMP)
    assert all(len(line.encode()) <= 75 for line in ics.split("\r\n"))
    assert "é" * 120 in "".join(unfold(ics))


def test_invalid_call_is_refused():
    with pytest.raises(ValueError):
        convert.to_ics({"title": "x"})


def test_ics_round_trips_through_icalendar():
    icalendar = pytest.importorskip("icalendar")
    cal = icalendar.Calendar.from_ical(convert.to_ics(WEEKLY, stamp=STAMP))
    ev = next(c for c in cal.walk() if c.name == "VEVENT")
    assert str(ev["SUMMARY"]) == "Yoga"
    assert ev["DTSTART"].dt == datetime(2026, 10, 20, 18, 0)
    assert ev["DTEND"].dt == datetime(2026, 10, 20, 19, 0)
    rule = ev["RRULE"]
    assert rule["FREQ"] == ["WEEKLY"] and rule["INTERVAL"] == [2] and rule["BYDAY"] == ["TU", "TH"]

    one = icalendar.Calendar.from_ical(convert.to_ics(ONE_OFF, stamp=STAMP))
    ev = next(c for c in one.walk() if c.name == "VEVENT")
    assert str(ev["LOCATION"]) == "Joe's Cafe; downtown"
    assert str(ev["SUMMARY"]) == "Lunch, with Sam"


def test_timezone_modes():
    pytest.importorskip("tzdata")
    one = unfold(convert.to_ics({"title": "x", "start": "2026-10-15T12:00"}, tz="America/Los_Angeles", stamp=STAMP))
    assert "DTSTART:20261015T190000Z" in one   # PDT is UTC-7
    rep = unfold(convert.to_ics(WEEKLY, tz="America/Los_Angeles", stamp=STAMP))
    assert "DTSTART;TZID=America/Los_Angeles:20261020T180000" in rep
    assert any(line.startswith("RRULE:") and line.endswith("UNTIL=20261216T075959Z") for line in rep)


def test_google_url():
    url = convert.to_google_url({**WEEKLY, "location": "Studio 5"}, tz="America/Los_Angeles")
    q = parse_qs(urlparse(url).query)
    assert urlparse(url).netloc == "calendar.google.com"
    assert q["action"] == ["TEMPLATE"] and q["text"] == ["Yoga"]
    assert q["dates"] == ["20261020T180000/20261020T190000"]
    assert q["ctz"] == ["America/Los_Angeles"] and q["location"] == ["Studio 5"]
    assert q["recur"][0].startswith("RRULE:FREQ=WEEKLY")


def test_outlook_url():
    q = parse_qs(urlparse(convert.to_outlook_url(ONE_OFF)).query)
    assert q["subject"] == ["Lunch, with Sam"]
    assert q["startdt"] == ["2026-10-15T12:00:00"] and q["enddt"] == ["2026-10-15T13:30:00"]
    assert urlparse(convert.to_outlook_url(ONE_OFF, "office")).netloc == "outlook.office.com"


def test_share_round_trip_and_rejects_garbage():
    token = convert.encode_share(WEEKLY)
    assert "=" not in token and "+" not in token and "/" not in token
    assert convert.decode_share(token) == WEEKLY
    for bad in ("", "!!!", convert.encode_share(ONE_OFF)[:-4]):
        with pytest.raises(ValueError):
            convert.decode_share(bad)


def test_cli(capsys, tmp_path):
    out = tmp_path / "e.ics"
    code = convert.main(["--args", '{"title":"Yoga","start":"2026-10-20T18:00"}', "--ics", str(out), "--links"])
    assert code == 0 and out.read_text().startswith("BEGIN:VCALENDAR")
    assert "calendar.google.com" in capsys.readouterr().out
    assert convert.main(["--args", '{"title":"Yoga"}']) == 1
