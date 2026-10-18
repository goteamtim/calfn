"""The date language, the resolver, and the link between the generator's expressions and both."""

import json
import random
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import refparse as P  # noqa: E402

from calfn import dates  # noqa: E402
from calfn.resolve import CallError, event_to_call, is_future, resolve  # noqa: E402
from calfn.schema import validate_call, validate_event  # noqa: E402
from gen import resolver as R  # noqa: E402

TODAY = date(2026, 10, 14)            # a Wednesday
NOW = datetime(2026, 10, 14, 9, 0)
DATA = Path(__file__).parent.parent / "data"


@pytest.mark.parametrize("expr,want", [
    ("in:0d", date(2026, 10, 14)),
    ("in:1d", date(2026, 10, 15)),
    ("in:30d", date(2026, 11, 13)),
    ("in:3w", date(2026, 11, 4)),
    ("wd:TH", date(2026, 10, 15)),
    ("wd:WE", date(2026, 10, 21)),         # a weekday name never means today
    ("wd:TH:2", date(2026, 10, 22)),
    ("nw:MO", date(2026, 10, 19)),
    ("nw:FR", date(2026, 10, 23)),
    ("md:10-14", date(2026, 10, 14)),      # month/day: today counts
    ("md:10-13", date(2027, 10, 13)),
    ("md:02-29", date(2028, 2, 29)),
    ("md:2027-03-03", date(2027, 3, 3)),
    ("dom:14", date(2026, 10, 14)),
    ("dom:13", date(2026, 11, 13)),
    ("dom:31", date(2026, 10, 31)),
    ("eom:0", date(2026, 10, 31)),
    ("eom:1", date(2026, 11, 30)),
    ("nth:1:WE:1", date(2026, 11, 4)),
    ("nth:L:FR:1", date(2026, 11, 27)),
    ("nth:3:TH:0", date(2026, 10, 15)),
    ("dnm:15", date(2026, 11, 15)),
])
def test_resolve_date(expr, want):
    assert dates.resolve_date(expr, TODAY) == want


@pytest.mark.parametrize("expr,today", [
    ("md:02-30", TODAY), ("md:2027-02-29", TODAY), ("dnm:31", date(2026, 1, 20)),   # no Feb 31
    ("dom:32", TODAY), ("md:04-31", TODAY),
])
def test_impossible_dates_raise(expr, today):
    with pytest.raises(dates.DateError):
        dates.resolve_date(expr, today)


@pytest.mark.parametrize("expr", ["tomorrow", "", "wd:", "wd:XX", "wd:TH:1", "in:3", "in:3m", "md:1-5", "eom:2",
                                  "nth:5:TU:0", "nth:L:TU:2", "dom:", "dnm:abc", "IN:1D"])
def test_malformed_expressions(expr):
    with pytest.raises(dates.DateError):
        dates.parse_expr(expr)


def test_first_occurrence_of_a_series():
    assert dates.first_occurrence("daily", (), TODAY) == date(2026, 10, 15)
    assert dates.first_occurrence("weekly", (2,), TODAY) == date(2026, 10, 21)          # not today
    assert dates.first_occurrence("weekly", (0, 2, 4), TODAY) == date(2026, 10, 16)
    assert dates.first_occurrence("weekly", (0, 1, 2, 3, 4), date(2026, 10, 16)) == date(2026, 10, 19)
    with pytest.raises(dates.DateError):
        dates.first_occurrence("monthly", (), TODAY)


# ------------------------------------------------------------------ resolve()

def test_resolve_one_off():
    call = {"title": "Dentist", "date": "in:1d", "time": "15:00"}
    assert resolve(call, NOW) == {"title": "Dentist", "start": "2026-10-15T15:00"}


def test_resolve_end_time_becomes_duration():
    call = {"title": "Poker", "date": "wd:SA", "time": "19:30", "end": "22:00", "location": "the lobby"}
    assert resolve(call, NOW) == {"title": "Poker", "start": "2026-10-17T19:30", "duration_minutes": 150,
                                  "location": "the lobby"}


def test_resolve_series():
    undated = {"title": "Yoga", "time": "18:00", "recurrence_freq": "weekly", "recurrence_days": "TU,TH",
               "recurrence_interval": 1}
    ev = resolve(undated, NOW)
    assert ev["start"] == "2026-10-15T18:00" and ev["recurrence_days"] == "TU,TH"

    monthly = {"title": "Rent", "date": "dom:1", "time": "08:00", "recurrence_freq": "monthly"}
    assert resolve(monthly, NOW)["start"] == "2026-11-01T08:00"

    until = {"title": "Run", "date": "in:1d", "time": "07:00", "recurrence_freq": "daily", "recurrence_until": "01-05"}
    assert resolve(until, NOW)["recurrence_until"] == "2027-01-05"            # next Jan 5
    explicit = {**until, "recurrence_until": "2026-12-01"}
    assert resolve(explicit, NOW)["recurrence_until"] == "2026-12-01"


def test_resolve_refuses_what_cannot_be_an_event():
    with pytest.raises(CallError):                                              # malformed call
        resolve({"title": "x", "date": "tomorrow", "time": "15:00"}, NOW)
    with pytest.raises(CallError):                                              # impossible date
        resolve({"title": "x", "date": "dnm:31", "time": "15:00"}, datetime(2026, 1, 20, 9, 0))
    with pytest.raises(CallError):                                              # series ends before it starts
        resolve({"title": "x", "date": "in:30d", "time": "15:00", "recurrence_freq": "daily",
                 "recurrence_until": "2026-10-20"}, NOW)


def test_is_future():
    ev = resolve({"title": "x", "date": "in:0d", "time": "08:00"}, NOW)         # resolves, but already past
    assert not is_future(ev, NOW)
    assert is_future(resolve({"title": "x", "date": "in:0d", "time": "10:00"}, NOW), NOW)


# ------------------------------------------------------------------ the generator's side

def test_every_generated_expression_resolves_to_the_phrases_target():
    """text -> target (independent parser, in test_resolver.py) and expr -> target (here): so text == expr."""
    rng = random.Random(21)
    kinds = set()
    for _ in range(4000):
        today = date(2026, 1, 1) + timedelta(days=rng.randrange(0, 900))
        target = today + timedelta(days=rng.choice([0, 1, 2, 3, 5, 7, 9, 14, 20, 28, 35, 60, 150, 400]) + rng.choice([0, 0, 1, 3]))
        for family, phrases in R.date_candidates(target, today, rng).items():
            for ph in phrases:
                assert dates.resolve_date(ph.expr, today) == target, (ph.text, ph.expr, today, target)
                assert P.parse_date(ph.text, today) == target
                kinds.add(dates.parse_expr(ph.expr)[0])
    assert kinds == {"in", "wd", "nw", "md", "dom", "eom", "nth", "dnm"}        # every kind is exercised


def test_until_phrase_value_resolves_to_its_date():
    rng = random.Random(22)
    for _ in range(1500):
        today = date(2026, 1, 1) + timedelta(days=rng.randrange(0, 900))
        until = today + timedelta(days=rng.randint(10, 700))
        text, value = R.until_date_phrase(until, today, rng)
        assert dates.resolve_until(value, today) == until, (text, value)


# ------------------------------------------------------------------ the frozen eval files

@pytest.mark.skipif(not (DATA / "eval" / "eval_v1.jsonl").exists(), reason="data not generated")
@pytest.mark.parametrize("name", ["eval_v1", "test_v1", "natural_v1"])
def test_every_eval_event_is_reachable_by_a_call(name):
    """event -> call -> event: the converter-side oracle can express every expected answer."""
    n = 0
    for line in open(DATA / "eval" / f"{name}.jsonl", encoding="utf-8"):
        row = json.loads(line)
        asst = row["messages"][2]
        if not asst.get("tool_calls"):
            continue
        event = asst["tool_calls"][0]["function"]["arguments"]
        now = datetime.strptime(row["meta"]["now"], "%Y-%m-%dT%H:%M")
        assert validate_event(event) == (True, "OK")
        call = event_to_call(event)
        assert validate_call(call) == (True, "OK"), call
        assert resolve(call, now) == event
        n += 1
    assert n > 20
