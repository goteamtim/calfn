from datetime import datetime

import pytest

from calfn import parse, schema, scoring

OK = {"title": "Dentist appointment", "start": "2026-10-15T15:00"}


def test_validate_accepts_minimal_and_full():
    assert schema.validate(OK) == (True, "OK")
    full = {**OK, "duration_minutes": 90, "location": "the library", "recurrence_freq": "weekly",
            "recurrence_interval": 2, "recurrence_days": "MO,WE", "recurrence_until": "2026-12-31"}
    assert schema.validate(full) == (True, "OK")


@pytest.mark.parametrize("bad", [
    {"start": "2026-10-15T15:00"},                                     # missing title
    {**OK, "color": "red"},                                            # unknown key
    {**OK, "start": "2026-10-15 15:00"},                               # wrong format
    {**OK, "start": "2026-02-30T15:00"},                               # impossible date
    {**OK, "duration_minutes": 0},
    {**OK, "duration_minutes": 90.5},
    {**OK, "duration_minutes": True},
    {**OK, "recurrence_interval": 2},                                  # without freq
    {**OK, "recurrence_freq": "hourly"},
    {**OK, "recurrence_freq": "daily", "recurrence_days": "MO"},       # days need weekly
    {**OK, "recurrence_freq": "weekly", "recurrence_days": "WE,MO"},   # not in week order
    {**OK, "recurrence_freq": "weekly", "recurrence_days": "MO,MO"},
    {**OK, "recurrence_freq": "daily", "recurrence_count": 3, "recurrence_until": "2026-12-01"},
    {**OK, "recurrence_freq": "daily", "recurrence_until": "2026-10-01"},   # before start
    {**OK, "recurrence_freq": "daily", "recurrence_count": 61},
    {**OK, "location": " "},
])
def test_validate_rejects(bad):
    assert schema.validate(bad)[0] is False


def test_normalize_treats_defaults_as_omitted():
    a = {**OK}
    b = {**OK, "duration_minutes": 60}
    assert schema.same_event(a, b)
    c = {**OK, "recurrence_freq": "daily"}
    d = {**OK, "recurrence_freq": "daily", "recurrence_interval": 1}
    assert schema.same_event(c, d)
    assert not schema.same_event(a, {**OK, "duration_minutes": 45})


def test_same_event_ignores_case_and_punctuation_in_text_only():
    assert schema.same_event(OK, {**OK, "title": "dentist appointment!"})
    assert not schema.same_event(OK, {**OK, "start": "2026-10-15T15:30"})
    loc = {**OK, "location": "Joe's Cafe"}
    assert schema.same_event(loc, {**OK, "location": "joes cafe"})


def test_developer_message_has_locale_proof_now_line():
    msg = schema.developer_message(datetime(2026, 10, 14, 9, 0))
    assert msg.endswith("\nNow: 2026-10-14 Wed 09:00")
    assert msg.startswith(schema.SYSTEM_PROMPT)


def test_format_and_parse_round_trip():
    args = {"title": "Lunch, with Sam: noon", "start": "2026-10-15T12:00", "duration_minutes": 90,
            "recurrence_freq": "weekly", "recurrence_days": "TU,TH"}
    raw = "junk " + parse.format_call(args) + "<start_function_response>"
    assert parse.parse_tool_call(raw) == args


def test_parse_handles_no_call_and_json_fallback():
    assert parse.parse_tool_call("I need a start time to create this event.") is None
    raw = '<tool_call>{"name":"create_event","arguments":{"title":"A","start":"2026-10-15T15:00"}}</tool_call>'
    assert parse.parse_tool_call(raw) == {"title": "A", "start": "2026-10-15T15:00"}


def test_coerce_does_not_truncate_floats():
    out = parse.coerce_types({"duration_minutes": 2.5, "recurrence_count": 3.0, "recurrence_interval": "2"})
    assert out["duration_minutes"] == 2.5          # stays a float, so validation fails
    assert out["recurrence_count"] == 3 and out["recurrence_interval"] == 2
    assert schema.validate({**OK, "duration_minutes": out["duration_minutes"]})[0] is False


def test_classify():
    gold = dict(OK)
    assert scoring.classify(dict(OK), gold) == "exact"
    assert scoring.classify({**OK, "start": "2026-10-16T15:00"}, gold) == "wrong_args"
    assert scoring.classify({**OK, "color": "x"}, gold) == "invalid"
    assert scoring.classify(None, gold) == "missed_tool_call"
    assert scoring.classify(None, None) == "correct_rejection"
    assert scoring.classify(dict(OK), None) == "false_positive"


def test_wilson_matches_known_values():
    lo, hi = scoring.wilson(163, 200)
    assert round(lo, 3) == 0.755 and round(hi, 3) == 0.863
    assert scoring.wilson(0, 0) == (0.0, 0.0)


def test_summarize_counts():
    rs = [
        scoring.make_result(0, "a", "", dict(OK), dict(OK), ["x", "y"]),
        scoring.make_result(1, "b", "", {**OK, "start": "2026-10-16T15:00"}, dict(OK), ["x"]),
        scoring.make_result(2, "c", "", None, None, ["y", "reject"]),
        scoring.make_result(3, "d", "", dict(OK), None, ["reject"]),
    ]
    s = scoring.summarize(rs)
    assert s["overall"]["pass"] == 2 and s["overall"]["n"] == 4
    assert s["per_tag"]["x"]["rate"] == 0.5
    assert s["breakdown"]["false_positive"] == 1
    assert s["rejection"] == {"gold_rejections": 2, "correct": 1, "false_positives": 1,
                              "gold_calls": 2, "missed_calls": 0}
    assert s["per_field"]["start"] == {"correct": 1, "total": 2, "accuracy": 0.5}
    assert "duration_minutes" not in s["per_field"]   # nobody stated a duration


def test_spurious_and_stated_fields():
    gold = {**OK}
    pred = {**OK, "duration_minutes": 30, "location": "home"}
    assert schema.spurious_fields(pred, gold) == ["duration_minutes", "location"]
    assert schema.spurious_fields({**OK, "duration_minutes": 60}, gold) == []   # an explicit default is harmless
    assert schema.field_matches(pred, gold) == {"title": True, "start": True}
    assert schema.field_matches({**OK, "duration_minutes": 45}, {**OK, "duration_minutes": 60}) ==         {"title": True, "start": True, "duration_minutes": False}
