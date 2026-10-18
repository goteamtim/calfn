from datetime import datetime

import pytest

from calfn import parse, schema, scoring

EVENT = {"title": "Dentist appointment", "start": "2026-10-15T15:00"}
CALL = {"title": "Dentist appointment", "date": "wd:TH", "time": "15:00"}


# ------------------------------------------------------------------ resolved event

def test_validate_event_accepts_minimal_and_full():
    assert schema.validate_event(EVENT) == (True, "OK")
    full = {**EVENT, "duration_minutes": 90, "location": "the library", "recurrence_freq": "weekly",
            "recurrence_interval": 2, "recurrence_days": "MO,WE", "recurrence_until": "2026-12-31"}
    assert schema.validate_event(full) == (True, "OK")


@pytest.mark.parametrize("bad", [
    {"start": "2026-10-15T15:00"},                                     # missing title
    {**EVENT, "color": "red"},                                         # unknown key
    {**EVENT, "start": "2026-10-15 15:00"},                            # wrong format
    {**EVENT, "start": "2026-02-30T15:00"},                            # impossible date
    {**EVENT, "duration_minutes": 0},
    {**EVENT, "duration_minutes": 90.5},
    {**EVENT, "duration_minutes": True},
    {**EVENT, "recurrence_interval": 2},                               # without freq
    {**EVENT, "recurrence_freq": "hourly"},
    {**EVENT, "recurrence_freq": "daily", "recurrence_days": "MO"},    # days need weekly
    {**EVENT, "recurrence_freq": "weekly", "recurrence_days": "WE,MO"},   # not in week order
    {**EVENT, "recurrence_freq": "weekly", "recurrence_days": "MO,MO"},
    {**EVENT, "recurrence_freq": "daily", "recurrence_count": 3, "recurrence_until": "2026-12-01"},
    {**EVENT, "recurrence_freq": "daily", "recurrence_until": "2026-10-01"},   # before start
    {**EVENT, "recurrence_freq": "daily", "recurrence_count": 61},
    {**EVENT, "location": " "},
])
def test_validate_event_rejects(bad):
    assert schema.validate_event(bad)[0] is False


def test_normalize_treats_defaults_as_omitted():
    assert schema.same_event(EVENT, {**EVENT, "duration_minutes": 60})
    c = {**EVENT, "recurrence_freq": "daily"}
    assert schema.same_event(c, {**c, "recurrence_interval": 1})
    assert not schema.same_event(EVENT, {**EVENT, "duration_minutes": 45})


def test_same_event_ignores_case_and_punctuation_in_text_only():
    assert schema.same_event(EVENT, {**EVENT, "title": "dentist appointment!"})
    assert not schema.same_event(EVENT, {**EVENT, "start": "2026-10-15T15:30"})
    loc = {**EVENT, "location": "Joe's Cafe"}
    assert schema.same_event(loc, {**EVENT, "location": "joes cafe"})


def test_spurious_and_stated_fields():
    pred = {**EVENT, "duration_minutes": 30, "location": "home"}
    assert schema.spurious_fields(pred, EVENT) == ["duration_minutes", "location"]
    assert schema.spurious_fields({**EVENT, "duration_minutes": 60}, EVENT) == []   # explicit default is harmless
    assert schema.field_matches(pred, EVENT) == {"title": True, "start_date": True, "start_time": True}
    assert schema.field_matches({**EVENT, "start": "2026-10-16T15:00", "duration_minutes": 45},
                                {**EVENT, "duration_minutes": 60}) == \
        {"title": True, "start_date": False, "start_time": True, "duration_minutes": False}


# ------------------------------------------------------------------ call (what the model writes)

def test_validate_call_accepts():
    assert schema.validate_call(CALL) == (True, "OK")
    assert schema.validate_call({**CALL, "end": "16:30", "location": "x"}) == (True, "OK")
    assert schema.validate_call({**CALL, "duration_minutes": 45}) == (True, "OK")
    # daily / weekly series may omit the date; monthly / yearly may not
    undated = {"title": "Yoga", "time": "18:00", "recurrence_freq": "weekly", "recurrence_days": "TU"}
    assert schema.validate_call(undated) == (True, "OK")
    assert schema.validate_call({"title": "x", "time": "07:00", "recurrence_freq": "daily"}) == (True, "OK")
    dated = {**CALL, "date": "dom:15", "recurrence_freq": "monthly", "recurrence_until": "12-15"}
    assert schema.validate_call(dated) == (True, "OK")
    assert schema.validate_call({**CALL, "date": "md:02-29"}) == (True, "OK")      # leap day exists


@pytest.mark.parametrize("bad", [
    {"title": "x", "date": "wd:TH"},                                   # no time
    {**CALL, "start": "2026-10-15T15:00"},                             # the old field
    {**CALL, "color": "red"},
    {**CALL, "time": "3pm"},
    {**CALL, "time": "25:00"},
    {**CALL, "time": "9:00"},
    {**CALL, "end": "14:00"},                                          # before time
    {**CALL, "end": "15:00"},                                          # equal to time
    {**CALL, "end": "16:00", "duration_minutes": 60},                  # exclusive
    {**CALL, "date": "tomorrow"},
    {**CALL, "date": "wd:XX"},
    {**CALL, "date": "wd:TH:1"},                                       # n starts at 2
    {**CALL, "date": "md:02-30"},
    {**CALL, "date": "md:13-01"},
    {**CALL, "date": "nth:5:TU:1"},
    {**CALL, "date": "eom:2"},
    {"title": "x", "time": "15:00"},                                   # one-off with no date
    {"title": "x", "time": "15:00", "recurrence_freq": "monthly"},     # monthly needs a date
    {"title": "x", "time": "15:00", "recurrence_freq": "yearly"},
    {"title": "x", "time": "15:00", "recurrence_freq": "weekly"},      # undated weekly needs days
    {**CALL, "recurrence_freq": "daily", "recurrence_until": "12/15"},
    {**CALL, "recurrence_freq": "daily", "recurrence_until": "02-30"},
    {**CALL, "recurrence_freq": "daily", "recurrence_count": 3, "recurrence_until": "12-15"},
    {**CALL, "recurrence_interval": 2},                                # without freq
])
def test_validate_call_rejects(bad):
    assert schema.validate_call(bad)[0] is False


def test_prompts():
    assert "Now:" not in schema.developer_message()          # the model never sees the date
    msg = schema.direct_developer_message(datetime(2026, 10, 14, 9, 0))
    assert msg.endswith("\nNow: 2026-10-14 Wed 09:00")        # locale-proof weekday
    assert msg.startswith(schema.DIRECT_SYSTEM_PROMPT)


# ------------------------------------------------------------------ parse + score

def test_format_and_parse_round_trip():
    args = {"title": "Lunch, with Sam: noon", "date": "wd:FR", "time": "12:00", "duration_minutes": 90,
            "recurrence_freq": "weekly", "recurrence_days": "TU,TH"}
    raw = "junk " + parse.format_call(args) + "<start_function_response>"
    assert parse.parse_tool_call(raw) == args


def test_parse_handles_no_call_and_json_fallback():
    assert parse.parse_tool_call("I need a start time to create this event.") is None
    raw = '<tool_call>{"name":"create_event","arguments":{"title":"A","time":"15:00"}}</tool_call>'
    assert parse.parse_tool_call(raw) == {"title": "A", "time": "15:00"}


def test_coerce_does_not_truncate_floats():
    out = parse.coerce_types({"duration_minutes": 2.5, "recurrence_count": 3.0, "recurrence_interval": "2"})
    assert out["duration_minutes"] == 2.5          # stays a float, so validation fails
    assert out["recurrence_count"] == 3 and out["recurrence_interval"] == 2
    assert schema.validate_call({**CALL, "duration_minutes": out["duration_minutes"]})[0] is False


def test_classify():
    ev = dict(EVENT)
    assert scoring.classify(CALL, ev, None, EVENT) == "exact"
    assert scoring.classify(CALL, {**EVENT, "start": "2026-10-16T15:00"}, None, EVENT) == "wrong_args"
    assert scoring.classify({**CALL, "color": "x"}, None, "unknown params", EVENT) == "invalid"
    assert scoring.classify(None, None, None, EVENT) == "missed_tool_call"
    assert scoring.classify(None, None, None, None) == "correct_rejection"
    assert scoring.classify(CALL, ev, None, None) == "false_positive"
    assert scoring.classify(CALL, None, "cannot resolve", None) == "false_positive"


def test_wilson_matches_known_values():
    lo, hi = scoring.wilson(163, 200)
    assert round(lo, 3) == 0.755 and round(hi, 3) == 0.863
    assert scoring.wilson(0, 0) == (0.0, 0.0)


def test_summarize_counts():
    other = {**EVENT, "start": "2026-10-16T15:00"}
    rs = [
        scoring.make_result(0, "a", "", CALL, EVENT, ["x", "y"], event=dict(EVENT)),
        scoring.make_result(1, "b", "", CALL, EVENT, ["x"], event=other),
        scoring.make_result(2, "c", "", None, None, ["y", "reject"]),
        scoring.make_result(3, "d", "", CALL, None, ["reject"], event=dict(EVENT)),
    ]
    s = scoring.summarize(rs)
    assert s["overall"]["pass"] == 2 and s["overall"]["n"] == 4
    assert s["per_tag"]["x"]["rate"] == 0.5
    assert s["breakdown"]["false_positive"] == 1
    assert s["rejection"] == {"gold_rejections": 2, "correct": 1, "false_positives": 1,
                              "gold_calls": 2, "missed_calls": 0}
    assert s["per_field"]["start_date"] == {"correct": 1, "total": 2, "accuracy": 0.5}
    assert s["per_field"]["start_time"]["accuracy"] == 1.0
    assert "duration_minutes" not in s["per_field"]   # nobody stated a duration
