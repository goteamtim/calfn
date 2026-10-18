import json
from datetime import datetime
from pathlib import Path

import pytest

from calfn import schema
from calfn.resolve import resolve
from gen import generate, resolver as R

DATA = Path(__file__).parent.parent / "data"


def rows_of(split, n):
    return generate.generate_split(split, n, set())


def test_generation_is_deterministic():
    assert rows_of("train", 60) == rows_of("train", 60)


def test_train_rows_hold_a_call_that_resolves_to_the_event():
    for row in rows_of("train", 600):
        now = datetime.strptime(row["meta"]["now"], "%Y-%m-%dT%H:%M")
        dev, user, asst = row["messages"]
        assert dev["content"] == schema.developer_message() and "Now:" not in dev["content"]
        call = (asst.get("tool_calls") or [None])[0]
        if call is None:
            assert "reject" in row["meta"]["tags"] and asst["content"] and "event" not in row["meta"]
            continue
        args, event = call["function"]["arguments"], row["meta"]["event"]
        assert schema.validate_call(args) == (True, "OK"), args
        assert schema.validate_event(event) == (True, "OK"), event
        assert resolve(args, now) == event, (user["content"], args, event)
        assert list(args) == sorted(args, key=schema.CALL_ORDER.index)
        assert schema.parse_start(event["start"]) > now, (user["content"], event, now)


def test_eval_and_test_rows_hold_the_resolved_event_with_the_direct_prompt():
    for split in ("eval", "test"):
        for row in rows_of(split, 400):
            now = datetime.strptime(row["meta"]["now"], "%Y-%m-%dT%H:%M")
            dev, user, asst = row["messages"]
            assert dev["content"] == schema.direct_developer_message(now)
            call = (asst.get("tool_calls") or [None])[0]
            if call is None:
                assert "reject" in row["meta"]["tags"]
                continue
            event = call["function"]["arguments"]
            assert schema.validate_event(event) == (True, "OK"), event
            assert schema.parse_start(event["start"]) > now
            assert "event" not in row["meta"]


def test_dated_and_undated_series_starts():
    for row in rows_of("train", 1500):
        if "rec_undated" not in row["meta"]["tags"] or "event" not in row["meta"]:
            continue
        now = datetime.strptime(row["meta"]["now"], "%Y-%m-%dT%H:%M")
        a = row["meta"]["event"]
        start = datetime.strptime(a["start"], "%Y-%m-%dT%H:%M").date()
        spec = R.RecurrenceSpec(
            a["recurrence_freq"], a.get("recurrence_interval", 1),
            tuple(schema.DAY_CODES.index(c) for c in a["recurrence_days"].split(",")) if "recurrence_days" in a else (),
            month=start.month, day=start.day)
        assert R.recurrence_first_date(spec, now.date()) == start, (row["messages"][1]["content"], a)


def test_undated_daily_and_weekly_series_have_no_date_in_the_call():
    seen = set()
    for row in rows_of("train", 1500):
        tags, asst = row["meta"]["tags"], row["messages"][2]
        if "rec_undated" in tags and asst.get("tool_calls"):
            args = asst["tool_calls"][0]["function"]["arguments"]
            freq = args["recurrence_freq"]
            assert ("date" in args) == (freq in ("monthly", "yearly")), args
            seen.add(freq)
        if "rec_dated" in tags:
            assert "date" in asst["tool_calls"][0]["function"]["arguments"]
    assert seen == {"daily", "weekly", "monthly", "yearly"}


def test_end_time_replaces_duration_for_ranges_and_until():
    n = 0
    for row in rows_of("train", 800):
        tags, asst = row["meta"]["tags"], row["messages"][2]
        if not asst.get("tool_calls"):
            continue
        args = asst["tool_calls"][0]["function"]["arguments"]
        if "dur_range" in tags or "dur_until" in tags:
            assert "end" in args and "duration_minutes" not in args
            n += 1
        if "dur" in tags:
            assert "duration_minutes" in args and "end" not in args
    assert n > 20


def test_reject_rate_is_about_fifteen_percent():
    rows = rows_of("train", 1500)
    rate = sum("reject" in r["meta"]["tags"] for r in rows) / len(rows)
    assert 0.11 < rate < 0.19


def test_train_vocab_never_leaks_heldout():
    held_titles = {t for t, _ in generate.HELDOUT_VOCAB.titles}
    train_text = " ".join(r["messages"][1]["content"].lower() for r in rows_of("train", 800))
    assert not any(t.lower() in train_text for t in held_titles)
    assert not any(p.lower() in train_text for p in generate.HELDOUT_VOCAB.offtopic)


@pytest.mark.skipif(not (DATA / "train.jsonl").exists(), reason="data not generated")
def test_committed_splits_are_disjoint():
    def texts(path):
        return {generate.norm_text(json.loads(l)["messages"][1]["content"]) for l in open(path, encoding="utf-8")}
    train = texts(DATA / "train.jsonl")
    ev, test = texts(DATA / "eval" / "eval_v1.jsonl"), texts(DATA / "eval" / "test_v1.jsonl")
    assert not train & ev and not train & test and not ev & test


@pytest.mark.skipif(not (DATA / "eval" / "eval_v1.jsonl").exists(), reason="data not generated")
def test_frozen_eval_and_test_still_reproduce_byte_for_byte():
    train_text = generate.user_texts(rows_of("train", 3000))
    eval_rows = generate.generate_split("eval", 300, train_text)
    test_rows = generate.generate_split("test", 200, train_text | generate.user_texts(eval_rows))
    assert (DATA / "eval" / "eval_v1.jsonl").read_bytes() == generate._jsonl_bytes(eval_rows)
    assert (DATA / "eval" / "test_v1.jsonl").read_bytes() == generate._jsonl_bytes(test_rows)


@pytest.mark.skipif(not (DATA / "train.jsonl").exists(), reason="data not generated")
def test_committed_train_matches_the_generator():
    assert (DATA / "train.jsonl").read_bytes() == generate._jsonl_bytes(rows_of("train", 3000))
