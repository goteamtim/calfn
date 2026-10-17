import json
from datetime import datetime
from pathlib import Path

import pytest

from calfn import schema
from gen import generate, resolver as R

DATA = Path(__file__).parent.parent / "data"


def rows_of(split, n):
    return generate.generate_split(split, n, set())


def test_generation_is_deterministic():
    assert rows_of("train", 60) == rows_of("train", 60)


def test_every_label_is_valid_and_in_the_future():
    for split in ("train", "eval", "test"):
        for row in rows_of(split, 400):
            now = datetime.strptime(row["meta"]["now"], "%Y-%m-%dT%H:%M")
            dev, user, asst = row["messages"]
            assert dev["content"] == schema.developer_message(now)
            call = (asst.get("tool_calls") or [None])[0]
            if call is None:
                assert "reject" in row["meta"]["tags"] and asst["content"]
                continue
            args = call["function"]["arguments"]
            assert schema.validate(args) == (True, "OK"), args
            assert schema.parse_start(args["start"]) > now, (user["content"], args, now)
            assert list(args) == sorted(args, key=schema.PARAM_ORDER.index)
            # a title that is the user's own words
            assert schema.same_text(args["title"], args["title"])


def test_dated_and_undated_recurrence_starts():
    for row in rows_of("train", 1500):
        tags = row["meta"]["tags"]
        if "rec_undated" not in tags:
            continue
        now = datetime.strptime(row["meta"]["now"], "%Y-%m-%dT%H:%M")
        a = row["messages"][2]["tool_calls"][0]["function"]["arguments"]
        start = datetime.strptime(a["start"], "%Y-%m-%dT%H:%M").date()
        spec = R.RecurrenceSpec(
            a["recurrence_freq"], a.get("recurrence_interval", 1),
            tuple(schema.DAY_CODES.index(c) for c in a["recurrence_days"].split(",")) if "recurrence_days" in a else (),
            month=start.month, day=start.day)
        assert R.recurrence_first_date(spec, now.date()) == start, (row["messages"][1]["content"], a)


def test_reject_rate_is_about_fifteen_percent():
    rows = rows_of("train", 1500)
    rate = sum("reject" in r["meta"]["tags"] for r in rows) / len(rows)
    assert 0.11 < rate < 0.19


def test_train_vocab_never_leaks_heldout_and_vice_versa():
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
