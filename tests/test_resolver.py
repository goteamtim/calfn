"""The labeler is checked against an independent reference parser (tests/refparse.py)."""

import random
import sys
from datetime import date, time, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import refparse as P  # noqa: E402

from gen import resolver as R  # noqa: E402

TODAY = date(2026, 10, 14)  # a Wednesday


def random_today(rng):
    return date(2026, 1, 1) + timedelta(days=rng.randrange(0, 900))


def test_wednesday_anchor():
    assert TODAY.weekday() == 2


def test_date_phrases_parse_back_to_their_target():
    rng = random.Random(7)
    checked = 0
    for _ in range(4000):
        today = random_today(rng)
        delta = rng.choice([0, 1, 2, 3, 5, 7, 9, 14, 20, 28, 35, 60, 150, 400])
        target = today + timedelta(days=delta + rng.choice([0, 0, 1, 3]))
        for family, phrases in R.date_candidates(target, today, rng).items():
            for ph in phrases:
                for text in (ph.text, ph.on):
                    got = P.parse_date(text, today)
                    assert got == target, f"{text!r} said on {today} ({today:%a}) -> {got}, wanted {target}"
                    checked += 1
    assert checked > 20000


def test_this_and_next_only_when_unambiguous():
    rng = random.Random(8)
    for _ in range(3000):
        today = random_today(rng)
        target = today + timedelta(days=rng.randint(1, 20))
        for ph in R.date_candidates(target, today, rng).get("weekday", []):
            if ph.text.startswith("this "):
                assert P.monday_of(target) == P.monday_of(today)
            if ph.text.startswith("next "):
                assert P.monday_of(target) == P.monday_of(today) + timedelta(days=7)


def test_every_target_has_a_phrase():
    rng = random.Random(9)
    for _ in range(2000):
        today = random_today(rng)
        target = today + timedelta(days=rng.randint(0, 400))
        assert R.date_candidates(target, today, rng)


def test_convention_examples():
    f = lambda s: P.parse_date(s, TODAY)  # noqa: E731
    assert f("Thursday") == date(2026, 10, 15)
    assert f("next Monday") == date(2026, 10, 19)
    assert f("Friday next week") == date(2026, 10, 23)
    assert f("a week from Friday") == date(2026, 10, 23)
    assert f("Oct 14") == TODAY
    assert f("the 14th") == TODAY
    assert f("the 13th") == date(2026, 11, 13)
    assert f("Feb 29") == date(2028, 2, 29)
    assert f("the end of the month") == date(2026, 10, 31)
    assert f("the last Friday of next month") == date(2026, 11, 27)
    # the resolver's own arithmetic agrees
    assert R.next_weekday(TODAY, 2) == date(2026, 10, 21)  # Wednesday names the NEXT Wednesday
    assert R.next_month_day(TODAY, 2, 29) == date(2028, 2, 29)
    assert R.next_day_of_month(TODAY, 31) == date(2026, 10, 31)
    assert R.next_day_of_month(date(2026, 11, 14), 31) == date(2026, 12, 31)


def test_time_phrases_parse_back():
    rng = random.Random(10)
    for hour in range(6, 22):
        for minute in range(0, 60, 5):
            t = time(hour, minute)
            for _ in range(6):
                for family, phrases in R.time_candidates(t, rng).items():
                    for ph in phrases:
                        assert P.parse_clock(ph.bare) == t, f"{ph.bare!r} -> {P.parse_clock(ph.bare)} not {t}"
                        assert ph.text == f"at {ph.bare}"


def test_ranges_durations_and_until():
    rng = random.Random(11)
    for _ in range(2000):
        start_min = rng.randrange(6 * 60, 20 * 60, 5)
        dur = rng.choice(R.__dict__.get("DURS", [15, 30, 45, 60, 75, 90, 120, 150, 180, 240]))
        end_min = start_min + dur
        if end_min > 23 * 60 + 45:
            continue
        t1, t2 = time(start_min // 60, start_min % 60), time(end_min // 60, end_min % 60)
        assert P.parse_range(R.range_phrase(t1, t2, rng).bare) == (t1, t2)
        assert P.parse_duration(R.duration_phrase(dur, rng)) == dur
        phrase = R.until_phrase(t2, rng)
        assert P.parse_clock(phrase.split(" ", 1)[1]) == t2


def test_recurrence_phrases_parse_back():
    rng = random.Random(12)
    for _ in range(3000):
        freq = rng.choice(["daily", "weekly", "monthly", "yearly"])
        if freq == "daily":
            spec = R.RecurrenceSpec("daily", rng.choice([1, 2, 3, 4]))
        elif freq == "weekly":
            days = tuple(sorted(rng.sample(range(7), rng.choice([1, 2, 3]))))
            spec = R.RecurrenceSpec("weekly", rng.choice([1, 2, 3]) if len(days) == 1 else 1, days)
        elif freq == "monthly":
            spec = R.RecurrenceSpec("monthly", rng.choice([1, 2, 3]), day=rng.randint(1, 28))
        else:
            spec = R.RecurrenceSpec("yearly", 1, month=rng.randint(1, 12), day=rng.randint(1, 28))
        text = R.recurrence_phrase(spec, rng)
        got = P.parse_recurrence(text)
        want = (spec.freq, spec.interval, spec.days, spec.month, spec.day)
        assert got == want, f"{text!r} -> {got}, wanted {want}"


def test_weekday_sets_have_dedicated_phrases():
    rng = random.Random(13)
    assert "weekday" in R.recurrence_phrase(R.RecurrenceSpec("weekly", 1, (0, 1, 2, 3, 4)), rng).lower() \
        or "Friday" in R.recurrence_phrase(R.RecurrenceSpec("weekly", 1, (0, 1, 2, 3, 4)), rng)


def test_recurrence_first_date():
    spec = R.RecurrenceSpec
    assert R.recurrence_first_date(spec("daily"), TODAY) == date(2026, 10, 15)
    assert R.recurrence_first_date(spec("weekly", 1, (2,)), TODAY) == date(2026, 10, 21)  # not today
    assert R.recurrence_first_date(spec("weekly", 1, (0, 2, 4)), TODAY) == date(2026, 10, 16)
    assert R.recurrence_first_date(spec("weekly", 1, (0, 1, 2, 3, 4)), date(2026, 10, 16)) == date(2026, 10, 19)
    assert R.recurrence_first_date(spec("monthly", 1, day=14), TODAY) == TODAY
    assert R.recurrence_first_date(spec("yearly", 1, month=10, day=13), TODAY) == date(2027, 10, 13)


def test_ordinals():
    assert [R.ordinal(n) for n in (1, 2, 3, 4, 11, 12, 13, 21, 22, 23, 30, 31)] == \
        ["1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "22nd", "23rd", "30th", "31st"]
