"""
Seeded generator for train / eval / test data. No LLM involved: every label is computed.

    python -m gen.generate                 # writes data/train.jsonl, data/eval/eval_v1.jsonl, data/eval/test_v1.jsonl
    python -m gen.generate --force         # overwrite the frozen eval/test files (don't, unless eval_v2)

Train uses TRAIN vocabulary only. Eval and test use 70% HELDOUT vocabulary (tag `ood`) and 30%
TRAIN vocabulary with fresh dates and combinations (tag `ind`). No eval/test user text appears
in train, and no test text appears in eval.
"""

import argparse
import json
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from calfn.schema import DAY_CODES, PARAM_ORDER, developer_message, validate  # noqa: E402
from gen import resolver as R  # noqa: E402
from gen import vocab as V  # noqa: E402

SEEDS = {"train": 101, "eval": 202, "test": 303}
REJECT_RATE = 0.15
RECURRING_RATE = {"train": 0.25, "eval": 0.30, "test": 0.30}


@dataclass
class Vocab:
    name: str  # "ind" or "ood"
    titles: list
    locations: list
    frames: list
    rec_frames: list
    short_frames: list
    offtopic: list


TRAIN_VOCAB = Vocab("ind", V.TRAIN_TITLES, V.TRAIN_LOCATIONS, V.TRAIN_FRAMES, V.TRAIN_REC_FRAMES,
                    V.TRAIN_SHORT_FRAMES, V.TRAIN_OFFTOPIC)
HELDOUT_VOCAB = Vocab("ood", V.HELDOUT_TITLES, V.HELDOUT_LOCATIONS, V.HELDOUT_FRAMES, V.HELDOUT_REC_FRAMES,
                      V.HELDOUT_SHORT_FRAMES, V.HELDOUT_OFFTOPIC)


# ------------------------------------------------------------------ samplers

def sample_now(rng: random.Random) -> datetime:
    day = date(2026, 1, 1) + timedelta(days=rng.randrange(0, 730))
    return datetime.combine(day, time(rng.randrange(6, 22), rng.choice([0, 15, 30, 45])))


def sample_delta(rng: random.Random) -> int:
    lo, hi = rng.choices([(0, 0), (1, 7), (8, 30), (31, 120), (121, 400)], [5, 40, 35, 15, 5])[0]
    return rng.randint(lo, hi)


def sample_time(rng: random.Random) -> time:
    hour = rng.choices(range(6, 22), [1, 2, 4, 5, 5, 4, 4, 4, 4, 4, 5, 5, 4, 3, 2, 1])[0]
    minute = rng.choices([0, 30, 15, 45, None], [60, 14, 8, 8, 10])[0]
    if minute is None:
        minute = rng.choice([5, 10, 20, 25, 35, 40, 50, 55])
    return time(hour, minute)


DURATIONS = [15, 30, 30, 45, 60, 60, 90, 90, 120, 120, 150, 180, 240]


def cap(s: str) -> str:
    return s[0].upper() + s[1:]


def style_text(text: str, rng: random.Random) -> str:
    text = " ".join(text.split())
    r = rng.random()
    if r < 0.12:
        text = text.lower()
    elif r < 0.20:
        text = text[0].lower() + text[1:]
    if not re.search(r"[.?!]$", text) and rng.random() < 0.15:
        text += "."
    return text


def pick_frame(frames: list, title_kind: str, rng: random.Random) -> str:
    ok = [p for p, k in frames if k in ("any", title_kind)]
    return rng.choice(ok)


def attach(pattern: str, slots: dict, tails: list[str], rng: random.Random) -> str:
    """Fill a frame; tails (duration, location, series end, ...) go right after the time."""
    rng.shuffle(tails)
    tail_text = (", " if rng.random() < 0.25 else " ").join(tails) if tails else ""
    stripped = re.sub(r"[?.!]+$", "", pattern)
    if stripped.endswith(("{time}", "{tbare}")) and tail_text:
        key = "time" if stripped.endswith("{time}") else "tbare"
        return pattern.format(**{**slots, key: f"{slots[key]} {tail_text}"})
    text = pattern.format(**slots)
    if tail_text:
        trail = re.search(r"[?.!]+$", text)
        text = (text[: trail.start()] if trail else text) + " " + tail_text + (trail.group() if trail else "")
    return text


# ------------------------------------------------------------------ time / duration plan

def plan_time(rng: random.Random, t: time) -> dict:
    """Pick how the start time (and optional length) is phrased; returns phrase pieces and the duration label."""
    mode = rng.choices(["none", "dur", "until", "range"], [50, 30, 10, 10])[0]
    start_min = t.hour * 60 + t.minute
    dur = rng.choice(DURATIONS)
    if mode != "none" and mode != "dur" and start_min + dur > 23 * 60 + 45:
        mode = "none"
    if mode == "none":
        return {"tp": R.pick_time(t, rng), "tails": [], "duration": None, "tags": []}
    if mode == "dur":
        return {"tp": R.pick_time(t, rng), "tails": [R.duration_phrase(dur, rng)], "duration": dur,
                "tags": ["dur"]}
    end = time((start_min + dur) // 60, (start_min + dur) % 60)
    if mode == "until":
        return {"tp": R.pick_time(t, rng), "tails": [R.until_phrase(end, rng)], "duration": dur,
                "tags": ["dur_until"]}
    return {"tp": R.range_phrase(t, end, rng), "tails": [], "duration": dur, "tags": ["dur_range"]}


def plan_location(rng: random.Random, vocab: Vocab):
    if rng.random() < 0.25:
        prep, name = rng.choice(vocab.locations)
        return f"{prep} {name}", name
    return None, None


def tags_for(vocab: Vocab, dp, tp, plan, loc, title: str) -> list[str]:
    tags = [vocab.name]
    if dp is not None:
        tags.append(f"date_{dp.kind}")
        if re.search(r"\b20\d\d\b", dp.text):
            tags.append("year_explicit")
    tags.append(f"time_{tp.kind}")
    tags += plan["tags"]
    if loc:
        tags.append("loc")
    if re.search(r"\d", title) or (loc and re.search(r"\d", loc)):
        tags.append("distractor")
    return tags


# ------------------------------------------------------------------ builders

def build_single(rng, now, vocab):
    today = now.date()
    for _ in range(30):
        target = today + timedelta(days=sample_delta(rng))
        t = sample_time(rng)
        if datetime.combine(target, t) <= now:
            continue
        plan = plan_time(rng, t)
        dp = R.pick_date(target, today, rng)
        title, kind = rng.choice(vocab.titles)
        loc_tail, loc = plan_location(rng, vocab)
        tails = plan["tails"] + ([loc_tail] if loc_tail else [])
        tp = plan["tp"]
        slots = {"title": title, "Title": cap(title), "date": dp.text, "on_date": dp.on,
                 "time": tp.text, "tbare": tp.bare}
        text = attach(pick_frame(vocab.frames, kind, rng), slots, tails, rng)
        gold = {"title": cap(title), "start": datetime.combine(target, t).strftime("%Y-%m-%dT%H:%M")}
        if plan["duration"]:
            gold["duration_minutes"] = plan["duration"]
        if loc:
            gold["location"] = loc
        return text, gold, tags_for(vocab, dp, tp, plan, loc, title)
    return None


def sample_spec(rng: random.Random) -> R.RecurrenceSpec:
    freq = rng.choices(["daily", "weekly", "monthly", "yearly"], [15, 55, 18, 12])[0]
    if freq == "daily":
        return R.RecurrenceSpec("daily", rng.choice([1, 1, 1, 1, 2, 3]))
    if freq == "weekly":
        shape = rng.choices(["one", "two", "three", "weekdays", "weekend"], [55, 20, 10, 10, 5])[0]
        if shape == "weekdays":
            return R.RecurrenceSpec("weekly", 1, (0, 1, 2, 3, 4))
        if shape == "weekend":
            return R.RecurrenceSpec("weekly", 1, (5, 6))
        k = {"one": 1, "two": 2, "three": 3}[shape]
        days = tuple(sorted(rng.sample(range(7), k)))
        interval = rng.choices([1, 2, 3], [70, 22, 8])[0] if k == 1 else 1
        return R.RecurrenceSpec("weekly", interval, days)
    if freq == "monthly":
        return R.RecurrenceSpec("monthly", rng.choices([1, 2, 3], [80, 12, 8])[0], day=rng.randint(1, 28))
    month = rng.randint(1, 12)
    return R.RecurrenceSpec("yearly", 1, month=month, day=rng.randint(1, 28 if month == 2 else 30))


def build_recurring(rng, now, vocab):
    today = now.date()
    for _ in range(30):
        spec = sample_spec(rng)
        t = sample_time(rng)

        explicit = None
        if spec.freq in ("daily", "weekly") and rng.random() < 0.35:
            cands = [today + timedelta(days=d) for d in range(0, 60)
                     if spec.freq == "daily" or (today + timedelta(days=d)).weekday() in spec.days]
            explicit = rng.choice(cands)
        start_d = explicit or R.recurrence_first_date(spec, today)
        if datetime.combine(start_d, t) <= now:
            continue

        tails = []
        tags = []
        if explicit:
            tails.append(f"starting {R.pick_date(explicit, today, rng).text}")
            tags.append("rec_dated")
        else:
            tags.append("rec_undated")

        gold = {"title": "", "start": datetime.combine(start_d, t).strftime("%Y-%m-%dT%H:%M"),
                "recurrence_freq": spec.freq}
        if spec.interval > 1:
            gold["recurrence_interval"] = spec.interval
            tags.append("rec_interval")
        if spec.freq == "weekly":
            gold["recurrence_days"] = ",".join(DAY_CODES[d] for d in spec.days)

        end = rng.choices(["none", "count", "until"], [55, 25, 20])[0]
        if end == "count":
            count = rng.randint(3, 14) if spec.freq == "daily" else rng.randint(2, 12)
            gold["recurrence_count"] = count
            tails.append(R.count_phrase(spec, count, rng))
            tags.append("rec_count")
        elif end == "until":
            span = {"daily": (7, 120), "weekly": (14, 300), "monthly": (60, 700), "yearly": (400, 2000)}[spec.freq]
            until = start_d + timedelta(days=rng.randint(*span))
            gold["recurrence_until"] = until.isoformat()
            tails.append(R.until_date_phrase(until, today, rng))
            tags.append("rec_until")

        plan = plan_time(rng, t)
        tp = plan["tp"]
        tails += plan["tails"]
        if plan["duration"]:
            gold["duration_minutes"] = plan["duration"]
        loc_tail, loc = plan_location(rng, vocab)
        if loc:
            tails.append(loc_tail)
            gold["location"] = loc

        title, kind = rng.choice(vocab.titles)
        gold["title"] = cap(title)
        slots = {"title": title, "Title": cap(title), "rec": R.recurrence_phrase(spec, rng),
                 "time": tp.text, "tbare": tp.bare}
        text = attach(pick_frame(vocab.rec_frames, kind, rng), slots, tails, rng)

        base = tags_for(vocab, None, tp, plan, loc, title)
        base += ["recurring", f"rec_{spec.freq}"] + tags
        gold = {k: gold[k] for k in sorted(gold, key=PARAM_ORDER.index)}
        return text, gold, base
    return None


REJECT_WEIGHTS = {"offtopic": 25, "notime": 22, "nodate": 12, "vague": 16, "ambiguous": 10, "past": 7, "baddate": 8}


def build_rejection(rng, now, vocab, allow_offtopic: bool = True):
    today = now.date()
    weights = dict(REJECT_WEIGHTS)
    if not allow_offtopic:
        weights.pop("offtopic")
    cat = rng.choices(list(weights), list(weights.values()))[0]
    tags = [vocab.name, "reject", f"reject_{cat}"]

    if cat == "offtopic":
        return rng.choice(vocab.offtopic), cat, tags

    title, kind = rng.choice(vocab.titles)
    target = today + timedelta(days=sample_delta(rng))
    t = sample_time(rng)
    dp = R.pick_date(target, today, rng)
    tp = R.pick_time(t, rng)

    if cat == "notime":
        if rng.random() < 0.3:
            when = R.recurrence_phrase(sample_spec(rng), rng)
        else:
            when = dp.on
    elif cat == "nodate":
        when = tp.text
    elif cat == "vague":
        when = rng.choice(V.VAGUE_WHEN)
    elif cat == "ambiguous":
        hour = rng.randint(1, 12)
        when = f"{dp.on} {R.ambiguous_time_phrase(hour, rng.choice([0, 0, 30]))}"
    elif cat == "past":
        when = f"{rng.choice(V.PAST_WHEN)} {tp.text}"
    else:  # baddate
        when = f"on {rng.choice(V.BAD_DATES)} {tp.text}"
    pattern = pick_frame(vocab.short_frames, kind, rng)
    text = pattern.format(title=title, Title=cap(title), when=when)
    return text, cat, tags


# ------------------------------------------------------------------ rows

def make_row(split: str, idx: int, now: datetime, text: str, gold, tags, reject_cat=None, rng=None) -> dict:
    messages = [
        {"role": "developer", "content": developer_message(now)},
        {"role": "user", "content": text},
    ]
    if gold is None:
        messages.append({"role": "assistant", "content": rng.choice(V.REJECTION_TEXT[reject_cat])})
    else:
        messages.append({"role": "assistant", "content": None, "tool_calls": [{
            "type": "function", "function": {"name": "create_event", "arguments": gold}}]})
    return {"messages": messages,
            "meta": {"id": f"{split}-{idx:04d}", "now": now.strftime("%Y-%m-%dT%H:%M"), "tags": tags}}


def norm_text(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()


def generate_split(split: str, n: int, forbidden: set[str]) -> list[dict]:
    rng = random.Random(SEEDS[split])
    rows, seen = [], set()
    attempts = 0
    while len(rows) < n:
        attempts += 1
        if attempts > n * 200:
            raise RuntimeError(f"could not generate {n} unique {split} rows")
        now = sample_now(rng)
        if split == "train":
            vocab = TRAIN_VOCAB
        else:
            vocab = HELDOUT_VOCAB if rng.random() < 0.7 else TRAIN_VOCAB

        if rng.random() < REJECT_RATE:
            # held-out off-topic prompts are the only ones guaranteed unseen in train
            built = build_rejection(rng, now, vocab, allow_offtopic=(split == "train" or vocab.name == "ood"))
            text, cat, tags = built
            gold = None
        else:
            builder = build_recurring if rng.random() < RECURRING_RATE[split] else build_single
            built = builder(rng, now, vocab)
            if built is None:
                continue
            text, gold, tags = built
            cat = None
            ok, msg = validate(gold)
            assert ok, f"generator produced an invalid label: {msg} {gold}"

        text = style_text(text, rng)
        key = norm_text(text)
        if split == "train":
            key = f"{key}|{now.date()}"
        if key in seen or norm_text(text) in forbidden:
            continue
        seen.add(key)
        rows.append(make_row(split, len(rows), now, text, gold, tags, cat, rng))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def user_texts(rows: list[dict]) -> set[str]:
    return {norm_text(next(m["content"] for m in r["messages"] if m["role"] == "user")) for r in rows}


def describe(name: str, rows: list[dict]) -> None:
    tags = Counter(t for r in rows for t in r["meta"]["tags"])
    rej = sum(1 for r in rows if "reject" in r["meta"]["tags"])
    print(f"{name}: {len(rows)} rows, {rej} rejections")
    print("  " + ", ".join(f"{t}={c}" for t, c in sorted(tags.items())))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data")
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-eval", type=int, default=300)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--force", action="store_true", help="overwrite the frozen eval/test files")
    args = ap.parse_args()

    out = Path(args.out)
    eval_path, test_path = out / "eval" / "eval_v1.jsonl", out / "eval" / "test_v1.jsonl"
    if (eval_path.exists() or test_path.exists()) and not args.force:
        sys.exit(f"{eval_path} / {test_path} exist and are frozen. Pass --force only to make a new eval version.")

    train = generate_split("train", args.n_train, set())
    train_text = user_texts(train)
    eval_rows = generate_split("eval", args.n_eval, train_text)
    test_rows = generate_split("test", args.n_test, train_text | user_texts(eval_rows))

    write_jsonl(out / "train.jsonl", train)
    write_jsonl(eval_path, eval_rows)
    write_jsonl(test_path, test_rows)
    for name, rows in (("train", train), ("eval", eval_rows), ("test", test_rows)):
        describe(name, rows)


if __name__ == "__main__":
    main()
