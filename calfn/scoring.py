"""
Scoring for create_event predictions. No torch or API dependency, so it runs anywhere.

Headline metric: pass rate = exact event match, or a correct rejection.
"exact" means the prediction describes the same event after normalize() (explicit defaults
equal omitted ones, titles/locations compare ignoring case and punctuation).
"""

import math
from collections import defaultdict

from .schema import field_matches, same_event, spurious_fields, validate

MATCH_TYPES = (
    "exact", "wrong_args", "invalid", "missed_tool_call", "correct_rejection", "false_positive",
)
PASSING = {"exact", "correct_rejection"}


def classify(pred: dict | None, gold: dict | None) -> str:
    if gold is None:
        return "correct_rejection" if pred is None else "false_positive"
    if pred is None:
        return "missed_tool_call"
    ok, _ = validate(pred)
    if not ok:
        return "invalid"
    return "exact" if same_event(pred, gold) else "wrong_args"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k successes in n trials."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def make_result(index: int, user: str, raw: str, pred, gold, tags, latency_ms: float = 0.0) -> dict:
    match = classify(pred, gold)
    return {
        "index": index,
        "user": user,
        "raw_output": raw,
        "pred": pred,
        "gold": gold,
        "match": match,
        "tags": list(tags or []),
        "latency_ms": round(latency_ms, 1),
    }


def _rate(results: list[dict]) -> dict:
    n = len(results)
    k = sum(1 for r in results if r["match"] in PASSING)
    lo, hi = wilson(k, n)
    return {"n": n, "pass": k, "rate": round(k / n, 4) if n else 0.0,
            "ci95": [round(lo, 4), round(hi, 4)]}


def summarize(results: list[dict]) -> dict:
    """Aggregate per-example results into the numbers that go in results/*.json."""
    overall = _rate(results)

    by_tag = defaultdict(list)
    for r in results:
        for t in r["tags"]:
            by_tag[t].append(r)
    per_tag = {t: _rate(rs) for t, rs in sorted(by_tag.items())}

    breakdown = {m: sum(1 for r in results if r["match"] == m) for m in MATCH_TYPES}

    # Per-field accuracy over examples where both sides made a call, for fields the gold call states;
    # `spurious` counts non-default fields the prediction added that the user never asked for
    field = defaultdict(lambda: [0, 0])
    for r in results:
        if r["gold"] is not None and r["pred"] is not None and r["match"] in ("exact", "wrong_args"):
            for k, ok in field_matches(r["pred"], r["gold"]).items():
                field[k][1] += 1
                field[k][0] += int(ok)
    spurious = defaultdict(int)
    for r in results:
        if r["gold"] is not None and r["pred"] is not None and r["match"] in ("exact", "wrong_args"):
            for k in spurious_fields(r["pred"], r["gold"]):
                spurious[k] += 1
    per_field = {k: {"correct": c, "total": t, "accuracy": round(c / t, 4)}
                 for k, (c, t) in sorted(field.items())}
    for k, n in spurious.items():
        per_field.setdefault(k, {"correct": 0, "total": 0, "accuracy": 0.0})["spurious"] = n

    gold_rej = [r for r in results if r["gold"] is None]
    gold_call = [r for r in results if r["gold"] is not None]
    rejection = {
        "gold_rejections": len(gold_rej),
        "correct": sum(1 for r in gold_rej if r["match"] == "correct_rejection"),
        "false_positives": sum(1 for r in gold_rej if r["match"] == "false_positive"),
        "gold_calls": len(gold_call),
        "missed_calls": sum(1 for r in gold_call if r["match"] == "missed_tool_call"),
    }

    lat = [r["latency_ms"] for r in results if r["latency_ms"]]
    return {
        "overall": overall,
        "breakdown": breakdown,
        "rejection": rejection,
        "per_field": per_field,
        "per_tag": per_tag,
        "mean_latency_ms": round(sum(lat) / len(lat), 1) if lat else None,
    }
