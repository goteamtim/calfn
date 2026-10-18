"""
Turn results/*.json into a markdown table (results/summary.md).

    python scripts/summarize.py                 # all files in results/
    python scripts/summarize.py results/dev     # a subfolder

Run it per eval file: results from dev (eval_v1) and test (test_v1) should never share a table
unless you want them to, so keep them in separate folders (results/dev, results/test).
"""

import json
import sys
from pathlib import Path


def main() -> None:
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    runs = []
    for p in sorted(folder.glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        if "summary" in d:
            runs.append((p.stem, d))
    if not runs:
        sys.exit(f"no result files in {folder}")

    lines = [f"# Results ({folder})", "",
             "| run | task | eval file | pass | 95% CI | rejections ok | false pos | missed calls | date | time | ms |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, d in runs:
        s = d["summary"]
        o, r = s["overall"], s["rejection"]
        pf = s["per_field"]
        # v1 result files predate the split of `start` into date and time (and the `task` key)
        date_acc = (pf.get("start_date") or pf.get("start") or {}).get("accuracy")
        time_acc = (pf.get("start_time") or {}).get("accuracy")
        pct = lambda v: "" if v is None else f"{v:.0%}"  # noqa: E731
        lat = s["mean_latency_ms"]
        lines.append(
            f"| {name} | {d.get('task', 'direct')} | {Path(d['eval_file']).name} | {o['rate']:.1%} | "
            f"{o['ci95'][0]:.0%}-{o['ci95'][1]:.0%} | {r['correct']}/{r['gold_rejections']} | {r['false_positives']} | "
            f"{r['missed_calls']}/{r['gold_calls']} | {pct(date_acc)} | {pct(time_acc)} | {'' if not lat else f'{lat:.0f}'} |")

    tags = sorted({t for _, d in runs for t in d["summary"]["per_tag"]})
    lines += ["", "## Pass rate by tag", "",
              "| tag | n | " + " | ".join(n for n, _ in runs) + " |",
              "|---|---|" + "---|" * len(runs)]
    for t in tags:
        n = max(d["summary"]["per_tag"].get(t, {"n": 0})["n"] for _, d in runs)
        cells = []
        for _, d in runs:
            v = d["summary"]["per_tag"].get(t)
            cells.append(f"{v['rate']:.0%}" if v else "")
        lines.append(f"| {t} | {n} | " + " | ".join(cells) + " |")

    out = folder / "summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
