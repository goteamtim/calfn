"""
Score a prompted Claude model on the same eval, with the same developer message and tool.

    python scripts/baseline_claude.py --eval data/eval/eval_v1.jsonl --name claude-opus-5-5
    python scripts/baseline_claude.py --eval data/eval/eval_v1.jsonl --model claude-haiku-5-5 --name claude-haiku-5-5

Needs ANTHROPIC_API_KEY (or an `ant auth login` profile). Every request carries the same
SYSTEM_PROMPT conventions and Now line the fine-tuned model sees, so the comparison is fair.
Scoring is just as strict: an extra argument or a wrong type counts as invalid.

Forced tool use is not available on current models, so tool_choice stays "auto" and the
system prompt tells the model to skip the tool when the request can't be completed.
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from inference import load_examples, print_summary, split_example  # noqa: E402

from calfn.schema import TOOL  # noqa: E402
from calfn.scoring import make_result, summarize  # noqa: E402

CLAUDE_TOOL = {"name": TOOL["name"], "description": TOOL["description"], "input_schema": TOOL["parameters"]}


def ask(client, model: str, effort: str | None, dev: str, user: str):
    """Returns (pred or None, raw text, seconds, usage)."""
    kwargs = {}
    if effort:
        kwargs["extra_body"] = {"output_config": {"effort": effort}}
    t0 = time.perf_counter()
    resp = client.messages.create(
        model=model, max_tokens=4096, system=dev, tools=[CLAUDE_TOOL],
        messages=[{"role": "user", "content": user}], **kwargs,
    )
    seconds = time.perf_counter() - t0
    call = next((b for b in resp.content if b.type == "tool_use"), None)
    text = " ".join(b.text for b in resp.content if b.type == "text")
    raw = json.dumps({"name": call.name, "input": call.input}) if call else (text or f"[{resp.stop_reason}]")
    return (dict(call.input) if call else None), raw, seconds, resp.usage


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--effort", choices=["low", "medium", "high"], default=None,
                    help="thinking effort; omit for the model's default")
    ap.add_argument("--name")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    import anthropic
    client = anthropic.Anthropic()
    examples = load_examples(args.eval)[: args.limit]

    def one(i_ex):
        i, ex = i_ex
        dev, user, gold, tags = split_example(ex)
        pred, raw, seconds, usage = ask(client, args.model, args.effort, dev, user)
        return make_result(i, user, raw, pred, gold, tags, seconds * 1000), usage

    with ThreadPoolExecutor(args.workers) as pool:
        done = list(pool.map(one, enumerate(examples)))
    results = [r for r, _ in done]
    tokens_in = sum(u.input_tokens for _, u in done)
    tokens_out = sum(u.output_tokens for _, u in done)

    summary = summarize(results)
    print_summary(summary)
    print(f"Tokens: {tokens_in:,} in, {tokens_out:,} out")

    out = Path(args.out_dir) / f"{args.name or args.model}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "run_date": datetime.now().isoformat(timespec="seconds"),
        "model": args.model, "effort": args.effort, "eval_file": args.eval,
        "tokens": {"input": tokens_in, "output": tokens_out},
        "summary": summary, "results": results,
    }, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
