"""
Score a local Ollama model on the same eval, with the same developer message and tool.

    python scripts/baseline_ollama.py --eval data/eval/eval_v1.jsonl --model gemma4:12b --name ollama-gemma4-12b

Needs a running Ollama server (default http://localhost:11434). Same SYSTEM_PROMPT conventions, Now line
and strict scoring as baseline_claude.py. Greedy decoding (temperature 0). Thinking is off unless
--think is passed, so latency is comparable to the fine-tuned model.
"""

import argparse
import json
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from inference import load_examples, print_summary, split_example  # noqa: E402

from calfn.schema import TOOL  # noqa: E402
from calfn.scoring import make_result, summarize  # noqa: E402

OLLAMA_TOOL = {"type": "function", "function": TOOL}


def ask(url: str, model: str, think: bool, num_ctx: int, dev: str, user: str):
    """Returns (pred or None, raw text, seconds, (prompt tokens, output tokens))."""
    body = {
        "model": model, "stream": False, "think": think, "tools": [OLLAMA_TOOL],
        "messages": [{"role": "system", "content": dev}, {"role": "user", "content": user}],
        "options": {"temperature": 0, "num_ctx": num_ctx},
    }
    req = urllib.request.Request(f"{url}/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as r:
        resp = json.load(r)
    seconds = time.perf_counter() - t0
    msg = resp["message"]
    calls = msg.get("tool_calls") or []
    call = calls[0]["function"] if calls else None
    raw = json.dumps({"name": call["name"], "input": call["arguments"]}) if call else (msg.get("content") or "[no output]")
    return (dict(call["arguments"]) if call else None), raw, seconds, (resp.get("prompt_eval_count", 0), resp.get("eval_count", 0))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--name")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--think", action="store_true", help="let a reasoning model think before answering")
    ap.add_argument("--num-ctx", type=int, default=4096)
    ap.add_argument("--url", default="http://localhost:11434")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    examples = load_examples(args.eval)[: args.limit]
    results, tokens_in, tokens_out = [], 0, 0
    for i, ex in enumerate(examples):
        dev, user, gold, tags = split_example(ex)
        pred, raw, seconds, (t_in, t_out) = ask(args.url, args.model, args.think, args.num_ctx, dev, user)
        tokens_in += t_in
        tokens_out += t_out
        results.append(make_result(i, user, raw, pred, gold, tags, seconds * 1000))
        print(f"[{i + 1:>3}/{len(examples)}] {results[-1]['match']:<18} | {user[:60]}", flush=True)

    summary = summarize(results)
    print_summary(summary)
    print(f"Tokens: {tokens_in:,} in, {tokens_out:,} out")

    out = Path(args.out_dir) / f"{args.name or args.model.replace(':', '-')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "run_date": datetime.now().isoformat(timespec="seconds"),
        "model": args.model, "think": args.think, "eval_file": args.eval,
        "tokens": {"input": tokens_in, "output": tokens_out},
        "summary": summary, "results": results,
    }, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
