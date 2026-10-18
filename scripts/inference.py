"""
Run a model on calfn prompts and score it.

    # score a fine-tuned model (or the untuned base) on the dev eval
    python scripts/inference.py --model output/calfn-v2 --eval data/eval/eval_v1.jsonl --name calfn-v2 --out-dir results/dev
    python scripts/inference.py --model unsloth/functiongemma-270m-it --eval data/eval/eval_v1.jsonl --name base-extract --out-dir results/dev

    # a v1 checkpoint (it wrote the event itself, told "Now: ..."): score it with the direct task
    python scripts/inference.py --model output/calfn-v1 --task direct --eval data/eval/eval_v1.jsonl --name calfn-v1

    # one prompt (--now defaults to the current local time)
    python scripts/inference.py --model output/calfn-v2 --prompt "lunch with Sam next Friday at noon"

    # check the harness without a model: answers are the gold labels, so everything must pass
    python scripts/inference.py --oracle --eval data/eval/eval_v1.jsonl --name oracle

Tasks (calfn/task.py): `extract` (default) the model writes the date and time as said and code resolves
the real date; `direct` the model is told "Now: ..." and writes the event itself (the v1 approach).
Either way the result is scored as an event against the frozen eval labels.

Results are written to <out-dir>/<name>.json. Greedy decoding, so scores are reproducible.
The test set (data/eval/test_v1.jsonl) is scored once, at the end. Never pick settings with it.

torch and transformers are imported only when a model is loaded; scoring runs anywhere.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from calfn.parse import format_call, parse_tool_call  # noqa: E402
from calfn.resolve import event_to_call  # noqa: E402
from calfn.scoring import make_result, summarize  # noqa: E402
from calfn.task import TASKS, Task, apply  # noqa: E402

BASE_MODEL = "unsloth/functiongemma-270m-it"  # ungated copy of google/functiongemma-270m-it


def load_examples(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("//"):
                rows.append(json.loads(line))
    return rows


def split_example(ex: dict):
    """(user text, expected event or None, tags, now). Eval rows hold the resolved event as the answer."""
    msgs = ex["messages"]
    user = next(m["content"] for m in msgs if m["role"] == "user")
    asst = next(m for m in msgs if m["role"] == "assistant")
    gold = asst["tool_calls"][0]["function"]["arguments"] if asst.get("tool_calls") else None
    meta = ex.get("meta", {})
    now = datetime.strptime(meta["now"], "%Y-%m-%dT%H:%M")
    return user, gold, meta.get("tags", []), now


def build_prompt(tokenizer, developer: str, user: str, tools: list[dict]) -> str:
    """The exact prompt shape used in training: developer message, user turn, tool declaration."""
    text = tokenizer.apply_chat_template(
        [{"role": "developer", "content": developer}, {"role": "user", "content": user}],
        tools=tools, tokenize=False, add_generation_prompt=True,
    )
    # The template already starts with <bos>; the tokenizer adds one too. Keep exactly one.
    return text.removeprefix("<bos>")


def load_model(path: str):
    """bf16 model; a LoRA adapter directory is merged into the base first."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path)
    if (Path(path) / "adapter_config.json").exists():
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(base, path).merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(device).eval(), tokenizer


def model_generator(model, tokenizer, task: Task, max_new_tokens: int = 160):
    import torch

    # Stop at end of turn, or where the model would wait for the tool's reply
    stop_ids = [tokenizer.eos_token_id] + tokenizer.convert_tokens_to_ids(
        ["<end_of_turn>", "<start_function_response>"])

    def generate(developer: str, user: str) -> tuple[str, float]:
        prompt = build_prompt(tokenizer, developer, user, task.tools)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                 eos_token_id=stop_ids, pad_token_id=tokenizer.eos_token_id)
        elapsed = time.perf_counter() - t0
        new = out[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new, skip_special_tokens=False).strip(), elapsed

    return generate


def oracle_generator(examples: list[dict], task: Task):
    """Answers with a correct output in the model's own format. Exercises parse + resolve + score."""
    gold = {split_example(e)[0]: split_example(e)[1] for e in examples}

    def generate(developer: str, user: str) -> tuple[str, float]:
        g = gold[user]
        if g is None:
            return "I need a date and time to create the event.", 0.0
        return format_call(event_to_call(g) if task.name == "extract" else g), 0.0

    return generate


def evaluate(examples: list[dict], task: Task, generate, limit: int | None = None, quiet: bool = False) -> list[dict]:
    results = []
    for i, ex in enumerate(examples[:limit]):
        user, gold, tags, now = split_example(ex)
        raw, seconds = generate(task.developer(now), user)
        pred = parse_tool_call(raw)
        event, error = apply(task, pred, now)
        results.append(make_result(i, user, raw, pred, gold, tags, seconds * 1000, event=event, error=error))
        if not quiet:
            n = sum(r["match"] in ("exact", "correct_rejection") for r in results)
            print(f"[{i + 1:>4}/{len(examples[:limit])}] {results[-1]['match']:18s} pass={n / (i + 1):.3f} | {user[:60]}")
    return results


def print_summary(summary: dict) -> None:
    o = summary["overall"]
    print("\n" + "=" * 60)
    print(f"Pass rate: {o['rate']:.1%}  ({o['pass']}/{o['n']})  95% CI {o['ci95'][0]:.1%}-{o['ci95'][1]:.1%}")
    print(f"Breakdown: {summary['breakdown']}")
    r = summary["rejection"]
    print(f"Rejections: {r['correct']}/{r['gold_rejections']} correct, {r['false_positives']} false positives; "
          f"{r['missed_calls']}/{r['gold_calls']} real requests missed")
    for field, v in summary["per_field"].items():
        extra = f"  (+{v['spurious']} spurious)" if v.get("spurious") else ""
        acc = f"{v['accuracy']:.1%}" if v["total"] else "  -  "
        print(f"  {field:22s} {v['correct']:>4}/{v['total']:<4} {acc}{extra}")
    if summary["mean_latency_ms"]:
        print(f"Mean latency: {summary['mean_latency_ms']:.0f} ms")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="model directory or hub id")
    ap.add_argument("--task", choices=sorted(TASKS), default="extract")
    ap.add_argument("--oracle", action="store_true", help="answer with correct outputs (harness check)")
    ap.add_argument("--eval", help="JSONL file to score")
    ap.add_argument("--prompt", help="a single request to run")
    ap.add_argument("--now", help="local time for --prompt, YYYY-MM-DDTHH:MM (default: now)")
    ap.add_argument("--name", help="<out-dir>/<name>.json (default: model directory name)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()
    if not (args.model or args.oracle):
        ap.error("give --model or --oracle")
    task = TASKS[args.task]

    examples = load_examples(args.eval) if args.eval else []
    if args.oracle:
        generate = oracle_generator(examples, task)
    else:
        print(f"Loading {args.model} ...")
        model, tokenizer = load_model(args.model)
        generate = model_generator(model, tokenizer, task)

    if args.prompt:
        now = datetime.strptime(args.now, "%Y-%m-%dT%H:%M") if args.now else datetime.now().replace(second=0, microsecond=0)
        raw, seconds = generate(task.developer(now), args.prompt)
        print(f"Raw output ({seconds * 1000:.0f} ms):\n  {raw}")
        call = parse_tool_call(raw)
        if call is None:
            print("No tool call.")
            return
        print("Call:", json.dumps(call, ensure_ascii=False))
        event, error = apply(task, call, now)
        print("Event:", json.dumps(event, ensure_ascii=False) if event else f"cannot resolve: {error}")
        return

    if not args.eval:
        ap.error("give --eval or --prompt")
    results = evaluate(examples, task, generate, args.limit)
    summary = summarize(results)
    print_summary(summary)

    name = args.name or ("oracle" if args.oracle else Path(args.model.rstrip("/\\")).name)
    out = Path(args.out_dir) / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "run_date": datetime.now().isoformat(timespec="seconds"),
        "model": "oracle" if args.oracle else args.model,
        "task": task.name,
        "eval_file": args.eval,
        "summary": summary,
        "results": results,
    }, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
