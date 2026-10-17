"""
Fine-tune FunctionGemma 270M on calfn data with Unsloth + TRL.

    # full fine-tune, bf16 (the default recipe; see docs/decisions.md D2)
    python scripts/finetune.py --data data/train.jsonl --output output/calfn-v1

    # LoRA instead
    python scripts/finetune.py --data data/train.jsonl --output output/calfn-lora64 --lora_rank 64 --lr 1e-4

Run it in the pinned training environment (requirements-train.txt). The versions matter:
newer unsloth/peft/torch diverged on the Gridfinity project with the same data.

Never pass data/eval/* as --val_data. Final epoch is kept.
"""

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from calfn.schema import TOOL  # noqa: E402

TOOLS = [{"type": "function", "function": TOOL}]


class LossLogger:
    """Collects train loss per logging step and eval loss per epoch (wrapped as a TrainerCallback below)."""

    def __init__(self):
        self.loss_log: list[dict] = []
        self.eval_loss: float | None = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.loss_log.append({"step": state.global_step, "loss": round(logs["loss"], 6),
                                  "learning_rate": logs.get("learning_rate")})
            print(f"  step {state.global_step:>5}/{state.max_steps}  loss={logs['loss']:.4f}")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "eval_loss" in metrics:
            self.eval_loss = round(metrics["eval_loss"], 6)
            print(f"  eval_loss={self.eval_loss:.4f}")


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip() and not line.lstrip().startswith("//")]


def format_example(example: dict, tokenizer) -> str:
    """
    Render one example with the model's own chat template, including the tool declaration,
    exactly as scripts/inference.py builds prompts. The leading <bos> is dropped because the
    tokenizer adds its own, and two would differ from what the model sees at inference.
    """
    text = tokenizer.apply_chat_template(example["messages"], tools=TOOLS, tokenize=False,
                                         add_generation_prompt=False)
    return text.removeprefix("<bos>")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="unsloth/functiongemma-270m-it")
    ap.add_argument("--data", default="data/train.jsonl")
    ap.add_argument("--val_data", default=None, help="a held-out split of the TRAIN data; never data/eval/*")
    ap.add_argument("--output", default=None)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--lora_rank", type=int, default=0, help="0 = full fine-tune (default)")
    ap.add_argument("--lora_alpha", type=int, default=None, help="default: 2 x rank")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_seq_len", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.val_data and "eval" in Path(args.val_data).parts:
        ap.error("do not use eval data for validation (it would leak into checkpoint choice)")
    full = args.lora_rank == 0
    args.output = args.output or f"output/calfn_e{args.epochs}_{'full' if full else f'r{args.lora_rank}'}_{datetime.now():%Y%m%d_%H%M%S}"

    from unsloth import FastLanguageModel
    import torch

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model, max_seq_length=args.max_seq_len, dtype=None,
        load_in_4bit=False, full_finetuning=full,
    )
    if not full:
        model = FastLanguageModel.get_peft_model(
            model, r=args.lora_rank,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=args.lora_alpha or 2 * args.lora_rank, lora_dropout=0.05, bias="none",
            use_gradient_checkpointing=True, random_state=args.seed,
        )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    from datasets import Dataset

    def make_dataset(path: str) -> Dataset:
        return Dataset.from_list([{"text": format_example(ex, tokenizer)} for ex in load_jsonl(path)])

    train_ds = make_dataset(args.data)
    val_ds = make_dataset(args.val_data) if args.val_data else None

    # A sequence longer than max_seq_len is truncated at the END, which cuts off the tool call
    # the model is supposed to learn. Fail loudly instead of training on broken examples.
    lengths = [len(tokenizer(t, add_special_tokens=True)["input_ids"]) for t in train_ds["text"]]
    print(f"Train examples: {len(train_ds)}   tokens per example: max {max(lengths)}, mean {sum(lengths) / len(lengths):.0f}")
    too_long = sum(n > args.max_seq_len for n in lengths)
    if too_long:
        sys.exit(f"{too_long} examples exceed --max_seq_len {args.max_seq_len}; raise it (longest is {max(lengths)}).")
    print("\n--- first training example (formatted) ---")
    print(train_ds[0]["text"])
    print("---\n")

    from trl import SFTTrainer
    from transformers import TrainerCallback, TrainingArguments

    class _Callback(LossLogger, TrainerCallback):
        pass

    logger = _Callback()
    total_steps = math.ceil(len(train_ds) / (args.batch_size * args.grad_accum)) * args.epochs
    warmup = max(1, round(total_steps * 0.05))
    bf16 = torch.cuda.is_bf16_supported()

    print(f"model {args.model} | {'full fine-tune' if full else f'LoRA r={args.lora_rank}'} | "
          f"trainable {trainable:,} | lr {args.lr} | epochs {args.epochs} | steps {total_steps} | out {args.output}\n")

    trainer = SFTTrainer(
        model=model, processing_class=tokenizer, train_dataset=train_ds, eval_dataset=val_ds,
        dataset_text_field="text", max_seq_length=args.max_seq_len, callbacks=[logger],
        args=TrainingArguments(
            output_dir=args.output, num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr, lr_scheduler_type="cosine", warmup_steps=warmup,
            fp16=not bf16, bf16=bf16, logging_steps=10,
            eval_strategy="epoch" if val_ds else "no", save_strategy="epoch",
            load_best_model_at_end=bool(val_ds), seed=args.seed, report_to="none",
        ),
    )
    result = trainer.train()

    Path(args.output).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"\nModel saved to {args.output}")

    metrics = getattr(result, "metrics", {})
    final_loss = logger.loss_log[-1]["loss"] if logger.loss_log else None
    Path("runs").mkdir(exist_ok=True)
    summary_path = Path("runs") / f"train_{datetime.now():%Y%m%d_%H%M%S}.json"
    summary_path.write_text(json.dumps({
        "run_date": datetime.now().isoformat(timespec="seconds"),
        "model": args.model, "full_finetune": full, "trainable_params": trainable,
        "output": args.output, "data": args.data, "val_data": args.val_data,
        "hyperparams": {k: getattr(args, k) for k in
                        ("epochs", "lr", "lora_rank", "lora_alpha", "batch_size", "grad_accum", "max_seq_len", "seed")},
        "n_train": len(train_ds), "max_tokens": max(lengths),
        "train_runtime_s": round(metrics.get("train_runtime", 0), 1),
        "final_train_loss": final_loss, "final_eval_loss": logger.eval_loss,
        "loss_log": logger.loss_log,
    }, indent=1), encoding="utf-8")
    print(f"Training summary: {summary_path}   (final train loss {final_loss})")


if __name__ == "__main__":
    main()
