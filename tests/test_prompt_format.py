"""
Training text and inference prompt must agree token for token up to the model's turn.
Needs transformers and the FunctionGemma tokenizer (downloads ~5 MB once); skipped without them.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

transformers = pytest.importorskip("transformers")
try:
    TOK = transformers.AutoTokenizer.from_pretrained("unsloth/functiongemma-270m-it")
except Exception as e:  # offline, gated, etc.
    pytest.skip(f"tokenizer unavailable: {e}", allow_module_level=True)

import finetune  # noqa: E402
import inference  # noqa: E402
from calfn.task import DIRECT, EXTRACT  # noqa: E402


def rows(path, n=200):
    return [json.loads(l) for _, l in zip(range(n), open(ROOT / path, encoding="utf-8"))]


def test_training_text_starts_with_inference_prompt():
    for ex in rows("data/train.jsonl"):
        user, _, _, now = inference.split_example({**ex, "messages": ex["messages"]})
        prompt = inference.build_prompt(TOK, EXTRACT.developer(now), user, EXTRACT.tools)
        assert finetune.format_example(ex, TOK).startswith(prompt)
        assert prompt.endswith("<start_of_turn>model\n")


def test_exactly_one_bos():
    ex = rows("data/train.jsonl", 1)[0]
    user, _, _, now = inference.split_example(ex)
    ids = TOK(inference.build_prompt(TOK, EXTRACT.developer(now), user, EXTRACT.tools))["input_ids"]
    assert ids[0] == TOK.bos_token_id and ids[1] != TOK.bos_token_id


def test_tool_call_ends_where_inference_stops():
    call_rows = [r for r in rows("data/train.jsonl") if r["messages"][2].get("tool_calls")]
    text = finetune.format_example(call_rows[0], TOK)
    assert text.rstrip().endswith("<end_function_call><start_function_response>")


def test_the_model_never_sees_the_date():
    ex = rows("data/train.jsonl", 1)[0]
    user, _, _, now = inference.split_example(ex)
    prompt = inference.build_prompt(TOK, EXTRACT.developer(now), user, EXTRACT.tools)
    assert "Now:" not in prompt and now.strftime("%Y-%m-%d") not in prompt
    direct = inference.build_prompt(TOK, DIRECT.developer(datetime(2026, 10, 14, 9, 0)), "x", DIRECT.tools)
    assert "Now: 2026-10-14 Wed 09:00" in direct       # the v1 prompt still can


def test_longest_training_example_fits_max_seq_len():
    longest = max(len(TOK(finetune.format_example(ex, TOK))["input_ids"]) for ex in rows("data/train.jsonl", 100000))
    print("longest training example (tokens):", longest)
    assert longest <= 1024     # the default --max_seq_len; a longer one would have its tool call truncated
