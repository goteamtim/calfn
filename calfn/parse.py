"""
Parse FunctionGemma's native tool-call output into an arguments dict, and format one back.

Adapted from function-gemma-fine-tune/scripts/inference.py (the parser was exercised on
~7,000 model outputs there). The model emits:

    <start_function_call>call:create_event{duration_minutes:90,start:<escape>2026-10-15T15:00<escape>,title:<escape>Dentist<escape>}<end_function_call>

Strings are wrapped in <escape>...<escape>; numbers and booleans are bare.
"""

import json
import re

from .schema import CALL_PROPS, EVENT_PROPS

_TYPES = {name: p["type"] for name, p in {**EVENT_PROPS, **CALL_PROPS}.items()}
_ESC = "<escape>"


def parse_fg_body(body: str) -> dict:
    """Parse the k:v body between the braces of a FunctionGemma call."""
    result = {}
    i = 0
    while i < len(body):
        while i < len(body) and body[i] in " ,\n\r\t":
            i += 1
        if i >= len(body):
            break
        colon = body.find(":", i)
        if colon == -1:
            break
        key = body[i:colon].strip()
        i = colon + 1

        if body.startswith(_ESC, i):
            end = body.find(_ESC, i + len(_ESC))
            if end == -1:
                result[key] = body[i + len(_ESC):]
                break
            result[key] = body[i + len(_ESC):end]
            i = end + len(_ESC)
        elif body.startswith("true", i):
            result[key] = True
            i += 4
        elif body.startswith("false", i):
            result[key] = False
            i += 5
        else:
            end = i
            while end < len(body) and body[end] not in ",}":
                end += 1
            raw = body[i:end].strip()
            try:
                result[key] = int(raw)
            except ValueError:
                try:
                    result[key] = float(raw)
                except ValueError:
                    result[key] = raw
            i = end
    return result


def coerce_types(args: dict) -> dict:
    """
    Bring parsed values to the schema's types without being generous: a float that isn't a
    whole number stays a float (so validation fails) instead of being truncated, and a
    string where a number belongs is only converted if it is exactly an integer.
    """
    out = {}
    for k, v in args.items():
        t = _TYPES.get(k)
        if t == "integer":
            if isinstance(v, float) and v.is_integer():
                v = int(v)
            elif isinstance(v, str) and re.fullmatch(r"-?\d+", v.strip()):
                v = int(v.strip())
        elif t == "string" and isinstance(v, (int, float)) and not isinstance(v, bool):
            v = str(v)
        out[k] = v
    return out


_CALL_RE = re.compile(r"<start_function_call>call:\w+\{(.*?)\}<end_function_call>", re.DOTALL)


def parse_tool_call(raw_output: str) -> dict | None:
    """
    Extract the call's arguments from raw model output; None if the model made no call.
    Falls back to a <tool_call>{json}</tool_call> block, which Claude-style or other models emit.
    """
    m = _CALL_RE.search(raw_output)
    if m:
        parsed = parse_fg_body(m.group(1))
        if parsed:
            return coerce_types(parsed)

    m = re.search(r"<tool_call>(.*?)</tool_call>", raw_output, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group(1).strip())
            return coerce_types(parsed.get("arguments") or parsed)
        except (json.JSONDecodeError, AttributeError):
            pass
    return None


def format_call(args: dict, name: str = "create_event") -> str:
    """Render arguments in the model's native format (used by tests and the oracle model)."""
    parts = []
    for k in sorted(args):
        v = args[k]
        if isinstance(v, bool):
            parts.append(f"{k}:{'true' if v else 'false'}")
        elif isinstance(v, (int, float)):
            parts.append(f"{k}:{v}")
        else:
            parts.append(f"{k}:{_ESC}{v}{_ESC}")
    return f"<start_function_call>call:{name}{{{','.join(parts)}}}<end_function_call>"
