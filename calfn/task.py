"""
The two ways to ask a model for an event, behind one interface.

  extract  (default) the model writes a CALL (date and time as said); resolve() makes the event.
  direct   the v1 approach: the model is told "Now: ..." and writes the event itself.

A Task builds the developer message and tool declaration for the prompt, and turns a model's
parsed arguments into an event (or raises CallError). Training, scoring and the baselines all go
through it, so they can't disagree about the prompt.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Callable

from . import schema
from .resolve import CallError, resolve


@dataclass(frozen=True)
class Task:
    name: str
    tool: dict
    developer: Callable[[datetime], str]
    to_event: Callable[[dict, datetime], dict]

    @property
    def tools(self) -> list[dict]:
        return [{"type": "function", "function": self.tool}]


def _direct_event(args: dict, now: datetime) -> dict:
    ok, msg = schema.validate_event(args)
    if not ok:
        raise CallError(msg)
    return args


def apply(task: "Task", pred: dict | None, now: datetime) -> tuple[dict | None, str | None]:
    """(event, error) for what a model wrote: (None, None) for no call, (None, why) if it can't be resolved."""
    if pred is None:
        return None, None
    try:
        return task.to_event(pred, now), None
    except CallError as e:
        return None, str(e)


EXTRACT = Task("extract", schema.TOOL, lambda now: schema.developer_message(), resolve)
DIRECT = Task("direct", schema.EVENT_TOOL, schema.direct_developer_message, _direct_event)
TASKS = {t.name: t for t in (EXTRACT, DIRECT)}
