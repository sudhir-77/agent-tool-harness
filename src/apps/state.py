"""In-memory app state. Success checks read this state, not model text."""

from __future__ import annotations

import copy
from typing import Any

from harness.registry import ToolSpec

DELETE_ACTIONS = frozenset({"delete", "cancel_meeting", "reset_preference"})


def empty_state(apps: list[str]) -> dict[str, Any]:
    return {app: {"docs": [], "activity": []} for app in apps}


def execute(state: dict[str, Any], tool: ToolSpec, arguments: dict[str, Any]) -> dict[str, Any]:
    """Apply a call that has already passed the permission gate."""
    bucket = state[tool.app]
    payload = dict(arguments)
    if tool.action in DELETE_ACTIONS:
        bucket["docs"] = [doc for doc in bucket["docs"] if doc.get("title") != payload.get("title")]
    else:
        bucket["docs"].append({"kind": tool.action, **payload})
    bucket["activity"].append({"tool": tool.name, "action": tool.action, **payload})
    return {"ok": True, "tool": tool.name, "action": tool.action}


def expectations_met(state: dict[str, Any], expectations: list[dict[str, Any]]) -> bool:
    for expectation in expectations:
        activity = state[expectation["app"]]["activity"]
        fields = expectation["fields"]
        if not any(_fields_match(row, fields) for row in activity):
            return False
    return True


def _fields_match(row: dict[str, Any], fields: dict[str, Any]) -> bool:
    return all(row.get(key) == value for key, value in fields.items())


def clone_state(state: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(state)
