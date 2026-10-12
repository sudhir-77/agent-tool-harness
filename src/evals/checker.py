"""Replay a gold plan through the gate and the state checker."""

from __future__ import annotations

from typing import Any

from apps.state import empty_state, execute, expectations_met
from harness.permissions import PermissionGate
from harness.registry import ToolCall, ToolRegistry, ToolSpec


def replay_gold(task: dict[str, Any], registry: ToolRegistry, apps: list[str]) -> bool:
    """Execute the labeled plan, confirming side effects, and check final state."""
    state = empty_state(apps)
    gate = PermissionGate()
    for step in task["plan"]:
        tool = registry.get(step["tool"])
        if tool is None:
            return False
        call = ToolCall(name=tool.name, arguments=dict(step["arguments"]), confirm=False)
        decision = gate.decide(tool, call)
        if not decision.allow:
            confirmed = ToolCall(name=tool.name, arguments=dict(step["arguments"]), confirm=True)
            if not gate.decide(tool, confirmed).allow:
                return False
            execute(state, tool, confirmed.arguments)
        else:
            execute(state, tool, call.arguments)
    return expectations_met(state, task["expectations"])


def tool_is_side_effect(tool: ToolSpec) -> bool:
    return bool(tool.side_effect)
