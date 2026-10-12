"""Permission gate for side-effecting tools.

A side-effecting call executes only when the model sets confirm=True.
Blocked calls leave app state unchanged and are not counted as executions.
"""

from __future__ import annotations

from dataclasses import dataclass

from harness.registry import ToolCall, ToolSpec


@dataclass(frozen=True)
class GateDecision:
    allow: bool
    reason: str


class PermissionGate:
    def decide(self, tool: ToolSpec, call: ToolCall) -> GateDecision:
        if not tool.side_effect:
            return GateDecision(True, "no_side_effect")
        if call.confirm is True:
            return GateDecision(True, "confirmed")
        return GateDecision(False, "confirmation_required")
