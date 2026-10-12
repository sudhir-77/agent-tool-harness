"""Failure labels for multi-step episodes.

The loop assigns exactly one label per episode. `success` means the final
app state satisfied every expectation. The other labels describe the first
decisive way the episode went wrong.
"""

from __future__ import annotations

from harness.loop import classify_failure

FAILURE_TYPES = (
    "success",
    "wrong_tool",
    "bad_arguments",
    "premature_stop",
    "loop",
    "schema_invalid",
    "budget_exceeded",
)

__all__ = ["FAILURE_TYPES", "classify_failure"]
