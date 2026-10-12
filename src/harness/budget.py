"""Step and latency budgets for one episode."""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class StepBudget:
    max_steps: int = 16
    max_seconds: float = 30.0
    started: float = 0.0
    steps: int = 0

    def start(self) -> None:
        self.started = time.perf_counter()
        self.steps = 0

    def tick(self) -> str | None:
        """Return a stop reason, or None if the episode may continue."""
        self.steps += 1
        if self.steps > self.max_steps:
            return "budget_exceeded"
        if time.perf_counter() - self.started > self.max_seconds:
            return "budget_exceeded"
        return None
