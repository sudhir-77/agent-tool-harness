"""Entrypoint smoke test. The quick path must finish in a few minutes."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_quick_entrypoint() -> None:
    env = os.environ.copy()
    env["HARNESS_SKIP_COVERAGE"] = "1"
    env["OMP_NUM_THREADS"] = "4"
    completed = subprocess.run(
        [sys.executable, "run.py", "--quick"],
        cwd=ROOT,
        env=env,
        timeout=170,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    metrics_path = ROOT / "results" / "quick" / "metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert payload["project"] == "agent-tool-harness"
    assert payload["seeds"] == [0]
    assert payload["splits"]["tasks"]["train"] == 0
    assert payload["n_test_tasks"] == 8
    assert payload["unconfirmed_side_effects"] == 0
    assert payload["schema_valid_rate"] >= 0.98
    assert "success_by_config" in payload
    assert "all_tools" in payload["success_by_config"]
    header = (ROOT / "results" / "quick" / "episodes.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header == "task_id,split,config,success,n_steps,wrong_tool,schema_valid_calls,prompt_tokens,latency_ms,failure_type"
    retrieval = (ROOT / "results" / "quick" / "retrieval.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "gold_tool" in retrieval and "hit_at_5" in retrieval
    assert (ROOT / "results" / "quick" / "success_by_config.png").exists()
