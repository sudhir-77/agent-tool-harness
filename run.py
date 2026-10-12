"""Reproduce the tool-harness evaluation.

Examples:
    python run.py
    python run.py --quick
    python run.py --device cpu
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HOME", str(ROOT / "data" / "hf_home"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(ROOT / "data" / "hf_home" / "hub"))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(ROOT / "data" / "st_home"))

sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the agent-tool-harness evaluation")
    parser.add_argument("--quick", action="store_true", help="Small test subset, skip the embedding model")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--skip-oracle", action="store_true")
    parser.add_argument("--skip-coverage", action="store_true")
    args = parser.parse_args()
    from evals.pipeline import run_pipeline

    payload = run_pipeline(
        ROOT,
        quick=args.quick,
        device=args.device,
        skip_oracle=args.skip_oracle,
        skip_coverage=args.skip_coverage or args.quick,
    )
    print(f"best_config={payload['metrics']['best_config']}")
    print(f"success_gap_points={payload['metrics']['success_gap_points']:.2f}")
    print(f"recall_at_5={payload['recall_at_5']:.4f}")
    print(f"token_reduction={payload['metrics']['token_reduction']:.4f}")
    print(f"schema_valid_rate={payload['schema_valid_rate']:.4f}")
    print(f"unconfirmed_side_effects={payload['unconfirmed_side_effects']}")
    print(f"n_test_tasks={payload['n_test_tasks']}")
    print(f"runtime_seconds={payload['runtime_seconds']}")
    for item in payload["success_criteria"]:
        status = "PASS" if item["passed"] else "FAIL"
        print(f"{status} {item['name']}: {item['value']}")


if __name__ == "__main__":
    main()
