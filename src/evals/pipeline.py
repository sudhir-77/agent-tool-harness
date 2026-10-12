"""End-to-end run: generate data, evaluate ablations, write results."""

from __future__ import annotations

import gc
import json
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import psutil

from apps.catalog import APPS, build_tools
from evals.checker import replay_gold
from evals.data import ensure_minilm_config, save_dataset
from evals.generate import build_tasks, split_tasks
from evals.metrics import (
    aggregate_episodes,
    recall_at_5,
    select_best,
    success_gap_points,
    token_reduction,
)
from evals.report import (
    RETRIEVAL_FIELDS,
    build_success_criteria,
    episode_row,
    headline_recall,
    render_results_markdown,
    render_resume,
    resume_figures,
    save_plots,
    update_readme,
    write_csv,
)
from evals.runner import CONFIGS, QUICK_CONFIGS, evaluate_tasks, score_retrieval
from harness.registry import ToolRegistry
from harness.retrieval import BM25Retriever, EmbeddingRetriever, HierarchicalRetriever, SentenceTransformerEncoder


class PeakSampler:
    def __init__(self) -> None:
        self.proc = psutil.Process()
        self.peak = self.proc.memory_info().rss
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.wait(0.25):
            try:
                rss = self.proc.memory_info().rss
            except psutil.Error:
                return
            if rss > self.peak:
                self.peak = rss

    def __enter__(self) -> "PeakSampler":
        self._thread.start()
        return self

    def __exit__(self, *args: Any) -> bool:
        self._stop.set()
        self._thread.join(timeout=2)
        try:
            self.peak = max(self.peak, self.proc.memory_info().rss)
        except psutil.Error:
            pass
        return False

    def gb(self) -> float:
        return self.peak / (1024**3)


def _hardware(peak_gb: float, device: str) -> dict[str, Any]:
    gpu = None
    if device == "cuda":
        gpu = "cuda"
    return {
        "cpu": platform.processor() or platform.machine(),
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "gpu": gpu,
        "python": platform.python_version(),
        "os": platform.platform(),
        "peak_rss_gb": round(peak_gb, 4),
        "device": device,
    }


def measure_coverage(root: Path) -> float | None:
    env = os.environ.copy()
    env["HARNESS_SKIP_COVERAGE"] = "1"
    out = root / "results" / "coverage.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "pytest",
        "--cov=harness",
        "--cov=apps",
        "--cov=evals",
        f"--cov-report=json:{out}",
        "--cov-report=term",
        "-q",
    ]
    subprocess.run(command, cwd=root, env=env, check=False)
    if not out.exists():
        return None
    payload = json.loads(out.read_text(encoding="utf-8"))
    return float(payload["totals"]["percent_covered"]) / 100.0


def _build_retrievers(tools: list[Any], names: set[str], data_dir: Path, device: str) -> dict[str, Any]:
    retrievers: dict[str, Any] = {}
    if "bm25" in names:
        retrievers["bm25"] = BM25Retriever(tools)
    if "hierarchical" in names:
        retrievers["hierarchical"] = HierarchicalRetriever(tools)
    if "embed" in names:
        ensure_minilm_config(data_dir)
        os.environ.setdefault("HF_HOME", str(data_dir / "hf_home"))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(data_dir / "hf_home" / "hub"))
        os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(data_dir / "st_home"))
        encoder = SentenceTransformerEncoder(device=device)
        retrievers["embed"] = EmbeddingRetriever(tools, encoder)
    return retrievers


def run_pipeline(
    root: Path,
    *,
    quick: bool = False,
    device: str = "cpu",
    skip_oracle: bool = False,
    skip_coverage: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    if device == "cuda":
        import torch

        if not torch.cuda.is_available():
            device = "cpu"
    data_dir = root / "data"
    out_dir = root / "results" / "quick" if quick else root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    with PeakSampler() as sampler:
        tools = build_tools()
        tasks = build_tasks(tools, seed=0)
        manifest = save_dataset(data_dir, tools, tasks)
        registry = ToolRegistry(tools)
        apps = [name for name, _noun in APPS]
        for task in tasks:
            if not replay_gold(task, registry, apps):
                raise RuntimeError(f"gold plan does not satisfy its checker: {task['task_id']}")
        dev, test = split_tasks(tasks)
        eval_tasks = test[:8] if quick else list(test)
        configs = QUICK_CONFIGS if quick else CONFIGS
        needed = {config["retriever"] for config in configs if config["retriever"]}
        retrievers = _build_retrievers(tools, needed, data_dir, device)
        retrieval_rows = score_retrieval(eval_tasks, retrievers)
        dev_rows = score_retrieval(dev, retrievers)
        lexical = evaluate_tasks(
            eval_tasks, tools, retrievers, configs, agent_kind="lexical_grounding"
        )
        oracle = []
        if not skip_oracle:
            oracle = evaluate_tasks(
                eval_tasks, tools, retrievers, configs, agent_kind="scripted_oracle"
            )
        # Drop the embedding model before coverage so the next process is not stacked on it.
        retrievers.clear()
        gc.collect()

        coverage = None
        if not quick and not skip_coverage and os.environ.get("HARNESS_SKIP_COVERAGE") != "1":
            coverage = measure_coverage(root)

        lexical_rows = [episode_row(episode) for episode in lexical]
        oracle_rows = [episode_row(episode) for episode in oracle]
        summary = aggregate_episodes(lexical_rows)
        success = summary["success_by_config"]
        tokens = summary["tokens_per_task_by_config"]
        best = select_best(success, tokens)
        gap = success_gap_points(success, best)
        reduction = token_reduction(tokens, best)
        recall, recall_by = headline_recall(retrieval_rows, best, split="test")
        unconfirmed = int(sum(episode.unconfirmed_side_effects for episode in lexical))
        unconfirmed += int(sum(episode.unconfirmed_side_effects for episode in oracle))
        scored_rows = lexical_rows + oracle_rows
        n_scored_steps = sum(int(row["n_steps"]) for row in scored_rows)
        n_scored_valid = sum(int(row["schema_valid_calls"]) for row in scored_rows)
        schema_rate = (n_scored_valid / n_scored_steps) if n_scored_steps else 1.0
        criteria = build_success_criteria(
            gap_points=gap,
            recall=recall,
            reduction=reduction,
            unconfirmed=unconfirmed,
            schema_rate=schema_rate,
            coverage=coverage,
        )
        figures = resume_figures(
            success=success,
            tokens=tokens,
            best=best,
            recall=recall,
            n_tools=len(tools),
        )
        dev_recall = {name: recall_at_5(dev_rows, name, split="dev") for name in {row["retriever"] for row in dev_rows}}
        oracle_summary = aggregate_episodes(oracle_rows) if oracle_rows else {}
        payload: dict[str, Any] = {
            "project": "agent-tool-harness",
            "success_criteria": criteria,
            "metrics": {
                "agent": "lexical_grounding",
                "best_config": best,
                "success_by_config": success,
                "tokens_per_task_by_config": tokens,
                "wrong_tool_mean_by_config": summary["wrong_tool_mean_by_config"],
                "latency_ms_mean_by_config": summary["latency_ms_mean_by_config"],
                "failure_counts_by_config": summary["failure_counts_by_config"],
                "recall_at_5": recall,
                "recall_at_5_by_retriever": recall_by,
                "dev_recall_at_5_by_retriever": dev_recall,
                "token_reduction": reduction,
                "success_gap_points": gap,
                "unconfirmed_side_effects": unconfirmed,
                "schema_valid_rate": schema_rate,
                "n_test_tasks": len(eval_tasks),
                "n_tools": len(tools),
                "n_apps": len(apps),
                "context_limit_tokens": 2048,
                "retrieval_k": 5,
                "seeds": [0],
                "resume_numbers": figures,
                "oracle_upper_bound": {
                    "success_by_config": oracle_summary.get("success_by_config", {}),
                    "tokens_per_task_by_config": oracle_summary.get("tokens_per_task_by_config", {}),
                },
                "dataset": manifest,
            },
            "baseline": {
                "config": "all_tools",
                "agent": "lexical_grounding",
                "success": success.get("all_tools", 0.0),
                "tokens_per_task": tokens.get("all_tools", 0.0),
                "description": (
                    "Same lexical model with every tool schema in the prompt, "
                    "no retrieval and no context compaction."
                ),
            },
            "splits": {
                "tasks": {
                    "train": 0,
                    "val": int(manifest["n_dev"]),
                    "test": int(manifest["n_test"]),
                }
            },
            "seeds": [0],
            "hardware": _hardware(sampler.gb(), device),
            "runtime_seconds": 0.0,
            "success_by_config": success,
            "recall_at_5": recall,
            "tokens_per_task_by_config": tokens,
            "unconfirmed_side_effects": unconfirmed,
            "schema_valid_rate": schema_rate,
            "n_test_tasks": len(eval_tasks),
        }
        write_csv(out_dir / "episodes.csv", lexical_rows, [
            "task_id", "split", "config", "success", "n_steps", "wrong_tool",
            "schema_valid_calls", "prompt_tokens", "latency_ms", "failure_type",
        ])
        if oracle_rows:
            write_csv(out_dir / "oracle_episodes.csv", oracle_rows, [
                "task_id", "split", "config", "success", "n_steps", "wrong_tool",
                "schema_valid_calls", "prompt_tokens", "latency_ms", "failure_type",
            ])
        write_csv(out_dir / "retrieval.csv", retrieval_rows, RETRIEVAL_FIELDS)
        write_csv(out_dir / "dev_retrieval.csv", dev_rows, RETRIEVAL_FIELDS)
        gate_rows = [
            {"task_id": episode.task_id, "config": episode.config, "agent": "lexical_grounding", "unconfirmed_side_effects": episode.unconfirmed_side_effects}
            for episode in lexical
        ]
        gate_rows.extend(
            {"task_id": episode.task_id, "config": episode.config, "agent": "scripted_oracle", "unconfirmed_side_effects": episode.unconfirmed_side_effects}
            for episode in oracle
        )
        write_csv(
            out_dir / "gate_log.csv",
            gate_rows,
            ["task_id", "config", "agent", "unconfirmed_side_effects"],
        )
        examples = []
        for row in lexical_rows:
            if row["failure_type"] == "success":
                continue
            seen = sum(1 for item in examples if item["failure_type"] == row["failure_type"])
            if seen >= 5:
                continue
            examples.append(
                {
                    "task_id": row["task_id"],
                    "config": row["config"],
                    "failure_type": row["failure_type"],
                    "n_steps": row["n_steps"],
                    "wrong_tool": row["wrong_tool"],
                }
            )
        (out_dir / "failure_examples.jsonl").write_text(
            "".join(json.dumps(item) + "\n" for item in examples),
            encoding="utf-8",
        )
        (out_dir / "dev_diagnostics.json").write_text(
            json.dumps({"recall_at_5_by_retriever": dev_recall, "seed": 0}, indent=2),
            encoding="utf-8",
        )
        save_plots(
            out_dir,
            success,
            tokens,
            recall_by,
            summary["failure_counts_by_config"],
        )
        payload["runtime_seconds"] = round(time.perf_counter() - started, 3)
        payload["hardware"]["peak_rss_gb"] = round(sampler.gb(), 4)
        (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        table = render_results_markdown(
            success=success,
            tokens=tokens,
            wrong=summary["wrong_tool_mean_by_config"],
            recall_by=recall_by,
            best=best,
            criteria=criteria,
        )
        (out_dir / "ablation_table.md").write_text(table, encoding="utf-8")
        if not quick:
            update_readme(root / "README.md", table)
            (root / "RESUME_NOTES.md").write_text(render_resume(figures), encoding="utf-8")
            # Official copy of the table also lives at results/metrics.json.
            if out_dir != root / "results":
                pass
    return payload
