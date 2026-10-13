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
from evals.data import download_verified_stream, ensure_minilm_config, save_dataset
from evals.generate import build_tasks, paraphrase_task, split_tasks
from evals.metrics import (
    aggregate_episodes,
    recall_at_5,
    select_best,
    success_gap_points,
    token_reduction,
)
from evals.report import (
    EPISODE_FIELDS,
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
from harness.generative import (
    CONTEXT_TOKENS,
    GGUF_FILENAME,
    GGUF_SHA256,
    GGUF_URL,
    LLM_MODEL_ID,
    REPORTED_AGENT,
    load_llama_backend,
)
from harness.registry import ToolRegistry
from harness.retrieval import (
    BM25Retriever,
    EmbeddingRetriever,
    FrozenRetriever,
    HierarchicalRetriever,
    SentenceTransformerEncoder,
)


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


def choose_retriever(scores: dict[str, float]) -> str:
    """Highest dev paraphrase recall. Ties prefer a retriever that is not the encoder."""

    def rank(name: str) -> tuple[float, int, str]:
        return (scores[name], 0 if name == "embed" else 1, name)

    return max(scores, key=rank)


def llm_configs(retriever: str) -> list[dict[str, Any]]:
    return [
        {"name": "all_tools", "retriever": None, "compact": False},
        {"name": "random5", "retriever": "random5", "compact": False},
        {"name": retriever, "retriever": retriever, "compact": False},
        {"name": f"{retriever}_compact", "retriever": retriever, "compact": True},
    ]


def _mention_rate(rows: list[dict[str, Any]], split: str) -> float:
    subset = [row for row in rows if row["split"] == split]
    if not subset:
        return 1.0
    quiet = sum(1 for row in subset if int(row["instruction_mentions_tool"]) == 0)
    return quiet / len(subset)


def _freeze_retriever(retriever: Any, tasks: list[dict[str, Any]], k: int = 5) -> FrozenRetriever:
    from harness.llm import split_clauses

    mapping: dict[str, list[str]] = {}
    for task in tasks:
        for clause in split_clauses(task["utterance"]):
            if clause not in mapping:
                mapping[clause] = list(retriever.retrieve(clause, k=k))
    return FrozenRetriever(str(retriever.name), mapping)


def _keep_retriever(retrievers: dict[str, Any], chosen: str, tasks: list[dict[str, Any]]) -> Any | None:
    retriever = retrievers.get(chosen)
    if retriever is None:
        return None
    if chosen == "embed":
        return _freeze_retriever(retriever, tasks)
    return retriever


def _release_embed(retrievers: dict[str, Any]) -> None:
    embed = retrievers.get("embed")
    if embed is None:
        return
    encoder = getattr(embed, "encoder", None)
    if encoder is None:
        return
    cache = getattr(encoder, "_cache", None)
    if isinstance(cache, dict):
        cache.clear()
    if hasattr(encoder, "model"):
        encoder.model = None


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
        by_name = {tool.name: tool for tool in tools}
        retrieval_rows = score_retrieval(eval_tasks, retrievers, tools)
        dev_rows = score_retrieval(dev, retrievers, tools)
        dev_para = [paraphrase_task(task, by_name) for task in dev]
        dev_para_rows = score_retrieval(dev_para, retrievers, tools)
        test_quiet = _mention_rate(retrieval_rows, "test")
        if retrieval_rows and test_quiet < 0.80:
            raise RuntimeError(f"test instructions mention the tool on too many rows: quiet_rate={test_quiet:.3f}")
        lexical = evaluate_tasks(
            eval_tasks, tools, retrievers, configs, agent_kind="lexical_grounding"
        )
        oracle = []
        if not skip_oracle:
            oracle = evaluate_tasks(
                eval_tasks, tools, retrievers, configs, agent_kind="scripted_oracle"
            )
        reported_kind = "lexical_grounding"
        reported = lexical
        llm_device = device
        llm_model = None
        context_limit = 2048
        para_recall = {
            name: recall_at_5(dev_para_rows, name, split="dev")
            for name in {row["retriever"] for row in dev_para_rows}
        }
        if not quick:
            chosen = choose_retriever(para_recall)
            # Freeze embeddings, then drop the encoder before the GGUF loads.
            kept = _keep_retriever(retrievers, chosen, eval_tasks)
            _release_embed(retrievers)
            retrievers.clear()
            gc.collect()
            if kept is not None:
                retrievers[chosen] = kept
            model_path = download_verified_stream(
                GGUF_URL,
                data_dir / "models" / GGUF_FILENAME,
                GGUF_SHA256,
            )
            backend = load_llama_backend(model_path, data_dir)
            llm_device = backend.device
            llm_model = LLM_MODEL_ID
            context_limit = CONTEXT_TOKENS
            print(
                f"llm_model={llm_model} device={llm_device} retriever={chosen} "
                f"dev_paraphrase_recall={para_recall}",
                flush=True,
            )
            try:
                reported = evaluate_tasks(
                    eval_tasks,
                    tools,
                    retrievers,
                    llm_configs(chosen),
                    agent_kind="generative",
                    generative_backend=backend,
                    catalog_order=True,
                    schema_retry_limit=1,
                    context_limit=1_000_000,
                )
            finally:
                backend.close()
                del backend
                gc.collect()
            reported_kind = REPORTED_AGENT
        # Drop retrievers before coverage so the next process is not stacked on them.
        retrievers.clear()
        gc.collect()

        coverage = None
        if not quick and not skip_coverage and os.environ.get("HARNESS_SKIP_COVERAGE") != "1":
            coverage = measure_coverage(root)

        lexical_rows = [episode_row(episode) for episode in lexical]
        reported_rows = [episode_row(episode) for episode in reported]
        oracle_rows = [episode_row(episode) for episode in oracle]
        summary = aggregate_episodes(reported_rows)
        success = summary["success_by_config"]
        tokens = summary["tokens_per_task_by_config"]
        best = select_best(success, tokens)
        gap = success_gap_points(success, best)
        reduction = token_reduction(tokens, best)
        recall, recall_by = headline_recall(retrieval_rows, best, split="test")
        unconfirmed = int(sum(episode.unconfirmed_side_effects for episode in reported))
        unconfirmed += int(sum(episode.unconfirmed_side_effects for episode in oracle))
        n_scored_steps = sum(int(row["n_steps"]) for row in reported_rows)
        n_scored_valid = sum(int(row["schema_valid_calls"]) for row in reported_rows)
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
            agent=reported_kind,
        )
        dev_recall = {name: recall_at_5(dev_rows, name, split="dev") for name in {row["retriever"] for row in dev_rows}}
        oracle_summary = aggregate_episodes(oracle_rows) if oracle_rows else {}
        payload: dict[str, Any] = {
            "project": "agent-tool-harness",
            "success_criteria": criteria,
            "metrics": {
                "agent": reported_kind,
                "llm_model": llm_model,
                "llm_device": llm_device if llm_model else None,
                "best_config": best,
                "success_by_config": success,
                "tokens_per_task_by_config": tokens,
                "wrong_tool_mean_by_config": summary["wrong_tool_mean_by_config"],
                "latency_ms_mean_by_config": summary["latency_ms_mean_by_config"],
                "failure_counts_by_config": summary["failure_counts_by_config"],
                "recall_at_5": recall,
                "recall_at_5_by_retriever": recall_by,
                "dev_recall_at_5_by_retriever": dev_recall,
                "dev_paraphrase_recall_at_5_by_retriever": para_recall,
                "test_instruction_quiet_rate": test_quiet,
                "token_reduction": reduction,
                "success_gap_points": gap,
                "unconfirmed_side_effects": unconfirmed,
                "schema_valid_rate": schema_rate,
                "n_test_tasks": len(eval_tasks),
                "n_tools": len(tools),
                "n_apps": len(apps),
                "context_limit_tokens": context_limit,
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
                "agent": reported_kind,
                "success": success.get("all_tools", 0.0),
                "tokens_per_task": tokens.get("all_tools", 0.0),
                "description": (
                    "Same agent with tool schemas in catalog order, truncated to the context budget, "
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
            "hardware": _hardware(sampler.gb(), llm_device if llm_model else device),
            "runtime_seconds": 0.0,
            "success_by_config": success,
            "recall_at_5": recall,
            "tokens_per_task_by_config": tokens,
            "unconfirmed_side_effects": unconfirmed,
            "schema_valid_rate": schema_rate,
            "n_test_tasks": len(eval_tasks),
        }
        write_csv(out_dir / "episodes.csv", reported_rows, EPISODE_FIELDS)
        if not quick and reported is not lexical:
            write_csv(out_dir / "lexical_episodes.csv", lexical_rows, EPISODE_FIELDS)
        if oracle_rows:
            write_csv(out_dir / "oracle_episodes.csv", oracle_rows, EPISODE_FIELDS)
        write_csv(out_dir / "retrieval.csv", retrieval_rows, RETRIEVAL_FIELDS)
        write_csv(out_dir / "dev_retrieval.csv", dev_rows, RETRIEVAL_FIELDS)
        gate_rows = [
            {
                "task_id": episode.task_id,
                "config": episode.config,
                "agent": episode.agent,
                "unconfirmed_side_effects": episode.unconfirmed_side_effects,
            }
            for episode in reported
        ]
        if reported is not lexical:
            gate_rows.extend(
                {
                    "task_id": episode.task_id,
                    "config": episode.config,
                    "agent": episode.agent,
                    "unconfirmed_side_effects": episode.unconfirmed_side_effects,
                }
                for episode in lexical
            )
        gate_rows.extend(
            {
                "task_id": episode.task_id,
                "config": episode.config,
                "agent": episode.agent,
                "unconfirmed_side_effects": episode.unconfirmed_side_effects,
            }
            for episode in oracle
        )
        write_csv(
            out_dir / "gate_log.csv",
            gate_rows,
            ["task_id", "config", "agent", "unconfirmed_side_effects"],
        )
        examples = []
        for row in reported_rows:
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
            json.dumps(
                {
                    "recall_at_5_by_retriever": dev_recall,
                    "paraphrase_recall_at_5_by_retriever": para_recall,
                    "seed": 0,
                },
                indent=2,
            ),
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
