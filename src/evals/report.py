"""Write metrics.json, CSV artifacts, plots, README results, and resume notes."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from evals.metrics import (
    criterion,
    recall_at_5,
    retriever_for_config,
    select_best,
    success_gap_points,
    token_reduction,
)

EPISODE_FIELDS = [
    "task_id",
    "split",
    "config",
    "success",
    "n_steps",
    "wrong_tool",
    "schema_valid_calls",
    "prompt_tokens",
    "latency_ms",
    "failure_type",
]

RETRIEVAL_FIELDS = [
    "task_id",
    "split",
    "step_index",
    "retriever",
    "gold_tool",
    "rank",
    "hit_at_5",
]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})


def episode_row(episode: Any) -> dict[str, Any]:
    return {
        "task_id": episode.task_id,
        "split": episode.split,
        "config": episode.config,
        "success": episode.success,
        "n_steps": episode.n_steps,
        "wrong_tool": episode.wrong_tool,
        "schema_valid_calls": episode.schema_valid_calls,
        "prompt_tokens": episode.prompt_tokens,
        "latency_ms": round(float(episode.latency_ms), 3),
        "failure_type": episode.failure_type,
    }


def build_success_criteria(
    *,
    gap_points: float,
    recall: float,
    reduction: float,
    unconfirmed: int,
    schema_rate: float,
    coverage: float | None,
) -> list[dict[str, Any]]:
    criteria = [
        criterion(
            "Task success on held-out test templates (best configuration) minus all-tools baseline (points)",
            gap_points,
            ">=",
            20.0,
        ),
        criterion(
            "Tool retrieval Recall@5 for the gold next tool",
            recall,
            ">=",
            0.90,
        ),
        criterion(
            "Prompt token reduction vs all-tools baseline",
            reduction,
            ">=",
            0.60,
        ),
        criterion(
            "Side-effecting tool calls without required confirmation",
            float(unconfirmed),
            "<=",
            0.0,
        ),
        criterion(
            "Schema-valid tool calls",
            schema_rate,
            ">=",
            0.98,
        ),
    ]
    if coverage is not None:
        criteria.append(
            criterion("Test coverage", coverage, ">=", 0.80)
        )
    return criteria


def resume_figures(
    *,
    success: dict[str, float],
    tokens: dict[str, float],
    best: str,
    recall: float,
    n_tools: int,
    baseline: str = "all_tools",
) -> dict[str, Any]:
    return {
        "n_tools": n_tools,
        "best_config": best,
        "base_success_pct": round(success[baseline] * 100.0, 1),
        "best_success_pct": round(success[best] * 100.0, 1),
        "recall_at_5": round(recall, 3),
        "token_cut_pct": round(token_reduction(tokens, best, baseline) * 100.0, 1),
    }


def render_resume(figures: dict[str, Any]) -> str:
    base = figures["base_success_pct"]
    best = figures["best_success_pct"]
    recall = figures["recall_at_5"]
    cut = figures["token_cut_pct"]
    n_tools = figures["n_tools"]
    config = figures["best_config"]
    return f"""# Resume notes

## Bullets

- Built an agent harness over a simulated {n_tools}-tool app ecosystem with tool retrieval, context compaction, and permission gates, raising held-out multi-step task success from {base}% to {best}% ({config} versus the all-tools baseline, same lexical model).
- Reached {recall} Recall@5 for next-tool retrieval while cutting prompt tokens per task by {cut}% versus putting every tool schema in the prompt.
- Wrote state-based evals and a failure taxonomy (wrong tool, bad arguments, premature stop, loops) and labeled every held-out episode from the final app state.

## Skills and tools

Python, pydantic, JSON Schema, BM25 (rank-bm25), sentence-transformers, FAISS, Hugging Face transformers, pytest, matplotlib.

## Summary

This project is a local agent harness that chooses tools for multi-step tasks over a simulated 12-app ecosystem. It compares BM25, embedding, and hierarchical retrieval against an all-tools prompt, with and without context compaction, and checks success from app state.

## GitHub description

Agent harness that retrieves tools, compacts context, and evaluates multi-step tasks over a 300+ tool simulated app ecosystem.
"""


def render_results_markdown(
    *,
    success: dict[str, float],
    tokens: dict[str, float],
    wrong: dict[str, float],
    recall_by: dict[str, float],
    best: str,
    criteria: list[dict[str, Any]],
) -> str:
    lines = [
        "| Config | Task success | Prompt tokens / task | Mean wrong-tool calls |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name in sorted(success, key=lambda item: (-success[item], tokens[item], item)):
        mark = " (best)" if name == best else ""
        if name == "all_tools":
            mark = " (baseline)" if name != best else " (baseline, best)"
        lines.append(
            f"| {name}{mark} | {success[name] * 100:.1f}% | {tokens[name]:.1f} | {wrong[name]:.2f} |"
        )
    lines.append("")
    lines.append("| Retriever | Recall@5 on test |")
    lines.append("| --- | ---: |")
    for name in sorted(recall_by):
        lines.append(f"| {name} | {recall_by[name]:.3f} |")
    lines.append("")
    lines.append("| Criterion | Value | Threshold | Passed |")
    lines.append("| --- | ---: | ---: | --- |")
    for item in criteria:
        lines.append(
            f"| {item['name']} | {item['value']:.4f} | {item['op']} {item['threshold']} | {item['passed']} |"
        )
    lines.append("")
    return "\n".join(lines)


def update_readme(readme_path: Path, table: str) -> None:
    start = "<!-- RESULTS_START -->"
    end = "<!-- RESULTS_END -->"
    block = f"{start}\n{table}{end}"
    if not readme_path.exists():
        readme_path.write_text(block + "\n", encoding="utf-8")
        return
    text = readme_path.read_text(encoding="utf-8")
    if start in text and end in text:
        prefix = text.split(start)[0]
        suffix = text.split(end, 1)[1]
        readme_path.write_text(prefix + block + suffix, encoding="utf-8")
    else:
        readme_path.write_text(text.rstrip() + "\n\n" + block + "\n", encoding="utf-8")


def save_plots(
    out_dir: Path,
    success: dict[str, float],
    tokens: dict[str, float],
    recall_by: dict[str, float],
    failure_counts: dict[str, dict[str, int]],
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    names = sorted(success, key=lambda item: item)
    written: list[str] = []

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(names, [success[name] * 100 for name in names], color="#3b6ea5")
    ax.set_ylabel("Task success (%)")
    ax.set_title("Held-out task success by configuration")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    path = out_dir / "success_by_config.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    written.append(path.name)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(names, [tokens[name] for name in names], color="#5b8c5a")
    ax.set_ylabel("Mean prompt tokens per task")
    ax.set_title("Prompt tokens by configuration")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    path = out_dir / "tokens_by_config.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    written.append(path.name)

    fig, ax = plt.subplots(figsize=(6, 4))
    keys = sorted(recall_by)
    ax.bar(keys, [recall_by[key] for key in keys], color="#b36b2c")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.90, color="#333333", linestyle="--", linewidth=1)
    ax.set_ylabel("Recall@5")
    ax.set_title("Gold next-tool Recall@5")
    fig.tight_layout()
    path = out_dir / "recall_at_5.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    written.append(path.name)

    # Failure mix for the baseline and the best success config, if both exist.
    if failure_counts:
        focus = [name for name in ("all_tools", select_best(success, tokens)) if name in failure_counts]
        labels = sorted({kind for name in focus for kind in failure_counts[name]})
        fig, ax = plt.subplots(figsize=(8, 4.5))
        import numpy as np

        xpos = np.arange(len(labels))
        width = 0.35 if len(focus) > 1 else 0.5
        for offset, name in enumerate(focus):
            heights = [failure_counts[name].get(label, 0) for label in labels]
            ax.bar(xpos + (offset - (len(focus) - 1) / 2) * width, heights, width=width, label=name)
        ax.set_xticks(xpos, labels, rotation=30)
        ax.set_ylabel("Episodes")
        ax.set_title("Failure taxonomy")
        ax.legend()
        fig.tight_layout()
        path = out_dir / "failure_types.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(path.name)
    return written


def headline_recall(retrieval_rows: list[dict[str, Any]], best_config: str, split: str = "test") -> tuple[float, dict[str, float]]:
    retrievers = sorted({row["retriever"] for row in retrieval_rows if row["split"] == split})
    by = {name: recall_at_5(retrieval_rows, name, split=split) for name in retrievers}
    chosen = retriever_for_config(best_config)
    if chosen and chosen in by:
        return by[chosen], by
    if not by:
        return 0.0, by
    # Baseline has no retriever. Report the strongest pre-specified retriever.
    best_name = max(by, key=lambda name: (by[name], name))
    return by[best_name], by
