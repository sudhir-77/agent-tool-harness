"""Metric aggregation. Numbers written to disk come from these functions."""

from __future__ import annotations

from typing import Any


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def compare(value: float, op: str, threshold: float, threshold_high: float | None = None) -> bool:
    if op == ">=":
        return value >= threshold
    if op == "<=":
        return value <= threshold
    if op == ">":
        return value > threshold
    if op == "<":
        return value < threshold
    if op == "between":
        if threshold_high is None:
            raise ValueError("between requires threshold_high")
        return threshold <= value <= threshold_high
    raise ValueError(f"unknown op {op}")


def criterion(
    name: str,
    value: float,
    op: str,
    threshold: float,
    threshold_high: float | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "value": value,
        "op": op,
        "threshold": threshold,
        "threshold_high": threshold_high,
        "passed": compare(value, op, threshold, threshold_high),
    }


def aggregate_episodes(rows: list[dict[str, Any]]) -> dict[str, Any]:
    configs = sorted({row["config"] for row in rows})
    success: dict[str, float] = {}
    tokens: dict[str, float] = {}
    wrong: dict[str, float] = {}
    latency: dict[str, float] = {}
    failures: dict[str, dict[str, int]] = {}
    for config in configs:
        subset = [row for row in rows if row["config"] == config]
        success[config] = mean([float(row["success"]) for row in subset])
        tokens[config] = mean([float(row["prompt_tokens"]) for row in subset])
        wrong[config] = mean([float(row["wrong_tool"]) for row in subset])
        latency[config] = mean([float(row["latency_ms"]) for row in subset])
        counts: dict[str, int] = {}
        for row in subset:
            kind = str(row["failure_type"])
            counts[kind] = counts.get(kind, 0) + 1
        failures[config] = counts
    n_steps = sum(int(row["n_steps"]) for row in rows)
    n_valid = sum(int(row["schema_valid_calls"]) for row in rows)
    schema_rate = (n_valid / n_steps) if n_steps else 1.0
    return {
        "success_by_config": success,
        "tokens_per_task_by_config": tokens,
        "wrong_tool_mean_by_config": wrong,
        "latency_ms_mean_by_config": latency,
        "failure_counts_by_config": failures,
        "schema_valid_rate": schema_rate,
        "n_episodes": len(rows),
    }


def select_best(success: dict[str, float], tokens: dict[str, float]) -> str:
    """Highest success, then fewer prompt tokens, then config name."""
    return min(success, key=lambda name: (-success[name], tokens[name], name))


def success_gap_points(success: dict[str, float], best: str, baseline: str = "all_tools") -> float:
    return (success[best] - success[baseline]) * 100.0


def token_reduction(tokens: dict[str, float], best: str, baseline: str = "all_tools") -> float:
    base = tokens[baseline]
    if base <= 0:
        return 0.0
    return 1.0 - (tokens[best] / base)


def recall_at_5(rows: list[dict[str, Any]], retriever: str, split: str = "test") -> float:
    subset = [row for row in rows if row["retriever"] == retriever and row["split"] == split]
    if not subset:
        return 0.0
    return mean([float(row["hit_at_5"]) for row in subset])


def retriever_for_config(config: str) -> str | None:
    if config.startswith("bm25"):
        return "bm25"
    if config.startswith("embed"):
        return "embed"
    if config.startswith("hierarchical"):
        return "hierarchical"
    return None
