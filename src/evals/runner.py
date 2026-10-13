"""Run one agent over a task list for each ablation configuration."""

from __future__ import annotations

import random
from typing import Any

from apps.catalog import APPS
from apps.state import empty_state, execute, expectations_met
from harness.context import CONTEXT_LIMIT, RETRIEVAL_K, render_tools
from harness.generative import GenerativeToolModel
from harness.llm import LexicalGroundingModel, PlannedCall, ScriptedOracle, phrase_in_text, split_clauses
from harness.loop import EpisodeResult, run_episode
from harness.permissions import PermissionGate
from harness.registry import ToolRegistry, ToolSpec

# Pre-specified ablation. all_tools is the baseline: every schema in the prompt,
# no retrieval, no compaction.
CONFIGS: list[dict[str, Any]] = [
    {"name": "all_tools", "retriever": None, "compact": False},
    {"name": "all_tools_compact", "retriever": None, "compact": True},
    {"name": "bm25", "retriever": "bm25", "compact": False},
    {"name": "bm25_compact", "retriever": "bm25", "compact": True},
    {"name": "embed", "retriever": "embed", "compact": False},
    {"name": "embed_compact", "retriever": "embed", "compact": True},
    {"name": "hierarchical", "retriever": "hierarchical", "compact": False},
    {"name": "hierarchical_compact", "retriever": "hierarchical", "compact": True},
]

QUICK_CONFIGS: list[dict[str, Any]] = [
    {"name": "all_tools", "retriever": None, "compact": False},
    {"name": "bm25", "retriever": "bm25", "compact": False},
    {"name": "bm25_compact", "retriever": "bm25", "compact": True},
    {"name": "hierarchical", "retriever": "hierarchical", "compact": False},
    {"name": "hierarchical_compact", "retriever": "hierarchical", "compact": True},
]


def baseline_order(tools: list[ToolSpec], seed: int = 0, catalog: bool = False) -> list[ToolSpec]:
    ordered = list(tools)
    if not catalog:
        random.Random(seed).shuffle(ordered)
    return ordered


def instruction_mentions_tool(text: str, tool: ToolSpec) -> int:
    if tool.name in text or phrase_in_text(tool.phrase, text):
        return 1
    return 0


def score_retrieval(
    tasks: list[dict[str, Any]],
    retrievers: dict[str, Any],
    tools: list[ToolSpec] | None = None,
    k: int = RETRIEVAL_K,
) -> list[dict[str, Any]]:
    """Gold-next-tool recall. Labels are used only to score a finished ranking."""
    by_name = {tool.name: tool for tool in tools} if tools else {}
    rows: list[dict[str, Any]] = []
    for task in tasks:
        clauses = split_clauses(task["utterance"])
        for index, step in enumerate(task["plan"]):
            query = clauses[index]
            gold = step["tool"]
            for name, retriever in retrievers.items():
                ranked = list(retriever.retrieve(query, k=k))
                hit = int(gold in ranked)
                rank = ranked.index(gold) + 1 if hit else -1
                tool = by_name.get(gold)
                mentions = instruction_mentions_tool(query, tool) if tool is not None else 0
                rows.append(
                    {
                        "task_id": task["task_id"],
                        "split": task["split"],
                        "step_index": index,
                        "retriever": name,
                        "gold_tool": gold,
                        "rank": rank,
                        "hit_at_5": hit,
                        "instruction_mentions_tool": mentions,
                    }
                )
    return rows


def evaluate_tasks(
    tasks: list[dict[str, Any]],
    tools: list[ToolSpec],
    retrievers: dict[str, Any],
    configs: list[dict[str, Any]],
    *,
    agent_kind: str,
    context_limit: int = CONTEXT_LIMIT,
    seed: int = 0,
    generative_backend: Any | None = None,
    catalog_order: bool = False,
    schema_retry_limit: int | None = None,
) -> list[EpisodeResult]:
    registry = ToolRegistry(tools)
    apps = [name for name, _noun in APPS]
    rendered = {item.tool.name: item for item in render_tools(tools)}
    baseline = [rendered[tool.name] for tool in baseline_order(tools, seed=seed, catalog=catalog_order)]
    episodes: list[EpisodeResult] = []
    for config in configs:
        retriever_name = config["retriever"]
        if retriever_name not in (None, "random5") and retriever_name not in retrievers:
            raise KeyError(f"missing retriever {retriever_name}")
        if agent_kind == "generative":
            print(f"generative start config={config['name']} tasks={len(tasks)}", flush=True)
        for task in tasks:
            if agent_kind == "lexical_grounding":
                agent: Any = LexicalGroundingModel()
            elif agent_kind == "scripted_oracle":
                agent = ScriptedOracle(
                    [PlannedCall(step["tool"], dict(step["arguments"])) for step in task["plan"]]
                )
            elif agent_kind == "generative":
                if generative_backend is None:
                    raise ValueError("generative agent requires a backend")
                agent = GenerativeToolModel(generative_backend)
            else:
                raise ValueError(agent_kind)

            def select_rendered(clause: str, _config: dict[str, Any] = config, _task: dict[str, Any] = task) -> list[Any]:
                if _config["retriever"] is None:
                    return baseline
                if _config["retriever"] == "random5":
                    local = random.Random(f"{seed}:{_task['task_id']}")
                    names = local.sample([item.name for item in tools], k=min(5, len(tools)))
                    return [rendered[name] for name in names]
                names = retrievers[_config["retriever"]].retrieve(clause, k=RETRIEVAL_K)
                return [rendered[name] for name in names]

            episodes.append(
                run_episode(
                    task_id=task["task_id"],
                    split=task["split"],
                    config=config["name"],
                    utterance=task["utterance"],
                    gold=[(step["tool"], dict(step["arguments"])) for step in task["plan"]],
                    expectations=task["expectations"],
                    state=empty_state(apps),
                    registry=registry,
                    select_rendered=select_rendered,
                    agent=agent,
                    gate=PermissionGate(),
                    execute_fn=execute,
                    check_fn=expectations_met,
                    compact=bool(config["compact"]),
                    context_limit=context_limit,
                    schema_retry_limit=schema_retry_limit,
                )
            )
            if agent_kind == "generative" and len(episodes) % 25 == 0:
                done = sum(1 for episode in episodes if episode.config == config["name"])
                print(f"generative config={config['name']} done={done}/{len(tasks)}", flush=True)
    return episodes
