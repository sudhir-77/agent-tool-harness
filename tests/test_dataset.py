"""Dataset integrity, dev-split retrieval, and the context-window mechanism."""

from __future__ import annotations

import copy

import pytest

from apps.catalog import build_tools
from apps.state import empty_state, execute
from evals.checker import replay_gold
from evals.data import save_dataset, verify_task_split
from evals.generate import build_tasks, render_clause, split_tasks
from evals.runner import evaluate_tasks
from harness.context import CONTEXT_LIMIT, assemble_prompt, render_tools
from harness.llm import parse_args, phrase_in_text, split_clauses
from harness.permissions import PermissionGate
from harness.registry import ToolCall, ToolRegistry
from harness.retrieval import BM25Retriever, HierarchicalRetriever
from harness.schema import validate_call
from harness.tokens import count_tokens


@pytest.fixture(scope="module")
def tools():
    return build_tools()


@pytest.fixture(scope="module")
def tasks(tools):
    return build_tasks(tools, seed=0)


def test_catalog_size_and_phrase_uniqueness(tools) -> None:
    assert len(tools) >= 300
    assert len({tool.app for tool in tools}) == 12
    assert len({tool.name for tool in tools}) == len(tools)
    phrases = [tool.phrase for tool in tools]
    assert len(phrases) == len(set(phrases))
    for left in tools:
        for right in tools:
            if left.name != right.name:
                assert not phrase_in_text(left.phrase, right.phrase)


def test_split_is_disjoint_and_templates_do_not_leak(tasks) -> None:
    counts = verify_task_split(tasks, 336)
    assert counts["n_tasks"] == 400
    assert counts["n_dev"] == 200
    assert counts["n_test"] == 200
    dev, test = split_tasks(tasks)
    dev_templates = {task["template_id"] for task in dev}
    test_templates = {task["template_id"] for task in test}
    assert dev_templates.isdisjoint(test_templates)
    assert all(task["utterance"].startswith("Please ") for task in dev)
    assert all(task["utterance"].startswith("User request: ") for task in test)
    assert {task["task_id"] for task in dev}.isdisjoint({task["task_id"] for task in test})
    lengths = {task["n_steps"] for task in tasks}
    assert lengths == {1, 2, 3, 4, 5, 6}
    # Five instantiations of one template use different titles.
    group = [task for task in tasks if task["template_id"] == "tpl000"]
    titles = {task["plan"][0]["arguments"]["title"] for task in group}
    assert len(titles) == 5


def test_gold_plans_validate_parse_and_pass_the_checker(tools, tasks) -> None:
    registry = ToolRegistry(tools)
    by_name = registry.by_name
    apps = sorted({tool.app for tool in tools})
    for task in tasks:
        clauses = split_clauses(task["utterance"])
        assert len(clauses) == len(task["plan"])
        for clause, step in zip(clauses, task["plan"], strict=True):
            tool = by_name[step["tool"]]
            assert validate_call(
                ToolCall(name=tool.name, arguments=step["arguments"]),
                registry,
            )
            assert parse_args(clause, tool.parameters) == step["arguments"]
            if task["split"] == "dev":
                assert phrase_in_text(tool.phrase, clause)
            else:
                assert not phrase_in_text(tool.phrase, clause)
                assert tool.name not in clause
        assert replay_gold(task, registry, apps)


def test_permission_matrix_never_mutates_without_confirmation(tools) -> None:
    gate = PermissionGate()
    apps = sorted({tool.app for tool in tools})
    side_effects = [tool for tool in tools if tool.side_effect]
    assert side_effects
    for tool in side_effects:
        state = empty_state(apps)
        arguments = {field: ("Case 1" if tool.parameters["properties"][field]["type"] == "string" else 1) for field in tool.parameters["required"]}
        decision = gate.decide(tool, ToolCall(name=tool.name, arguments=arguments, confirm=False))
        assert decision.allow is False
        assert state[tool.app]["activity"] == []
        assert state[tool.app]["docs"] == []
        allowed = gate.decide(tool, ToolCall(name=tool.name, arguments=arguments, confirm=True))
        assert allowed.allow is True
        before_docs = len(state[tool.app]["docs"])
        execute(state, tool, arguments)
        assert state[tool.app]["activity"]
        if tool.action == "delete":
            assert len(state[tool.app]["docs"]) <= before_docs


def test_dev_retrieval_recall_and_success_gap(tools, tasks) -> None:
    dev, test = split_tasks(tasks)
    assert test  # held out; this test does not score it
    bm25 = BM25Retriever(tools)
    hierarchical = HierarchicalRetriever(tools)
    retrievers = {"bm25": bm25, "hierarchical": hierarchical}
    from evals.runner import score_retrieval

    rows = score_retrieval(dev, retrievers)
    from evals.metrics import recall_at_5

    assert recall_at_5(rows, "bm25", split="dev") >= 0.90
    assert recall_at_5(rows, "hierarchical", split="dev") >= 0.90
    # Known first-step ranking: the dev clause's gold tool is rank 1 under BM25.
    clause = split_clauses(dev[0]["utterance"])[0]
    assert bm25.retrieve(clause, k=1) == [dev[0]["plan"][0]["tool"]]

    configs = [
        {"name": "all_tools", "retriever": None, "compact": False},
        {"name": "bm25", "retriever": "bm25", "compact": False},
    ]
    episodes = evaluate_tasks(dev, tools, retrievers, configs, agent_kind="lexical_grounding")
    from evals.metrics import aggregate_episodes, success_gap_points
    from evals.report import episode_row

    summary = aggregate_episodes([episode_row(episode) for episode in episodes])
    gap = success_gap_points(summary["success_by_config"], "bm25")
    assert summary["success_by_config"]["bm25"] >= 0.90
    assert gap >= 20
    assert all(episode.unconfirmed_side_effects == 0 for episode in episodes)
    assert summary["schema_valid_rate"] >= 0.98


def test_context_window_cuts_tools_and_tokens(tools) -> None:
    utterance = "Please create the event in calendar title='Case 1' note='alpha detail'."
    rendered = render_tools(tools)
    full = assemble_prompt(utterance, rendered, [], compact=False, context_limit=CONTEXT_LIMIT)
    assert count_tokens(full.full_text) == full.full_tokens
    assert len(full.visible) < 80
    assert full.full_tokens > full.window_tokens
    small = assemble_prompt(utterance, rendered[:5], [], compact=False, context_limit=CONTEXT_LIMIT)
    assert len(small.visible) == 5
    assert small.full_tokens <= 0.4 * full.full_tokens
    history = [f"OK tool=calendar.create args={{}} step={index} extra={'word ' * 20}" for index in range(4)]
    raw = assemble_prompt(utterance, rendered[:5], history, compact=False)
    compact = assemble_prompt(utterance, rendered[:5], history, compact=True)
    assert compact.full_tokens < raw.full_tokens


def test_saved_split_is_stable(tmp_path, tools, tasks) -> None:
    first = save_dataset(tmp_path, tools, tasks)
    second = save_dataset(tmp_path, tools, tasks)
    assert first["tasks_sha256"] == second["tasks_sha256"]
    changed = copy.deepcopy(tasks)
    changed[0]["utterance"] = "Please changed."
    with pytest.raises(RuntimeError):
        save_dataset(tmp_path, tools, changed)


def test_render_clause_matches_parser(tools) -> None:
    tool = next(tool for tool in tools if tool.name == "calendar.schedule_meeting")
    arguments = {"title": "Case 42", "date": "2026-04-02", "time": "10:00", "duration_minutes": 45}
    clause = render_clause(tool.phrase, arguments, list(tool.parameters["required"]))
    assert parse_args(clause, tool.parameters) == arguments
