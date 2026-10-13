"""Branch coverage for error paths, the quick pipeline, and the encoder wrapper."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apps.catalog import build_tools
from apps.state import empty_state, execute, expectations_met
from evals.checker import replay_gold
from evals.data import download_verified, verify_task_split
from evals.failure import FAILURE_TYPES, classify_failure
from evals.generate import build_tasks
from evals.metrics import compare, mean, recall_at_5, retriever_for_config, token_reduction
from evals.pipeline import _keep_retriever, _release_embed, run_pipeline
from evals.report import headline_recall
from harness.context import render_tools
from harness.llm import LexicalGroundingModel, ToolCall, to_openai_response
from harness.loop import run_episode
from harness.permissions import GateDecision, PermissionGate
from harness.registry import ToolRegistry, ToolSpec, schema_for_fields
from harness.retrieval import EmbeddingRetriever, SentenceTransformerEncoder
from harness.schema import validate_call
from harness.tokens import tokenize_text, truncate_to_tokens

ROOT = Path(__file__).resolve().parents[1]


def test_metric_edge_cases() -> None:
    assert mean([]) == 0.0
    assert compare(1, ">", 0)
    assert compare(0, "<", 1)
    with pytest.raises(ValueError):
        compare(0, "between", 0, None)
    with pytest.raises(ValueError):
        compare(0, "nope", 0)
    assert token_reduction({"all_tools": 0.0, "bm25": 1.0}, "bm25") == 0.0
    assert recall_at_5([], "bm25") == 0.0
    assert retriever_for_config("embed_compact") == "embed"
    assert retriever_for_config("hierarchical") == "hierarchical"
    assert retriever_for_config("all_tools") is None
    assert "wrong_tool" in FAILURE_TYPES
    assert classify_failure(
        success=False, schema_invalid=False, looped=False, wrong_tool=0,
        completed_steps=0, gold_steps=1, hit_budget=False,
    ) == "premature_stop"
    value, by = headline_recall(
        [{"retriever": "bm25", "split": "test", "hit_at_5": 1}],
        "all_tools",
    )
    assert value == 1.0 and by["bm25"] == 1.0
    assert headline_recall([], "bm25") == (0.0, {})


def test_registry_schema_and_replay_misses() -> None:
    tool = ToolSpec(
        name="calendar.create",
        app="calendar",
        action="create",
        phrase="create the event in calendar",
        description="create the event in calendar",
        side_effect=False,
        parameters=schema_for_fields(["title"]),
        retrieval_text="calendar create",
    )
    with pytest.raises(ValueError):
        ToolRegistry([tool, tool])
    registry = ToolRegistry([tool])
    assert registry.get("missing") is None
    assert len(registry) == 1
    broken = ToolSpec(
        name="calendar.bad",
        app="calendar",
        action="bad",
        phrase="bad phrase unique",
        description="bad",
        side_effect=False,
        parameters={"type": "object", "required": "title"},
        retrieval_text="bad",
    )
    assert validate_call(ToolCall(name="calendar.bad", arguments={}), ToolRegistry([broken])) is False
    assert replay_gold(
        {"plan": [{"tool": "missing.tool", "arguments": {}}], "expectations": []},
        registry,
        ["calendar"],
    ) is False
    assert truncate_to_tokens("alpha", 0) == ""
    assert tokenize_text("a b") == ["a", "b"]
    with pytest.raises(RuntimeError):
        verify_task_split([], 10)


def test_unconfirmed_execution_is_counted_and_loops_stop() -> None:
    tool = ToolSpec(
        name="files.share",
        app="files",
        action="share",
        phrase="share the file in files",
        description="share the file in files",
        side_effect=True,
        parameters=schema_for_fields(["title", "note"]),
        retrieval_text="files share the file in files",
    )
    registry = ToolRegistry([tool])
    rendered = {item.tool.name: item for item in render_tools([tool])}

    class OpenGate:
        def decide(self, _tool, _call):
            return GateDecision(True, "forced")

    class Repeat:
        def complete(self, visible, clause, history, n_ok, n_clauses):
            call = ToolCall(name="files.share", arguments={"title": "Case 1", "note": "alpha detail"}, confirm=False)
            return to_openai_response(call)

    episode = run_episode(
        task_id="loop",
        split="dev",
        config="toy",
        utterance=(
            "Please share the file in files title='Case 9' note='alpha detail'."
            " THEN share the file in files title='Case 8' note='bravo detail'."
        ),
        gold=[("files.share", {"title": "Case 9", "note": "alpha detail"})],
        expectations=[{"app": "files", "fields": {"tool": "files.share", "title": "Case 9"}}],
        state=empty_state(["files"]),
        registry=registry,
        select_rendered=lambda _clause: [rendered["files.share"]],
        agent=Repeat(),
        gate=OpenGate(),
        execute_fn=execute,
        check_fn=expectations_met,
        compact=False,
        context_limit=2048,
    )
    assert episode.unconfirmed_side_effects >= 1
    assert episode.failure_type == "loop"
    # Real gate still blocks the same call.
    blocked = run_episode(
        task_id="blocked",
        split="dev",
        config="toy",
        utterance="Please share the file in files title='Case 9' note='alpha detail'.",
        gold=[("files.share", {"title": "Case 9", "note": "alpha detail"})],
        expectations=[{"app": "files", "fields": {"tool": "files.share", "title": "Case 9"}}],
        state=empty_state(["files"]),
        registry=registry,
        select_rendered=lambda _clause: [rendered["files.share"]],
        agent=Repeat(),
        gate=PermissionGate(),
        execute_fn=execute,
        check_fn=expectations_met,
        compact=False,
        context_limit=2048,
        max_steps=3,
    )
    assert blocked.unconfirmed_side_effects == 0
    assert blocked.success == 0


def test_download_checksum_mismatch(tmp_path) -> None:
    class Response:
        content = b"nope"

        def raise_for_status(self) -> None:
            return None

    class Session:
        def get(self, url: str, timeout: int = 0) -> Response:
            return Response()

    with pytest.raises(RuntimeError):
        download_verified("https://example.test/x", tmp_path / "x.bin", "0" * 64, session=Session())


def test_random5_is_sampled_without_a_fitted_retriever() -> None:
    from evals.runner import evaluate_tasks

    tools = build_tools()
    task = build_tasks(tools, seed=0)[0]
    episodes = evaluate_tasks(
        [task],
        tools,
        {},
        [{"name": "random5", "retriever": "random5", "compact": False}],
        agent_kind="lexical_grounding",
    )
    assert len(episodes) == 1
    assert episodes[0].config == "random5"


def test_runner_rejects_unknown_agent_and_retriever() -> None:
    from evals.runner import evaluate_tasks

    tools = build_tools()
    task = build_tasks(tools, seed=0)[0]
    with pytest.raises(KeyError):
        evaluate_tasks(
            [task],
            tools,
            {},
            [{"name": "bm25", "retriever": "bm25", "compact": False}],
            agent_kind="lexical_grounding",
        )
    with pytest.raises(ValueError):
        evaluate_tasks(
            [task],
            tools,
            {"bm25": object()},
            [{"name": "bm25", "retriever": "bm25", "compact": False}],
            agent_kind="nope",
        )


def test_embed_choice_is_frozen_before_release() -> None:
    class Encoder:
        def __init__(self) -> None:
            self.model = object()
            self._cache = {"q": 1}

    class Retriever:
        def __init__(self, name: str) -> None:
            self.name = name
            self.encoder = Encoder()

        def retrieve(self, query: str, k: int = 5) -> list[str]:
            del k
            return [f"{self.name}:{query}"]

    retrievers = {"embed": Retriever("embed"), "bm25": Retriever("bm25")}
    kept_bm25 = _keep_retriever(retrievers, "bm25", [])
    assert kept_bm25 is retrievers["bm25"]
    kept = _keep_retriever(retrievers, "embed", [{"utterance": "Please hello."}])
    assert kept is not None
    assert kept.retrieve("hello") == ["embed:hello"]
    _release_embed(retrievers)
    assert retrievers["embed"].encoder.model is None
    assert retrievers["embed"].encoder._cache == {}
    assert _keep_retriever({}, "embed", []) is None


def test_encoder_wrapper_and_quick_pipeline(monkeypatch) -> None:
    import sentence_transformers
    import torch

    class FakeModel:
        def __init__(self, model_id: str, device: str = "cpu") -> None:
            assert "MiniLM" in model_id

        def encode(self, texts, **kwargs):
            return np.ones((len(texts), 8), dtype=np.float32)

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", FakeModel)
    monkeypatch.setattr(torch, "set_num_threads", lambda _n: None)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda _n: None)
    encoder = SentenceTransformerEncoder(device="cpu")
    tools = build_tools()[:6]
    retriever = EmbeddingRetriever(tools, encoder)
    assert len(retriever.retrieve("create the event in calendar title='Case 1'", k=3)) == 3
    # The lexical model is constructed so this test also imports the policy module path.
    assert LexicalGroundingModel().name == "lexical_grounding"
    payload = run_pipeline(ROOT, quick=True, device="cpu", skip_coverage=True)
    assert payload["n_test_tasks"] == 8
    assert payload["unconfirmed_side_effects"] == 0
    assert (ROOT / "results" / "quick" / "metrics.json").exists()
