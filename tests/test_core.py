"""Hand-checked tokenizer, metrics, permissions, retrieval, and loop cases."""

from __future__ import annotations

import numpy as np

from apps.state import empty_state, execute, expectations_met
from evals.data import download_verified, sha256_bytes
from evals.metrics import (
    aggregate_episodes,
    compare,
    recall_at_5,
    select_best,
    success_gap_points,
    token_reduction,
)
from harness.context import compact_history, render_tools
from harness.llm import (
    PlannedCall,
    ScriptedOracle,
    ToolCall,
    call_from_openai,
    parse_args,
    parse_tool_call_text,
    phrase_in_text,
    split_clauses,
    to_openai_response,
)
from harness.loop import classify_failure, run_episode
from harness.permissions import PermissionGate
from harness.registry import ToolRegistry, ToolSpec, schema_for_fields
from harness.retrieval import BM25Retriever, EmbeddingRetriever, FaissEmbeddingRetriever
from harness.schema import validate_call
from harness.tokens import count_tokens, truncate_to_tokens


def _tool(name: str, phrase: str, fields: list[str], side_effect: bool = False) -> ToolSpec:
    app, action = name.split(".")
    return ToolSpec(
        name=name,
        app=app,
        action=action,
        phrase=phrase,
        description=phrase,
        side_effect=side_effect,
        parameters=schema_for_fields(fields),
        retrieval_text=f"{name.replace('.', ' ')} {phrase}",
    )


def test_count_tokens_hand_computed() -> None:
    assert count_tokens("") == 0
    assert count_tokens("alpha beta") == 2
    assert count_tokens("x=1") == 3  # x, =, 1
    assert count_tokens("a\nb") == count_tokens("a") + count_tokens("b")
    assert truncate_to_tokens("alpha beta gamma", 2) == "alpha beta"
    assert count_tokens(truncate_to_tokens("alpha beta gamma", 10)) == 3


def test_metric_functions_hand_computed() -> None:
    rows = [
        {"config": "all_tools", "success": 0, "prompt_tokens": 1000, "n_steps": 2, "schema_valid_calls": 2, "wrong_tool": 1, "failure_type": "wrong_tool", "latency_ms": 10},
        {"config": "all_tools", "success": 1, "prompt_tokens": 1000, "n_steps": 1, "schema_valid_calls": 1, "wrong_tool": 0, "failure_type": "success", "latency_ms": 30},
        {"config": "bm25", "success": 1, "prompt_tokens": 100, "n_steps": 1, "schema_valid_calls": 1, "wrong_tool": 0, "failure_type": "success", "latency_ms": 5},
        {"config": "bm25", "success": 1, "prompt_tokens": 200, "n_steps": 1, "schema_valid_calls": 0, "wrong_tool": 0, "failure_type": "success", "latency_ms": 7},
    ]
    summary = aggregate_episodes(rows)
    assert summary["success_by_config"]["all_tools"] == 0.5
    assert summary["success_by_config"]["bm25"] == 1.0
    assert summary["tokens_per_task_by_config"]["all_tools"] == 1000
    assert summary["tokens_per_task_by_config"]["bm25"] == 150
    assert summary["schema_valid_rate"] == 4 / 5
    assert summary["latency_ms_mean_by_config"]["all_tools"] == 20
    best = select_best(summary["success_by_config"], summary["tokens_per_task_by_config"])
    assert best == "bm25"
    assert success_gap_points(summary["success_by_config"], best) == 50
    assert token_reduction(summary["tokens_per_task_by_config"], best) == 0.85
    assert compare(0.85, ">=", 0.60)
    assert compare(0, "<=", 0)
    assert compare(0.5, "between", 0.4, 0.6)
    retrieval = [
        {"retriever": "bm25", "split": "test", "hit_at_5": 1},
        {"retriever": "bm25", "split": "test", "hit_at_5": 1},
        {"retriever": "bm25", "split": "test", "hit_at_5": 0},
        {"retriever": "bm25", "split": "test", "hit_at_5": 1},
        {"retriever": "bm25", "split": "dev", "hit_at_5": 0},
    ]
    assert recall_at_5(retrieval, "bm25", split="test") == 0.75


def test_phrase_boundaries_do_not_match_unpin_as_pin() -> None:
    clause = "unpin the event in calendar title='Case 1' note='alpha detail'"
    assert phrase_in_text("unpin the event in calendar", clause)
    assert not phrase_in_text("pin the event in calendar", clause)
    assert not phrase_in_text("unstar the event in calendar", "star the event in calendar title='Case 1'")


def test_parse_args_round_trip() -> None:
    schema = schema_for_fields(["title", "note", "duration_minutes"])
    clause = "schedule the meeting in calendar title='Case 1001' note='alpha detail' duration_minutes=30"
    assert parse_args(clause, schema) == {
        "title": "Case 1001",
        "note": "alpha detail",
        "duration_minutes": 30,
    }


def test_split_clauses_dev_and_test_wrappers() -> None:
    body = "create the event in calendar title='Case 1' note='alpha detail' THEN list the event in calendar title='Case 1' note='bravo detail'"
    dev = split_clauses(f"Please {body}.")
    test = split_clauses(f"User request: {body}.")
    assert dev == test
    assert len(dev) == 2
    assert dev[0].startswith("create the event")


def test_openai_tool_call_round_trip() -> None:
    call = ToolCall(name="calendar.create", arguments={"title": "Case 1", "note": "alpha detail"}, confirm=True)
    restored = call_from_openai(to_openai_response(call))
    assert restored is not None
    assert restored.name == call.name
    assert restored.arguments == call.arguments
    assert restored.confirm is True
    assert call_from_openai(to_openai_response(None)) is None
    parsed = parse_tool_call_text('prefix TOOLCALL {"name": "mail.send_email", "arguments": {"title": "A"}, "confirm": false}')
    assert parsed is not None
    assert parsed.name == "mail.send_email"
    assert parsed.arguments == {"title": "A"}
    assert parse_tool_call_text("no call here") is None


def test_permission_gate_blocks_unconfirmed_side_effect() -> None:
    tool = _tool("files.delete", "delete the file in files", ["title", "note"], side_effect=True)
    gate = PermissionGate()
    state = empty_state(["files"])
    denied = gate.decide(tool, ToolCall(name=tool.name, arguments={"title": "Case 1", "note": "alpha detail"}, confirm=False))
    assert denied.allow is False
    snapshot = empty_state(["files"])
    assert state == snapshot
    allowed = gate.decide(tool, ToolCall(name=tool.name, arguments={"title": "Case 1", "note": "alpha detail"}, confirm=True))
    assert allowed.allow is True
    execute(state, tool, {"title": "Case 1", "note": "alpha detail"})
    assert state["files"]["docs"] == []
    assert state["files"]["activity"][0]["tool"] == "files.delete"
    safe = _tool("files.create", "create the file in files", ["title", "note"], side_effect=False)
    assert gate.decide(safe, ToolCall(name=safe.name, arguments={"title": "Case 1", "note": "n"}, confirm=False)).allow


def test_schema_validation_known_cases() -> None:
    tool = _tool("notes.create", "create the note in notes", ["title", "note"])
    registry = ToolRegistry([tool])
    good = ToolCall(name=tool.name, arguments={"title": "Case 1", "note": "alpha detail"})
    bad = ToolCall(name=tool.name, arguments={"title": "Case 1"})
    extra = ToolCall(name=tool.name, arguments={"title": "Case 1", "note": "alpha detail", "extra": 1})
    missing = ToolCall(name="nope.tool", arguments={})
    assert validate_call(good, registry)
    assert not validate_call(bad, registry)
    assert not validate_call(extra, registry)
    assert not validate_call(missing, registry)


def test_bm25_known_case_prefers_calendar_create() -> None:
    from harness.retrieval import retrieval_query

    query = "create the event in calendar title='Case 1' note='alpha detail'"
    assert retrieval_query(query) == "create the event in calendar"
    tools = [
        _tool("calendar.create", "create the event in calendar", ["title", "note"]),
        _tool("notes.create", "create the note in notes", ["title", "note"]),
        _tool("calendar.list", "list the event in calendar", ["title", "note"]),
    ]
    # Extra documents keep the IDF of shared words ("the", "in") from going negative.
    for index in range(8):
        tools.append(_tool(f"misc.filler{index}", f"ignore the filler number {index} in misc", ["title", "note"]))
    retriever = BM25Retriever(tools)
    ranked = retriever.retrieve(query, k=2)
    assert ranked[0] == "calendar.create"


def test_faiss_and_embedding_retriever_known_vector() -> None:
    tools = [
        _tool("calendar.create", "create the event in calendar", ["title", "note"]),
        _tool("notes.create", "create the note in notes", ["title", "note"]),
        _tool("mail.send_email", "send the email in mail", ["title", "note"]),
    ]
    eye = np.eye(3, dtype=np.float32)
    faiss_retriever = FaissEmbeddingRetriever(tools, eye)
    assert faiss_retriever.search_vector(np.array([1.0, 0.0, 0.0], dtype=np.float32), k=1) == ["calendar.create"]

    class FakeEncoder:
        def encode_documents(self, texts: list[str]) -> np.ndarray:
            rows = []
            for index, _text in enumerate(texts):
                row = np.zeros(len(texts), dtype=np.float32)
                row[index] = 1.0
                rows.append(row)
            return np.stack(rows)

        def encode_query(self, text: str) -> np.ndarray:
            vector = np.zeros(len(tools), dtype=np.float32)
            for index, tool in enumerate(tools):
                if tool.phrase in text:
                    vector[index] = 1.0
                    return vector
            vector[0] = 1.0
            return vector

    retriever = EmbeddingRetriever(tools, FakeEncoder())
    assert retriever.retrieve("send the email in mail title='Case 1'", k=1) == ["mail.send_email"]


def test_classify_failure_branches() -> None:
    assert classify_failure(success=True, schema_invalid=False, looped=False, wrong_tool=0, completed_steps=1, gold_steps=1, hit_budget=False) == "success"
    assert classify_failure(success=False, schema_invalid=True, looped=True, wrong_tool=1, completed_steps=0, gold_steps=1, hit_budget=True) == "schema_invalid"
    assert classify_failure(success=False, schema_invalid=False, looped=True, wrong_tool=1, completed_steps=2, gold_steps=1, hit_budget=False) == "loop"
    assert classify_failure(success=False, schema_invalid=False, looped=False, wrong_tool=1, completed_steps=1, gold_steps=1, hit_budget=False) == "wrong_tool"
    assert classify_failure(success=False, schema_invalid=False, looped=False, wrong_tool=0, completed_steps=0, gold_steps=2, hit_budget=False) == "premature_stop"
    assert classify_failure(success=False, schema_invalid=False, looped=False, wrong_tool=0, completed_steps=1, gold_steps=3, hit_budget=True) == "budget_exceeded"
    assert classify_failure(success=False, schema_invalid=False, looped=False, wrong_tool=0, completed_steps=1, gold_steps=1, hit_budget=False) == "bad_arguments"


def _episode(agent, tools, utterance, gold, expectations, limit=2048, compact=False, selector=None):
    registry = ToolRegistry(tools)
    rendered = {item.tool.name: item for item in render_tools(tools)}

    def select(clause: str):
        if selector is None:
            return list(rendered.values())
        return [rendered[name] for name in selector(clause)]

    return run_episode(
        task_id="toy",
        split="dev",
        config="toy",
        utterance=utterance,
        gold=gold,
        expectations=expectations,
        state=empty_state(["calendar", "notes", "files"]),
        registry=registry,
        select_rendered=select,
        agent=agent,
        gate=PermissionGate(),
        execute_fn=execute,
        check_fn=expectations_met,
        compact=compact,
        context_limit=limit,
    )


def test_loop_success_wrong_tool_bad_args_and_permission() -> None:
    create = _tool("calendar.create", "create the event in calendar", ["title", "note"])
    share = _tool("calendar.share", "share the event in calendar", ["title", "note"], side_effect=True)
    other = _tool("notes.create", "create the note in notes", ["title", "note"])
    tools = [create, share, other]
    utterance = "Please create the event in calendar title='Case 1' note='alpha detail'."
    gold = [("calendar.create", {"title": "Case 1", "note": "alpha detail"})]
    expectations = [{"app": "calendar", "fields": {"tool": "calendar.create", "title": "Case 1", "note": "alpha detail"}}]

    class GoldVisible:
        def complete(self, visible, clause, history, n_ok, n_clauses):
            from harness.llm import LexicalGroundingModel

            return LexicalGroundingModel().complete(visible, clause, history, n_ok, n_clauses)

    # Fresh model inside complete loses last_call. Use one instance.
    model = __import__("harness.llm", fromlist=["LexicalGroundingModel"]).LexicalGroundingModel()
    success = _episode(model, tools, utterance, gold, expectations, selector=lambda _c: ["calendar.create"])
    assert success.success == 1
    assert success.failure_type == "success"
    assert success.unconfirmed_side_effects == 0
    assert success.schema_valid_calls == success.n_steps

    wrong_model = __import__("harness.llm", fromlist=["LexicalGroundingModel"]).LexicalGroundingModel()
    wrong = _episode(wrong_model, tools, utterance, gold, expectations, selector=lambda _c: ["notes.create"])
    assert wrong.success == 0
    assert wrong.failure_type == "wrong_tool"
    assert wrong.wrong_tool >= 1

    class BadArgs:
        def complete(self, visible, clause, history, n_ok, n_clauses):
            call = ToolCall(name="calendar.create", arguments={"title": "", "note": ""}, confirm=False)
            return to_openai_response(call)

    bad = _episode(BadArgs(), tools, utterance, gold, expectations, selector=lambda _c: ["calendar.create"])
    assert bad.success == 0
    assert bad.failure_type == "bad_arguments"

    class Invalid:
        def complete(self, visible, clause, history, n_ok, n_clauses):
            return to_openai_response(ToolCall(name="calendar.create", arguments={}, confirm=False))

    invalid = _episode(Invalid(), tools, utterance, gold, expectations, selector=lambda _c: ["calendar.create"], limit=4096)
    assert invalid.failure_type == "schema_invalid"
    assert invalid.schema_valid_calls < invalid.n_steps

    share_utterance = "Please share the event in calendar title='Case 1' note='alpha detail'."
    share_gold = [("calendar.share", {"title": "Case 1", "note": "alpha detail"})]
    share_expect = [{"app": "calendar", "fields": {"tool": "calendar.share", "title": "Case 1"}}]
    share_model = __import__("harness.llm", fromlist=["LexicalGroundingModel"]).LexicalGroundingModel()
    shared = _episode(
        share_model,
        tools,
        share_utterance,
        share_gold,
        share_expect,
        selector=lambda _c: ["calendar.share"],
    )
    assert shared.success == 1
    assert shared.unconfirmed_side_effects == 0
    assert shared.n_steps == 2

    oracle = ScriptedOracle([PlannedCall("calendar.create", {"title": "Case 1", "note": "alpha detail"})])
    hidden = _episode(oracle, tools, utterance, gold, expectations, selector=lambda _c: ["notes.create"])
    assert hidden.success == 0
    assert hidden.failure_type == "premature_stop"


def test_compaction_shrinks_long_history() -> None:
    history = [
        "OK tool=calendar.create args={\"title\": \"Case 1\", \"note\": \"alpha detail\"}",
        "OK tool=calendar.update args={\"title\": \"Case 1\", \"note\": \"bravo detail with a longer observation\"}",
        "OK tool=calendar.tag args={\"title\": \"Case 1\", \"note\": \"cedar detail and more words here\"}",
    ]
    compact = compact_history(history)
    assert compact[-1] == history[-1]
    assert len(" ".join(compact)) < len(" ".join(history))
    assert compact_history(history[:1]) == history[:1]


def test_download_verified_checksum(tmp_path) -> None:
    payload = b"abc"
    digest = sha256_bytes(payload)

    class Response:
        content = payload

        def raise_for_status(self) -> None:
            return None

    class Session:
        def get(self, url: str, timeout: int = 0) -> Response:
            assert url == "https://example.test/file"
            return Response()

    dest = tmp_path / "nested" / "file.txt"
    download_verified("https://example.test/file", dest, digest, session=Session())
    assert dest.read_bytes() == payload
    download_verified("https://example.test/file", dest, digest, session=Session())
