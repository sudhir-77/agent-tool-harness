"""Caller wording, model-output parsing, and a fake generative backend."""

from __future__ import annotations

from apps.catalog import build_tools
from apps.state import empty_state, execute, expectations_met
from evals.data import download_verified_stream
from evals.generate import build_tasks, render_caller_clause, split_tasks
from harness.generative import (
    GenerativeToolModel,
    LlamaToolBackend,
    _pending_confirmation,
    build_messages,
    call_schema,
    coerce_call,
    fit_visible,
    parse_json_object,
)
from harness.retrieval import FrozenRetriever
from harness.llm import phrase_in_text
from harness.loop import run_episode
from harness.permissions import PermissionGate
from harness.registry import ToolRegistry


class FakeBackend:
    def __init__(self, payload: str, budget: int = 100000) -> None:
        self.payload = payload
        self.prompt_budget = budget
        self.agent_name = "qwen2.5-1.5b-instruct"
        self.device = "cpu"
        self.calls = 0

    def count_chat_tokens(self, messages: list[dict[str, str]]) -> int:
        return sum(len(message["content"]) for message in messages)

    def generate(self, messages: list[dict[str, str]], schema: dict) -> tuple[str, int]:
        del schema
        self.calls += 1
        return self.payload, 32


def test_test_split_hides_tool_phrase_and_name() -> None:
    tools = build_tools()
    by_name = {tool.name: tool for tool in tools}
    _dev, test = split_tasks(build_tasks(tools, seed=0))
    mentions = 0
    total = 0
    for task in test:
        clauses = __import__("harness.llm", fromlist=["split_clauses"]).split_clauses(task["utterance"])
        for clause, step in zip(clauses, task["plan"], strict=True):
            tool = by_name[step["tool"]]
            total += 1
            if phrase_in_text(tool.phrase, clause) or tool.name in clause:
                mentions += 1
            assert render_caller_clause(tool, step["arguments"]) == clause
    assert total > 0
    assert mentions / total <= 0.20


def test_parse_coerce_and_fit() -> None:
    tools = build_tools()
    tool = next(item for item in tools if item.name == "calendar.schedule_meeting")
    text = '```json\n{"name":"calendar.schedule_meeting","arguments":{"title":"Case 1","date":"2026-01-02","time":"09:00","duration_minutes":"30"},"confirm":false}\n```'
    call = coerce_call(parse_json_object(text), [tool])
    assert call is not None
    assert call.arguments["duration_minutes"] == 30
    assert call.confirm is False
    assert parse_json_object("no json here") is None
    messages = build_messages([tool], "Book a sitdown", [])
    assert "calendar.schedule_meeting" in messages[1]["content"]
    assert fit_visible([tool, tool], "step", [], lambda _messages: 10, budget=100) == [tool, tool]
    assert fit_visible([tool, tool], "step", [], lambda _messages: 10**6, budget=100) == []


def test_fake_generative_backend_executes_a_visible_tool() -> None:
    tools = build_tools()
    payload = (
        '{"name":"calendar.create","arguments":{"title":"Case 1","note":"alpha detail"},"confirm":false}'
    )
    backend = FakeBackend(payload)
    agent = GenerativeToolModel(backend)
    registry = ToolRegistry(tools)
    utterance = "Please create the event in calendar title='Case 1' note='alpha detail'."
    gold = [("calendar.create", {"title": "Case 1", "note": "alpha detail"})]
    expectations = [{"app": "calendar", "fields": {"tool": "calendar.create", "title": "Case 1", "note": "alpha detail"}}]

    def select(_clause: str):
        from harness.context import render_tools

        rendered = {item.tool.name: item for item in render_tools(tools)}
        return [rendered["calendar.create"]]

    episode = run_episode(
        task_id="toy",
        split="test",
        config="bm25",
        utterance=utterance,
        gold=gold,
        expectations=expectations,
        state=empty_state(["calendar"]),
        registry=registry,
        select_rendered=select,
        agent=agent,
        gate=PermissionGate(),
        execute_fn=execute,
        check_fn=expectations_met,
        compact=False,
        context_limit=4096,
        schema_retry_limit=1,
    )
    assert episode.success == 1
    assert episode.agent == "qwen2.5-1.5b-instruct"
    assert episode.n_calls >= 1
    assert episode.schema_retries == 0
    assert episode.unconfirmed_side_effects == 0
    assert backend.calls == episode.n_calls


def test_confirmation_schema_locks_the_pending_tool() -> None:
    tools = build_tools()
    tool = next(item for item in tools if item.name == "calendar.share")
    history = ['CONFIRMATION_REQUIRED tool=calendar.share args={"note": "alpha detail", "title": "Case 1"}']
    schema = call_schema([tool], history)
    assert schema["properties"]["confirm"]["const"] is True
    assert schema["properties"]["name"]["const"] == "calendar.share"
    assert "title" in schema["properties"]["arguments"]["required"]
    assert "set confirm to true" in build_messages([tool], "step", history)[1]["content"]
    assert _pending_confirmation(history + ["OK tool=calendar.share args={}"]) is None
    idle = call_schema([tool], [])
    assert idle["properties"]["confirm"]["const"] is False
    other = next(item for item in tools if item.name == "notes.create")
    both = call_schema([tool, other], [])
    assert len(both["oneOf"]) == 2
    assert call_schema([], [])["properties"]["name"]["enum"] == ["none"]


def test_extract_cuda_dlls_from_zip(tmp_path) -> None:
    import zipfile

    from harness.cuda_runtime import _extract_dlls

    archive = tmp_path / "cuda.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("pkg/bin/cudart64_12.dll", b"runtime")
        bundle.writestr("pkg/README.txt", b"skip")
    dest = tmp_path / "bin"
    dest.mkdir()
    _extract_dlls(archive, dest)
    _extract_dlls(archive, dest)
    assert (dest / "cudart64_12.dll").read_bytes() == b"runtime"
    assert list(dest.glob("*.txt")) == []


def test_stream_download_verifies_checksum(tmp_path) -> None:
    payload = b"gguf-bytes"
    import hashlib

    digest = hashlib.sha256(payload).hexdigest()

    class Response:
        def raise_for_status(self) -> None:
            return None

        def iter_content(self, chunk_size: int = 0):
            del chunk_size
            yield payload

        def close(self) -> None:
            return None

    class Session:
        def get(self, url: str, stream: bool = False, timeout: int = 0) -> Response:
            assert stream is True
            assert url == "https://example.test/model.gguf"
            return Response()

    dest = tmp_path / "models" / "model.gguf"
    download_verified_stream("https://example.test/model.gguf", dest, digest, session=Session())
    assert dest.read_bytes() == payload
    download_verified_stream("https://example.test/model.gguf", dest, digest, session=Session())


def test_frozen_retriever_replays_cached_names() -> None:
    frozen = FrozenRetriever("bm25", {"step": ["a", "b", "c"]})
    assert frozen.name == "bm25"
    assert frozen.retrieve("step", k=2) == ["a", "b"]
    assert frozen.retrieve("missing", k=5) == []


def test_llama_backend_counts_tokens_and_reads_usage() -> None:
    backend = LlamaToolBackend.__new__(LlamaToolBackend)
    backend._grammars = {}
    backend._formatter = None
    backend._bias = 2

    class Fake:
        closed = False

        def tokenize(self, data: bytes, add_bos: bool = False, special: bool = True) -> list[int]:
            assert add_bos is False
            assert special is True
            assert data
            return [1, 2, 3]

        def create_chat_completion(self, **kwargs: object) -> dict:
            assert kwargs.get("temperature") == 0.0
            return {
                "choices": [{"message": {"content": '{"ok": true}'}}],
                "usage": {"prompt_tokens": 11},
            }

        def close(self) -> None:
            self.closed = True

    fake = Fake()
    backend.llm = fake
    assert backend.count_chat_tokens([{"role": "user", "content": "hi"}]) == 5
    text, tokens = backend.generate(
        [{"role": "user", "content": "hi"}],
        {
            "type": "object",
            "properties": {"ok": {"type": "boolean", "const": True}},
            "required": ["ok"],
            "additionalProperties": False,
        },
    )
    assert text == '{"ok": true}'
    assert tokens == 11
    backend.close()
    assert fake.closed is True
    assert backend.llm is None
