"""Local generative tool caller.

The reported agent is Qwen2.5-1.5B-Instruct (Apache-2.0) in a Q4_K_M GGUF.
It emits one JSON tool call. The harness may retry that call once after a
schema error. Lexical grounding and the scripted oracle stay available as
labeled comparison policies.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Protocol

from harness.llm import ToolCall, to_openai_response
from harness.registry import ToolSpec

REPORTED_AGENT = "qwen2.5-1.5b-instruct"
LLM_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
GGUF_FILENAME = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
GGUF_URL = (
    "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/"
    "qwen2.5-1.5b-instruct-q4_k_m.gguf"
)
GGUF_SHA256 = "6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e"
CONTEXT_TOKENS = 8192
MAX_NEW_TOKENS = 160
PROMPT_MARGIN = 128

SYSTEM_PROMPT = (
    "You call exactly one tool. Reply with one JSON object and no other text. "
    "Keys are name, arguments, and confirm. "
    "name must be one of the listed tool names. "
    "Copy argument values from the current step. Integer fields are JSON numbers, not strings. "
    "Set confirm to false unless the user message says the previous call needs confirmation. "
    "If the last history line starts with SCHEMA_INVALID, fix the arguments so they match the tool schema."
)


class TextBackend(Protocol):
    prompt_budget: int
    agent_name: str
    device: str

    def count_chat_tokens(self, messages: list[dict[str, str]]) -> int:
        """Token count of the chat prompt the model will see."""

    def generate(self, messages: list[dict[str, str]], schema: dict[str, Any]) -> tuple[str, int]:
        """Return completion text and the prompt token count."""


def tool_card(tool: ToolSpec) -> str:
    fields = []
    for name, spec in tool.parameters.get("properties", {}).items():
        fields.append(f"{name} ({spec.get('type', 'string')})")
    side = "true" if tool.side_effect else "false"
    wording = tool.description.split("Callers may say: ", 1)[-1].rstrip(".")
    return (
        f"- {tool.name} side_effect={side} caller=\"{wording}\" "
        f"arguments: {', '.join(fields)}"
    )


def build_messages(visible: list[ToolSpec], clause: str, history: list[str]) -> list[dict[str, str]]:
    history_text = "\n".join(history) if history else "(none)"
    cards = "\n".join(tool_card(tool) for tool in visible) if visible else "(none)"
    pending = _pending_confirmation(history)
    if pending is not None:
        confirm_hint = (
            "The previous call needs confirmation. "
            "Repeat that tool name and every argument value exactly, and set confirm to true.\n\n"
        )
        tail = ""
    else:
        confirm_hint = ""
        tail = (
            "\n\nMatch the tool whose caller text appears in CURRENT STEP. "
            "Copy each key=value from CURRENT STEP into arguments. "
            "Do not invent values."
        )
    # Tools come first so a repeated catalog can reuse the llama.cpp prompt cache.
    user = (
        f"TOOLS:\n{cards}\n\n"
        f"{confirm_hint}"
        f"CURRENT STEP:\n{clause}\n\n"
        f"HISTORY:\n{history_text}"
        f"{tail}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def _pending_confirmation(history: list[str]) -> str | None:
    """Tool name from the latest confirmation request that has not been executed."""
    for line in reversed(history):
        if line.startswith("OK "):
            return None
        if line.startswith("CONFIRMATION_REQUIRED"):
            match = re.search(r"\btool=(\S+)", line)
            return match.group(1) if match else None
    return None


def _argument_schema(tool: ToolSpec) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    for name, spec in tool.parameters.get("properties", {}).items():
        properties[name] = {"type": spec.get("type", "string")}
    return {
        "type": "object",
        "properties": properties,
        "required": [key for key in tool.parameters.get("required", []) if key in properties],
        "additionalProperties": False,
    }


def _call_branch(tool: ToolSpec, confirm: bool) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string", "const": tool.name},
            "arguments": _argument_schema(tool),
            "confirm": {"type": "boolean", "const": confirm},
        },
        "required": ["name", "arguments", "confirm"],
        "additionalProperties": False,
    }


def call_schema(visible: list[ToolSpec], history: list[str] | None = None) -> dict[str, Any]:
    """JSON schema for one call. Each branch requires that tool's argument keys."""
    history = list(history or [])
    pending = _pending_confirmation(history)
    if pending is not None:
        chosen = [tool for tool in visible if tool.name == pending]
        tools = chosen or list(visible)
        confirm = True
    else:
        tools = list(visible)
        confirm = False
    if not tools:
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string", "enum": ["none"]},
                "arguments": {"type": "object", "properties": {}, "additionalProperties": False},
                "confirm": {"type": "boolean", "const": False},
            },
            "required": ["name", "arguments", "confirm"],
            "additionalProperties": False,
        }
    branches = [_call_branch(tool, confirm) for tool in tools]
    if len(branches) == 1:
        return branches[0]
    return {"oneOf": branches}


def serialize_chat(messages: list[dict[str, str]]) -> str:
    """Qwen2 chat wrapper. Special tokens are counted by the model tokenizer."""
    parts: list[str] = []
    for message in messages:
        parts.append(f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def parse_json_object(text: str) -> dict[str, Any] | None:
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, flags=re.DOTALL)
    if fenced:
        cleaned = fenced.group(1)
    tagged = re.search(r"<tool_call>\s*(\{.*\})\s*</tool_call>", cleaned, flags=re.DOTALL)
    if tagged:
        cleaned = tagged.group(1)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def coerce_call(payload: dict[str, Any] | None, visible: list[ToolSpec]) -> ToolCall | None:
    """Normalize a model JSON object into a tool call. Returns None if it is not an object."""
    if payload is None:
        return None
    name = payload.get("name")
    if not isinstance(name, str) or not name:
        return None
    raw_args = payload.get("arguments", {})
    if isinstance(raw_args, str):
        try:
            raw_args = json.loads(raw_args)
        except json.JSONDecodeError:
            raw_args = {}
    if not isinstance(raw_args, dict):
        raw_args = {}
    confirm = _as_bool(payload.get("confirm", False))
    tool = next((item for item in visible if item.name == name), None)
    if tool is None:
        return ToolCall(name=name, arguments={str(k): v for k, v in raw_args.items()}, confirm=confirm)
    cleaned: dict[str, Any] = {}
    properties = tool.parameters.get("properties", {})
    for key, spec in properties.items():
        if key not in raw_args:
            continue
        value = raw_args[key]
        if spec.get("type") == "integer":
            if isinstance(value, bool):
                continue
            if isinstance(value, int):
                cleaned[key] = value
            elif isinstance(value, float) and value.is_integer():
                cleaned[key] = int(value)
            elif isinstance(value, str) and re.fullmatch(r"-?\d+", value.strip()):
                cleaned[key] = int(value.strip())
        else:
            cleaned[key] = value if isinstance(value, str) else str(value)
    return ToolCall(name=name, arguments=cleaned, confirm=confirm)


def fit_visible(
    visible: list[ToolSpec],
    clause: str,
    history: list[str],
    count_tokens: Any,
    budget: int,
) -> list[ToolSpec]:
    """Keep a prefix of visible tools whose chat prompt fits the context budget."""
    if not visible:
        return []

    def size(count: int) -> int:
        return int(count_tokens(build_messages(visible[:count], clause, history)))

    if size(len(visible)) <= budget:
        return list(visible)
    best = 0
    lo = 0
    hi = len(visible) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        count = mid + 1
        if size(count) <= budget:
            best = count
            lo = mid + 1
        else:
            hi = mid - 1
    return list(visible[:best])


class GenerativeToolModel:
    """One-step tool caller backed by a local instruct model."""

    def __init__(self, backend: TextBackend) -> None:
        self.backend = backend
        self.name = backend.agent_name

    def complete(
        self,
        visible: list[ToolSpec],
        clause: str,
        history: list[str],
        n_ok: int,
        n_clauses: int,
    ) -> dict[str, Any]:
        del n_ok, n_clauses
        shown = fit_visible(
            visible,
            clause,
            history,
            self.backend.count_chat_tokens,
            self.backend.prompt_budget,
        )
        if not shown:
            payload = to_openai_response(None)
            payload["prompt_tokens"] = 0
            return payload
        messages = build_messages(shown, clause, history)
        text, prompt_tokens = self.backend.generate(messages, call_schema(shown, history))
        call = coerce_call(parse_json_object(text), shown)
        payload = to_openai_response(call)
        payload["prompt_tokens"] = int(prompt_tokens)
        return payload


class LlamaToolBackend:
    """llama.cpp runtime for the Qwen GGUF. GPU offload is used when the wheel supports it."""

    agent_name = REPORTED_AGENT

    def __init__(self, model_path: Path, n_ctx: int = CONTEXT_TOKENS) -> None:
        from llama_cpp import Llama

        layers = 0
        try:
            from llama_cpp import llama_supports_gpu_offload

            if llama_supports_gpu_offload():
                layers = -1
        except Exception:
            layers = 0
        self.n_ctx = n_ctx
        self.prompt_budget = n_ctx - MAX_NEW_TOKENS - PROMPT_MARGIN
        self._layers = layers
        self._grammars: dict[str, Any] = {}
        self._formatter: Any | None = None
        self._bias = 0
        self.llm = Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            n_gpu_layers=layers,
            n_threads=4,
            n_threads_batch=4,
            # One batch for the whole context. Smaller CUDA batches changed
            # argmax on the long all-tools prompt between processes.
            n_batch=n_ctx,
            n_ubatch=n_ctx,
            seed=0,
            verbose=False,
            use_mmap=True,
            use_mlock=False,
            logits_all=False,
            offload_kqv=True,
        )
        self.device = "cuda" if layers != 0 else "cpu"
        self._formatter = self._make_formatter()
        self._calibrate()

    def _make_formatter(self) -> Any | None:
        template = self.llm.metadata.get("tokenizer.chat_template")
        if not isinstance(template, str) or not template:
            return None
        try:
            from llama_cpp.llama_chat_format import Jinja2ChatFormatter

            eos_id = self.llm.token_eos()
            bos_id = self.llm.token_bos()
            eos = self.llm._model.token_get_text(eos_id) if eos_id != -1 else ""
            bos = self.llm._model.token_get_text(bos_id) if bos_id != -1 else ""
            return Jinja2ChatFormatter(template=template, eos_token=eos, bos_token=bos)
        except Exception:
            return None

    def _render_chat(self, messages: list[dict[str, str]]) -> str:
        formatter = self._formatter
        if formatter is not None:
            try:
                return str(formatter(messages=messages).prompt)
            except Exception:
                pass
        return serialize_chat(messages)

    def count_chat_tokens(self, messages: list[dict[str, str]]) -> int:
        text = self._render_chat(messages)
        token_ids = self.llm.tokenize(text.encode("utf-8"), add_bos=False, special=True)
        return max(1, len(token_ids) + int(self._bias))

    def _grammar_for(self, schema: dict[str, Any]) -> Any | None:
        key = json.dumps(schema, sort_keys=True, separators=(",", ":"))
        if key in self._grammars:
            return self._grammars[key]
        grammar = None
        try:
            from llama_cpp import LlamaGrammar

            grammar = LlamaGrammar.from_json_schema(key, verbose=False)
        except Exception as exc:
            print(f"grammar_error={type(exc).__name__}: {exc}", flush=True)
            grammar = None
        self._grammars[key] = grammar
        return grammar

    def generate(self, messages: list[dict[str, str]], schema: dict[str, Any]) -> tuple[str, int]:
        kwargs: dict[str, Any] = {
            "messages": messages,
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 1.0,
            "min_p": 0.0,
            "max_tokens": MAX_NEW_TOKENS,
            "repeat_penalty": 1.0,
            "seed": 0,
        }
        grammar = self._grammar_for(schema)
        if grammar is not None:
            kwargs["grammar"] = grammar
        try:
            result = self.llm.create_chat_completion(**kwargs)
        except TypeError:
            kwargs.pop("seed", None)
            try:
                result = self.llm.create_chat_completion(**kwargs)
            except Exception as exc:
                print(f"generate_error={type(exc).__name__}: {exc}", flush=True)
                return "", self.count_chat_tokens(messages)
        except Exception as exc:
            print(f"generate_error={type(exc).__name__}: {exc}", flush=True)
            return "", self.count_chat_tokens(messages)
        message = result["choices"][0]["message"]
        text = message.get("content") or ""
        usage = result.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or self.count_chat_tokens(messages))
        return text, prompt_tokens

    def _calibrate(self) -> None:
        """Align the fitter's token count with usage.prompt_tokens from one short call."""
        messages = [
            {"role": "system", "content": "Reply with one JSON object."},
            {"role": "user", "content": "ping"},
        ]
        self._bias = 0
        estimated = self.count_chat_tokens(messages)
        schema = {
            "type": "object",
            "properties": {"ok": {"type": "boolean", "const": True}},
            "required": ["ok"],
            "additionalProperties": False,
        }
        _text, actual = self.generate(messages, schema)
        if actual <= 0:
            self._bias = 0
        else:
            self._bias = max(-8, min(64, int(actual) - int(estimated)))
        print(f"llm_device={self.device} token_bias={self._bias}", flush=True)

    def close(self) -> None:
        llm = getattr(self, "llm", None)
        close = getattr(llm, "close", None)
        if close is not None:
            close()
        self.llm = None


def load_llama_backend(model_path: Path, data_dir: Path) -> LlamaToolBackend:
    from harness.cuda_runtime import prepare_windows_cuda_runtime

    prepare_windows_cuda_runtime(data_dir)
    return LlamaToolBackend(model_path)
