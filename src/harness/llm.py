"""Model-agnostic tool-calling policies behind one OpenAI-style response.

The evaluated policy is lexical grounding over tools that are actually inside
the context window. The scripted oracle is an upper bound: it may emit the
gold call only when that tool's schema is visible. An optional transformers
backend parses a generated tool call from a local instruct model.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from rank_bm25 import BM25Okapi

from harness.registry import ToolCall, ToolSpec
from harness.retrieval import retrieval_query, tokenize


def phrase_in_text(phrase: str, text: str) -> bool:
    """True when phrase occurs on word boundaries (so 'pin' does not match 'unpin')."""
    pattern = rf"(?<!\w){re.escape(phrase)}(?!\w)"
    return re.search(pattern, text) is not None


def parse_args(clause: str, schema: dict[str, Any]) -> dict[str, Any]:
    arguments: dict[str, Any] = {}
    properties = schema.get("properties", {})
    for name, spec in properties.items():
        escaped = re.escape(name)
        if spec.get("type") == "integer":
            match = re.search(rf"\b{escaped}=(-?\d+)\b", clause)
            arguments[name] = int(match.group(1)) if match else 0
        else:
            match = re.search(rf"\b{escaped}='([^']*)'", clause)
            arguments[name] = match.group(1) if match else ""
    return arguments


def split_clauses(utterance: str) -> list[str]:
    text = utterance.strip()
    for prefix in ("Please ", "User request: "):
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break
    if text.endswith("."):
        text = text[:-1]
    return [part.strip() for part in text.split(" THEN ") if part.strip()]


def to_openai_response(call: ToolCall | None) -> dict[str, Any]:
    if call is None:
        message: dict[str, Any] = {"role": "assistant", "content": "DONE", "tool_calls": []}
    else:
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "type": "function",
                    "confirm": call.confirm,
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments, sort_keys=True),
                    },
                }
            ],
        }
    return {"choices": [{"message": message}]}


def call_from_openai(response: dict[str, Any]) -> ToolCall | None:
    message = response["choices"][0]["message"]
    tool_calls = message.get("tool_calls") or []
    if not tool_calls:
        return None
    function = tool_calls[0]["function"]
    raw_args = function["arguments"]
    arguments = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
    confirm = bool(tool_calls[0].get("confirm", False))
    return ToolCall(name=function["name"], arguments=arguments, confirm=confirm)


def parse_tool_call_text(text: str) -> ToolCall | None:
    """Parse a model completion of the form TOOLCALL {json}."""
    match = re.search(r"TOOLCALL\s*(\{.*\})", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    name = payload.get("name")
    arguments = payload.get("arguments", {})
    if not isinstance(name, str) or not isinstance(arguments, dict):
        return None
    return ToolCall(name=name, arguments=arguments, confirm=bool(payload.get("confirm", False)))


@dataclass
class PlannedCall:
    name: str
    arguments: dict[str, Any]


def _bm25_best(clause: str, visible: list[ToolSpec]) -> ToolSpec:
    docs = [tokenize(tool.retrieval_text) or ["empty"] for tool in visible]
    scorer = BM25Okapi(docs)
    scores = scorer.get_scores(tokenize(retrieval_query(clause)))
    best = max(range(len(visible)), key=lambda i: (float(scores[i]), len(visible[i].phrase), visible[i].name))
    return visible[best]


class LexicalGroundingModel:
    """Pick a visible tool whose action phrase is in the current clause.

    This is the policy used for the ablation table. It does not read gold labels.
    The same policy runs for every retrieval configuration.
    """

    name = "lexical_grounding"

    def __init__(self) -> None:
        self.last_call: ToolCall | None = None

    def decide(self, visible: list[ToolSpec], clause: str, history: list[str], n_ok: int, n_clauses: int) -> ToolCall | None:
        if history and history[-1].startswith("CONFIRMATION_REQUIRED") and self.last_call is not None:
            return ToolCall(name=self.last_call.name, arguments=dict(self.last_call.arguments), confirm=True)
        if n_ok >= n_clauses or not visible or not clause:
            return None
        matches = [tool for tool in visible if phrase_in_text(tool.phrase, clause)]
        if matches:
            tool = max(matches, key=lambda item: (len(item.phrase), item.name))
        else:
            tool = _bm25_best(clause, visible)
        call = ToolCall(name=tool.name, arguments=parse_args(clause, tool.parameters), confirm=False)
        self.last_call = call
        return call

    def complete(self, visible: list[ToolSpec], clause: str, history: list[str], n_ok: int, n_clauses: int) -> dict[str, Any]:
        return to_openai_response(self.decide(visible, clause, history, n_ok, n_clauses))


class ScriptedOracle:
    """Upper bound that emits the gold call only if that tool is visible."""

    name = "scripted_oracle"

    def __init__(self, plan: list[PlannedCall]) -> None:
        self.plan = list(plan)
        self.idx = 0

    def decide(self, visible: list[ToolSpec], clause: str, history: list[str], n_ok: int, n_clauses: int) -> ToolCall | None:
        del clause, n_ok, n_clauses
        if history and history[-1].startswith("CONFIRMATION_REQUIRED") and self.idx < len(self.plan):
            step = self.plan[self.idx]
            self.idx += 1
            return ToolCall(name=step.name, arguments=dict(step.arguments), confirm=True)
        if self.idx >= len(self.plan):
            return None
        step = self.plan[self.idx]
        names = {tool.name for tool in visible}
        if step.name not in names:
            return None
        tool = next(tool for tool in visible if tool.name == step.name)
        if tool.side_effect:
            return ToolCall(name=step.name, arguments=dict(step.arguments), confirm=False)
        self.idx += 1
        return ToolCall(name=step.name, arguments=dict(step.arguments), confirm=False)

    def complete(self, visible: list[ToolSpec], clause: str, history: list[str], n_ok: int, n_clauses: int) -> dict[str, Any]:
        return to_openai_response(self.decide(visible, clause, history, n_ok, n_clauses))


class TransformersToolClient:
    """Local instruct model behind the same OpenAI-style tool-call interface.

    Not used for the main ablation. A 135M-class model on CPU is too slow to
    score every configuration on the full task set inside the memory budget.
    """

    def __init__(self, model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct", device: str = "cpu") -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.set_num_threads(4)
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

    def generate(self, prompt: str, max_new_tokens: int = 64) -> str:
        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        new_tokens = output[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def complete_from_prompt(self, prompt: str) -> dict[str, Any]:
        text = self.generate(prompt)
        return to_openai_response(parse_tool_call_text(text))
