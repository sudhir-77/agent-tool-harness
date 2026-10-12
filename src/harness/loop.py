"""Planner loop: build a prompt, call the model, validate, gate, observe."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from harness.budget import StepBudget
from harness.context import PromptPacket, RenderedTool, assemble_prompt
from harness.llm import call_from_openai, split_clauses
from harness.permissions import PermissionGate
from harness.registry import ToolCall, ToolRegistry, ToolSpec
from harness.schema import validate_call


ExecuteFn = Callable[[dict[str, Any], ToolSpec, dict[str, Any]], dict[str, Any]]
CheckFn = Callable[[dict[str, Any], list[dict[str, Any]]], bool]


@dataclass
class EpisodeResult:
    task_id: str
    split: str
    config: str
    success: int
    n_steps: int
    wrong_tool: int
    schema_valid_calls: int
    prompt_tokens: int
    latency_ms: float
    failure_type: str
    unconfirmed_side_effects: int = 0
    executed: list[str] = field(default_factory=list)
    hit_budget: bool = False


def classify_failure(
    *,
    success: bool,
    schema_invalid: bool,
    looped: bool,
    wrong_tool: int,
    completed_steps: int,
    gold_steps: int,
    hit_budget: bool,
) -> str:
    if success:
        return "success"
    if schema_invalid:
        return "schema_invalid"
    if looped:
        return "loop"
    if wrong_tool > 0:
        return "wrong_tool"
    if completed_steps < gold_steps and not hit_budget:
        return "premature_stop"
    if hit_budget:
        return "budget_exceeded"
    return "bad_arguments"


def _history_line(prefix: str, call: ToolCall) -> str:
    payload = json.dumps(call.arguments, sort_keys=True)
    return f"{prefix} tool={call.name} args={payload}"


def run_episode(
    *,
    task_id: str,
    split: str,
    config: str,
    utterance: str,
    gold: list[tuple[str, dict[str, Any]]],
    expectations: list[dict[str, Any]],
    state: dict[str, Any],
    registry: ToolRegistry,
    select_rendered: Callable[[str], list[RenderedTool]],
    agent: Any,
    gate: PermissionGate,
    execute_fn: ExecuteFn,
    check_fn: CheckFn,
    compact: bool,
    context_limit: int,
    max_steps: int = 16,
) -> EpisodeResult:
    started = time.perf_counter()
    history: list[str] = []
    executed: list[ToolCall] = []
    n_steps = 0
    schema_valid_calls = 0
    schema_invalid = False
    prompt_tokens = 0
    unconfirmed = 0
    hit_budget = False
    clauses = split_clauses(utterance)
    budget = StepBudget(max_steps=max_steps)
    budget.start()
    seen_executions: set[tuple[str, str]] = set()
    looped = False

    while True:
        reason = budget.tick()
        if reason is not None:
            hit_budget = True
            break
        n_ok = sum(1 for line in history if line.startswith("OK "))
        if n_ok >= len(clauses) or check_fn(state, expectations):
            break
        clause = clauses[min(n_ok, len(clauses) - 1)] if clauses else ""
        packet: PromptPacket = assemble_prompt(
            utterance,
            select_rendered(clause),
            history,
            compact=compact,
            context_limit=context_limit,
        )
        prompt_tokens += packet.full_tokens
        response = agent.complete(packet.visible, clause, history, n_ok, len(clauses))
        call = call_from_openai(response)
        if call is None:
            break
        n_steps += 1
        if not validate_call(call, registry):
            schema_invalid = True
            history.append(_history_line("SCHEMA_INVALID", call))
            continue
        schema_valid_calls += 1
        tool = registry.get(call.name)
        assert tool is not None
        decision = gate.decide(tool, call)
        if not decision.allow:
            history.append(_history_line("CONFIRMATION_REQUIRED", call))
            continue
        if tool.side_effect and call.confirm is not True:
            # The gate should have blocked this. Count it if it ever executes.
            unconfirmed += 1
        result = execute_fn(state, tool, call.arguments)
        key = (call.name, json.dumps(call.arguments, sort_keys=True))
        if key in seen_executions:
            looped = True
        seen_executions.add(key)
        executed.append(call)
        history.append(_history_line("OK", call) + f" result={json.dumps(result, sort_keys=True)}")
        if looped:
            break

    success = check_fn(state, expectations)
    wrong = 0
    for index, call in enumerate(executed):
        gold_name = gold[index][0] if index < len(gold) else None
        if call.name != gold_name:
            wrong += 1
    failure = classify_failure(
        success=success,
        schema_invalid=schema_invalid,
        looped=looped,
        wrong_tool=wrong,
        completed_steps=len(executed),
        gold_steps=len(gold),
        hit_budget=hit_budget,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return EpisodeResult(
        task_id=task_id,
        split=split,
        config=config,
        success=int(success),
        n_steps=n_steps,
        wrong_tool=wrong,
        schema_valid_calls=schema_valid_calls,
        prompt_tokens=prompt_tokens,
        latency_ms=elapsed_ms,
        failure_type=failure,
        unconfirmed_side_effects=unconfirmed,
        executed=[call.name for call in executed],
        hit_budget=hit_budget,
    )
