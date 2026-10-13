"""Seeded task templates. Dev and test template ids are disjoint by construction.

Split rule (fixed, not tuned): even template indexes are dev, odd indexes are test.
Slot values change with the seed. The test templates are never read by retrieval
fitting or by the selection rule's hyperparameters.
"""

from __future__ import annotations

import random
from typing import Any

from apps.catalog import APPS, caller_text
from harness.llm import phrase_in_text, split_clauses
from harness.registry import ToolSpec

SEED = 0
INSTANCES_PER_TEMPLATE = 5

NOTES = ["alpha detail", "bravo detail", "cedar detail", "delta detail"]
STATUSES = ["open", "done", "pending"]
TAGS = ["home", "work", "urgent"]


def _sample_value(field: str, rng: random.Random, title: str) -> Any:
    if field == "title":
        return title
    if field == "note":
        return rng.choice(NOTES)
    if field == "date":
        return f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
    if field == "time":
        return f"{rng.randint(8, 17):02d}:{rng.choice(['00', '30'])}"
    if field == "duration_minutes":
        return int(rng.choice([15, 30, 45, 60]))
    if field == "amount":
        return int(rng.randint(10, 80))
    if field == "quantity":
        return int(rng.randint(1, 5))
    if field == "status":
        return rng.choice(STATUSES)
    if field == "tag":
        return rng.choice(TAGS)
    raise KeyError(field)


def _render_value(field: str, value: Any) -> str:
    if isinstance(value, int) and not isinstance(value, bool):
        return f"{field}={value}"
    return f"{field}='{value}'"


def render_clause(phrase: str, arguments: dict[str, Any], fields: list[str]) -> str:
    parts = [phrase]
    for field in fields:
        parts.append(_render_value(field, arguments[field]))
    return " ".join(parts)


def render_caller_clause(tool: ToolSpec, arguments: dict[str, Any]) -> str:
    """Paraphrase used on test templates. Slot values stay in key=value form."""
    noun = dict(APPS)[tool.app]
    parts = [caller_text(tool.app, noun, tool.action)]
    for field in tool.parameters["required"]:
        parts.append(_render_value(field, arguments[field]))
    clause = " ".join(parts)
    if phrase_in_text(tool.phrase, clause) or tool.name in clause:
        raise RuntimeError(f"test clause mentions {tool.name}")
    return clause


def paraphrase_task(task: dict[str, Any], by_name: dict[str, ToolSpec]) -> dict[str, Any]:
    """Same plan and split, with every step rewritten in caller wording."""
    clauses = [render_caller_clause(by_name[step["tool"]], dict(step["arguments"])) for step in task["plan"]]
    copied = dict(task)
    copied["utterance"] = f"User request: {' THEN '.join(clauses)}."
    return copied


def template_specs() -> list[list[str]]:
    """80 templates. Lengths range from 1 to 6 tool calls."""
    apps = [
        "calendar",
        "notes",
        "messages",
        "files",
        "settings",
        "shopping",
        "mail",
        "contacts",
        "tasks",
        "weather",
        "navigation",
        "banking",
    ]
    domain_action = {
        "calendar": "schedule_meeting",
        "notes": "add_notebook",
        "messages": "send_message",
        "files": "upload_file",
        "settings": "set_preference",
        "shopping": "add_cart_item",
        "mail": "send_email",
        "contacts": "add_contact",
        "tasks": "add_task",
        "weather": "get_forecast",
        "navigation": "plan_route",
        "banking": "transfer_funds",
    }
    specs: list[list[str]] = []
    for app in apps:
        specs.append([f"{app}.create"])
        specs.append([f"{app}.{domain_action[app]}"])
        specs.append([f"{app}.share"])
    for app in apps:
        specs.append([f"{app}.create", f"{app}.list"])
    for app in apps:
        specs.append([f"{app}.create", f"{app}.share"])
    for app in apps[:8]:
        specs.append([f"{app}.create", f"{app}.tag", f"{app}.star"])
    for app in apps[:6]:
        specs.append([f"{app}.create", f"{app}.update", f"{app}.tag", f"{app}.export"])
    for app in apps[:4]:
        specs.append([f"{app}.create", f"{app}.update", f"{app}.tag", f"{app}.star", f"{app}.export"])
    for app in apps[:2]:
        specs.append(
            [
                f"{app}.create",
                f"{app}.update",
                f"{app}.tag",
                f"{app}.star",
                f"{app}.pin",
                f"{app}.export",
            ]
        )
    if len(specs) != 80:
        raise RuntimeError(f"expected 80 templates, got {len(specs)}")
    return specs


def build_tasks(tools: list[ToolSpec], seed: int = SEED) -> list[dict[str, Any]]:
    by_name = {tool.name: tool for tool in tools}
    tasks: list[dict[str, Any]] = []
    for index, spec in enumerate(template_specs()):
        split = "dev" if index % 2 == 0 else "test"
        template_id = f"tpl{index:03d}"
        missing = [name for name in spec if name not in by_name]
        if missing:
            raise RuntimeError(f"unknown tools in template {template_id}: {missing}")
        for instance in range(INSTANCES_PER_TEMPLATE):
            rng = random.Random(seed + index * 17 + instance * 1009)
            title = f"Case {rng.randint(1000, 9999)}"
            plan = []
            clauses = []
            expectations = []
            for tool_name in spec:
                tool = by_name[tool_name]
                fields = list(tool.parameters["required"])
                arguments = {field: _sample_value(field, rng, title) for field in fields}
                plan.append({"tool": tool.name, "arguments": arguments})
                if split == "test":
                    clauses.append(render_caller_clause(tool, arguments))
                else:
                    clauses.append(render_clause(tool.phrase, arguments, fields))
                expectations.append(
                    {
                        "app": tool.app,
                        "fields": {"tool": tool.name, "action": tool.action, **arguments},
                    }
                )
            body = " THEN ".join(clauses)
            if split == "dev":
                utterance = f"Please {body}."
            else:
                utterance = f"User request: {body}."
            # Round-trip so retrieval queries stay aligned with plan steps.
            if split_clauses(utterance) != clauses:
                raise RuntimeError(f"clause split failed for {template_id}")
            tasks.append(
                {
                    "task_id": f"{template_id}-i{instance}",
                    "template_id": template_id,
                    "split": split,
                    "utterance": utterance,
                    "plan": plan,
                    "expectations": expectations,
                    "n_steps": len(plan),
                    "seed": seed,
                }
            )
    return tasks


def split_tasks(tasks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dev = [task for task in tasks if task["split"] == "dev"]
    test = [task for task in tasks if task["split"] == "test"]
    return dev, test
