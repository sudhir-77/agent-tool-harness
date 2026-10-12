"""Typed tool schemas and the in-memory registry."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


INT_FIELDS = frozenset({"duration_minutes", "amount", "quantity"})


class ToolSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    app: str
    action: str
    phrase: str
    description: str
    side_effect: bool
    parameters: dict[str, Any]
    retrieval_text: str


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    confirm: bool = False


class ToolRegistry:
    def __init__(self, tools: list[ToolSpec]) -> None:
        names = [tool.name for tool in tools]
        if len(names) != len(set(names)):
            raise ValueError("duplicate tool name")
        self.tools = list(tools)
        self.by_name = {tool.name: tool for tool in tools}

    def __len__(self) -> int:
        return len(self.tools)

    def get(self, name: str) -> ToolSpec | None:
        return self.by_name.get(name)


def schema_for_fields(fields: list[str]) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    for field in fields:
        if field in INT_FIELDS:
            properties[field] = {"type": "integer"}
        else:
            properties[field] = {"type": "string"}
    return {
        "type": "object",
        "properties": properties,
        "required": list(fields),
        "additionalProperties": False,
    }


def render_tool_block(tool: ToolSpec) -> str:
    schema = json.dumps(tool.parameters, sort_keys=True, separators=(",", ":"))
    side = "true" if tool.side_effect else "false"
    return (
        f"TOOL {tool.name}\n"
        f"description: {tool.description}\n"
        f"phrase: {tool.phrase}\n"
        f"side_effect: {side}\n"
        f"schema: {schema}\n"
        f"END {tool.name}\n"
    )
