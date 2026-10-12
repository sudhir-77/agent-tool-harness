"""JSON-schema validation for tool calls."""

from __future__ import annotations

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from harness.registry import ToolCall, ToolRegistry


def validate_call(call: ToolCall, registry: ToolRegistry) -> bool:
    tool = registry.get(call.name)
    if tool is None:
        return False
    if not isinstance(call.arguments, dict):
        return False
    try:
        validator = Draft202012Validator(tool.parameters)
    except SchemaError:
        return False
    return validator.is_valid(call.arguments)
