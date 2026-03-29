from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping


@dataclass
class ToolContext:
    run_id: str
    graph_id: str
    node_id: str
    state_snapshot: Mapping[str, Any]


@dataclass
class ToolResult:
    status: str
    output: Any = None
    error: dict[str, Any] | None = None
    summary: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolDefinition:
    name: str
    description: str
    input_schema: Mapping[str, Any]
    executor: Callable[[Mapping[str, Any], ToolContext], ToolResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": dict(self.input_schema),
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered.")
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolDefinition:
        if name not in self._tools:
            raise KeyError(f"Unknown tool '{name}'.")
        return self._tools[name]

    def list_definitions(self) -> list[ToolDefinition]:
        return list(self._tools.values())

    def validate_input(self, schema: Mapping[str, Any], payload: Mapping[str, Any]) -> dict[str, Any] | None:
        if schema.get("type") != "object":
            return {"message": "Only object schemas are supported in v1."}

        required = schema.get("required", [])
        properties = schema.get("properties", {})

        missing = [field for field in required if field not in payload]
        if missing:
            return {
                "message": "Missing required fields.",
                "missing_fields": missing,
            }

        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "object": dict,
            "array": list,
        }

        type_errors: list[dict[str, str]] = []
        for key, value in payload.items():
            if key not in properties:
                continue
            expected_type = properties[key].get("type")
            if expected_type is None:
                continue
            python_type = type_map.get(expected_type)
            if python_type is None:
                continue
            if not isinstance(value, python_type):
                type_errors.append(
                    {
                        "field": key,
                        "expected": expected_type,
                        "received": type(value).__name__,
                    }
                )

        if type_errors:
            return {
                "message": "One or more fields have invalid types.",
                "type_errors": type_errors,
            }

        return None

    def invoke(self, name: str, payload: Mapping[str, Any], context: ToolContext) -> ToolResult:
        tool = self.get(name)
        validation_error = self.validate_input(tool.input_schema, payload)
        if validation_error is not None:
            return ToolResult(
                status="validation_error",
                error=validation_error,
                summary=f"Tool '{name}' rejected the payload.",
            )
        return tool.executor(payload, context)
