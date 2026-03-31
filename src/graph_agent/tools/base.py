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
    executor: Callable[[Mapping[str, Any], ToolContext], ToolResult] | None = None
    source_type: str = "builtin"
    server_id: str | None = None
    enabled: bool = True
    available: bool = True
    availability_error: str = ""
    schema_origin: str = "static"
    schema_warning: str = ""
    managed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": dict(self.input_schema),
            "source_type": self.source_type,
            "server_id": self.server_id,
            "enabled": self.enabled,
            "available": self.available,
            "availability_error": self.availability_error,
            "schema_origin": self.schema_origin,
            "schema_warning": self.schema_warning,
            "managed": self.managed,
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered.")
        self._tools[tool.name] = tool

    def upsert(self, tool: ToolDefinition) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolDefinition:
        if name not in self._tools:
            raise KeyError(f"Unknown tool '{name}'.")
        return self._tools[name]

    def get_optional(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def list_definitions(self) -> list[ToolDefinition]:
        return list(self._tools.values())

    def list_server_tool_names(self, server_id: str) -> list[str]:
        return sorted(tool.name for tool in self._tools.values() if tool.server_id == server_id)

    def exposable_definitions(self, names: list[str]) -> list[ToolDefinition]:
        definitions: list[ToolDefinition] = []
        for name in names:
            tool = self.get_optional(str(name))
            if tool is None or not tool.enabled or not tool.available:
                continue
            definitions.append(tool)
        return definitions

    def require_exposable(self, name: str) -> ToolDefinition:
        tool = self.get(name)
        if not tool.enabled:
            raise ValueError(f"Tool '{name}' is disabled.")
        if not tool.available:
            detail = f" {tool.availability_error}" if tool.availability_error else ""
            raise ValueError(f"Tool '{name}' is unavailable.{detail}")
        return tool

    def require_invocable(self, name: str) -> ToolDefinition:
        tool = self.require_exposable(name)
        if tool.executor is None:
            raise ValueError(f"Tool '{name}' is not executable.")
        return tool

    def set_tool_enabled(self, name: str, enabled: bool) -> ToolDefinition:
        tool = self.get(name)
        updated = ToolDefinition(
            name=tool.name,
            description=tool.description,
            input_schema=tool.input_schema,
            executor=tool.executor,
            source_type=tool.source_type,
            server_id=tool.server_id,
            enabled=enabled,
            available=tool.available,
            availability_error=tool.availability_error,
            schema_origin=tool.schema_origin,
            schema_warning=tool.schema_warning,
            managed=tool.managed,
        )
        self._tools[name] = updated
        return updated

    def mark_tool_unavailable(self, name: str, reason: str) -> None:
        tool = self.get_optional(name)
        if tool is None:
            return
        self._tools[name] = ToolDefinition(
            name=tool.name,
            description=tool.description,
            input_schema=tool.input_schema,
            executor=tool.executor,
            source_type=tool.source_type,
            server_id=tool.server_id,
            enabled=tool.enabled,
            available=False,
            availability_error=reason,
            schema_origin=tool.schema_origin,
            schema_warning=tool.schema_warning,
            managed=tool.managed,
        )

    def mark_server_tools_unavailable(self, server_id: str, reason: str) -> None:
        for tool_name in self.list_server_tool_names(server_id):
            self.mark_tool_unavailable(tool_name, reason)

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
        if not tool.enabled:
            return ToolResult(
                status="unavailable",
                error={"message": f"Tool '{name}' is disabled."},
                summary=f"Tool '{name}' is disabled.",
            )
        if not tool.available or tool.executor is None:
            detail = tool.availability_error or "Tool is unavailable."
            return ToolResult(
                status="unavailable",
                error={"message": detail},
                summary=f"Tool '{name}' is unavailable.",
            )
        validation_error = self.validate_input(tool.input_schema, payload)
        if validation_error is not None:
            return ToolResult(
                status="validation_error",
                error=validation_error,
                summary=f"Tool '{name}' rejected the payload.",
            )
        return tool.executor(payload, context)
