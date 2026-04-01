from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ModelMessage:
    role: str
    content: str


@dataclass
class ModelToolDefinition:
    name: str
    description: str
    input_schema: Mapping[str, Any]


@dataclass
class ModelToolCall:
    tool_name: str
    arguments: Any
    provider_tool_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderPreflightResult:
    status: str
    ok: bool
    message: str = ""
    warnings: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelRequest:
    prompt_name: str
    messages: list[ModelMessage]
    response_schema: Mapping[str, Any] | None = None
    provider_config: Mapping[str, Any] | None = None
    available_tools: list[ModelToolDefinition] = field(default_factory=list)
    preferred_tool_name: str | None = None
    response_mode: str = "message"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelResponse:
    content: str
    structured_output: Any = None
    tool_calls: list[ModelToolCall] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def _as_mapping_dict(value: Any) -> dict[str, Any] | None:
    if isinstance(value, Mapping):
        return dict(value)
    return None


def _normalize_tool_call_entry(value: Any) -> dict[str, Any] | None:
    if isinstance(value, ModelToolCall):
        return {
            "tool_name": value.tool_name,
            "arguments": value.arguments,
            "provider_tool_id": value.provider_tool_id,
            "metadata": dict(value.metadata),
        }
    mapping_value = _as_mapping_dict(value)
    if mapping_value is None:
        return None
    tool_name = mapping_value.get("tool_name")
    if not isinstance(tool_name, str) or not tool_name.strip():
        return None
    metadata = mapping_value.get("metadata")
    normalized_metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
    provider_tool_id = mapping_value.get("provider_tool_id")
    return {
        "tool_name": tool_name.strip(),
        "arguments": mapping_value.get("arguments"),
        "provider_tool_id": str(provider_tool_id) if provider_tool_id else None,
        "metadata": normalized_metadata,
    }


def api_decision_response_schema(
    *,
    final_message_schema: Mapping[str, Any] | None = None,
    available_tools: Sequence[ModelToolDefinition] | None = None,
    allow_tool_calls: bool = True,
) -> dict[str, Any]:
    tool_names = [tool.name for tool in (available_tools or []) if isinstance(tool.name, str) and tool.name.strip()]
    tool_call_item_schema: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "tool_name": {"type": "string"},
            "arguments": {"type": "object", "additionalProperties": True},
        },
        "required": ["tool_name", "arguments"],
    }
    if tool_names:
        tool_call_item_schema["properties"]["tool_name"] = {"type": "string", "enum": tool_names}

    final_payload_schema: dict[str, Any]
    if isinstance(final_message_schema, Mapping):
        final_payload_schema = dict(final_message_schema)
    else:
        final_payload_schema = {
            "type": ["string", "object", "array", "number", "boolean", "null"],
        }

    schema: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "should_call_tools": {"type": "boolean"},
            "tool_calls": {
                "type": "array",
                "items": tool_call_item_schema,
            },
            "final_message": final_payload_schema,
        },
        "required": ["should_call_tools", "tool_calls", "final_message"],
    }
    if not allow_tool_calls:
        schema["properties"]["should_call_tools"] = {"const": False}
        schema["properties"]["tool_calls"] = {
            "type": "array",
            "items": tool_call_item_schema,
            "maxItems": 0,
        }
    return schema


def normalize_api_decision_output(
    structured_output: Any,
    *,
    content: str = "",
    tool_calls: Sequence[ModelToolCall | Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    normalized_tool_calls = [
        normalized
        for normalized in (_normalize_tool_call_entry(tool_call) for tool_call in (tool_calls or []))
        if normalized is not None
    ]
    mapping_output = _as_mapping_dict(structured_output)
    if mapping_output is not None and isinstance(mapping_output.get("should_call_tools"), bool):
        raw_tool_calls = mapping_output.get("tool_calls", [])
        normalized_structured_tool_calls = (
            [
                normalized
                for normalized in (_normalize_tool_call_entry(tool_call) for tool_call in raw_tool_calls)
                if normalized is not None
            ]
            if isinstance(raw_tool_calls, list)
            else []
        )
        return {
            "should_call_tools": bool(mapping_output.get("should_call_tools")),
            "tool_calls": normalized_structured_tool_calls,
            "final_message": mapping_output.get("final_message"),
        }
    if normalized_tool_calls:
        return {
            "should_call_tools": True,
            "tool_calls": normalized_tool_calls,
            "final_message": None,
        }
    final_message = structured_output if structured_output is not None else (content if content.strip() else None)
    return {
        "should_call_tools": False,
        "tool_calls": [],
        "final_message": final_message,
    }


def validate_api_decision_output(
    decision: Mapping[str, Any],
    *,
    callable_tool_names: set[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(decision.get("should_call_tools"), bool):
        raise ValueError("Structured API output must include boolean 'should_call_tools'.")
    should_call_tools = bool(decision["should_call_tools"])
    raw_tool_calls = decision.get("tool_calls", [])
    if not isinstance(raw_tool_calls, list):
        raise ValueError("Structured API output field 'tool_calls' must be a list.")
    normalized_tool_calls = [
        normalized
        for normalized in (_normalize_tool_call_entry(tool_call) for tool_call in raw_tool_calls)
        if normalized is not None
    ]
    if should_call_tools:
        if not normalized_tool_calls:
            raise ValueError("Structured API output requires at least one tool call when 'should_call_tools' is true.")
        if callable_tool_names:
            unknown_tool_names = sorted(
                {
                    str(tool_call["tool_name"])
                    for tool_call in normalized_tool_calls
                    if str(tool_call["tool_name"]) not in callable_tool_names
                }
            )
            if unknown_tool_names:
                joined = ", ".join(unknown_tool_names)
                raise ValueError(f"Structured API output requested unavailable tool(s): {joined}.")
        return {
            "should_call_tools": True,
            "tool_calls": normalized_tool_calls,
            "final_message": None,
        }
    if normalized_tool_calls:
        raise ValueError("Structured API output must leave 'tool_calls' empty when 'should_call_tools' is false.")
    if "final_message" not in decision or decision.get("final_message") is None:
        raise ValueError("Structured API output requires 'final_message' when no tool calls are requested.")
    return {
        "should_call_tools": False,
        "tool_calls": [],
        "final_message": decision.get("final_message"),
    }


class ModelProvider(Protocol):
    name: str

    def generate(self, request: ModelRequest) -> ModelResponse:
        ...

    def preflight(self, provider_config: Mapping[str, Any] | None = None) -> ProviderPreflightResult:
        ...
