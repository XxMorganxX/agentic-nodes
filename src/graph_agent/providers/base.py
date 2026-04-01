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
    response_mode: str = "auto",
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
            "message": final_payload_schema,
            "need_tool": {"type": "boolean"},
            "tool_calls": {
                "type": "array",
                "items": tool_call_item_schema,
            },
        },
        "required": ["message", "need_tool", "tool_calls"],
    }
    if not allow_tool_calls:
        schema["properties"]["need_tool"] = {"const": False}
        schema["properties"]["tool_calls"] = {
            "type": "array",
            "items": tool_call_item_schema,
            "maxItems": 0,
        }
    elif response_mode == "tool_call":
        schema["properties"]["need_tool"] = {"const": True}
        schema["properties"]["tool_calls"] = {
            "type": "array",
            "items": tool_call_item_schema,
            "minItems": 1,
        }
    elif response_mode == "message":
        schema["properties"]["need_tool"] = {"const": False}
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
    if mapping_output is not None:
        need_tool_value = mapping_output.get("need_tool")
        if not isinstance(need_tool_value, bool):
            legacy_should_call_tools = mapping_output.get("should_call_tools")
            need_tool_value = legacy_should_call_tools if isinstance(legacy_should_call_tools, bool) else None
        if isinstance(need_tool_value, bool):
            raw_message = mapping_output.get("message", mapping_output.get("final_message"))
            if (raw_message is None or raw_message == "") and content.strip():
                raw_message = content
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
                "message": raw_message,
                "need_tool": bool(need_tool_value),
                "tool_calls": normalized_structured_tool_calls,
            }
    if normalized_tool_calls:
        fallback_message = content if content.strip() else ""
        return {
            "message": fallback_message,
            "need_tool": True,
            "tool_calls": normalized_tool_calls,
        }
    final_message = structured_output if structured_output is not None else (content if content.strip() else None)
    return {
        "message": final_message,
        "need_tool": False,
        "tool_calls": [],
    }


def validate_api_decision_output(
    decision: Mapping[str, Any],
    *,
    callable_tool_names: set[str] | None = None,
    response_mode: str = "auto",
) -> dict[str, Any]:
    need_tool_value = decision.get("need_tool")
    if not isinstance(need_tool_value, bool):
        legacy_should_call_tools = decision.get("should_call_tools")
        need_tool_value = legacy_should_call_tools if isinstance(legacy_should_call_tools, bool) else None
    if not isinstance(need_tool_value, bool):
        raise ValueError("Structured API output must include boolean 'need_tool'.")
    should_call_tools = bool(need_tool_value)
    raw_tool_calls = decision.get("tool_calls", [])
    if not isinstance(raw_tool_calls, list):
        raise ValueError("Structured API output field 'tool_calls' must be a list.")
    message = decision.get("message", decision.get("final_message"))
    if message is None:
        raise ValueError("Structured API output must include 'message'.")
    normalized_tool_calls = [
        normalized
        for normalized in (_normalize_tool_call_entry(tool_call) for tool_call in raw_tool_calls)
        if normalized is not None
    ]
    if should_call_tools:
        if not normalized_tool_calls:
            raise ValueError("Structured API output requires at least one tool call when 'need_tool' is true.")
        if response_mode == "message":
            raise ValueError("Structured API output requires 'need_tool' to be false in message mode.")
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
            "message": message,
            "need_tool": True,
            "should_call_tools": True,
            "tool_calls": normalized_tool_calls,
            "final_message": message,
        }
    if normalized_tool_calls:
        raise ValueError("Structured API output must leave 'tool_calls' empty when 'need_tool' is false.")
    if response_mode == "tool_call":
        raise ValueError("Structured API output requires 'need_tool' to be true in tool_call mode.")
    return {
        "message": message,
        "need_tool": False,
        "should_call_tools": False,
        "tool_calls": [],
        "final_message": message,
    }


class ModelProvider(Protocol):
    name: str

    def generate(self, request: ModelRequest) -> ModelResponse:
        ...

    def preflight(self, provider_config: Mapping[str, Any] | None = None) -> ProviderPreflightResult:
        ...
