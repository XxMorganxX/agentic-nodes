from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol


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


class ModelProvider(Protocol):
    name: str

    def generate(self, request: ModelRequest) -> ModelResponse:
        ...

    def preflight(self, provider_config: Mapping[str, Any] | None = None) -> ProviderPreflightResult:
        ...
