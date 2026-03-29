from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol


@dataclass
class ModelMessage:
    role: str
    content: str


@dataclass
class ModelRequest:
    prompt_name: str
    messages: list[ModelMessage]
    response_schema: Mapping[str, Any] | None = None
    provider_config: Mapping[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelResponse:
    content: str
    structured_output: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelProvider(Protocol):
    name: str

    def generate(self, request: ModelRequest) -> ModelResponse:
        ...
