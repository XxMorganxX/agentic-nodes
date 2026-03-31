from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class NodeCategory(str, Enum):
    START = "start"
    API = "api"
    PROVIDER = "provider"
    TOOL = "tool"
    DATA = "data"
    END = "end"


@dataclass
class ProviderConfigOptionDefinition:
    value: str
    label: str

    def to_dict(self) -> dict[str, str]:
        return {
            "value": self.value,
            "label": self.label,
        }


@dataclass
class ProviderConfigFieldDefinition:
    key: str
    label: str
    input_type: str = "text"
    help_text: str = ""
    placeholder: str = ""
    options: list[ProviderConfigOptionDefinition] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "label": self.label,
            "input_type": self.input_type,
            "help_text": self.help_text,
            "placeholder": self.placeholder,
            "options": [option.to_dict() for option in self.options],
        }


@dataclass
class NodeProviderDefinition:
    provider_id: str
    display_name: str
    category: NodeCategory
    node_kind: str
    description: str
    capabilities: list[str] = field(default_factory=list)
    model_provider_name: str | None = None
    default_config: dict[str, Any] = field(default_factory=dict)
    config_fields: list[ProviderConfigFieldDefinition] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["category"] = self.category.value
        payload["config_fields"] = [field.to_dict() for field in self.config_fields]
        return payload


@dataclass(frozen=True)
class CategoryContract:
    category: NodeCategory
    accepted_inputs: list[str]
    produced_outputs: list[str]
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category.value,
            "accepted_inputs": self.accepted_inputs,
            "produced_outputs": self.produced_outputs,
            "description": self.description,
        }


@dataclass(frozen=True)
class ConnectionRule:
    source_category: NodeCategory
    target_category: NodeCategory
    rationale: str

    def to_dict(self) -> dict[str, str]:
        return {
            "source_category": self.source_category.value,
            "target_category": self.target_category.value,
            "rationale": self.rationale,
        }


class NodeProviderRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, NodeProviderDefinition] = {}

    def register(self, provider: NodeProviderDefinition) -> None:
        if provider.provider_id in self._providers:
            raise ValueError(f"Node provider '{provider.provider_id}' is already registered.")
        self._providers[provider.provider_id] = provider

    def get(self, provider_id: str) -> NodeProviderDefinition:
        if provider_id not in self._providers:
            raise KeyError(f"Unknown node provider '{provider_id}'.")
        return self._providers[provider_id]

    def list_definitions(self) -> list[NodeProviderDefinition]:
        return list(self._providers.values())


DEFAULT_CATEGORY_CONTRACTS: dict[NodeCategory, CategoryContract] = {
    NodeCategory.START: CategoryContract(
        category=NodeCategory.START,
        accepted_inputs=["external_input"],
        produced_outputs=["message_envelope"],
        description="Captures a user or system supplied request and normalizes it into the shared graph envelope.",
    ),
    NodeCategory.API: CategoryContract(
        category=NodeCategory.API,
        accepted_inputs=["message_envelope", "tool_result_envelope", "data_envelope"],
        produced_outputs=["message_envelope", "tool_call_envelope"],
        description="Consumes a normalized envelope, can expose tools at prompt time, and emits either a response envelope or a tool call envelope.",
    ),
    NodeCategory.PROVIDER: CategoryContract(
        category=NodeCategory.PROVIDER,
        accepted_inputs=[],
        produced_outputs=["provider_binding"],
        description="Stores model provider configuration that can be visually bound into generic API nodes without becoming part of the main execution path.",
    ),
    NodeCategory.TOOL: CategoryContract(
        category=NodeCategory.TOOL,
        accepted_inputs=["tool_call_envelope", "message_envelope", "data_envelope"],
        produced_outputs=["tool_result_envelope"],
        description="Executes a registered tool using a validated payload and returns a structured tool result envelope.",
    ),
    NodeCategory.DATA: CategoryContract(
        category=NodeCategory.DATA,
        accepted_inputs=["message_envelope", "tool_result_envelope", "data_envelope"],
        produced_outputs=["data_envelope"],
        description="Transforms or enriches deterministic context while preserving the normalized graph envelope shape.",
    ),
    NodeCategory.END: CategoryContract(
        category=NodeCategory.END,
        accepted_inputs=["message_envelope", "tool_result_envelope", "data_envelope"],
        produced_outputs=["final_output"],
        description="Terminates a run and renders the final user-facing output from the latest envelope.",
    ),
}


DEFAULT_CONNECTION_RULES: list[ConnectionRule] = [
    ConnectionRule(NodeCategory.START, NodeCategory.API, "Input nodes can hand user requests to model providers."),
    ConnectionRule(NodeCategory.START, NodeCategory.DATA, "Input nodes can feed deterministic preprocessing steps."),
    ConnectionRule(NodeCategory.START, NodeCategory.TOOL, "Input nodes can route directly into tools when the input already satisfies a tool contract."),
    ConnectionRule(NodeCategory.START, NodeCategory.END, "Input nodes can terminate directly for passthrough or echo-style flows."),
    ConnectionRule(NodeCategory.PROVIDER, NodeCategory.API, "Provider nodes plug concrete vendor settings into model-agnostic API nodes."),
    ConnectionRule(NodeCategory.API, NodeCategory.API, "Model/API nodes can chain planning, routing, and synthesis steps."),
    ConnectionRule(NodeCategory.API, NodeCategory.TOOL, "Model/API nodes can route tool calls to executable tools."),
    ConnectionRule(NodeCategory.API, NodeCategory.DATA, "Model/API nodes can send outputs into deterministic processing steps."),
    ConnectionRule(NodeCategory.API, NodeCategory.END, "Model/API nodes can finalize a run directly."),
    ConnectionRule(NodeCategory.TOOL, NodeCategory.API, "Tool results commonly return to a model for reasoning or repair."),
    ConnectionRule(NodeCategory.TOOL, NodeCategory.DATA, "Tool results can be normalized or enriched by data nodes."),
    ConnectionRule(NodeCategory.TOOL, NodeCategory.END, "Tool results can finalize a run without another model step."),
    ConnectionRule(NodeCategory.DATA, NodeCategory.API, "Processed context can feed model/API nodes."),
    ConnectionRule(NodeCategory.DATA, NodeCategory.TOOL, "Deterministic transforms can prepare payloads for tools."),
    ConnectionRule(NodeCategory.DATA, NodeCategory.DATA, "Data nodes can compose deterministic transforms."),
    ConnectionRule(NodeCategory.DATA, NodeCategory.END, "Data nodes can finalize a run after deterministic formatting."),
]


def list_connection_rules() -> list[ConnectionRule]:
    return list(DEFAULT_CONNECTION_RULES)


def get_category_contract(category: NodeCategory) -> CategoryContract:
    return DEFAULT_CATEGORY_CONTRACTS[category]


def is_valid_category_connection(source_category: NodeCategory, target_category: NodeCategory) -> bool:
    return any(
        rule.source_category == source_category and rule.target_category == target_category
        for rule in DEFAULT_CONNECTION_RULES
    )
