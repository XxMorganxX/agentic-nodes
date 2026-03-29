from __future__ import annotations

from graph_agent import config
from graph_agent.providers.mock import MockModelProvider
from graph_agent.providers.vendor_api import ClaudeMessagesModelProvider, OpenAIChatModelProvider
from graph_agent.runtime.core import GraphDefinition, RuntimeServices
from graph_agent.runtime.node_providers import NodeCategory, NodeProviderDefinition, NodeProviderRegistry
from graph_agent.tools.base import ToolRegistry
from graph_agent.tools.example_tools import build_search_catalog_tool


def build_example_services() -> RuntimeServices:
    registry = ToolRegistry()
    registry.register(build_search_catalog_tool())
    node_providers = NodeProviderRegistry()
    node_providers.register(
        NodeProviderDefinition(
            provider_id="core.input",
            display_name="Core Input Node",
            category=NodeCategory.START,
            node_kind="input",
            description="Captures the starting payload for a graph run.",
            capabilities=["accepts external input"],
        )
    )
    node_providers.register(
        NodeProviderDefinition(
            provider_id="core.api",
            display_name="API Call Node",
            category=NodeCategory.API,
            node_kind="model",
            description="Runs a model-agnostic API call step and selects its concrete provider from node configuration.",
            capabilities=["schema proposal", "schema repair", "response composition"],
        )
    )
    node_providers.register(
        NodeProviderDefinition(
            provider_id="provider.mock",
            display_name="Mock Provider",
            category=NodeCategory.PROVIDER,
            node_kind="provider",
            description="Provides mock-model settings to a generic API call node.",
            capabilities=["local test provider", "schema proposal", "response composition"],
        )
    )
    node_providers.register(
        NodeProviderDefinition(
            provider_id="provider.openai",
            display_name="OpenAI Provider",
            category=NodeCategory.PROVIDER,
            node_kind="provider",
            description="Provides OpenAI chat-completions settings to a generic API call node.",
            capabilities=["structured output", "tool-schema generation", "response composition"],
        )
    )
    node_providers.register(
        NodeProviderDefinition(
            provider_id="provider.claude",
            display_name="Claude Provider",
            category=NodeCategory.PROVIDER,
            node_kind="provider",
            description="Provides Anthropic Claude messages settings to a generic API call node.",
            capabilities=["structured output", "tool-schema generation", "response composition"],
        )
    )
    node_providers.register(
        NodeProviderDefinition(
            provider_id="tool.registry",
            display_name="Registry Tool Node",
            category=NodeCategory.TOOL,
            node_kind="tool",
            description="Dispatches a registered tool and surfaces validation failures as routable results.",
            capabilities=["tool dispatch", "schema validation"],
        )
    )
    node_providers.register(
        NodeProviderDefinition(
            provider_id="core.output",
            display_name="Core Output Node",
            category=NodeCategory.END,
            node_kind="output",
            description="Returns the terminal response for the run.",
            capabilities=["final output"],
        )
    )
    return RuntimeServices(
        model_providers={
            "claude": ClaudeMessagesModelProvider(),
            "mock": MockModelProvider(),
            "openai": OpenAIChatModelProvider(),
        },
        node_provider_registry=node_providers,
        tool_registry=registry,
        config={
            "max_steps": config.DEFAULT_RUN_MAX_STEPS,
            "max_visits_per_node": config.DEFAULT_MAX_VISITS_PER_NODE,
            "max_repair_attempts": config.DEFAULT_MAX_REPAIR_ATTEMPTS,
        },
    )


def build_example_graph_payload() -> dict[str, object]:
    return {
        "graph_id": config.DEFAULT_GRAPH_ID,
        "name": "Tool Schema Repair Example",
        "description": "Demonstrates a model -> tool -> repair -> model loop using the shared graph envelope contract.",
        "version": "1.0",
        "start_node_id": "start",
        "nodes": [
            {
                "id": "start",
                "kind": "input",
                "category": "start",
                "label": "Start Input",
                "provider_id": "core.input",
                "provider_label": "Core Input Node",
                "description": "Captures the starting payload for a graph run.",
                "position": {"x": 120, "y": 120},
                "config": {
                    "input_binding": {"type": "input_payload"},
                },
            },
            {
                "id": "propose_tool",
                "kind": "model",
                "category": "api",
                "label": "Propose Tool Payload",
                "provider_id": "core.api",
                "provider_label": "API Call Node",
                "description": "Creates the first tool payload attempt and routes it to the tool node.",
                "model_provider_name": "mock",
                "prompt_name": "schema_proposal",
                "position": {"x": 420, "y": 120},
                "config": {
                    "provider_name": "mock",
                    "model": "mock-default",
                    "prompt_name": "schema_proposal",
                    "mode": "schema_proposal",
                    "system_prompt": config.SCHEMA_PROPOSAL_PROMPT,
                    "user_message_template": "Request: {user_request}\\nAvailable tools: {available_tools}",
                    "response_mode": "tool_call",
                    "preferred_tool_name": "search_catalog",
                    "allowed_tool_names": ["search_catalog"],
                    "tool_target_node_ids": ["run_tool"],
                    "metadata_bindings": {
                        "user_request": {"type": "latest_payload", "source": "start"},
                    },
                },
            },
            {
                "id": "run_tool",
                "kind": "tool",
                "category": "tool",
                "label": "Run Tool",
                "provider_id": "tool.registry",
                "provider_label": "Registry Tool Node",
                "description": "Validates and runs the selected tool.",
                "tool_name": "search_catalog",
                "position": {"x": 760, "y": 300},
                "config": {
                    "tool_name": "search_catalog",
                    "input_binding": {
                        "type": "first_available_envelope",
                        "sources": ["repair_tool", "propose_tool"],
                    },
                },
            },
            {
                "id": "repair_tool",
                "kind": "model",
                "category": "api",
                "label": "Repair Tool Payload",
                "provider_id": "core.api",
                "provider_label": "API Call Node",
                "description": "Repairs the tool payload after validation failure.",
                "model_provider_name": "mock",
                "prompt_name": "schema_repair",
                "position": {"x": 420, "y": 420},
                "config": {
                    "provider_name": "mock",
                    "model": "mock-default",
                    "prompt_name": "schema_repair",
                    "mode": "schema_repair",
                    "system_prompt": config.SCHEMA_REPAIR_PROMPT,
                    "user_message_template": "Request: {user_request}\\nValidation error: {tool_error}\\nAvailable tools: {available_tools}",
                    "response_mode": "tool_call",
                    "preferred_tool_name": "search_catalog",
                    "allowed_tool_names": ["search_catalog"],
                    "tool_target_node_ids": ["run_tool"],
                    "metadata_bindings": {
                        "user_request": {"type": "latest_payload", "source": "start"},
                        "tool_error": {"type": "latest_error", "source": "run_tool"},
                    },
                },
            },
            {
                "id": "compose_response",
                "kind": "model",
                "category": "api",
                "label": "Compose Final Response",
                "provider_id": "core.api",
                "provider_label": "API Call Node",
                "description": "Builds the response shown to the user.",
                "model_provider_name": "mock",
                "prompt_name": "final_response",
                "position": {"x": 1080, "y": 120},
                "config": {
                    "provider_name": "mock",
                    "model": "mock-default",
                    "prompt_name": "final_response",
                    "mode": "final_response",
                    "system_prompt": config.FINAL_RESPONSE_PROMPT,
                    "user_message_template": "Request: {user_request}\\nTool result: {tool_result}",
                    "response_mode": "message",
                    "metadata_bindings": {
                        "user_request": {"type": "latest_payload", "source": "start"},
                        "tool_result": {"type": "latest_payload", "source": "run_tool"},
                    },
                },
            },
            {
                "id": "finish",
                "kind": "output",
                "category": "end",
                "label": "Output Response",
                "provider_id": "core.output",
                "provider_label": "Core Output Node",
                "description": "Returns the final response payload.",
                "position": {"x": 1380, "y": 120},
                "config": {
                    "source_binding": {"type": "latest_envelope", "source": "compose_response"},
                },
            },
        ],
        "edges": [
            {
                "id": "edge-start-propose",
                "source_id": "start",
                "target_id": "propose_tool",
                "label": "begin",
                "kind": "standard",
                "priority": 100,
                "condition": None,
            },
            {
                "id": "edge-propose-run",
                "source_id": "propose_tool",
                "target_id": "run_tool",
                "label": "try tool",
                "kind": "standard",
                "priority": 100,
                "condition": None,
            },
            {
                "id": "edge-run-repair",
                "source_id": "run_tool",
                "target_id": "repair_tool",
                "label": "repair invalid schema",
                "kind": "conditional",
                "priority": 10,
                "condition": {
                    "id": "tool_validation_failed",
                    "label": "Tool validation failed",
                    "type": "result_status_equals",
                    "value": "validation_error",
                },
            },
            {
                "id": "edge-run-compose",
                "source_id": "run_tool",
                "target_id": "compose_response",
                "label": "tool succeeded",
                "kind": "standard",
                "priority": 100,
                "condition": None,
            },
            {
                "id": "edge-repair-run",
                "source_id": "repair_tool",
                "target_id": "run_tool",
                "label": "retry tool",
                "kind": "standard",
                "priority": 100,
                "condition": None,
            },
            {
                "id": "edge-compose-finish",
                "source_id": "compose_response",
                "target_id": "finish",
                "label": "finalize",
                "kind": "standard",
                "priority": 100,
                "condition": None,
            },
        ],
    }


def build_example_graph() -> GraphDefinition:
    return GraphDefinition.from_dict(build_example_graph_payload())
