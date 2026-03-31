from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import os
import re
from typing import Any, Mapping, Sequence
from uuid import uuid4

from graph_agent.providers.base import ModelMessage, ModelProvider, ModelRequest, ModelToolDefinition
from graph_agent.runtime.node_providers import (
    NodeCategory,
    NodeProviderRegistry,
    get_category_contract,
    is_valid_category_connection,
)
from graph_agent.tools.base import ToolContext, ToolRegistry
from graph_agent.tools.mcp import McpServerManager


def utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


API_TOOL_CALL_HANDLE_ID = "api-tool-call"
API_MESSAGE_HANDLE_ID = "api-message"


def _json_safe(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=2, sort_keys=True)


def _deep_get(value: Any, path: str | None) -> Any:
    if path in {None, "", "$"}:
        return value
    current = value
    for segment in path.split("."):
        if isinstance(current, Mapping):
            current = current.get(segment)
        elif isinstance(current, list) and segment.isdigit():
            index = int(segment)
            current = current[index] if index < len(current) else None
        else:
            return None
    return current


class SafeFormatDict(dict[str, Any]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


DEFAULT_GRAPH_ENV_VARS = {
    "OPENAI_API_KEY": "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY": "ANTHROPIC_API_KEY",
    "DISCORD_BOT_TOKEN": "DISCORD_BOT_TOKEN",
}

GRAPH_ENV_REFERENCE_PATTERN = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _normalize_graph_env_vars(payload: Mapping[str, Any] | None) -> dict[str, str]:
    env_vars = dict(DEFAULT_GRAPH_ENV_VARS)
    if not isinstance(payload, Mapping):
        return env_vars
    for key, value in payload.items():
        normalized_key = str(key).strip()
        if not normalized_key:
            continue
        env_vars[normalized_key] = str(value if value is not None else "")
    return env_vars


def _resolve_graph_env_string(value: str, env_vars: Mapping[str, str]) -> str:
    return GRAPH_ENV_REFERENCE_PATTERN.sub(lambda match: env_vars.get(match.group(1), match.group(0)), value)


def resolve_graph_env_value(value: Any, env_vars: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        return _resolve_graph_env_string(value, env_vars)
    if isinstance(value, Mapping):
        return {str(key): resolve_graph_env_value(child, env_vars) for key, child in value.items()}
    if isinstance(value, list):
        return [resolve_graph_env_value(item, env_vars) for item in value]
    return value


def resolve_graph_env_var_name(value: str, env_vars: Mapping[str, str]) -> str:
    return str(resolve_graph_env_value(value, env_vars)).strip()


def resolve_graph_process_env(value: str, env_vars: Mapping[str, str]) -> str:
    env_var_name = resolve_graph_env_var_name(value, env_vars)
    if not env_var_name:
        return ""
    return os.environ.get(env_var_name, "")


@dataclass
class MessageEnvelope:
    schema_version: str
    from_node_id: str
    from_category: str
    payload: Any
    artifacts: dict[str, Any] = field(default_factory=dict)
    errors: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MessageEnvelope:
        return cls(
            schema_version=str(payload.get("schema_version", "1.0")),
            from_node_id=str(payload.get("from_node_id", "")),
            from_category=str(payload.get("from_category", "")),
            payload=payload.get("payload"),
            artifacts=dict(payload.get("artifacts", {})),
            errors=list(payload.get("errors", [])),
            tool_calls=list(payload.get("tool_calls", [])),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass
class NodeExecutionResult:
    status: str
    output: Any = None
    error: dict[str, Any] | None = None
    summary: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RuntimeEvent:
    event_type: str
    summary: str
    payload: dict[str, Any]
    run_id: str
    agent_id: str | None = None
    parent_run_id: str | None = None
    timestamp: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Condition:
    id: str
    label: str
    condition_type: str
    value: Any = None
    path: str | None = None

    def evaluate(self, state: RunState, result: NodeExecutionResult) -> bool:
        if self.condition_type == "result_status_equals":
            return result.status == self.value
        if self.condition_type == "result_has_error":
            return result.error is not None
        if self.condition_type == "result_payload_path_equals":
            return _deep_get(result.output, self.path) == self.value
        if self.condition_type == "state_error_path_equals":
            return _deep_get(state.node_errors, self.path) == self.value
        raise ValueError(f"Unsupported condition type '{self.condition_type}'.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "type": self.condition_type,
            "value": self.value,
            "path": self.path,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Condition:
        return cls(
            id=str(payload["id"]),
            label=str(payload.get("label", payload["id"])),
            condition_type=str(payload.get("type", "result_status_equals")),
            value=payload.get("value"),
            path=payload.get("path"),
        )


def _is_tool_call_contract_condition(condition: Condition | None) -> bool:
    return bool(
        condition is not None
        and condition.condition_type == "result_payload_path_equals"
        and condition.path == "metadata.contract"
        and condition.value == "tool_call_envelope"
    )


def _is_message_contract_condition(condition: Condition | None) -> bool:
    return bool(
        condition is not None
        and condition.condition_type == "result_payload_path_equals"
        and condition.path == "metadata.contract"
        and condition.value == "message_envelope"
    )


@dataclass
class Edge:
    id: str
    source_id: str
    target_id: str
    source_handle_id: str | None = None
    target_handle_id: str | None = None
    label: str = ""
    kind: str = "standard"
    priority: int = 100
    waypoints: list[dict[str, float]] = field(default_factory=list)
    condition: Condition | None = None

    def is_match(self, state: RunState, result: NodeExecutionResult) -> bool:
        if self.kind != "conditional":
            return True
        if self.condition is None:
            raise ValueError(f"Conditional edge '{self.id}' is missing a condition.")
        return self.condition.evaluate(state, result)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "source_handle_id": self.source_handle_id,
            "target_handle_id": self.target_handle_id,
            "label": self.label,
            "kind": self.kind,
            "priority": self.priority,
            "waypoints": self.waypoints,
            "condition": None if self.condition is None else self.condition.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Edge:
        condition = payload.get("condition")
        raw_waypoints = payload.get("waypoints", [])
        waypoints = []
        if isinstance(raw_waypoints, Sequence) and not isinstance(raw_waypoints, (str, bytes)):
            for waypoint in raw_waypoints:
                if isinstance(waypoint, Mapping):
                    waypoints.append(
                        {
                            "x": float(waypoint.get("x", 0)),
                            "y": float(waypoint.get("y", 0)),
                        }
                    )
        return cls(
            id=str(payload["id"]),
            source_id=str(payload["source_id"]),
            target_id=str(payload["target_id"]),
            source_handle_id=str(payload.get("source_handle_id")) if payload.get("source_handle_id") is not None else None,
            target_handle_id=str(payload.get("target_handle_id")) if payload.get("target_handle_id") is not None else None,
            label=str(payload.get("label", "")),
            kind=str(payload.get("kind", "standard")),
            priority=int(payload.get("priority", 100)),
            waypoints=waypoints,
            condition=Condition.from_dict(condition) if isinstance(condition, Mapping) else None,
        )


@dataclass
class TransitionRecord:
    edge_id: str
    source_id: str
    target_id: str
    timestamp: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunState:
    graph_id: str
    input_payload: Any
    run_id: str = field(default_factory=lambda: str(uuid4()))
    agent_id: str | None = None
    parent_run_id: str | None = None
    current_node_id: str | None = None
    status: str = "pending"
    started_at: str = field(default_factory=utc_now_iso)
    ended_at: str | None = None
    node_outputs: dict[str, Any] = field(default_factory=dict)
    node_errors: dict[str, Any] = field(default_factory=dict)
    visit_counts: dict[str, int] = field(default_factory=dict)
    transition_history: list[TransitionRecord] = field(default_factory=list)
    event_history: list[RuntimeEvent] = field(default_factory=list)
    final_output: Any = None
    terminal_error: dict[str, Any] | None = None
    agent_runs: dict[str, Any] = field(default_factory=dict)

    def snapshot(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "graph_id": self.graph_id,
            "agent_id": self.agent_id,
            "parent_run_id": self.parent_run_id,
            "current_node_id": self.current_node_id,
            "status": self.status,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "input_payload": self.input_payload,
            "node_outputs": self.node_outputs,
            "node_errors": self.node_errors,
            "visit_counts": self.visit_counts,
            "transition_history": [transition.to_dict() for transition in self.transition_history],
            "event_history": [event.to_dict() for event in self.event_history],
            "final_output": self.final_output,
            "terminal_error": self.terminal_error,
            "agent_runs": self.agent_runs,
        }


@dataclass
class RuntimeServices:
    model_providers: dict[str, ModelProvider] = field(default_factory=dict)
    node_provider_registry: NodeProviderRegistry = field(default_factory=NodeProviderRegistry)
    tool_registry: ToolRegistry = field(default_factory=ToolRegistry)
    mcp_server_manager: McpServerManager | None = None
    config: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class NodeContext:
    graph: GraphDefinition
    state: RunState
    services: RuntimeServices
    node_id: str

    def latest_output(self, node_id: str) -> Any:
        return self.state.node_outputs.get(node_id)

    def latest_error(self, node_id: str) -> Any:
        return self.state.node_errors.get(node_id)

    def latest_envelope(self, node_id: str) -> MessageEnvelope | None:
        output = self.latest_output(node_id)
        if isinstance(output, Mapping) and "schema_version" in output and "payload" in output:
            return MessageEnvelope.from_dict(output)
        return None

    def latest_payload(self, node_id: str) -> Any:
        envelope = self.latest_envelope(node_id)
        if envelope is not None:
            return envelope.payload
        return self.latest_output(node_id)

    def graph_env_vars(self) -> dict[str, str]:
        return dict(self.graph.env_vars)

    def resolve_graph_env_value(self, value: Any) -> Any:
        return resolve_graph_env_value(value, self.graph.env_vars)

    def available_tool_definitions(self, names: list[str]) -> list[dict[str, Any]]:
        definitions: list[dict[str, Any]] = []
        for tool in self.services.tool_registry.exposable_definitions(names):
            definitions.append(self._apply_tool_node_overrides(tool.name, tool.to_dict()))
        return definitions

    def _candidate_mcp_context_nodes_for_model(self, node_id: str | None = None) -> list[McpContextProviderNode]:
        target_node_id = node_id or self.node_id
        target_node = self.graph.nodes.get(target_node_id)
        if not isinstance(target_node, ModelNode):
            return []

        candidate_tool_ids: list[str] = []
        configured_target_ids = target_node.config.get("tool_target_node_ids", [])
        if isinstance(configured_target_ids, Sequence) and not isinstance(configured_target_ids, (str, bytes)):
            candidate_tool_ids.extend(str(tool_id) for tool_id in configured_target_ids if str(tool_id).strip())
        for edge in self.graph.get_incoming_edges(target_node_id):
            if edge.kind != "binding":
                continue
            candidate_tool_ids.append(str(edge.source_id))

        seen_tool_ids: set[str] = set()
        candidates: list[McpContextProviderNode] = []
        for tool_node_id in candidate_tool_ids:
            if tool_node_id in seen_tool_ids:
                continue
            seen_tool_ids.add(tool_node_id)
            candidate = self.graph.nodes.get(tool_node_id)
            if isinstance(candidate, McpContextProviderNode):
                candidates.append(candidate)
        return candidates

    def _mcp_tool_prompt_enabled(self, node: BaseNode) -> bool:
        return bool(node.config.get("include_mcp_tool_context", False))

    def _mcp_tool_exposure_enabled(self, node: BaseNode) -> bool:
        return bool(node.config.get("expose_mcp_tools", True))

    def _exposable_mcp_tool(self, tool_name: str) -> dict[str, Any] | None:
        try:
            registry_tool = self.services.tool_registry.require_exposable(tool_name)
        except (KeyError, ValueError):
            return None
        if registry_tool.source_type != "mcp":
            return None
        return registry_tool.to_dict()

    def _configured_tool_names(self, node: BaseNode) -> list[str]:
        tool_names: list[str] = []
        raw_tool_names = node.config.get("tool_names", [])
        if isinstance(raw_tool_names, Sequence) and not isinstance(raw_tool_names, (str, bytes)):
            tool_names.extend(str(tool_name).strip() for tool_name in raw_tool_names if str(tool_name).strip())
        configured_name = str(node.config.get("tool_name", "") or getattr(node, "tool_name", "")).strip()
        if configured_name:
            tool_names.append(configured_name)
        deduped: list[str] = []
        seen: set[str] = set()
        for tool_name in tool_names:
            if tool_name in seen:
                continue
            seen.add(tool_name)
            deduped.append(tool_name)
        return deduped

    def _matching_tool_node(self, tool_name: str) -> BaseNode | None:
        for node in self.graph.nodes.values():
            if node.kind not in {"tool", "mcp_context_provider"}:
                continue
            if tool_name in self._configured_tool_names(node):
                return node
        return None

    def _apply_tool_node_overrides(self, tool_name: str, definition: Mapping[str, Any]) -> dict[str, Any]:
        tool_node = self._matching_tool_node(tool_name)
        if tool_node is None:
            return dict(definition)

        user_description_text = self.resolve_graph_env_value(
            str(tool_node.config.get("tool_user_description") or definition.get("description", ""))
        )
        agent_description_text = self.resolve_graph_env_value(
            str(
                tool_node.config.get("tool_agent_description")
                or tool_node.config.get("tool_model_description")
                or definition.get("description", "")
            )
        )
        schema_text = self.resolve_graph_env_value(
            str(
                tool_node.config.get("tool_model_schema_text")
                or json.dumps(definition.get("input_schema", {}), indent=2, sort_keys=True)
            )
        )
        template = self.resolve_graph_env_value(
            str(
                tool_node.config.get("tool_model_template")
                or "Tool: {tool_name}\nDescription:\n{tool_agent_description}\n\nSchema:\n{tool_schema}"
            )
        )

        resolved_schema: Any = definition.get("input_schema", {})
        try:
            parsed_schema = json.loads(schema_text)
            if isinstance(parsed_schema, Mapping):
                resolved_schema = parsed_schema
        except json.JSONDecodeError:
            pass

        return {
            **dict(definition),
            "description": template.format_map(
                SafeFormatDict(
                    self.template_variables(
                        {
                            "tool_name": tool_name,
                            "tool_user_description": user_description_text,
                            "tool_agent_description": agent_description_text,
                            "tool_description": agent_description_text,
                            "tool_schema": schema_text,
                        }
                    )
                )
            ),
            "input_schema": resolved_schema,
        }

    def resolve_binding(self, binding: Mapping[str, Any] | None) -> Any:
        if not binding:
            incoming_edges = self.graph.get_incoming_edges(self.node_id)
            if not incoming_edges:
                return self.state.input_payload
            for edge in reversed(incoming_edges):
                if edge.source_id in self.state.node_outputs:
                    return self.latest_output(edge.source_id)
            return None

        binding_type = str(binding.get("type", "latest_output"))
        if binding_type == "input_payload":
            return self.state.input_payload
        if binding_type == "latest_output":
            return self.latest_output(str(binding["source"]))
        if binding_type == "latest_payload":
            return self.latest_payload(str(binding["source"]))
        if binding_type == "latest_envelope":
            envelope = self.latest_envelope(str(binding["source"]))
            return envelope.to_dict() if envelope else None
        if binding_type == "latest_error":
            return self.latest_error(str(binding["source"]))
        if binding_type == "first_available_payload":
            for source in binding.get("sources", []):
                payload = self.latest_payload(str(source))
                if payload is not None:
                    return payload
            return None
        if binding_type == "first_available_envelope":
            for source in binding.get("sources", []):
                envelope = self.latest_envelope(str(source))
                if envelope is not None:
                    return envelope.to_dict()
            return None
        if binding_type == "available_tools":
            return self.available_tool_definitions(list(binding.get("names", [])))
        raise ValueError(f"Unsupported binding type '{binding_type}'.")

    def template_variables(self, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
        variables = {
            **self.graph.env_vars,
            "input_payload": self.state.input_payload,
            "run_id": self.state.run_id,
            "graph_id": self.state.graph_id,
            "current_node_id": self.state.current_node_id,
        }
        if extra:
            variables.update(extra)
        return {key: _json_safe(value) for key, value in variables.items()}

    def render_template(self, template: str, extra: Mapping[str, Any] | None = None) -> str:
        resolved_template = self.resolve_graph_env_value(template)
        return resolved_template.format_map(SafeFormatDict(self.template_variables(extra)))

    def bound_provider_node(self, node_id: str | None = None) -> ProviderNode | None:
        target_node_id = node_id or self.node_id
        target_node = self.graph.nodes.get(target_node_id)
        if target_node is None:
            return None

        binding_node_id = str(target_node.config.get("provider_binding_node_id", "")).strip()
        if binding_node_id:
            candidate = self.graph.nodes.get(binding_node_id)
            if isinstance(candidate, ProviderNode):
                return candidate

        for edge in self.graph.get_incoming_edges(target_node_id):
            candidate = self.graph.nodes.get(edge.source_id)
            if isinstance(candidate, ProviderNode):
                return candidate
        return None

    def mcp_tool_context_for_model(self, node_id: str | None = None) -> dict[str, Any] | None:
        target_node_id = node_id or self.node_id
        context_tools: list[dict[str, Any]] = []
        server_ids: set[str] = set()
        prompt_blocks: list[str] = []
        for candidate in self._candidate_mcp_context_nodes_for_model(target_node_id):
            if not self._mcp_tool_prompt_enabled(candidate):
                continue
            for tool_name in self._configured_tool_names(candidate):
                registry_tool = self._exposable_mcp_tool(tool_name)
                if registry_tool is None:
                    continue
                model_tool_definition = self._apply_tool_node_overrides(tool_name, registry_tool)
                prompt_blocks.append(str(model_tool_definition.get("description", "")).strip())
                server = None
                server_id = str(registry_tool.get("server_id", "") or "")
                if server_id and self.services.mcp_server_manager is not None:
                    try:
                        server = self.services.mcp_server_manager.get_server(server_id)
                        server_ids.add(server_id)
                    except KeyError:
                        server = None
                context_tools.append(
                    {
                        "tool_node_id": candidate.id,
                        "tool_node_label": candidate.label,
                        "tool_name": tool_name,
                        "tool": registry_tool,
                        "model_tool_definition": model_tool_definition,
                        "server": server,
                        "include_mcp_tool_context": self._mcp_tool_prompt_enabled(candidate),
                        "expose_mcp_tools": self._mcp_tool_exposure_enabled(candidate),
                    }
                )

        if not context_tools:
            return None

        servers: list[dict[str, Any]] = []
        if self.services.mcp_server_manager is not None:
            for server_id in sorted(server_ids):
                try:
                    servers.append(self.services.mcp_server_manager.get_server(server_id))
                except KeyError:
                    continue

        return {
            "tool_names": [tool["tool_name"] for tool in context_tools],
            "tool_nodes": context_tools,
            "servers": servers,
            "prompt_blocks": [block for block in prompt_blocks if block],
            "rendered_prompt_text": "\n\n".join(block for block in prompt_blocks if block),
            "run_context": {
                "run_id": self.state.run_id,
                "graph_id": self.state.graph_id,
                "node_id": target_node_id,
            },
        }

    def mcp_tool_definitions_for_model(self, node_id: str | None = None) -> list[dict[str, Any]]:
        definitions_by_name: dict[str, dict[str, Any]] = {}
        for candidate in self._candidate_mcp_context_nodes_for_model(node_id):
            if not self._mcp_tool_exposure_enabled(candidate):
                continue
            for tool_name in self._configured_tool_names(candidate):
                registry_tool = self._exposable_mcp_tool(tool_name)
                if registry_tool is None:
                    continue
                definitions_by_name[tool_name] = self._apply_tool_node_overrides(tool_name, registry_tool)
        return list(definitions_by_name.values())


class BaseNode(ABC):
    kind = "base"

    def __init__(
        self,
        node_id: str,
        label: str,
        *,
        category: NodeCategory,
        provider_id: str,
        provider_label: str | None = None,
        description: str = "",
        config: Mapping[str, Any] | None = None,
        position: Mapping[str, Any] | None = None,
    ) -> None:
        self.id = node_id
        self.label = label
        self.category = category
        self.provider_id = provider_id
        self.provider_label = provider_label or provider_id
        self.description = description
        self.config = dict(config or {})
        self.position = {
            "x": float((position or {}).get("x", 0)),
            "y": float((position or {}).get("y", 0)),
        }

    @abstractmethod
    def execute(self, context: NodeContext) -> NodeExecutionResult:
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "category": self.category.value,
            "label": self.label,
            "provider_id": self.provider_id,
            "provider_label": self.provider_label,
            "description": self.description,
            "position": self.position,
            "config": self.config,
        }


class InputNode(BaseNode):
    kind = "input"

    def __init__(
        self,
        node_id: str,
        label: str,
        provider_id: str = "core.input",
        provider_label: str = "Core Input Node",
        description: str = "",
        config: Mapping[str, Any] | None = None,
        position: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            node_id=node_id,
            label=label,
            category=NodeCategory.START,
            provider_id=provider_id,
            provider_label=provider_label,
            description=description,
            config=config,
            position=position,
        )

    def execute(self, context: NodeContext) -> NodeExecutionResult:
        payload = context.resolve_binding(self.config.get("input_binding"))
        if payload is None:
            payload = context.state.input_payload
        envelope = MessageEnvelope(
            schema_version="1.0",
            from_node_id=self.id,
            from_category=self.category.value,
            payload=payload,
            metadata={"contract": "message_envelope", "node_kind": self.kind},
        )
        return NodeExecutionResult(status="success", output=envelope.to_dict(), summary="Input payload captured.")


class DataNode(BaseNode):
    kind = "data"

    def __init__(
        self,
        node_id: str,
        label: str,
        provider_id: str = "core.data",
        provider_label: str = "Core Data Node",
        description: str = "",
        config: Mapping[str, Any] | None = None,
        position: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            node_id=node_id,
            label=label,
            category=NodeCategory.DATA,
            provider_id=provider_id,
            provider_label=provider_label,
            description=description,
            config=config,
            position=position,
        )

    def execute(self, context: NodeContext) -> NodeExecutionResult:
        source_value = context.resolve_binding(self.config.get("input_binding"))
        display_value = source_value
        source_envelope: MessageEnvelope | None = None
        if isinstance(display_value, Mapping) and "metadata" in display_value:
            try:
                source_envelope = MessageEnvelope.from_dict(display_value)
            except Exception:  # noqa: BLE001
                source_envelope = None
        if isinstance(source_value, Mapping) and "payload" in source_value:
            source_value = source_value.get("payload")
        display_only = bool(self.config.get("show_input_envelope", False))
        mode = "passthrough" if bool(self.config.get("lock_passthrough", False)) else self.config.get("mode", "passthrough")
        if mode == "template":
            payload = context.render_template(str(self.config.get("template", "{input_payload}")), {"source": source_value})
        else:
            payload = source_value
        artifacts: dict[str, Any] = {}
        if display_only:
            artifacts["display_envelope"] = display_value
        metadata = {
            "contract": "data_envelope",
            "node_kind": self.kind,
            "display_only": display_only,
        }
        errors: list[dict[str, Any]] = []
        tool_calls: list[dict[str, Any]] = []
        schema_version = "1.0"
        if display_only and source_envelope is not None:
            schema_version = source_envelope.schema_version
            artifacts = {**source_envelope.artifacts, **artifacts}
            errors = list(source_envelope.errors)
            tool_calls = list(source_envelope.tool_calls)
            metadata = {
                **dict(source_envelope.metadata),
                "contract": str(source_envelope.metadata.get("contract", "data_envelope")),
                "node_kind": self.kind,
                "display_only": True,
                "display_provider_id": self.provider_id,
            }
        envelope = MessageEnvelope(
            schema_version=schema_version,
            from_node_id=self.id,
            from_category=self.category.value,
            payload=payload,
            artifacts=artifacts,
            errors=errors,
            tool_calls=tool_calls,
            metadata=metadata,
        )
        return NodeExecutionResult(
            status="success",
            output=envelope.to_dict(),
            summary="Display node captured the upstream envelope." if display_only else "Data node completed.",
        )


class ProviderNode(BaseNode):
    kind = "provider"

    def __init__(
        self,
        node_id: str,
        label: str,
        provider_name: str,
        provider_id: str = "provider.generic",
        provider_label: str = "Generic Model Provider",
        description: str = "",
        config: Mapping[str, Any] | None = None,
        position: Mapping[str, Any] | None = None,
    ) -> None:
        merged_config = {"provider_name": provider_name, **dict(config or {})}
        super().__init__(
            node_id=node_id,
            label=label,
            category=NodeCategory.PROVIDER,
            provider_id=provider_id,
            provider_label=provider_label,
            description=description,
            config=merged_config,
            position=position,
        )
        self.provider_name = provider_name

    def execute(self, context: NodeContext) -> NodeExecutionResult:
        return NodeExecutionResult(
            status="success",
            output={
                "provider_name": self.provider_name,
                "provider_id": self.provider_id,
                "provider_label": self.provider_label,
                "config": dict(self.config),
            },
            summary=f"Provider node '{self.label}' is available for API node bindings.",
            metadata={"provider_name": self.provider_name, "binding_only": True},
        )

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload["model_provider_name"] = self.provider_name
        return payload


class ModelNode(BaseNode):
    kind = "model"

    _RUNTIME_CONFIG_KEYS = {
        "allowed_tool_names",
        "metadata_bindings",
        "mode",
        "preferred_tool_name",
        "prompt_name",
        "provider_config",
        "provider_name",
        "response_mode",
        "response_schema",
        "system_prompt",
        "tool_target_node_ids",
        "user_message_template",
    }

    def __init__(
        self,
        node_id: str,
        label: str,
        provider_name: str,
        prompt_name: str,
        node_provider_id: str = "model.generic",
        node_provider_label: str = "Generic Model Node",
        description: str = "",
        config: Mapping[str, Any] | None = None,
        position: Mapping[str, Any] | None = None,
    ) -> None:
        merged_config = {"provider_name": provider_name, "prompt_name": prompt_name, **dict(config or {})}
        super().__init__(
            node_id=node_id,
            label=label,
            category=NodeCategory.API,
            provider_id=node_provider_id,
            provider_label=node_provider_label,
            description=description,
            config=merged_config,
            position=position,
        )
        self.provider_name = provider_name
        self.prompt_name = prompt_name

    def execute(self, context: NodeContext) -> NodeExecutionResult:
        bound_provider_node = context.bound_provider_node(self.id)
        provider_name = str(
            (
                bound_provider_node.config.get("provider_name")
                if bound_provider_node is not None
                else self.config.get("provider_name", self.provider_name)
            )
            or self.provider_name
        )
        provider = context.services.model_providers[provider_name]
        metadata_bindings = dict(self.config.get("metadata_bindings", {}))
        metadata: dict[str, Any] = {}
        for key, binding in metadata_bindings.items():
            if isinstance(binding, Mapping):
                metadata[key] = context.resolve_binding(binding)
            else:
                metadata[key] = binding
        mcp_tool_context = context.mcp_tool_context_for_model(self.id)
        if mcp_tool_context is not None and "mcp_tool_context" not in metadata:
            metadata["mcp_tool_context"] = mcp_tool_context
            metadata["mcp_tool_context_prompt"] = mcp_tool_context.get("rendered_prompt_text", "")
        allowed_tool_names = list(self.config.get("allowed_tool_names", []))
        available_tool_payloads = context.available_tool_definitions(allowed_tool_names)
        for tool_definition in context.mcp_tool_definitions_for_model(self.id):
            tool_name = str(tool_definition.get("name", "")).strip()
            if not tool_name:
                continue
            if any(str(existing.get("name", "")).strip() == tool_name for existing in available_tool_payloads):
                continue
            available_tool_payloads.append(tool_definition)
        available_tools = [
            ModelToolDefinition(
                name=str(tool.get("name", "")),
                description=str(tool.get("description", "")),
                input_schema=dict(tool.get("input_schema", {})),
            )
            for tool in available_tool_payloads
            if isinstance(tool, Mapping) and isinstance(tool.get("name"), str) and isinstance(tool.get("input_schema"), Mapping)
        ]
        response_mode = str(self.config.get("response_mode", "message"))
        mcp_available_tool_names = sorted(
            {
                str(tool.get("name", "")).strip()
                for tool in available_tool_payloads
                if str(tool.get("source_type", "")).strip() == "mcp" and str(tool.get("name", "")).strip()
            }
        )
        metadata["available_tools"] = available_tool_payloads
        metadata["mcp_available_tool_names"] = mcp_available_tool_names
        metadata["mode"] = self.config.get("mode", self.prompt_name)
        metadata["preferred_tool_name"] = self.config.get("preferred_tool_name")
        metadata["response_mode"] = response_mode
        system_prompt_template = str(self.config.get("system_prompt", ""))
        user_template = str(self.config.get("user_message_template", "{input_payload}"))
        system_prompt = context.render_template(system_prompt_template, metadata)
        mcp_tool_prompt = str(metadata.get("mcp_tool_context_prompt", "") or "").strip()
        mcp_prompt_sections: list[str] = []
        if mcp_available_tool_names:
            guidance_lines = [
                "MCP Tool Guidance",
                "Use MCP tools only when a listed live capability is needed to answer the request or complete the task.",
                "Call only MCP tools that are explicitly exposed to you and follow their schemas exactly.",
                "Do not invent MCP tool names or arguments.",
                "If no exposed MCP tool is necessary, continue without calling one.",
            ]
            mcp_prompt_sections.append("\n".join(guidance_lines))
        if mcp_tool_prompt:
            mcp_prompt_sections.append(f"MCP Tool Context\n{mcp_tool_prompt}")
        if mcp_prompt_sections:
            appended_prompt = "\n\n".join(section.strip() for section in mcp_prompt_sections if section.strip())
            if system_prompt.strip():
                system_prompt = f"{system_prompt.rstrip()}\n\n{appended_prompt}"
            else:
                system_prompt = appended_prompt
        provider_config = self._provider_config(context, bound_provider_node)
        request = ModelRequest(
            prompt_name=self.prompt_name,
            messages=[
                ModelMessage(role="system", content=system_prompt),
                ModelMessage(role="user", content=context.render_template(user_template, metadata)),
            ],
            response_schema=self.config.get("response_schema"),
            provider_config=provider_config,
            available_tools=available_tools,
            preferred_tool_name=str(self.config.get("preferred_tool_name", "") or "") or None,
            response_mode=response_mode,
            metadata=metadata,
        )
        response = provider.generate(request)
        output = response.structured_output if response.structured_output is not None else response.content
        normalized_tool_calls = [
            {
                "tool_name": tool_call.tool_name,
                "arguments": tool_call.arguments,
                "provider_tool_id": tool_call.provider_tool_id,
                "metadata": dict(tool_call.metadata),
            }
            for tool_call in response.tool_calls
        ]
        if response_mode == "tool_call" and not normalized_tool_calls and response.structured_output is not None:
            available_tool_names = [str(tool.get("name", "")).strip() for tool in available_tool_payloads if str(tool.get("name", "")).strip()]
            target_tool = self.config.get("preferred_tool_name") or (available_tool_names[0] if available_tool_names else None)
            if target_tool:
                normalized_tool_calls = [
                    {
                        "tool_name": target_tool,
                        "arguments": response.structured_output,
                        "provider_tool_id": None,
                        "metadata": {"fallback": True},
                    }
                ]
        emit_tool_call_envelope = response_mode == "tool_call" or (response_mode == "auto" and bool(normalized_tool_calls))
        envelope_artifacts: dict[str, Any] = {}
        if emit_tool_call_envelope and response.content:
            envelope_artifacts["assistant_message"] = response.content
        envelope = MessageEnvelope(
            schema_version="1.0",
            from_node_id=self.id,
            from_category=self.category.value,
            payload=None if emit_tool_call_envelope else output,
            artifacts=envelope_artifacts,
            metadata={
                "contract": "tool_call_envelope" if emit_tool_call_envelope else "message_envelope",
                "node_kind": self.kind,
                "provider": provider.name,
                "prompt_name": self.prompt_name,
                "response_mode": response_mode,
                "tool_call_count": len(response.tool_calls),
                **response.metadata,
            },
        )
        if emit_tool_call_envelope:
            envelope.tool_calls = normalized_tool_calls
        return NodeExecutionResult(
            status="success",
            output=envelope.to_dict(),
            summary=f"Model node '{self.label}' completed.",
            metadata=envelope.metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload["model_provider_name"] = self.provider_name
        payload["prompt_name"] = self.prompt_name
        return payload

    def _provider_config(self, context: NodeContext, bound_provider_node: ProviderNode | None = None) -> dict[str, Any]:
        raw_provider_config = self.config.get("provider_config", {})
        provider_config = dict(raw_provider_config) if isinstance(raw_provider_config, Mapping) else {}
        if bound_provider_node is not None:
            for key, value in bound_provider_node.config.items():
                if key != "provider_binding_node_id":
                    provider_config[key] = value
        for key, value in self.config.items():
            if key not in self._RUNTIME_CONFIG_KEYS:
                provider_config[key] = value
        return context.resolve_graph_env_value(provider_config)


class ToolNode(BaseNode):
    kind = "tool"

    def __init__(
        self,
        node_id: str,
        label: str,
        tool_name: str,
        provider_id: str = "tool.registry",
        provider_label: str = "Registry Tool Node",
        description: str = "",
        config: Mapping[str, Any] | None = None,
        position: Mapping[str, Any] | None = None,
    ) -> None:
        merged_config = {"tool_name": tool_name, **dict(config or {})}
        super().__init__(
            node_id=node_id,
            label=label,
            category=NodeCategory.TOOL,
            provider_id=provider_id,
            provider_label=provider_label,
            description=description,
            config=merged_config,
            position=position,
        )
        self.tool_name = tool_name

    def execute(self, context: NodeContext) -> NodeExecutionResult:
        bound_value = context.resolve_binding(self.config.get("input_binding"))
        payload: Any = bound_value
        if isinstance(bound_value, Mapping) and "tool_calls" in bound_value:
            tool_calls = list(bound_value.get("tool_calls", []))
            if tool_calls:
                payload = tool_calls[0].get("arguments")
        elif isinstance(bound_value, Mapping) and "payload" in bound_value:
            payload = bound_value.get("payload")
        if not isinstance(payload, Mapping):
            payload = {}
        tool_context = ToolContext(
            run_id=context.state.run_id,
            graph_id=context.state.graph_id,
            node_id=context.node_id,
            state_snapshot=context.state.snapshot(),
        )
        tool_result = context.services.tool_registry.invoke(self.tool_name, payload, tool_context)
        envelope = MessageEnvelope(
            schema_version="1.0",
            from_node_id=self.id,
            from_category=self.category.value,
            payload=tool_result.output,
            errors=[tool_result.error] if tool_result.error else [],
            metadata={
                "contract": "tool_result_envelope",
                "node_kind": self.kind,
                "tool_name": self.tool_name,
                **tool_result.metadata,
            },
        )
        return NodeExecutionResult(
            status=tool_result.status,
            output=envelope.to_dict(),
            error=tool_result.error,
            summary=tool_result.summary or f"Tool '{self.tool_name}' completed.",
            metadata=envelope.metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload["tool_name"] = self.tool_name
        return payload


class McpContextProviderNode(BaseNode):
    kind = "mcp_context_provider"

    def __init__(
        self,
        node_id: str,
        label: str,
        provider_id: str = "tool.mcp_context_provider",
        provider_label: str = "MCP Context Provider",
        description: str = "",
        config: Mapping[str, Any] | None = None,
        position: Mapping[str, Any] | None = None,
    ) -> None:
        tool_names: list[str] = []
        raw_config = dict(config or {})
        raw_tool_names = raw_config.get("tool_names", [])
        if isinstance(raw_tool_names, Sequence) and not isinstance(raw_tool_names, (str, bytes)):
            tool_names.extend(str(tool_name).strip() for tool_name in raw_tool_names if str(tool_name).strip())
        fallback_tool_name = str(raw_config.get("tool_name", "")).strip()
        if fallback_tool_name and fallback_tool_name not in tool_names:
            tool_names.append(fallback_tool_name)
        merged_config = {
            "tool_names": tool_names,
            "include_mcp_tool_context": bool(raw_config.get("include_mcp_tool_context", False)),
            "expose_mcp_tools": bool(raw_config.get("expose_mcp_tools", True)),
            **raw_config,
        }
        super().__init__(
            node_id=node_id,
            label=label,
            category=NodeCategory.TOOL,
            provider_id=provider_id,
            provider_label=provider_label,
            description=description,
            config=merged_config,
            position=position,
        )

    def execute(self, context: NodeContext) -> NodeExecutionResult:
        output = {
            "tool_names": list(self.config.get("tool_names", [])),
            "include_mcp_tool_context": bool(self.config.get("include_mcp_tool_context", False)),
            "expose_mcp_tools": bool(self.config.get("expose_mcp_tools", True)),
        }
        envelope = MessageEnvelope(
            schema_version="1.0",
            from_node_id=self.id,
            from_category=self.category.value,
            payload=output,
            metadata={
                "contract": "data_envelope",
                "node_kind": self.kind,
                "binding_only": True,
            },
        )
        return NodeExecutionResult(
            status="success",
            output=envelope.to_dict(),
            summary=f"MCP context provider '{self.label}' prepared context metadata.",
            metadata=envelope.metadata,
        )


class McpToolExecutorNode(BaseNode):
    kind = "mcp_tool_executor"

    def __init__(
        self,
        node_id: str,
        label: str,
        provider_id: str = "tool.mcp_tool_executor",
        provider_label: str = "MCP Tool Executor",
        description: str = "",
        config: Mapping[str, Any] | None = None,
        position: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            node_id=node_id,
            label=label,
            category=NodeCategory.TOOL,
            provider_id=provider_id,
            provider_label=provider_label,
            description=description,
            config=config,
            position=position,
        )

    def execute(self, context: NodeContext) -> NodeExecutionResult:
        bound_value = context.resolve_binding(self.config.get("input_binding"))
        tool_call = None
        if isinstance(bound_value, Mapping) and "tool_calls" in bound_value:
            tool_calls = list(bound_value.get("tool_calls", []))
            if tool_calls:
                tool_call = tool_calls[0]
        tool_name = str((tool_call or {}).get("tool_name", "")).strip()
        payload = (tool_call or {}).get("arguments", {})
        if isinstance(bound_value, Mapping) and "payload" in bound_value and not tool_name:
            payload = bound_value.get("payload")
        if not isinstance(payload, Mapping):
            payload = {}
        if not tool_name:
            error = {"type": "missing_tool_call", "node_id": self.id, "message": "No MCP tool call was available to dispatch."}
            envelope = MessageEnvelope(
                schema_version="1.0",
                from_node_id=self.id,
                from_category=self.category.value,
                payload=None,
                errors=[error],
                metadata={"contract": "tool_result_envelope", "node_kind": self.kind},
            )
            return NodeExecutionResult(
                status="failed",
                output=envelope.to_dict(),
                error=error,
                summary=f"MCP executor '{self.label}' did not receive a tool call.",
                metadata=envelope.metadata,
            )
        try:
            tool_definition = context.services.tool_registry.require_invocable(tool_name)
            if tool_definition.source_type != "mcp":
                raise ValueError(f"Tool '{tool_name}' is not an MCP tool.")
            tool_context = ToolContext(
                run_id=context.state.run_id,
                graph_id=context.state.graph_id,
                node_id=context.node_id,
                state_snapshot=context.state.snapshot(),
            )
            tool_result = context.services.tool_registry.invoke(tool_name, payload, tool_context)
            envelope = MessageEnvelope(
                schema_version="1.0",
                from_node_id=self.id,
                from_category=self.category.value,
                payload=tool_result.output,
                errors=[tool_result.error] if tool_result.error else [],
                metadata={
                    "contract": "tool_result_envelope",
                    "node_kind": self.kind,
                    "tool_name": tool_name,
                    **tool_result.metadata,
                },
            )
            return NodeExecutionResult(
                status=tool_result.status,
                output=envelope.to_dict(),
                error=tool_result.error,
                summary=tool_result.summary or f"MCP tool '{tool_name}' completed.",
                metadata=envelope.metadata,
            )
        except (KeyError, ValueError) as exc:
            error = {"type": "mcp_tool_dispatch_error", "node_id": self.id, "tool_name": tool_name, "message": str(exc)}
            envelope = MessageEnvelope(
                schema_version="1.0",
                from_node_id=self.id,
                from_category=self.category.value,
                payload=None,
                errors=[error],
                metadata={
                    "contract": "tool_result_envelope",
                    "node_kind": self.kind,
                    "tool_name": tool_name,
                },
            )
            return NodeExecutionResult(
                status="failed",
                output=envelope.to_dict(),
                error=error,
                summary=f"MCP executor '{self.label}' could not dispatch '{tool_name}'.",
                metadata=envelope.metadata,
            )


class OutputNode(BaseNode):
    kind = "output"

    def __init__(
        self,
        node_id: str,
        label: str,
        provider_id: str = "core.output",
        provider_label: str = "Core Output Node",
        description: str = "",
        config: Mapping[str, Any] | None = None,
        position: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            node_id=node_id,
            label=label,
            category=NodeCategory.END,
            provider_id=provider_id,
            provider_label=provider_label,
            description=description,
            config=config,
            position=position,
        )

    def execute(self, context: NodeContext) -> NodeExecutionResult:
        bound_value = context.resolve_binding(self.config.get("source_binding"))
        if isinstance(bound_value, Mapping) and "payload" in bound_value:
            output = bound_value.get("payload")
        else:
            output = bound_value
        return NodeExecutionResult(status="success", output=output, summary="Output prepared.")


class GraphValidationError(ValueError):
    pass


def _node_from_dict(payload: Mapping[str, Any]) -> BaseNode:
    kind = str(payload["kind"])
    common = {
        "node_id": str(payload["id"]),
        "label": str(payload.get("label", payload["id"])),
        "description": str(payload.get("description", "")),
        "position": payload.get("position"),
        "config": payload.get("config"),
    }
    if kind == "input":
        return InputNode(
            provider_id=str(payload.get("provider_id", "core.input")),
            provider_label=str(payload.get("provider_label", "Core Input Node")),
            **common,
        )
    if kind == "data":
        return DataNode(
            provider_id=str(payload.get("provider_id", "core.data")),
            provider_label=str(payload.get("provider_label", "Core Data Node")),
            **common,
        )
    if kind == "provider":
        return ProviderNode(
            provider_name=str(payload.get("model_provider_name") or payload.get("config", {}).get("provider_name", "")),
            provider_id=str(payload.get("provider_id", "provider.generic")),
            provider_label=str(payload.get("provider_label", "Generic Model Provider")),
            **common,
        )
    if kind == "model":
        return ModelNode(
            provider_name=str(payload.get("model_provider_name") or payload.get("config", {}).get("provider_name", "")),
            prompt_name=str(payload.get("prompt_name") or payload.get("config", {}).get("prompt_name", "")),
            node_provider_id=str(payload.get("provider_id", "model.generic")),
            node_provider_label=str(payload.get("provider_label", "Generic Model Node")),
            **common,
        )
    if kind == "tool":
        config_payload = payload.get("config", {})
        include_mcp_context = bool(config_payload.get("include_mcp_tool_context", False)) if isinstance(config_payload, Mapping) else False
        if include_mcp_context:
            return McpContextProviderNode(
                provider_id=str(payload.get("provider_id", "tool.mcp_context_provider")),
                provider_label=str(payload.get("provider_label", "MCP Context Provider")),
                **common,
            )
        return ToolNode(
            tool_name=str(payload.get("tool_name") or payload.get("config", {}).get("tool_name", "")),
            provider_id=str(payload.get("provider_id", "tool.registry")),
            provider_label=str(payload.get("provider_label", "Registry Tool Node")),
            **common,
        )
    if kind == "mcp_context_provider":
        return McpContextProviderNode(
            provider_id=str(payload.get("provider_id", "tool.mcp_context_provider")),
            provider_label=str(payload.get("provider_label", "MCP Context Provider")),
            **common,
        )
    if kind == "mcp_tool_executor":
        return McpToolExecutorNode(
            provider_id=str(payload.get("provider_id", "tool.mcp_tool_executor")),
            provider_label=str(payload.get("provider_label", "MCP Tool Executor")),
            **common,
        )
    if kind == "output":
        return OutputNode(
            provider_id=str(payload.get("provider_id", "core.output")),
            provider_label=str(payload.get("provider_label", "Core Output Node")),
            **common,
        )
    raise GraphValidationError(f"Unsupported node kind '{kind}'.")


class GraphDefinition:
    def __init__(
        self,
        graph_id: str,
        name: str,
        start_node_id: str,
        nodes: list[BaseNode],
        edges: list[Edge],
        *,
        description: str = "",
        version: str = "1.0",
        env_vars: Mapping[str, Any] | None = None,
    ) -> None:
        node_ids = [node.id for node in nodes]
        if len(node_ids) != len(set(node_ids)):
            raise GraphValidationError("Graph node identifiers must be unique.")
        self.graph_id = graph_id
        self.name = name
        self.description = description
        self.version = version
        self.start_node_id = start_node_id
        self.env_vars = _normalize_graph_env_vars(env_vars)
        self.nodes = {node.id: node for node in nodes}
        self.edges = edges
        self.validate()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GraphDefinition:
        nodes = [_node_from_dict(node) for node in payload.get("nodes", [])]
        edges = [Edge.from_dict(edge) for edge in payload.get("edges", [])]
        return cls(
            graph_id=str(payload["graph_id"]),
            name=str(payload["name"]),
            description=str(payload.get("description", "")),
            version=str(payload.get("version", "1.0")),
            start_node_id=str(payload["start_node_id"]),
            env_vars=payload.get("env_vars"),
            nodes=nodes,
            edges=edges,
        )

    def validate(self) -> None:
        if self.start_node_id not in self.nodes:
            raise GraphValidationError(f"Unknown start node '{self.start_node_id}'.")
        if not self.nodes:
            raise GraphValidationError("Graphs must contain at least one node.")
        start_node = self.nodes[self.start_node_id]
        if start_node.category != NodeCategory.START:
            raise GraphValidationError("The graph start node must use the 'start' category.")

        output_nodes = [node for node in self.nodes.values() if node.category == NodeCategory.END]
        if not output_nodes:
            raise GraphValidationError("Graphs must include at least one 'end' category node.")

        standard_edge_counts: dict[str, int] = {}
        for edge in self.edges:
            if edge.source_id not in self.nodes:
                raise GraphValidationError(f"Edge '{edge.id}' references unknown source '{edge.source_id}'.")
            if edge.target_id not in self.nodes:
                raise GraphValidationError(f"Edge '{edge.id}' references unknown target '{edge.target_id}'.")
            source_node = self.nodes[edge.source_id]
            target_node = self.nodes[edge.target_id]
            if not is_valid_category_connection(source_node.category, target_node.category):
                raise GraphValidationError(
                    f"Edge '{edge.id}' uses an invalid category connection: "
                    f"{source_node.category.value} -> {target_node.category.value}."
                )
            if source_node.category == NodeCategory.END:
                raise GraphValidationError(f"End node '{source_node.id}' cannot have outgoing edges.")
            if edge.kind == "conditional" and edge.condition is None:
                raise GraphValidationError(f"Edge '{edge.id}' is conditional but missing a condition.")
            if edge.kind not in {"conditional", "binding"}:
                if source_node.category == NodeCategory.PROVIDER:
                    continue
                standard_edge_counts[edge.source_id] = standard_edge_counts.get(edge.source_id, 0) + 1
                if standard_edge_counts[edge.source_id] > 1:
                    raise GraphValidationError(
                        f"Node '{edge.source_id}' has more than one standard outgoing edge."
                    )

    def start_node(self) -> BaseNode:
        return self.nodes[self.start_node_id]

    def start_node_config(self) -> dict[str, Any]:
        return dict(self.start_node().config)

    def resolved_start_node_config(self) -> dict[str, Any]:
        return resolve_graph_env_value(self.start_node_config(), self.env_vars)

    def validate_against_services(self, services: RuntimeServices) -> None:
        for node in self.nodes.values():
            try:
                provider_definition = services.node_provider_registry.get(node.provider_id)
            except KeyError as exc:
                raise GraphValidationError(str(exc)) from exc
            if provider_definition.category != node.category:
                raise GraphValidationError(
                    f"Node '{node.id}' uses provider '{node.provider_id}' with category "
                    f"'{provider_definition.category.value}', but the node is '{node.category.value}'."
                )
            if provider_definition.node_kind != node.kind:
                raise GraphValidationError(
                    f"Node '{node.id}' uses provider '{node.provider_id}' for kind "
                    f"'{provider_definition.node_kind}', but the node kind is '{node.kind}'."
                )
            contract = get_category_contract(node.category)
            if not contract.produced_outputs:
                raise GraphValidationError(f"Node category '{node.category.value}' is missing a contract.")

            if node.kind == "provider":
                provider_name = str(node.config.get("provider_name", "") or getattr(node, "provider_name", ""))
                if provider_name not in services.model_providers:
                    raise GraphValidationError(
                        f"Provider node '{node.id}' references unknown model provider '{provider_name}'."
                    )
            if node.kind == "model":
                bound_provider = self._resolve_provider_binding(node)
                model_provider_name = (
                    str(bound_provider.config.get("provider_name", ""))
                    if bound_provider is not None
                    else str(node.config.get("provider_name", ""))
                )
                if model_provider_name not in services.model_providers:
                    raise GraphValidationError(
                        f"Model node '{node.id}' references unknown model provider '{model_provider_name}'."
                    )
                allowed_tool_names = [str(tool_name) for tool_name in node.config.get("allowed_tool_names", [])]
                for tool_name in allowed_tool_names:
                    try:
                        services.tool_registry.require_exposable(str(tool_name))
                    except (KeyError, ValueError) as exc:
                        raise GraphValidationError(str(exc)) from exc
                mcp_context_tool_names: list[str] = []
                response_mode = str(node.config.get("response_mode", "message"))
                if response_mode not in {"message", "tool_call", "auto"}:
                    raise GraphValidationError(
                        f"Model node '{node.id}' uses unsupported response mode '{response_mode}'."
                    )
                candidate_context_nodes: list[McpContextProviderNode] = []
                seen_context_node_ids: set[str] = set()
                tool_target_node_ids = node.config.get("tool_target_node_ids", [])
                if tool_target_node_ids:
                    if not isinstance(tool_target_node_ids, Sequence) or isinstance(tool_target_node_ids, (str, bytes)):
                        raise GraphValidationError(
                            f"Model node '{node.id}' must declare tool_target_node_ids as a list of tool node ids."
                        )
                    for target_node_id in tool_target_node_ids:
                        target_node = self.nodes.get(str(target_node_id))
                        if not isinstance(target_node, McpContextProviderNode):
                            raise GraphValidationError(
                                f"Model node '{node.id}' references unknown MCP context provider node '{target_node_id}'."
                            )
                        if target_node.id not in seen_context_node_ids:
                            candidate_context_nodes.append(target_node)
                            seen_context_node_ids.add(target_node.id)
                for edge in self.get_incoming_edges(node.id):
                    if edge.kind != "binding":
                        continue
                    source_node = self.nodes.get(edge.source_id)
                    if isinstance(source_node, McpContextProviderNode) and source_node.id not in seen_context_node_ids:
                        candidate_context_nodes.append(source_node)
                        seen_context_node_ids.add(source_node.id)
                for target_node in candidate_context_nodes:
                    tool_names = target_node.config.get("tool_names", [])
                    if not isinstance(tool_names, Sequence) or isinstance(tool_names, (str, bytes)):
                        raise GraphValidationError(
                            f"MCP context provider '{target_node.id}' must declare tool_names as a list."
                        )
                    for tool_name in tool_names:
                        tool_name_str = str(tool_name).strip()
                        if not tool_name_str:
                            continue
                        try:
                            tool_definition = services.tool_registry.require_exposable(tool_name_str)
                        except (KeyError, ValueError) as exc:
                            raise GraphValidationError(str(exc)) from exc
                        if tool_definition.source_type != "mcp":
                            raise GraphValidationError(
                                f"MCP context provider '{target_node.id}' references non-MCP tool '{tool_name_str}'."
                            )
                        if bool(target_node.config.get("expose_mcp_tools", True)):
                            mcp_context_tool_names.append(tool_name_str)
                combined_tool_names = [*allowed_tool_names, *[tool_name for tool_name in mcp_context_tool_names if tool_name not in allowed_tool_names]]
                if response_mode == "tool_call" and not combined_tool_names and not isinstance(node.config.get("response_schema"), Mapping):
                    raise GraphValidationError(
                        f"Model node '{node.id}' uses tool_call mode but does not expose any allowed tools."
                    )
                preferred_tool_name = str(node.config.get("preferred_tool_name", "") or "").strip()
                if preferred_tool_name and combined_tool_names and preferred_tool_name not in combined_tool_names:
                    raise GraphValidationError(
                        f"Model node '{node.id}' prefers tool '{preferred_tool_name}', but it is not exposed to the node."
                    )
                model_outgoing_edges = [edge for edge in self.get_outgoing_edges(node.id) if edge.kind != "binding"]
                api_tool_call_edges = [edge for edge in model_outgoing_edges if edge.source_handle_id == API_TOOL_CALL_HANDLE_ID]
                api_message_edges = [edge for edge in model_outgoing_edges if edge.source_handle_id == API_MESSAGE_HANDLE_ID]
                if len(api_tool_call_edges) > 1:
                    raise GraphValidationError(
                        f"Model node '{node.id}' can only declare one api-tool-call route."
                    )
                if len(api_message_edges) > 1:
                    raise GraphValidationError(
                        f"Model node '{node.id}' can only declare one api-message route."
                    )
                for edge in model_outgoing_edges:
                    target_node = self.nodes.get(edge.target_id)
                    if target_node is None:
                        continue
                    if edge.source_handle_id == API_TOOL_CALL_HANDLE_ID:
                        if target_node.category != NodeCategory.TOOL and not (
                            target_node.category == NodeCategory.DATA and target_node.provider_id == "core.data_display"
                        ):
                            raise GraphValidationError(
                                f"Model node '{node.id}' tool-call output must route to a tool node, but '{target_node.id}' is '{target_node.category.value}'."
                            )
                        if edge.kind != "conditional" or not _is_tool_call_contract_condition(edge.condition):
                            raise GraphValidationError(
                                f"Model node '{node.id}' tool-call output edge '{edge.id}' must match tool_call_envelope."
                            )
                    if edge.source_handle_id == API_MESSAGE_HANDLE_ID:
                        if target_node.category not in {NodeCategory.API, NodeCategory.DATA, NodeCategory.END}:
                            raise GraphValidationError(
                                f"Model node '{node.id}' message output must route to api, data, or end nodes, but '{target_node.id}' is '{target_node.category.value}'."
                            )
                        if edge.kind != "conditional" or not _is_message_contract_condition(edge.condition):
                            raise GraphValidationError(
                                f"Model node '{node.id}' message output edge '{edge.id}' must match message_envelope."
                            )
            if node.kind == "tool":
                try:
                    services.tool_registry.require_invocable(str(node.config.get("tool_name", "")))
                except (KeyError, ValueError) as exc:
                    raise GraphValidationError(str(exc)) from exc
            if node.kind == "mcp_context_provider":
                tool_names = node.config.get("tool_names", [])
                if not isinstance(tool_names, Sequence) or isinstance(tool_names, (str, bytes)):
                    raise GraphValidationError(f"MCP context provider '{node.id}' must declare tool_names as a list.")
                if not [str(tool_name).strip() for tool_name in tool_names if str(tool_name).strip()]:
                    raise GraphValidationError(f"MCP context provider '{node.id}' must register at least one MCP tool.")
                for tool_name in tool_names:
                    tool_name_str = str(tool_name).strip()
                    if not tool_name_str:
                        continue
                    try:
                        tool_definition = services.tool_registry.require_exposable(tool_name_str)
                    except (KeyError, ValueError) as exc:
                        raise GraphValidationError(str(exc)) from exc
                    if tool_definition.source_type != "mcp":
                        raise GraphValidationError(
                            f"MCP context provider '{node.id}' references non-MCP tool '{tool_name_str}'."
                        )
            if node.kind == "mcp_tool_executor":
                input_binding = node.config.get("input_binding")
                if input_binding is not None:
                    if not isinstance(input_binding, Mapping):
                        raise GraphValidationError(
                            f"MCP tool executor '{node.id}' must declare input_binding as an object."
                        )
                    binding_type = str(input_binding.get("type", "latest_output"))
                    if binding_type not in {"latest_output", "latest_envelope", "first_available_envelope"}:
                        raise GraphValidationError(
                            f"MCP tool executor '{node.id}' uses unsupported input binding '{binding_type}'."
                        )
                    if binding_type == "first_available_envelope":
                        raw_sources = input_binding.get("sources", [])
                        if not isinstance(raw_sources, Sequence) or isinstance(raw_sources, (str, bytes)):
                            raise GraphValidationError(
                                f"MCP tool executor '{node.id}' must declare sources as a list for first_available_envelope bindings."
                            )
                        binding_source_ids = [str(source_id).strip() for source_id in raw_sources if str(source_id).strip()]
                    else:
                        binding_source_id = str(input_binding.get("source", "")).strip()
                        binding_source_ids = [binding_source_id] if binding_source_id else []
                    if not binding_source_ids:
                        raise GraphValidationError(
                            f"MCP tool executor '{node.id}' must reference at least one upstream source."
                        )
                    for source_id in binding_source_ids:
                        source_node = self.nodes.get(source_id)
                        if source_node is None:
                            raise GraphValidationError(
                                f"MCP tool executor '{node.id}' references missing source node '{source_id}'."
                            )
                        if isinstance(source_node, ModelNode):
                            source_response_mode = str(source_node.config.get("response_mode", "message"))
                            if source_response_mode == "message":
                                raise GraphValidationError(
                                    f"MCP tool executor '{node.id}' must bind to a tool_call model output, but '{source_id}' uses response mode '{source_response_mode}'."
                                )
                else:
                    incoming_edges = self.get_incoming_edges(node.id)
                    if not incoming_edges:
                        raise GraphValidationError(
                            f"MCP tool executor '{node.id}' must have an incoming edge or an explicit input_binding."
                        )
                    valid_tool_call_routes = 0
                    for edge in incoming_edges:
                        source_node = self.nodes.get(edge.source_id)
                        if source_node is None:
                            continue
                        if not isinstance(source_node, ModelNode):
                            valid_tool_call_routes += 1
                            continue
                        source_response_mode = str(source_node.config.get("response_mode", "message"))
                        if edge.source_handle_id == API_MESSAGE_HANDLE_ID:
                            continue
                        if edge.source_handle_id == API_TOOL_CALL_HANDLE_ID:
                            valid_tool_call_routes += 1
                            continue
                        if source_response_mode == "tool_call":
                            valid_tool_call_routes += 1
                            continue
                        if source_response_mode == "auto" and _is_tool_call_contract_condition(edge.condition):
                            valid_tool_call_routes += 1
                            continue
                    if valid_tool_call_routes == 0:
                        raise GraphValidationError(
                            f"MCP tool executor '{node.id}' must receive a tool_call envelope from at least one upstream model node."
                        )
                    for edge in incoming_edges:
                        source_node = self.nodes.get(edge.source_id)
                        if not isinstance(source_node, ModelNode):
                            continue
                        source_response_mode = str(source_node.config.get("response_mode", "message"))
                        if edge.source_handle_id == API_MESSAGE_HANDLE_ID:
                            raise GraphValidationError(
                                f"MCP tool executor '{node.id}' cannot receive api-message output from model node '{source_node.id}'."
                            )
                        if edge.source_handle_id == API_TOOL_CALL_HANDLE_ID:
                            if edge.kind != "conditional" or not _is_tool_call_contract_condition(edge.condition):
                                raise GraphValidationError(
                                    f"MCP tool executor '{node.id}' must use a tool_call_envelope condition on api-tool-call edge '{edge.id}'."
                                )
                            continue
                        if source_response_mode == "message":
                            raise GraphValidationError(
                                f"MCP tool executor '{node.id}' cannot receive direct message-mode output from model node '{source_node.id}'."
                            )
                        if source_response_mode == "auto" and not _is_tool_call_contract_condition(edge.condition):
                            raise GraphValidationError(
                                f"MCP tool executor '{node.id}' must use a tool_call_envelope condition on edges from auto-mode model node '{source_node.id}'."
                            )

    def _resolve_provider_binding(self, node: BaseNode) -> ProviderNode | None:
        binding_node_id = str(node.config.get("provider_binding_node_id", "")).strip()
        if binding_node_id:
            bound_node = self.nodes.get(binding_node_id)
            if bound_node is None:
                raise GraphValidationError(
                    f"Node '{node.id}' references missing provider binding node '{binding_node_id}'."
                )
            if not isinstance(bound_node, ProviderNode):
                raise GraphValidationError(
                    f"Node '{node.id}' references '{binding_node_id}', but it is not a provider node."
                )
            if not any(edge.source_id == binding_node_id and edge.target_id == node.id for edge in self.get_incoming_edges(node.id)):
                raise GraphValidationError(
                    f"Node '{node.id}' is missing a provider edge from '{binding_node_id}'."
                )
            return bound_node

        for edge in self.get_incoming_edges(node.id):
            bound_node = self.nodes.get(edge.source_id)
            if isinstance(bound_node, ProviderNode):
                return bound_node
        return None

    def get_node(self, node_id: str) -> BaseNode:
        return self.nodes[node_id]

    def get_outgoing_edges(self, node_id: str) -> list[Edge]:
        return sorted(
            [edge for edge in self.edges if edge.source_id == node_id],
            key=lambda edge: edge.priority,
        )

    def get_incoming_edges(self, node_id: str) -> list[Edge]:
        return [edge for edge in self.edges if edge.target_id == node_id]

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "start_node_id": self.start_node_id,
            "env_vars": dict(self.env_vars),
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [edge.to_dict() for edge in self.edges],
        }
