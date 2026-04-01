from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import os
import re
from typing import Any, Mapping, Sequence
from uuid import uuid4

from graph_agent.providers.base import (
    ModelMessage,
    ModelProvider,
    ModelRequest,
    ModelToolDefinition,
    api_decision_response_schema,
    validate_api_decision_output,
)
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
MCP_TERMINAL_OUTPUT_HANDLE_ID = "mcp-terminal-output"
PROMPT_BLOCK_PROVIDER_ID = "core.prompt_block"
PROMPT_BLOCK_MODE = "prompt_block"
PROMPT_BLOCK_ROLES = {"system", "user", "assistant"}


def _json_safe(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=2, sort_keys=True)


def _normalize_prompt_block_role(value: Any) -> str:
    role = str(value or "user").strip().lower()
    return role if role in PROMPT_BLOCK_ROLES else "user"


def _is_prompt_block_payload(value: Any) -> bool:
    return bool(isinstance(value, Mapping) and str(value.get("kind", "")).strip() == "prompt_block")


def _prompt_block_text(value: Mapping[str, Any]) -> str:
    return str(value.get("content", "") or "")


def _render_prompt_block_text(value: Mapping[str, Any]) -> str:
    role = _normalize_prompt_block_role(value.get("role"))
    label = role.capitalize()
    name = str(value.get("name", "") or "").strip()
    header = f"{label} ({name})" if name else label
    content = _prompt_block_text(value).strip()
    return f"{header}: {content}" if content else f"{header}:"


def _render_chatgpt_style_messages(prompt_blocks: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    rendered_messages: list[dict[str, str]] = []
    for payload in prompt_blocks:
        content = _prompt_block_text(payload).strip()
        if not content:
            continue
        message_payload: dict[str, str] = {
            "role": _normalize_prompt_block_role(payload.get("role")),
            "content": content,
        }
        name = str(payload.get("name", "") or "").strip()
        if name:
            message_payload["name"] = name
        rendered_messages.append(message_payload)
    return rendered_messages


def _is_chat_message_payload(value: Any) -> bool:
    return bool(
        isinstance(value, Mapping)
        and str(value.get("role", "")).strip().lower() in PROMPT_BLOCK_ROLES
        and "content" in value
    )


def _normalize_chat_message_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "kind": "prompt_block",
        "role": _normalize_prompt_block_role(value.get("role")),
        "content": _prompt_block_text(value),
    }
    name = str(value.get("name", "") or "").strip()
    if name:
        payload["name"] = name
    return payload


def _infer_message_role_from_envelope(
    envelope: "MessageEnvelope",
    *,
    source_node_kind: str | None = None,
) -> str | None:
    explicit_role = envelope.metadata.get("prompt_block_role") or envelope.metadata.get("role")
    if isinstance(explicit_role, str) and explicit_role.strip().lower() in PROMPT_BLOCK_ROLES:
        return _normalize_prompt_block_role(explicit_role)
    if source_node_kind == "input":
        return "user"
    contract = str(envelope.metadata.get("contract", "") or "").strip()
    node_kind = str(envelope.metadata.get("node_kind", "") or "").strip()
    if contract == "message_envelope" and node_kind == "model":
        return "assistant"
    return None


def _extract_prompt_like_payloads(
    value: Any,
    *,
    source_node_kind: str | None = None,
) -> list[dict[str, Any]]:
    if _is_prompt_block_payload(value):
        return [dict(value)]
    if _is_chat_message_payload(value):
        return [_normalize_chat_message_payload(value)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        payloads: list[dict[str, Any]] = []
        for item in value:
            item_payloads = _extract_prompt_like_payloads(item, source_node_kind=source_node_kind)
            if not item_payloads:
                return []
            payloads.extend(item_payloads)
        return payloads
    if isinstance(value, Mapping) and "schema_version" in value and "payload" in value:
        try:
            envelope = MessageEnvelope.from_dict(value)
        except Exception:  # noqa: BLE001
            return []
        display_envelope = envelope.artifacts.get("display_envelope")
        if isinstance(display_envelope, Mapping):
            payloads = _extract_prompt_like_payloads(display_envelope, source_node_kind=source_node_kind)
            if payloads:
                return payloads
        payloads = _extract_prompt_like_payloads(envelope.payload, source_node_kind=source_node_kind)
        if payloads:
            return payloads
        role = _infer_message_role_from_envelope(envelope, source_node_kind=source_node_kind)
        content = envelope.payload
        if role is None or content is None or content == "":
            return []
        if isinstance(content, Mapping):
            if _is_chat_message_payload(content):
                return [_normalize_chat_message_payload(content)]
            content = next(
                (
                    str(content[key]).strip()
                    for key in ("message", "content", "text", "summary")
                    if isinstance(content.get(key), str) and str(content.get(key)).strip()
                ),
                "",
            )
            if not content:
                return []
        elif isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
            return []
        return [
            {
                "kind": "prompt_block",
                "role": role,
                "content": content if isinstance(content, str) else _json_safe(content),
            }
        ]
    return []


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
CONTEXT_BUILDER_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
CONTEXT_BUILDER_SLUG_PATTERN = re.compile(r"[^A-Za-z0-9_]+")


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


def _slugify_context_builder_placeholder(value: Any, *, fallback: str = "source") -> str:
    raw_value = str(value or "").strip().lower()
    normalized = CONTEXT_BUILDER_SLUG_PATTERN.sub("_", raw_value).strip("_")
    if not normalized:
        normalized = CONTEXT_BUILDER_SLUG_PATTERN.sub("_", fallback.strip().lower()).strip("_")
    if not normalized:
        normalized = "source"
    if normalized[0].isdigit():
        normalized = f"source_{normalized}"
    return normalized


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
    route_outputs: dict[str, Any] = field(default_factory=dict)

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


def _is_terminal_output_contract_condition(condition: Condition | None) -> bool:
    return bool(
        condition is not None
        and condition.condition_type == "result_payload_path_equals"
        and condition.path == "metadata.contract"
        and condition.value == "terminal_output_envelope"
    )


def _model_has_exposed_tool_context(graph: GraphDefinition, node: BaseNode) -> bool:
    candidate_node_ids: set[str] = set()
    configured_target_ids = node.config.get("tool_target_node_ids", [])
    if isinstance(configured_target_ids, Sequence) and not isinstance(configured_target_ids, (str, bytes)):
        candidate_node_ids.update(str(node_id).strip() for node_id in configured_target_ids if str(node_id).strip())
    for edge in graph.get_incoming_edges(node.id):
        if edge.kind == "binding":
            candidate_node_ids.add(edge.source_id)
    for node_id in candidate_node_ids:
        candidate = graph.nodes.get(node_id)
        if candidate is None or candidate.kind != "mcp_context_provider":
            continue
        if not bool(candidate.config.get("expose_mcp_tools", True)):
            continue
        tool_names = candidate.config.get("tool_names", [])
        if isinstance(tool_names, Sequence) and not isinstance(tool_names, (str, bytes)):
            if any(str(tool_name).strip() for tool_name in tool_names):
                return True
    return False


def _node_supports_mcp_tool_context(node: BaseNode | None) -> bool:
    return isinstance(node, ModelNode) or (
        isinstance(node, McpToolExecutorNode) and bool(node.config.get("enable_follow_up_decision", False))
    )


def _model_has_tool_output_route(graph: GraphDefinition, node: BaseNode) -> bool:
    for edge in graph.get_outgoing_edges(node.id):
        if edge.kind == "binding":
            continue
        if edge.source_handle_id == API_TOOL_CALL_HANDLE_ID:
            return True
        if edge.source_handle_id == API_MESSAGE_HANDLE_ID:
            continue
        target_node = graph.nodes.get(edge.target_id)
        if target_node is None:
            continue
        if target_node.category == NodeCategory.TOOL or _is_tool_call_contract_condition(edge.condition):
            return True
    return False


def _model_has_message_output_route(graph: GraphDefinition, node: BaseNode) -> bool:
    for edge in graph.get_outgoing_edges(node.id):
        if edge.kind == "binding":
            continue
        if edge.source_handle_id == API_MESSAGE_HANDLE_ID:
            return True
        if edge.source_handle_id == API_TOOL_CALL_HANDLE_ID:
            continue
        target_node = graph.nodes.get(edge.target_id)
        if target_node is None:
            continue
        if _is_message_contract_condition(edge.condition):
            return True
        if target_node.category in {NodeCategory.API, NodeCategory.DATA, NodeCategory.END}:
            return True
    return False


def infer_model_response_mode(graph: GraphDefinition, node: BaseNode) -> str:
    has_tool_output_route = _model_has_tool_output_route(graph, node)
    has_message_output_route = _model_has_message_output_route(graph, node)
    if has_tool_output_route and has_message_output_route:
        return "auto"
    if has_tool_output_route:
        return "tool_call"
    if _model_has_exposed_tool_context(graph, node) and not has_message_output_route:
        return "tool_call"
    return "message"


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
    current_edge_id: str | None = None
    status: str = "pending"
    started_at: str = field(default_factory=utc_now_iso)
    ended_at: str | None = None
    node_outputs: dict[str, Any] = field(default_factory=dict)
    edge_outputs: dict[str, Any] = field(default_factory=dict)
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
            "current_edge_id": self.current_edge_id,
            "status": self.status,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "input_payload": self.input_payload,
            "node_outputs": self.node_outputs,
            "edge_outputs": self.edge_outputs,
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

    def current_input_edge(self) -> Edge | None:
        current_edge_id = self.state.current_edge_id
        if not current_edge_id:
            return None
        for edge in self.graph.get_incoming_edges(self.node_id):
            if edge.id == current_edge_id:
                return edge
        return None

    def _current_route_output_for_source(self, node_id: str) -> Any:
        current_edge = self.current_input_edge()
        if current_edge is None or current_edge.source_id != node_id:
            return None
        return self.state.edge_outputs.get(current_edge.id)

    def _binding_sources_in_resolution_order(self, sources: Sequence[Any]) -> list[str]:
        ordered_sources = [str(source) for source in sources]
        current_edge = self.current_input_edge()
        if current_edge is None or current_edge.source_id not in ordered_sources:
            return ordered_sources
        prioritized_sources = [current_edge.source_id]
        prioritized_sources.extend(source for source in ordered_sources if source != current_edge.source_id)
        return prioritized_sources

    def latest_output(self, node_id: str) -> Any:
        route_output = self._current_route_output_for_source(node_id)
        if route_output is not None:
            return route_output
        if node_id in self.state.node_outputs:
            return self.state.node_outputs.get(node_id)
        prompt_block_envelope = self.prompt_block_envelope_for_node(node_id)
        if prompt_block_envelope is not None:
            return prompt_block_envelope.to_dict()
        display_node_output = self.display_node_output_for_node(node_id)
        if display_node_output is not None:
            return display_node_output
        return None

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
        if not _node_supports_mcp_tool_context(target_node):
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

    def prompt_block_payload_for_node(self, node_id: str) -> dict[str, Any] | None:
        candidate = self.graph.nodes.get(node_id)
        if candidate is None or candidate.kind != "data" or candidate.provider_id != PROMPT_BLOCK_PROVIDER_ID:
            return None
        role = _normalize_prompt_block_role(candidate.config.get("role"))
        content = self.render_template(str(candidate.config.get("content", "") or ""))
        name = self.render_template(str(candidate.config.get("name", "") or "")).strip()
        payload = {
            "kind": "prompt_block",
            "role": role,
            "content": content,
        }
        if name:
            payload["name"] = name
        return payload

    def prompt_block_envelope_for_node(self, node_id: str) -> MessageEnvelope | None:
        payload = self.prompt_block_payload_for_node(node_id)
        if payload is None:
            return None
        return MessageEnvelope(
            schema_version="1.0",
            from_node_id=node_id,
            from_category=NodeCategory.DATA.value,
            payload=payload,
            metadata={
                "contract": "data_envelope",
                "node_kind": "data",
                "data_mode": PROMPT_BLOCK_MODE,
                "provider_id": PROMPT_BLOCK_PROVIDER_ID,
                "binding_only": True,
                "prompt_block_role": payload["role"],
            },
        )

    def display_node_output_for_node(self, node_id: str) -> dict[str, Any] | None:
        candidate = self.graph.nodes.get(node_id)
        if candidate is None or candidate.kind != "data" or candidate.provider_id != "core.data_display":
            return None
        if not self.graph.get_incoming_edges(node_id) and not candidate.config.get("input_binding"):
            return None
        display_context = NodeContext(graph=self.graph, state=self.state, services=self.services, node_id=node_id)
        try:
            result = candidate.execute(display_context)
        except Exception:  # noqa: BLE001
            return None
        if not isinstance(result.output, Mapping):
            return None
        output = dict(result.output)
        artifacts = output.get("artifacts")
        has_display_envelope = isinstance(artifacts, Mapping) and artifacts.get("display_envelope") is not None
        if output.get("payload") is None and not has_display_envelope:
            return None
        return output

    def _bound_prompt_block_node_ids(self, node_id: str) -> list[str]:
        target_node = self.graph.nodes.get(node_id)
        if target_node is None:
            return []
        candidate_node_ids: list[str] = []
        configured_node_ids = target_node.config.get("prompt_block_node_ids", [])
        if isinstance(configured_node_ids, Sequence) and not isinstance(configured_node_ids, (str, bytes)):
            candidate_node_ids.extend(str(candidate_id).strip() for candidate_id in configured_node_ids if str(candidate_id).strip())
        for edge in self.graph.get_incoming_edges(node_id):
            if edge.kind == "binding":
                candidate_node_ids.append(edge.source_id)
        ordered_ids: list[str] = []
        seen: set[str] = set()
        for candidate_id in candidate_node_ids:
            if candidate_id in seen:
                continue
            seen.add(candidate_id)
            ordered_ids.append(candidate_id)
        return ordered_ids

    def prompt_block_payloads_for_node(self, node_id: str) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for candidate_id in self._bound_prompt_block_node_ids(node_id):
            payload = self.prompt_block_payload_for_node(candidate_id)
            if payload is not None:
                payloads.append(payload)
        return payloads

    def prompt_block_messages_for_model(self, node_id: str | None = None) -> list[ModelMessage]:
        target_node_id = node_id or self.node_id
        messages: list[ModelMessage] = []
        for payload in self.prompt_block_payloads_for_node(target_node_id):
            content = _prompt_block_text(payload).strip()
            if not content:
                continue
            messages.append(ModelMessage(role=_normalize_prompt_block_role(payload.get("role")), content=content))
        return messages

    def resolve_binding(self, binding: Mapping[str, Any] | None) -> Any:
        if not binding:
            current_edge = self.current_input_edge()
            if current_edge is not None and current_edge.id in self.state.edge_outputs:
                return self.state.edge_outputs[current_edge.id]
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
            for source in self._binding_sources_in_resolution_order(binding.get("sources", [])):
                payload = self.latest_payload(str(source))
                if payload is not None:
                    return payload
            return None
        if binding_type == "first_available_envelope":
            for source in self._binding_sources_in_resolution_order(binding.get("sources", [])):
                envelope = self.latest_envelope(str(source))
                if envelope is not None:
                    return envelope.to_dict()
            return None
        if binding_type == "available_tools":
            return self.available_tool_definitions(list(binding.get("names", [])))
        raise ValueError(f"Unsupported binding type '{binding_type}'.")

    def template_variables(self, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
        input_payload = self.resolve_binding(None)
        if isinstance(input_payload, Mapping) and "payload" in input_payload:
            input_payload = input_payload.get("payload")
        if input_payload is None:
            input_payload = self.state.input_payload
        variables = {
            **self.graph.env_vars,
            "input_payload": input_payload,
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

    def is_ready(self, context: NodeContext) -> bool:
        return True

    def runtime_input_preview(self, context: NodeContext) -> Any:
        return None

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

    def runtime_input_preview(self, context: NodeContext) -> Any:
        payload = context.resolve_binding(self.config.get("input_binding"))
        if payload is None:
            payload = context.state.input_payload
        return payload


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

    def _context_builder_bindings(self, context: NodeContext) -> list[dict[str, Any]]:
        incoming_edges = context.graph.get_incoming_edges(self.id)
        binding_source_ids = [edge.source_id for edge in incoming_edges if edge.kind == "binding"]
        preferred_source_ids: list[str] = []
        current_edge = context.current_input_edge()
        if current_edge is not None:
            preferred_source_ids.append(current_edge.source_id)
        preferred_source_ids.extend(binding_source_ids)
        preferred_source_ids.extend(edge.source_id for edge in incoming_edges)
        incoming_source_ids: list[str] = []
        for source_id in preferred_source_ids:
            if source_id == self.id or source_id in incoming_source_ids:
                continue
            incoming_source_ids.append(source_id)
        incoming_source_set = set(incoming_source_ids)

        bindings: list[dict[str, Any]] = []
        configured_bindings = self.config.get("input_bindings", [])
        if isinstance(configured_bindings, Sequence) and not isinstance(configured_bindings, (str, bytes)):
            for index, raw_binding in enumerate(configured_bindings):
                if not isinstance(raw_binding, Mapping):
                    continue
                source_node_id = str(raw_binding.get("source_node_id") or raw_binding.get("source") or "").strip()
                if not source_node_id or source_node_id not in incoming_source_set:
                    continue
                source_node = context.graph.nodes.get(source_node_id)
                placeholder = _slugify_context_builder_placeholder(
                    raw_binding.get("placeholder"),
                    fallback=(source_node.label if source_node is not None else f"source_{index + 1}"),
                )
                binding = raw_binding.get("binding")
                if not isinstance(binding, Mapping):
                    binding = {"type": "latest_payload", "source": source_node_id}
                bindings.append(
                    {
                        "source_node_id": source_node_id,
                        "placeholder": placeholder,
                        "binding": dict(binding),
                    }
                )

        if bindings:
            return bindings

        configured_sources = {str(binding["source_node_id"]) for binding in bindings}
        for index, source_node_id in enumerate(incoming_source_ids):
            if source_node_id in configured_sources:
                continue
            source_node = context.graph.nodes.get(source_node_id)
            placeholder = _slugify_context_builder_placeholder(
                source_node.label if source_node is not None else source_node_id,
                fallback=f"source_{index + 1}",
            )
            bindings.append(
                {
                    "source_node_id": source_node_id,
                    "placeholder": placeholder,
                    "binding": {"type": "latest_payload", "source": source_node_id},
                }
            )
        return bindings

    def _execute_context_builder(self, context: NodeContext) -> NodeExecutionResult:
        bindings = self._context_builder_bindings(context)
        resolved_variables: dict[str, Any] = {}
        ordered_values: list[str] = []
        ordered_prompt_blocks: list[dict[str, Any]] = []
        saw_non_prompt_value = False
        for binding in bindings:
            source_binding = binding.get("binding")
            source_node = context.graph.nodes.get(str(binding.get("source_node_id", "")).strip())
            resolved_value = context.resolve_binding(source_binding if isinstance(source_binding, Mapping) else None)
            prompt_source_value = resolved_value
            if source_node is not None and source_node.provider_id == "core.data_display":
                prompt_source_value = context.latest_output(source_node.id) or resolved_value
            prompt_like_values = _extract_prompt_like_payloads(
                prompt_source_value,
                source_node_kind=(source_node.kind if source_node is not None else None),
            )
            value = resolved_value
            if isinstance(value, Mapping) and "payload" in value:
                value = value.get("payload")
            placeholder = str(binding.get("placeholder", "")).strip()
            synthetic_prompt_block: dict[str, Any] | None = None
            if not prompt_like_values and source_node is not None and source_node.kind == "input" and value is not None and value != "":
                synthetic_prompt_block = {
                    "kind": "prompt_block",
                    "role": "user",
                    "content": value if isinstance(value, str) else _json_safe(value),
                }
            prompt_like_value = (
                [dict(prompt_like_value) for prompt_like_value in prompt_like_values]
                if prompt_like_values
                else ([synthetic_prompt_block] if synthetic_prompt_block is not None else [])
            )
            if prompt_like_value:
                rendered_value = "\n\n".join(_render_prompt_block_text(payload) for payload in prompt_like_value)
            else:
                rendered_value = value
            resolved_variables[placeholder] = rendered_value
            if prompt_like_value:
                ordered_prompt_blocks.extend(prompt_like_value)
            elif value is not None and value != "":
                saw_non_prompt_value = True
            if value is not None and value != "":
                ordered_values.append(rendered_value if isinstance(rendered_value, str) else _json_safe(rendered_value))

        template = str(self.config.get("template", "") or "")
        if template.strip():
            payload = context.render_template(template, resolved_variables)
        else:
            joiner = str(self.config.get("joiner", "\n\n") or "\n\n")
            should_compile_chatgpt_messages = bool(ordered_prompt_blocks) and not saw_non_prompt_value
            if should_compile_chatgpt_messages:
                rendered_messages = _render_chatgpt_style_messages(ordered_prompt_blocks)
                payload = rendered_messages if rendered_messages else joiner.join(ordered_values)
            else:
                payload = joiner.join(ordered_values)

        envelope = MessageEnvelope(
            schema_version="1.0",
            from_node_id=self.id,
            from_category=self.category.value,
            payload=payload,
            metadata={
                "contract": "data_envelope",
                "node_kind": self.kind,
                "data_mode": "context_builder",
                "binding_count": len(bindings),
                "placeholders": [str(binding.get("placeholder", "")) for binding in bindings],
                "prompt_blocks": ordered_prompt_blocks,
            },
        )
        return NodeExecutionResult(
            status="success",
            output=envelope.to_dict(),
            summary="Context builder rendered a prompt block." if bindings else "Context builder rendered an empty prompt block.",
        )

    def _is_context_builder_ready(self, context: NodeContext) -> bool:
        for binding in self._context_builder_bindings(context):
            source_node_id = str(binding.get("source_node_id", "")).strip()
            if not source_node_id:
                continue
            if context.latest_output(source_node_id) is None:
                return False
        return True

    def _execute_prompt_block(self, context: NodeContext) -> NodeExecutionResult:
        payload = context.prompt_block_payload_for_node(self.id) or {
            "kind": "prompt_block",
            "role": "user",
            "content": "",
        }
        envelope = MessageEnvelope(
            schema_version="1.0",
            from_node_id=self.id,
            from_category=self.category.value,
            payload=payload,
            metadata={
                "contract": "data_envelope",
                "node_kind": self.kind,
                "data_mode": PROMPT_BLOCK_MODE,
                "provider_id": self.provider_id,
                "binding_only": True,
                "prompt_block_role": payload["role"],
            },
        )
        return NodeExecutionResult(
            status="success",
            output=envelope.to_dict(),
            summary=f"Prompt block '{self.label}' prepared a {payload['role']} message.",
        )

    def runtime_input_preview(self, context: NodeContext) -> Any:
        mode = "passthrough" if bool(self.config.get("lock_passthrough", False)) else self.config.get("mode", "passthrough")
        if self.provider_id == PROMPT_BLOCK_PROVIDER_ID or mode == PROMPT_BLOCK_MODE:
            return context.prompt_block_payload_for_node(self.id)
        if mode == "context_builder":
            preview_bindings: list[dict[str, Any]] = []
            for binding in self._context_builder_bindings(context):
                source_binding = binding.get("binding")
                value = context.resolve_binding(source_binding if isinstance(source_binding, Mapping) else None)
                if isinstance(value, Mapping) and "payload" in value:
                    value = value.get("payload")
                preview_bindings.append(
                    {
                        "source_node_id": str(binding.get("source_node_id", "")),
                        "placeholder": str(binding.get("placeholder", "")),
                        "value": _render_prompt_block_text(value) if _is_prompt_block_payload(value) else value,
                    }
                )
            return preview_bindings
        source_value = context.resolve_binding(self.config.get("input_binding"))
        if bool(self.config.get("show_input_envelope", False)):
            return source_value
        if isinstance(source_value, Mapping) and "payload" in source_value:
            return source_value.get("payload")
        return source_value

    def is_ready(self, context: NodeContext) -> bool:
        mode = "passthrough" if bool(self.config.get("lock_passthrough", False)) else self.config.get("mode", "passthrough")
        if mode == "context_builder":
            return self._is_context_builder_ready(context)
        return True

    def execute(self, context: NodeContext) -> NodeExecutionResult:
        mode = "passthrough" if bool(self.config.get("lock_passthrough", False)) else self.config.get("mode", "passthrough")
        if self.provider_id == PROMPT_BLOCK_PROVIDER_ID or mode == PROMPT_BLOCK_MODE:
            return self._execute_prompt_block(context)
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
        if mode == "context_builder":
            return self._execute_context_builder(context)
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
        "prompt_block_node_ids",
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

    def _prompt_only_mcp_tool_decision_contract(
        self,
        *,
        mcp_tool_context: Mapping[str, Any] | None,
        mcp_available_tool_names: Sequence[str],
    ) -> str:
        if mcp_tool_context is None or mcp_available_tool_names:
            return ""
        return "\n".join(
            [
                "MCP Tool Decision Output",
                "When MCP tool metadata is present in prompt context but no MCP tools are directly callable, you must respond using this exact structure:",
                "",
                "Uses Tool: True|False",
                'Tool Call Schema: {"tool_name":"<tool name>","arguments":{...}} or NA',
                "DELIMITER",
                "<Explain why the tool schema is needed or why no tool is needed, and describe the next step required to finish the user's request.>",
                "",
                "Rules",
                "- Set `Uses Tool` to `True` only when one of the tools described in the MCP Tool Context is required.",
                "- When `Uses Tool` is `True`, `Tool Call Schema` must be a single JSON object containing exactly `tool_name` and `arguments`.",
                "- When `Uses Tool` is `False`, `Tool Call Schema` must be `NA`.",
                "- Do not claim that you already called a tool unless you were given an actual tool result.",
                "- The content after `DELIMITER` must be plain-language guidance for the next processing step.",
            ]
        )

    def _build_request(self, context: NodeContext, bound_provider_node: ProviderNode | None = None) -> ModelRequest:
        if bound_provider_node is None:
            bound_provider_node = context.bound_provider_node(self.id)
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
        prompt_block_payloads = context.prompt_block_payloads_for_node(self.id)
        if prompt_block_payloads and "prompt_blocks" not in metadata:
            metadata["prompt_blocks"] = prompt_block_payloads
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
        response_mode = infer_model_response_mode(context.graph, self)
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
        prompt_only_tool_contract = self._prompt_only_mcp_tool_decision_contract(
            mcp_tool_context=mcp_tool_context,
            mcp_available_tool_names=mcp_available_tool_names,
        )
        if prompt_only_tool_contract:
            mcp_prompt_sections.append(prompt_only_tool_contract)
        if mcp_prompt_sections:
            appended_prompt = "\n\n".join(section.strip() for section in mcp_prompt_sections if section.strip())
            if system_prompt.strip():
                system_prompt = f"{system_prompt.rstrip()}\n\n{appended_prompt}"
            else:
                system_prompt = appended_prompt
        provider_config = self._provider_config(context, bound_provider_node)
        decision_response_schema = api_decision_response_schema(
            final_message_schema=self.config.get("response_schema")
            if isinstance(self.config.get("response_schema"), Mapping)
            else None,
            available_tools=available_tools,
            allow_tool_calls=bool(available_tools),
        )
        return ModelRequest(
            prompt_name=self.prompt_name,
            messages=[
                ModelMessage(role="system", content=system_prompt),
                *context.prompt_block_messages_for_model(self.id),
                ModelMessage(role="user", content=context.render_template(user_template, metadata)),
            ],
            response_schema=decision_response_schema,
            provider_config=provider_config,
            available_tools=available_tools,
            preferred_tool_name=str(self.config.get("preferred_tool_name", "") or "") or None,
            response_mode=response_mode,
            metadata=metadata,
        )

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
        request = self._build_request(context, bound_provider_node)
        metadata = request.metadata
        response_mode_hint = request.response_mode
        available_tool_payloads = list(metadata.get("available_tools", []))
        callable_tool_names = {
            str(tool.get("name", "")).strip()
            for tool in available_tool_payloads
            if isinstance(tool, Mapping) and str(tool.get("name", "")).strip()
        }
        response = provider.generate(request)
        try:
            decision_output = validate_api_decision_output(
                response.structured_output if isinstance(response.structured_output, Mapping) else {},
                callable_tool_names=callable_tool_names,
            )
        except ValueError as exc:
            error = {"message": str(exc), "type": "structured_api_output_error"}
            envelope = MessageEnvelope(
                schema_version="1.0",
                from_node_id=self.id,
                from_category=self.category.value,
                payload=None,
                errors=[error],
                metadata={
                    "contract": "message_envelope",
                    "node_kind": self.kind,
                    "provider": provider.name,
                    "prompt_name": self.prompt_name,
                    "response_mode": response_mode_hint,
                    **response.metadata,
                },
            )
            return NodeExecutionResult(
                status="validation_error",
                output=envelope.to_dict(),
                error=error,
                summary=f"Model node '{self.label}' returned invalid structured API output.",
                metadata=envelope.metadata,
            )

        normalized_tool_calls = list(decision_output["tool_calls"])
        emit_tool_call_envelope = bool(decision_output["should_call_tools"])
        emit_message_envelope = not emit_tool_call_envelope

        base_metadata = {
            "node_kind": self.kind,
            "provider": provider.name,
            "prompt_name": self.prompt_name,
            "response_mode": response_mode_hint,
            "should_call_tools": bool(decision_output["should_call_tools"]),
            "tool_call_count": len(response.tool_calls),
            **response.metadata,
        }
        route_outputs: dict[str, Any] = {}

        tool_envelope: MessageEnvelope | None = None
        if emit_tool_call_envelope:
            tool_artifacts: dict[str, Any] = {}
            source_input = context.resolve_binding(None)
            source_input_envelope: MessageEnvelope | None = None
            if isinstance(source_input, Mapping) and "metadata" in source_input:
                try:
                    source_input_envelope = MessageEnvelope.from_dict(source_input)
                except Exception:  # noqa: BLE001
                    source_input_envelope = None
            if source_input_envelope is not None:
                tool_artifacts["source_input_payload"] = source_input_envelope.payload
                tool_artifacts["source_input_metadata"] = dict(source_input_envelope.metadata)
            elif source_input is not None:
                tool_artifacts["source_input_payload"] = source_input
            tool_envelope = MessageEnvelope(
                schema_version="1.0",
                from_node_id=self.id,
                from_category=self.category.value,
                payload=None,
                artifacts=tool_artifacts,
                tool_calls=normalized_tool_calls,
                metadata={
                    "contract": "tool_call_envelope",
                    **base_metadata,
                },
            )
            route_outputs[API_TOOL_CALL_HANDLE_ID] = tool_envelope.to_dict()

        message_envelope: MessageEnvelope | None = None
        if emit_message_envelope:
            message_payload = decision_output["final_message"]
            message_envelope = MessageEnvelope(
                schema_version="1.0",
                from_node_id=self.id,
                from_category=self.category.value,
                payload=message_payload,
                metadata={
                    "contract": "message_envelope",
                    **base_metadata,
                },
            )
            route_outputs[API_MESSAGE_HANDLE_ID] = message_envelope.to_dict()

        envelope = tool_envelope or message_envelope or MessageEnvelope(
            schema_version="1.0",
            from_node_id=self.id,
            from_category=self.category.value,
            payload=decision_output["final_message"],
            metadata={
                "contract": "message_envelope",
                **base_metadata,
            },
        )
        return NodeExecutionResult(
            status="success",
            output=envelope.to_dict(),
            summary=f"Model node '{self.label}' completed.",
            metadata=envelope.metadata,
            route_outputs=route_outputs,
        )

    def runtime_input_preview(self, context: NodeContext) -> Any:
        request = self._build_request(context)
        return {
            "messages": [{"role": message.role, "content": message.content} for message in request.messages],
            "response_mode": request.response_mode,
        }

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

    def runtime_input_preview(self, context: NodeContext) -> Any:
        bound_value = context.resolve_binding(self.config.get("input_binding"))
        payload: Any = bound_value
        if isinstance(bound_value, Mapping) and "tool_calls" in bound_value:
            tool_calls = list(bound_value.get("tool_calls", []))
            if tool_calls:
                payload = tool_calls[0].get("arguments")
        elif isinstance(bound_value, Mapping) and "payload" in bound_value:
            payload = bound_value.get("payload")
        return payload if isinstance(payload, Mapping) else {}

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

    def runtime_input_preview(self, context: NodeContext) -> Any:
        return {
            "tool_names": list(self.config.get("tool_names", [])),
            "include_mcp_tool_context": bool(self.config.get("include_mcp_tool_context", False)),
            "expose_mcp_tools": bool(self.config.get("expose_mcp_tools", True)),
        }


class McpToolExecutorNode(BaseNode):
    kind = "mcp_tool_executor"
    _RUNTIME_CONFIG_KEYS = {
        "allowed_tool_names",
        "enable_follow_up_decision",
        "input_binding",
        "metadata_bindings",
        "mode",
        "preferred_tool_name",
        "prompt_name",
        "provider_name",
        "response_mode",
        "response_schema",
        "system_prompt",
        "tool_target_node_ids",
        "user_message_template",
        "validate_last_tool_success",
    }
    _FOLLOW_UP_STATE_CONTRACT = "mcp_executor_follow_up_envelope"

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

    def _source_envelope_from_value(self, bound_value: Any) -> MessageEnvelope | None:
        source_envelope: MessageEnvelope | None = None
        if isinstance(bound_value, Mapping) and "schema_version" in bound_value and "payload" in bound_value:
            try:
                source_envelope = MessageEnvelope.from_dict(bound_value)
            except Exception:  # noqa: BLE001
                source_envelope = None
        return source_envelope

    def _resolve_tool_call_input(self, context: NodeContext) -> tuple[MessageEnvelope | None, str, dict[str, Any]]:
        bound_value = context.resolve_binding(self.config.get("input_binding"))
        source_envelope = self._source_envelope_from_value(bound_value)
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
        return source_envelope, tool_name, dict(payload)

    def _provider_config(self, context: NodeContext) -> dict[str, Any]:
        provider_config: dict[str, Any] = {}
        for key, value in self.config.items():
            if key not in self._RUNTIME_CONFIG_KEYS:
                provider_config[key] = value
        return context.resolve_graph_env_value(provider_config)

    def _tool_call_signature(self, tool_name: str, arguments: Any) -> str:
        try:
            normalized_arguments = json.dumps(arguments, sort_keys=True, separators=(",", ":"))
        except TypeError:
            normalized_arguments = json.dumps(_json_safe(arguments), sort_keys=True, separators=(",", ":"))
        return f"{tool_name}:{normalized_arguments}"

    def _successful_tool_call_signatures(self, normalized_payload: Mapping[str, Any]) -> set[str]:
        successful_signatures: set[str] = set()
        raw_tool_history = normalized_payload.get("tool_history", [])
        if not isinstance(raw_tool_history, Sequence) or isinstance(raw_tool_history, (str, bytes)):
            return successful_signatures
        for entry in raw_tool_history:
            if not isinstance(entry, Mapping):
                continue
            if str(entry.get("tool_status", "")).strip() != "success":
                continue
            tool_name = str(entry.get("tool_name", "")).strip()
            if tool_name:
                successful_signatures.add(self._tool_call_signature(tool_name, entry.get("tool_arguments", {})))
        return successful_signatures

    def _configured_follow_up_response_mode(self) -> str:
        response_mode = str(self.config.get("response_mode", "auto") or "auto").strip()
        if response_mode not in {"message", "tool_call", "auto"}:
            return "auto"
        return response_mode

    def _dispatch_tool_call(
        self,
        context: NodeContext,
        *,
        tool_name: str,
        payload: Mapping[str, Any],
        source_envelope: MessageEnvelope | None,
    ) -> NodeExecutionResult:
        payload_dict = dict(payload)
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
            tool_result = context.services.tool_registry.invoke(tool_name, payload_dict, tool_context)
            route_outputs: dict[str, Any] = {}
            execution_summary = tool_result.summary or f"MCP tool '{tool_name}' completed."
            tool_metadata = dict(tool_result.metadata)
            terminal_output = tool_metadata.pop("terminal_output", None)
            tool_artifacts: dict[str, Any] = {}
            if source_envelope is not None:
                tool_artifacts["source_tool_call_envelope"] = source_envelope.to_dict()
                if source_envelope.tool_calls:
                    tool_artifacts["requested_tool_call"] = dict(source_envelope.tool_calls[0])
                assistant_message = source_envelope.artifacts.get("assistant_message")
                if assistant_message:
                    tool_artifacts["assistant_message"] = assistant_message
            if isinstance(terminal_output, Mapping):
                terminal_envelope = MessageEnvelope(
                    schema_version="1.0",
                    from_node_id=self.id,
                    from_category=self.category.value,
                    payload=dict(terminal_output),
                    errors=[tool_result.error] if tool_result.error else [],
                    metadata={
                        "contract": "terminal_output_envelope",
                        "node_kind": self.kind,
                        "tool_name": tool_name,
                    },
                )
                route_outputs[MCP_TERMINAL_OUTPUT_HANDLE_ID] = terminal_envelope.to_dict()
                tool_artifacts["terminal_output"] = dict(terminal_output)
                tool_artifacts["terminal_output_envelope"] = terminal_envelope.to_dict()
            envelope = MessageEnvelope(
                schema_version="1.0",
                from_node_id=self.id,
                from_category=self.category.value,
                payload=tool_result.output,
                artifacts=tool_artifacts,
                errors=[tool_result.error] if tool_result.error else [],
                metadata={
                    "contract": "tool_result_envelope",
                    "node_kind": self.kind,
                    "tool_name": tool_name,
                    **tool_metadata,
                    "tool_status": tool_result.status,
                    "tool_summary": execution_summary,
                    "terminal_output_present": isinstance(terminal_output, Mapping),
                },
            )
            return NodeExecutionResult(
                status=tool_result.status,
                output=envelope.to_dict(),
                error=tool_result.error,
                summary=execution_summary,
                metadata=envelope.metadata,
                route_outputs=route_outputs,
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

    def _invalid_follow_up_result(self, message: str, *, route_outputs: Mapping[str, Any] | None = None) -> NodeExecutionResult:
        error = {"type": "mcp_executor_follow_up_error", "node_id": self.id, "message": message}
        envelope = MessageEnvelope(
            schema_version="1.0",
            from_node_id=self.id,
            from_category=self.category.value,
            payload=None,
            errors=[error],
            metadata={"contract": "message_envelope", "node_kind": self.kind},
        )
        return NodeExecutionResult(
            status="failed",
            output=envelope.to_dict(),
            error=error,
            summary=f"MCP executor '{self.label}' could not prepare follow-up context.",
            metadata=envelope.metadata,
            route_outputs=dict(route_outputs or {}),
        )

    def _build_follow_up_payload(
        self,
        context: NodeContext,
        source_envelope: MessageEnvelope,
    ) -> tuple[dict[str, Any], Mapping[str, Any] | None, Mapping[str, Any] | None]:
        requested_tool_call = source_envelope.artifacts.get("requested_tool_call")
        source_tool_call_envelope = source_envelope.artifacts.get("source_tool_call_envelope")
        source_tool_call_artifacts = (
            dict(source_tool_call_envelope.get("artifacts", {}))
            if isinstance(source_tool_call_envelope, Mapping) and isinstance(source_tool_call_envelope.get("artifacts"), Mapping)
            else {}
        )
        source_input_metadata = (
            dict(source_tool_call_artifacts.get("source_input_metadata", {}))
            if isinstance(source_tool_call_artifacts.get("source_input_metadata"), Mapping)
            else {}
        )
        source_input_payload = source_tool_call_artifacts.get("source_input_payload")
        previous_follow_up_payload = (
            dict(source_input_payload)
            if isinstance(source_input_payload, Mapping)
            and str(source_input_metadata.get("contract", "")).strip() == self._FOLLOW_UP_STATE_CONTRACT
            else None
        )
        terminal_output = source_envelope.artifacts.get("terminal_output")
        terminal_output_envelope = source_envelope.artifacts.get("terminal_output_envelope")
        tool_name = str(source_envelope.metadata.get("tool_name", "")).strip()
        tool_error = source_envelope.errors[0] if source_envelope.errors else None
        prior_tool_history = []
        if previous_follow_up_payload is not None:
            raw_tool_history = previous_follow_up_payload.get("tool_history", [])
            if isinstance(raw_tool_history, Sequence) and not isinstance(raw_tool_history, (str, bytes)):
                prior_tool_history = [dict(entry) for entry in raw_tool_history if isinstance(entry, Mapping)]
        current_tool_entry = {
            "tool_name": tool_name,
            "tool_status": str(source_envelope.metadata.get("tool_status", "") or ("failed" if tool_error else "success")),
            "tool_summary": str(source_envelope.metadata.get("tool_summary", "") or ""),
            "tool_arguments": dict(requested_tool_call.get("arguments", {})) if isinstance(requested_tool_call, Mapping) else {},
            "tool_call": dict(requested_tool_call) if isinstance(requested_tool_call, Mapping) else None,
            "tool_output": source_envelope.payload,
            "tool_error": tool_error,
            "tool_errors": list(source_envelope.errors),
            "tool_metadata": dict(source_envelope.metadata),
            "terminal_output": dict(terminal_output) if isinstance(terminal_output, Mapping) else None,
        }
        normalized_payload = {
            "original_input_payload": (
                previous_follow_up_payload.get("original_input_payload")
                if previous_follow_up_payload is not None and "original_input_payload" in previous_follow_up_payload
                else source_input_payload if source_input_payload is not None else context.state.input_payload
            ),
            "tool_name": tool_name,
            "tool_history": [*prior_tool_history, current_tool_entry],
            **current_tool_entry,
            "follow_up_context": {
                "executor_node_id": self.id,
                "run_id": context.state.run_id,
                "graph_id": context.state.graph_id,
                "tool_history_length": len(prior_tool_history) + 1,
            },
        }
        return normalized_payload, source_tool_call_envelope, terminal_output_envelope

    def _build_follow_up_request(
        self,
        context: NodeContext,
        normalized_payload: Mapping[str, Any],
        *,
        forbidden_tool_call_signatures: set[str] | None = None,
        force_response_mode: str | None = None,
        include_available_tools: bool = True,
    ) -> ModelRequest:
        forbidden_signatures = forbidden_tool_call_signatures or set()
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
        available_tool_payloads: list[dict[str, Any]] = []
        if include_available_tools:
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
        response_mode = force_response_mode or self._configured_follow_up_response_mode()
        mcp_available_tool_names = sorted(
            {
                str(tool.get("name", "")).strip()
                for tool in available_tool_payloads
                if str(tool.get("source_type", "")).strip() == "mcp" and str(tool.get("name", "")).strip()
            }
        )
        if response_mode != "message" and not available_tools:
            response_mode = "message"
        metadata["available_tools"] = available_tool_payloads
        metadata["mcp_available_tool_names"] = mcp_available_tool_names
        metadata["forbidden_tool_call_signatures"] = sorted(forbidden_signatures)
        metadata["mode"] = self.config.get("mode", self.config.get("prompt_name", "mcp_executor_follow_up"))
        metadata["preferred_tool_name"] = self.config.get("preferred_tool_name")
        metadata["response_mode"] = response_mode
        metadata["original_input_payload"] = normalized_payload.get("original_input_payload")
        metadata["tool_history"] = normalized_payload.get("tool_history", [])
        metadata["last_tool_name"] = normalized_payload.get("tool_name")
        metadata["last_tool_status"] = normalized_payload.get("tool_status")
        metadata["input_payload"] = normalized_payload
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
                "Do not repeat a successful MCP tool call already present in tool_history.",
                "If no exposed MCP tool is necessary, continue without calling one.",
            ]
            mcp_prompt_sections.append("\n".join(guidance_lines))
        if forbidden_signatures:
            mcp_prompt_sections.append(
                "Do not repeat any already satisfied MCP tool call signatures from tool_history. "
                + "Forbidden successful call signatures for this step: "
                + ", ".join(sorted(forbidden_signatures))
            )
        if not include_available_tools:
            mcp_prompt_sections.append(
                "No further tool calls are allowed in this step. Use tool_history to produce the final answer."
            )
        if mcp_tool_prompt:
            mcp_prompt_sections.append(f"MCP Tool Context\n{mcp_tool_prompt}")
        if mcp_prompt_sections:
            appended_prompt = "\n\n".join(section.strip() for section in mcp_prompt_sections if section.strip())
            if system_prompt.strip():
                system_prompt = f"{system_prompt.rstrip()}\n\n{appended_prompt}"
            else:
                system_prompt = appended_prompt
        decision_response_schema = api_decision_response_schema(
            final_message_schema=self.config.get("response_schema")
            if isinstance(self.config.get("response_schema"), Mapping)
            else None,
            available_tools=available_tools,
            allow_tool_calls=response_mode != "message" and bool(available_tools),
        )
        return ModelRequest(
            prompt_name=str(self.config.get("prompt_name", "mcp_executor_follow_up")),
            messages=[
                ModelMessage(role="system", content=system_prompt),
                ModelMessage(role="user", content=context.render_template(user_template, metadata)),
            ],
            response_schema=decision_response_schema,
            provider_config=self._provider_config(context),
            available_tools=available_tools,
            preferred_tool_name=str(self.config.get("preferred_tool_name", "") or "") or None,
            response_mode=response_mode,
            metadata=metadata,
        )

    def _follow_up_failure_result(
        self,
        normalized_payload: Mapping[str, Any],
        source_envelope: MessageEnvelope,
        *,
        route_outputs: Mapping[str, Any] | None = None,
    ) -> NodeExecutionResult:
        output = source_envelope.to_dict()
        if isinstance(output.get("artifacts"), Mapping):
            output["artifacts"] = {
                **dict(output["artifacts"]),
                "follow_up_payload": dict(normalized_payload),
                "validation_message": (
                    f"Skipping further MCP tool checks because '{normalized_payload.get('tool_name', 'tool')}' "
                    "did not complete successfully."
                ),
            }
        return NodeExecutionResult(
            status="failed",
            output=output,
            error=source_envelope.errors[0] if source_envelope.errors else {
                "type": "mcp_executor_follow_up_failed_tool",
                "node_id": self.id,
                "tool_name": normalized_payload.get("tool_name"),
                "message": "An MCP tool call failed during executor follow-up.",
            },
            summary=f"MCP executor '{self.label}' halted after failed MCP execution.",
            metadata=dict(source_envelope.metadata),
            route_outputs=dict(route_outputs or {}),
        )

    def _follow_up_iteration_limit(self) -> int:
        raw_limit = self.config.get("max_turns", 3)
        try:
            limit = int(raw_limit)
        except (TypeError, ValueError):
            limit = 3
        return max(1, min(limit, 12))

    def _follow_up_tool_call_envelope(
        self,
        *,
        tool_call: Mapping[str, Any],
        normalized_payload: Mapping[str, Any],
        source_tool_result_envelope: MessageEnvelope,
        source_tool_call_envelope: Mapping[str, Any] | None,
        terminal_output_envelope: Mapping[str, Any] | None,
        metadata: Mapping[str, Any],
        assistant_message: str | None,
    ) -> MessageEnvelope:
        tool_artifacts: dict[str, Any] = {
            "follow_up_payload": normalized_payload,
            "source_input_payload": normalized_payload,
            "source_input_metadata": {"contract": self._FOLLOW_UP_STATE_CONTRACT, "node_kind": self.kind},
            "source_tool_result_envelope": source_tool_result_envelope.to_dict(),
            "source_tool_call_envelope": dict(source_tool_call_envelope) if isinstance(source_tool_call_envelope, Mapping) else None,
            "terminal_output_envelope": dict(terminal_output_envelope) if isinstance(terminal_output_envelope, Mapping) else None,
        }
        if assistant_message:
            tool_artifacts["assistant_message"] = assistant_message
        return MessageEnvelope(
            schema_version="1.0",
            from_node_id=self.id,
            from_category=self.category.value,
            payload=None,
            artifacts=tool_artifacts,
            tool_calls=[dict(tool_call)],
            metadata={
                "contract": "tool_call_envelope",
                **dict(metadata),
            },
        )

    def _follow_up_message_result(
        self,
        *,
        normalized_payload: Mapping[str, Any],
        source_tool_result_envelope: MessageEnvelope,
        source_tool_call_envelope: Mapping[str, Any] | None,
        terminal_output_envelope: Mapping[str, Any] | None,
        metadata: Mapping[str, Any],
        payload: Any,
        summary: str,
        route_outputs: Mapping[str, Any] | None = None,
    ) -> NodeExecutionResult:
        message_envelope = MessageEnvelope(
            schema_version="1.0",
            from_node_id=self.id,
            from_category=self.category.value,
            payload=payload,
            artifacts={
                "follow_up_payload": normalized_payload,
                "source_tool_result_envelope": source_tool_result_envelope.to_dict(),
                "source_tool_call_envelope": dict(source_tool_call_envelope) if isinstance(source_tool_call_envelope, Mapping) else None,
                "terminal_output_envelope": dict(terminal_output_envelope) if isinstance(terminal_output_envelope, Mapping) else None,
            },
            metadata={
                "contract": "message_envelope",
                **dict(metadata),
            },
        )
        return NodeExecutionResult(
            status="success",
            output=message_envelope.to_dict(),
            summary=summary,
            metadata=message_envelope.metadata,
            route_outputs=dict(route_outputs or {}),
        )

    def _execute_follow_up(self, context: NodeContext, dispatch_result: NodeExecutionResult) -> NodeExecutionResult:
        source_envelope = self._source_envelope_from_value(dispatch_result.output)
        if source_envelope is None:
            return self._invalid_follow_up_result(
                "No MCP tool result envelope was available for follow-up evaluation.",
                route_outputs=dispatch_result.route_outputs,
            )
        if str(source_envelope.metadata.get("contract", "")).strip() != "tool_result_envelope":
            return self._invalid_follow_up_result(
                "MCP follow-up evaluation requires a tool_result_envelope.",
                route_outputs=dispatch_result.route_outputs,
            )
        normalized_payload, source_tool_call_envelope, terminal_output_envelope = self._build_follow_up_payload(context, source_envelope)
        if bool(self.config.get("validate_last_tool_success", True)) and normalized_payload["tool_status"] != "success":
            return self._follow_up_failure_result(normalized_payload, source_envelope, route_outputs=dispatch_result.route_outputs)

        provider_name = str(self.config.get("provider_name", "") or "claude_code")
        provider = context.services.model_providers[provider_name]
        route_outputs = dict(dispatch_result.route_outputs)
        iteration_budget = self._follow_up_iteration_limit()

        while True:
            successful_tool_call_signatures = self._successful_tool_call_signatures(normalized_payload)
            request = self._build_follow_up_request(
                context,
                normalized_payload,
                forbidden_tool_call_signatures=successful_tool_call_signatures,
            )
            metadata = request.metadata
            response_mode = request.response_mode
            available_tool_payloads = list(metadata.get("available_tools", []))
            callable_tool_names = {
                str(tool.get("name", "")).strip()
                for tool in available_tool_payloads
                if isinstance(tool, Mapping) and str(tool.get("name", "")).strip()
            }
            response = provider.generate(request)
            try:
                decision_output = validate_api_decision_output(
                    response.structured_output if isinstance(response.structured_output, Mapping) else {},
                    callable_tool_names=callable_tool_names,
                )
            except ValueError as exc:
                return self._invalid_follow_up_result(str(exc), route_outputs=route_outputs)
            output = decision_output["final_message"]
            normalized_tool_calls = [
                tool_call
                for tool_call in list(decision_output["tool_calls"])
                if self._tool_call_signature(str(tool_call["tool_name"]), tool_call.get("arguments"))
                not in successful_tool_call_signatures
            ]
            base_metadata = {
                "node_kind": self.kind,
                "provider": provider.name,
                "prompt_name": request.prompt_name,
                "response_mode": response_mode,
                "should_call_tools": bool(decision_output["should_call_tools"]),
                "tool_call_count": len(response.tool_calls),
                "tool_name": normalized_payload.get("tool_name"),
                "tool_status": normalized_payload.get("tool_status"),
                **response.metadata,
            }
            if normalized_tool_calls:
                if iteration_budget <= 0:
                    return self._invalid_follow_up_result(
                        "MCP executor follow-up exceeded the configured iteration limit.",
                        route_outputs=route_outputs,
                    )
                next_tool_call = normalized_tool_calls[0]
                tool_call_envelope = self._follow_up_tool_call_envelope(
                    tool_call=next_tool_call,
                    normalized_payload=normalized_payload,
                    source_tool_result_envelope=source_envelope,
                    source_tool_call_envelope=source_tool_call_envelope,
                    terminal_output_envelope=terminal_output_envelope,
                    metadata=base_metadata,
                    assistant_message=response.content,
                )
                dispatch_result = self._dispatch_tool_call(
                    context,
                    tool_name=str(next_tool_call.get("tool_name", "")).strip(),
                    payload=dict(next_tool_call.get("arguments", {})) if isinstance(next_tool_call.get("arguments", {}), Mapping) else {},
                    source_envelope=tool_call_envelope,
                )
                route_outputs.update(dispatch_result.route_outputs)
                source_envelope = self._source_envelope_from_value(dispatch_result.output)
                if source_envelope is None:
                    return self._invalid_follow_up_result(
                        "MCP executor follow-up lost the downstream tool result envelope.",
                        route_outputs=route_outputs,
                    )
                normalized_payload, source_tool_call_envelope, terminal_output_envelope = self._build_follow_up_payload(
                    context,
                    source_envelope,
                )
                if bool(self.config.get("validate_last_tool_success", True)) and normalized_payload["tool_status"] != "success":
                    return self._follow_up_failure_result(
                        normalized_payload,
                        source_envelope,
                        route_outputs=route_outputs,
                    )
                iteration_budget -= 1
                continue

            if response_mode != "message":
                request = self._build_follow_up_request(
                    context,
                    normalized_payload,
                    forbidden_tool_call_signatures=successful_tool_call_signatures,
                    force_response_mode="message",
                    include_available_tools=False,
                )
                response = provider.generate(request)
                try:
                    decision_output = validate_api_decision_output(
                        response.structured_output if isinstance(response.structured_output, Mapping) else {},
                        callable_tool_names=None,
                    )
                except ValueError as exc:
                    return self._invalid_follow_up_result(str(exc), route_outputs=route_outputs)
                output = decision_output["final_message"]
                base_metadata = {
                    "node_kind": self.kind,
                    "provider": provider.name,
                    "prompt_name": request.prompt_name,
                    "response_mode": request.response_mode,
                    "should_call_tools": bool(decision_output["should_call_tools"]),
                    "tool_call_count": len(response.tool_calls),
                    "tool_name": normalized_payload.get("tool_name"),
                    "tool_status": normalized_payload.get("tool_status"),
                    **response.metadata,
                }
            return self._follow_up_message_result(
                normalized_payload=normalized_payload,
                source_tool_result_envelope=source_envelope,
                source_tool_call_envelope=source_tool_call_envelope,
                terminal_output_envelope=terminal_output_envelope,
                metadata=base_metadata,
                payload=output,
                summary=f"MCP executor '{self.label}' completed follow-up evaluation.",
                route_outputs=route_outputs,
            )

    def execute(self, context: NodeContext) -> NodeExecutionResult:
        source_envelope, tool_name, payload = self._resolve_tool_call_input(context)
        dispatch_result = self._dispatch_tool_call(
            context,
            tool_name=tool_name,
            payload=payload,
            source_envelope=source_envelope,
        )
        if not bool(self.config.get("enable_follow_up_decision", False)):
            return dispatch_result
        return self._execute_follow_up(context, dispatch_result)

    def runtime_input_preview(self, context: NodeContext) -> Any:
        _, tool_name, payload = self._resolve_tool_call_input(context)
        return {
            "tool_name": tool_name,
            "arguments": payload,
            "follow_up_enabled": bool(self.config.get("enable_follow_up_decision", False)),
            "provider_name": self.config.get("provider_name", "claude_code"),
            "response_mode": self._configured_follow_up_response_mode(),
        }


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

    def runtime_input_preview(self, context: NodeContext) -> Any:
        bound_value = context.resolve_binding(self.config.get("source_binding"))
        if isinstance(bound_value, Mapping) and "payload" in bound_value:
            return bound_value.get("payload")
        return bound_value


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
        default_input: str = "",
        env_vars: Mapping[str, Any] | None = None,
    ) -> None:
        node_ids = [node.id for node in nodes]
        if len(node_ids) != len(set(node_ids)):
            raise GraphValidationError("Graph node identifiers must be unique.")
        self.graph_id = graph_id
        self.name = name
        self.description = description
        self.version = version
        self.default_input = default_input
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
            default_input=str(payload.get("default_input", "")),
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
            if source_node.kind == "data" and source_node.provider_id == PROMPT_BLOCK_PROVIDER_ID:
                if edge.kind != "binding":
                    raise GraphValidationError(
                        f"Prompt block '{source_node.id}' can only create binding edges."
                    )
                if target_node.kind not in {"model", "data"}:
                    raise GraphValidationError(
                        f"Prompt block '{source_node.id}' can only bind into model or data nodes."
                    )
            if target_node.kind == "data" and target_node.provider_id == PROMPT_BLOCK_PROVIDER_ID:
                raise GraphValidationError(
                    f"Prompt block '{target_node.id}' is source-only and cannot receive incoming edges."
                )
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
                        services.tool_registry.require_graph_reference(str(tool_name))
                    except (KeyError, ValueError) as exc:
                        raise GraphValidationError(str(exc)) from exc
                mcp_context_tool_names: list[str] = []
                response_mode = infer_model_response_mode(self, node)
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
                prompt_block_node_ids = node.config.get("prompt_block_node_ids", [])
                if prompt_block_node_ids:
                    if not isinstance(prompt_block_node_ids, Sequence) or isinstance(prompt_block_node_ids, (str, bytes)):
                        raise GraphValidationError(
                            f"Model node '{node.id}' must declare prompt_block_node_ids as a list of prompt block node ids."
                        )
                    for prompt_block_node_id in prompt_block_node_ids:
                        prompt_block_node_id_str = str(prompt_block_node_id).strip()
                        if not prompt_block_node_id_str:
                            continue
                        target_node = self.nodes.get(prompt_block_node_id_str)
                        if target_node is None or target_node.kind != "data" or target_node.provider_id != PROMPT_BLOCK_PROVIDER_ID:
                            raise GraphValidationError(
                                f"Model node '{node.id}' references unknown prompt block node '{prompt_block_node_id_str}'."
                            )
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
                            tool_definition = services.tool_registry.require_graph_reference(tool_name_str)
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
                    services.tool_registry.require_graph_reference(str(node.config.get("tool_name", "")), require_executor=True)
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
                        tool_definition = services.tool_registry.require_graph_reference(tool_name_str)
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
                            source_response_mode = infer_model_response_mode(self, source_node)
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
                        source_response_mode = infer_model_response_mode(self, source_node)
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
                        source_response_mode = infer_model_response_mode(self, source_node)
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
                terminal_output_edges = [
                    edge
                    for edge in self.get_outgoing_edges(node.id)
                    if edge.kind != "binding" and edge.source_handle_id == MCP_TERMINAL_OUTPUT_HANDLE_ID
                ]
                if len(terminal_output_edges) > 1:
                    raise GraphValidationError(
                        f"MCP tool executor '{node.id}' can only declare one terminal output route."
                    )
                for edge in terminal_output_edges:
                    if edge.kind != "conditional" or not _is_terminal_output_contract_condition(edge.condition):
                        raise GraphValidationError(
                            f"MCP tool executor '{node.id}' terminal output edge '{edge.id}' must match terminal_output_envelope."
                        )
                if bool(node.config.get("enable_follow_up_decision", False)):
                    provider_name = str(node.config.get("provider_name", "") or "claude_code")
                    if provider_name not in services.model_providers:
                        raise GraphValidationError(
                            f"MCP tool executor '{node.id}' references unknown follow-up model provider '{provider_name}'."
                        )
                    response_mode = str(node.config.get("response_mode", "auto") or "auto").strip()
                    if response_mode not in {"message", "tool_call", "auto"}:
                        raise GraphValidationError(
                            f"MCP tool executor '{node.id}' uses unsupported follow-up response_mode '{response_mode}'."
                        )
                    allowed_tool_names = [str(tool_name) for tool_name in node.config.get("allowed_tool_names", [])]
                    for tool_name in allowed_tool_names:
                        try:
                            services.tool_registry.require_graph_reference(str(tool_name))
                        except (KeyError, ValueError) as exc:
                            raise GraphValidationError(str(exc)) from exc
                    mcp_context_tool_names: list[str] = []
                    candidate_context_nodes: list[McpContextProviderNode] = []
                    seen_context_node_ids: set[str] = set()
                    tool_target_node_ids = node.config.get("tool_target_node_ids", [])
                    if tool_target_node_ids:
                        if not isinstance(tool_target_node_ids, Sequence) or isinstance(tool_target_node_ids, (str, bytes)):
                            raise GraphValidationError(
                                f"MCP tool executor '{node.id}' must declare tool_target_node_ids as a list of MCP context provider node ids."
                            )
                        for target_node_id in tool_target_node_ids:
                            target_node = self.nodes.get(str(target_node_id))
                            if not isinstance(target_node, McpContextProviderNode):
                                raise GraphValidationError(
                                    f"MCP tool executor '{node.id}' references unknown MCP context provider node '{target_node_id}'."
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
                                tool_definition = services.tool_registry.require_graph_reference(tool_name_str)
                            except (KeyError, ValueError) as exc:
                                raise GraphValidationError(str(exc)) from exc
                            if tool_definition.source_type != "mcp":
                                raise GraphValidationError(
                                    f"MCP context provider '{target_node.id}' references non-MCP tool '{tool_name_str}'."
                                )
                            if bool(target_node.config.get("expose_mcp_tools", True)):
                                mcp_context_tool_names.append(tool_name_str)
                    combined_tool_names = [
                        *allowed_tool_names,
                        *[tool_name for tool_name in mcp_context_tool_names if tool_name not in allowed_tool_names],
                    ]
                    if response_mode == "tool_call" and not combined_tool_names and not isinstance(node.config.get("response_schema"), Mapping):
                        raise GraphValidationError(
                            f"MCP tool executor '{node.id}' uses tool_call follow-up mode but does not expose any allowed tools."
                        )
                    preferred_tool_name = str(node.config.get("preferred_tool_name", "") or "").strip()
                    if preferred_tool_name and combined_tool_names and preferred_tool_name not in combined_tool_names:
                        raise GraphValidationError(
                            f"MCP tool executor '{node.id}' prefers tool '{preferred_tool_name}', but it is not exposed to the node."
                        )
            if node.kind == "data" and node.provider_id == "core.context_builder":
                raw_bindings = node.config.get("input_bindings", [])
                if not isinstance(raw_bindings, Sequence) or isinstance(raw_bindings, (str, bytes)):
                    raise GraphValidationError(f"Context builder '{node.id}' must declare input_bindings as a list.")
                incoming_source_ids = {edge.source_id for edge in self.get_incoming_edges(node.id)}
                seen_placeholders: set[str] = set()
                for index, raw_binding in enumerate(raw_bindings):
                    if not isinstance(raw_binding, Mapping):
                        raise GraphValidationError(
                            f"Context builder '{node.id}' binding at index {index} must be an object."
                        )
                    source_node_id = str(raw_binding.get("source_node_id") or raw_binding.get("source") or "").strip()
                    if source_node_id and source_node_id not in incoming_source_ids:
                        raise GraphValidationError(
                            f"Context builder '{node.id}' references '{source_node_id}', but it is not currently connected."
                        )
                    placeholder = _slugify_context_builder_placeholder(
                        raw_binding.get("placeholder"),
                        fallback=f"source_{index + 1}",
                    )
                    if not CONTEXT_BUILDER_IDENTIFIER_PATTERN.match(placeholder):
                        raise GraphValidationError(
                            f"Context builder '{node.id}' uses invalid placeholder '{placeholder}'."
                        )
                    if placeholder in seen_placeholders:
                        raise GraphValidationError(
                            f"Context builder '{node.id}' uses duplicate placeholder '{placeholder}'."
                        )
                    seen_placeholders.add(placeholder)
            if node.kind == "data" and node.provider_id == PROMPT_BLOCK_PROVIDER_ID:
                raw_role = str(node.config.get("role", "user") or "user").strip().lower()
                if raw_role not in PROMPT_BLOCK_ROLES:
                    raise GraphValidationError(
                        f"Prompt block '{node.id}' uses unsupported role '{raw_role}'."
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
            "default_input": self.default_input,
            "start_node_id": self.start_node_id,
            "env_vars": dict(self.env_vars),
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [edge.to_dict() for edge in self.edges],
        }
