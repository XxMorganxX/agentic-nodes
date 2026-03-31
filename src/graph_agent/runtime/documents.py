from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from graph_agent.runtime.core import DEFAULT_GRAPH_ENV_VARS, Edge, GraphDefinition, GraphValidationError, RuntimeServices, _node_from_dict


def _normalize_env_vars(payload: Mapping[str, Any] | None) -> dict[str, str]:
    env_vars = dict(DEFAULT_GRAPH_ENV_VARS)
    if not isinstance(payload, Mapping):
        return env_vars
    for key, value in payload.items():
        normalized_key = str(key).strip()
        if not normalized_key:
            continue
        env_vars[normalized_key] = str(value if value is not None else "")
    return env_vars


def _normalize_legacy_mcp_nodes(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized_nodes = [dict(node) for node in nodes]
    node_lookup = {str(node.get("id", "")): node for node in normalized_nodes}
    context_provider_ids = {
        node_id
        for node_id, node in node_lookup.items()
        if str(node.get("kind", "")) == "mcp_context_provider"
        or (
            str(node.get("kind", "")) == "tool"
            and isinstance(node.get("config"), Mapping)
            and bool(node["config"].get("include_mcp_tool_context", False))
        )
    }
    for node in normalized_nodes:
        if str(node.get("kind", "")) != "model":
            continue
        config = node.get("config")
        if not isinstance(config, Mapping):
            continue
        raw_target_ids = config.get("tool_target_node_ids", [])
        if not isinstance(raw_target_ids, list):
            continue
        filtered_target_ids = [str(target_id) for target_id in raw_target_ids if str(target_id) in context_provider_ids]
        next_config = dict(config)
        if filtered_target_ids:
            next_config["tool_target_node_ids"] = filtered_target_ids
        else:
            next_config.pop("tool_target_node_ids", None)
        node["config"] = next_config
    return normalized_nodes


@dataclass
class AgentDefinition:
    agent_id: str
    name: str
    start_node_id: str
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    description: str = ""
    version: str = "1.0"
    env_vars: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_GRAPH_ENV_VARS))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AgentDefinition:
        normalized_nodes = _normalize_legacy_mcp_nodes(
            [dict(node) for node in payload.get("nodes", []) if isinstance(node, Mapping)]
        )
        graph = GraphDefinition.from_dict(
            {
                "graph_id": str(payload.get("agent_id", payload.get("graph_id", "agent"))),
                "name": str(payload.get("name", payload.get("agent_id", "Agent"))),
                "description": str(payload.get("description", "")),
                "version": str(payload.get("version", "1.0")),
                "start_node_id": str(payload["start_node_id"]),
                "env_vars": payload.get("env_vars"),
                "nodes": normalized_nodes,
                "edges": payload.get("edges", []),
            }
        )
        return cls.from_graph(graph, agent_id=str(payload.get("agent_id", graph.graph_id)))

    @classmethod
    def from_graph(cls, graph: GraphDefinition, *, agent_id: str | None = None) -> AgentDefinition:
        return cls(
            agent_id=str(agent_id or graph.graph_id),
            name=graph.name,
            description=graph.description,
            version=graph.version,
            start_node_id=graph.start_node_id,
            env_vars=dict(graph.env_vars),
            nodes=[node.to_dict() for node in graph.nodes.values()],
            edges=[edge.to_dict() for edge in graph.edges],
        )

    def to_graph(self, *, graph_id: str, shared_env_vars: Mapping[str, Any] | None = None) -> GraphDefinition:
        merged_env_vars = _normalize_env_vars(shared_env_vars)
        merged_env_vars.update(_normalize_env_vars(self.env_vars))
        nodes = [_node_from_dict(node) for node in self.nodes]
        edges = [Edge.from_dict(edge) for edge in self.edges]
        return GraphDefinition(
            graph_id=graph_id,
            name=self.name,
            description=self.description,
            version=self.version,
            start_node_id=self.start_node_id,
            env_vars=merged_env_vars,
            nodes=nodes,
            edges=edges,
        )

    def validate(self, *, graph_id: str, shared_env_vars: Mapping[str, Any] | None = None) -> None:
        self.to_graph(graph_id=graph_id, shared_env_vars=shared_env_vars).validate()

    def validate_against_services(self, services: RuntimeServices, *, graph_id: str, shared_env_vars: Mapping[str, Any] | None = None) -> None:
        self.to_graph(graph_id=graph_id, shared_env_vars=shared_env_vars).validate_against_services(services)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "start_node_id": self.start_node_id,
            "env_vars": dict(self.env_vars),
            "nodes": self.nodes,
            "edges": self.edges,
        }


@dataclass
class TestEnvironmentDefinition:
    graph_id: str
    name: str
    agents: list[AgentDefinition]
    description: str = ""
    version: str = "1.0"
    graph_type: str = "test_environment"
    default_input: str = ""
    env_vars: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_GRAPH_ENV_VARS))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TestEnvironmentDefinition:
        if "agents" not in payload:
            normalized_payload = dict(payload)
            normalized_payload["nodes"] = _normalize_legacy_mcp_nodes(
                [dict(node) for node in payload.get("nodes", []) if isinstance(node, Mapping)]
            )
            legacy_graph = GraphDefinition.from_dict(normalized_payload)
            legacy_agent_id = str(payload.get("agent_id", f"{legacy_graph.graph_id}-agent"))
            graph_type = "graph"
            agents = [AgentDefinition.from_graph(legacy_graph, agent_id=legacy_agent_id)]
        else:
            agents = [AgentDefinition.from_dict(agent) for agent in payload.get("agents", [])]
            graph_type = str(payload.get("graph_type", "test_environment"))
        document = cls(
            graph_id=str(payload["graph_id"]),
            name=str(payload.get("name", payload["graph_id"])),
            description=str(payload.get("description", "")),
            version=str(payload.get("version", "1.0")),
            graph_type=graph_type,
            default_input=str(payload.get("default_input", "")),
            env_vars=_normalize_env_vars(payload.get("env_vars")),
            agents=agents,
        )
        document.validate()
        return document

    def validate(self) -> None:
        if not self.agents:
            raise GraphValidationError("Test environments must include at least one agent.")
        seen_agent_ids: set[str] = set()
        for agent in self.agents:
            if agent.agent_id in seen_agent_ids:
                raise GraphValidationError(f"Duplicate agent identifier '{agent.agent_id}'.")
            seen_agent_ids.add(agent.agent_id)
            agent.validate(graph_id=self.graph_id, shared_env_vars=self.env_vars)

    def validate_against_services(self, services: RuntimeServices) -> None:
        self.validate()
        for agent in self.agents:
            agent.validate_against_services(services, graph_id=self.graph_id, shared_env_vars=self.env_vars)

    def get_agent(self, agent_id: str) -> AgentDefinition:
        for agent in self.agents:
            if agent.agent_id == agent_id:
                return agent
        raise KeyError(agent_id)

    @property
    def default_agent_id(self) -> str:
        return self.agents[0].agent_id

    @property
    def is_multi_agent(self) -> bool:
        return self.graph_type == "test_environment" or len(self.agents) > 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "graph_type": self.graph_type,
            "default_input": self.default_input,
            "env_vars": dict(self.env_vars),
            "agents": [agent.to_dict() for agent in self.agents],
        }

    def as_graph(self, agent_id: str | None = None) -> GraphDefinition:
        selected_agent = self.get_agent(agent_id or self.default_agent_id)
        return selected_agent.to_graph(graph_id=self.graph_id, shared_env_vars=self.env_vars)


def load_graph_document(payload: Mapping[str, Any]) -> TestEnvironmentDefinition:
    return TestEnvironmentDefinition.from_dict(payload)
