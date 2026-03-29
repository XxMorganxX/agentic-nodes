from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from graph_agent.examples.tool_schema_repair import build_example_graph_payload
from graph_agent.runtime.core import GraphDefinition, GraphValidationError, RuntimeServices
from graph_agent.runtime.node_providers import DEFAULT_CATEGORY_CONTRACTS, list_connection_rules


class GraphStore:
    def __init__(self, services: RuntimeServices, path: Path | None = None) -> None:
        self.services = services
        self.path = path or Path(__file__).resolve().with_name("graphs_store.json")
        self._ensure_seed_data()

    def list_graphs(self) -> list[dict[str, Any]]:
        payload = self._load_all()
        return payload["graphs"]

    def get_graph(self, graph_id: str) -> dict[str, Any]:
        for graph in self.list_graphs():
            if graph["graph_id"] == graph_id:
                return graph
        raise KeyError(graph_id)

    def create_graph(self, graph_payload: dict[str, Any]) -> dict[str, Any]:
        self._validate_graph_payload(graph_payload)
        payload = self._load_all()
        existing_ids = {graph["graph_id"] for graph in payload["graphs"]}
        if graph_payload["graph_id"] in existing_ids:
            raise ValueError(f"Graph '{graph_payload['graph_id']}' already exists.")
        payload["graphs"].append(graph_payload)
        self._save_all(payload)
        return graph_payload

    def update_graph(self, graph_id: str, graph_payload: dict[str, Any]) -> dict[str, Any]:
        self._validate_graph_payload(graph_payload)
        payload = self._load_all()
        updated = False
        for index, graph in enumerate(payload["graphs"]):
            if graph["graph_id"] == graph_id:
                payload["graphs"][index] = graph_payload
                updated = True
                break
        if not updated:
            raise KeyError(graph_id)
        self._save_all(payload)
        return graph_payload

    def delete_graph(self, graph_id: str) -> None:
        payload = self._load_all()
        next_graphs = [graph for graph in payload["graphs"] if graph["graph_id"] != graph_id]
        if len(next_graphs) == len(payload["graphs"]):
            raise KeyError(graph_id)
        payload["graphs"] = next_graphs
        self._save_all(payload)

    def catalog(self) -> dict[str, Any]:
        return {
            "node_providers": [
                provider.to_dict() for provider in self.services.node_provider_registry.list_definitions()
            ],
            "tools": [tool.to_dict() for tool in self.services.tool_registry.list_definitions()],
            "connection_rules": [rule.to_dict() for rule in list_connection_rules()],
            "contracts": {
                category.value: contract.to_dict()
                for category, contract in DEFAULT_CATEGORY_CONTRACTS.items()
            },
        }

    def _validate_graph_payload(self, payload: dict[str, Any]) -> None:
        try:
            graph = GraphDefinition.from_dict(payload)
            graph.validate_against_services(self.services)
        except (GraphValidationError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(str(exc)) from exc

    def _ensure_seed_data(self) -> None:
        if self.path.exists():
            return
        self._save_all({"graphs": [build_example_graph_payload()]})

    def _load_all(self) -> dict[str, Any]:
        return json.loads(self.path.read_text())

    def _save_all(self, payload: dict[str, Any]) -> None:
        self.path.write_text(json.dumps(payload, indent=2))
