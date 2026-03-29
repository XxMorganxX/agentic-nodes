from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from graph_agent.runtime.core import GraphDefinition, GraphValidationError, RuntimeServices
from graph_agent.runtime.node_providers import DEFAULT_CATEGORY_CONTRACTS, list_connection_rules


class GraphStore:
    def __init__(
        self,
        services: RuntimeServices,
        path: Path | None = None,
        bundled_path: Path | None = None,
    ) -> None:
        self.services = services
        self.bundled_path = bundled_path or Path(__file__).resolve().with_name("graphs_store.json")
        self.path = path or Path(__file__).resolve().parents[3] / ".graph-agent" / "graphs_store.json"
        self._ensure_user_store()

    def list_graphs(self) -> list[dict[str, Any]]:
        return list(self._merged_graphs().values())

    def get_graph(self, graph_id: str) -> dict[str, Any]:
        graph = self._merged_graphs().get(graph_id)
        if graph is None:
            raise KeyError(graph_id)
        return graph

    def create_graph(self, graph_payload: dict[str, Any]) -> dict[str, Any]:
        self._validate_graph_payload(graph_payload)
        payload = self._load_user_all()
        existing_ids = set(self._merged_graphs())
        if graph_payload["graph_id"] in existing_ids:
            raise ValueError(f"Graph '{graph_payload['graph_id']}' already exists.")
        payload["graphs"].append(graph_payload)
        self._save_user_all(payload)
        return graph_payload

    def update_graph(self, graph_id: str, graph_payload: dict[str, Any]) -> dict[str, Any]:
        self._validate_graph_payload(graph_payload)
        if graph_id not in self._merged_graphs():
            raise KeyError(graph_id)

        payload = self._load_user_all()
        updated = False
        for index, graph in enumerate(payload["graphs"]):
            if graph["graph_id"] == graph_id:
                payload["graphs"][index] = graph_payload
                updated = True
                break

        # Persist edits as user-local overrides so built-in sample graphs stay tracked.
        if not updated:
            payload["graphs"].append(graph_payload)

        self._save_user_all(payload)
        return graph_payload

    def delete_graph(self, graph_id: str) -> None:
        payload = self._load_user_all()
        next_graphs = [graph for graph in payload["graphs"] if graph["graph_id"] != graph_id]
        if len(next_graphs) != len(payload["graphs"]):
            payload["graphs"] = next_graphs
            self._save_user_all(payload)
            return

        if graph_id in self._bundled_graph_ids():
            raise ValueError(f"Cannot delete built-in graph '{graph_id}'.")

        if len(next_graphs) == len(payload["graphs"]):
            raise KeyError(graph_id)

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

    def _ensure_user_store(self) -> None:
        if self.path.exists():
            return
        self._save_user_all({"graphs": []})

    def _load_bundled_all(self) -> dict[str, Any]:
        return json.loads(self.bundled_path.read_text())

    def _load_user_all(self) -> dict[str, Any]:
        return json.loads(self.path.read_text())

    def _save_user_all(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2))

    def _bundled_graph_ids(self) -> set[str]:
        return {graph["graph_id"] for graph in self._load_bundled_all()["graphs"]}

    def _merged_graphs(self) -> dict[str, dict[str, Any]]:
        merged = {
            graph["graph_id"]: graph for graph in self._load_bundled_all()["graphs"]
        }
        for graph in self._load_user_all()["graphs"]:
            merged[graph["graph_id"]] = graph
        return merged
