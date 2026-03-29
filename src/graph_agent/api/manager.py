from __future__ import annotations

import json
from queue import Queue
from threading import Lock, Thread
from typing import Any
from uuid import uuid4

from graph_agent.api.graph_store import GraphStore
from graph_agent.examples.tool_schema_repair import build_example_services
from graph_agent.runtime.core import GraphDefinition
from graph_agent.runtime.engine import GraphRuntime
from graph_agent.runtime.serialization import serialize_run_state


class GraphRunManager:
    def __init__(self) -> None:
        self._services = build_example_services()
        self._store = GraphStore(self._services)
        self._lock = Lock()
        self._run_states: dict[str, dict[str, Any]] = {}
        self._event_backlog: dict[str, list[dict[str, Any]]] = {}
        self._subscribers: dict[str, list[Queue[str | None]]] = {}

    def list_graphs(self) -> list[dict[str, Any]]:
        return [self._graph_payload(graph["graph_id"]) for graph in self._store.list_graphs()]

    def get_graph(self, graph_id: str) -> dict[str, Any]:
        return self._graph_payload(graph_id)

    def create_graph(self, graph_payload: dict[str, Any]) -> dict[str, Any]:
        graph = self._store.create_graph(graph_payload)
        return self._decorate_graph(graph)

    def update_graph(self, graph_id: str, graph_payload: dict[str, Any]) -> dict[str, Any]:
        graph = self._store.update_graph(graph_id, graph_payload)
        return self._decorate_graph(graph)

    def delete_graph(self, graph_id: str) -> None:
        self._store.delete_graph(graph_id)

    def get_catalog(self) -> dict[str, Any]:
        return self._store.catalog()

    def get_run(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            if run_id not in self._run_states:
                raise KeyError(run_id)
            return self._run_states[run_id]

    def start_run(self, graph_id: str, input_payload: Any) -> str:
        graph_payload = self._store.get_graph(graph_id)
        graph = GraphDefinition.from_dict(graph_payload)
        graph.validate_against_services(self._services)
        run_id = str(uuid4())

        with self._lock:
            self._run_states[run_id] = {
                "run_id": run_id,
                "graph_id": graph_id,
                "status": "queued",
                "input_payload": input_payload,
                "current_node_id": None,
                "started_at": None,
                "ended_at": None,
                "node_outputs": {},
                "node_errors": {},
                "visit_counts": {},
                "transition_history": [],
                "event_history": [],
                "final_output": None,
                "terminal_error": None,
            }
            self._event_backlog[run_id] = []
            self._subscribers.setdefault(run_id, [])

        runtime = GraphRuntime(
            services=self._services,
            max_steps=self._services.config["max_steps"],
            max_visits_per_node=self._services.config["max_visits_per_node"],
            event_listeners=[lambda event: self._record_event(run_id, event.to_dict())],
        )

        thread = Thread(
            target=self._execute_run,
            args=(runtime, graph, input_payload, run_id),
            daemon=True,
        )
        thread.start()
        return run_id

    def subscribe(self, run_id: str) -> tuple[list[dict[str, Any]], Queue[str | None]]:
        queue: Queue[str | None] = Queue()
        with self._lock:
            if run_id not in self._run_states:
                raise KeyError(run_id)
            backlog = list(self._event_backlog.get(run_id, []))
            self._subscribers.setdefault(run_id, []).append(queue)
            status = self._run_states[run_id]["status"]

        if status in {"completed", "failed"}:
            queue.put(None)

        return backlog, queue

    def unsubscribe(self, run_id: str, queue: Queue[str | None]) -> None:
        with self._lock:
            subscribers = self._subscribers.get(run_id, [])
            if queue in subscribers:
                subscribers.remove(queue)

    def _execute_run(self, runtime: GraphRuntime, graph, input_payload: Any, run_id: str) -> None:
        state = runtime.run(graph, input_payload, run_id=run_id)
        snapshot = serialize_run_state(state)
        with self._lock:
            self._run_states[run_id] = snapshot
        self._close_streams(run_id)

    def _record_event(self, run_id: str, event: dict[str, Any]) -> None:
        encoded = json.dumps(event)
        with self._lock:
            self._event_backlog.setdefault(run_id, []).append(event)
            state = self._run_states.get(run_id)
            if state is not None:
                self._apply_event_to_state(state, event)
            subscribers = list(self._subscribers.get(run_id, []))

        for subscriber in subscribers:
            subscriber.put(encoded)

    def _apply_event_to_state(self, state: dict[str, Any], event: dict[str, Any]) -> None:
        state["event_history"].append(event)
        event_type = event["event_type"]
        payload = event["payload"]

        if event_type == "run.started":
            state["status"] = "running"
            state["started_at"] = event["timestamp"]
        elif event_type == "node.started":
            state["current_node_id"] = payload["node_id"]
            state["visit_counts"][payload["node_id"]] = payload["visit_count"]
        elif event_type == "node.completed":
            if payload.get("output") is not None:
                state["node_outputs"][payload["node_id"]] = payload["output"]
            if payload.get("error") is not None:
                state["node_errors"][payload["node_id"]] = payload["error"]
        elif event_type == "edge.selected":
            state["transition_history"].append(
                {
                    "edge_id": payload["id"],
                    "source_id": payload["source_id"],
                    "target_id": payload["target_id"],
                    "timestamp": event["timestamp"],
                }
            )
        elif event_type == "run.completed":
            state["status"] = "completed"
            state["final_output"] = payload["final_output"]
            state["ended_at"] = event["timestamp"]
        elif event_type == "run.failed":
            state["status"] = "failed"
            state["terminal_error"] = payload["error"]
            state["ended_at"] = event["timestamp"]

    def _close_streams(self, run_id: str) -> None:
        with self._lock:
            subscribers = list(self._subscribers.get(run_id, []))

        for subscriber in subscribers:
            subscriber.put(None)

    def _graph_payload(self, graph_id: str) -> dict[str, Any]:
        graph = self._store.get_graph(graph_id)
        return self._decorate_graph(graph)

    def _decorate_graph(self, graph: dict[str, Any]) -> dict[str, Any]:
        return {
            **graph,
            "node_providers": [
                provider.to_dict() for provider in self._services.node_provider_registry.list_definitions()
            ],
        }
