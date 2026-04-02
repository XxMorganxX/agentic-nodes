from __future__ import annotations

from typing import Any
from uuid import uuid4


_MISSING = object()


def build_run_state(
    run_id: str,
    graph_id: str,
    input_payload: Any,
    *,
    agent_id: str | None = None,
    parent_run_id: str | None = None,
    agent_name: str | None = None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "graph_id": graph_id,
        "agent_id": agent_id,
        "agent_name": agent_name,
        "parent_run_id": parent_run_id,
        "status": "queued",
        "status_reason": None,
        "input_payload": input_payload,
        "current_node_id": None,
        "current_edge_id": None,
        "started_at": None,
        "ended_at": None,
        "runtime_instance_id": None,
        "last_heartbeat_at": None,
        "node_inputs": {},
        "node_outputs": {},
        "edge_outputs": {},
        "node_errors": {},
        "visit_counts": {},
        "transition_history": [],
        "event_history": [],
        "final_output": None,
        "terminal_error": None,
        "agent_runs": {},
    }


def _omit_run_state_entry(record: dict[str, Any] | None, key: str) -> dict[str, Any]:
    if not record or key not in record:
        return record or {}
    next_record = dict(record)
    next_record.pop(key, None)
    return next_record


def _resolve_edge_output_from_event_history(
    event_history: list[dict[str, Any]],
    edge_payload: dict[str, Any],
) -> Any:
    source_node_id = edge_payload.get("source_id")
    source_handle_id = edge_payload.get("source_handle_id")
    if not isinstance(source_node_id, str) or not source_node_id:
        return _MISSING
    for candidate in reversed(event_history):
        if candidate.get("event_type") != "node.completed":
            continue
        candidate_payload = candidate.get("payload")
        if not isinstance(candidate_payload, dict):
            continue
        if candidate_payload.get("node_id") != source_node_id:
            continue
        if isinstance(source_handle_id, str) and source_handle_id:
            route_outputs = candidate_payload.get("route_outputs")
            if isinstance(route_outputs, dict) and source_handle_id in route_outputs:
                return route_outputs[source_handle_id]
        if "output" in candidate_payload:
            return candidate_payload["output"]
        return _MISSING
    return _MISSING


def apply_single_run_event(previous: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
    next_state = {
        **previous,
        "event_history": [*previous.get("event_history", []), event],
    }
    event_type = event.get("event_type")
    payload = event.get("payload")
    if not isinstance(payload, dict):
        payload = {}

    if event_type == "run.started":
        next_state["status"] = "running"
        next_state["status_reason"] = None
        next_state["started_at"] = event.get("timestamp")

    if event_type == "node.started":
        node_id = payload.get("node_id")
        visit_count = payload.get("visit_count")
        next_state["current_node_id"] = node_id if isinstance(node_id, str) else None
        next_state["current_edge_id"] = None
        if isinstance(node_id, str):
            next_state["visit_counts"] = {
                **next_state.get("visit_counts", {}),
                node_id: int(visit_count) if isinstance(visit_count, int) else 0,
            }
            next_state["node_inputs"] = {
                **next_state.get("node_inputs", {}),
                node_id: payload.get("received_input"),
            }
            next_state["node_errors"] = _omit_run_state_entry(next_state.get("node_errors"), node_id)

    if event_type == "node.completed":
        node_id = payload.get("node_id")
        if next_state.get("current_node_id") == node_id:
            next_state["current_node_id"] = None
        if isinstance(node_id, str) and "output" in payload:
            next_state["node_outputs"] = {
                **next_state.get("node_outputs", {}),
                node_id: payload.get("output"),
            }
        if isinstance(node_id, str) and payload.get("error") is not None:
            next_state["node_errors"] = {
                **next_state.get("node_errors", {}),
                node_id: payload.get("error"),
            }
        elif isinstance(node_id, str):
            next_state["node_errors"] = _omit_run_state_entry(next_state.get("node_errors"), node_id)

    if event_type == "edge.selected":
        selected_edge_id = payload.get("id")
        next_state["current_edge_id"] = selected_edge_id if isinstance(selected_edge_id, str) else None
        selected_edge_output = _resolve_edge_output_from_event_history(previous.get("event_history", []), payload)
        if isinstance(selected_edge_id, str) and selected_edge_output is not _MISSING:
            next_state["edge_outputs"] = {
                **next_state.get("edge_outputs", {}),
                selected_edge_id: selected_edge_output,
            }
        next_state["transition_history"] = [
            *next_state.get("transition_history", []),
            {
                "edge_id": payload.get("id"),
                "source_id": payload.get("source_id"),
                "target_id": payload.get("target_id"),
                "timestamp": event.get("timestamp"),
            },
        ]

    if event_type == "run.completed":
        next_state["status"] = "completed"
        next_state["status_reason"] = None
        next_state["current_node_id"] = None
        next_state["current_edge_id"] = None
        next_state["ended_at"] = event.get("timestamp")
        next_state["final_output"] = payload.get("final_output")

    if event_type == "run.failed":
        next_state["status"] = "failed"
        next_state["status_reason"] = None
        next_state["current_node_id"] = None
        next_state["current_edge_id"] = None
        next_state["ended_at"] = event.get("timestamp")
        next_state["terminal_error"] = payload.get("error")
        if "final_output" in payload:
            next_state["final_output"] = payload.get("final_output")

    if event_type == "run.cancelled":
        next_state["status"] = "cancelled"
        next_state["status_reason"] = None
        next_state["current_node_id"] = None
        next_state["current_edge_id"] = None
        next_state["ended_at"] = event.get("timestamp")
        next_state["terminal_error"] = payload.get("error")
        if "final_output" in payload:
            next_state["final_output"] = payload.get("final_output")

    if event_type == "run.interrupted":
        next_state["status"] = "interrupted"
        next_state["status_reason"] = payload.get("reason")
        next_state["current_node_id"] = None
        next_state["current_edge_id"] = None
        next_state["ended_at"] = event.get("timestamp")
        next_state["terminal_error"] = payload.get("error")
        if "final_output" in payload:
            next_state["final_output"] = payload.get("final_output")

    return next_state


def apply_event(previous: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
    event_type = str(event.get("event_type") or "")
    if not event_type.startswith("agent."):
        return apply_single_run_event(previous, event)

    payload = event.get("payload")
    if not isinstance(payload, dict):
        payload = {}
    next_state = {
        **previous,
        "event_history": [*previous.get("event_history", []), event],
        "agent_runs": dict(previous.get("agent_runs", {})),
    }
    agent_id = str(event.get("agent_id") or payload.get("agent_id") or "")
    if not agent_id:
        return next_state
    agent_state = next_state["agent_runs"].get(agent_id)
    if agent_state is None:
        agent_state = build_run_state(
            str(payload.get("child_run_id") or uuid4()),
            str(previous.get("graph_id") or ""),
            previous.get("input_payload"),
            agent_id=agent_id,
            parent_run_id=str(previous.get("run_id") or ""),
            agent_name=str(payload.get("agent_name") or agent_id),
        )
    normalized_event = {
        **event,
        "event_type": event_type.removeprefix("agent."),
        "run_id": agent_state["run_id"],
    }
    next_state["agent_runs"][agent_id] = apply_single_run_event(agent_state, normalized_event)
    return next_state


def replay_events(initial_state: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any]:
    state = initial_state
    for event in events:
        state = apply_event(state, event)
    return state
