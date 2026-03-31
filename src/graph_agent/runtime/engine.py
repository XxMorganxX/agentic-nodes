from __future__ import annotations

from typing import Any
from collections.abc import Callable
from uuid import uuid4

from graph_agent.runtime.core import (
    API_MESSAGE_HANDLE_ID,
    API_TOOL_CALL_HANDLE_ID,
    Edge,
    GraphDefinition,
    NodeContext,
    NodeExecutionResult,
    RunState,
    RuntimeEvent,
    RuntimeServices,
    TransitionRecord,
)


class GraphRuntime:
    def __init__(
        self,
        services: RuntimeServices,
        *,
        max_steps: int,
        max_visits_per_node: int,
        event_listeners: list[Callable[[RuntimeEvent], None]] | None = None,
    ) -> None:
        self.services = services
        self.max_steps = max_steps
        self.max_visits_per_node = max_visits_per_node
        self.event_listeners = event_listeners or []

    def add_event_listener(self, listener: Callable[[RuntimeEvent], None]) -> None:
        self.event_listeners.append(listener)

    def emit(self, state: RunState, event_type: str, summary: str, payload: dict[str, Any]) -> RuntimeEvent:
        event = RuntimeEvent(
            event_type=event_type,
            summary=summary,
            payload=payload,
            run_id=state.run_id,
        )
        state.event_history.append(event)
        for listener in self.event_listeners:
            listener(event)
        return event

    def run(self, graph: GraphDefinition, input_payload: Any, *, run_id: str | None = None) -> RunState:
        state = RunState(
            graph_id=graph.graph_id,
            input_payload=input_payload,
            run_id=run_id or str(uuid4()),
            status="running",
        )
        current_node_id = graph.start_node_id

        self.emit(
            state,
            "run.started",
            f"Run started for graph '{graph.name}'.",
            {"graph_id": graph.graph_id, "graph_name": graph.name},
        )

        step = 0
        while step < self.max_steps:
            state.current_node_id = current_node_id
            visit_count = state.visit_counts.get(current_node_id, 0) + 1
            state.visit_counts[current_node_id] = visit_count

            if visit_count > self.max_visits_per_node:
                return self.fail_run(
                    state,
                    summary=f"Node '{current_node_id}' exceeded the visit limit.",
                    error={
                        "type": "loop_guard",
                        "node_id": current_node_id,
                        "max_visits_per_node": self.max_visits_per_node,
                    },
                )

            node = graph.get_node(current_node_id)
            self.emit(
                state,
                "node.started",
                f"Started node '{node.label}'.",
                {
                    "node_id": node.id,
                    "node_kind": node.kind,
                    "node_category": node.category.value,
                    "node_provider_id": node.provider_id,
                    "node_provider_label": node.provider_label,
                    "visit_count": visit_count,
                },
            )

            context = NodeContext(graph=graph, state=state, services=self.services, node_id=node.id)

            try:
                result = node.execute(context)
            except Exception as exc:  # noqa: BLE001
                return self.fail_run(
                    state,
                    summary=f"Node '{node.label}' raised an exception.",
                    error={"type": "node_exception", "node_id": node.id, "message": str(exc)},
                )

            if result.error is not None:
                state.node_errors[node.id] = result.error
            if result.output is not None:
                state.node_outputs[node.id] = result.output

            self.emit(
                state,
                "node.completed",
                result.summary or f"Completed node '{node.label}'.",
                {
                    "node_id": node.id,
                    "node_kind": node.kind,
                    "node_category": node.category.value,
                    "node_provider_id": node.provider_id,
                    "node_provider_label": node.provider_label,
                    "status": result.status,
                    "output": result.output,
                    "error": result.error,
                    "metadata": result.metadata,
                },
            )

            if node.kind == "output":
                state.final_output = result.output
                state.status = "completed"
                completion_event = self.emit(
                    state,
                    "run.completed",
                    "Run completed successfully.",
                    {"final_output": state.final_output, "terminal_node_id": node.id},
                )
                state.ended_at = completion_event.timestamp
                return state

            next_edge = self.select_edge(graph, state, node.id, result)
            if next_edge is None:
                return self.fail_run(
                    state,
                    summary=f"No outgoing edge matched after node '{node.id}'.",
                    error={"type": "no_matching_edge", "node_id": node.id},
                )

            state.transition_history.append(
                TransitionRecord(
                    edge_id=next_edge.id,
                    source_id=next_edge.source_id,
                    target_id=next_edge.target_id,
                )
            )
            self.emit(
                state,
                "edge.selected",
                f"Transitioning from '{next_edge.source_id}' to '{next_edge.target_id}'.",
                next_edge.to_dict(),
            )

            current_node_id = next_edge.target_id
            step += 1

        return self.fail_run(
            state,
            summary="Run exceeded the maximum number of steps.",
            error={"type": "max_steps_exceeded", "max_steps": self.max_steps},
        )

    def select_edge(
        self,
        graph: GraphDefinition,
        state: RunState,
        node_id: str,
        result: NodeExecutionResult,
    ) -> Edge | None:
        outgoing = graph.get_outgoing_edges(node_id)
        outgoing = self._filter_api_output_edges(graph, node_id, outgoing, result)
        conditional_edges = [edge for edge in outgoing if edge.kind == "conditional"]
        standard_edges = [edge for edge in outgoing if edge.kind == "standard"]

        for edge in conditional_edges:
            matched = edge.is_match(state, result)
            self.emit(
                state,
                "condition.evaluated",
                f"Condition '{edge.condition.label}' evaluated to {matched}.",
                {
                    "edge_id": edge.id,
                    "condition_id": edge.condition.id if edge.condition else None,
                    "matched": matched,
                },
            )
            if matched:
                if result.status != "success":
                    self.emit(
                        state,
                        "retry.triggered",
                        f"Retry path selected through edge '{edge.id}'.",
                        {"edge_id": edge.id, "node_id": node_id, "result_status": result.status},
                    )
                return edge

        if standard_edges:
            return standard_edges[0]

        return None

    def _result_contract(self, result: NodeExecutionResult) -> str | None:
        output = result.output
        if not isinstance(output, dict):
            return None
        metadata = output.get("metadata")
        if not isinstance(metadata, dict):
            return None
        contract = metadata.get("contract")
        return contract if isinstance(contract, str) and contract else None

    def _filter_api_output_edges(
        self,
        graph: GraphDefinition,
        node_id: str,
        outgoing: list[Edge],
        result: NodeExecutionResult,
    ) -> list[Edge]:
        source_node = graph.get_node(node_id)
        if source_node.kind != "model":
            return outgoing
        if not any(edge.source_handle_id in {API_TOOL_CALL_HANDLE_ID, API_MESSAGE_HANDLE_ID} for edge in outgoing):
            return outgoing
        contract = self._result_contract(result)
        if contract == "tool_call_envelope":
            return [edge for edge in outgoing if edge.source_handle_id != API_MESSAGE_HANDLE_ID]
        if contract == "message_envelope":
            return [edge for edge in outgoing if edge.source_handle_id != API_TOOL_CALL_HANDLE_ID]
        return [edge for edge in outgoing if edge.source_handle_id not in {API_TOOL_CALL_HANDLE_ID, API_MESSAGE_HANDLE_ID}]

    def fail_run(self, state: RunState, summary: str, error: dict[str, Any]) -> RunState:
        state.status = "failed"
        state.terminal_error = error
        failure_event = self.emit(state, "run.failed", summary, {"error": error})
        state.ended_at = failure_event.timestamp
        return state
