from __future__ import annotations

from collections import deque
from typing import Any
from collections.abc import Callable
from uuid import uuid4

from graph_agent.runtime.core import (
    API_MESSAGE_HANDLE_ID,
    API_TOOL_CALL_HANDLE_ID,
    Edge,
    GraphDefinition,
    MCP_TERMINAL_OUTPUT_HANDLE_ID,
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
        pending_nodes = deque([{"node_id": graph.start_node_id, "incoming_edge_id": None}])

        self.emit(
            state,
            "run.started",
            f"Run started for graph '{graph.name}'.",
            {"graph_id": graph.graph_id, "graph_name": graph.name},
        )

        step = 0
        while pending_nodes and step < self.max_steps:
            frame = pending_nodes.popleft()
            current_node_id = str(frame["node_id"])
            state.current_node_id = current_node_id
            incoming_edge_id = frame.get("incoming_edge_id")
            state.current_edge_id = str(incoming_edge_id) if incoming_edge_id is not None else None
            node = graph.get_node(current_node_id)
            context = NodeContext(graph=graph, state=state, services=self.services, node_id=node.id)
            if not node.is_ready(context):
                pending_nodes.append(frame)
                step += 1
                continue
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

            try:
                received_input = node.runtime_input_preview(context)
            except Exception:  # noqa: BLE001
                received_input = None
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
                    "received_input": received_input,
                },
            )

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
                    "route_outputs": result.route_outputs,
                    "error": result.error,
                    "metadata": result.metadata,
                },
            )

            if node.kind == "output":
                if self._should_promote_output_result(graph, state, node.id, result):
                    state.final_output = result.output
                if not pending_nodes:
                    state.status = "completed"
                    completion_event = self.emit(
                        state,
                        "run.completed",
                        "Run completed successfully.",
                        {"final_output": state.final_output, "terminal_node_id": node.id},
                    )
                    state.ended_at = completion_event.timestamp
                    return state
                step += 1
                continue

            next_edges = self.select_edges(graph, state, node.id, result)
            if not next_edges:
                return self.fail_run(
                    state,
                    summary=f"No outgoing edge matched after node '{node.id}'.",
                    error={"type": "no_matching_edge", "node_id": node.id},
                )

            for next_edge, edge_result in next_edges:
                if edge_result.output is not None:
                    state.edge_outputs[next_edge.id] = edge_result.output
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
                pending_nodes.append({"node_id": next_edge.target_id, "incoming_edge_id": next_edge.id})
            step += 1

        return self.fail_run(
            state,
            summary="Run exceeded the maximum number of steps.",
            error={"type": "max_steps_exceeded", "max_steps": self.max_steps},
        )

    def select_edges(
        self,
        graph: GraphDefinition,
        state: RunState,
        node_id: str,
        result: NodeExecutionResult,
    ) -> list[tuple[Edge, NodeExecutionResult]]:
        outgoing = graph.get_outgoing_edges(node_id)
        source_node = graph.get_node(node_id)
        if source_node.kind == "mcp_tool_executor":
            remaining_outgoing = [
                edge for edge in outgoing if edge.source_handle_id != MCP_TERMINAL_OUTPUT_HANDLE_ID
            ]
            selected_edges = self._select_matching_edges(state, node_id, remaining_outgoing, result)
            route_result = self._route_result(result, MCP_TERMINAL_OUTPUT_HANDLE_ID)
            if route_result is not None:
                handle_edges = [edge for edge in outgoing if edge.source_handle_id == MCP_TERMINAL_OUTPUT_HANDLE_ID]
                selected_edges.extend(self._select_matching_edges(state, node_id, handle_edges, route_result))
            return selected_edges
        has_explicit_api_outputs = source_node.kind == "model" and any(
            edge.source_handle_id in {API_TOOL_CALL_HANDLE_ID, API_MESSAGE_HANDLE_ID} for edge in outgoing
        )
        if has_explicit_api_outputs:
            selected_edges: list[tuple[Edge, NodeExecutionResult]] = []
            for handle_id in (API_TOOL_CALL_HANDLE_ID, API_MESSAGE_HANDLE_ID):
                route_result = self._route_result(result, handle_id)
                if route_result is None:
                    continue
                handle_edges = [edge for edge in outgoing if edge.source_handle_id == handle_id]
                selected_edges.extend(
                    self._select_matching_edges(
                        state,
                        node_id,
                        handle_edges,
                        route_result,
                        allow_parallel=True,
                    )
                )
            if selected_edges:
                return selected_edges
            outgoing = [edge for edge in outgoing if edge.source_handle_id not in {API_TOOL_CALL_HANDLE_ID, API_MESSAGE_HANDLE_ID}]
        return self._select_matching_edges(state, node_id, outgoing, result)

    def select_edge(
        self,
        graph: GraphDefinition,
        state: RunState,
        node_id: str,
        result: NodeExecutionResult,
    ) -> Edge | None:
        selected = self.select_edges(graph, state, node_id, result)
        return selected[0][0] if selected else None

    def _should_promote_output_result(
        self,
        graph: GraphDefinition,
        state: RunState,
        node_id: str,
        result: NodeExecutionResult,
    ) -> bool:
        if result.output is None:
            return False
        current_edge_id = state.current_edge_id
        if not current_edge_id:
            return True
        incoming_edge = next((edge for edge in graph.get_incoming_edges(node_id) if edge.id == current_edge_id), None)
        if incoming_edge is None:
            return True
        if incoming_edge.source_handle_id == MCP_TERMINAL_OUTPUT_HANDLE_ID:
            return state.final_output is None
        if incoming_edge.source_handle_id == API_MESSAGE_HANDLE_ID:
            edge_output = state.edge_outputs.get(incoming_edge.id)
            if isinstance(edge_output, dict):
                metadata = edge_output.get("metadata")
                if isinstance(metadata, dict) and (
                    metadata.get("should_call_tools") is True or metadata.get("need_tool") is True
                ):
                    return state.final_output is None
        return True

    def _select_matching_edges(
        self,
        state: RunState,
        node_id: str,
        outgoing: list[Edge],
        result: NodeExecutionResult,
        *,
        allow_parallel: bool = False,
    ) -> list[tuple[Edge, NodeExecutionResult]]:
        conditional_edges = [edge for edge in outgoing if edge.kind == "conditional"]
        standard_edges = [edge for edge in outgoing if edge.kind == "standard"]

        matched_conditional_edges: list[tuple[Edge, NodeExecutionResult]] = []
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
                matched_conditional_edges.append((edge, result))
                if not allow_parallel:
                    return matched_conditional_edges

        if matched_conditional_edges:
            return matched_conditional_edges

        if standard_edges:
            if allow_parallel:
                return [(edge, result) for edge in standard_edges]
            return [(standard_edges[0], result)]

        return []

    def _route_result(self, result: NodeExecutionResult, handle_id: str) -> NodeExecutionResult | None:
        if handle_id not in result.route_outputs:
            return None
        route_output = result.route_outputs[handle_id]
        route_metadata = result.metadata
        if isinstance(route_output, dict):
            output_metadata = route_output.get("metadata")
            if isinstance(output_metadata, dict):
                route_metadata = dict(output_metadata)
        return NodeExecutionResult(
            status=result.status,
            output=route_output,
            error=result.error,
            summary=result.summary,
            metadata=dict(route_metadata),
        )

    def _result_contract(self, result: NodeExecutionResult) -> str | None:
        output = result.output
        if not isinstance(output, dict):
            return None
        metadata = output.get("metadata")
        if not isinstance(metadata, dict):
            return None
        contract = metadata.get("contract")
        return contract if isinstance(contract, str) and contract else None

    def fail_run(self, state: RunState, summary: str, error: dict[str, Any]) -> RunState:
        state.status = "failed"
        state.terminal_error = error
        failure_event = self.emit(state, "run.failed", summary, {"error": error})
        state.ended_at = failure_event.timestamp
        return state
