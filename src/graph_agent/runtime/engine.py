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


def _base_node_label(node: Any) -> str:
    explicit_label = str(getattr(node, "label", "") or "").strip()
    if explicit_label:
        return explicit_label
    provider_label = str(getattr(node, "provider_label", "") or "").strip()
    if provider_label:
        return provider_label
    return str(getattr(node, "id", "node"))


def _node_instance_labels(graph: GraphDefinition) -> dict[str, str]:
    groups: dict[str, list[Any]] = {}
    for node in graph.nodes.values():
        base_label = _base_node_label(node)
        groups.setdefault(base_label, []).append(node)
    labels: dict[str, str] = {}
    for base_label, nodes in groups.items():
        if len(nodes) == 1:
            labels[str(nodes[0].id)] = base_label
            continue
        for index, node in enumerate(nodes, start=1):
            labels[str(node.id)] = f"{base_label} {index}"
    return labels


class GraphRuntime:
    def __init__(
        self,
        services: RuntimeServices,
        *,
        max_steps: int,
        max_visits_per_node: int,
        event_listeners: list[Callable[[RuntimeEvent], None]] | None = None,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> None:
        self.services = services
        self.max_steps = max_steps
        self.max_visits_per_node = max_visits_per_node
        self.event_listeners = event_listeners or []
        self.cancel_requested = cancel_requested or (lambda: False)

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

    def _describe_edge(self, edge: Edge, graph: GraphDefinition) -> dict[str, Any]:
        source_node = graph.nodes.get(edge.source_id)
        target_node = graph.nodes.get(edge.target_id)
        instance_labels = _node_instance_labels(graph)
        condition = edge.condition
        return {
            "edge_id": edge.id,
            "kind": edge.kind,
            "label": edge.label or None,
            "source_handle_id": edge.source_handle_id,
            "target_handle_id": edge.target_handle_id,
            "target_node_id": edge.target_id,
            "target_node_label": instance_labels.get(edge.target_id, target_node.label if target_node is not None else edge.target_id),
            "target_node_kind": target_node.kind if target_node is not None else None,
            "source_node_label": instance_labels.get(edge.source_id, source_node.label if source_node is not None else edge.source_id),
            "condition_label": condition.label if condition is not None else None,
            "condition_type": condition.condition_type if condition is not None else None,
            "condition_path": condition.path if condition is not None else None,
            "condition_value": condition.value if condition is not None else None,
        }

    def _no_matching_edge_error(
        self,
        graph: GraphDefinition,
        node_id: str,
        result: NodeExecutionResult,
    ) -> tuple[str, dict[str, Any]]:
        node = graph.nodes.get(node_id)
        instance_labels = _node_instance_labels(graph)
        node_label = instance_labels.get(node_id, node.label if node is not None else node_id)
        outgoing_edges = graph.get_outgoing_edges(node_id)
        available_routes = [self._describe_edge(edge, graph) for edge in outgoing_edges]
        result_contract = self._result_contract(result)
        route_count = len(outgoing_edges)
        if route_count == 0:
            message = f"Node '{node_label}' completed, but it has no outgoing execution edges."
        else:
            route_target_labels = ", ".join(route["target_node_label"] for route in available_routes) or "none"
            contract_clause = f" Output contract was '{result_contract}'." if result_contract else ""
            message = (
                f"Node '{node_label}' completed, but no outgoing edge matched its result."
                f"{contract_clause} Available routes: {route_target_labels}."
            )
        error = {
            "type": "no_matching_edge",
            "node_id": node_id,
            "node_label": node_label,
            "node_kind": node.kind if node is not None else None,
            "node_provider_id": node.provider_id if node is not None else None,
            "node_provider_label": node.provider_label if node is not None else None,
            "result_status": result.status,
            "result_contract": result_contract,
            "available_routes": available_routes,
            "message": message,
        }
        summary = f"No outgoing edge matched after node '{node_label}'."
        return summary, error

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
            if self.cancel_requested():
                return self.cancel_run(state, summary="Run cancelled before starting the next node.")
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

            if self.cancel_requested():
                return self.cancel_run(state, summary=f"Run cancelled before executing node '{node.label}'.")
            try:
                result = node.execute(context)
            except Exception as exc:  # noqa: BLE001
                return self.fail_run(
                    state,
                    summary=f"Node '{node.label}' raised an exception.",
                    error={"type": "node_exception", "node_id": node.id, "message": str(exc)},
                )

            if self.cancel_requested():
                return self.cancel_run(state, summary=f"Run cancelled while node '{node.label}' was executing.")
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
            binding_edges = [
                edge
                for edge in graph.get_outgoing_edges(node.id)
                if edge.kind == "binding"
                and (target := graph.nodes.get(edge.target_id)) is not None
                and getattr(target, "provider_id", None) == "core.context_builder"
            ]
            hold_outgoing = bool(result.metadata.get("hold_outgoing_edges"))
            if hold_outgoing:
                next_edges = []
                binding_edges = []
            if not next_edges and not binding_edges:
                if hold_outgoing:
                    state.current_node_id = None
                    state.current_edge_id = None
                    step += 1
                    continue
                failure_summary, failure_error = self._no_matching_edge_error(graph, node.id, result)
                return self.fail_run(
                    state,
                    summary=failure_summary,
                    error=failure_error,
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
                frame = {"node_id": next_edge.target_id, "incoming_edge_id": next_edge.id}
                target = graph.nodes.get(next_edge.target_id)
                if target is not None and getattr(target, "provider_id", None) == "core.context_builder":
                    pending_nodes.appendleft(frame)
                else:
                    pending_nodes.append(frame)

            for edge in binding_edges:
                state.transition_history.append(
                    TransitionRecord(
                        edge_id=edge.id,
                        source_id=edge.source_id,
                        target_id=edge.target_id,
                    )
                )
                self.emit(
                    state,
                    "edge.selected",
                    f"Transitioning from '{edge.source_id}' to '{edge.target_id}'.",
                    edge.to_dict(),
                )
                pending_nodes.appendleft(
                    {"node_id": edge.target_id, "incoming_edge_id": edge.id},
                )
            step += 1

        if self.cancel_requested():
            return self.cancel_run(state, summary="Run cancelled before completion.")
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

    def cancel_run(self, state: RunState, summary: str) -> RunState:
        state.status = "cancelled"
        state.current_node_id = None
        state.current_edge_id = None
        state.terminal_error = {"type": "run_cancelled", "message": summary}
        cancelled_event = self.emit(state, "run.cancelled", summary, {"error": state.terminal_error})
        state.ended_at = cancelled_event.timestamp
        return state
