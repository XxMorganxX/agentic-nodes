from __future__ import annotations

from typing import Any

from graph_agent.runtime.core import GraphDefinition, RunState


def serialize_graph(graph: GraphDefinition) -> dict[str, Any]:
    return graph.to_dict()


def serialize_run_state(state: RunState) -> dict[str, Any]:
    return state.snapshot()
