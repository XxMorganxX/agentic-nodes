from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from graph_agent.api.graph_store import GraphStore
from graph_agent.api.manager import GraphRunManager
from graph_agent.examples.tool_schema_repair import build_example_services
from graph_agent.runtime.core import GraphDefinition, MessageEnvelope, RuntimeServices
from graph_agent.runtime.documents import AgentDefinition, TestEnvironmentDefinition, load_graph_document


def _selected_graph(graph_payload: Mapping[str, Any], agent_id: str | None = None) -> GraphDefinition:
    document = load_graph_document(graph_payload)
    if document.is_multi_agent and len(document.agents) > 1 and not agent_id:
        raise ValueError(
            f"Graph '{document.graph_id}' is already a multi-agent environment. Pass --agent-id to probe one agent."
        )
    return document.as_graph(agent_id)


def bake_graph_into_environment(
    graph_payload: Mapping[str, Any],
    *,
    agent_id: str | None = None,
    baked_graph_id: str | None = None,
) -> dict[str, Any]:
    graph = _selected_graph(graph_payload, agent_id)
    baked_agent_id = agent_id or f"{graph.graph_id}-agent"
    baked_environment = TestEnvironmentDefinition(
        graph_id=baked_graph_id or f"{graph.graph_id}-probe-{uuid4().hex[:8]}",
        name=f"{graph.name} Probe Environment",
        description=f"Temporary MCP probe environment for '{graph.name}'.",
        version=graph.version,
        graph_type="test_environment",
        default_input=graph.default_input,
        env_vars=dict(graph.env_vars),
        agents=[AgentDefinition.from_graph(graph, agent_id=baked_agent_id)],
    )
    return baked_environment.to_dict()


def _graph_tool_candidates(graph: GraphDefinition) -> list[str]:
    candidates: list[str] = []
    for node in graph.nodes.values():
        if node.kind == "mcp_context_provider":
            raw_tool_names = node.config.get("tool_names", [])
            if isinstance(raw_tool_names, list):
                candidates.extend(str(tool_name).strip() for tool_name in raw_tool_names if str(tool_name).strip())
        if node.kind == "model":
            raw_allowed_tool_names = node.config.get("allowed_tool_names", [])
            if isinstance(raw_allowed_tool_names, list):
                candidates.extend(str(tool_name).strip() for tool_name in raw_allowed_tool_names if str(tool_name).strip())
        configured_tool_name = str(node.config.get("tool_name", "") or getattr(node, "tool_name", "")).strip()
        if configured_tool_name:
            candidates.append(configured_tool_name)
    return sorted({candidate for candidate in candidates if candidate})


def _prepare_graph_mcp_tools(manager: GraphRunManager, graph: GraphDefinition) -> dict[str, Any]:
    catalog = manager.get_catalog()
    tool_by_name = {
        str(tool.get("name", "")).strip(): tool
        for tool in catalog.get("tools", [])
        if isinstance(tool, Mapping) and str(tool.get("name", "")).strip()
    }
    requested_tools = _graph_tool_candidates(graph)
    mcp_tools = []
    booted_servers: list[str] = []

    for tool_name in requested_tools:
        tool = tool_by_name.get(tool_name)
        if not tool or str(tool.get("source_type", "")).strip() != "mcp":
            continue
        mcp_tools.append(tool_name)

    for server_id in sorted(
        {
            str(tool_by_name[tool_name].get("server_id", "")).strip()
            for tool_name in mcp_tools
            if tool_name in tool_by_name and str(tool_by_name[tool_name].get("server_id", "")).strip()
        }
    ):
        manager.boot_mcp_server(server_id)
        booted_servers.append(server_id)

    enabled_tools: list[str] = []
    for tool_name in mcp_tools:
        manager.set_mcp_tool_enabled(tool_name, True)
        enabled_tools.append(tool_name)

    return {
        "requested_tools": requested_tools,
        "configured_mcp_tools": mcp_tools,
        "booted_servers": booted_servers,
        "enabled_tools": enabled_tools,
    }


def _wait_for_run_completion(manager: GraphRunManager, run_id: str, timeout_seconds: float) -> dict[str, Any]:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        state = manager.get_run(run_id)
        if state["status"] in {"completed", "failed"}:
            return state
        time.sleep(0.05)
    raise TimeoutError(f"Run '{run_id}' did not finish within {timeout_seconds:.1f}s.")


def _agent_run_state(environment_state: Mapping[str, Any]) -> dict[str, Any]:
    agent_runs = environment_state.get("agent_runs", {})
    if not isinstance(agent_runs, Mapping) or not agent_runs:
        raise ValueError("Baked environment run did not produce any agent runs.")
    return next(iter(agent_runs.values()))


def _is_mapping(value: Any) -> bool:
    return isinstance(value, Mapping)


def _collect_mcp_activity(graph: GraphDefinition, run_state: Mapping[str, Any]) -> dict[str, Any]:
    requested_tool_calls: list[dict[str, Any]] = []
    executor_results: list[dict[str, Any]] = []
    terminal_outputs: list[dict[str, Any]] = []
    seen_tool_call_entries: set[str] = set()
    seen_executor_entries: set[str] = set()

    node_outputs = run_state.get("node_outputs", {})
    if not isinstance(node_outputs, Mapping):
        node_outputs = {}

    for node_id, output in node_outputs.items():
        if not _is_mapping(output):
            continue
        envelope = MessageEnvelope.from_dict(output)
        contract = str(envelope.metadata.get("contract", "")).strip()
        node = graph.nodes.get(str(node_id))
        node_kind = node.kind if node is not None else str(envelope.metadata.get("node_kind", "")).strip()
        node_label = node.label if node is not None else str(node_id)

        if contract == "tool_call_envelope" and envelope.tool_calls:
            for tool_call in envelope.tool_calls:
                signature = json.dumps(
                    {
                        "node_id": str(node_id),
                        "tool_name": str(tool_call.get("tool_name", "")).strip(),
                        "arguments": tool_call.get("arguments"),
                    },
                    sort_keys=True,
                    default=str,
                )
                if signature in seen_tool_call_entries:
                    continue
                seen_tool_call_entries.add(signature)
                requested_tool_calls.append(
                    {
                        "node_id": str(node_id),
                        "node_label": node_label,
                        "tool_name": str(tool_call.get("tool_name", "")).strip(),
                        "arguments": tool_call.get("arguments"),
                    }
                )

        if contract == "tool_result_envelope" and node_kind == "mcp_tool_executor":
            terminal_output_envelope = envelope.artifacts.get("terminal_output_envelope")
            result_entry = {
                "node_id": str(node_id),
                "node_label": node_label,
                "tool_name": str(envelope.metadata.get("tool_name", "")).strip(),
                "tool_status": str(envelope.metadata.get("tool_status", "")).strip(),
                "has_requested_tool_call": bool(envelope.artifacts.get("requested_tool_call")),
                "has_source_tool_call_envelope": bool(envelope.artifacts.get("source_tool_call_envelope")),
                "has_terminal_output_envelope": bool(terminal_output_envelope),
                "error_count": len(envelope.errors),
            }
            result_signature = json.dumps(result_entry, sort_keys=True, default=str)
            if result_signature not in seen_executor_entries:
                seen_executor_entries.add(result_signature)
                executor_results.append(result_entry)
            if _is_mapping(terminal_output_envelope):
                terminal_outputs.append(
                    {
                        "node_id": str(node_id),
                        "node_label": node_label,
                        "tool_name": str(envelope.metadata.get("tool_name", "")).strip(),
                        "terminal_output_envelope": terminal_output_envelope,
                    }
                )
        if node_kind == "mcp_tool_executor":
            follow_up_payload = envelope.artifacts.get("follow_up_payload")
            if _is_mapping(follow_up_payload):
                raw_tool_history = follow_up_payload.get("tool_history", [])
                if isinstance(raw_tool_history, list):
                    for entry in raw_tool_history:
                        if not _is_mapping(entry):
                            continue
                        tool_name = str(entry.get("tool_name", "")).strip()
                        tool_arguments = entry.get("tool_arguments")
                        call_signature = json.dumps(
                            {
                                "node_id": str(node_id),
                                "tool_name": tool_name,
                                "arguments": tool_arguments,
                            },
                            sort_keys=True,
                            default=str,
                        )
                        if tool_name and call_signature not in seen_tool_call_entries:
                            seen_tool_call_entries.add(call_signature)
                            requested_tool_calls.append(
                                {
                                    "node_id": str(node_id),
                                    "node_label": node_label,
                                    "tool_name": tool_name,
                                    "arguments": tool_arguments,
                                }
                            )
                        result_entry = {
                            "node_id": str(node_id),
                            "node_label": node_label,
                            "tool_name": tool_name,
                            "tool_status": str(entry.get("tool_status", "")).strip(),
                            "has_requested_tool_call": entry.get("tool_call") is not None,
                            "has_source_tool_call_envelope": True,
                            "has_terminal_output_envelope": entry.get("terminal_output") is not None,
                            "error_count": len(entry.get("tool_errors", [])) if isinstance(entry.get("tool_errors"), list) else 0,
                        }
                        result_signature = json.dumps(result_entry, sort_keys=True, default=str)
                        if tool_name and result_signature not in seen_executor_entries:
                            seen_executor_entries.add(result_signature)
                            executor_results.append(result_entry)

    return {
        "requested_tool_calls": requested_tool_calls,
        "executor_results": executor_results,
        "terminal_outputs": terminal_outputs,
        "called_tools": sorted(
            {
                result["tool_name"]
                for result in executor_results
                if isinstance(result.get("tool_name"), str) and str(result["tool_name"]).strip()
            }
        ),
    }


def run_baked_environment_probe(
    graph_payload: Mapping[str, Any],
    *,
    input_payload: Any,
    services: RuntimeServices | None = None,
    agent_id: str | None = None,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    services = services or build_example_services()
    baked_environment_payload = bake_graph_into_environment(graph_payload, agent_id=agent_id)
    baked_graph = _selected_graph(baked_environment_payload, baked_environment_payload["agents"][0]["agent_id"])

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        empty_bundled_store = temp_root / "bundled_graphs.json"
        empty_bundled_store.write_text(json.dumps({"graphs": []}, indent=2), encoding="utf-8")
        temp_store = GraphStore(
            services,
            path=temp_root / "graphs_store.json",
            bundled_path=empty_bundled_store,
        )
        manager = GraphRunManager(services=services, store=temp_store)
        try:
            mcp_setup = _prepare_graph_mcp_tools(manager, baked_graph)
            temp_store.create_graph(baked_environment_payload)
            run_id = manager.start_run(baked_environment_payload["graph_id"], input_payload)
            environment_state = _wait_for_run_completion(manager, run_id, timeout_seconds=timeout_seconds)
            agent_state = _agent_run_state(environment_state)
            return {
                "graph_id": str(graph_payload.get("graph_id", "")),
                "baked_environment_graph_id": baked_environment_payload["graph_id"],
                "agent_id": baked_environment_payload["agents"][0]["agent_id"],
                "status": environment_state["status"],
                "agent_status": agent_state["status"],
                "node_errors": agent_state.get("node_errors", {}),
                "visit_counts": agent_state.get("visit_counts", {}),
                "mcp_setup": mcp_setup,
                "mcp_activity": _collect_mcp_activity(baked_graph, agent_state),
            }
        finally:
            manager.stop_background_services()


def run_saved_graph_probe(
    graph_id: str,
    *,
    input_payload: Any,
    services: RuntimeServices | None = None,
    agent_id: str | None = None,
    store_path: Path | None = None,
    bundled_path: Path | None = None,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    services = services or build_example_services()
    source_store = GraphStore(services, path=store_path, bundled_path=bundled_path)
    graph_payload = source_store.get_graph(graph_id)
    return run_baked_environment_probe(
        graph_payload,
        input_payload=input_payload,
        services=services,
        agent_id=agent_id,
        timeout_seconds=timeout_seconds,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Bake a saved graph into a temporary one-agent environment and probe MCP activity.")
    parser.add_argument("--graph-id", required=True, help="Saved graph id to probe.")
    parser.add_argument("--input", default="", help="Run input payload. Defaults to the graph's saved default_input when omitted.")
    parser.add_argument("--agent-id", help="Agent id to probe when the saved graph is already a multi-agent environment.")
    parser.add_argument("--timeout", type=float, default=30.0, help="Seconds to wait for the run to finish.")
    parser.add_argument("--store-path", type=Path, help="Optional override for the user graph store path.")
    parser.add_argument("--bundled-path", type=Path, help="Optional override for the bundled graph store path.")
    args = parser.parse_args()

    services = build_example_services()
    source_store = GraphStore(services, path=args.store_path, bundled_path=args.bundled_path)
    graph_payload = source_store.get_graph(args.graph_id)
    document = load_graph_document(graph_payload)
    selected_graph = _selected_graph(graph_payload, args.agent_id)
    input_payload = args.input or document.default_input or selected_graph.default_input
    summary = run_baked_environment_probe(
        graph_payload,
        input_payload=input_payload,
        services=services,
        agent_id=args.agent_id,
        timeout_seconds=args.timeout,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
