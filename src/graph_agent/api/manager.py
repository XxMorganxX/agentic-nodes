from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import os
import shutil
from queue import Queue
from threading import Event, Lock, Thread
from typing import Any
from uuid import uuid4

from graph_agent.api.graph_store import GraphStore
from graph_agent.api.run_store import RunStore, build_default_run_store
from graph_agent.api.run_state_reducer import apply_event, build_run_state
from graph_agent.examples.tool_schema_repair import build_example_services
from graph_agent.providers.discord import DiscordMessageEvent, DiscordTriggerService, normalize_discord_message_payload
from graph_agent.runtime.core import GraphDefinition, resolve_graph_process_env, utc_now_iso
from graph_agent.runtime.documents import AgentDefinition, TestEnvironmentDefinition, load_graph_document
from graph_agent.runtime.engine import GraphRuntime
from graph_agent.tools.mcp import McpServerDefinition


LOGGER = logging.getLogger(__name__)
DISCORD_START_PROVIDER_ID = "start.discord_message"
TERMINAL_RUN_STATUSES = {"completed", "failed", "cancelled", "interrupted"}


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    if value is None:
        return default
    return bool(value)


def _read_interval_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return max(float(raw), 0.1)
    except ValueError:
        return default


@dataclass
class RunControl:
    run_id: str
    cancel_event: Event
    thread: Thread | None = None


class GraphRunManager:
    def __init__(
        self,
        *,
        services: Any | None = None,
        store: GraphStore | None = None,
        run_store: RunStore | None = None,
        run_log_store: RunStore | None = None,
        discord_service: DiscordTriggerService | None = None,
    ) -> None:
        self._services = services or build_example_services(include_user_mcp_servers=True)
        self._store = store or GraphStore(self._services)
        self._lock = Lock()
        self._run_states: dict[str, dict[str, Any]] = {}
        self._run_controls: dict[str, RunControl] = {}
        self._event_backlog: dict[str, list[dict[str, Any]]] = {}
        self._subscribers: dict[str, list[Queue[str | None]]] = {}
        self._run_store = run_store or run_log_store or build_default_run_store()
        self._discord_service = discord_service or DiscordTriggerService(self.handle_discord_message)
        self._runtime_instance_id = str(uuid4())
        self._heartbeat_interval_seconds = _read_interval_env("GRAPH_AGENT_RUN_HEARTBEAT_INTERVAL_SECONDS", 1.0)
        self._heartbeat_timeout_seconds = _read_interval_env("GRAPH_AGENT_RUN_HEARTBEAT_TIMEOUT_SECONDS", 5.0)
        self._heartbeat_stop = Event()
        self._heartbeat_thread: Thread | None = None
        self._start_heartbeat_loop()

    def list_graphs(self) -> list[dict[str, Any]]:
        return [self._graph_payload(graph["graph_id"]) for graph in self._store.list_graphs()]

    def get_graph(self, graph_id: str) -> dict[str, Any]:
        return self._graph_payload(graph_id)

    def create_graph(self, graph_payload: dict[str, Any]) -> dict[str, Any]:
        graph = self._store.create_graph(graph_payload)
        self._sync_discord_service()
        return self._decorate_graph(graph)

    def update_graph(self, graph_id: str, graph_payload: dict[str, Any]) -> dict[str, Any]:
        graph = self._store.update_graph(graph_id, graph_payload)
        self._sync_discord_service()
        return self._decorate_graph(graph)

    def delete_graph(self, graph_id: str) -> None:
        self._store.delete_graph(graph_id)
        self._sync_discord_service()

    def get_catalog(self) -> dict[str, Any]:
        provider_statuses: dict[str, Any] = {}
        for provider_name, provider in self._services.model_providers.items():
            result = provider.preflight()
            provider_statuses[provider_name] = {
                "status": result.status,
                "ok": result.ok,
                "message": result.message,
                "warnings": result.warnings,
                "details": result.details,
            }
        return {
            **self._store.catalog(),
            "provider_statuses": provider_statuses,
            "mcp_servers": (
                self._services.mcp_server_manager.list_servers() if self._services.mcp_server_manager is not None else []
            ),
        }

    def boot_mcp_server(self, server_id: str) -> dict[str, Any]:
        if self._services.mcp_server_manager is None:
            raise RuntimeError("MCP server manager is not configured.")
        return self._services.mcp_server_manager.boot_server(server_id)

    def stop_mcp_server(self, server_id: str) -> dict[str, Any]:
        if self._services.mcp_server_manager is None:
            raise RuntimeError("MCP server manager is not configured.")
        return self._services.mcp_server_manager.stop_server(server_id)

    def refresh_mcp_server(self, server_id: str) -> dict[str, Any]:
        if self._services.mcp_server_manager is None:
            raise RuntimeError("MCP server manager is not configured.")
        return self._services.mcp_server_manager.refresh_server(server_id)

    def set_mcp_tool_enabled(self, tool_name: str, enabled: bool) -> dict[str, Any]:
        if self._services.mcp_server_manager is None:
            raise RuntimeError("MCP server manager is not configured.")
        return self._services.mcp_server_manager.set_tool_enabled(tool_name, enabled)

    def create_mcp_server(self, server_payload: dict[str, Any]) -> dict[str, Any]:
        if self._services.mcp_server_manager is None:
            raise RuntimeError("MCP server manager is not configured.")
        definition = McpServerDefinition.from_dict(server_payload)
        return self._services.mcp_server_manager.create_server(definition)

    def update_mcp_server(self, server_id: str, server_payload: dict[str, Any]) -> dict[str, Any]:
        if self._services.mcp_server_manager is None:
            raise RuntimeError("MCP server manager is not configured.")
        definition = McpServerDefinition.from_dict(server_payload)
        return self._services.mcp_server_manager.update_server(server_id, definition)

    def delete_mcp_server(self, server_id: str) -> None:
        if self._services.mcp_server_manager is None:
            raise RuntimeError("MCP server manager is not configured.")
        self._services.mcp_server_manager.delete_server(server_id)

    def test_mcp_server(self, server_payload: dict[str, Any]) -> dict[str, Any]:
        if self._services.mcp_server_manager is None:
            raise RuntimeError("MCP server manager is not configured.")
        definition = McpServerDefinition.from_dict(server_payload)
        return self._services.mcp_server_manager.validate_server(definition)

    def preflight_provider(
        self,
        provider_name: str,
        provider_config: dict[str, Any] | None = None,
        *,
        live: bool = False,
    ) -> dict[str, Any]:
        provider = self._services.model_providers[provider_name]
        config = dict(provider_config or {})
        if live:
            config["check_auth"] = True
        result = provider.preflight(config)
        return {
            "provider_name": provider_name,
            "status": result.status,
            "ok": result.ok,
            "message": result.message,
            "warnings": result.warnings,
            "details": result.details,
        }

    def provider_diagnostics(
        self,
        provider_name: str,
        provider_config: dict[str, Any] | None = None,
        *,
        live: bool = False,
    ) -> dict[str, Any]:
        config = dict(provider_config or {})
        if live:
            config["check_auth"] = True

        preflight = self.preflight_provider(provider_name, config, live=live)
        cli_path = str(config.get("cli_path") or "claude").strip() or "claude"
        anthropic_api_key_present = bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())
        is_mock = provider_name == "mock"
        is_claude_code = provider_name == "claude_code"
        is_anthropic_api = provider_name == "claude"
        claude_binary_path = shutil.which(cli_path) if is_claude_code else None

        warning = None
        if is_claude_code and anthropic_api_key_present:
            warning = (
                "ANTHROPIC_API_KEY is present in the host environment. Claude Code may use API billing instead of subscription "
                "auth if that variable reaches the child process. This app strips it for Claude Code runs."
            )

        auth_status = "not_checked"
        if is_mock:
            auth_status = "not_applicable"
        elif live:
            auth_status = preflight["status"]
        elif is_anthropic_api:
            auth_status = "api_key_present" if anthropic_api_key_present else "missing_api_key"

        return {
            "provider_name": provider_name,
            "active_backend": "claude_code" if is_claude_code else "anthropic_api" if is_anthropic_api else provider_name,
            "claude_binary_exists": bool(claude_binary_path),
            "claude_binary_path": claude_binary_path,
            "anthropic_api_key_present": anthropic_api_key_present,
            "warning": warning,
            "child_env_sanitized": is_claude_code,
            "sanitized_env_removed_vars": ["ANTHROPIC_API_KEY"] if is_claude_code else [],
            "authentication_status": auth_status,
            "preflight": preflight,
        }

    def get_run(self, run_id: str) -> dict[str, Any]:
        recovered = self._recover_run_state(run_id)
        if recovered is not None:
            reconciled = self._reconcile_run_state(run_id)
            return reconciled if reconciled is not None else recovered
        with self._lock:
            if run_id not in self._run_states:
                raise KeyError(run_id)
            return self._run_states[run_id]

    def list_runs(self, graph_id: str | None = None, *, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._run_store.list_runs(graph_id=graph_id, limit=limit)
        history: list[dict[str, Any]] = []
        for row in rows:
            run_id = str(row.get("run_id") or "")
            if not run_id:
                continue
            state = self._recover_run_state(run_id)
            state = self._reconcile_run_state(run_id) if state is not None else None
            if state is None:
                history.append(row)
                continue
            history.append(
                {
                    **row,
                    "run_id": state.get("run_id"),
                    "graph_id": state.get("graph_id"),
                    "status": state.get("status"),
                    "status_reason": state.get("status_reason"),
                    "started_at": state.get("started_at"),
                    "ended_at": state.get("ended_at"),
                    "agent_id": state.get("agent_id"),
                    "agent_name": state.get("agent_name"),
                    "parent_run_id": state.get("parent_run_id"),
                    "runtime_instance_id": state.get("runtime_instance_id"),
                    "last_heartbeat_at": state.get("last_heartbeat_at"),
                }
            )
        return history

    def start_run(self, graph_id: str, input_payload: Any, agent_ids: list[str] | None = None) -> str:
        self._start_heartbeat_loop()
        document = load_graph_document(self._store.get_graph(graph_id))
        document.validate_against_services(self._services)
        run_id = str(uuid4())
        selected_agents = self._resolve_environment_agents(document, agent_ids) if document.is_multi_agent else None

        with self._lock:
            self._run_controls[run_id] = RunControl(run_id=run_id, cancel_event=Event())
            self._event_backlog[run_id] = []
            self._subscribers.setdefault(run_id, [])
            if document.is_multi_agent:
                self._run_states[run_id] = self._build_environment_run_state(run_id, document, input_payload, selected_agents or [])
            else:
                default_agent = document.agents[0]
                self._run_states[run_id] = self._build_run_state(
                    run_id,
                    graph_id,
                    input_payload,
                    agent_id=default_agent.agent_id,
                    agent_name=default_agent.name,
                )
            states_to_initialize = [self._run_states[run_id], *self._run_states[run_id]["agent_runs"].values()]

        for state in states_to_initialize:
            self._run_store.initialize_run(state)
        self._touch_run_liveness(run_id)

        if document.is_multi_agent:
            thread = Thread(
                target=self._execute_environment_run,
                args=(document, selected_agents or [], input_payload, run_id),
                daemon=True,
            )
        else:
            graph = document.as_graph()
            runtime = GraphRuntime(
                services=self._services,
                max_steps=self._services.config["max_steps"],
                max_visits_per_node=self._services.config["max_visits_per_node"],
                event_listeners=[lambda event: self._record_event(run_id, event.to_dict())],
                cancel_requested=lambda current_run_id=run_id: self._cancel_requested(current_run_id),
            )
            thread = Thread(
                target=self._execute_run,
                args=(runtime, graph, input_payload, run_id),
                daemon=True,
            )

        with self._lock:
            control = self._run_controls.get(run_id)
            if control is not None:
                control.thread = thread
        thread.start()
        return run_id

    def subscribe(self, run_id: str) -> tuple[list[dict[str, Any]], Queue[str | None]]:
        queue: Queue[str | None] = Queue()
        self._recover_run_state(run_id)
        self._reconcile_run_state(run_id)
        with self._lock:
            if run_id not in self._run_states:
                raise KeyError(run_id)
            backlog = list(self._event_backlog.get(run_id, []))
            self._subscribers.setdefault(run_id, []).append(queue)
            status = self._run_states[run_id]["status"]

        if status in TERMINAL_RUN_STATUSES:
            queue.put(None)

        return backlog, queue

    def unsubscribe(self, run_id: str, queue: Queue[str | None]) -> None:
        with self._lock:
            subscribers = self._subscribers.get(run_id, [])
            if queue in subscribers:
                subscribers.remove(queue)

    def start_background_services(self) -> None:
        self._start_heartbeat_loop()
        self._reconcile_persisted_runs()
        self._sync_discord_service()
        if self._services.mcp_server_manager is not None:
            self._services.mcp_server_manager.start_auto_boot()

    def stop_background_services(self) -> None:
        self._heartbeat_stop.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=self._heartbeat_interval_seconds + 1.0)
            self._heartbeat_thread = None
        self._discord_service.stop()
        if self._services.mcp_server_manager is not None:
            self._services.mcp_server_manager.shutdown_all(preserve_desired_running=True)

    def reset_runtime(self) -> dict[str, Any]:
        with self._lock:
            active_controls = [
                control
                for control in self._run_controls.values()
                if control.thread is not None and control.thread.is_alive()
            ]
            for control in active_controls:
                control.cancel_event.set()

        running_server_ids: list[str] = []
        if self._services.mcp_server_manager is not None:
            running_server_ids = [
                str(server["server_id"])
                for server in self._services.mcp_server_manager.list_servers()
                if bool(server.get("running"))
            ]

        self._discord_service.stop()
        if self._services.mcp_server_manager is not None:
            self._services.mcp_server_manager.shutdown_all(preserve_desired_running=False)

        return {
            "cancelled_run_ids": [control.run_id for control in active_controls],
            "cancelled_run_count": len(active_controls),
            "stopped_mcp_server_ids": running_server_ids,
            "stopped_mcp_server_count": len(running_server_ids),
            "discord_stopped": True,
        }

    def handle_discord_message(self, message: DiscordMessageEvent) -> list[str]:
        run_ids: list[str] = []
        input_payload = normalize_discord_message_payload(message)
        for graph in self._iter_discord_trigger_graphs():
            if not self._graph_matches_discord_message(graph, message):
                continue
            try:
                run_ids.append(self.start_run(graph.graph_id, dict(input_payload)))
            except Exception:  # noqa: BLE001
                LOGGER.exception("Failed to start Discord-triggered run for graph '%s'.", graph.graph_id)
        return run_ids

    def _execute_run(self, runtime: GraphRuntime, graph: GraphDefinition, input_payload: Any, run_id: str) -> None:
        try:
            runtime.run(graph, input_payload, run_id=run_id)
            with self._lock:
                snapshot = deepcopy(self._run_states[run_id])
            self._run_store.write_state(run_id, snapshot)
            self._close_streams(run_id)
        finally:
            self._release_run_control(run_id)

    def _execute_environment_run(
        self,
        document: TestEnvironmentDefinition,
        agents: list[AgentDefinition],
        input_payload: Any,
        parent_run_id: str,
    ) -> None:
        try:
            self._record_event(
                parent_run_id,
                {
                    "event_type": "run.started",
                    "summary": f"Run started for environment '{document.name}'.",
                    "payload": {
                        "graph_id": document.graph_id,
                        "graph_name": document.name,
                        "agent_count": len(agents),
                    },
                    "run_id": parent_run_id,
                    "timestamp": utc_now_iso(),
                    "agent_id": None,
                    "parent_run_id": None,
                },
            )

            final_output: dict[str, Any] = {}
            failed_agent_ids: list[str] = []
            for agent in agents:
                if self._cancel_requested(parent_run_id):
                    self._record_run_cancelled(
                        parent_run_id,
                        f"Environment run '{document.name}' was cancelled before starting the next agent.",
                    )
                    with self._lock:
                        snapshot = deepcopy(self._run_states[parent_run_id])
                    self._run_store.write_state(parent_run_id, snapshot)
                    self._close_streams(parent_run_id)
                    return
                child_snapshot = self._run_agent_in_environment(document, agent, input_payload, parent_run_id)
                final_output[agent.agent_id] = child_snapshot.get("final_output")
                if child_snapshot.get("status") == "cancelled":
                    self._record_run_cancelled(
                        parent_run_id,
                        f"Environment run '{document.name}' was cancelled.",
                        agent_id=agent.agent_id,
                    )
                    with self._lock:
                        snapshot = deepcopy(self._run_states[parent_run_id])
                    self._run_store.write_state(parent_run_id, snapshot)
                    self._close_streams(parent_run_id)
                    return
                if child_snapshot.get("status") == "failed":
                    failed_agent_ids.append(agent.agent_id)
            self._record_event(
                parent_run_id,
                {
                    "event_type": "run.failed" if failed_agent_ids else "run.completed",
                    "summary": "Environment run finished with agent failures." if failed_agent_ids else "Environment run completed successfully.",
                    "payload": (
                        {"error": {"type": "agent_failure", "agent_ids": failed_agent_ids}, "final_output": final_output}
                        if failed_agent_ids
                        else {"final_output": final_output}
                    ),
                    "run_id": parent_run_id,
                    "timestamp": utc_now_iso(),
                    "agent_id": None,
                    "parent_run_id": None,
                },
            )
            with self._lock:
                snapshot = deepcopy(self._run_states[parent_run_id])
            self._run_store.write_state(parent_run_id, snapshot)
            self._close_streams(parent_run_id)
        except Exception as exc:  # noqa: BLE001
            self._record_event(
                parent_run_id,
                {
                    "event_type": "run.failed",
                    "summary": "Environment run failed unexpectedly.",
                    "payload": {"error": {"type": "environment_exception", "message": str(exc)}},
                    "run_id": parent_run_id,
                    "timestamp": utc_now_iso(),
                    "agent_id": None,
                    "parent_run_id": None,
                },
            )
            with self._lock:
                snapshot = deepcopy(self._run_states[parent_run_id])
            self._run_store.write_state(parent_run_id, snapshot)
            self._close_streams(parent_run_id)
        finally:
            self._release_run_control(parent_run_id)

    def _run_agent_in_environment(
        self,
        document: TestEnvironmentDefinition,
        agent: AgentDefinition,
        input_payload: Any,
        parent_run_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            parent_state = self._run_states[parent_run_id]
            child_state = parent_state["agent_runs"][agent.agent_id]
            child_run_id = child_state["run_id"]

        runtime = GraphRuntime(
            services=self._services,
            max_steps=self._services.config["max_steps"],
            max_visits_per_node=self._services.config["max_visits_per_node"],
            event_listeners=[
                lambda event, current_agent_id=agent.agent_id, current_agent_name=agent.name: self._record_agent_event(
                    parent_run_id,
                    current_agent_id,
                    current_agent_name,
                    event.to_dict(),
                )
            ],
            cancel_requested=lambda current_run_id=parent_run_id: self._cancel_requested(current_run_id),
        )
        graph = agent.to_graph(graph_id=document.graph_id, shared_env_vars=document.env_vars)
        runtime.run(graph, input_payload, run_id=child_run_id)

        with self._lock:
            snapshot = deepcopy(self._run_states[parent_run_id]["agent_runs"][agent.agent_id])
            self._run_states[child_run_id] = snapshot

        self._run_store.write_state(child_run_id, snapshot)
        return snapshot

    def _record_event(self, run_id: str, event: dict[str, Any]) -> None:
        encoded = json.dumps(event)
        parent_snapshot: dict[str, Any] | None = None
        child_run_id: str | None = None
        child_event: dict[str, Any] | None = None
        child_snapshot: dict[str, Any] | None = None
        with self._lock:
            self._event_backlog.setdefault(run_id, []).append(event)
            state = self._run_states.get(run_id)
            if state is not None:
                next_state = apply_event(state, event)
                self._run_states[run_id] = next_state
                parent_snapshot = deepcopy(next_state)
                if event["event_type"].startswith("agent."):
                    payload = event.get("payload", {})
                    agent_id = str(event.get("agent_id") or payload.get("agent_id") or "")
                    agent_state = next_state["agent_runs"].get(agent_id) if agent_id else None
                    if agent_state is not None:
                        child_run_id = str(agent_state["run_id"])
                        self._run_states[child_run_id] = deepcopy(agent_state)
                        child_event = {
                            **event,
                            "event_type": event["event_type"].removeprefix("agent."),
                            "run_id": child_run_id,
                        }
                        child_snapshot = deepcopy(agent_state)
            subscribers = list(self._subscribers.get(run_id, []))

        self._run_store.append_event(run_id, event)
        if parent_snapshot is not None:
            self._run_store.write_state(run_id, parent_snapshot)
        if child_run_id is not None and child_event is not None and child_snapshot is not None:
            self._run_store.append_event(child_run_id, child_event)
            self._run_store.write_state(child_run_id, child_snapshot)

        for subscriber in subscribers:
            subscriber.put(encoded)

    def _record_agent_event(
        self,
        parent_run_id: str,
        agent_id: str,
        agent_name: str,
        event: dict[str, Any],
    ) -> None:
        child_run_id = str(event.get("run_id", ""))
        wrapped_event = {
            **event,
            "event_type": f"agent.{event['event_type']}",
            "run_id": parent_run_id,
            "agent_id": agent_id,
            "parent_run_id": parent_run_id,
            "summary": f"[{agent_name}] {event['summary']}",
            "payload": {
                **event.get("payload", {}),
                "agent_id": agent_id,
                "agent_name": agent_name,
                "child_run_id": child_run_id,
            },
        }
        self._record_event(parent_run_id, wrapped_event)

    def _close_streams(self, run_id: str) -> None:
        with self._lock:
            subscribers = list(self._subscribers.get(run_id, []))

        for subscriber in subscribers:
            subscriber.put(None)

    def _cancel_requested(self, run_id: str) -> bool:
        with self._lock:
            control = self._run_controls.get(run_id)
            return bool(control is not None and control.cancel_event.is_set())

    def _release_run_control(self, run_id: str) -> None:
        with self._lock:
            self._run_controls.pop(run_id, None)

    def _record_run_cancelled(self, run_id: str, summary: str, *, agent_id: str | None = None) -> None:
        error_payload = {"type": "run_cancelled", "message": summary}
        if agent_id is not None:
            error_payload["agent_id"] = agent_id
        self._record_event(
            run_id,
            {
                "event_type": "run.cancelled",
                "summary": summary,
                "payload": {"error": error_payload},
                "run_id": run_id,
                "timestamp": utc_now_iso(),
                "agent_id": None,
                "parent_run_id": None,
            },
        )

    def _start_heartbeat_loop(self) -> None:
        if self._heartbeat_thread is not None and self._heartbeat_thread.is_alive():
            return
        if self._heartbeat_stop.is_set():
            self._heartbeat_stop = Event()
        self._heartbeat_thread = Thread(target=self._heartbeat_loop, name="graph-run-heartbeat", daemon=True)
        self._heartbeat_thread.start()

    def _heartbeat_loop(self) -> None:
        while not self._heartbeat_stop.wait(self._heartbeat_interval_seconds):
            self._refresh_active_run_heartbeats()

    def _refresh_active_run_heartbeats(self) -> None:
        with self._lock:
            active_run_ids = [
                control.run_id
                for control in self._run_controls.values()
                if control.thread is not None and control.thread.is_alive()
            ]
        if not active_run_ids:
            return
        heartbeat_at = utc_now_iso()
        for run_id in active_run_ids:
            self._touch_run_liveness(run_id, heartbeat_at)

    def _touch_run_liveness(self, run_id: str, heartbeat_at: str | None = None) -> None:
        heartbeat_value = heartbeat_at or utc_now_iso()
        parent_snapshot: dict[str, Any] | None = None
        child_snapshots: dict[str, dict[str, Any]] = {}
        with self._lock:
            state = self._run_states.get(run_id)
            if state is None:
                return
            next_state = deepcopy(state)
            self._stamp_run_liveness(next_state, heartbeat_value)
            for agent_id, agent_state in next_state.get("agent_runs", {}).items():
                self._stamp_run_liveness(agent_state, heartbeat_value)
                child_run_id = str(agent_state.get("run_id") or "")
                if child_run_id:
                    child_snapshots[child_run_id] = deepcopy(agent_state)
                    self._run_states[child_run_id] = deepcopy(agent_state)
                next_state["agent_runs"][agent_id] = agent_state
            self._run_states[run_id] = next_state
            parent_snapshot = deepcopy(next_state)
        if parent_snapshot is not None:
            self._run_store.write_state(run_id, parent_snapshot)
        for child_run_id, snapshot in child_snapshots.items():
            self._run_store.write_state(child_run_id, snapshot)

    def _stamp_run_liveness(self, state: dict[str, Any], heartbeat_at: str) -> None:
        status = str(state.get("status") or "")
        if status in TERMINAL_RUN_STATUSES:
            return
        state["runtime_instance_id"] = self._runtime_instance_id
        state["last_heartbeat_at"] = heartbeat_at

    def _reconcile_persisted_runs(self, *, limit: int = 200) -> None:
        for row in self._run_store.list_runs(limit=limit):
            run_id = str(row.get("run_id") or "")
            if not run_id or row.get("status") != "running":
                continue
            self._recover_run_state(run_id)
            self._reconcile_run_state(run_id)

    def _reconcile_run_state(self, run_id: str) -> dict[str, Any] | None:
        if self._has_live_run_control(run_id):
            self._touch_run_liveness(run_id)
            with self._lock:
                current = self._run_states.get(run_id)
                return deepcopy(current) if current is not None else None

        with self._lock:
            current = deepcopy(self._run_states.get(run_id))
        if current is None:
            return None
        if current.get("status") != "running":
            return current
        if not self._heartbeat_expired(current.get("last_heartbeat_at")):
            return current

        self._interrupt_stale_run(current)
        with self._lock:
            interrupted = self._run_states.get(run_id)
            return deepcopy(interrupted) if interrupted is not None else None

    def _has_live_run_control(self, run_id: str) -> bool:
        with self._lock:
            control = self._run_controls.get(run_id)
            return bool(control is not None and control.thread is not None and control.thread.is_alive())

    def _heartbeat_expired(self, timestamp: Any) -> bool:
        if not isinstance(timestamp, str) or not timestamp.strip():
            return True
        try:
            last_heartbeat = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError:
            return True
        if last_heartbeat.tzinfo is None:
            last_heartbeat = last_heartbeat.replace(tzinfo=timezone.utc)
        age_seconds = (datetime.now(timezone.utc) - last_heartbeat).total_seconds()
        return age_seconds > self._heartbeat_timeout_seconds

    def _interrupt_stale_run(self, snapshot: dict[str, Any]) -> None:
        run_id = str(snapshot.get("run_id") or "")
        if not run_id:
            return
        if snapshot.get("agent_runs"):
            for agent_id, agent_state in snapshot.get("agent_runs", {}).items():
                if agent_state.get("status") != "running":
                    continue
                self._record_stale_agent_interruption(run_id, str(agent_id), agent_state)
        with self._lock:
            current = self._run_states.get(run_id)
            if current is None or current.get("status") != "running":
                return
        self._record_event(run_id, self._build_interrupted_event(snapshot))

    def _record_stale_agent_interruption(self, parent_run_id: str, agent_id: str, agent_state: dict[str, Any]) -> None:
        with self._lock:
            parent_state = self._run_states.get(parent_run_id)
            current_agent_state = parent_state.get("agent_runs", {}).get(agent_id) if parent_state is not None else None
            if current_agent_state is None or current_agent_state.get("status") != "running":
                return
        event = self._build_interrupted_event(agent_state)
        self._record_event(
            parent_run_id,
            {
                **event,
                "event_type": "agent.run.interrupted",
                "run_id": parent_run_id,
                "agent_id": agent_id,
                "parent_run_id": parent_run_id,
                "summary": f"[{agent_state.get('agent_name') or agent_id}] {event['summary']}",
                "payload": {
                    **event["payload"],
                    "agent_id": agent_id,
                    "agent_name": agent_state.get("agent_name") or agent_id,
                    "child_run_id": agent_state.get("run_id"),
                },
            },
        )

    def _build_interrupted_event(self, state: dict[str, Any]) -> dict[str, Any]:
        runtime_instance_id = state.get("runtime_instance_id")
        last_heartbeat_at = state.get("last_heartbeat_at")
        detail_parts = []
        if isinstance(runtime_instance_id, str) and runtime_instance_id:
            detail_parts.append(f"runtime instance {runtime_instance_id}")
        if isinstance(last_heartbeat_at, str) and last_heartbeat_at:
            detail_parts.append(f"last heartbeat {last_heartbeat_at}")
        detail_suffix = f" ({', '.join(detail_parts)})" if detail_parts else ""
        return {
            "event_type": "run.interrupted",
            "summary": f"Run interrupted after runtime heartbeat expired{detail_suffix}.",
            "payload": {
                "reason": "runtime_heartbeat_expired",
                "error": {
                    "type": "runtime_interrupted",
                    "message": "The runtime process stopped updating this run heartbeat before the run completed.",
                    "runtime_instance_id": runtime_instance_id,
                    "last_heartbeat_at": last_heartbeat_at,
                },
            },
            "run_id": state.get("run_id"),
            "timestamp": utc_now_iso(),
            "agent_id": state.get("agent_id"),
            "parent_run_id": state.get("parent_run_id"),
        }

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

    def _iter_discord_trigger_graphs(self) -> list[GraphDefinition]:
        graphs: list[GraphDefinition] = []
        for graph_payload in self._store.list_graphs():
            try:
                document = load_graph_document(graph_payload)
                document.validate_against_services(self._services)
            except Exception:  # noqa: BLE001
                LOGGER.warning("Skipping invalid graph '%s' while syncing Discord triggers.", graph_payload.get("graph_id"))
                continue
            if document.is_multi_agent:
                continue
            graph = document.as_graph()
            if graph.start_node().provider_id == DISCORD_START_PROVIDER_ID:
                graphs.append(graph)
        return graphs

    def _graph_matches_discord_message(self, graph: GraphDefinition, message: DiscordMessageEvent) -> bool:
        start_node = graph.start_node()
        if start_node.provider_id != DISCORD_START_PROVIDER_ID:
            return False
        config = graph.resolved_start_node_config()
        channel_id = str(config.get("discord_channel_id", "") or "").strip()
        if not channel_id or channel_id != message.channel_id:
            return False
        if _as_bool(config.get("ignore_bot_messages"), True) and message.author_is_bot:
            return False
        if _as_bool(config.get("ignore_self_messages"), True) and message.author_is_self:
            return False
        return True

    def _sync_discord_service(self) -> None:
        token = self._resolve_discord_service_token()
        if not token:
            self._discord_service.stop()
            return
        try:
            self._discord_service.start(token)
        except RuntimeError:
            LOGGER.exception("Unable to start Discord trigger service.")

    def _resolve_discord_service_token(self) -> str:
        tokens: set[str] = set()
        for graph in self._iter_discord_trigger_graphs():
            config = graph.resolved_start_node_config()
            token = resolve_graph_process_env(
                str(config.get("discord_bot_token_env_var", "{DISCORD_BOT_TOKEN}") or "{DISCORD_BOT_TOKEN}"),
                graph.env_vars,
            ).strip()
            if token:
                tokens.add(token)
        if not tokens:
            return ""
        if len(tokens) > 1:
            LOGGER.warning("Multiple Discord bot tokens are configured across graphs; Discord triggers are disabled.")
            return ""
        return next(iter(tokens))

    def _build_environment_run_state(
        self,
        run_id: str,
        document: TestEnvironmentDefinition,
        input_payload: Any,
        agents: list[AgentDefinition],
    ) -> dict[str, Any]:
        state = self._build_run_state(run_id, document.graph_id, input_payload)
        state["agent_runs"] = {
            agent.agent_id: self._build_run_state(
                str(uuid4()),
                document.graph_id,
                input_payload,
                agent_id=agent.agent_id,
                parent_run_id=run_id,
                agent_name=agent.name,
            )
            for agent in agents
        }
        return state

    def _resolve_environment_agents(
        self,
        document: TestEnvironmentDefinition,
        requested_agent_ids: list[str] | None,
    ) -> list[AgentDefinition]:
        if requested_agent_ids is None:
            return list(document.agents)
        normalized_ids = [str(agent_id).strip() for agent_id in requested_agent_ids if str(agent_id).strip()]
        if not normalized_ids:
            raise ValueError("Select at least one agent before running the environment.")
        requested_id_set = set(normalized_ids)
        available_ids = {agent.agent_id for agent in document.agents}
        unknown_ids = sorted(requested_id_set - available_ids)
        if unknown_ids:
            raise ValueError(f"Unknown environment agent selection: {', '.join(unknown_ids)}.")
        return [agent for agent in document.agents if agent.agent_id in requested_id_set]

    def _build_run_state(
        self,
        run_id: str,
        graph_id: str,
        input_payload: Any,
        *,
        agent_id: str | None = None,
        parent_run_id: str | None = None,
        agent_name: str | None = None,
    ) -> dict[str, Any]:
        return build_run_state(
            run_id,
            graph_id,
            input_payload,
            agent_id=agent_id,
            parent_run_id=parent_run_id,
            agent_name=agent_name,
        )

    def _recover_run_state(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            existing = self._run_states.get(run_id)
        if existing is not None:
            return existing
        snapshot = self._run_store.recover_run_state(run_id)
        if snapshot is None:
            return None
        backlog = self._run_store.load_events(run_id)
        with self._lock:
            existing = self._run_states.get(run_id)
            if existing is not None:
                return existing
            self._run_states[run_id] = snapshot
            self._event_backlog.setdefault(run_id, backlog)
            self._subscribers.setdefault(run_id, [])
            for agent_state in snapshot.get("agent_runs", {}).values():
                child_run_id = str(agent_state.get("run_id") or "")
                if child_run_id:
                    self._run_states.setdefault(child_run_id, deepcopy(agent_state))
                    self._event_backlog.setdefault(child_run_id, self._run_store.load_events(child_run_id))
                    self._subscribers.setdefault(child_run_id, [])
            return self._run_states[run_id]
