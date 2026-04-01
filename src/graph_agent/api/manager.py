from __future__ import annotations

from copy import deepcopy
import json
import logging
import os
import shutil
from queue import Queue
from threading import Lock, Thread
from typing import Any
from uuid import uuid4

from graph_agent.api.graph_store import GraphStore
from graph_agent.api.run_log_store import RunLogStore
from graph_agent.examples.tool_schema_repair import build_example_services
from graph_agent.providers.discord import DiscordMessageEvent, DiscordTriggerService, normalize_discord_message_payload
from graph_agent.runtime.core import GraphDefinition, resolve_graph_process_env, utc_now_iso
from graph_agent.runtime.documents import AgentDefinition, TestEnvironmentDefinition, load_graph_document
from graph_agent.runtime.engine import GraphRuntime
from graph_agent.runtime.serialization import serialize_run_state
from graph_agent.tools.mcp import McpServerDefinition


LOGGER = logging.getLogger(__name__)
DISCORD_START_PROVIDER_ID = "start.discord_message"


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


class GraphRunManager:
    def __init__(
        self,
        *,
        services: Any | None = None,
        store: GraphStore | None = None,
        run_log_store: RunLogStore | None = None,
        discord_service: DiscordTriggerService | None = None,
    ) -> None:
        self._services = services or build_example_services(include_user_mcp_servers=True)
        self._store = store or GraphStore(self._services)
        self._lock = Lock()
        self._run_states: dict[str, dict[str, Any]] = {}
        self._event_backlog: dict[str, list[dict[str, Any]]] = {}
        self._subscribers: dict[str, list[Queue[str | None]]] = {}
        self._run_log_store = run_log_store or RunLogStore()
        self._discord_service = discord_service or DiscordTriggerService(self.handle_discord_message)

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
        with self._lock:
            if run_id not in self._run_states:
                raise KeyError(run_id)
            return self._run_states[run_id]

    def start_run(self, graph_id: str, input_payload: Any) -> str:
        document = load_graph_document(self._store.get_graph(graph_id))
        document.validate_against_services(self._services)
        run_id = str(uuid4())

        with self._lock:
            self._event_backlog[run_id] = []
            self._subscribers.setdefault(run_id, [])
            if document.is_multi_agent:
                self._run_states[run_id] = self._build_environment_run_state(run_id, document, input_payload)
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
            self._run_log_store.initialize_run(state)

        if document.is_multi_agent:
            thread = Thread(
                target=self._execute_environment_run,
                args=(document, input_payload, run_id),
                daemon=True,
            )
        else:
            graph = document.as_graph()
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

    def start_background_services(self) -> None:
        self._sync_discord_service()
        if self._services.mcp_server_manager is not None:
            self._services.mcp_server_manager.start_auto_boot()

    def stop_background_services(self) -> None:
        self._discord_service.stop()
        if self._services.mcp_server_manager is not None:
            self._services.mcp_server_manager.shutdown_all(preserve_desired_running=True)

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
        state = runtime.run(graph, input_payload, run_id=run_id)
        snapshot = serialize_run_state(state)
        with self._lock:
            self._run_states[run_id] = snapshot
        self._run_log_store.write_state(run_id, snapshot)
        self._close_streams(run_id)

    def _execute_environment_run(
        self,
        document: TestEnvironmentDefinition,
        input_payload: Any,
        parent_run_id: str,
    ) -> None:
        self._record_event(
            parent_run_id,
            {
                "event_type": "run.started",
                "summary": f"Run started for environment '{document.name}'.",
                "payload": {
                    "graph_id": document.graph_id,
                    "graph_name": document.name,
                    "agent_count": len(document.agents),
                },
                "run_id": parent_run_id,
                "timestamp": utc_now_iso(),
                "agent_id": None,
                "parent_run_id": None,
            },
        )

        final_output: dict[str, Any] = {}
        failed_agent_ids: list[str] = []
        try:
            for agent in document.agents:
                child_snapshot = self._run_agent_in_environment(document, agent, input_payload, parent_run_id)
                final_output[agent.agent_id] = child_snapshot.get("final_output")
                if child_snapshot.get("status") == "failed":
                    failed_agent_ids.append(agent.agent_id)
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
            self._run_log_store.write_state(parent_run_id, snapshot)
            self._close_streams(parent_run_id)
            return

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
        self._run_log_store.write_state(parent_run_id, snapshot)
        self._close_streams(parent_run_id)

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
        )
        graph = agent.to_graph(graph_id=document.graph_id, shared_env_vars=document.env_vars)
        state = runtime.run(graph, input_payload, run_id=child_run_id)
        snapshot = {
            **serialize_run_state(state),
            "agent_id": agent.agent_id,
            "agent_name": agent.name,
            "parent_run_id": parent_run_id,
        }

        with self._lock:
            self._run_states[child_run_id] = snapshot
            self._run_states[parent_run_id]["agent_runs"][agent.agent_id] = snapshot

        self._run_log_store.write_state(child_run_id, snapshot)
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
                self._apply_event_to_state(state, event)
                parent_snapshot = deepcopy(state)
                if event["event_type"].startswith("agent."):
                    payload = event.get("payload", {})
                    agent_id = str(event.get("agent_id") or payload.get("agent_id") or "")
                    agent_state = state["agent_runs"].get(agent_id) if agent_id else None
                    if agent_state is not None:
                        child_run_id = str(agent_state["run_id"])
                        child_event = {
                            **event,
                            "event_type": event["event_type"].removeprefix("agent."),
                            "run_id": child_run_id,
                        }
                        child_snapshot = deepcopy(agent_state)
            subscribers = list(self._subscribers.get(run_id, []))

        self._run_log_store.append_event(run_id, event)
        if parent_snapshot is not None:
            self._run_log_store.write_state(run_id, parent_snapshot)
        if child_run_id is not None and child_event is not None and child_snapshot is not None:
            self._run_log_store.append_event(child_run_id, child_event)
            self._run_log_store.write_state(child_run_id, child_snapshot)

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

    def _apply_event_to_state(self, state: dict[str, Any], event: dict[str, Any]) -> None:
        if event["event_type"].startswith("agent."):
            self._apply_agent_event_to_state(state, event)
            return
        self._apply_run_event_to_state(state, event)

    def _apply_agent_event_to_state(self, state: dict[str, Any], event: dict[str, Any]) -> None:
        state["event_history"].append(event)
        payload = event["payload"]
        agent_id = str(event.get("agent_id") or payload.get("agent_id") or "")
        if not agent_id:
            return

        agent_state = state["agent_runs"].get(agent_id)
        if agent_state is None:
            agent_state = self._build_run_state(
                str(payload.get("child_run_id") or uuid4()),
                state["graph_id"],
                state["input_payload"],
                agent_id=agent_id,
                parent_run_id=state["run_id"],
                agent_name=str(payload.get("agent_name") or agent_id),
            )
            state["agent_runs"][agent_id] = agent_state

        normalized_event = {
            **event,
            "event_type": event["event_type"].removeprefix("agent."),
            "run_id": agent_state["run_id"],
        }
        self._apply_run_event_to_state(agent_state, normalized_event)

    def _apply_run_event_to_state(self, state: dict[str, Any], event: dict[str, Any]) -> None:
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
            if state.get("current_node_id") == payload["node_id"]:
                state["current_node_id"] = None
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
            state["current_node_id"] = None
            state["final_output"] = payload["final_output"]
            state["ended_at"] = event["timestamp"]
        elif event_type == "run.failed":
            state["status"] = "failed"
            state["current_node_id"] = None
            state["terminal_error"] = payload["error"]
            if "final_output" in payload:
                state["final_output"] = payload["final_output"]
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
            for agent in document.agents
        }
        return state

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
        return {
            "run_id": run_id,
            "graph_id": graph_id,
            "agent_id": agent_id,
            "agent_name": agent_name,
            "parent_run_id": parent_run_id,
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
            "agent_runs": {},
        }
