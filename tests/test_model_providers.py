from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
import unittest
from pathlib import Path
from typing import Any, Mapping
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from graph_agent.api.manager import GraphRunManager
from graph_agent.examples.tool_schema_repair import build_example_graph_payload, build_example_services
from graph_agent.providers.base import ModelMessage, ModelRequest, ModelToolDefinition
from graph_agent.providers.base import ModelProvider, ModelResponse, ProviderPreflightResult
from graph_agent.providers.claude_code import ClaudeCodeCLIModelProvider
from graph_agent.providers.vendor_api import ClaudeMessagesModelProvider, OpenAIChatModelProvider
from graph_agent.runtime.core import GraphDefinition, GraphValidationError
from graph_agent.runtime.engine import GraphRuntime
from graph_agent.tools.base import ToolContext, ToolDefinition


SEARCH_CATALOG_TOOL = ModelToolDefinition(
    name="search_catalog",
    description="Search the catalog.",
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer"},
        },
        "required": ["query", "limit"],
    },
)


def _debug_enabled() -> bool:
    return os.environ.get("GRAPH_AGENT_TEST_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}


def _extract_claude_code_debug_payload(command: list[str] | None) -> dict[str, Any]:
    if not command:
        return {}

    def option_value(flag: str) -> str | None:
        if flag not in command:
            return None
        index = command.index(flag)
        if index + 1 >= len(command):
            return None
        return command[index + 1]

    schema_text = option_value("--json-schema")
    return {
        "model": option_value("--model"),
        "system_prompt": option_value("--system-prompt"),
        "prompt": option_value("-p"),
        "max_turns": option_value("--max-turns"),
        "json_schema": json.loads(schema_text) if schema_text else None,
    }


def _print_model_debug(
    *,
    title: str,
    request: ModelRequest,
    response: Any,
    sent_payload: Mapping[str, Any] | None = None,
    raw_output: Mapping[str, Any] | None = None,
) -> None:
    if not _debug_enabled():
        return

    debug_payload = {
        "test": title,
        "request_messages": [{"role": message.role, "content": message.content} for message in request.messages],
        "provider_config": dict(request.provider_config or {}),
        "response_mode": request.response_mode,
        "preferred_tool_name": request.preferred_tool_name,
        "available_tools": [tool.name for tool in request.available_tools],
        "sent_payload": dict(sent_payload) if sent_payload is not None else None,
        "raw_output": dict(raw_output) if raw_output is not None else None,
        "normalized_response": {
            "content": response.content,
            "structured_output": response.structured_output,
            "tool_calls": [
                {
                    "tool_name": tool_call.tool_name,
                    "arguments": tool_call.arguments,
                    "provider_tool_id": tool_call.provider_tool_id,
                    "metadata": tool_call.metadata,
                }
                for tool_call in response.tool_calls
            ],
            "metadata": response.metadata,
        },
    }
    print(f"\n=== {title} ===")
    print(json.dumps(debug_payload, indent=2, sort_keys=True))


class _WeatherStubHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        payload = {
            "current_condition": [
                {
                    "temp_C": "22",
                    "temp_F": "72",
                    "FeelsLikeC": "24",
                    "FeelsLikeF": "75",
                    "humidity": "76",
                    "windspeedKmph": "14",
                    "winddir16Point": "SW",
                    "observation_time": "04:08 PM",
                    "weatherDesc": [{"value": "Partly cloudy"}],
                }
            ],
            "nearest_area": [{"areaName": [{"value": "Testville"}]}],
        }
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return


class WeatherStubServer:
    def __enter__(self) -> str:
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _WeatherStubHandler)
        self._thread = Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        host, port = self._server.server_address
        return f"http://{host}:{port}"

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2)


class StubOpenAIProvider(OpenAIChatModelProvider):
    def __init__(self) -> None:
        self.last_payload: Mapping[str, Any] | None = None
        self.last_response_payload: Mapping[str, Any] | None = None

    def _post_json(
        self,
        url: str,
        payload: Mapping[str, Any],
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> Mapping[str, Any]:
        self.last_payload = payload
        self.last_response_payload = {
            "model": payload.get("model"),
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "search_catalog",
                                    "arguments": '{"query":"graph agents","limit":3}',
                                },
                            }
                        ],
                    },
                }
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 7},
        }
        return self.last_response_payload

    def _headers(self, provider_config: Mapping[str, Any]) -> dict[str, str]:
        return {"Authorization": "Bearer test"}


class StubClaudeProvider(ClaudeMessagesModelProvider):
    def __init__(self) -> None:
        self.last_payload: Mapping[str, Any] | None = None
        self.last_response_payload: Mapping[str, Any] | None = None

    def _post_json(
        self,
        url: str,
        payload: Mapping[str, Any],
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> Mapping[str, Any]:
        self.last_payload = payload
        self.last_response_payload = {
            "model": payload.get("model"),
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "search_catalog",
                    "input": {"query": "graph agents", "limit": 3},
                }
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 11, "output_tokens": 6},
        }
        return self.last_response_payload

    def _headers(self, provider_config: Mapping[str, Any]) -> dict[str, str]:
        return {"x-api-key": "test", "anthropic-version": "2023-06-01"}


class StubClaudeCodeProvider(ClaudeCodeCLIModelProvider):
    def __init__(self) -> None:
        self.last_command: list[str] | None = None
        self.last_cwd: str | None = None
        self.last_timeout_seconds: float | None = None
        self.last_payload: Mapping[str, Any] | None = None

    def _run_command(self, command: list[str], cwd: str | None, timeout_seconds: float) -> Mapping[str, Any]:
        self.last_command = command
        self.last_cwd = cwd
        self.last_timeout_seconds = timeout_seconds
        self.last_payload = {
            "result": "",
            "structured_output": {"tool_name": "search_catalog", "arguments": {"query": "graph agents", "limit": 3}},
            "session_id": "session-123",
            "duration_ms": 42,
            "usage": {"input_tokens": 10, "output_tokens": 6},
        }
        return self.last_payload


class ContextEchoProvider(ModelProvider):
    name = "context_echo"

    def generate(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(content="", structured_output=request.metadata.get("mcp_tool_context"))

    def preflight(self, provider_config: Mapping[str, Any] | None = None) -> ProviderPreflightResult:
        return ProviderPreflightResult(status="available", ok=True, message="ok")


class ModelProviderTests(unittest.TestCase):
    def test_openai_provider_normalizes_tool_calls_for_tool_call_nodes(self) -> None:
        provider = StubOpenAIProvider()
        request = ModelRequest(
            prompt_name="schema_proposal",
            messages=[ModelMessage(role="user", content="Build the tool payload.")],
            provider_config={"model": "gpt-4.1-mini"},
            response_mode="tool_call",
            preferred_tool_name="search_catalog",
            available_tools=[SEARCH_CATALOG_TOOL],
        )

        response = provider.generate(request)
        _print_model_debug(
            title="OpenAI tool-call normalization",
            request=request,
            response=response,
            sent_payload=provider.last_payload,
            raw_output=provider.last_response_payload,
        )

        self.assertEqual(response.structured_output, {"query": "graph agents", "limit": 3})
        self.assertEqual(len(response.tool_calls), 1)
        self.assertEqual(response.tool_calls[0].tool_name, "search_catalog")
        self.assertIsNotNone(provider.last_payload)
        assert provider.last_payload is not None
        self.assertIn("tools", provider.last_payload)
        self.assertEqual(provider.last_payload["tools"][0]["function"]["name"], "search_catalog")
        self.assertEqual(
            provider.last_payload["tools"][0]["function"]["parameters"]["required"],
            ["query", "limit"],
        )

    def test_claude_provider_normalizes_tool_calls_for_tool_call_nodes(self) -> None:
        provider = StubClaudeProvider()
        request = ModelRequest(
            prompt_name="schema_proposal",
            messages=[
                ModelMessage(role="system", content="Return structured JSON."),
                ModelMessage(role="user", content="Build the tool payload."),
            ],
            provider_config={"model": "claude-3-5-haiku-latest"},
            response_mode="tool_call",
            preferred_tool_name="search_catalog",
            available_tools=[SEARCH_CATALOG_TOOL],
        )

        response = provider.generate(request)
        _print_model_debug(
            title="Claude API tool-call normalization",
            request=request,
            response=response,
            sent_payload=provider.last_payload,
            raw_output=provider.last_response_payload,
        )

        self.assertEqual(response.structured_output, {"query": "graph agents", "limit": 3})
        self.assertEqual(len(response.tool_calls), 1)
        self.assertEqual(response.tool_calls[0].tool_name, "search_catalog")
        self.assertIsNotNone(provider.last_payload)
        assert provider.last_payload is not None
        self.assertIn("tools", provider.last_payload)
        self.assertEqual(provider.last_payload["tools"][0]["name"], "search_catalog")
        self.assertEqual(
            provider.last_payload["tools"][0]["input_schema"]["required"],
            ["query", "limit"],
        )

    def test_claude_code_provider_normalizes_tool_calls_from_structured_output(self) -> None:
        provider = StubClaudeCodeProvider()
        request = ModelRequest(
            prompt_name="schema_proposal",
            messages=[
                ModelMessage(role="system", content="Return structured JSON."),
                ModelMessage(role="user", content="Build the tool payload."),
            ],
            provider_config={
                "model": "sonnet",
                "cli_path": "claude",
                "working_directory": str(ROOT),
                "timeout_seconds": 15,
                "max_turns": 1,
            },
            response_mode="tool_call",
            preferred_tool_name="search_catalog",
            available_tools=[SEARCH_CATALOG_TOOL],
        )

        response = provider.generate(request)
        _print_model_debug(
            title="Claude Code CLI structured output normalization",
            request=request,
            response=response,
            sent_payload=_extract_claude_code_debug_payload(provider.last_command),
            raw_output=provider.last_payload,
        )

        self.assertEqual(response.structured_output, {"query": "graph agents", "limit": 3})
        self.assertEqual(len(response.tool_calls), 1)
        self.assertEqual(response.tool_calls[0].tool_name, "search_catalog")
        self.assertIsNotNone(provider.last_command)
        assert provider.last_command is not None
        self.assertIn("--json-schema", provider.last_command)
        self.assertIn("--system-prompt", provider.last_command)
        self.assertIn("--tools", provider.last_command)
        self.assertEqual(provider.last_cwd, str(ROOT))
        self.assertEqual(provider.last_timeout_seconds, 15.0)

    def test_claude_code_preflight_reports_live_auth_success(self) -> None:
        provider = StubClaudeCodeProvider()
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}, clear=False):
            with patch("graph_agent.providers.claude_code.subprocess.run") as run_mock:
                run_mock.return_value = Mock(returncode=0, stdout="claude 1.0.0", stderr="")
                result = provider.preflight({"cli_path": "claude", "check_auth": True})

        self.assertTrue(result.ok)
        self.assertEqual(result.status, "available")
        self.assertIsNotNone(provider.last_command)
        assert provider.last_command is not None
        turns_index = provider.last_command.index("--max-turns")
        self.assertEqual(provider.last_command[turns_index + 1], "2")
        self.assertTrue(result.warnings)
        self.assertTrue(result.details["sanitized_child_env"])
        self.assertTrue(result.details["anthropic_api_key_present"])

    def test_claude_code_healthcheck_preserves_higher_max_turns(self) -> None:
        provider = StubClaudeCodeProvider()
        with patch("graph_agent.providers.claude_code.subprocess.run") as run_mock:
            run_mock.return_value = Mock(returncode=0, stdout="claude 1.0.0", stderr="")
            result = provider.preflight({"cli_path": "claude", "check_auth": True, "max_turns": 4})

        self.assertTrue(result.ok)
        self.assertIsNotNone(provider.last_command)
        assert provider.last_command is not None
        turns_index = provider.last_command.index("--max-turns")
        self.assertEqual(provider.last_command[turns_index + 1], "4")

    def test_claude_code_preflight_reports_unauthenticated_state(self) -> None:
        class AuthFailingClaudeCodeProvider(StubClaudeCodeProvider):
            def _run_command(self, command: list[str], cwd: str | None, timeout_seconds: float) -> Mapping[str, Any]:
                raise RuntimeError("not authenticated with Claude Code subscription")

        provider = AuthFailingClaudeCodeProvider()
        with patch("graph_agent.providers.claude_code.subprocess.run") as run_mock:
            run_mock.return_value = Mock(returncode=0, stdout="claude 1.0.0", stderr="")
            result = provider.preflight({"cli_path": "claude", "check_auth": True})

        self.assertFalse(result.ok)
        self.assertEqual(result.status, "unauthenticated")

    def test_claude_code_provider_strips_anthropic_api_key_from_child_process(self) -> None:
        provider = ClaudeCodeCLIModelProvider()
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key", "PATH": "test-path"}, clear=True):
            with patch("graph_agent.providers.claude_code.subprocess.run") as run_mock:
                run_mock.return_value = Mock(
                    returncode=0,
                    stdout='{"result":"ok","session_id":"session-123"}',
                    stderr="",
                )
                provider.generate(
                    ModelRequest(
                        prompt_name="smoke_test",
                        messages=[ModelMessage(role="user", content="Say ok.")],
                        provider_config={"cli_path": "claude", "model": "sonnet"},
                    )
                )

        self.assertTrue(run_mock.called)
        child_env = run_mock.call_args.kwargs["env"]
        self.assertEqual(child_env["PATH"], "test-path")
        self.assertNotIn("ANTHROPIC_API_KEY", child_env)

    def test_manager_reports_claude_code_diagnostics_with_billing_warning(self) -> None:
        manager = GraphRunManager(services=build_example_services())
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}, clear=False):
            with patch("graph_agent.providers.claude_code.subprocess.run") as run_mock:
                run_mock.return_value = Mock(returncode=0, stdout="claude 1.0.0", stderr="")
                diagnostics = manager.provider_diagnostics("claude_code", {"cli_path": "claude"})

        self.assertEqual(diagnostics["active_backend"], "claude_code")
        self.assertTrue(diagnostics["anthropic_api_key_present"])
        self.assertTrue(diagnostics["child_env_sanitized"])
        self.assertIn("API billing", diagnostics["warning"])
        self.assertEqual(diagnostics["authentication_status"], "not_checked")
        self.assertEqual(diagnostics["preflight"]["status"], "installed")

    def test_mock_provider_live_preflight_is_marked_not_applicable(self) -> None:
        manager = GraphRunManager(services=build_example_services())

        preflight = manager.preflight_provider("mock", {}, live=True)
        diagnostics = manager.provider_diagnostics("mock", {}, live=True)

        self.assertEqual(preflight["message"], "Mock provider does not require live verification.")
        self.assertEqual(preflight["details"]["auth_mode"], "not_applicable")
        self.assertEqual(diagnostics["authentication_status"], "not_applicable")
        self.assertFalse(diagnostics["claude_binary_exists"])

    def test_graph_validation_rejects_tool_call_mode_without_tools(self) -> None:
        services = build_example_services()
        payload = build_example_graph_payload()
        for node in payload["nodes"]:
            if node["kind"] == "model":
                node["config"]["response_mode"] = "tool_call"
                node["config"]["allowed_tool_names"] = []
                node["config"].pop("response_schema", None)

        with self.assertRaises(GraphValidationError):
            GraphDefinition.from_dict(payload).validate_against_services(services)

    def test_manager_catalog_includes_provider_statuses_and_metadata(self) -> None:
        manager = GraphRunManager(services=build_example_services())
        with patch("graph_agent.providers.claude_code.subprocess.run") as run_mock:
            run_mock.return_value = Mock(returncode=0, stdout="claude 1.0.0", stderr="")
            catalog = manager.get_catalog()

        self.assertIn("provider_statuses", catalog)
        self.assertIn("claude_code", catalog["provider_statuses"])
        claude_code_provider = next(
            provider for provider in catalog["node_providers"] if provider["provider_id"] == "provider.claude_code"
        )
        self.assertEqual(claude_code_provider["model_provider_name"], "claude_code")
        self.assertTrue(claude_code_provider["config_fields"])

    def test_manager_catalog_includes_mcp_servers_and_tool_metadata(self) -> None:
        manager = GraphRunManager(services=build_example_services())
        catalog = manager.get_catalog()

        self.assertIn("mcp_servers", catalog)
        self.assertTrue(any(server["server_id"] == "weather_mcp" for server in catalog["mcp_servers"]))
        weather_tool = next(tool for tool in catalog["tools"] if tool["name"] == "weather_current")
        self.assertEqual(weather_tool["source_type"], "mcp")
        self.assertFalse(weather_tool["enabled"])
        self.assertFalse(weather_tool["available"])
        self.assertEqual(weather_tool["schema_origin"], "static")

    def test_graph_validation_rejects_offline_mcp_tools(self) -> None:
        services = build_example_services()
        payload = build_example_graph_payload()

        for node in payload["nodes"]:
            if node["kind"] == "model":
                node["config"]["response_mode"] = "tool_call"
                node["config"]["allowed_tool_names"] = ["weather_current"]
                node["config"]["preferred_tool_name"] = "weather_current"
            if node["kind"] == "tool":
                node["tool_name"] = "weather_current"
                node["config"]["tool_name"] = "weather_current"

        with self.assertRaises(GraphValidationError):
            GraphDefinition.from_dict(payload).validate_against_services(services)

    def test_booted_enabled_mcp_tools_can_validate_and_execute(self) -> None:
        services = build_example_services()
        manager = GraphRunManager(services=services)
        with WeatherStubServer() as weather_base:
            try:
                with patch.dict("os.environ", {"GRAPH_AGENT_WEATHER_API_BASE": weather_base}, clear=False):
                    server = manager.boot_mcp_server("weather_mcp")
                    self.assertTrue(server["running"])
                    manager.set_mcp_tool_enabled("weather_current", True)

                    catalog = manager.get_catalog()
                    weather_tool = next(tool for tool in catalog["tools"] if tool["name"] == "weather_current")
                    self.assertTrue(weather_tool["available"])
                    self.assertEqual(weather_tool["schema_origin"], "discovered")

                    payload = build_example_graph_payload()
                    for node in payload["nodes"]:
                        if node["kind"] == "model":
                            node["config"]["response_mode"] = "tool_call"
                            node["config"]["allowed_tool_names"] = ["weather_current"]
                            node["config"]["preferred_tool_name"] = "weather_current"
                        if node["kind"] == "tool":
                            node["tool_name"] = "weather_current"
                            node["config"]["tool_name"] = "weather_current"

                    GraphDefinition.from_dict(payload).validate_against_services(services)

                    result = services.tool_registry.invoke(
                        "weather_current",
                        {"location": "Austin"},
                        ToolContext(run_id="run-1", graph_id="graph-1", node_id="tool-node", state_snapshot={}),
                    )
                    self.assertEqual(result.status, "success")
                    self.assertEqual(result.output["resolved_location"], "Testville")
                    self.assertEqual(result.output["condition"], "Partly cloudy")
                    self.assertEqual(result.output["temperature_c"], "22")
            finally:
                manager.stop_background_services()

    def test_mcp_boot_reports_schema_drift_when_live_tool_metadata_changes(self) -> None:
        services = build_example_services()
        manager = GraphRunManager(services=services)
        services.tool_registry.upsert(
            ToolDefinition(
                name="weather_current",
                description="Outdated preregistered schema.",
                input_schema={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
                source_type="mcp",
                server_id="weather_mcp",
                enabled=False,
                available=False,
                availability_error="MCP server is offline.",
                managed=True,
            )
        )
        try:
            manager.boot_mcp_server("weather_mcp")
            catalog = manager.get_catalog()
            weather_tool = next(tool for tool in catalog["tools"] if tool["name"] == "weather_current")
            self.assertEqual(weather_tool["schema_origin"], "discovered")
            self.assertIn("differs", weather_tool["schema_warning"])
        finally:
            manager.stop_background_services()

    def test_model_nodes_receive_targeted_tool_node_mcp_context_metadata(self) -> None:
        services = build_example_services()
        services.model_providers["context_echo"] = ContextEchoProvider()
        runtime = GraphRuntime(
            services=services,
            max_steps=services.config["max_steps"],
            max_visits_per_node=services.config["max_visits_per_node"],
        )
        graph_payload = {
            "graph_id": "mcp-context-graph",
            "name": "MCP Context Graph",
            "description": "",
            "version": "1.0",
            "start_node_id": "start",
            "nodes": [
                {
                    "id": "start",
                    "kind": "input",
                    "category": "start",
                    "label": "Start",
                    "provider_id": "start.manual_run",
                    "provider_label": "Run Button Start",
                    "config": {"input_binding": {"type": "input_payload"}},
                    "position": {"x": 0, "y": 0},
                },
                {
                    "id": "weather_tool",
                    "kind": "tool",
                    "category": "tool",
                    "label": "Weather Tool",
                    "provider_id": "tool.registry",
                    "provider_label": "Registry Tool Node",
                    "tool_name": "weather_current",
                    "config": {"tool_name": "weather_current", "include_mcp_tool_context": True},
                    "position": {"x": 100, "y": 0},
                },
                {
                    "id": "model",
                    "kind": "model",
                    "category": "api",
                    "label": "Model",
                    "provider_id": "core.api",
                    "provider_label": "API Call Node",
                    "model_provider_name": "context_echo",
                    "prompt_name": "context_prompt",
                    "config": {
                        "provider_name": "context_echo",
                        "prompt_name": "context_prompt",
                        "system_prompt": "Use the MCP context.",
                        "user_message_template": "{mcp_tool_context}",
                        "response_mode": "message",
                        "tool_target_node_ids": ["weather_tool"],
                    },
                    "position": {"x": 200, "y": 0},
                },
                {
                    "id": "finish",
                    "kind": "output",
                    "category": "end",
                    "label": "Finish",
                    "provider_id": "core.output",
                    "provider_label": "Core Output Node",
                    "config": {"source_binding": {"type": "latest_envelope", "source": "model"}},
                    "position": {"x": 300, "y": 0},
                },
            ],
            "edges": [
                {"id": "e1", "source_id": "start", "target_id": "model", "label": "", "kind": "standard", "priority": 100},
                {"id": "e2", "source_id": "model", "target_id": "finish", "label": "", "kind": "standard", "priority": 100},
            ],
        }
        graph = GraphDefinition.from_dict(graph_payload)
        graph.validate_against_services(services)

        state = runtime.run(graph, {"request": "weather please"}, run_id="run-mcp-context")
        self.assertEqual(state.status, "completed")
        payload = state.final_output
        assert isinstance(payload, dict)
        self.assertEqual(payload["tool_names"], ["weather_current"])
        self.assertEqual(payload["tool_nodes"][0]["tool_name"], "weather_current")
        self.assertEqual(payload["tool_nodes"][0]["tool_node_id"], "weather_tool")
        self.assertEqual(payload["tool_nodes"][0]["server"]["server_id"], "weather_mcp")

    def test_graphs_can_swap_between_registered_model_providers(self) -> None:
        services = build_example_services()
        payload = build_example_graph_payload()

        for node in payload["nodes"]:
            if node["kind"] == "model":
                node["model_provider_name"] = "openai"
                node["config"]["provider_name"] = "openai"
                node["config"]["model"] = "gpt-4.1-mini"

        GraphDefinition.from_dict(payload).validate_against_services(services)

        for node in payload["nodes"]:
            if node["kind"] == "model":
                node["model_provider_name"] = "claude"
                node["config"]["provider_name"] = "claude"
                node["config"]["model"] = "claude-3-5-haiku-latest"
                node["config"]["max_tokens"] = 1024

        GraphDefinition.from_dict(payload).validate_against_services(services)

        for node in payload["nodes"]:
            if node["kind"] == "model":
                node["model_provider_name"] = "claude_code"
                node["config"]["provider_name"] = "claude_code"
                node["config"]["model"] = "sonnet"

        GraphDefinition.from_dict(payload).validate_against_services(services)


if __name__ == "__main__":
    unittest.main()
