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
from graph_agent.examples.tool_schema_repair import (
    build_auto_branching_graph_payload,
    build_example_graph_payload,
    build_example_services,
)
from graph_agent.providers.base import ModelMessage, ModelRequest, ModelToolCall, ModelToolDefinition
from graph_agent.providers.base import ModelProvider, ModelResponse, ProviderPreflightResult
from graph_agent.providers.claude_code import ClaudeCodeCLIModelProvider
from graph_agent.providers.vendor_api import ClaudeMessagesModelProvider, OpenAIChatModelProvider
from graph_agent.runtime.core import GraphDefinition, GraphValidationError
from graph_agent.runtime.documents import load_graph_document
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
        system_prompt = request.messages[0].content if request.messages else ""
        return ModelResponse(
            content="",
            structured_output={
                "system_prompt": system_prompt,
                "mcp_tool_context": request.metadata.get("mcp_tool_context"),
                "available_tool_names": [tool.name for tool in request.available_tools],
                "mcp_available_tool_names": request.metadata.get("mcp_available_tool_names", []),
            },
        )

    def preflight(self, provider_config: Mapping[str, Any] | None = None) -> ProviderPreflightResult:
        return ProviderPreflightResult(status="available", ok=True, message="ok")


class McpDispatchProvider(ModelProvider):
    name = "mcp_dispatch"

    def generate(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(
            content="",
            structured_output={"location": "Austin"},
            tool_calls=[
                ModelToolCall(
                    tool_name="weather_current",
                    arguments={"location": "Austin"},
                    provider_tool_id="provider-tool-1",
                    metadata={},
                )
            ],
        )

    def preflight(self, provider_config: Mapping[str, Any] | None = None) -> ProviderPreflightResult:
        return ProviderPreflightResult(status="available", ok=True, message="ok")


class AutoMessageProvider(ModelProvider):
    name = "auto_message"

    def generate(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(
            content="A direct reply is enough.",
            structured_output={"message": "A direct reply is enough."},
            metadata={"variant": "message"},
        )

    def preflight(self, provider_config: Mapping[str, Any] | None = None) -> ProviderPreflightResult:
        return ProviderPreflightResult(status="available", ok=True, message="ok")


class AutoToolCallProvider(ModelProvider):
    name = "auto_tool_call"

    def generate(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(
            content="",
            structured_output={"location": "Austin"},
            tool_calls=[
                ModelToolCall(
                    tool_name="weather_current",
                    arguments={"location": "Austin"},
                    provider_tool_id="auto-tool-1",
                    metadata={"variant": "tool_call"},
                )
            ],
            metadata={"variant": "tool_call"},
        )

    def preflight(self, provider_config: Mapping[str, Any] | None = None) -> ProviderPreflightResult:
        return ProviderPreflightResult(status="available", ok=True, message="ok")


class AutoMixedProvider(ModelProvider):
    name = "auto_mixed"

    def generate(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(
            content="Checking the live weather tool now.",
            structured_output={"location": "Austin"},
            tool_calls=[
                ModelToolCall(
                    tool_name="weather_current",
                    arguments={"location": "Austin"},
                    provider_tool_id="auto-mixed-1",
                    metadata={"variant": "mixed"},
                )
            ],
            metadata={"variant": "mixed"},
        )

    def preflight(self, provider_config: Mapping[str, Any] | None = None) -> ProviderPreflightResult:
        return ProviderPreflightResult(status="available", ok=True, message="ok")


class ModelProviderTests(unittest.TestCase):
    def _build_auto_branch_graph(self, provider_name: str, provider: ModelProvider | None = None) -> tuple[Any, GraphRuntime, dict[str, Any]]:
        services = build_example_services()
        if provider is not None:
            services.model_providers[provider_name] = provider
        runtime = GraphRuntime(
            services=services,
            max_steps=services.config["max_steps"],
            max_visits_per_node=services.config["max_visits_per_node"],
        )
        graph_payload = build_auto_branching_graph_payload()
        for node in graph_payload["nodes"]:
            if node["id"] == "model":
                node["model_provider_name"] = provider_name
                node["config"]["provider_name"] = provider_name
        return services, runtime, graph_payload

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
            child_env = provider._child_env()
            payload = provider._run_command(
                [
                    sys.executable,
                    "-c",
                    (
                        "import json, os; "
                        "print(json.dumps({'result': os.environ.get('ANTHROPIC_API_KEY', 'missing'), 'session_id': 'session-123'}))"
                    ),
                ],
                cwd=str(ROOT),
                timeout_seconds=5,
            )

        self.assertEqual(payload["result"], "missing")
        self.assertNotIn("ANTHROPIC_API_KEY", child_env)
        self.assertEqual(child_env["PATH"], "test-path")

    def test_claude_code_provider_resets_timeout_when_output_progress_continues(self) -> None:
        provider = ClaudeCodeCLIModelProvider()
        payload = provider._run_command(
            [
                sys.executable,
                "-c",
                (
                    "import json, sys, time; "
                    "sys.stderr.write('tick1\\n'); sys.stderr.flush(); "
                    "time.sleep(0.1); "
                    "sys.stderr.write('tick2\\n'); sys.stderr.flush(); "
                    "time.sleep(0.1); "
                    "sys.stdout.write(json.dumps({'result': 'ok', 'session_id': 'session-123'})); "
                    "sys.stdout.flush()"
                ),
            ],
            cwd=str(ROOT),
            timeout_seconds=0.15,
        )

        self.assertEqual(payload["result"], "ok")

    def test_claude_code_provider_times_out_after_output_stalls(self) -> None:
        provider = ClaudeCodeCLIModelProvider()
        with self.assertRaises(RuntimeError) as context:
            provider._run_command(
                [
                    sys.executable,
                    "-c",
                    (
                        "import json, time; "
                        "time.sleep(0.3); "
                        "print(json.dumps({'result': 'too late'}))"
                    ),
                ],
                cwd=str(ROOT),
                timeout_seconds=0.1,
            )

        self.assertIn("without output progress", str(context.exception))

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

    def test_model_nodes_receive_mcp_context_provider_in_system_prompt(self) -> None:
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
                    "id": "weather_context",
                    "kind": "mcp_context_provider",
                    "category": "tool",
                    "label": "Weather Context",
                    "provider_id": "tool.mcp_context_provider",
                    "provider_label": "MCP Context Provider",
                    "config": {"tool_names": ["weather_current"], "include_mcp_tool_context": True},
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
                {
                    "id": "ctx-binding",
                    "source_id": "weather_context",
                    "target_id": "model",
                    "source_handle_id": "tool-context",
                    "target_handle_id": "api-tool-context",
                    "label": "tool context",
                    "kind": "binding",
                    "priority": 0,
                },
                {"id": "e2", "source_id": "model", "target_id": "finish", "label": "", "kind": "standard", "priority": 100},
            ],
        }
        graph = GraphDefinition.from_dict(graph_payload)
        manager = GraphRunManager(services=services)
        with WeatherStubServer() as weather_base:
            try:
                with patch.dict("os.environ", {"GRAPH_AGENT_WEATHER_API_BASE": weather_base}, clear=False):
                    manager.boot_mcp_server("weather_mcp")
                    manager.set_mcp_tool_enabled("weather_current", True)
                    graph.validate_against_services(services)

                    state = runtime.run(graph, {"request": "weather please"}, run_id="run-mcp-context")
                    self.assertEqual(state.status, "completed")
                    payload = state.final_output
                    assert isinstance(payload, dict)
                    self.assertIn("MCP Tool Guidance", payload["system_prompt"])
                    self.assertIn("MCP Tool Context", payload["system_prompt"])
                    self.assertIn("Tool: weather_current", payload["system_prompt"])
                    tool_context = payload["mcp_tool_context"]
                    assert isinstance(tool_context, dict)
                    self.assertEqual(tool_context["tool_names"], ["weather_current"])
                    self.assertEqual(tool_context["tool_nodes"][0]["tool_name"], "weather_current")
                    self.assertEqual(tool_context["tool_nodes"][0]["tool_node_id"], "weather_context")
                    self.assertEqual(tool_context["tool_nodes"][0]["server"]["server_id"], "weather_mcp")
                    self.assertEqual(payload["available_tool_names"], ["weather_current"])
            finally:
                manager.stop_background_services()

    def test_mcp_context_provider_can_inject_prompt_without_exposing_tools(self) -> None:
        services = build_example_services()
        services.model_providers["context_echo"] = ContextEchoProvider()
        runtime = GraphRuntime(
            services=services,
            max_steps=services.config["max_steps"],
            max_visits_per_node=services.config["max_visits_per_node"],
        )
        graph_payload = {
            "graph_id": "mcp-prompt-only-graph",
            "name": "MCP Prompt Only Graph",
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
                    "id": "weather_context",
                    "kind": "mcp_context_provider",
                    "category": "tool",
                    "label": "Weather Context",
                    "provider_id": "tool.mcp_context_provider",
                    "provider_label": "MCP Context Provider",
                    "config": {
                        "tool_names": ["weather_current"],
                        "expose_mcp_tools": False,
                        "include_mcp_tool_context": True,
                    },
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
                        "user_message_template": "{input_payload}",
                        "response_mode": "message",
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
                {
                    "id": "ctx-binding",
                    "source_id": "weather_context",
                    "target_id": "model",
                    "source_handle_id": "tool-context",
                    "target_handle_id": "api-tool-context",
                    "label": "tool context",
                    "kind": "binding",
                    "priority": 0,
                },
                {"id": "e2", "source_id": "model", "target_id": "finish", "label": "", "kind": "standard", "priority": 100},
            ],
        }
        graph = GraphDefinition.from_dict(graph_payload)
        manager = GraphRunManager(services=services)
        with WeatherStubServer() as weather_base:
            try:
                with patch.dict("os.environ", {"GRAPH_AGENT_WEATHER_API_BASE": weather_base}, clear=False):
                    manager.boot_mcp_server("weather_mcp")
                    manager.set_mcp_tool_enabled("weather_current", True)
                    graph.validate_against_services(services)

                    state = runtime.run(graph, {"request": "weather please"}, run_id="run-mcp-prompt-only")
                    self.assertEqual(state.status, "completed")
                    payload = state.final_output
                    assert isinstance(payload, dict)
                    self.assertNotIn("MCP Tool Guidance", payload["system_prompt"])
                    self.assertIn("MCP Tool Context", payload["system_prompt"])
                    self.assertEqual(payload["available_tool_names"], [])
                    self.assertEqual(payload["mcp_available_tool_names"], [])
                    assert isinstance(payload["mcp_tool_context"], dict)
                    self.assertEqual(payload["mcp_tool_context"]["tool_names"], ["weather_current"])
            finally:
                manager.stop_background_services()

    def test_disabled_mcp_tools_are_not_exposed_at_runtime(self) -> None:
        services = build_example_services()
        services.model_providers["context_echo"] = ContextEchoProvider()
        runtime = GraphRuntime(
            services=services,
            max_steps=services.config["max_steps"],
            max_visits_per_node=services.config["max_visits_per_node"],
        )
        graph_payload = {
            "graph_id": "mcp-disabled-runtime-graph",
            "name": "MCP Disabled Runtime Graph",
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
                    "id": "weather_context",
                    "kind": "mcp_context_provider",
                    "category": "tool",
                    "label": "Weather Context",
                    "provider_id": "tool.mcp_context_provider",
                    "provider_label": "MCP Context Provider",
                    "config": {"tool_names": ["weather_current"], "include_mcp_tool_context": True},
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
                        "system_prompt": "Use MCP if available.",
                        "user_message_template": "{input_payload}",
                        "response_mode": "message",
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
                {
                    "id": "ctx-binding",
                    "source_id": "weather_context",
                    "target_id": "model",
                    "source_handle_id": "tool-context",
                    "target_handle_id": "api-tool-context",
                    "label": "tool context",
                    "kind": "binding",
                    "priority": 0,
                },
                {"id": "e2", "source_id": "model", "target_id": "finish", "label": "", "kind": "standard", "priority": 100},
            ],
        }
        graph = GraphDefinition.from_dict(graph_payload)
        manager = GraphRunManager(services=services)
        with WeatherStubServer() as weather_base:
            try:
                with patch.dict("os.environ", {"GRAPH_AGENT_WEATHER_API_BASE": weather_base}, clear=False):
                    manager.boot_mcp_server("weather_mcp")
                    state = runtime.run(graph, {"request": "weather please"}, run_id="run-mcp-disabled")
                    self.assertEqual(state.status, "completed")
                    payload = state.final_output
                    assert isinstance(payload, dict)
                    self.assertEqual(payload["available_tool_names"], [])
                    self.assertEqual(payload["mcp_available_tool_names"], [])
                    self.assertNotIn("MCP Tool Guidance", payload["system_prompt"])
                    self.assertIsNone(payload["mcp_tool_context"])
            finally:
                manager.stop_background_services()

    def test_mcp_executor_dispatches_selected_mcp_tool(self) -> None:
        services = build_example_services()
        services.model_providers["mcp_dispatch"] = McpDispatchProvider()
        runtime = GraphRuntime(
            services=services,
            max_steps=services.config["max_steps"],
            max_visits_per_node=services.config["max_visits_per_node"],
        )
        graph_payload = {
            "graph_id": "mcp-executor-graph",
            "name": "MCP Executor Graph",
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
                    "id": "weather_context",
                    "kind": "mcp_context_provider",
                    "category": "tool",
                    "label": "Weather Context",
                    "provider_id": "tool.mcp_context_provider",
                    "provider_label": "MCP Context Provider",
                    "config": {"tool_names": ["weather_current"], "include_mcp_tool_context": False},
                    "position": {"x": 120, "y": 0},
                },
                {
                    "id": "model",
                    "kind": "model",
                    "category": "api",
                    "label": "Model",
                    "provider_id": "core.api",
                    "provider_label": "API Call Node",
                    "model_provider_name": "mcp_dispatch",
                    "prompt_name": "dispatch_prompt",
                    "config": {
                        "provider_name": "mcp_dispatch",
                        "prompt_name": "dispatch_prompt",
                        "system_prompt": "Call the MCP weather tool.",
                        "user_message_template": "{input_payload}",
                        "response_mode": "tool_call",
                    },
                    "position": {"x": 240, "y": 0},
                },
                {
                    "id": "executor",
                    "kind": "mcp_tool_executor",
                    "category": "tool",
                    "label": "Executor",
                    "provider_id": "tool.mcp_tool_executor",
                    "provider_label": "MCP Tool Executor",
                    "config": {},
                    "position": {"x": 420, "y": 0},
                },
                {
                    "id": "finish",
                    "kind": "output",
                    "category": "end",
                    "label": "Finish",
                    "provider_id": "core.output",
                    "provider_label": "Core Output Node",
                    "config": {"source_binding": {"type": "latest_payload", "source": "executor"}},
                    "position": {"x": 600, "y": 0},
                },
            ],
            "edges": [
                {"id": "start-model", "source_id": "start", "target_id": "model", "label": "", "kind": "standard", "priority": 100},
                {
                    "id": "ctx-binding",
                    "source_id": "weather_context",
                    "target_id": "model",
                    "source_handle_id": "tool-context",
                    "target_handle_id": "api-tool-context",
                    "label": "tool context",
                    "kind": "binding",
                    "priority": 0,
                },
                {"id": "model-executor", "source_id": "model", "target_id": "executor", "label": "", "kind": "standard", "priority": 100},
                {"id": "executor-finish", "source_id": "executor", "target_id": "finish", "label": "", "kind": "standard", "priority": 100},
            ],
        }
        graph = GraphDefinition.from_dict(graph_payload)
        manager = GraphRunManager(services=services)
        with WeatherStubServer() as weather_base:
            try:
                with patch.dict("os.environ", {"GRAPH_AGENT_WEATHER_API_BASE": weather_base}, clear=False):
                    manager.boot_mcp_server("weather_mcp")
                    manager.set_mcp_tool_enabled("weather_current", True)
                    graph.validate_against_services(services)
                    state = runtime.run(graph, {"request": "weather please"}, run_id="run-mcp-executor")
                    self.assertEqual(state.status, "completed")
                    payload = state.final_output
                    assert isinstance(payload, dict)
                    self.assertEqual(payload["resolved_location"], "Testville")
                    self.assertEqual(payload["condition"], "Partly cloudy")
            finally:
                manager.stop_background_services()

    def test_auto_mode_emits_message_envelope_for_message_only_responses(self) -> None:
        services, runtime, graph_payload = self._build_auto_branch_graph("auto_message", AutoMessageProvider())
        graph = GraphDefinition.from_dict(graph_payload)
        manager = GraphRunManager(services=services)
        with WeatherStubServer() as weather_base:
            try:
                with patch.dict("os.environ", {"GRAPH_AGENT_WEATHER_API_BASE": weather_base}, clear=False):
                    manager.boot_mcp_server("weather_mcp")
                    manager.set_mcp_tool_enabled("weather_current", True)
                    graph.validate_against_services(services)

                    state = runtime.run(graph, {"request": "say hi"}, run_id="run-auto-message")
                    self.assertEqual(state.status, "completed")
                    self.assertEqual(state.final_output, {"message": "A direct reply is enough."})
                    model_output = state.node_outputs["model"]
                    assert isinstance(model_output, dict)
                    self.assertEqual(model_output["metadata"]["contract"], "message_envelope")
                    self.assertEqual(model_output["payload"], {"message": "A direct reply is enough."})
                    self.assertEqual(model_output["tool_calls"], [])
            finally:
                manager.stop_background_services()

    def test_auto_mode_routes_tool_call_responses_to_mcp_executor(self) -> None:
        services, runtime, graph_payload = self._build_auto_branch_graph("auto_tool_call", AutoToolCallProvider())
        graph = GraphDefinition.from_dict(graph_payload)
        manager = GraphRunManager(services=services)
        with WeatherStubServer() as weather_base:
            try:
                with patch.dict("os.environ", {"GRAPH_AGENT_WEATHER_API_BASE": weather_base}, clear=False):
                    manager.boot_mcp_server("weather_mcp")
                    manager.set_mcp_tool_enabled("weather_current", True)
                    graph.validate_against_services(services)

                    state = runtime.run(graph, {"request": "weather please"}, run_id="run-auto-tool-call")
                    self.assertEqual(state.status, "completed")
                    payload = state.final_output
                    assert isinstance(payload, dict)
                    self.assertEqual(payload["resolved_location"], "Testville")
                    self.assertEqual(payload["condition"], "Partly cloudy")
                    model_output = state.node_outputs["model"]
                    assert isinstance(model_output, dict)
                    self.assertEqual(model_output["metadata"]["contract"], "tool_call_envelope")
                    self.assertEqual(model_output["payload"], None)
                    self.assertEqual(model_output["tool_calls"][0]["tool_name"], "weather_current")
                    tool_call_edge = next(edge for edge in graph_payload["edges"] if edge["id"] == "edge-model-executor")
                    self.assertEqual(tool_call_edge["source_handle_id"], "api-tool-call")
            finally:
                manager.stop_background_services()

    def test_auto_mode_routes_both_tool_call_and_message_handles_for_mixed_responses(self) -> None:
        services, runtime, graph_payload = self._build_auto_branch_graph("auto_mixed", AutoMixedProvider())
        for node in graph_payload["nodes"]:
            if node["id"] == "model":
                node["config"]["response_mode"] = "tool_call"
        graph = GraphDefinition.from_dict(graph_payload)
        manager = GraphRunManager(services=services)
        with WeatherStubServer() as weather_base:
            try:
                with patch.dict("os.environ", {"GRAPH_AGENT_WEATHER_API_BASE": weather_base}, clear=False):
                    manager.boot_mcp_server("weather_mcp")
                    manager.set_mcp_tool_enabled("weather_current", True)
                    graph.validate_against_services(services)

                    state = runtime.run(graph, {"request": "weather please"}, run_id="run-auto-mixed")
                    self.assertEqual(state.status, "completed")
                    payload = state.final_output
                    self.assertEqual(payload, "Checking the live weather tool now.")
                    self.assertEqual(state.visit_counts["finish"], 2)
                    model_output = state.node_outputs["model"]
                    assert isinstance(model_output, dict)
                    self.assertEqual(model_output["metadata"]["contract"], "tool_call_envelope")
                    self.assertEqual(model_output["metadata"]["response_mode"], "auto")
                    self.assertEqual(model_output["artifacts"], {})
                    self.assertEqual(model_output["tool_calls"][0]["tool_name"], "weather_current")
                    tool_route_output = state.edge_outputs["edge-model-executor"]
                    assert isinstance(tool_route_output, dict)
                    self.assertEqual(tool_route_output["metadata"]["contract"], "tool_call_envelope")
                    self.assertEqual(tool_route_output["artifacts"], {})
                    self.assertEqual(tool_route_output["tool_calls"][0]["tool_name"], "weather_current")
                    message_route_output = state.edge_outputs["edge-model-finish"]
                    assert isinstance(message_route_output, dict)
                    self.assertEqual(message_route_output["metadata"]["contract"], "message_envelope")
                    self.assertEqual(message_route_output["payload"], "Checking the live weather tool now.")
                    transition_edges = [transition.edge_id for transition in state.transition_history]
                    self.assertIn("edge-model-executor", transition_edges)
                    self.assertIn("edge-model-finish", transition_edges)
                    finish_outputs = [
                        event.payload["output"]
                        for event in state.event_history
                        if event.event_type == "node.completed" and event.payload.get("node_id") == "finish"
                    ]
                    self.assertIn("Checking the live weather tool now.", finish_outputs)
                    self.assertTrue(any(isinstance(item, dict) and item.get("resolved_location") == "Testville" for item in finish_outputs))
            finally:
                manager.stop_background_services()

    def test_mcp_executor_validation_rejects_auto_mode_sources_without_tool_call_condition(self) -> None:
        services, _runtime, graph_payload = self._build_auto_branch_graph("mock")
        for edge in graph_payload["edges"]:
            if edge["id"] == "edge-model-executor":
                edge["kind"] = "standard"
                edge["condition"] = None

        with self.assertRaises(GraphValidationError):
            GraphDefinition.from_dict(graph_payload).validate_against_services(services)

    def test_mcp_executor_validation_rejects_api_message_output_routes(self) -> None:
        services, _runtime, graph_payload = self._build_auto_branch_graph("mock")
        for edge in graph_payload["edges"]:
            if edge["id"] == "edge-model-finish":
                edge["target_id"] = "executor"

        with self.assertRaises(GraphValidationError):
            GraphDefinition.from_dict(graph_payload).validate_against_services(services)

    def test_legacy_auto_branch_graphs_without_api_output_handles_still_run(self) -> None:
        services, runtime, graph_payload = self._build_auto_branch_graph("auto_tool_call", AutoToolCallProvider())
        for edge in graph_payload["edges"]:
            if edge["source_id"] == "model":
                edge["source_handle_id"] = None
        graph = GraphDefinition.from_dict(graph_payload)
        manager = GraphRunManager(services=services)
        with WeatherStubServer() as weather_base:
            try:
                with patch.dict("os.environ", {"GRAPH_AGENT_WEATHER_API_BASE": weather_base}, clear=False):
                    manager.boot_mcp_server("weather_mcp")
                    manager.set_mcp_tool_enabled("weather_current", True)
                    graph.validate_against_services(services)

                    state = runtime.run(graph, {"request": "weather please"}, run_id="run-legacy-auto-tool-call")
                    self.assertEqual(state.status, "completed")
                    payload = state.final_output
                    assert isinstance(payload, dict)
                    self.assertEqual(payload["resolved_location"], "Testville")
                    self.assertEqual(payload["condition"], "Partly cloudy")
            finally:
                manager.stop_background_services()

    def test_display_node_preserves_tool_call_contract_for_downstream_executor(self) -> None:
        services, runtime, graph_payload = self._build_auto_branch_graph("auto_tool_call", AutoToolCallProvider())
        graph_payload["nodes"].insert(
            4,
            {
                "id": "display",
                "kind": "data",
                "category": "data",
                "label": "Display Envelope",
                "provider_id": "core.data_display",
                "provider_label": "Envelope Display Node",
                "config": {
                    "mode": "passthrough",
                    "show_input_envelope": True,
                    "lock_passthrough": True,
                },
                "position": {"x": 900, "y": 120},
            },
        )
        for edge in graph_payload["edges"]:
            if edge["id"] == "edge-model-executor":
                edge["target_id"] = "display"
            if edge["id"] == "edge-model-finish":
                edge["priority"] = 30
        graph_payload["edges"].append(
            {
                "id": "edge-display-executor",
                "source_id": "display",
                "target_id": "executor",
                "label": "forward tool call",
                "kind": "standard",
                "priority": 100,
                "condition": None,
            }
        )

        graph = GraphDefinition.from_dict(graph_payload)
        manager = GraphRunManager(services=services)
        with WeatherStubServer() as weather_base:
            try:
                with patch.dict("os.environ", {"GRAPH_AGENT_WEATHER_API_BASE": weather_base}, clear=False):
                    manager.boot_mcp_server("weather_mcp")
                    manager.set_mcp_tool_enabled("weather_current", True)
                    graph.validate_against_services(services)

                    state = runtime.run(graph, {"request": "weather please"}, run_id="run-display-tool-call")
                    self.assertEqual(state.status, "completed")
                    payload = state.final_output
                    assert isinstance(payload, dict)
                    self.assertEqual(payload["resolved_location"], "Testville")
                    self.assertEqual(payload["condition"], "Partly cloudy")
                    display_output = state.node_outputs["display"]
                    assert isinstance(display_output, dict)
                    self.assertEqual(display_output["metadata"]["contract"], "tool_call_envelope")
                    self.assertEqual(display_output["tool_calls"][0]["tool_name"], "weather_current")
                    self.assertEqual(display_output["artifacts"]["display_envelope"]["metadata"]["contract"], "tool_call_envelope")
            finally:
                manager.stop_background_services()

    def test_model_validation_rejects_non_context_tool_targets(self) -> None:
        services = build_example_services()
        graph_payload = {
            "graph_id": "invalid-context-target",
            "name": "Invalid MCP Context Target",
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
                    "id": "executor",
                    "kind": "mcp_tool_executor",
                    "category": "tool",
                    "label": "Executor",
                    "provider_id": "tool.mcp_tool_executor",
                    "provider_label": "MCP Tool Executor",
                    "config": {},
                    "position": {"x": 120, "y": 0},
                },
                {
                    "id": "model",
                    "kind": "model",
                    "category": "api",
                    "label": "Model",
                    "provider_id": "core.api",
                    "provider_label": "API Call Node",
                    "model_provider_name": "mock",
                    "prompt_name": "invalid_context",
                    "config": {
                        "provider_name": "mock",
                        "prompt_name": "invalid_context",
                        "system_prompt": "Invalid target",
                        "user_message_template": "{input_payload}",
                        "response_mode": "tool_call",
                        "tool_target_node_ids": ["executor"],
                    },
                    "position": {"x": 240, "y": 0},
                },
                {
                    "id": "finish",
                    "kind": "output",
                    "category": "end",
                    "label": "Finish",
                    "provider_id": "core.output",
                    "provider_label": "Core Output Node",
                    "config": {"source_binding": {"type": "latest_envelope", "source": "model"}},
                    "position": {"x": 360, "y": 0},
                },
            ],
            "edges": [
                {"id": "e1", "source_id": "start", "target_id": "model", "label": "", "kind": "standard", "priority": 100},
                {"id": "e2", "source_id": "model", "target_id": "finish", "label": "", "kind": "standard", "priority": 100},
            ],
        }
        with self.assertRaises(GraphValidationError):
            GraphDefinition.from_dict(graph_payload).validate_against_services(services)

    def test_mcp_executor_validation_rejects_non_tool_call_model_sources(self) -> None:
        services = build_example_services()
        graph_payload = {
            "graph_id": "invalid-mcp-executor-source",
            "name": "Invalid MCP Executor Source",
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
                    "id": "model",
                    "kind": "model",
                    "category": "api",
                    "label": "Model",
                    "provider_id": "core.api",
                    "provider_label": "API Call Node",
                    "model_provider_name": "mock",
                    "prompt_name": "invalid_executor",
                    "config": {
                        "provider_name": "mock",
                        "prompt_name": "invalid_executor",
                        "system_prompt": "Produce a message.",
                        "user_message_template": "{input_payload}",
                        "response_mode": "message",
                    },
                    "position": {"x": 120, "y": 0},
                },
                {
                    "id": "executor",
                    "kind": "mcp_tool_executor",
                    "category": "tool",
                    "label": "Executor",
                    "provider_id": "tool.mcp_tool_executor",
                    "provider_label": "MCP Tool Executor",
                    "config": {},
                    "position": {"x": 240, "y": 0},
                },
            ],
            "edges": [
                {"id": "start-model", "source_id": "start", "target_id": "model", "label": "", "kind": "standard", "priority": 100},
                {"id": "model-executor", "source_id": "model", "target_id": "executor", "label": "", "kind": "standard", "priority": 100},
            ],
        }

        with self.assertRaises(GraphValidationError):
            GraphDefinition.from_dict(graph_payload).validate_against_services(services)

    def test_load_graph_document_drops_legacy_tool_targets_for_non_context_tools(self) -> None:
        payload = {
            "graph_id": "legacy-tool-targets",
            "name": "Legacy Tool Targets",
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
                    "id": "model",
                    "kind": "model",
                    "category": "api",
                    "label": "Model",
                    "provider_id": "core.api",
                    "provider_label": "API Call Node",
                    "model_provider_name": "mock",
                    "prompt_name": "legacy_tool_targets",
                    "config": {
                        "provider_name": "mock",
                        "prompt_name": "legacy_tool_targets",
                        "system_prompt": "legacy",
                        "user_message_template": "{input_payload}",
                        "response_mode": "tool_call",
                        "allowed_tool_names": ["search_catalog"],
                        "tool_target_node_ids": ["legacy_tool"],
                    },
                    "position": {"x": 120, "y": 0},
                },
                {
                    "id": "legacy_tool",
                    "kind": "tool",
                    "category": "tool",
                    "label": "Legacy Tool",
                    "provider_id": "tool.registry",
                    "provider_label": "Registry Tool Node",
                    "tool_name": "search_catalog",
                    "config": {"tool_name": "search_catalog"},
                    "position": {"x": 240, "y": 0},
                },
                {
                    "id": "finish",
                    "kind": "output",
                    "category": "end",
                    "label": "Finish",
                    "provider_id": "core.output",
                    "provider_label": "Core Output Node",
                    "config": {"source_binding": {"type": "latest_envelope", "source": "model"}},
                    "position": {"x": 360, "y": 0},
                },
            ],
            "edges": [
                {"id": "e1", "source_id": "start", "target_id": "model", "label": "", "kind": "standard", "priority": 100},
                {"id": "e2", "source_id": "model", "target_id": "finish", "label": "", "kind": "standard", "priority": 100},
            ],
        }

        document = load_graph_document(payload)
        graph = document.as_graph()
        model_node = next(node for node in graph.to_dict()["nodes"] if node["id"] == "model")
        self.assertNotIn("tool_target_node_ids", model_node["config"])

    def test_load_graph_document_defaults_mcp_context_providers_to_callable_tool_exposure(self) -> None:
        payload = {
            "graph_id": "mcp-default-exposure",
            "name": "MCP Default Exposure",
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
                    "id": "weather_context",
                    "kind": "mcp_context_provider",
                    "category": "tool",
                    "label": "Weather Context",
                    "provider_id": "tool.mcp_context_provider",
                    "provider_label": "MCP Context Provider",
                    "config": {"tool_names": ["weather_current"], "include_mcp_tool_context": False},
                    "position": {"x": 120, "y": 0},
                },
                {
                    "id": "finish",
                    "kind": "output",
                    "category": "end",
                    "label": "Finish",
                    "provider_id": "core.output",
                    "provider_label": "Core Output Node",
                    "config": {"source_binding": {"type": "input_payload"}},
                    "position": {"x": 240, "y": 0},
                },
            ],
            "edges": [
                {"id": "e1", "source_id": "start", "target_id": "finish", "label": "", "kind": "standard", "priority": 100},
            ],
        }

        document = load_graph_document(payload)
        graph = document.as_graph()
        weather_context = next(node for node in graph.to_dict()["nodes"] if node["id"] == "weather_context")
        self.assertTrue(weather_context["config"]["expose_mcp_tools"])

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
