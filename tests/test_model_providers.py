from __future__ import annotations

from datetime import datetime
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from tempfile import TemporaryDirectory
from threading import Thread
import unittest
from pathlib import Path
from typing import Any, Mapping, Sequence
from unittest.mock import Mock, patch
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from graph_agent.api.graph_store import GraphStore
from graph_agent.api.manager import GraphRunManager
from graph_agent.examples.tool_schema_repair import (
    build_auto_branching_graph_payload,
    build_example_graph_payload,
    build_example_services,
)
from graph_agent.providers.base import ModelMessage, ModelRequest, ModelToolCall, ModelToolDefinition, api_decision_response_schema
from graph_agent.providers.base import ModelProvider, ModelResponse, ProviderPreflightResult
from graph_agent.providers.claude_code import ClaudeCodeCLIModelProvider
from graph_agent.providers.mock import MockModelProvider
from graph_agent.providers.vendor_api import ClaudeMessagesModelProvider, OpenAIChatModelProvider
from graph_agent.runtime.core import GraphDefinition, GraphValidationError, NodeContext, RunState
from graph_agent.runtime.documents import load_graph_document
from graph_agent.runtime.engine import GraphRuntime
from graph_agent.tools.base import ToolContext, ToolDefinition, ToolRegistry, ToolResult
from graph_agent.tools.mcp import McpServerDefinition, McpServerManager, _McpStdioSession


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


def _decision(
    *,
    message: Any = None,
    final_message: Any = None,
    need_tool: bool | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    normalized_tool_calls = list(tool_calls or [])
    resolved_message = message if message is not None else final_message
    resolved_need_tool = bool(normalized_tool_calls) if need_tool is None else bool(need_tool)
    return {
        "message": "" if resolved_need_tool and resolved_message is None else resolved_message,
        "need_tool": resolved_need_tool,
        "tool_calls": normalized_tool_calls,
    }


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


class _HttpMcpStubHandler(BaseHTTPRequestHandler):
    requests: list[dict[str, Any]] = []

    def do_POST(self) -> None:  # noqa: N802
        content_length = int(self.headers.get("Content-Length", "0") or "0")
        payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
        self.__class__.requests.append(payload)
        method = payload.get("method")
        message_id = payload.get("id")
        result: dict[str, Any]

        if method == "initialize":
            result = {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "http-mcp-stub", "version": "0.1.0"},
            }
        elif method == "tools/list":
            result = {
                "tools": [
                    {
                        "name": "echo_http",
                        "description": "Echo a message over HTTP MCP.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"message": {"type": "string"}},
                            "required": ["message"],
                        },
                    }
                ]
            }
        elif method == "tools/call":
            params = payload.get("params", {})
            result = {
                "content": [{"type": "text", "text": "echoed"}],
                "structuredContent": {
                    "tool_name": params.get("name"),
                    "arguments": dict(params.get("arguments", {})),
                },
                "isError": False,
            }
        else:
            result = {}

        if message_id is None:
            self.send_response(202)
            self.end_headers()
            return

        body = json.dumps({"jsonrpc": "2.0", "id": message_id, "result": result}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return


class HttpMcpStubServer:
    def __enter__(self) -> str:
        _HttpMcpStubHandler.requests = []
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _HttpMcpStubHandler)
        self._thread = Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        host, port = self._server.server_address
        return f"http://{host}:{port}/mcp"

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
            structured_output=_decision(
                final_message={
                    "system_prompt": system_prompt,
                    "mcp_tool_context": request.metadata.get("mcp_tool_context"),
                    "available_tool_names": [tool.name for tool in request.available_tools],
                    "mcp_available_tool_names": request.metadata.get("mcp_available_tool_names", []),
                }
            ),
        )

    def preflight(self, provider_config: Mapping[str, Any] | None = None) -> ProviderPreflightResult:
        return ProviderPreflightResult(status="available", ok=True, message="ok")


class McpDispatchProvider(ModelProvider):
    name = "mcp_dispatch"

    def generate(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(
            content="",
            structured_output=_decision(
                tool_calls=[
                    {
                        "tool_name": "weather_current",
                        "arguments": {"location": "Austin"},
                        "provider_tool_id": "provider-tool-1",
                        "metadata": {},
                    }
                ]
            ),
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
            structured_output=_decision(final_message="A direct reply is enough."),
            metadata={"variant": "message"},
        )

    def preflight(self, provider_config: Mapping[str, Any] | None = None) -> ProviderPreflightResult:
        return ProviderPreflightResult(status="available", ok=True, message="ok")


class AutoToolCallProvider(ModelProvider):
    name = "auto_tool_call"

    def generate(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(
            content="",
            structured_output=_decision(
                tool_calls=[
                    {
                        "tool_name": "weather_current",
                        "arguments": {"location": "Austin"},
                        "provider_tool_id": "auto-tool-1",
                        "metadata": {"variant": "tool_call"},
                    }
                ]
            ),
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


class AutoTwoToolCallsProvider(ModelProvider):
    name = "auto_two_tool_calls"

    def generate(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(
            content="",
            structured_output=_decision(
                tool_calls=[
                    {
                        "tool_name": "weather_current",
                        "arguments": {"location": "Austin"},
                        "provider_tool_id": "auto-two-tools-1",
                        "metadata": {"variant": "tool_call"},
                    },
                    {
                        "tool_name": "time_current_minute",
                        "arguments": {},
                        "provider_tool_id": "auto-two-tools-2",
                        "metadata": {"variant": "tool_call"},
                    },
                ]
            ),
            tool_calls=[
                ModelToolCall(
                    tool_name="weather_current",
                    arguments={"location": "Austin"},
                    provider_tool_id="auto-two-tools-1",
                    metadata={"variant": "tool_call"},
                ),
                ModelToolCall(
                    tool_name="time_current_minute",
                    arguments={},
                    provider_tool_id="auto-two-tools-2",
                    metadata={"variant": "tool_call"},
                ),
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
            structured_output=_decision(
                tool_calls=[
                    {
                        "tool_name": "weather_current",
                        "arguments": {"location": "Austin"},
                        "provider_tool_id": "auto-mixed-1",
                        "metadata": {"variant": "mixed"},
                    }
                ]
            ),
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


class PromptBlockEchoProvider(ModelProvider):
    name = "prompt_block_echo"

    def __init__(self) -> None:
        self.last_request: ModelRequest | None = None

    def generate(self, request: ModelRequest) -> ModelResponse:
        self.last_request = request
        return ModelResponse(content="ok", structured_output=_decision(final_message="ok"))

    def preflight(self, provider_config: Mapping[str, Any] | None = None) -> ProviderPreflightResult:
        return ProviderPreflightResult(status="available", ok=True, message="ok")


class UserMessageJsonEchoProvider(ModelProvider):
    name = "user_message_json_echo"

    def __init__(self) -> None:
        self.last_request: ModelRequest | None = None

    def generate(self, request: ModelRequest) -> ModelResponse:
        self.last_request = request
        user_content = request.messages[-1].content if request.messages else ""
        return ModelResponse(
            content="",
            structured_output=_decision(final_message=json.loads(user_content) if user_content else None),
        )

    def preflight(self, provider_config: Mapping[str, Any] | None = None) -> ProviderPreflightResult:
        return ProviderPreflightResult(status="available", ok=True, message="ok")


class RecheckLoopProvider(ModelProvider):
    name = "recheck_loop"

    def __init__(self) -> None:
        self.last_request: ModelRequest | None = None
        self.seen_payloads: list[dict[str, Any]] = []

    def generate(self, request: ModelRequest) -> ModelResponse:
        self.last_request = request
        payload = json.loads(request.messages[-1].content) if request.messages else {}
        assert isinstance(payload, dict)
        self.seen_payloads.append(payload)
        current_location = payload.get("tool_arguments", {}).get("location")
        if current_location == "Austin":
            return ModelResponse(
                content="",
                structured_output=_decision(
                    tool_calls=[
                        {
                            "tool_name": "weather_current",
                            "arguments": {"location": "Seattle"},
                            "provider_tool_id": "recheck-tool-1",
                            "metadata": {"variant": "loop"},
                        }
                    ]
                ),
                tool_calls=[
                    ModelToolCall(
                        tool_name="weather_current",
                        arguments={"location": "Seattle"},
                        provider_tool_id="recheck-tool-1",
                        metadata={"variant": "loop"},
                    )
                ],
                metadata={"variant": "loop"},
            )
        return ModelResponse(
            content="",
            structured_output=_decision(
                final_message=f"No additional tools needed after {payload.get('tool_name', 'tool')}."
            ),
            metadata={"variant": "message"},
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

    def _build_mcp_followup_executor_graph(
        self,
        *,
        initial_provider_name: str,
        followup_provider_name: str,
        tool_name: str,
        allowed_tool_names: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        tool_names = list(allowed_tool_names) if allowed_tool_names else [tool_name]
        return {
            "graph_id": f"mcp-followup-executor-{tool_name}",
            "name": "MCP Follow-Up Executor Graph",
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
                    "id": "tool_context",
                    "kind": "mcp_context_provider",
                    "category": "tool",
                    "label": "Tool Context",
                    "provider_id": "tool.mcp_context_provider",
                    "provider_label": "MCP Context Provider",
                    "config": {"tool_names": tool_names, "include_mcp_tool_context": False},
                    "position": {"x": 120, "y": 0},
                },
                {
                    "id": "model",
                    "kind": "model",
                    "category": "api",
                    "label": "Initial Model",
                    "provider_id": "core.api",
                    "provider_label": "API Call Node",
                    "model_provider_name": initial_provider_name,
                    "prompt_name": "initial_prompt",
                    "config": {
                        "provider_name": initial_provider_name,
                        "prompt_name": "initial_prompt",
                        "system_prompt": "Choose an MCP tool call.",
                        "user_message_template": "{input_payload}",
                    },
                    "position": {"x": 260, "y": 0},
                },
                {
                    "id": "executor",
                    "kind": "mcp_tool_executor",
                    "category": "tool",
                    "label": "Executor",
                    "provider_id": "tool.mcp_tool_executor",
                    "provider_label": "MCP Tool Executor",
                    "config": {
                        "enable_follow_up_decision": True,
                        "provider_name": followup_provider_name,
                        "prompt_name": "followup_prompt",
                        "mode": "followup_prompt",
                        "system_prompt": "Review the last MCP execution and decide whether another tool is needed.",
                        "user_message_template": "{input_payload}",
                        "response_mode": "auto",
                        "allowed_tool_names": tool_names,
                    },
                    "position": {"x": 500, "y": 0},
                },
                {
                    "id": "finish",
                    "kind": "output",
                    "category": "end",
                    "label": "Finish",
                    "provider_id": "core.output",
                    "provider_label": "Core Output Node",
                    "config": {"source_binding": {"type": "latest_payload", "source": "executor"}},
                    "position": {"x": 760, "y": 0},
                },
            ],
            "edges": [
                {"id": "start-model", "source_id": "start", "target_id": "model", "label": "", "kind": "standard", "priority": 100},
                {
                    "id": "ctx-binding",
                    "source_id": "tool_context",
                    "target_id": "model",
                    "source_handle_id": "tool-context",
                    "target_handle_id": "api-tool-context",
                    "label": "tool context",
                    "kind": "binding",
                    "priority": 0,
                },
                {
                    "id": "model-executor",
                    "source_id": "model",
                    "target_id": "executor",
                    "source_handle_id": "api-tool-call",
                    "label": "tool call",
                    "kind": "conditional",
                    "priority": 100,
                    "condition": {
                        "id": "model-executor-condition",
                        "label": "Tool call output",
                        "type": "result_payload_path_equals",
                        "value": "tool_call_envelope",
                        "path": "metadata.contract",
                    },
                },
                {"id": "executor-finish", "source_id": "executor", "target_id": "finish", "label": "", "kind": "standard", "priority": 100},
            ],
        }

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

        self.assertEqual(
            response.structured_output,
            _decision(
                tool_calls=[
                    {
                        "tool_name": "search_catalog",
                        "arguments": {"query": "graph agents", "limit": 3},
                        "provider_tool_id": "call-1",
                        "metadata": {"provider_type": "function"},
                    }
                ]
            ),
        )
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

        self.assertEqual(
            response.structured_output,
            _decision(
                tool_calls=[
                    {
                        "tool_name": "search_catalog",
                        "arguments": {"query": "graph agents", "limit": 3},
                        "provider_tool_id": "toolu_1",
                        "metadata": {},
                    }
                ]
            ),
        )
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

        self.assertEqual(
            response.structured_output,
            _decision(
                tool_calls=[
                    {
                        "tool_name": "search_catalog",
                        "arguments": {"query": "graph agents", "limit": 3},
                        "provider_tool_id": None,
                        "metadata": {},
                    }
                ]
            ),
        )
        self.assertEqual(len(response.tool_calls), 1)
        self.assertEqual(response.tool_calls[0].tool_name, "search_catalog")
        self.assertIsNotNone(provider.last_command)
        assert provider.last_command is not None
        self.assertIn("--json-schema", provider.last_command)
        self.assertIn("--system-prompt", provider.last_command)
        self.assertIn("--tools", provider.last_command)
        self.assertEqual(provider.last_cwd, str(ROOT))
        self.assertEqual(provider.last_timeout_seconds, 15.0)

    def test_claude_code_provider_uses_object_schema_for_multiple_tools(self) -> None:
        provider = ClaudeCodeCLIModelProvider()
        tools = [
            ModelToolDefinition(
                name="weather_current",
                description="Fetch the weather.",
                input_schema={
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            ),
            ModelToolDefinition(
                name="time_current_minute",
                description="Fetch the current time.",
                input_schema={
                    "type": "object",
                    "properties": {},
                },
            ),
        ]
        request = ModelRequest(
            prompt_name="followup_prompt",
            messages=[ModelMessage(role="user", content="Decide whether another tool is needed.")],
            response_schema=api_decision_response_schema(available_tools=tools),
            available_tools=tools,
            response_mode="auto",
        )

        schema = provider._response_schema(request, tools)

        assert schema is not None
        self.assertEqual(schema["type"], "object")
        self.assertEqual(schema["properties"]["need_tool"]["type"], "boolean")
        self.assertEqual(
            schema["properties"]["tool_calls"]["items"]["properties"]["tool_name"]["enum"],
            ["weather_current", "time_current_minute"],
        )
        self.assertEqual(schema["properties"]["tool_calls"]["items"]["properties"]["arguments"]["type"], "object")
        self.assertEqual(schema["required"], ["message", "need_tool", "tool_calls"])

    def test_mock_provider_emits_tool_calls_for_tool_call_modes(self) -> None:
        provider = MockModelProvider()

        for prompt_name in ("schema_proposal", "schema_repair", "research_query", "executor_plan"):
            with self.subTest(prompt_name=prompt_name):
                request = ModelRequest(
                    prompt_name=prompt_name,
                    messages=[ModelMessage(role="user", content="Build the tool payload.")],
                    provider_config={"model": "mock-default"},
                    response_mode="tool_call",
                    preferred_tool_name="search_catalog",
                    available_tools=[SEARCH_CATALOG_TOOL],
                    metadata={"mode": prompt_name, "user_request": "graph agents"},
                )

                response = provider.generate(request)

                self.assertEqual(
                    response.structured_output,
                    _decision(
                        tool_calls=[
                            {
                                "tool_name": "search_catalog",
                                "arguments": {
                                    "query": "graph agents",
                                    "limit": "three" if prompt_name == "schema_proposal" else 3,
                                },
                            }
                        ]
                    ),
                )
                self.assertEqual(len(response.tool_calls), 1)
                self.assertEqual(response.tool_calls[0].tool_name, "search_catalog")

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
        self.assertTrue(any(server["server_id"] == "time_mcp" for server in catalog["mcp_servers"]))
        weather_tool = next(tool for tool in catalog["tools"] if tool["name"] == "weather_current")
        time_tool = next(tool for tool in catalog["tools"] if tool["name"] == "time_current_minute")
        self.assertEqual(weather_tool["source_type"], "mcp")
        self.assertTrue(weather_tool["enabled"])
        self.assertFalse(weather_tool["available"])
        self.assertEqual(weather_tool["schema_origin"], "static")
        self.assertEqual(time_tool["source_type"], "mcp")
        self.assertTrue(time_tool["enabled"])
        self.assertFalse(time_tool["available"])
        self.assertEqual(time_tool["schema_origin"], "static")

    def test_offline_mcp_tools_do_not_block_graph_save_or_run_start(self) -> None:
        services = build_example_services()
        with TemporaryDirectory() as tmp_dir:
            bundled_path = Path(tmp_dir) / "bundled_graphs_store.json"
            bundled_path.write_text('{"graphs":[]}', encoding="utf-8")
            store = GraphStore(
                services,
                path=Path(tmp_dir) / "graphs_store.json",
                bundled_path=bundled_path,
            )
            manager = GraphRunManager(services=services, store=store)
            payload = build_example_graph_payload()
            graph_id = f"offline-mcp-graph-{uuid4()}"
            payload["graph_id"] = graph_id
            payload["name"] = "Offline MCP Graph"

            for node in payload["nodes"]:
                if node["kind"] == "model":
                    node["config"]["response_mode"] = "tool_call"
                    node["config"]["allowed_tool_names"] = ["weather_current"]
                    node["config"]["preferred_tool_name"] = "weather_current"
                if node["kind"] == "tool":
                    node["tool_name"] = "weather_current"
                    node["config"]["tool_name"] = "weather_current"

            manager.set_mcp_tool_enabled("weather_current", True)
            graph = GraphDefinition.from_dict(payload)
            graph.validate_against_services(services)

            saved_graph = manager.create_graph(payload)
            self.assertEqual(saved_graph["graph_id"], graph_id)

            run_id = manager.start_run(graph_id, {"request": "weather please"})
            snapshot = manager.get_run(run_id)
            self.assertIn(snapshot["status"], {"queued", "running", "failed", "completed"})

            result = services.tool_registry.invoke(
                "weather_current",
                {"location": "Austin"},
                ToolContext(run_id="run-offline", graph_id=graph_id, node_id="tool-node", state_snapshot={}),
            )
            self.assertEqual(result.status, "unavailable")
            assert result.error is not None
            self.assertIn("offline", result.error["message"].lower())

    def test_booted_mcp_tools_default_to_enabled_and_execute(self) -> None:
        services = build_example_services()
        manager = GraphRunManager(services=services)
        with WeatherStubServer() as weather_base:
            try:
                with patch.dict("os.environ", {"GRAPH_AGENT_WEATHER_API_BASE": weather_base}, clear=False):
                    server = manager.boot_mcp_server("weather_mcp")
                    self.assertTrue(server["running"])

                    catalog = manager.get_catalog()
                    weather_tool = next(tool for tool in catalog["tools"] if tool["name"] == "weather_current")
                    self.assertTrue(weather_tool["enabled"])
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

    def test_booted_enabled_time_mcp_tool_returns_current_minute(self) -> None:
        services = build_example_services()
        manager = GraphRunManager(services=services)
        try:
            server = manager.boot_mcp_server("time_mcp")
            self.assertTrue(server["running"])

            result = services.tool_registry.invoke(
                "time_current_minute",
                {},
                ToolContext(run_id="run-time", graph_id="graph-time", node_id="tool-node", state_snapshot={}),
            )
            self.assertEqual(result.status, "success")
            payload = result.output
            self.assertIsInstance(payload, dict)
            parsed = datetime.fromisoformat(payload["local_iso_minute"])
            self.assertEqual(parsed.second, 0)
            self.assertEqual(parsed.microsecond, 0)
            self.assertEqual(payload["hour_24"], parsed.hour)
            self.assertEqual(payload["minute"], parsed.minute)
            self.assertIsInstance(payload["timezone"], str)
            self.assertIsInstance(payload["utc_offset"], str)
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
            self.assertFalse(weather_tool["enabled"])
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
                    self.assertNotIn("MCP Tool Decision Output", payload["system_prompt"])
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
                    self.assertIn("MCP Tool Decision Output", payload["system_prompt"])
                    self.assertIn("Uses Tool: True|False", payload["system_prompt"])
                    self.assertIn('Tool Call Schema: {"tool_name":"<tool name>","arguments":{...}} or NA', payload["system_prompt"])
                    self.assertIn("DELIMITER", payload["system_prompt"])
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
                    manager.set_mcp_tool_enabled("weather_current", False)
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
                    self.assertEqual(state.final_output, "A direct reply is enough.")
                    model_output = state.node_outputs["model"]
                    assert isinstance(model_output, dict)
                    self.assertEqual(model_output["metadata"]["contract"], "message_envelope")
                    self.assertEqual(model_output["payload"], "A direct reply is enough.")
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
                    assert isinstance(payload, dict)
                    self.assertEqual(payload["resolved_location"], "Testville")
                    self.assertEqual(payload["condition"], "Partly cloudy")
                    self.assertEqual(state.visit_counts["finish"], 2)
                    model_output = state.node_outputs["model"]
                    assert isinstance(model_output, dict)
                    self.assertEqual(model_output["metadata"]["contract"], "tool_call_envelope")
                    self.assertEqual(model_output["metadata"]["response_mode"], "tool_call")
                    self.assertEqual(model_output["artifacts"]["source_input_payload"], {"request": "weather please"})
                    self.assertEqual(model_output["tool_calls"][0]["tool_name"], "weather_current")
                    tool_route_output = state.edge_outputs["edge-model-executor"]
                    assert isinstance(tool_route_output, dict)
                    self.assertEqual(tool_route_output["metadata"]["contract"], "tool_call_envelope")
                    self.assertEqual(tool_route_output["artifacts"]["source_input_payload"], {"request": "weather please"})
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
                    self.assertTrue(any(item == "Checking the live weather tool now." for item in finish_outputs))
                    self.assertTrue(any(isinstance(item, dict) and item.get("resolved_location") == "Testville" for item in finish_outputs))
            finally:
                manager.stop_background_services()

    def test_tool_call_output_fans_out_to_multiple_parallel_branches(self) -> None:
        services, runtime, graph_payload = self._build_auto_branch_graph("parallel_tool_call", AutoToolCallProvider())
        for node in graph_payload["nodes"]:
            if node["id"] == "model":
                node["config"]["response_mode"] = "tool_call"
        graph_payload["nodes"].extend(
            [
                {
                    "id": "tool_display",
                    "kind": "data",
                    "category": "data",
                    "label": "Tool Display",
                    "provider_id": "core.data_display",
                    "provider_label": "Envelope Display Node",
                    "config": {
                        "mode": "passthrough",
                        "show_input_envelope": True,
                        "lock_passthrough": True,
                    },
                    "position": {"x": 520, "y": 180},
                },
                {
                    "id": "display_finish",
                    "kind": "output",
                    "category": "end",
                    "label": "Display Finish",
                    "provider_id": "core.output",
                    "provider_label": "Core Output Node",
                    "config": {"source_binding": {"type": "latest_payload", "source": "tool_display"}},
                    "position": {"x": 760, "y": 180},
                },
            ]
        )
        graph_payload["edges"].extend(
            [
                {
                    "id": "edge-model-display",
                    "source_id": "model",
                    "target_id": "tool_display",
                    "source_handle_id": "api-tool-call",
                    "label": "tool call display",
                    "kind": "conditional",
                    "priority": 110,
                    "condition": {
                        "id": "edge-model-display-condition",
                        "label": "Tool call output",
                        "type": "result_payload_path_equals",
                        "value": "tool_call_envelope",
                        "path": "metadata.contract",
                    },
                },
                {
                    "id": "edge-display-finish",
                    "source_id": "tool_display",
                    "target_id": "display_finish",
                    "label": "",
                    "kind": "standard",
                    "priority": 100,
                },
            ]
        )
        graph = GraphDefinition.from_dict(graph_payload)
        manager = GraphRunManager(services=services)
        with WeatherStubServer() as weather_base:
            try:
                with patch.dict("os.environ", {"GRAPH_AGENT_WEATHER_API_BASE": weather_base}, clear=False):
                    manager.boot_mcp_server("weather_mcp")
                    manager.set_mcp_tool_enabled("weather_current", True)
                    graph.validate_against_services(services)

                    state = runtime.run(graph, {"request": "weather please"}, run_id="run-parallel-tool-fanout")
                    self.assertEqual(state.status, "completed")
                    payload = state.final_output
                    assert isinstance(payload, dict)
                    self.assertEqual(payload["resolved_location"], "Testville")
                    self.assertEqual(state.visit_counts["executor"], 1)
                    self.assertEqual(state.visit_counts["tool_display"], 1)
                    self.assertEqual(state.visit_counts["display_finish"], 1)
                    transition_edges = [transition.edge_id for transition in state.transition_history]
                    self.assertIn("edge-model-executor", transition_edges)
                    self.assertIn("edge-model-display", transition_edges)
                    display_edge_output = state.edge_outputs["edge-model-display"]
                    assert isinstance(display_edge_output, dict)
                    self.assertEqual(display_edge_output["metadata"]["contract"], "tool_call_envelope")
                    self.assertEqual(display_edge_output["tool_calls"][0]["tool_name"], "weather_current")
                    display_output = state.node_outputs["tool_display"]
                    assert isinstance(display_output, dict)
                    self.assertEqual(display_output["metadata"]["contract"], "tool_call_envelope")
                    self.assertEqual(display_output["tool_calls"][0]["tool_name"], "weather_current")
            finally:
                manager.stop_background_services()

    def test_mcp_executor_receives_tool_call_through_display_passthrough_when_envelope_preview_is_disabled(self) -> None:
        services, runtime, graph_payload = self._build_auto_branch_graph("auto_tool_call", AutoToolCallProvider())
        for node in graph_payload["nodes"]:
            if node["id"] == "executor":
                node["config"]["input_binding"] = {"type": "latest_envelope", "source": "start"}
        graph_payload["edges"] = [edge for edge in graph_payload["edges"] if edge["id"] != "edge-model-executor"]
        graph_payload["nodes"].append(
            {
                "id": "tool_display",
                "kind": "data",
                "category": "data",
                "label": "Tool Display",
                "provider_id": "core.data_display",
                "provider_label": "Envelope Display Node",
                "config": {
                    "mode": "passthrough",
                    "show_input_envelope": False,
                    "lock_passthrough": True,
                },
                "position": {"x": 520, "y": 180},
            }
        )
        graph_payload["edges"].extend(
            [
                {
                    "id": "edge-model-display",
                    "source_id": "model",
                    "target_id": "tool_display",
                    "source_handle_id": "api-tool-call",
                    "label": "tool call display",
                    "kind": "conditional",
                    "priority": 110,
                    "condition": {
                        "id": "edge-model-display-condition",
                        "label": "Tool call output",
                        "type": "result_payload_path_equals",
                        "value": "tool_call_envelope",
                        "path": "metadata.contract",
                    },
                },
                {
                    "id": "edge-display-executor",
                    "source_id": "tool_display",
                    "target_id": "executor",
                    "label": "to executor",
                    "kind": "standard",
                    "priority": 100,
                },
            ]
        )
        graph = GraphDefinition.from_dict(graph_payload)
        manager = GraphRunManager(services=services)
        with WeatherStubServer() as weather_base:
            try:
                with patch.dict("os.environ", {"GRAPH_AGENT_WEATHER_API_BASE": weather_base}, clear=False):
                    manager.boot_mcp_server("weather_mcp")
                    manager.set_mcp_tool_enabled("weather_current", True)
                    graph.validate_against_services(services)

                    state = runtime.run(graph, {"request": "weather please"}, run_id="run-display-passthrough-to-mcp-executor")
                    self.assertEqual(state.status, "completed")
                    self.assertNotIn("executor", state.node_errors)
                    executor_output = state.node_outputs["executor"]
                    assert isinstance(executor_output, dict)
                    self.assertEqual(executor_output["metadata"]["tool_status"], "success")
                    self.assertEqual(executor_output["metadata"]["tool_name"], "weather_current")
            finally:
                manager.stop_background_services()

    def test_api_model_preserves_multiple_tool_call_schemas_in_tool_envelope(self) -> None:
        class MultiToolProvider(ModelProvider):
            name = "multi_tool"

            def generate(self, request: ModelRequest) -> ModelResponse:
                return ModelResponse(
                    content="",
                    structured_output=_decision(
                        tool_calls=[
                            {"tool_name": "weather_current", "arguments": {"location": "Austin"}},
                            {"tool_name": "weather_current", "arguments": {"location": "Seattle"}},
                        ]
                    ),
                    tool_calls=[
                        ModelToolCall(tool_name="weather_current", arguments={"location": "Austin"}),
                        ModelToolCall(tool_name="weather_current", arguments={"location": "Seattle"}),
                    ],
                )

            def preflight(self, provider_config: Mapping[str, Any] | None = None) -> ProviderPreflightResult:
                return ProviderPreflightResult(status="available", ok=True, message="ok")

        services, runtime, graph_payload = self._build_auto_branch_graph("multi_tool", MultiToolProvider())
        graph = GraphDefinition.from_dict(graph_payload)
        manager = GraphRunManager(services=services)
        with WeatherStubServer() as weather_base:
            try:
                with patch.dict("os.environ", {"GRAPH_AGENT_WEATHER_API_BASE": weather_base}, clear=False):
                    manager.boot_mcp_server("weather_mcp")
                    manager.set_mcp_tool_enabled("weather_current", True)
                    graph.validate_against_services(services)

                    state = runtime.run(graph, {"request": "weather please"}, run_id="run-multi-tool")
                    self.assertEqual(state.status, "completed")
                    model_output = state.node_outputs["model"]
                    assert isinstance(model_output, dict)
                    self.assertEqual(model_output["metadata"]["contract"], "tool_call_envelope")
                    self.assertEqual(len(model_output["tool_calls"]), 2)
                    self.assertEqual(model_output["tool_calls"][0]["arguments"], {"location": "Austin"})
                    self.assertEqual(model_output["tool_calls"][1]["arguments"], {"location": "Seattle"})
            finally:
                manager.stop_background_services()

    def test_api_model_rejects_invalid_structured_output_combinations(self) -> None:
        class InvalidStructuredProvider(ModelProvider):
            name = "invalid_structured"

            def generate(self, request: ModelRequest) -> ModelResponse:
                return ModelResponse(
                    content="",
                    structured_output={
                        "should_call_tools": False,
                        "tool_calls": [{"tool_name": "weather_current", "arguments": {"location": "Austin"}}],
                        "final_message": "This should have been empty.",
                    },
                )

            def preflight(self, provider_config: Mapping[str, Any] | None = None) -> ProviderPreflightResult:
                return ProviderPreflightResult(status="available", ok=True, message="ok")

        services, runtime, graph_payload = self._build_auto_branch_graph("invalid_structured", InvalidStructuredProvider())
        graph = GraphDefinition.from_dict(graph_payload)
        manager = GraphRunManager(services=services)
        with WeatherStubServer() as weather_base:
            try:
                with patch.dict("os.environ", {"GRAPH_AGENT_WEATHER_API_BASE": weather_base}, clear=False):
                    manager.boot_mcp_server("weather_mcp")
                    manager.set_mcp_tool_enabled("weather_current", True)
                    graph.validate_against_services(services)

                    state = runtime.run(graph, {"request": "weather please"}, run_id="run-invalid-structured")
                    self.assertEqual(state.status, "failed")
                    assert isinstance(state.terminal_error, dict)
                    self.assertEqual(state.terminal_error["type"], "no_matching_edge")
                    self.assertEqual(state.terminal_error["node_id"], "model")
                    self.assertEqual(state.terminal_error["node_label"], graph.get_node("model").label)
                    self.assertIn("no outgoing edge matched", state.terminal_error["message"].lower())
                    self.assertEqual(state.terminal_error["result_contract"], "message_envelope")
                    self.assertTrue(state.terminal_error["available_routes"])
                    self.assertIn("model", state.node_errors)
                    self.assertEqual(state.node_errors["model"]["type"], "structured_api_output_error")
                    self.assertIn("must leave 'tool_calls' empty", state.node_errors["model"]["message"])
            finally:
                manager.stop_background_services()

    def test_tool_call_mode_rejects_final_message_decisions(self) -> None:
        class ToolCallBypassProvider(ModelProvider):
            name = "tool_call_bypass"

            def generate(self, request: ModelRequest) -> ModelResponse:
                return ModelResponse(
                    content="",
                    structured_output={
                        "should_call_tools": False,
                        "tool_calls": [],
                        "final_message": "I answered directly instead of calling a tool.",
                    },
                )

            def preflight(self, provider_config: Mapping[str, Any] | None = None) -> ProviderPreflightResult:
                return ProviderPreflightResult(status="available", ok=True, message="ok")

        services, runtime, graph_payload = self._build_auto_branch_graph("tool_call_bypass", ToolCallBypassProvider())
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

                    state = runtime.run(graph, {"request": "weather please"}, run_id="run-tool-call-bypass")
                    self.assertEqual(state.status, "failed")
                    assert isinstance(state.terminal_error, dict)
                    self.assertEqual(state.terminal_error["type"], "no_matching_edge")
                    self.assertEqual(state.terminal_error["node_id"], "model")
                    self.assertEqual(state.terminal_error["node_label"], graph.get_node("model").label)
                    self.assertIn("no outgoing edge matched", state.terminal_error["message"].lower())
                    self.assertEqual(state.terminal_error["result_contract"], "message_envelope")
                    self.assertTrue(state.terminal_error["available_routes"])
                    self.assertIn("model", state.node_errors)
                    self.assertEqual(state.node_errors["model"]["type"], "structured_api_output_error")
                    self.assertIn("requires 'need_tool' to be true", state.node_errors["model"]["message"])
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

    def test_mcp_executor_validation_rejects_mcp_context_provider_bindings(self) -> None:
        services = build_example_services()
        graph_payload = self._build_mcp_followup_executor_graph(
            initial_provider_name="mock",
            followup_provider_name="mock",
            tool_name="weather_current",
        )
        graph_payload["edges"].append(
            {
                "id": "ctx-to-executor",
                "source_id": "tool_context",
                "target_id": "executor",
                "source_handle_id": "tool-context",
                "target_handle_id": None,
                "label": "tool context",
                "kind": "binding",
                "priority": 0,
            }
        )

        with self.assertRaises(GraphValidationError):
            GraphDefinition.from_dict(graph_payload).validate_against_services(services)

    def test_prompt_block_node_emits_structured_data_envelope(self) -> None:
        services = build_example_services()
        graph = GraphDefinition.from_dict(
            {
                "graph_id": "prompt-block-runtime",
                "name": "Prompt Block Runtime",
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
                        "id": "prompt_block",
                        "kind": "data",
                        "category": "data",
                        "label": "System Prompt",
                        "provider_id": "core.prompt_block",
                        "provider_label": "Prompt Block",
                        "config": {
                            "mode": "prompt_block",
                            "role": "system",
                            "name": "instructions",
                            "content": "You are helping with {graph_id}.",
                        },
                        "position": {"x": 120, "y": 0},
                    },
                    {
                        "id": "finish",
                        "kind": "output",
                        "category": "end",
                        "label": "Finish",
                        "provider_id": "core.output",
                        "provider_label": "Core Output Node",
                        "config": {"source_binding": {"type": "latest_payload", "source": "start"}},
                        "position": {"x": 240, "y": 0},
                    },
                ],
                "edges": [
                    {"id": "start-finish", "source_id": "start", "target_id": "finish", "label": "", "kind": "standard", "priority": 100},
                ],
            }
        )
        graph.validate_against_services(services)

        prompt_node = graph.get_node("prompt_block")
        context = NodeContext(graph=graph, state=RunState(graph_id=graph.graph_id, input_payload="hello"), services=services, node_id="prompt_block")
        result = prompt_node.execute(context)

        self.assertEqual(result.status, "success")
        assert isinstance(result.output, dict)
        self.assertEqual(result.output["metadata"]["contract"], "data_envelope")
        self.assertEqual(result.output["metadata"]["data_mode"], "prompt_block")
        self.assertEqual(
            result.output["payload"],
            {
                "kind": "prompt_block",
                "role": "system",
                "name": "instructions",
                "content": "You are helping with prompt-block-runtime.",
            },
        )

    def test_context_builder_renders_bound_prompt_blocks_without_executing_them(self) -> None:
        services = build_example_services()
        runtime = GraphRuntime(
            services=services,
            max_steps=services.config["max_steps"],
            max_visits_per_node=services.config["max_visits_per_node"],
        )
        graph = GraphDefinition.from_dict(
            {
                "graph_id": "prompt-block-context-builder",
                "name": "Prompt Block Context Builder",
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
                        "id": "system_block",
                        "kind": "data",
                        "category": "data",
                        "label": "System Block",
                        "provider_id": "core.prompt_block",
                        "provider_label": "Prompt Block",
                        "config": {"mode": "prompt_block", "role": "system", "content": "Follow the graph rules."},
                        "position": {"x": 120, "y": 0},
                    },
                    {
                        "id": "user_block",
                        "kind": "data",
                        "category": "data",
                        "label": "User Block",
                        "provider_id": "core.prompt_block",
                        "provider_label": "Prompt Block",
                        "config": {"mode": "prompt_block", "role": "user", "content": "Question: {input_payload}"},
                        "position": {"x": 120, "y": 120},
                    },
                    {
                        "id": "compose",
                        "kind": "data",
                        "category": "data",
                        "label": "Compose",
                        "provider_id": "core.context_builder",
                        "provider_label": "Context Builder",
                        "config": {"mode": "context_builder", "template": "", "input_bindings": [], "joiner": "\n\n"},
                        "position": {"x": 320, "y": 0},
                    },
                    {
                        "id": "finish",
                        "kind": "output",
                        "category": "end",
                        "label": "Finish",
                        "provider_id": "core.output",
                        "provider_label": "Core Output Node",
                        "config": {"source_binding": {"type": "latest_payload", "source": "compose"}},
                        "position": {"x": 520, "y": 0},
                    },
                ],
                "edges": [
                    {"id": "start-compose", "source_id": "start", "target_id": "compose", "label": "", "kind": "standard", "priority": 100},
                    {"id": "system-compose", "source_id": "system_block", "target_id": "compose", "label": "prompt block", "kind": "binding", "priority": 0},
                    {"id": "user-compose", "source_id": "user_block", "target_id": "compose", "label": "prompt block", "kind": "binding", "priority": 0},
                    {"id": "compose-finish", "source_id": "compose", "target_id": "finish", "label": "", "kind": "standard", "priority": 100},
                ],
            }
        )
        graph.validate_against_services(services)

        state = runtime.run(graph, "How do prompt blocks work?", run_id="run-prompt-context-builder")

        self.assertEqual(state.status, "completed")
        self.assertEqual(
            state.final_output,
            [
                {"role": "user", "content": "How do prompt blocks work?"},
                {"role": "system", "content": "Follow the graph rules."},
                {"role": "user", "content": "Question: How do prompt blocks work?"},
            ],
        )
        self.assertNotIn("system_block", state.node_outputs)
        self.assertNotIn("user_block", state.node_outputs)

    def test_context_builder_compiles_multiple_user_prompt_blocks_into_one_turn(self) -> None:
        services = build_example_services()
        runtime = GraphRuntime(
            services=services,
            max_steps=services.config["max_steps"],
            max_visits_per_node=services.config["max_visits_per_node"],
        )
        graph = GraphDefinition.from_dict(
            {
                "graph_id": "prompt-block-context-builder-multi-user",
                "name": "Prompt Block Context Builder Multi User",
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
                        "id": "first_user_block",
                        "kind": "data",
                        "category": "data",
                        "label": "First Request",
                        "provider_id": "core.prompt_block",
                        "provider_label": "Prompt Block",
                        "config": {"mode": "prompt_block", "role": "user", "content": "Summarize the issue."},
                        "position": {"x": 120, "y": 0},
                    },
                    {
                        "id": "second_user_block",
                        "kind": "data",
                        "category": "data",
                        "label": "Second Request",
                        "provider_id": "core.prompt_block",
                        "provider_label": "Prompt Block",
                        "config": {"mode": "prompt_block", "role": "user", "content": "Suggest the next debugging step."},
                        "position": {"x": 120, "y": 120},
                    },
                    {
                        "id": "compose",
                        "kind": "data",
                        "category": "data",
                        "label": "Compose",
                        "provider_id": "core.context_builder",
                        "provider_label": "Context Builder",
                        "config": {"mode": "context_builder", "template": "", "input_bindings": [], "joiner": "\n\n"},
                        "position": {"x": 320, "y": 0},
                    },
                    {
                        "id": "finish",
                        "kind": "output",
                        "category": "end",
                        "label": "Finish",
                        "provider_id": "core.output",
                        "provider_label": "Core Output Node",
                        "config": {"source_binding": {"type": "latest_payload", "source": "compose"}},
                        "position": {"x": 520, "y": 0},
                    },
                ],
                "edges": [
                    {"id": "start-compose", "source_id": "start", "target_id": "compose", "label": "", "kind": "standard", "priority": 100},
                    {"id": "first-compose", "source_id": "first_user_block", "target_id": "compose", "label": "prompt block", "kind": "binding", "priority": 0},
                    {"id": "second-compose", "source_id": "second_user_block", "target_id": "compose", "label": "prompt block", "kind": "binding", "priority": 0},
                    {"id": "compose-finish", "source_id": "compose", "target_id": "finish", "label": "", "kind": "standard", "priority": 100},
                ],
            }
        )
        graph.validate_against_services(services)

        state = runtime.run(graph, "unused input", run_id="run-prompt-context-builder-multi-user")

        self.assertEqual(state.status, "completed")
        self.assertEqual(
            state.final_output,
            [
                {"role": "user", "content": "unused input"},
                {"role": "user", "content": "Summarize the issue."},
                {"role": "user", "content": "Suggest the next debugging step."},
            ],
        )

    def test_context_builder_flattens_multiple_display_node_message_outputs_into_one_stack(self) -> None:
        services = build_example_services()
        services.model_providers["auto_message"] = AutoMessageProvider()
        runtime = GraphRuntime(
            services=services,
            max_steps=services.config["max_steps"],
            max_visits_per_node=services.config["max_visits_per_node"],
        )
        graph = GraphDefinition.from_dict(
            {
                "graph_id": "display-message-context-builder",
                "name": "Display Message Context Builder",
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
                        "id": "model_a",
                        "kind": "model",
                        "category": "api",
                        "label": "Model A",
                        "provider_id": "core.api",
                        "provider_label": "API Call Node",
                        "model_provider_name": "auto_message",
                        "prompt_name": "auto_message_a",
                        "config": {
                            "provider_name": "auto_message",
                            "prompt_name": "auto_message_a",
                            "mode": "auto_message_a",
                            "system_prompt": "Return a direct reply.",
                            "user_message_template": "{input_payload}",
                            "response_mode": "message",
                        },
                        "position": {"x": 180, "y": 0},
                    },
                    {
                        "id": "model_b",
                        "kind": "model",
                        "category": "api",
                        "label": "Model B",
                        "provider_id": "core.api",
                        "provider_label": "API Call Node",
                        "model_provider_name": "auto_message",
                        "prompt_name": "auto_message_b",
                        "config": {
                            "provider_name": "auto_message",
                            "prompt_name": "auto_message_b",
                            "mode": "auto_message_b",
                            "system_prompt": "Return a direct reply.",
                            "user_message_template": "{input_payload}",
                            "response_mode": "message",
                        },
                        "position": {"x": 180, "y": 140},
                    },
                    {
                        "id": "display_a",
                        "kind": "data",
                        "category": "data",
                        "label": "Display A",
                        "provider_id": "core.data_display",
                        "provider_label": "Envelope Display Node",
                        "config": {
                            "mode": "passthrough",
                            "show_input_envelope": True,
                            "lock_passthrough": True,
                        },
                        "position": {"x": 360, "y": 0},
                    },
                    {
                        "id": "display_b",
                        "kind": "data",
                        "category": "data",
                        "label": "Display B",
                        "provider_id": "core.data_display",
                        "provider_label": "Envelope Display Node",
                        "config": {
                            "mode": "passthrough",
                            "show_input_envelope": True,
                            "lock_passthrough": True,
                        },
                        "position": {"x": 360, "y": 140},
                    },
                    {
                        "id": "compose",
                        "kind": "data",
                        "category": "data",
                        "label": "Compose",
                        "provider_id": "core.context_builder",
                        "provider_label": "Context Builder",
                        "config": {
                            "mode": "context_builder",
                            "template": "",
                            "joiner": "\n\n",
                            "input_bindings": [
                                {"source_node_id": "display_a", "placeholder": "display_a", "binding": {"type": "latest_payload", "source": "display_a"}},
                                {"source_node_id": "display_b", "placeholder": "display_b", "binding": {"type": "latest_payload", "source": "display_b"}},
                            ],
                        },
                        "position": {"x": 560, "y": 70},
                    },
                    {
                        "id": "finish",
                        "kind": "output",
                        "category": "end",
                        "label": "Finish",
                        "provider_id": "core.output",
                        "provider_label": "Core Output Node",
                        "config": {"source_binding": {"type": "latest_payload", "source": "compose"}},
                        "position": {"x": 760, "y": 70},
                    },
                ],
                "edges": [
                    {"id": "start-model-a", "source_id": "start", "target_id": "model_a", "label": "", "kind": "standard", "priority": 100},
                    {"id": "model-a-display", "source_id": "model_a", "target_id": "display_a", "label": "", "kind": "standard", "priority": 100},
                    {"id": "display-a-model-b", "source_id": "display_a", "target_id": "model_b", "label": "", "kind": "standard", "priority": 100},
                    {"id": "model-b-display", "source_id": "model_b", "target_id": "display_b", "label": "", "kind": "standard", "priority": 100},
                    {"id": "display-a-compose", "source_id": "display_a", "target_id": "compose", "label": "bind prior display", "kind": "binding", "priority": 0},
                    {"id": "display-b-compose", "source_id": "display_b", "target_id": "compose", "label": "", "kind": "standard", "priority": 100},
                    {"id": "compose-finish", "source_id": "compose", "target_id": "finish", "label": "", "kind": "standard", "priority": 100},
                ],
            }
        )
        graph.validate_against_services(services)

        state = runtime.run(graph, "current request", run_id="run-display-message-context-builder")

        self.assertEqual(state.status, "completed")
        self.assertEqual(
            state.final_output,
            [
                {"role": "assistant", "content": "A direct reply is enough."},
                {"role": "assistant", "content": "A direct reply is enough."},
            ],
        )

    def test_context_builder_includes_newly_connected_sources_when_some_bindings_are_configured(self) -> None:
        services = build_example_services()
        services.model_providers["auto_message"] = AutoMessageProvider()
        runtime = GraphRuntime(
            services=services,
            max_steps=services.config["max_steps"],
            max_visits_per_node=services.config["max_visits_per_node"],
        )
        graph = GraphDefinition.from_dict(
            {
                "graph_id": "context-builder-partial-bindings",
                "name": "Context Builder Partial Bindings",
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
                        "id": "model_a",
                        "kind": "model",
                        "category": "api",
                        "label": "Model A",
                        "provider_id": "core.api",
                        "provider_label": "API Call Node",
                        "model_provider_name": "auto_message",
                        "prompt_name": "auto_message_a",
                        "config": {
                            "provider_name": "auto_message",
                            "prompt_name": "auto_message_a",
                            "mode": "auto_message_a",
                            "system_prompt": "Return a direct reply.",
                            "user_message_template": "{input_payload}",
                            "response_mode": "message",
                        },
                        "position": {"x": 180, "y": 0},
                    },
                    {
                        "id": "model_b",
                        "kind": "model",
                        "category": "api",
                        "label": "Model B",
                        "provider_id": "core.api",
                        "provider_label": "API Call Node",
                        "model_provider_name": "auto_message",
                        "prompt_name": "auto_message_b",
                        "config": {
                            "provider_name": "auto_message",
                            "prompt_name": "auto_message_b",
                            "mode": "auto_message_b",
                            "system_prompt": "Return a direct reply.",
                            "user_message_template": "{input_payload}",
                            "response_mode": "message",
                        },
                        "position": {"x": 180, "y": 140},
                    },
                    {
                        "id": "display_a",
                        "kind": "data",
                        "category": "data",
                        "label": "Display A",
                        "provider_id": "core.data_display",
                        "provider_label": "Envelope Display Node",
                        "config": {
                            "mode": "passthrough",
                            "show_input_envelope": True,
                            "lock_passthrough": True,
                        },
                        "position": {"x": 360, "y": 0},
                    },
                    {
                        "id": "display_b",
                        "kind": "data",
                        "category": "data",
                        "label": "Display B",
                        "provider_id": "core.data_display",
                        "provider_label": "Envelope Display Node",
                        "config": {
                            "mode": "passthrough",
                            "show_input_envelope": True,
                            "lock_passthrough": True,
                        },
                        "position": {"x": 360, "y": 140},
                    },
                    {
                        "id": "compose",
                        "kind": "data",
                        "category": "data",
                        "label": "Compose",
                        "provider_id": "core.context_builder",
                        "provider_label": "Context Builder",
                        "config": {
                            "mode": "context_builder",
                            "template": "",
                            "joiner": "\n\n",
                            "input_bindings": [
                                {
                                    "source_node_id": "display_a",
                                    "placeholder": "display_a",
                                    "binding": {"type": "latest_payload", "source": "display_a"},
                                }
                            ],
                        },
                        "position": {"x": 560, "y": 70},
                    },
                    {
                        "id": "finish",
                        "kind": "output",
                        "category": "end",
                        "label": "Finish",
                        "provider_id": "core.output",
                        "provider_label": "Core Output Node",
                        "config": {"source_binding": {"type": "latest_payload", "source": "compose"}},
                        "position": {"x": 760, "y": 70},
                    },
                ],
                "edges": [
                    {"id": "start-model-a", "source_id": "start", "target_id": "model_a", "label": "", "kind": "standard", "priority": 100},
                    {"id": "model-a-display", "source_id": "model_a", "target_id": "display_a", "label": "", "kind": "standard", "priority": 100},
                    {"id": "display-a-model-b", "source_id": "display_a", "target_id": "model_b", "label": "", "kind": "standard", "priority": 100},
                    {"id": "model-b-display", "source_id": "model_b", "target_id": "display_b", "label": "", "kind": "standard", "priority": 100},
                    {"id": "display-a-compose", "source_id": "display_a", "target_id": "compose", "label": "bind prior display", "kind": "binding", "priority": 0},
                    {"id": "display-b-compose", "source_id": "display_b", "target_id": "compose", "label": "", "kind": "standard", "priority": 100},
                    {"id": "compose-finish", "source_id": "compose", "target_id": "finish", "label": "", "kind": "standard", "priority": 100},
                ],
            }
        )
        graph.validate_against_services(services)

        state = runtime.run(graph, "current request", run_id="run-context-builder-partial-bindings")

        self.assertEqual(state.status, "completed")
        self.assertEqual(
            state.final_output,
            [
                {"role": "assistant", "content": "A direct reply is enough."},
                {"role": "assistant", "content": "A direct reply is enough."},
            ],
        )
        # display_a→compose is a binding edge: it schedules an early compose pass (partial
        # merge, hold_outgoing). display_b→compose (standard) schedules the final pass.
        compose_events = [
            e
            for e in state.event_history
            if e.event_type == "node.completed" and e.payload.get("node_id") == "compose"
        ]
        self.assertEqual(len(compose_events), 2)
        self.assertGreaterEqual(state.visit_counts.get("compose", 0), 2)
        first_out = compose_events[0].payload.get("output")
        self.assertIsInstance(first_out, dict)
        assert isinstance(first_out, dict)
        self.assertEqual(first_out.get("metadata", {}).get("binding_count"), 2)
        self.assertFalse(first_out.get("metadata", {}).get("context_builder_complete", True))
        self.assertTrue(compose_events[0].payload.get("metadata", {}).get("hold_outgoing_edges"))
        last_out = compose_events[-1].payload.get("output")
        self.assertIsInstance(last_out, dict)
        assert isinstance(last_out, dict)
        self.assertTrue(last_out.get("metadata", {}).get("context_builder_complete", False))
        self.assertFalse(bool(compose_events[-1].payload.get("metadata", {}).get("hold_outgoing_edges")))

    def test_context_builder_is_ready_when_any_connected_source_has_output(self) -> None:
        services = build_example_services()
        graph = GraphDefinition.from_dict(
            {
                "graph_id": "context-builder-ready-check",
                "name": "Context Builder Ready Check",
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
                        "id": "display_a",
                        "kind": "data",
                        "category": "data",
                        "label": "Display A",
                        "provider_id": "core.data_display",
                        "provider_label": "Envelope Display Node",
                        "config": {
                            "mode": "passthrough",
                            "show_input_envelope": True,
                            "lock_passthrough": True,
                        },
                        "position": {"x": 160, "y": 0},
                    },
                    {
                        "id": "display_b",
                        "kind": "data",
                        "category": "data",
                        "label": "Display B",
                        "provider_id": "core.data_display",
                        "provider_label": "Envelope Display Node",
                        "config": {
                            "mode": "passthrough",
                            "show_input_envelope": True,
                            "lock_passthrough": True,
                        },
                        "position": {"x": 160, "y": 140},
                    },
                    {
                        "id": "compose",
                        "kind": "data",
                        "category": "data",
                        "label": "Compose",
                        "provider_id": "core.context_builder",
                        "provider_label": "Context Builder",
                        "config": {
                            "mode": "context_builder",
                            "template": "",
                            "joiner": "\n\n",
                            "input_bindings": [
                                {"source_node_id": "display_a", "placeholder": "display_a", "binding": {"type": "latest_payload", "source": "display_a"}},
                                {"source_node_id": "display_b", "placeholder": "display_b", "binding": {"type": "latest_payload", "source": "display_b"}},
                            ],
                        },
                        "position": {"x": 360, "y": 70},
                    },
                    {
                        "id": "finish",
                        "kind": "output",
                        "category": "end",
                        "label": "Finish",
                        "provider_id": "core.output",
                        "provider_label": "Core Output Node",
                        "config": {"source_binding": {"type": "latest_payload", "source": "compose"}},
                        "position": {"x": 560, "y": 70},
                    },
                ],
                "edges": [
                    {"id": "start-display-a", "source_id": "start", "target_id": "display_a", "label": "", "kind": "standard", "priority": 100},
                    {"id": "display-a-compose", "source_id": "display_a", "target_id": "compose", "label": "", "kind": "standard", "priority": 100},
                    {"id": "display-b-compose", "source_id": "display_b", "target_id": "compose", "label": "bind delayed input", "kind": "binding", "priority": 0},
                ],
            }
        )
        graph.validate_against_services(services)

        compose_node = graph.get_node("compose")
        partial_state = RunState(graph_id=graph.graph_id, input_payload="request")
        partial_state.node_outputs["display_a"] = {
            "schema_version": "1.0",
            "from_node_id": "display_a",
            "from_category": "data",
            "payload": "first",
            "artifacts": {},
            "errors": [],
            "tool_calls": [],
            "metadata": {"contract": "message_envelope", "node_kind": "data"},
        }
        partial_context = NodeContext(graph=graph, state=partial_state, services=services, node_id="compose")
        self.assertTrue(compose_node.is_ready(partial_context))

        partial_state.node_outputs["display_b"] = {
            "schema_version": "1.0",
            "from_node_id": "display_b",
            "from_category": "data",
            "payload": "second",
            "artifacts": {},
            "errors": [],
            "tool_calls": [],
            "metadata": {"contract": "message_envelope", "node_kind": "data"},
        }
        complete_context = NodeContext(graph=graph, state=partial_state, services=services, node_id="compose")
        self.assertTrue(compose_node.is_ready(complete_context))

    def test_context_builder_accepts_multiple_binding_only_display_node_inputs(self) -> None:
        services = build_example_services()
        services.model_providers["auto_message"] = AutoMessageProvider()
        runtime = GraphRuntime(
            services=services,
            max_steps=services.config["max_steps"],
            max_visits_per_node=services.config["max_visits_per_node"],
        )
        graph = GraphDefinition.from_dict(
            {
                "graph_id": "binding-only-display-context-builder",
                "name": "Binding Only Display Context Builder",
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
                        "id": "model_a",
                        "kind": "model",
                        "category": "api",
                        "label": "Model A",
                        "provider_id": "core.api",
                        "provider_label": "API Call Node",
                        "model_provider_name": "auto_message",
                        "prompt_name": "auto_message_a",
                        "config": {
                            "provider_name": "auto_message",
                            "prompt_name": "auto_message_a",
                            "mode": "auto_message_a",
                            "system_prompt": "Return a direct reply.",
                            "user_message_template": "{input_payload}",
                            "response_mode": "message",
                        },
                        "position": {"x": 160, "y": 0},
                    },
                    {
                        "id": "model_b",
                        "kind": "model",
                        "category": "api",
                        "label": "Model B",
                        "provider_id": "core.api",
                        "provider_label": "API Call Node",
                        "model_provider_name": "auto_message",
                        "prompt_name": "auto_message_b",
                        "config": {
                            "provider_name": "auto_message",
                            "prompt_name": "auto_message_b",
                            "mode": "auto_message_b",
                            "system_prompt": "Return a direct reply.",
                            "user_message_template": "{input_payload}",
                            "response_mode": "message",
                        },
                        "position": {"x": 340, "y": 0},
                    },
                    {
                        "id": "display_a",
                        "kind": "data",
                        "category": "data",
                        "label": "Display A",
                        "provider_id": "core.data_display",
                        "provider_label": "Envelope Display Node",
                        "config": {
                            "mode": "passthrough",
                            "show_input_envelope": True,
                            "lock_passthrough": True,
                        },
                        "position": {"x": 160, "y": 140},
                    },
                    {
                        "id": "display_b",
                        "kind": "data",
                        "category": "data",
                        "label": "Display B",
                        "provider_id": "core.data_display",
                        "provider_label": "Envelope Display Node",
                        "config": {
                            "mode": "passthrough",
                            "show_input_envelope": True,
                            "lock_passthrough": True,
                        },
                        "position": {"x": 340, "y": 140},
                    },
                    {
                        "id": "compose",
                        "kind": "data",
                        "category": "data",
                        "label": "Compose",
                        "provider_id": "core.context_builder",
                        "provider_label": "Context Builder",
                        "config": {
                            "mode": "context_builder",
                            "template": "",
                            "joiner": "\n\n",
                            "input_bindings": [
                                {"source_node_id": "display_a", "placeholder": "display_a", "binding": {"type": "latest_payload", "source": "display_a"}},
                                {"source_node_id": "display_b", "placeholder": "display_b", "binding": {"type": "latest_payload", "source": "display_b"}},
                            ],
                        },
                        "position": {"x": 520, "y": 0},
                    },
                    {
                        "id": "finish",
                        "kind": "output",
                        "category": "end",
                        "label": "Finish",
                        "provider_id": "core.output",
                        "provider_label": "Core Output Node",
                        "config": {"source_binding": {"type": "latest_payload", "source": "compose"}},
                        "position": {"x": 700, "y": 0},
                    },
                ],
                "edges": [
                    {"id": "start-model-a", "source_id": "start", "target_id": "model_a", "label": "", "kind": "standard", "priority": 100},
                    {"id": "model-a-model-b", "source_id": "model_a", "target_id": "model_b", "label": "", "kind": "standard", "priority": 100},
                    {"id": "model-b-compose", "source_id": "model_b", "target_id": "compose", "label": "", "kind": "standard", "priority": 100},
                    {"id": "model-a-display-a", "source_id": "model_a", "target_id": "display_a", "label": "display source", "kind": "binding", "priority": 0},
                    {"id": "model-b-display-b", "source_id": "model_b", "target_id": "display_b", "label": "display source", "kind": "binding", "priority": 0},
                    {"id": "display-a-compose", "source_id": "display_a", "target_id": "compose", "label": "display text", "kind": "binding", "priority": 0},
                    {"id": "display-b-compose", "source_id": "display_b", "target_id": "compose", "label": "display text", "kind": "binding", "priority": 0},
                    {"id": "compose-finish", "source_id": "compose", "target_id": "finish", "label": "", "kind": "standard", "priority": 100},
                ],
            }
        )
        graph.validate_against_services(services)

        state = runtime.run(graph, "current request", run_id="run-binding-only-display-context-builder")

        self.assertEqual(state.status, "completed")
        self.assertEqual(
            state.final_output,
            [
                {"role": "assistant", "content": "A direct reply is enough."},
                {"role": "assistant", "content": "A direct reply is enough."},
            ],
        )
        self.assertNotIn("display_a", state.node_outputs)
        self.assertNotIn("display_b", state.node_outputs)

    def test_model_node_consumes_bound_prompt_blocks_as_messages(self) -> None:
        services = build_example_services()
        provider = PromptBlockEchoProvider()
        services.model_providers["prompt_block_echo"] = provider
        runtime = GraphRuntime(
            services=services,
            max_steps=services.config["max_steps"],
            max_visits_per_node=services.config["max_visits_per_node"],
        )
        graph = GraphDefinition.from_dict(
            {
                "graph_id": "prompt-block-model",
                "name": "Prompt Block Model",
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
                        "id": "assistant_block",
                        "kind": "data",
                        "category": "data",
                        "label": "Assistant Block",
                        "provider_id": "core.prompt_block",
                        "provider_label": "Prompt Block",
                        "config": {"mode": "prompt_block", "role": "assistant", "content": "Previous answer draft."},
                        "position": {"x": 120, "y": 0},
                    },
                    {
                        "id": "user_block",
                        "kind": "data",
                        "category": "data",
                        "label": "User Block",
                        "provider_id": "core.prompt_block",
                        "provider_label": "Prompt Block",
                        "config": {"mode": "prompt_block", "role": "user", "content": "Bound user context."},
                        "position": {"x": 120, "y": 120},
                    },
                    {
                        "id": "model",
                        "kind": "model",
                        "category": "api",
                        "label": "Model",
                        "provider_id": "core.api",
                        "provider_label": "API Call Node",
                        "model_provider_name": "prompt_block_echo",
                        "prompt_name": "prompt_block_echo",
                        "config": {
                            "provider_name": "prompt_block_echo",
                            "prompt_name": "prompt_block_echo",
                            "system_prompt": "Base system prompt",
                            "user_message_template": "Live input: {input_payload}",
                            "response_mode": "message",
                        },
                        "position": {"x": 320, "y": 0},
                    },
                    {
                        "id": "finish",
                        "kind": "output",
                        "category": "end",
                        "label": "Finish",
                        "provider_id": "core.output",
                        "provider_label": "Core Output Node",
                        "config": {"source_binding": {"type": "latest_payload", "source": "model"}},
                        "position": {"x": 520, "y": 0},
                    },
                ],
                "edges": [
                    {"id": "start-model", "source_id": "start", "target_id": "model", "label": "", "kind": "standard", "priority": 100},
                    {"id": "assistant-model", "source_id": "assistant_block", "target_id": "model", "label": "prompt block", "kind": "binding", "priority": 0},
                    {"id": "user-model", "source_id": "user_block", "target_id": "model", "label": "prompt block", "kind": "binding", "priority": 0},
                    {"id": "model-finish", "source_id": "model", "target_id": "finish", "label": "", "kind": "standard", "priority": 100},
                ],
            }
        )
        graph.validate_against_services(services)

        state = runtime.run(graph, "current request", run_id="run-prompt-model")

        self.assertEqual(state.status, "completed")
        self.assertEqual(state.final_output, "ok")
        self.assertIsNotNone(provider.last_request)
        assert provider.last_request is not None
        self.assertEqual(
            [(message.role, message.content) for message in provider.last_request.messages],
            [
                ("system", "Base system prompt"),
                ("assistant", "Previous answer draft."),
                ("user", "Bound user context."),
                ("user", "Live input: current request"),
            ],
        )
        self.assertEqual(
            provider.last_request.metadata.get("prompt_blocks"),
            [
                {"kind": "prompt_block", "role": "assistant", "content": "Previous answer draft."},
                {"kind": "prompt_block", "role": "user", "content": "Bound user context."},
            ],
        )
        started_events = {
            str(event.payload.get("node_id")): event
            for event in state.event_history
            if event.event_type == "node.started"
        }
        self.assertEqual(started_events["start"].payload.get("received_input"), "current request")
        self.assertEqual(
            started_events["model"].payload.get("received_input"),
            {
                "messages": [
                    {"role": "system", "content": "Base system prompt"},
                    {"role": "assistant", "content": "Previous answer draft."},
                    {"role": "user", "content": "Bound user context."},
                    {"role": "user", "content": "Live input: current request"},
                ],
                "response_mode": "message",
            },
        )
        self.assertEqual(started_events["finish"].payload.get("received_input"), "ok")

    def test_model_node_consumes_context_builder_payload_as_user_message(self) -> None:
        services = build_example_services()
        provider = PromptBlockEchoProvider()
        services.model_providers["prompt_block_echo"] = provider
        runtime = GraphRuntime(
            services=services,
            max_steps=services.config["max_steps"],
            max_visits_per_node=services.config["max_visits_per_node"],
        )
        graph = GraphDefinition.from_dict(
            {
                "graph_id": "context-builder-model",
                "name": "Context Builder Model",
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
                        "id": "first_user_block",
                        "kind": "data",
                        "category": "data",
                        "label": "First Request",
                        "provider_id": "core.prompt_block",
                        "provider_label": "Prompt Block",
                        "config": {"mode": "prompt_block", "role": "user", "content": "Summarize the issue."},
                        "position": {"x": 120, "y": 0},
                    },
                    {
                        "id": "second_user_block",
                        "kind": "data",
                        "category": "data",
                        "label": "Second Request",
                        "provider_id": "core.prompt_block",
                        "provider_label": "Prompt Block",
                        "config": {"mode": "prompt_block", "role": "user", "content": "Suggest the next debugging step."},
                        "position": {"x": 120, "y": 120},
                    },
                    {
                        "id": "compose",
                        "kind": "data",
                        "category": "data",
                        "label": "Compose",
                        "provider_id": "core.context_builder",
                        "provider_label": "Context Builder",
                        "config": {"mode": "context_builder", "template": "", "input_bindings": [], "joiner": "\n\n"},
                        "position": {"x": 320, "y": 0},
                    },
                    {
                        "id": "model",
                        "kind": "model",
                        "category": "api",
                        "label": "Model",
                        "provider_id": "core.api",
                        "provider_label": "API Call Node",
                        "model_provider_name": "prompt_block_echo",
                        "prompt_name": "prompt_block_echo",
                        "config": {
                            "provider_name": "prompt_block_echo",
                            "prompt_name": "prompt_block_echo",
                            "system_prompt": "Base system prompt",
                            "user_message_template": "{input_payload}",
                            "response_mode": "message",
                        },
                        "position": {"x": 520, "y": 0},
                    },
                    {
                        "id": "finish",
                        "kind": "output",
                        "category": "end",
                        "label": "Finish",
                        "provider_id": "core.output",
                        "provider_label": "Core Output Node",
                        "config": {"source_binding": {"type": "latest_payload", "source": "model"}},
                        "position": {"x": 720, "y": 0},
                    },
                ],
                "edges": [
                    {"id": "start-compose", "source_id": "start", "target_id": "compose", "label": "", "kind": "standard", "priority": 100},
                    {"id": "first-compose", "source_id": "first_user_block", "target_id": "compose", "label": "prompt block", "kind": "binding", "priority": 0},
                    {"id": "second-compose", "source_id": "second_user_block", "target_id": "compose", "label": "prompt block", "kind": "binding", "priority": 0},
                    {"id": "compose-model", "source_id": "compose", "target_id": "model", "label": "", "kind": "standard", "priority": 100},
                    {"id": "model-finish", "source_id": "model", "target_id": "finish", "label": "", "kind": "standard", "priority": 100},
                ],
            }
        )
        graph.validate_against_services(services)

        state = runtime.run(graph, "unused input", run_id="run-context-builder-model")

        self.assertEqual(state.status, "completed")
        self.assertEqual(state.final_output, "ok")
        self.assertIsNotNone(provider.last_request)
        assert provider.last_request is not None
        self.assertEqual(
            [message.role for message in provider.last_request.messages],
            ["system", "user"],
        )
        self.assertEqual(provider.last_request.messages[0].content, "Base system prompt")
        self.assertEqual(
            json.loads(provider.last_request.messages[1].content),
            [
                {"role": "user", "content": "unused input"},
                {"role": "user", "content": "Summarize the issue."},
                {"role": "user", "content": "Suggest the next debugging step."},
            ],
        )

    def test_mcp_executor_packages_follow_up_context_for_model(self) -> None:
        services = build_example_services()
        followup_provider = UserMessageJsonEchoProvider()
        services.model_providers["auto_tool_call"] = AutoToolCallProvider()
        services.model_providers["followup_echo"] = followup_provider
        runtime = GraphRuntime(
            services=services,
            max_steps=services.config["max_steps"],
            max_visits_per_node=services.config["max_visits_per_node"],
        )
        graph_payload = self._build_mcp_followup_executor_graph(
            initial_provider_name="auto_tool_call",
            followup_provider_name="followup_echo",
            tool_name="weather_current",
        )
        graph = GraphDefinition.from_dict(graph_payload)
        manager = GraphRunManager(services=services)
        with WeatherStubServer() as weather_base:
            try:
                with patch.dict("os.environ", {"GRAPH_AGENT_WEATHER_API_BASE": weather_base}, clear=False):
                    manager.boot_mcp_server("weather_mcp")
                    manager.set_mcp_tool_enabled("weather_current", True)
                    graph.validate_against_services(services)

                    state = runtime.run(graph, {"request": "weather please"}, run_id="run-mcp-followup-context")

                    self.assertEqual(state.status, "completed")
                    payload = state.final_output
                    assert isinstance(payload, dict)
                    self.assertFalse(payload["should_call_tool"])
                    self.assertEqual(payload["tool_calls"], [])
                    self.assertEqual(len(payload["tool_payloads"]), 1)
                    self.assertEqual(payload["tool_payloads"][0]["tool_name"], "weather_current")
                    self.assertEqual(payload["tool_payloads"][0]["tool_arguments"], {"location": "Austin"})
                    self.assertEqual(payload["tool_payloads"][0]["tool_output"]["resolved_location"], "Testville")
                    self.assertEqual(payload["tool_payloads"][0]["tool_status"], "success")
                    self.assertNotIn("terminal_output", payload["tool_payloads"][0])
                    executor_output = state.node_outputs["executor"]
                    assert isinstance(executor_output, dict)
                    self.assertEqual(executor_output["metadata"]["contract"], "tool_result_envelope")
                    self.assertEqual(executor_output["payload"]["tool_payloads"][0]["tool_name"], "weather_current")
                    self.assertEqual(
                        executor_output["artifacts"]["source_tool_call_envelope"]["metadata"]["contract"],
                        "tool_call_envelope",
                    )
                    self.assertEqual(
                        executor_output["artifacts"]["follow_up_payload"]["original_input_payload"],
                        {"request": "weather please"},
                    )
                    self.assertEqual(
                        executor_output["artifacts"]["follow_up_payload"]["tool_history"][0]["tool_arguments"],
                        {"location": "Austin"},
                    )
            finally:
                manager.stop_background_services()

    def test_mcp_executor_follow_up_can_loop_to_another_tool(self) -> None:
        services = build_example_services()
        followup_provider = RecheckLoopProvider()
        services.model_providers["auto_tool_call"] = AutoToolCallProvider()
        services.model_providers["recheck_loop"] = followup_provider
        runtime = GraphRuntime(
            services=services,
            max_steps=services.config["max_steps"],
            max_visits_per_node=services.config["max_visits_per_node"],
        )
        graph_payload = self._build_mcp_followup_executor_graph(
            initial_provider_name="auto_tool_call",
            followup_provider_name="recheck_loop",
            tool_name="weather_current",
        )
        graph = GraphDefinition.from_dict(graph_payload)
        manager = GraphRunManager(services=services)
        with WeatherStubServer() as weather_base:
            try:
                with patch.dict("os.environ", {"GRAPH_AGENT_WEATHER_API_BASE": weather_base}, clear=False):
                    manager.boot_mcp_server("weather_mcp")
                    manager.set_mcp_tool_enabled("weather_current", True)
                    graph.validate_against_services(services)

                    state = runtime.run(graph, {"request": "weather please"}, run_id="run-mcp-followup-context")

                    self.assertEqual(state.status, "completed")
                    assert isinstance(state.final_output, dict)
                    self.assertFalse(state.final_output["should_call_tool"])
                    self.assertEqual(state.final_output["tool_calls"], [])
                    self.assertEqual(
                        [entry["tool_arguments"] for entry in state.final_output["tool_payloads"]],
                        [{"location": "Austin"}, {"location": "Seattle"}],
                    )
                    self.assertEqual(state.visit_counts["executor"], 1)
                    self.assertEqual(len(followup_provider.seen_payloads), 2)
                    self.assertEqual(followup_provider.seen_payloads[0]["original_input_payload"], {"request": "weather please"})
                    self.assertEqual(followup_provider.seen_payloads[1]["original_input_payload"], {"request": "weather please"})
                    self.assertEqual(followup_provider.seen_payloads[0]["tool_arguments"], {"location": "Austin"})
                    self.assertEqual(followup_provider.seen_payloads[1]["tool_arguments"], {"location": "Seattle"})
                    self.assertEqual(len(followup_provider.seen_payloads[0]["tool_history"]), 1)
                    self.assertEqual(len(followup_provider.seen_payloads[1]["tool_history"]), 2)
                    executor_output = state.node_outputs["executor"]
                    assert isinstance(executor_output, dict)
                    self.assertEqual(executor_output["metadata"]["contract"], "tool_result_envelope")
                    follow_up_payload = executor_output["artifacts"]["follow_up_payload"]
                    assert isinstance(follow_up_payload, dict)
                    self.assertEqual(
                        [entry["tool_arguments"] for entry in follow_up_payload["tool_history"]],
                        [{"location": "Austin"}, {"location": "Seattle"}],
                    )
            finally:
                manager.stop_background_services()

    def test_mcp_executor_executes_all_requested_initial_tool_calls_before_follow_up(self) -> None:
        services = build_example_services()
        followup_provider = UserMessageJsonEchoProvider()
        services.model_providers["auto_two_tool_calls"] = AutoTwoToolCallsProvider()
        services.model_providers["followup_echo"] = followup_provider
        runtime = GraphRuntime(
            services=services,
            max_steps=services.config["max_steps"],
            max_visits_per_node=services.config["max_visits_per_node"],
        )
        graph_payload = self._build_mcp_followup_executor_graph(
            initial_provider_name="auto_two_tool_calls",
            followup_provider_name="followup_echo",
            tool_name="weather_current",
            allowed_tool_names=["weather_current", "time_current_minute"],
        )
        graph = GraphDefinition.from_dict(graph_payload)
        manager = GraphRunManager(services=services)
        with WeatherStubServer() as weather_base:
            try:
                with patch.dict("os.environ", {"GRAPH_AGENT_WEATHER_API_BASE": weather_base}, clear=False):
                    manager.boot_mcp_server("weather_mcp")
                    manager.boot_mcp_server("time_mcp")
                    manager.set_mcp_tool_enabled("weather_current", True)
                    manager.set_mcp_tool_enabled("time_current_minute", True)
                    graph.validate_against_services(services)

                    state = runtime.run(graph, {"request": "weather and time please"}, run_id="run-mcp-multi-tool-followup")

                    self.assertEqual(state.status, "completed")
                    assert isinstance(state.final_output, dict)
                    self.assertFalse(state.final_output["should_call_tool"])
                    self.assertEqual(state.final_output["tool_calls"], [])
                    self.assertEqual(
                        [entry["tool_name"] for entry in state.final_output["tool_payloads"]],
                        ["weather_current", "time_current_minute"],
                    )
                    self.assertEqual(state.final_output["tool_payloads"][0]["tool_arguments"], {"location": "Austin"})
                    self.assertIn("local_iso_minute", state.final_output["tool_payloads"][1]["tool_output"])

                    executor_output = state.node_outputs["executor"]
                    assert isinstance(executor_output, dict)
                    follow_up_payload = executor_output["artifacts"]["follow_up_payload"]
                    assert isinstance(follow_up_payload, dict)
                    self.assertEqual(
                        [entry["tool_name"] for entry in follow_up_payload["tool_history"]],
                        ["weather_current", "time_current_minute"],
                    )
                    self.assertEqual(follow_up_payload["pending_tool_calls"], [])

                    assert followup_provider.last_request is not None
                    echoed_payload = json.loads(followup_provider.last_request.messages[-1].content)
                    self.assertEqual(
                        [entry["tool_name"] for entry in echoed_payload["tool_history"]],
                        ["weather_current", "time_current_minute"],
                    )
            finally:
                manager.stop_background_services()

    def test_mcp_executor_follow_up_preserves_terminal_output(self) -> None:
        services = build_example_services()

        def _mock_mcp_tool(_payload: Mapping[str, Any], _context: ToolContext) -> ToolResult:
            return ToolResult(
                status="success",
                output={"ok": True},
                metadata={
                    "terminal_output": {
                        "server_id": "mock_mcp",
                        "tool_name": "mock_terminal_tool",
                        "stderr": "first line\nsecond line",
                        "stderr_lines": ["first line", "second line"],
                    }
                },
            )

        class TerminalToolCallProvider:
            name = "terminal_mock_followup"

            def generate(self, request: ModelRequest) -> ModelResponse:
                self.last_request = request
                return ModelResponse(
                    content="",
                    structured_output=_decision(tool_calls=[{"tool_name": "mock_terminal_tool", "arguments": {}}]),
                    tool_calls=[ModelToolCall(tool_name="mock_terminal_tool", arguments={})],
                )

            def preflight(self, provider_config: Mapping[str, Any] | None = None) -> ProviderPreflightResult:
                return ProviderPreflightResult(status="ready", ok=True)

        followup_provider = UserMessageJsonEchoProvider()
        services.model_providers["terminal_mock_followup"] = TerminalToolCallProvider()
        services.model_providers["followup_echo"] = followup_provider
        services.tool_registry.upsert(
            ToolDefinition(
                name="mock_terminal_tool",
                description="Mock MCP tool with terminal output.",
                input_schema={"type": "object", "properties": {}, "required": []},
                executor=_mock_mcp_tool,
                source_type="mcp",
                server_id="mock_mcp",
            )
        )
        runtime = GraphRuntime(
            services=services,
            max_steps=services.config["max_steps"],
            max_visits_per_node=services.config["max_visits_per_node"],
        )
        graph_payload = self._build_mcp_followup_executor_graph(
            initial_provider_name="terminal_mock_followup",
            followup_provider_name="followup_echo",
            tool_name="mock_terminal_tool",
        )
        graph = GraphDefinition.from_dict(graph_payload)
        graph.validate_against_services(services)

        state = runtime.run(graph, {"request": "run"}, run_id="run-mcp-followup-terminal")

        self.assertEqual(state.status, "completed")
        payload = state.final_output
        assert isinstance(payload, dict)
        self.assertFalse(payload["should_call_tool"])
        self.assertEqual(payload["tool_payloads"][0]["tool_name"], "mock_terminal_tool")
        self.assertEqual(payload["tool_payloads"][0]["terminal_output"]["stderr"], "first line\nsecond line")
        executor_output = state.node_outputs["executor"]
        assert isinstance(executor_output, dict)
        self.assertEqual(executor_output["artifacts"]["terminal_output_envelope"]["metadata"]["contract"], "terminal_output_envelope")
        self.assertEqual(executor_output["artifacts"]["follow_up_payload"]["terminal_output"]["tool_name"], "mock_terminal_tool")

    def test_mcp_executor_follow_up_stops_after_failed_tool_result(self) -> None:
        services = build_example_services()

        class FailedToolCallProvider(ModelProvider):
            name = "failed_tool_call"

            def generate(self, request: ModelRequest) -> ModelResponse:
                return ModelResponse(
                    content="",
                    structured_output=_decision(tool_calls=[{"tool_name": "mock_failure_tool", "arguments": {}}]),
                    tool_calls=[ModelToolCall(tool_name="mock_failure_tool", arguments={})],
                )

            def preflight(self, provider_config: Mapping[str, Any] | None = None) -> ProviderPreflightResult:
                return ProviderPreflightResult(status="ready", ok=True)

        def _failing_mcp_tool(_payload: Mapping[str, Any], _context: ToolContext) -> ToolResult:
            return ToolResult(
                status="failed",
                output={"ok": False},
                error={"type": "mock_failure", "message": "simulated MCP failure"},
            )

        followup_provider = RecheckLoopProvider()
        services.model_providers["failed_tool_call"] = FailedToolCallProvider()
        services.model_providers["recheck_loop"] = followup_provider
        services.tool_registry.upsert(
            ToolDefinition(
                name="mock_failure_tool",
                description="Mock MCP tool that fails.",
                input_schema={"type": "object", "properties": {}, "required": []},
                executor=_failing_mcp_tool,
                source_type="mcp",
                server_id="mock_mcp",
            )
        )
        runtime = GraphRuntime(
            services=services,
            max_steps=services.config["max_steps"],
            max_visits_per_node=services.config["max_visits_per_node"],
        )
        graph_payload = self._build_mcp_followup_executor_graph(
            initial_provider_name="failed_tool_call",
            followup_provider_name="recheck_loop",
            tool_name="mock_failure_tool",
        )
        finish_node = next(node for node in graph_payload["nodes"] if node["id"] == "finish")
        finish_node["config"] = {"source_binding": {"type": "latest_payload", "source": "executor"}}
        graph_payload["nodes"].append(
            {
                "id": "failure_finish",
                "kind": "output",
                "category": "end",
                "label": "Failure Finish",
                "provider_id": "core.output",
                "provider_label": "Core Output Node",
                "config": {"source_binding": {"type": "latest_payload", "source": "executor"}},
                "position": {"x": 760, "y": 120},
            }
        )
        graph_payload["edges"].append(
            {
                "id": "executor-failure-finish",
                "source_id": "executor",
                "target_id": "failure_finish",
                "source_handle_id": "tool-failure",
                "label": "on failure",
                "kind": "conditional",
                "priority": 10,
                "condition": {
                    "id": "executor-failure-finish-condition",
                    "label": "On failure",
                    "type": "result_has_error",
                    "value": True,
                },
            }
        )
        graph = GraphDefinition.from_dict(graph_payload)
        graph.validate_against_services(services)

        state = runtime.run(graph, {"request": "run"}, run_id="run-mcp-followup-failed-tool")

        self.assertEqual(state.status, "completed")
        self.assertEqual(state.visit_counts["executor"], 1)
        self.assertEqual(followup_provider.seen_payloads, [])
        payload = state.final_output
        assert isinstance(payload, dict)
        self.assertEqual(payload, {"ok": False})
        executor_output = state.node_outputs["executor"]
        assert isinstance(executor_output, dict)
        self.assertEqual(executor_output["metadata"]["contract"], "tool_result_envelope")
        self.assertEqual(executor_output["metadata"]["tool_status"], "failed")
        self.assertEqual(
            executor_output["artifacts"]["validation_message"],
            "Skipping further MCP tool checks because 'mock_failure_tool' did not complete successfully.",
        )
        transition_edges = [transition.edge_id for transition in state.transition_history]
        self.assertIn("executor-failure-finish", transition_edges)

    def test_removed_mcp_recheck_kind_is_rejected(self) -> None:
        services = build_example_services()
        graph_payload = {
            "graph_id": "invalid-recheck-upstream",
            "name": "Invalid Recheck Upstream",
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
                    "id": "recheck",
                    "kind": "mcp_recheck",
                    "category": "tool",
                    "label": "Recheck",
                    "provider_id": "tool.mcp_recheck",
                    "provider_label": "MCP Recheck Node",
                    "config": {},
                    "position": {"x": 200, "y": 0},
                },
                {
                    "id": "finish",
                    "kind": "output",
                    "category": "end",
                    "label": "Finish",
                    "provider_id": "core.output",
                    "provider_label": "Core Output Node",
                    "config": {"source_binding": {"type": "latest_payload", "source": "recheck"}},
                    "position": {"x": 400, "y": 0},
                },
            ],
            "edges": [
                {"id": "start-recheck", "source_id": "start", "target_id": "recheck", "label": "", "kind": "standard", "priority": 100},
                {"id": "recheck-finish", "source_id": "recheck", "target_id": "finish", "label": "", "kind": "standard", "priority": 100},
            ],
        }

        with self.assertRaises(GraphValidationError):
            GraphDefinition.from_dict(graph_payload).validate_against_services(services)

    def test_mcp_executor_routes_terminal_output_handle(self) -> None:
        services = build_example_services()

        def _mock_mcp_tool(_payload: Mapping[str, Any], _context: ToolContext) -> ToolResult:
            return ToolResult(
                status="success",
                output={"ok": True},
                metadata={
                    "terminal_output": {
                        "server_id": "mock_mcp",
                        "tool_name": "mock_terminal_tool",
                        "stderr": "first line\nsecond line",
                        "stderr_lines": ["first line", "second line"],
                    }
                },
            )

        class TerminalToolCallProvider:
            name = "terminal_mock"

            def generate(self, request: ModelRequest) -> ModelResponse:
                self.last_request = request
                return ModelResponse(
                    content="",
                    structured_output=_decision(tool_calls=[{"tool_name": "mock_terminal_tool", "arguments": {}}]),
                    tool_calls=[ModelToolCall(tool_name="mock_terminal_tool", arguments={})],
                )

            def preflight(self, provider_config: Mapping[str, Any] | None = None) -> ProviderPreflightResult:
                return ProviderPreflightResult(status="ready", ok=True)

        services.model_providers["terminal_mock"] = TerminalToolCallProvider()
        services.tool_registry.upsert(
            ToolDefinition(
                name="mock_terminal_tool",
                description="Mock MCP tool with terminal output.",
                input_schema={"type": "object", "properties": {}, "required": []},
                executor=_mock_mcp_tool,
                source_type="mcp",
                server_id="mock_mcp",
            )
        )

        graph_payload = {
            "graph_id": "mcp-executor-terminal-output",
            "name": "MCP Executor Terminal Output",
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
                    "id": "terminal_context",
                    "kind": "mcp_context_provider",
                    "category": "tool",
                    "label": "Terminal Context",
                    "provider_id": "tool.mcp_context_provider",
                    "provider_label": "MCP Context Provider",
                    "config": {"tool_names": ["mock_terminal_tool"], "include_mcp_tool_context": False},
                    "position": {"x": 120, "y": 0},
                },
                {
                    "id": "model",
                    "kind": "model",
                    "category": "api",
                    "label": "Model",
                    "provider_id": "core.api",
                    "provider_label": "API Call Node",
                    "model_provider_name": "terminal_mock",
                    "prompt_name": "terminal_prompt",
                    "config": {
                        "provider_name": "terminal_mock",
                        "prompt_name": "terminal_prompt",
                        "system_prompt": "Call the terminal MCP tool.",
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
                {
                    "id": "terminal_finish",
                    "kind": "output",
                    "category": "end",
                    "label": "Terminal Finish",
                    "provider_id": "core.output",
                    "provider_label": "Core Output Node",
                    "config": {"source_binding": {"type": "latest_payload", "source": "executor"}},
                    "position": {"x": 600, "y": 120},
                },
            ],
            "edges": [
                {"id": "start-model", "source_id": "start", "target_id": "model", "label": "", "kind": "standard", "priority": 100},
                {
                    "id": "ctx-binding",
                    "source_id": "terminal_context",
                    "target_id": "model",
                    "source_handle_id": "tool-context",
                    "target_handle_id": "api-tool-context",
                    "label": "tool context",
                    "kind": "binding",
                    "priority": 0,
                },
                {"id": "model-executor", "source_id": "model", "target_id": "executor", "label": "", "kind": "standard", "priority": 100},
                {"id": "executor-finish", "source_id": "executor", "target_id": "finish", "label": "", "kind": "standard", "priority": 100},
                {
                    "id": "executor-terminal",
                    "source_id": "executor",
                    "target_id": "terminal_finish",
                    "source_handle_id": "mcp-terminal-output",
                    "label": "terminal output",
                    "kind": "conditional",
                    "priority": 30,
                    "condition": {
                        "id": "executor-terminal-condition",
                        "label": "Terminal output",
                        "type": "result_payload_path_equals",
                        "value": "terminal_output_envelope",
                        "path": "metadata.contract",
                    },
                },
            ],
        }

        graph = GraphDefinition.from_dict(graph_payload)
        graph.validate_against_services(services)

        runtime = GraphRuntime(services=services, max_steps=10, max_visits_per_node=3)
        state = runtime.run(graph, {"request": "run"}, run_id="run-mcp-terminal-output")

        self.assertEqual(state.status, "completed")
        self.assertEqual(state.final_output, {"ok": True})
        terminal_output = state.edge_outputs["executor-terminal"]
        assert isinstance(terminal_output, dict)
        self.assertEqual(terminal_output["metadata"]["contract"], "terminal_output_envelope")
        self.assertEqual(terminal_output["payload"]["stderr"], "first line\nsecond line")
        self.assertEqual(terminal_output["payload"]["stderr_lines"], ["first line", "second line"])

    def test_mcp_executor_validation_rejects_terminal_routes_without_terminal_condition(self) -> None:
        services = build_example_services()
        services.tool_registry.upsert(
            ToolDefinition(
                name="mock_terminal_tool",
                description="Mock MCP tool for validation.",
                input_schema={"type": "object", "properties": {}, "required": []},
                source_type="mcp",
                server_id="mock_mcp",
            )
        )
        graph_payload = {
            "graph_id": "invalid-terminal-route",
            "name": "Invalid Terminal Route",
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
                    "id": "terminal_context",
                    "kind": "mcp_context_provider",
                    "category": "tool",
                    "label": "Terminal Context",
                    "provider_id": "tool.mcp_context_provider",
                    "provider_label": "MCP Context Provider",
                    "config": {"tool_names": ["mock_terminal_tool"], "include_mcp_tool_context": False},
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
                    "prompt_name": "terminal_validation",
                    "config": {
                        "provider_name": "mock",
                        "prompt_name": "terminal_validation",
                        "system_prompt": "Call the weather tool.",
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
                    "source_id": "terminal_context",
                    "target_id": "model",
                    "source_handle_id": "tool-context",
                    "target_handle_id": "api-tool-context",
                    "label": "tool context",
                    "kind": "binding",
                    "priority": 0,
                },
                {"id": "model-executor", "source_id": "model", "target_id": "executor", "label": "", "kind": "standard", "priority": 100},
                {
                    "id": "executor-terminal",
                    "source_id": "executor",
                    "target_id": "finish",
                    "source_handle_id": "mcp-terminal-output",
                    "label": "terminal output",
                    "kind": "standard",
                    "priority": 30,
                },
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

    def test_mcp_session_close_uses_shutdown_then_exit(self) -> None:
        definition = McpServerDefinition(
            server_id="weather_mcp",
            display_name="Weather MCP Server",
            description="Weather",
            command=["python", "-m", "weather"],
        )
        session = _McpStdioSession(definition)
        process = Mock()
        process.poll.return_value = None
        process.wait.return_value = 0
        process.stdin = Mock()
        process.stdout = Mock()
        process.stderr = Mock()
        session._process = process
        session.request = Mock(return_value={})
        session.notify = Mock()

        session.close()

        session.request.assert_called_once_with("shutdown", {})
        session.notify.assert_called_once_with("exit", {})
        process.terminate.assert_not_called()
        process.kill.assert_not_called()
        process.wait.assert_called_once_with(timeout=3)
        self.assertIsNone(session._process)

    def test_shutdown_all_preserves_desired_running_for_persistent_servers(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            state_path = Path(tmp_dir) / "mcp_servers_state.json"
            manager = McpServerManager(ToolRegistry(), state_path=state_path)
            definition = McpServerDefinition(
                server_id="weather_mcp",
                display_name="Weather MCP Server",
                description="Weather",
                command=["python", "-m", "weather"],
                persistent=True,
            )
            manager.register_server(definition)
            manager._set_desired_running("weather_mcp", True)

            manager.shutdown_all(preserve_desired_running=True)

            restarted = McpServerManager(ToolRegistry(), state_path=state_path)
            restarted.register_server(definition)
            with patch.object(restarted, "boot_server") as boot_mock:
                restarted.start_auto_boot()

            boot_mock.assert_called_once_with("weather_mcp")

    def test_http_mcp_servers_can_discover_and_invoke_tools(self) -> None:
        with TemporaryDirectory() as tmp_dir, HttpMcpStubServer() as base_url:
            state_path = Path(tmp_dir) / "mcp_servers_state.json"
            registry = ToolRegistry()
            manager = McpServerManager(registry, state_path=state_path)
            definition = McpServerDefinition(
                server_id="remote_http",
                display_name="Remote HTTP",
                description="HTTP MCP server",
                transport="http",
                base_url=base_url,
            )

            manager.create_server(definition)
            server_state = manager.boot_server("remote_http")

            self.assertEqual(server_state["transport"], "http")
            self.assertTrue(server_state["running"])
            self.assertIn("echo_http", server_state["tool_names"])
            manager.set_tool_enabled("echo_http", True)

            result = registry.invoke(
                "echo_http",
                {"message": "hello"},
                ToolContext(run_id="run-1", graph_id="graph-1", node_id="node-1", state_snapshot={}),
            )
            self.assertEqual(result.status, "success")
            self.assertEqual(result.output["arguments"], {"message": "hello"})
            self.assertEqual(_HttpMcpStubHandler.requests[0]["method"], "initialize")
            self.assertTrue(any(request["method"] == "tools/list" for request in _HttpMcpStubHandler.requests))

    def test_user_mcp_servers_persist_and_reload(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            state_path = Path(tmp_dir) / "mcp_servers_state.json"
            manager = McpServerManager(ToolRegistry(), state_path=state_path)
            definition = McpServerDefinition(
                server_id="remote_http",
                display_name="Remote HTTP",
                description="HTTP MCP server",
                transport="http",
                base_url="http://127.0.0.1:8123/mcp",
            )

            manager.create_server(definition)

            payload = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["servers"][0]["server_id"], "remote_http")

            restarted = McpServerManager(ToolRegistry(), state_path=state_path)
            restarted.load_user_servers()
            servers = restarted.list_servers()
            self.assertEqual(len(servers), 1)
            self.assertEqual(servers[0]["server_id"], "remote_http")
            self.assertEqual(servers[0]["source"], "user")

    def test_graph_run_manager_supports_mcp_server_crud(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            state_path = Path(tmp_dir) / "mcp_servers_state.json"
            services = build_example_services()
            services.mcp_server_manager = McpServerManager(services.tool_registry, state_path=state_path)
            manager = GraphRunManager(services=services)

            created = manager.create_mcp_server(
                {
                    "server_id": "remote_http",
                    "display_name": "Remote HTTP",
                    "description": "HTTP MCP server",
                    "transport": "http",
                    "base_url": "http://127.0.0.1:8123/mcp",
                    "timeout_seconds": 20,
                    "auto_boot": True,
                    "persistent": True,
                }
            )
            self.assertEqual(created["server_id"], "remote_http")
            self.assertTrue(any(server["server_id"] == "remote_http" for server in manager.get_catalog()["mcp_servers"]))

            updated = manager.update_mcp_server(
                "remote_http",
                {
                    "server_id": "remote_http",
                    "display_name": "Remote HTTP Updated",
                    "description": "Updated HTTP MCP server",
                    "transport": "http",
                    "base_url": "http://127.0.0.1:9000/mcp",
                    "timeout_seconds": 30,
                    "auto_boot": False,
                    "persistent": True,
                },
            )
            self.assertEqual(updated["display_name"], "Remote HTTP Updated")
            catalog = manager.get_catalog()
            catalog_server = next(server for server in catalog["mcp_servers"] if server["server_id"] == "remote_http")
            self.assertEqual(catalog_server["config"]["base_url"], "http://127.0.0.1:9000/mcp")

            manager.delete_mcp_server("remote_http")
            self.assertFalse(any(server["server_id"] == "remote_http" for server in manager.get_catalog()["mcp_servers"]))


if __name__ == "__main__":
    unittest.main()
