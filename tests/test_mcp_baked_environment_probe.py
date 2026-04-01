from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from graph_agent.diagnostics import run_baked_environment_probe
from graph_agent.examples.tool_schema_repair import build_example_services
from graph_agent.providers.base import ModelProvider, ModelRequest, ModelResponse, ModelToolCall, ProviderPreflightResult


class TimeToolCallerProvider(ModelProvider):
    name = "time_tool_caller"

    def generate(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(
            content="Calling the time MCP tool.",
            tool_calls=[ModelToolCall(tool_name="time_current_minute", arguments={})],
            metadata={"mode": request.metadata.get("mode")},
        )

    def preflight(self, provider_config: Mapping[str, Any] | None = None) -> ProviderPreflightResult:
        return ProviderPreflightResult(status="available", ok=True, message="ok", details={})


class BakedMcpEnvironmentProbeTests(unittest.TestCase):
    def test_probe_reports_requested_and_executed_mcp_tool_calls(self) -> None:
        services = build_example_services()
        services.model_providers["time_tool_caller"] = TimeToolCallerProvider()
        graph_payload = {
            "graph_id": "time-mcp-probe",
            "name": "Time MCP Probe",
            "description": "Exercise MCP tool execution through a baked test environment.",
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
                    "label": "Time Context",
                    "provider_id": "tool.mcp_context_provider",
                    "provider_label": "MCP Context Provider",
                    "config": {
                        "tool_names": ["time_current_minute"],
                        "expose_mcp_tools": True,
                        "include_mcp_tool_context": False,
                    },
                    "position": {"x": 120, "y": 0},
                },
                {
                    "id": "model",
                    "kind": "model",
                    "category": "api",
                    "label": "Model",
                    "provider_id": "core.api",
                    "provider_label": "API Call Node",
                    "model_provider_name": "time_tool_caller",
                    "prompt_name": "call_time_tool",
                    "config": {
                        "provider_name": "time_tool_caller",
                        "prompt_name": "call_time_tool",
                        "system_prompt": "Call the MCP time tool.",
                        "user_message_template": "{input_payload}",
                        "response_mode": "tool_call",
                    },
                    "position": {"x": 280, "y": 0},
                },
                {
                    "id": "executor",
                    "kind": "mcp_tool_executor",
                    "category": "tool",
                    "label": "Execute MCP Tool",
                    "provider_id": "tool.mcp_tool_executor",
                    "provider_label": "MCP Tool Executor",
                    "config": {},
                    "position": {"x": 480, "y": 0},
                },
                {
                    "id": "finish",
                    "kind": "output",
                    "category": "end",
                    "label": "Finish",
                    "provider_id": "core.output",
                    "provider_label": "Core Output Node",
                    "config": {"source_binding": {"type": "latest_envelope", "source": "executor"}},
                    "position": {"x": 680, "y": 0},
                },
            ],
            "edges": [
                {"id": "start-model", "source_id": "start", "target_id": "model", "label": "begin", "kind": "standard", "priority": 100, "condition": None},
                {"id": "context-model", "source_id": "tool_context", "target_id": "model", "label": "tools", "kind": "binding", "priority": 100, "condition": None},
                {"id": "model-executor", "source_id": "model", "target_id": "executor", "label": "execute", "kind": "standard", "priority": 100, "condition": None},
                {"id": "executor-finish", "source_id": "executor", "target_id": "finish", "label": "done", "kind": "standard", "priority": 100, "condition": None},
            ],
        }

        summary = run_baked_environment_probe(graph_payload, input_payload="What time is it?", services=services)

        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["agent_status"], "completed")
        self.assertEqual(summary["node_errors"], {})
        self.assertEqual(summary["mcp_setup"]["configured_mcp_tools"], ["time_current_minute"])
        self.assertEqual(summary["mcp_setup"]["enabled_tools"], ["time_current_minute"])
        self.assertEqual(summary["mcp_activity"]["called_tools"], ["time_current_minute"])
        self.assertTrue(summary["mcp_activity"]["requested_tool_calls"])
        self.assertEqual(summary["mcp_activity"]["requested_tool_calls"][0]["tool_name"], "time_current_minute")
        self.assertTrue(summary["mcp_activity"]["executor_results"])
        self.assertEqual(summary["mcp_activity"]["executor_results"][0]["tool_name"], "time_current_minute")
        self.assertTrue(summary["mcp_activity"]["executor_results"][0]["has_requested_tool_call"])


if __name__ == "__main__":
    unittest.main()
