from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
import unittest
from typing import Any
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from graph_agent.api.manager import GraphRunManager
from graph_agent.examples.tool_schema_repair import build_example_services
from graph_agent.tools.base import ToolContext, ToolRegistry
from graph_agent.tools.mcp import (
    MCP_CAPABILITY_PROMPT,
    MCP_CAPABILITY_RESOURCE,
    MCP_CAPABILITY_RESOURCE_TEMPLATE,
    MCP_CAPABILITY_TOOL,
    McpServerDefinition,
    McpServerManager,
    canonical_mcp_capability_name,
    canonical_mcp_tool_name,
)


DISCOVERY_SERVER_ID = "discovery_http"
DISCOVERY_TOOL_ID = canonical_mcp_tool_name(DISCOVERY_SERVER_ID, "echo_tool")
DISCOVERY_RESOURCE_ID = canonical_mcp_capability_name(
    DISCOVERY_SERVER_ID,
    MCP_CAPABILITY_RESOURCE,
    "docs://guide",
)
DISCOVERY_RESOURCE_TEMPLATE_ID = canonical_mcp_capability_name(
    DISCOVERY_SERVER_ID,
    MCP_CAPABILITY_RESOURCE_TEMPLATE,
    "docs://{slug}",
)
DISCOVERY_PROMPT_ID = canonical_mcp_capability_name(
    DISCOVERY_SERVER_ID,
    MCP_CAPABILITY_PROMPT,
    "summarize_docs",
)


class _DiscoveryStubHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        content_length = int(self.headers.get("Content-Length", "0") or "0")
        payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
        method = payload.get("method")
        message_id = payload.get("id")

        if method == "initialize":
            result = {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {"listChanged": False},
                    "resources": {"templates": True},
                    "prompts": {},
                },
                "serverInfo": {"name": "discovery-stub", "version": "0.1.0"},
            }
        elif method == "tools/list":
            result = {
                "tools": [
                    {
                        "name": "echo_tool",
                        "description": "Echo a message back.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"message": {"type": "string"}},
                            "required": ["message"],
                        },
                    }
                ]
            }
        elif method == "resources/list":
            result = {
                "resources": [
                    {
                        "uri": "docs://guide",
                        "name": "guide",
                        "title": "Guide Resource",
                        "description": "Primary guide document.",
                        "mimeType": "text/markdown",
                    }
                ]
            }
        elif method == "resources/templates/list":
            result = {
                "resourceTemplates": [
                    {
                        "uriTemplate": "docs://{slug}",
                        "name": "docs_template",
                        "title": "Docs Template",
                        "description": "Template for document lookups.",
                    }
                ]
            }
        elif method == "prompts/list":
            result = {
                "prompts": [
                    {
                        "name": "summarize_docs",
                        "title": "Summarize Docs",
                        "description": "Prompt for summarizing documentation.",
                        "arguments": [{"name": "slug", "required": True}],
                    }
                ]
            }
        elif method == "tools/call":
            arguments = dict(payload.get("params", {}).get("arguments", {}))
            result = {
                "structuredContent": {"echoed": arguments.get("message", "")},
                "content": [{"type": "text", "text": arguments.get("message", "")}],
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


class DiscoveryHttpServer:
    def __enter__(self) -> str:
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _DiscoveryStubHandler)
        self._thread = Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        host, port = self._server.server_address
        return f"http://{host}:{port}/mcp"

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2)


class _PartialDiscoveryStubHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        content_length = int(self.headers.get("Content-Length", "0") or "0")
        payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
        method = payload.get("method")
        message_id = payload.get("id")

        if method == "initialize":
            body = {
                "jsonrpc": "2.0",
                "id": message_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {},
                        "resources": {"templates": True},
                    },
                    "serverInfo": {"name": "partial-discovery-stub", "version": "0.1.0"},
                },
            }
            self._send_json(200, body)
            return
        if method == "tools/list":
            body = {
                "jsonrpc": "2.0",
                "id": message_id,
                "result": {
                    "tools": [
                        {
                            "name": "echo_tool",
                            "description": "Echo a message back.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {"message": {"type": "string"}},
                                "required": ["message"],
                            },
                        }
                    ]
                },
            }
            self._send_json(200, body)
            return
        if method in {"resources/list", "resources/templates/list"}:
            body = {
                "jsonrpc": "2.0",
                "id": message_id,
                "error": {"code": -32601, "message": f"{method} not implemented"},
            }
            self._send_json(200, body)
            return
        if message_id is None:
            self.send_response(202)
            self.end_headers()
            return
        self._send_json(200, {"jsonrpc": "2.0", "id": message_id, "result": {}})

    def _send_json(self, status_code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return


class PartialDiscoveryHttpServer:
    def __enter__(self) -> str:
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _PartialDiscoveryStubHandler)
        self._thread = Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        host, port = self._server.server_address
        return f"http://{host}:{port}/mcp"

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2)


class McpDiscoveryTests(unittest.TestCase):
    def test_manager_discovers_mixed_capability_types_and_keeps_tool_execution(self) -> None:
        registry = ToolRegistry()
        manager = McpServerManager(registry)
        with DiscoveryHttpServer() as base_url:
            definition = McpServerDefinition(
                server_id=DISCOVERY_SERVER_ID,
                display_name="Discovery Stub",
                description="Stub discovery server.",
                transport="http",
                base_url=base_url,
            )
            manager.register_server(definition)

            server_state = manager.boot_server(DISCOVERY_SERVER_ID)
            self.assertEqual(
                server_state["capability_types"],
                [
                    MCP_CAPABILITY_PROMPT,
                    MCP_CAPABILITY_RESOURCE,
                    MCP_CAPABILITY_RESOURCE_TEMPLATE,
                    MCP_CAPABILITY_TOOL,
                ],
            )
            self.assertEqual(server_state["capability_count"], 4)
            self.assertEqual(server_state["tool_names"], [DISCOVERY_TOOL_ID])
            self.assertIn("resources", server_state["declared_capabilities"])
            self.assertEqual(server_state["server_info"]["name"], "discovery-stub")

            capabilities = manager.list_capabilities(server_id=DISCOVERY_SERVER_ID)
            self.assertEqual(
                [item["canonical_name"] for item in capabilities],
                [
                    DISCOVERY_TOOL_ID,
                    DISCOVERY_PROMPT_ID,
                    DISCOVERY_RESOURCE_ID,
                    DISCOVERY_RESOURCE_TEMPLATE_ID,
                ],
            )
            self.assertEqual(registry.list_server_tool_names(DISCOVERY_SERVER_ID), [DISCOVERY_TOOL_ID])

            tool = registry.require_invocable(DISCOVERY_TOOL_ID)
            result = tool.executor(
                {"message": "hello"},
                ToolContext(run_id="run-1", graph_id="graph-1", node_id="node-1", state_snapshot={}),
            )
            self.assertEqual(result.status, "success")
            self.assertEqual(result.output, {"echoed": "hello"})

    def test_graph_run_manager_catalog_exposes_capabilities_and_templates(self) -> None:
        services = build_example_services(include_user_mcp_servers=False)
        manager = GraphRunManager(services=services)
        with DiscoveryHttpServer() as base_url:
            definition = McpServerDefinition(
                server_id=DISCOVERY_SERVER_ID,
                display_name="Discovery Stub",
                description="Stub discovery server.",
                transport="http",
                base_url=base_url,
            )
            services.mcp_server_manager.register_server(definition)
            services.mcp_server_manager.boot_server(DISCOVERY_SERVER_ID)

            with TemporaryDirectory() as tempdir:
                template_dir = Path(tempdir)
                (template_dir / "custom_templates.json").write_text(
                    json.dumps(
                        [
                            {
                                "template_id": "custom.directory",
                                "display_name": "Directory Template",
                                "description": "Loaded from a custom directory.",
                                "draft": {
                                    "server_id": "directory_template",
                                    "display_name": "Directory Template",
                                    "description": "Directory-provided MCP server draft.",
                                    "transport": "http",
                                    "base_url": "https://example.com/mcp",
                                    "command": [],
                                    "cwd": None,
                                    "env": {},
                                    "timeout_seconds": 15,
                                    "auto_boot": False,
                                    "persistent": True,
                                },
                                "capability_hints": ["tool", "prompt"],
                            }
                        ]
                    )
                )
                with patch.dict(os.environ, {"GRAPH_AGENT_MCP_TEMPLATE_DIR": str(template_dir)}):
                    catalog = manager.get_catalog()
                    capability_names = {item["canonical_name"] for item in catalog["mcp_capabilities"]}
                    template_ids = {item["template_id"] for item in catalog["mcp_server_templates"]}

            self.assertIn(DISCOVERY_RESOURCE_ID, capability_names)
            self.assertIn(DISCOVERY_PROMPT_ID, capability_names)
            self.assertIn("official.fetch", template_ids)
            self.assertIn("custom.directory", template_ids)

    def test_test_mcp_server_returns_rich_discovery_snapshot(self) -> None:
        services = build_example_services(include_user_mcp_servers=False)
        manager = GraphRunManager(services=services)
        with DiscoveryHttpServer() as base_url:
            result = manager.test_mcp_server(
                {
                    "server_id": DISCOVERY_SERVER_ID,
                    "display_name": "Discovery Stub",
                    "description": "Stub discovery server.",
                    "transport": "http",
                    "base_url": base_url,
                    "command": [],
                    "env": {},
                    "timeout_seconds": 15,
                    "auto_boot": False,
                    "persistent": True,
                }
            )

        self.assertTrue(result["ok"])
        self.assertEqual(result["capability_count"], 4)
        self.assertEqual(
            result["capability_types"],
            [
                MCP_CAPABILITY_PROMPT,
                MCP_CAPABILITY_RESOURCE,
                MCP_CAPABILITY_RESOURCE_TEMPLATE,
                MCP_CAPABILITY_TOOL,
            ],
        )
        self.assertEqual(result["tool_names"], [DISCOVERY_TOOL_ID])
        self.assertEqual(
            {item["canonical_name"] for item in result["capabilities"]},
            {
                DISCOVERY_TOOL_ID,
                DISCOVERY_RESOURCE_ID,
                DISCOVERY_RESOURCE_TEMPLATE_ID,
                DISCOVERY_PROMPT_ID,
            },
        )

    def test_boot_succeeds_when_non_tool_discovery_methods_fail(self) -> None:
        registry = ToolRegistry()
        manager = McpServerManager(registry)
        with PartialDiscoveryHttpServer() as base_url:
            definition = McpServerDefinition(
                server_id=DISCOVERY_SERVER_ID,
                display_name="Partial Discovery Stub",
                description="Stub with partial discovery support.",
                transport="http",
                base_url=base_url,
            )
            manager.register_server(definition)

            server_state = manager.boot_server(DISCOVERY_SERVER_ID)

        self.assertTrue(server_state["running"])
        self.assertEqual(server_state["tool_names"], [DISCOVERY_TOOL_ID])
        self.assertGreaterEqual(len(server_state["discovery_warnings"]), 1)
        self.assertIn("resources/list failed", "\n".join(server_state["discovery_warnings"]))


if __name__ == "__main__":
    unittest.main()
