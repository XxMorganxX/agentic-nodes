from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from graph_agent.examples.tool_schema_repair import build_example_graph_payload, build_example_services
from graph_agent.providers.base import ModelMessage, ModelRequest
from graph_agent.providers.vendor_api import ClaudeMessagesModelProvider, OpenAIChatModelProvider
from graph_agent.runtime.core import GraphDefinition


class StubOpenAIProvider(OpenAIChatModelProvider):
    def __init__(self) -> None:
        self.last_payload: Mapping[str, Any] | None = None

    def _post_json(
        self,
        url: str,
        payload: Mapping[str, Any],
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> Mapping[str, Any]:
        self.last_payload = payload
        return {
            "model": payload.get("model"),
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "emit_structured_output",
                                    "arguments": '{"query":"graph agents","limit":3}',
                                }
                            }
                        ],
                    },
                }
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 7},
        }

    def _headers(self, provider_config: Mapping[str, Any]) -> dict[str, str]:
        return {"Authorization": "Bearer test"}


class StubClaudeProvider(ClaudeMessagesModelProvider):
    def __init__(self) -> None:
        self.last_payload: Mapping[str, Any] | None = None

    def _post_json(
        self,
        url: str,
        payload: Mapping[str, Any],
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> Mapping[str, Any]:
        self.last_payload = payload
        return {
            "model": payload.get("model"),
            "content": [
                {
                    "type": "tool_use",
                    "name": "emit_structured_output",
                    "input": {"query": "graph agents", "limit": 3},
                }
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 11, "output_tokens": 6},
        }

    def _headers(self, provider_config: Mapping[str, Any]) -> dict[str, str]:
        return {"x-api-key": "test", "anthropic-version": "2023-06-01"}


class ModelProviderTests(unittest.TestCase):
    def test_openai_provider_derives_tool_schema_for_tool_call_nodes(self) -> None:
        provider = StubOpenAIProvider()
        request = ModelRequest(
            prompt_name="schema_proposal",
            messages=[ModelMessage(role="user", content="Build the tool payload.")],
            provider_config={"model": "gpt-4.1-mini"},
            metadata={
                "response_mode": "tool_call",
                "preferred_tool_name": "search_catalog",
                "available_tools": [
                    {
                        "name": "search_catalog",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"},
                                "limit": {"type": "integer"},
                            },
                            "required": ["query", "limit"],
                        },
                    }
                ],
            },
        )

        response = provider.generate(request)

        self.assertEqual(response.structured_output, {"query": "graph agents", "limit": 3})
        self.assertIsNotNone(provider.last_payload)
        assert provider.last_payload is not None
        self.assertIn("tools", provider.last_payload)
        self.assertEqual(
            provider.last_payload["tools"][0]["function"]["parameters"]["required"],
            ["query", "limit"],
        )

    def test_claude_provider_derives_tool_schema_for_tool_call_nodes(self) -> None:
        provider = StubClaudeProvider()
        request = ModelRequest(
            prompt_name="schema_proposal",
            messages=[
                ModelMessage(role="system", content="Return structured JSON."),
                ModelMessage(role="user", content="Build the tool payload."),
            ],
            provider_config={"model": "claude-3-5-haiku-latest"},
            metadata={
                "response_mode": "tool_call",
                "preferred_tool_name": "search_catalog",
                "available_tools": [
                    {
                        "name": "search_catalog",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"},
                                "limit": {"type": "integer"},
                            },
                            "required": ["query", "limit"],
                        },
                    }
                ],
            },
        )

        response = provider.generate(request)

        self.assertEqual(response.structured_output, {"query": "graph agents", "limit": 3})
        self.assertIsNotNone(provider.last_payload)
        assert provider.last_payload is not None
        self.assertIn("tools", provider.last_payload)
        self.assertEqual(
            provider.last_payload["tools"][0]["input_schema"]["required"],
            ["query", "limit"],
        )

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


if __name__ == "__main__":
    unittest.main()
