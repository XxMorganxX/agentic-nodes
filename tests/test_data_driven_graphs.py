from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from graph_agent.api.graph_store import GraphStore
from graph_agent.examples.tool_schema_repair import build_example_graph_payload, build_example_services
from graph_agent.providers.base import ModelRequest, ModelResponse
from graph_agent.runtime.core import GraphDefinition, GraphValidationError
from graph_agent.runtime.engine import GraphRuntime


class RecordingModelProvider:
    name = "openai"

    def __init__(self) -> None:
        self.last_request: ModelRequest | None = None

    def generate(self, request: ModelRequest) -> ModelResponse:
        self.last_request = request
        return ModelResponse(content="resolved-output")


class DataDrivenGraphTests(unittest.TestCase):
    def setUp(self) -> None:
        self.services = build_example_services()

    def test_example_graph_runs_end_to_end(self) -> None:
        graph = GraphDefinition.from_dict(build_example_graph_payload())
        graph.validate_against_services(self.services)

        runtime = GraphRuntime(
            services=self.services,
            max_steps=self.services.config["max_steps"],
            max_visits_per_node=self.services.config["max_visits_per_node"],
        )

        state = runtime.run(graph, "Find graph-agent references for a schema repair workflow.")

        self.assertEqual(state.status, "completed")
        self.assertIsNotNone(state.final_output)
        self.assertIn("message", state.final_output)
        self.assertEqual(state.transition_history[2].target_id, "repair_tool")

    def test_invalid_category_connection_is_rejected(self) -> None:
        payload = build_example_graph_payload()
        payload["edges"].append(
            {
                "id": "edge-finish-start",
                "source_id": "finish",
                "target_id": "start",
                "label": "invalid",
                "kind": "standard",
                "priority": 100,
                "condition": None,
            }
        )

        with self.assertRaises(GraphValidationError):
            GraphDefinition.from_dict(payload)

    def test_graph_env_vars_resolve_in_runtime_templates_and_provider_config(self) -> None:
        recording_provider = RecordingModelProvider()
        self.services.model_providers["openai"] = recording_provider

        payload: dict[str, Any] = {
            "graph_id": "env-agent",
            "name": "Env Agent",
            "description": "",
            "version": "1.0",
            "start_node_id": "start",
            "env_vars": {
                "OPENAI_API_KEY": "OPENAI_API_KEY",
                "SYSTEM_ROLE": "schema fixer",
                "MODEL_NAME": "gpt-4.1-mini",
            },
            "nodes": [
                {
                    "id": "start",
                    "kind": "input",
                    "category": "start",
                    "label": "Start",
                    "provider_id": "core.input",
                    "provider_label": "Core Input Node",
                    "description": "",
                    "position": {"x": 0, "y": 0},
                    "config": {"input_binding": {"type": "input_payload"}},
                },
                {
                    "id": "model",
                    "kind": "model",
                    "category": "api",
                    "label": "Call Model",
                    "provider_id": "core.api",
                    "provider_label": "Core API Node",
                    "description": "",
                    "position": {"x": 240, "y": 0},
                    "model_provider_name": "openai",
                    "prompt_name": "env_prompt",
                    "config": {
                        "provider_name": "openai",
                        "prompt_name": "env_prompt",
                        "model": "{MODEL_NAME}",
                        "api_key_env_var": "{OPENAI_API_KEY}",
                        "system_prompt": "You are a {SYSTEM_ROLE}.",
                        "user_message_template": "Input: {input_payload}",
                    },
                },
                {
                    "id": "finish",
                    "kind": "output",
                    "category": "end",
                    "label": "Finish",
                    "provider_id": "core.output",
                    "provider_label": "Core Output Node",
                    "description": "",
                    "position": {"x": 480, "y": 0},
                    "config": {},
                },
            ],
            "edges": [
                {
                    "id": "edge-start-model",
                    "source_id": "start",
                    "target_id": "model",
                    "label": "",
                    "kind": "standard",
                    "priority": 100,
                    "condition": None,
                },
                {
                    "id": "edge-model-finish",
                    "source_id": "model",
                    "target_id": "finish",
                    "label": "",
                    "kind": "standard",
                    "priority": 100,
                    "condition": None,
                },
            ],
        }

        graph = GraphDefinition.from_dict(payload)
        graph.validate_against_services(self.services)

        runtime = GraphRuntime(
            services=self.services,
            max_steps=self.services.config["max_steps"],
            max_visits_per_node=self.services.config["max_visits_per_node"],
        )
        state = runtime.run(graph, "repair the broken schema")

        self.assertEqual(state.status, "completed")
        self.assertIsNotNone(recording_provider.last_request)
        assert recording_provider.last_request is not None
        self.assertEqual(recording_provider.last_request.messages[0].content, "You are a schema fixer.")
        self.assertEqual(recording_provider.last_request.messages[1].content, "Input: repair the broken schema")
        self.assertEqual(recording_provider.last_request.provider_config["model"], "gpt-4.1-mini")
        self.assertEqual(recording_provider.last_request.provider_config["api_key_env_var"], "OPENAI_API_KEY")

    def test_graph_store_crud_and_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = GraphStore(self.services, path=Path(directory) / "graphs.json")

            catalog = store.catalog()
            self.assertTrue(catalog["node_providers"])
            self.assertTrue(catalog["tools"])
            self.assertTrue(catalog["connection_rules"])

            new_graph = build_example_graph_payload()
            new_graph["graph_id"] = "editable-agent"
            new_graph["name"] = "Editable Agent"

            created = store.create_graph(new_graph)
            self.assertEqual(created["graph_id"], "editable-agent")

            created["description"] = "Updated description"
            updated = store.update_graph("editable-agent", created)
            self.assertEqual(updated["description"], "Updated description")

            store.delete_graph("editable-agent")
            remaining_ids = {graph["graph_id"] for graph in store.list_graphs()}
            self.assertNotIn("editable-agent", remaining_ids)


if __name__ == "__main__":
    unittest.main()
