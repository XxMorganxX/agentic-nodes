from __future__ import annotations

import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from graph_agent.api.graph_store import GraphStore
from graph_agent.api.manager import GraphRunManager
from graph_agent.examples.tool_schema_repair import build_example_services
from graph_agent.runtime.documents import load_graph_document


class MultiAgentEnvironmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self.services = build_example_services()
        self.store = GraphStore(
            self.services,
            path=Path(self._temp_dir.name) / "graphs_store.json",
            bundled_path=ROOT / "src" / "graph_agent" / "api" / "graphs_store.json",
        )
        self.manager = GraphRunManager(services=self.services, store=self.store)

    def tearDown(self) -> None:
        self._temp_dir.cleanup()

    def test_bundled_test_environment_normalizes_to_multi_agent_document(self) -> None:
        payload = self.store.get_graph("test-environment")
        document = load_graph_document(payload)

        self.assertEqual(document.graph_type, "test_environment")
        self.assertEqual(document.graph_id, "test-environment")
        self.assertEqual([agent.agent_id for agent in document.agents], ["research-agent", "planner-agent", "executor-agent"])
        self.assertTrue(document.is_multi_agent)
        self.assertGreaterEqual(len(document.agents[1].edges), 2)

    def test_manager_tracks_isolated_agent_runs_for_environment_execution(self) -> None:
        run_id = self.manager.start_run("test-environment", "Find tools that can help plan and execute this task.")

        deadline = time.time() + 5
        state = self.manager.get_run(run_id)
        while state["status"] not in {"completed", "failed"} and time.time() < deadline:
            time.sleep(0.05)
            state = self.manager.get_run(run_id)

        self.assertEqual(state["status"], "completed")
        self.assertEqual(set(state["agent_runs"]), {"research-agent", "planner-agent", "executor-agent"})
        self.assertEqual(set(state["final_output"]), {"research-agent", "planner-agent", "executor-agent"})

        for agent_id, agent_state in state["agent_runs"].items():
            self.assertEqual(agent_state["agent_id"], agent_id)
            self.assertEqual(agent_state["parent_run_id"], run_id)
            self.assertEqual(agent_state["status"], "completed")
            self.assertIsNotNone(agent_state["run_id"])
            self.assertEqual(agent_state["node_errors"], {})

        agent_events = [event for event in state["event_history"] if event.get("agent_id")]
        self.assertTrue(agent_events)
        self.assertTrue(all(event["event_type"].startswith("agent.") for event in agent_events))


if __name__ == "__main__":
    unittest.main()
