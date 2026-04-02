from __future__ import annotations

from copy import deepcopy
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


class InMemoryRunStore:
    def __init__(self) -> None:
        self._manifests: dict[str, dict[str, object]] = {}
        self._events: dict[str, list[dict[str, object]]] = {}
        self._states: dict[str, dict[str, object]] = {}

    def initialize_run(self, state: dict[str, object]) -> None:
        run_id = str(state["run_id"])
        self._manifests[run_id] = {
            "run_id": run_id,
            "graph_id": state.get("graph_id"),
            "agent_id": state.get("agent_id"),
            "agent_name": state.get("agent_name"),
            "parent_run_id": state.get("parent_run_id"),
            "input_payload": state.get("input_payload"),
        }
        self._states[run_id] = deepcopy(state)
        self._events.setdefault(run_id, [])

    def append_event(self, run_id: str, event: dict[str, object]) -> None:
        self._events.setdefault(run_id, []).append(deepcopy(event))

    def write_state(self, run_id: str, state: dict[str, object]) -> None:
        self._states[run_id] = deepcopy(state)

    def load_manifest(self, run_id: str) -> dict[str, object] | None:
        manifest = self._manifests.get(run_id)
        return deepcopy(manifest) if manifest is not None else None

    def load_events(self, run_id: str) -> list[dict[str, object]]:
        return deepcopy(self._events.get(run_id, []))

    def load_state(self, run_id: str) -> dict[str, object] | None:
        state = self._states.get(run_id)
        return deepcopy(state) if state is not None else None

    def recover_run_state(self, run_id: str) -> dict[str, object] | None:
        return self.load_state(run_id)

    def list_runs(self, *, graph_id: str | None = None, limit: int = 50) -> list[dict[str, object]]:
        rows = [deepcopy(state) for state in self._states.values()]
        if graph_id is not None:
            rows = [row for row in rows if row.get("graph_id") == graph_id]
        return rows[:limit]


class MultiAgentEnvironmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self.services = build_example_services()
        self.store = GraphStore(
            self.services,
            path=Path(self._temp_dir.name) / "graphs_store.json",
            bundled_path=ROOT / "src" / "graph_agent" / "api" / "graphs_store.json",
        )
        self.manager = GraphRunManager(
            services=self.services,
            store=self.store,
            run_store=InMemoryRunStore(),
        )

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

    def test_manager_can_run_selected_environment_agents_only(self) -> None:
        run_id = self.manager.start_run(
            "test-environment",
            "Run only a subset of agents.",
            agent_ids=["planner-agent", "executor-agent"],
        )

        deadline = time.time() + 5
        state = self.manager.get_run(run_id)
        while state["status"] not in {"completed", "failed"} and time.time() < deadline:
            time.sleep(0.05)
            state = self.manager.get_run(run_id)

        self.assertEqual(state["status"], "completed")
        self.assertEqual(set(state["agent_runs"]), {"planner-agent", "executor-agent"})
        self.assertEqual(set(state["final_output"]), {"planner-agent", "executor-agent"})
        self.assertNotIn("research-agent", state["agent_runs"])

    def test_manager_rejects_empty_environment_agent_selection(self) -> None:
        with self.assertRaisesRegex(ValueError, "Select at least one agent"):
            self.manager.start_run("test-environment", "Run nothing.", agent_ids=[])


if __name__ == "__main__":
    unittest.main()
