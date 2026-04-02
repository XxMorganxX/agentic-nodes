from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from graph_agent.api.run_state_reducer import apply_event, apply_single_run_event, build_run_state


def _event(
    event_type: str,
    *,
    run_id: str,
    payload: dict[str, object] | None = None,
    agent_id: str | None = None,
    timestamp: str = "2026-04-02T00:00:00Z",
) -> dict[str, object]:
    return {
        "event_type": event_type,
        "summary": event_type,
        "payload": payload or {},
        "run_id": run_id,
        "timestamp": timestamp,
        "agent_id": agent_id,
        "parent_run_id": None,
    }


class RunStateReducerTests(unittest.TestCase):
    def test_single_run_reducer_tracks_inputs_edges_and_terminal_status(self) -> None:
        state = build_run_state("run-1", "graph-1", {"prompt": "hello"})
        state = apply_single_run_event(state, _event("run.started", run_id="run-1"))
        state = apply_single_run_event(
            state,
            _event(
                "node.started",
                run_id="run-1",
                payload={"node_id": "node-a", "visit_count": 1, "received_input": {"text": "hello"}},
            ),
        )
        state = apply_single_run_event(
            state,
            _event(
                "node.completed",
                run_id="run-1",
                payload={
                    "node_id": "node-a",
                    "output": {"answer": "ok"},
                    "route_outputs": {"success": {"answer": "routed"}},
                    "error": None,
                },
            ),
        )
        state = apply_single_run_event(
            state,
            _event(
                "edge.selected",
                run_id="run-1",
                payload={
                    "id": "edge-1",
                    "source_id": "node-a",
                    "target_id": "node-b",
                    "source_handle_id": "success",
                },
            ),
        )
        state = apply_single_run_event(
            state,
            _event("run.completed", run_id="run-1", payload={"final_output": {"answer": "done"}}),
        )

        self.assertEqual(state["status"], "completed")
        self.assertEqual(state["node_inputs"]["node-a"], {"text": "hello"})
        self.assertEqual(state["node_outputs"]["node-a"], {"answer": "ok"})
        self.assertEqual(state["edge_outputs"]["edge-1"], {"answer": "routed"})
        self.assertIsNone(state["current_node_id"])
        self.assertIsNone(state["current_edge_id"])
        self.assertEqual(state["final_output"], {"answer": "done"})

    def test_single_run_reducer_tracks_failure_and_cancellation(self) -> None:
        failed_state = build_run_state("run-failed", "graph-1", None)
        failed_state = apply_single_run_event(
            failed_state,
            _event(
                "node.completed",
                run_id="run-failed",
                payload={"node_id": "node-a", "output": None, "error": {"type": "boom"}},
            ),
        )
        failed_state = apply_single_run_event(
            failed_state,
            _event(
                "run.failed",
                run_id="run-failed",
                payload={"error": {"type": "node_exception", "message": "boom"}},
            ),
        )
        cancelled_state = apply_single_run_event(
            build_run_state("run-cancelled", "graph-1", None),
            _event(
                "run.cancelled",
                run_id="run-cancelled",
                payload={"error": {"type": "run_cancelled", "message": "stopped"}},
            ),
        )

        self.assertEqual(failed_state["status"], "failed")
        self.assertEqual(failed_state["node_errors"]["node-a"], {"type": "boom"})
        self.assertEqual(failed_state["terminal_error"], {"type": "node_exception", "message": "boom"})
        self.assertEqual(cancelled_state["status"], "cancelled")
        self.assertEqual(cancelled_state["terminal_error"], {"type": "run_cancelled", "message": "stopped"})

    def test_single_run_reducer_tracks_interruption(self) -> None:
        state = apply_single_run_event(
            build_run_state("run-interrupted", "graph-1", None),
            _event("run.started", run_id="run-interrupted"),
        )
        state = apply_single_run_event(
            state,
            _event(
                "run.interrupted",
                run_id="run-interrupted",
                payload={
                    "reason": "runtime_heartbeat_expired",
                    "error": {"type": "runtime_interrupted", "message": "heartbeat expired"},
                },
            ),
        )

        self.assertEqual(state["status"], "interrupted")
        self.assertEqual(state["status_reason"], "runtime_heartbeat_expired")
        self.assertEqual(state["terminal_error"], {"type": "runtime_interrupted", "message": "heartbeat expired"})

    def test_agent_events_create_and_update_nested_child_state(self) -> None:
        state = build_run_state("parent-run", "graph-1", {"prompt": "hello"})
        state = apply_event(
            state,
            _event(
                "agent.run.started",
                run_id="parent-run",
                agent_id="agent-a",
                payload={"agent_id": "agent-a", "agent_name": "Agent A", "child_run_id": "child-run"},
            ),
        )
        state = apply_event(
            state,
            _event(
                "agent.node.started",
                run_id="parent-run",
                agent_id="agent-a",
                payload={
                    "agent_id": "agent-a",
                    "agent_name": "Agent A",
                    "child_run_id": "child-run",
                    "node_id": "node-a",
                    "visit_count": 1,
                    "received_input": "hello",
                },
            ),
        )
        state = apply_event(
            state,
            _event(
                "agent.run.completed",
                run_id="parent-run",
                agent_id="agent-a",
                payload={
                    "agent_id": "agent-a",
                    "agent_name": "Agent A",
                    "child_run_id": "child-run",
                    "final_output": "done",
                },
            ),
        )

        child_state = state["agent_runs"]["agent-a"]
        self.assertEqual(child_state["run_id"], "child-run")
        self.assertEqual(child_state["agent_id"], "agent-a")
        self.assertEqual(child_state["agent_name"], "Agent A")
        self.assertEqual(child_state["status"], "completed")
        self.assertEqual(child_state["current_node_id"], None)
        self.assertEqual(child_state["node_inputs"]["node-a"], "hello")
        self.assertEqual(len(state["event_history"]), 3)


if __name__ == "__main__":
    unittest.main()
