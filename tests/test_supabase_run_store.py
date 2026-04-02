from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from threading import Thread
import sys
import tempfile
import time
import unittest
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from graph_agent.api.graph_store import GraphStore
from graph_agent.api.manager import GraphRunManager
from graph_agent.api.run_state_reducer import apply_single_run_event, build_run_state
from graph_agent.api.supabase_run_store import SupabaseRunStore
from graph_agent.examples.tool_schema_repair import build_example_graph_payload, build_example_services


class _SupabaseStubHandler(BaseHTTPRequestHandler):
    runs: dict[str, dict[str, object]] = {}
    events: dict[str, list[dict[str, object]]] = {}

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        if parsed.path == "/rest/v1/runs":
            rows = list(self.__class__.runs.values())
            run_id = _single_query_value(query, "run_id")
            graph_id = _single_query_value(query, "graph_id")
            if run_id:
                rows = [row for row in rows if row.get("run_id") == run_id.removeprefix("eq.")]
            if graph_id:
                rows = [row for row in rows if row.get("graph_id") == graph_id.removeprefix("eq.")]
            order = _single_query_value(query, "order")
            if order == "created_at.desc":
                rows.sort(key=lambda row: str(row.get("created_at") or ""), reverse=True)
            limit = _single_query_value(query, "limit")
            if limit:
                rows = rows[: int(limit)]
            return self._write_json(rows)
        if parsed.path == "/rest/v1/run_events":
            rows = list(self.__class__.events.get(_single_query_value(query, "run_id").removeprefix("eq."), []))
            order = _single_query_value(query, "order")
            reverse = order == "sequence_number.desc"
            rows.sort(key=lambda row: int(row.get("sequence_number", 0)), reverse=reverse)
            limit = _single_query_value(query, "limit")
            if limit:
                rows = rows[: int(limit)]
            select = _single_query_value(query, "select")
            if select == "sequence_number":
                rows = [{"sequence_number": row["sequence_number"]} for row in rows]
            return self._write_json(rows)
        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0") or "0")).decode("utf-8"))
        if parsed.path == "/rest/v1/runs":
            for row in payload:
                run_id = str(row["run_id"])
                existing = dict(self.__class__.runs.get(run_id, {}))
                existing.update(row)
                self.__class__.runs[run_id] = existing
            self.send_response(201)
            self.end_headers()
            return
        if parsed.path == "/rest/v1/run_events":
            for row in payload:
                run_id = str(row["run_id"])
                self.__class__.events.setdefault(run_id, []).append(dict(row))
            self.send_response(201)
            self.end_headers()
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return

    def _write_json(self, payload: object) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _single_query_value(query: dict[str, list[str]], key: str) -> str:
    values = query.get(key, [""])
    return values[0]


class SupabaseStubServer:
    def __enter__(self) -> str:
        _SupabaseStubHandler.runs = {}
        _SupabaseStubHandler.events = {}
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _SupabaseStubHandler)
        self._thread = Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        host, port = self._server.server_address
        return f"http://{host}:{port}"

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2)


def wait_for_run_completion(manager: GraphRunManager, run_id: str, timeout_seconds: float = 5.0) -> dict[str, object]:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        state = manager.get_run(run_id)
        if state["status"] in {"completed", "failed"}:
            return state
        time.sleep(0.05)
    raise AssertionError(f"Run '{run_id}' did not finish within {timeout_seconds} seconds.")


class SupabaseRunStoreTests(unittest.TestCase):
    def test_store_recovers_run_state_from_events(self) -> None:
        with SupabaseStubServer() as url:
            store = SupabaseRunStore(url=url, service_role_key="test-key")
            state = build_run_state("run-1", "graph-1", {"prompt": "hello"})
            store.initialize_run(state)
            state = apply_single_run_event(
                state,
                {
                    "event_type": "run.started",
                    "summary": "started",
                    "payload": {},
                    "run_id": "run-1",
                    "timestamp": "2026-04-02T00:00:00Z",
                    "agent_id": None,
                    "parent_run_id": None,
                },
            )
            store.append_event("run-1", state["event_history"][-1])
            state = apply_single_run_event(
                state,
                {
                    "event_type": "node.started",
                    "summary": "node started",
                    "payload": {"node_id": "node-a", "visit_count": 1, "received_input": {"prompt": "hello"}},
                    "run_id": "run-1",
                    "timestamp": "2026-04-02T00:00:01Z",
                    "agent_id": None,
                    "parent_run_id": None,
                },
            )
            store.append_event("run-1", state["event_history"][-1])
            state = apply_single_run_event(
                state,
                {
                    "event_type": "node.completed",
                    "summary": "node done",
                    "payload": {"node_id": "node-a", "output": {"answer": "ok"}, "error": None},
                    "run_id": "run-1",
                    "timestamp": "2026-04-02T00:00:02Z",
                    "agent_id": None,
                    "parent_run_id": None,
                },
            )
            store.append_event("run-1", state["event_history"][-1])
            state = apply_single_run_event(
                state,
                {
                    "event_type": "run.completed",
                    "summary": "done",
                    "payload": {"final_output": {"answer": "ok"}},
                    "run_id": "run-1",
                    "timestamp": "2026-04-02T00:00:03Z",
                    "agent_id": None,
                    "parent_run_id": None,
                },
            )
            store.append_event("run-1", state["event_history"][-1])
            store.write_state("run-1", state)

            recovered = store.recover_run_state("run-1")
            self.assertEqual(recovered, state)
            rows = store.list_runs(graph_id="graph-1")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["run_id"], "run-1")

    def test_manager_keeps_prior_runs_queryable_on_rerun(self) -> None:
        with SupabaseStubServer() as url, tempfile.TemporaryDirectory() as temp_dir:
            services = build_example_services()
            bundled_path = Path(temp_dir) / "bundled_graphs.json"
            bundled_path.write_text(json.dumps({"graphs": []}), encoding="utf-8")
            store = GraphStore(
                services,
                path=Path(temp_dir) / "graphs.json",
                bundled_path=bundled_path,
            )
            graph_payload = build_example_graph_payload()
            graph_payload["graph_id"] = "rerun-graph"
            store.create_graph(graph_payload)
            run_store = SupabaseRunStore(url=url, service_role_key="test-key")
            manager = GraphRunManager(services=services, store=store, run_store=run_store)

            first_run_id = manager.start_run("rerun-graph", "first request")
            first_state = wait_for_run_completion(manager, first_run_id)
            second_run_id = manager.start_run("rerun-graph", "second request")
            second_state = wait_for_run_completion(manager, second_run_id)

            self.assertNotEqual(first_run_id, second_run_id)
            self.assertEqual(manager.get_run(first_run_id), first_state)
            self.assertEqual(manager.get_run(second_run_id), second_state)

            history = manager.list_runs("rerun-graph", limit=10)
            self.assertEqual({row["run_id"] for row in history}, {first_run_id, second_run_id})

    def test_manager_reconciles_stale_running_run_to_interrupted(self) -> None:
        with SupabaseStubServer() as url, tempfile.TemporaryDirectory() as temp_dir:
            services = build_example_services()
            bundled_path = Path(temp_dir) / "bundled_graphs.json"
            bundled_path.write_text(json.dumps({"graphs": []}), encoding="utf-8")
            store = GraphStore(
                services,
                path=Path(temp_dir) / "graphs.json",
                bundled_path=bundled_path,
            )
            run_store = SupabaseRunStore(url=url, service_role_key="test-key")
            state = build_run_state("stale-run", "graph-1", {"prompt": "resume"})
            state = apply_single_run_event(
                state,
                {
                    "event_type": "run.started",
                    "summary": "started",
                    "payload": {},
                    "run_id": "stale-run",
                    "timestamp": "2026-04-02T00:00:00Z",
                    "agent_id": None,
                    "parent_run_id": None,
                },
            )
            state["runtime_instance_id"] = "old-runtime"
            state["last_heartbeat_at"] = "2026-04-02T00:00:01Z"
            run_store.initialize_run(state)
            run_store.append_event("stale-run", state["event_history"][-1])
            run_store.write_state("stale-run", state)

            manager = GraphRunManager(services=services, store=store, run_store=run_store)
            recovered = manager.get_run("stale-run")

            self.assertEqual(recovered["status"], "interrupted")
            self.assertEqual(recovered["status_reason"], "runtime_heartbeat_expired")
            history = manager.list_runs(limit=10)
            stale_row = next(row for row in history if row["run_id"] == "stale-run")
            self.assertEqual(stale_row["status"], "interrupted")


if __name__ == "__main__":
    unittest.main()
