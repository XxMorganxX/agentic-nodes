from __future__ import annotations

import json
import importlib
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from graph_agent.api.graph_store import GraphStore
from graph_agent.api.manager import GraphRunManager
from graph_agent.api.run_log_store import RunLogStore
from graph_agent.examples.tool_schema_repair import build_example_services
from graph_agent.runtime.core import GraphDefinition, NodeContext, RunState


def wait_for_run_completion(manager: GraphRunManager, run_id: str, timeout_seconds: float = 5.0) -> dict[str, object]:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        state = manager.get_run(run_id)
        if state["status"] in {"completed", "failed"}:
            return state
        time.sleep(0.05)
    raise AssertionError(f"Run '{run_id}' did not finish within {timeout_seconds} seconds.")


def sample_run_document() -> dict[str, object]:
    return {
        "document_id": "doc-1",
        "name": "brief.txt",
        "mime_type": "text/plain",
        "size_bytes": 24,
        "storage_path": "/tmp/brief.txt",
        "text_content": "Use the attached checklist.",
        "text_excerpt": "Use the attached checklist.",
        "status": "ready",
        "error": None,
    }


class RunDocumentTests(unittest.TestCase):
    def test_upload_endpoint_saves_and_extracts_text_document(self) -> None:
        app_module = importlib.import_module("graph_agent.api.app")
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            bundled_path = temp_path / "bundled_graphs.json"
            bundled_path.write_text(json.dumps({"graphs": []}), encoding="utf-8")
            services = build_example_services()
            manager = GraphRunManager(
                services=services,
                store=GraphStore(services, path=temp_path / "graphs.json", bundled_path=bundled_path),
                run_log_store=RunLogStore(temp_path / ".logs" / "runs"),
            )
            original_manager = app_module.manager
            app_module.manager = manager
            try:
                with patch.dict("os.environ", {"GRAPH_AGENT_UPLOAD_DIR": str(temp_path / ".graph-agent" / "uploads")}, clear=False):
                    with TestClient(app_module.app) as client:
                        response = client.post(
                            "/api/editor/documents/upload",
                            files=[("files", ("brief.txt", b"Use the attached checklist.", "text/plain"))],
                        )
                self.assertEqual(response.status_code, 200)
                payload = response.json()
                self.assertEqual(len(payload["documents"]), 1)
                document = payload["documents"][0]
                self.assertEqual(document["status"], "ready")
                self.assertEqual(document["name"], "brief.txt")
                self.assertIn("attached checklist", document["text_content"])
                self.assertTrue(Path(document["storage_path"]).exists())
            finally:
                app_module.manager = original_manager
                manager.stop_background_services()

    def test_runtime_exposes_documents_in_prompt_templates(self) -> None:
        services = build_example_services()
        graph = GraphDefinition.from_dict(
            {
                "graph_id": "document-prompt",
                "name": "Document Prompt",
                "description": "",
                "version": "1.0",
                "env_vars": {},
                "start_node_id": "start",
                "nodes": [
                    {
                        "id": "start",
                        "kind": "input",
                        "category": "start",
                        "label": "Start",
                        "provider_id": "core.input",
                        "provider_label": "Input",
                        "config": {},
                        "position": {"x": 0, "y": 0},
                    },
                    {
                        "id": "prompt_block",
                        "kind": "data",
                        "category": "data",
                        "label": "Prompt Block",
                        "provider_id": "core.prompt_block",
                        "provider_label": "Prompt Block",
                        "config": {
                            "mode": "prompt_block",
                            "role": "system",
                            "name": "doc_context",
                            "content": "Documents:\\n{documents}",
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
                        "config": {"source_binding": {"type": "latest_payload", "source": "prompt_block"}},
                        "position": {"x": 480, "y": 0},
                    },
                ],
                "edges": [
                    {"id": "start-finish", "source_id": "start", "target_id": "finish", "label": "", "kind": "standard", "priority": 100},
                ],
            }
        )
        graph.validate_against_services(services)
        prompt_node = graph.get_node("prompt_block")
        context = NodeContext(
            graph=graph,
            state=RunState(graph_id=graph.graph_id, input_payload="hello", documents=[sample_run_document()]),
            services=services,
            node_id="prompt_block",
        )
        result = prompt_node.execute(context)

        self.assertEqual(result.status, "success")
        self.assertIn("brief.txt", result.output["payload"]["content"])
        self.assertIn("Use the attached checklist.", result.output["payload"]["content"])

    def test_environment_runs_and_recovery_keep_documents(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            services = build_example_services()
            store = GraphStore(
                services,
                path=temp_path / "graphs_store.json",
                bundled_path=ROOT / "src" / "graph_agent" / "api" / "graphs_store.json",
            )
            run_store = RunLogStore(temp_path / ".logs" / "runs")
            manager = GraphRunManager(services=services, store=store, run_log_store=run_store)

            run_id = manager.start_run("test-environment", "Review the uploaded brief.", documents=[sample_run_document()])
            state = wait_for_run_completion(manager, run_id)

            self.assertEqual(state["documents"][0]["document_id"], "doc-1")
            self.assertTrue(state["agent_runs"])
            for agent_state in state["agent_runs"].values():
                self.assertEqual(agent_state["documents"][0]["document_id"], "doc-1")

            recovered_manager = GraphRunManager(services=services, store=store, run_log_store=run_store)
            recovered_state = recovered_manager.get_run(run_id)
            self.assertEqual(recovered_state["documents"][0]["document_id"], "doc-1")
            for agent_state in recovered_state["agent_runs"].values():
                self.assertEqual(agent_state["documents"][0]["document_id"], "doc-1")

    def test_upload_xlsx_stores_derived_csv(self) -> None:
        import io

        import openpyxl

        buf = io.BytesIO()
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet.append(["name", "qty"])
        sheet.append(["alpha", "10"])
        workbook.save(buf)
        xlsx_bytes = buf.getvalue()

        app_module = importlib.import_module("graph_agent.api.app")
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            bundled_path = temp_path / "bundled_graphs.json"
            bundled_path.write_text(json.dumps({"graphs": []}), encoding="utf-8")
            services = build_example_services()
            manager = GraphRunManager(
                services=services,
                store=GraphStore(services, path=temp_path / "graphs.json", bundled_path=bundled_path),
                run_log_store=RunLogStore(temp_path / ".logs" / "runs"),
            )
            original_manager = app_module.manager
            app_module.manager = manager
            try:
                with patch.dict("os.environ", {"GRAPH_AGENT_UPLOAD_DIR": str(temp_path / ".graph-agent" / "uploads")}, clear=False):
                    with TestClient(app_module.app) as client:
                        response = client.post(
                            "/api/editor/documents/upload",
                            files=[("files", ("grid.xlsx", xlsx_bytes, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"))],
                        )
                self.assertEqual(response.status_code, 200, msg=response.text)
                payload = response.json()
                document = payload["documents"][0]
                self.assertEqual(document["status"], "ready")
                self.assertEqual(document["name"], "grid.csv")
                self.assertEqual(document["mime_type"], "text/csv")
                stored = Path(document["storage_path"])
                self.assertTrue(stored.exists())
                self.assertEqual(stored.suffix.lower(), ".csv")
                csv_body = stored.read_text(encoding="utf-8-sig")
                self.assertIn("name", csv_body)
                self.assertIn("alpha", csv_body)
                self.assertIn("Sheet: Sheet", document["text_content"])
            finally:
                app_module.manager = original_manager
                manager.stop_background_services()


if __name__ == "__main__":
    unittest.main()
