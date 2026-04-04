from __future__ import annotations

import csv
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

from openpyxl import Workbook

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from graph_agent.examples.tool_schema_repair import build_example_services
from graph_agent.api.run_state_reducer import apply_single_run_event, build_run_state
from graph_agent.providers.base import ModelRequest, ModelResponse, ProviderPreflightResult
from graph_agent.runtime.core import GraphDefinition
from graph_agent.runtime.engine import GraphRuntime
from graph_agent.runtime.event_contract import normalize_runtime_state_snapshot
from graph_agent.runtime.spreadsheets import parse_spreadsheet, resolve_spreadsheet_path_from_run_documents


class SpreadsheetEchoProvider:
    name = "spreadsheet_echo"

    def __init__(self) -> None:
        self.user_messages: list[str] = []

    def generate(self, request: ModelRequest) -> ModelResponse:
        user_message = request.messages[-1].content if request.messages else ""
        self.user_messages.append(user_message)
        return ModelResponse(
            content=user_message,
            structured_output={
                "message": user_message,
                "need_tool": False,
                "tool_calls": [],
            },
        )

    def preflight(self, provider_config=None) -> ProviderPreflightResult:
        return ProviderPreflightResult(
            status="available",
            ok=True,
            message="Spreadsheet echo provider is available for tests.",
            details={"backend_type": "test"},
        )


class SpreadsheetRowTests(unittest.TestCase):
    def test_resolve_spreadsheet_path_from_run_documents(self) -> None:
        path = "/data/rows.csv"
        docs = [
            {
                "document_id": "d1",
                "name": "rows.csv",
                "status": "ready",
                "storage_path": path,
            }
        ]
        self.assertEqual(resolve_spreadsheet_path_from_run_documents(docs), path)
        two = [
            docs[0],
            {**docs[0], "document_id": "d2", "name": "b.csv", "storage_path": "/other.csv"},
        ]
        self.assertEqual(resolve_spreadsheet_path_from_run_documents(two), "")

    def test_runtime_uses_run_document_when_file_path_empty(self) -> None:
        with TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "rows.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["city", "temperature"])
                writer.writerow(["Seattle", "58"])

            services = build_example_services()
            provider = SpreadsheetEchoProvider()
            services.model_providers["spreadsheet_echo"] = provider
            runtime = GraphRuntime(
                services=services,
                max_steps=services.config["max_steps"],
                max_visits_per_node=services.config["max_visits_per_node"],
            )
            graph_payload = {
                "graph_id": "spreadsheet-row-docs-graph",
                "name": "Spreadsheet Row Docs Graph",
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
                        "id": "sheet",
                        "kind": "data",
                        "category": "data",
                        "label": "Spreadsheet Rows",
                        "provider_id": "core.spreadsheet_rows",
                        "provider_label": "Spreadsheet Rows",
                        "config": {
                            "mode": "spreadsheet_rows",
                            "file_format": "csv",
                            "file_path": "",
                            "sheet_name": "",
                            "header_row_index": 1,
                            "start_row_index": 2,
                            "empty_row_policy": "skip",
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
                        "model_provider_name": "spreadsheet_echo",
                        "prompt_name": "spreadsheet_prompt",
                        "config": {
                            "provider_name": "spreadsheet_echo",
                            "prompt_name": "spreadsheet_prompt",
                            "system_prompt": "Process the current spreadsheet row.",
                            "user_message_template": "{input_payload}",
                            "response_mode": "message",
                        },
                        "position": {"x": 220, "y": 0},
                    },
                    {
                        "id": "finish",
                        "kind": "output",
                        "category": "end",
                        "label": "Finish",
                        "provider_id": "core.output",
                        "provider_label": "Core Output Node",
                        "config": {"source_binding": {"type": "latest_payload", "source": "model"}},
                        "position": {"x": 340, "y": 0},
                    },
                ],
                "edges": [
                    {"id": "e1", "source_id": "start", "target_id": "sheet", "label": "", "kind": "standard", "priority": 100},
                    {"id": "e2", "source_id": "sheet", "target_id": "model", "label": "", "kind": "standard", "priority": 100},
                    {"id": "e3", "source_id": "model", "target_id": "finish", "label": "", "kind": "standard", "priority": 100},
                ],
            }
            graph = GraphDefinition.from_dict(graph_payload)
            graph.validate_against_services(services)
            run_docs = [
                {
                    "document_id": "doc-1",
                    "name": "rows.csv",
                    "mime_type": "text/csv",
                    "size_bytes": 1,
                    "storage_path": str(csv_path),
                    "text_content": "",
                    "text_excerpt": "",
                    "status": "ready",
                    "error": None,
                }
            ]
            state = runtime.run(
                graph,
                {"request": "Process spreadsheet rows"},
                run_id="spreadsheet-row-runtime-docs",
                documents=run_docs,
            )

        self.assertEqual(state.status, "completed")
        self.assertEqual(len(provider.user_messages), 1)
        self.assertIn('"city": "Seattle"', provider.user_messages[0])

    def test_parse_csv_normalizes_headers_and_skips_empty_rows(self) -> None:
        with TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "people.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["Name", "Name", ""])
                writer.writerow(["Alice", "Engineer", "Seattle"])
                writer.writerow(["", "", ""])
                writer.writerow(["Bob", "Manager", "Portland"])

            parsed = parse_spreadsheet(file_path=str(csv_path), file_format="csv")

        self.assertEqual(parsed.headers, ["Name", "Name_2", "column_3"])
        self.assertEqual(parsed.row_count, 2)
        self.assertEqual(parsed.rows[0].row_number, 2)
        self.assertEqual(
            parsed.rows[0].row_data,
            {"Name": "Alice", "Name_2": "Engineer", "column_3": "Seattle"},
        )
        self.assertEqual(parsed.rows[1].row_number, 4)

    def test_parse_spreadsheet_always_uses_first_row_as_headers(self) -> None:
        with TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "people.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["city", "temperature"])
                writer.writerow(["Seattle", "58"])
                writer.writerow(["Portland", "62"])

            parsed = parse_spreadsheet(
                file_path=str(csv_path),
                file_format="csv",
                header_row_index=9,
                start_row_index=10,
            )

        self.assertEqual(parsed.headers, ["city", "temperature"])
        self.assertEqual(parsed.rows[0].row_number, 2)
        self.assertEqual(parsed.rows[0].row_data, {"city": "Seattle", "temperature": "58"})

    def test_parse_xlsx_uses_selected_sheet(self) -> None:
        with TemporaryDirectory() as temp_dir:
            xlsx_path = Path(temp_dir) / "inventory.xlsx"
            workbook = Workbook()
            default_sheet = workbook.active
            default_sheet.title = "IgnoreMe"
            default_sheet.append(["unused"])
            data_sheet = workbook.create_sheet("Products")
            data_sheet.append(["sku", "qty"])
            data_sheet.append(["A-1", 3])
            data_sheet.append(["B-2", 9])
            workbook.save(xlsx_path)

            parsed = parse_spreadsheet(
                file_path=str(xlsx_path),
                file_format="xlsx",
                sheet_name="Products",
                header_row_index=1,
                start_row_index=2,
            )

        self.assertEqual(parsed.sheet_name, "Products")
        self.assertIn("IgnoreMe", parsed.sheet_names)
        self.assertEqual(parsed.headers, ["sku", "qty"])
        self.assertEqual(parsed.row_count, 2)
        self.assertEqual(parsed.rows[1].row_data, {"sku": "B-2", "qty": 9})

    def test_runtime_processes_spreadsheet_rows_sequentially(self) -> None:
        with TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "rows.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["city", "temperature"])
                writer.writerow(["Seattle", "58"])
                writer.writerow(["Portland", "62"])

            services = build_example_services()
            provider = SpreadsheetEchoProvider()
            services.model_providers["spreadsheet_echo"] = provider
            runtime = GraphRuntime(
                services=services,
                max_steps=services.config["max_steps"],
                max_visits_per_node=services.config["max_visits_per_node"],
            )
            graph_payload = {
                "graph_id": "spreadsheet-row-runtime-graph",
                "name": "Spreadsheet Row Runtime Graph",
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
                        "id": "sheet",
                        "kind": "data",
                        "category": "data",
                        "label": "Spreadsheet Rows",
                        "provider_id": "core.spreadsheet_rows",
                        "provider_label": "Spreadsheet Rows",
                        "config": {
                            "mode": "spreadsheet_rows",
                            "file_format": "csv",
                            "file_path": str(csv_path),
                            "sheet_name": "",
                            "header_row_index": 1,
                            "start_row_index": 2,
                            "empty_row_policy": "skip",
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
                        "model_provider_name": "spreadsheet_echo",
                        "prompt_name": "spreadsheet_prompt",
                        "config": {
                            "provider_name": "spreadsheet_echo",
                            "prompt_name": "spreadsheet_prompt",
                            "system_prompt": "Process the current spreadsheet row.",
                            "user_message_template": "{input_payload}",
                            "response_mode": "message",
                        },
                        "position": {"x": 220, "y": 0},
                    },
                    {
                        "id": "finish",
                        "kind": "output",
                        "category": "end",
                        "label": "Finish",
                        "provider_id": "core.output",
                        "provider_label": "Core Output Node",
                        "config": {"source_binding": {"type": "latest_payload", "source": "model"}},
                        "position": {"x": 340, "y": 0},
                    },
                ],
                "edges": [
                    {"id": "e1", "source_id": "start", "target_id": "sheet", "label": "", "kind": "standard", "priority": 100},
                    {"id": "e2", "source_id": "sheet", "target_id": "model", "label": "", "kind": "standard", "priority": 100},
                    {"id": "e3", "source_id": "model", "target_id": "finish", "label": "", "kind": "standard", "priority": 100},
                ],
            }
            graph = GraphDefinition.from_dict(graph_payload)
            graph.validate_against_services(services)

            state = runtime.run(graph, {"request": "Process spreadsheet rows"}, run_id="spreadsheet-row-runtime")

        self.assertEqual(state.status, "completed")
        self.assertEqual(len(provider.user_messages), 2)
        self.assertIn('"city": "Seattle"', provider.user_messages[0])
        self.assertIn('"city": "Portland"', provider.user_messages[1])
        self.assertEqual(state.visit_counts.get("model"), 2)
        self.assertEqual(state.iterator_states["sheet"]["status"], "completed")
        self.assertEqual(state.iterator_states["sheet"]["current_row_index"], 2)
        self.assertEqual(state.iterator_states["sheet"]["total_rows"], 2)
        self.assertIsInstance(state.final_output, str)
        self.assertIn("Portland", state.final_output)

    def test_context_builder_renders_spreadsheet_rows_as_llm_friendly_text(self) -> None:
        with TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "jobs.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["Company", "CEO", "Summer_2026_Internships?"])
                writer.writerow(["Scale AI", "Alexandr Wang", "YES"])

            services = build_example_services()
            runtime = GraphRuntime(
                services=services,
                max_steps=services.config["max_steps"],
                max_visits_per_node=services.config["max_visits_per_node"],
            )
            graph_payload = {
                "graph_id": "spreadsheet-row-context-builder-graph",
                "name": "Spreadsheet Row Context Builder Graph",
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
                        "id": "sheet",
                        "kind": "data",
                        "category": "data",
                        "label": "Spreadsheet Rows",
                        "provider_id": "core.spreadsheet_rows",
                        "provider_label": "Spreadsheet Rows",
                        "config": {
                            "mode": "spreadsheet_rows",
                            "file_format": "csv",
                            "file_path": str(csv_path),
                            "sheet_name": "",
                            "empty_row_policy": "skip",
                        },
                        "position": {"x": 100, "y": 0},
                    },
                    {
                        "id": "compose",
                        "kind": "data",
                        "category": "data",
                        "label": "Compose",
                        "provider_id": "core.context_builder",
                        "provider_label": "Context Builder",
                        "config": {"mode": "context_builder", "template": "", "input_bindings": [], "joiner": "\n\n"},
                        "position": {"x": 220, "y": 0},
                    },
                    {
                        "id": "finish",
                        "kind": "output",
                        "category": "end",
                        "label": "Finish",
                        "provider_id": "core.output",
                        "provider_label": "Core Output Node",
                        "config": {"source_binding": {"type": "latest_payload", "source": "compose"}},
                        "position": {"x": 340, "y": 0},
                    },
                ],
                "edges": [
                    {"id": "e1", "source_id": "start", "target_id": "sheet", "label": "", "kind": "standard", "priority": 100},
                    {"id": "e2", "source_id": "sheet", "target_id": "compose", "label": "", "kind": "standard", "priority": 100},
                    {"id": "e3", "source_id": "compose", "target_id": "finish", "label": "", "kind": "standard", "priority": 100},
                ],
            }
            graph = GraphDefinition.from_dict(graph_payload)
            graph.validate_against_services(services)

            state = runtime.run(graph, {"request": "Process spreadsheet rows"}, run_id="spreadsheet-row-context-builder")

        self.assertEqual(state.status, "completed")
        self.assertIsInstance(state.final_output, str)
        assert isinstance(state.final_output, str)
        self.assertIn("Spreadsheet record 1", state.final_output)
        self.assertIn("Company: Scale AI", state.final_output)
        self.assertIn("CEO: Alexandr Wang", state.final_output)
        self.assertIn("Summer_2026_Internships?: YES", state.final_output)
        self.assertNotIn('"row_data"', state.final_output)

    def test_run_state_reducer_tracks_iterator_updates(self) -> None:
        state = build_run_state("run-iterator", "graph-1", None, execution_node_ids=["sheet"])
        next_state = apply_single_run_event(
            state,
            {
                "event_type": "node.iterator.updated",
                "summary": "Iterator updated.",
                "payload": {
                    "node_id": "sheet",
                    "iterator_type": "spreadsheet_rows",
                    "status": "running",
                    "current_row_index": 1,
                    "total_rows": 3,
                    "headers": ["city", "temperature"],
                    "sheet_name": "Sheet1",
                    "source_file": "/tmp/test.csv",
                    "file_format": "csv",
                },
                "run_id": "run-iterator",
                "timestamp": "2026-04-02T00:00:00Z",
            },
        )
        self.assertEqual(next_state["iterator_states"]["sheet"]["current_row_index"], 1)
        self.assertEqual(next_state["iterator_states"]["sheet"]["total_rows"], 3)

    def test_runtime_state_snapshot_normalizes_iterator_states(self) -> None:
        normalized = normalize_runtime_state_snapshot(
            {
                "run_id": "run-1",
                "graph_id": "graph-1",
                "event_history": [],
                "node_statuses": {},
                "iterator_states": {
                    "sheet": {
                        "iterator_type": "spreadsheet_rows",
                        "status": "completed",
                        "current_row_index": 2,
                        "total_rows": 2,
                    }
                },
                "agent_runs": {},
            }
        )
        assert normalized is not None
        self.assertEqual(normalized["iterator_states"]["sheet"]["status"], "completed")


if __name__ == "__main__":
    unittest.main()
