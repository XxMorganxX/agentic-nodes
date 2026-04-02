from __future__ import annotations

import json
from queue import Empty
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict

from graph_agent.api.manager import GraphRunManager


class RunRequest(BaseModel):
    input: Any
    agent_ids: Optional[list[str]] = None


class ProviderPreflightRequest(BaseModel):
    provider_name: str
    provider_config: Optional[dict[str, Any]] = None
    live: bool = False


class GraphPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    graph_id: str
    name: str
    description: str = ""
    version: str = "1.0"
    graph_type: Optional[str] = None
    default_input: Optional[str] = None
    env_vars: Optional[dict[str, str]] = None
    start_node_id: Optional[str] = None
    nodes: Optional[list[dict[str, Any]]] = None
    edges: Optional[list[dict[str, Any]]] = None
    agents: Optional[list[dict[str, Any]]] = None


class ToolToggleRequest(BaseModel):
    enabled: bool


class McpServerPayload(BaseModel):
    server_id: str
    display_name: str
    description: str = ""
    transport: str = "stdio"
    command: list[str] = []
    cwd: Optional[str] = None
    env: Optional[dict[str, str]] = None
    base_url: Optional[str] = None
    timeout_seconds: int = 15
    auto_boot: bool = False
    persistent: bool = True


app = FastAPI(title="Graph Agent API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1):\d+$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

manager = GraphRunManager()


@app.on_event("startup")
def startup_event() -> None:
    manager.start_background_services()


@app.on_event("shutdown")
def shutdown_event() -> None:
    manager.stop_background_services()


@app.get("/api/graphs")
def list_graphs() -> dict[str, Any]:
    return {"graphs": manager.list_graphs()}


@app.get("/api/graphs/{graph_id}")
def get_graph(graph_id: str) -> dict[str, Any]:
    try:
        return manager.get_graph(graph_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown graph '{graph_id}'.") from exc


@app.post("/api/graphs")
def create_graph(graph: GraphPayload) -> dict[str, Any]:
    try:
        return manager.create_graph(graph.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.put("/api/graphs/{graph_id}")
def update_graph(graph_id: str, graph: GraphPayload) -> dict[str, Any]:
    try:
        return manager.update_graph(graph_id, graph.model_dump())
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown graph '{graph_id}'.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/api/graphs/{graph_id}")
def delete_graph(graph_id: str) -> dict[str, str]:
    try:
        manager.delete_graph(graph_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown graph '{graph_id}'.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"deleted": graph_id}


@app.get("/api/editor/catalog")
def get_editor_catalog() -> dict[str, Any]:
    return manager.get_catalog()


@app.post("/api/editor/providers/preflight")
def preflight_provider(request: ProviderPreflightRequest) -> dict[str, Any]:
    try:
        return manager.preflight_provider(
            request.provider_name,
            request.provider_config,
            live=request.live,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown provider '{request.provider_name}'.") from exc


@app.post("/api/editor/providers/diagnostics")
def provider_diagnostics(request: ProviderPreflightRequest) -> dict[str, Any]:
    try:
        return manager.provider_diagnostics(
            request.provider_name,
            request.provider_config,
            live=request.live,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown provider '{request.provider_name}'.") from exc


@app.post("/api/editor/mcp/servers/{server_id}/boot")
def boot_mcp_server(server_id: str) -> dict[str, Any]:
    try:
        return manager.boot_mcp_server(server_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown MCP server '{server_id}'.") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/editor/mcp/servers/{server_id}/stop")
def stop_mcp_server(server_id: str) -> dict[str, Any]:
    try:
        return manager.stop_mcp_server(server_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown MCP server '{server_id}'.") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/editor/mcp/servers/{server_id}/refresh")
def refresh_mcp_server(server_id: str) -> dict[str, Any]:
    try:
        return manager.refresh_mcp_server(server_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown MCP server '{server_id}'.") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/editor/mcp/servers")
def create_mcp_server(server: McpServerPayload) -> dict[str, Any]:
    try:
        return manager.create_mcp_server(server.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.put("/api/editor/mcp/servers/{server_id}")
def update_mcp_server(server_id: str, server: McpServerPayload) -> dict[str, Any]:
    try:
        return manager.update_mcp_server(server_id, server.model_dump())
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown MCP server '{server_id}'.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/api/editor/mcp/servers/{server_id}")
def delete_mcp_server(server_id: str) -> dict[str, str]:
    try:
        manager.delete_mcp_server(server_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown MCP server '{server_id}'.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"deleted": server_id}


@app.post("/api/editor/mcp/servers/test")
def test_mcp_server(server: McpServerPayload) -> dict[str, Any]:
    try:
        return manager.test_mcp_server(server.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/editor/mcp/tools/{tool_name}/toggle")
def toggle_mcp_tool(tool_name: str, request: ToolToggleRequest) -> dict[str, Any]:
    try:
        return manager.set_mcp_tool_enabled(tool_name, request.enabled)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown tool '{tool_name}'.") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/graphs/{graph_id}/runs")
def list_graph_runs(graph_id: str, limit: int = 50) -> dict[str, Any]:
    return {"runs": manager.list_runs(graph_id, limit=limit)}


@app.post("/api/graphs/{graph_id}/runs")
def start_run(graph_id: str, request: RunRequest) -> dict[str, str]:
    try:
        run_id = manager.start_run(graph_id, request.input, agent_ids=request.agent_ids)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown graph '{graph_id}'.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"run_id": run_id}


@app.post("/api/runtime/reset")
def reset_runtime() -> dict[str, Any]:
    return manager.reset_runtime()


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    try:
        return manager.get_run(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown run '{run_id}'.") from exc


@app.get("/api/runs/{run_id}/events")
def stream_run_events(run_id: str) -> StreamingResponse:
    try:
        backlog, queue = manager.subscribe(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown run '{run_id}'.") from exc

    def event_stream():
        try:
            for event in backlog:
                yield f"data: {json.dumps(event)}\n\n"

            while True:
                try:
                    item = queue.get(timeout=15)
                except Empty:
                    yield ": keep-alive\n\n"
                    continue

                if item is None:
                    break
                yield f"data: {item}\n\n"
        finally:
            manager.unsubscribe(run_id, queue)

    return StreamingResponse(event_stream(), media_type="text/event-stream")
