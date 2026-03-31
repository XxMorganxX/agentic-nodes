from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
from threading import Lock, Thread
from typing import Any

from graph_agent.tools.base import ToolContext, ToolDefinition, ToolRegistry, ToolResult


def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _is_mapping(value: Any) -> bool:
    return isinstance(value, Mapping)


def _merge_env(overrides: Mapping[str, str]) -> dict[str, str]:
    env = dict(os.environ)
    for key, value in overrides.items():
        env[str(key)] = str(value)
    return env


def _stdio_error(message: str, *, details: Mapping[str, Any] | None = None) -> RuntimeError:
    suffix = f" Details: {json.dumps(details, sort_keys=True)}" if details else ""
    return RuntimeError(f"MCP server error: {message}{suffix}")


@dataclass(frozen=True)
class McpServerDefinition:
    server_id: str
    display_name: str
    description: str
    command: list[str]
    cwd: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    auto_boot: bool = False
    persistent: bool = True


@dataclass
class McpServerState:
    server_id: str
    display_name: str
    description: str
    auto_boot: bool = False
    persistent: bool = True
    running: bool = False
    tool_names: list[str] = field(default_factory=list)
    error: str = ""
    pid: int | None = None
    booted_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "server_id": self.server_id,
            "display_name": self.display_name,
            "description": self.description,
            "auto_boot": self.auto_boot,
            "persistent": self.persistent,
            "running": self.running,
            "tool_names": list(self.tool_names),
            "error": self.error,
            "pid": self.pid,
            "booted_at": self.booted_at,
        }


class _McpStdioSession:
    def __init__(self, definition: McpServerDefinition) -> None:
        self.definition = definition
        self._lock = Lock()
        self._request_id = 0
        self._stderr_lines: deque[str] = deque(maxlen=20)
        self._stderr_thread: Thread | None = None
        self._process: subprocess.Popen[bytes] | None = None

    @property
    def pid(self) -> int | None:
        return self._process.pid if self._process is not None else None

    def start(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        command = list(self.definition.command)
        if not command:
            raise RuntimeError(f"MCP server '{self.definition.server_id}' is missing a launch command.")

        cwd = self.definition.cwd
        if cwd:
            Path(cwd).mkdir(parents=True, exist_ok=True)
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            env=_merge_env(self.definition.env),
        )
        self._stderr_thread = Thread(target=self._consume_stderr, daemon=True)
        self._stderr_thread.start()
        self.request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "graph-agent", "version": "0.1.0"},
            },
        )
        self.notify("notifications/initialized", {})

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        try:
            if process.poll() is None and process.stdin is not None:
                try:
                    self.notify("shutdown", {})
                except RuntimeError:
                    pass
                process.terminate()
                process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
        finally:
            if process.stdin is not None:
                process.stdin.close()
            if process.stdout is not None:
                process.stdout.close()
            if process.stderr is not None:
                process.stderr.close()

    def list_tools(self) -> list[dict[str, Any]]:
        response = self.request("tools/list", {})
        tools = response.get("tools", [])
        if not isinstance(tools, list):
            raise _stdio_error("Invalid tools/list response.", details={"response": response})
        normalized: list[dict[str, Any]] = []
        for tool in tools:
            if not _is_mapping(tool):
                continue
            name = tool.get("name")
            description = tool.get("description", "")
            input_schema = tool.get("inputSchema") or tool.get("input_schema") or {}
            if not isinstance(name, str) or not name.strip() or not _is_mapping(input_schema):
                continue
            normalized.append(
                {
                    "name": name,
                    "description": str(description or ""),
                    "input_schema": dict(input_schema),
                }
            )
        return normalized

    def call_tool(self, tool_name: str, arguments: Mapping[str, Any]) -> ToolResult:
        response = self.request("tools/call", {"name": tool_name, "arguments": dict(arguments)})
        if bool(response.get("isError")):
            details = response.get("structuredContent")
            error = details if isinstance(details, dict) else {"message": "MCP tool execution failed."}
            return ToolResult(status="error", error=error, summary=f"MCP tool '{tool_name}' failed.")

        structured = response.get("structuredContent")
        if structured is not None:
            output = structured
        else:
            content = response.get("content", [])
            output = response if not isinstance(content, list) else _normalize_content_blocks(content)
        return ToolResult(status="success", output=output, summary=f"MCP tool '{tool_name}' completed.")

    def notify(self, method: str, params: Mapping[str, Any]) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": dict(params)})

    def request(self, method: str, params: Mapping[str, Any]) -> dict[str, Any]:
        with self._lock:
            process = self._require_process()
            self._request_id += 1
            request_id = self._request_id
            self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": dict(params)})
            while True:
                message = self._read_message(process)
                if "id" not in message:
                    continue
                if message.get("id") != request_id:
                    continue
                if message.get("error"):
                    error = message["error"]
                    if _is_mapping(error):
                        raise _stdio_error(str(error.get("message", "Unknown error")), details=error)
                    raise _stdio_error(str(error))
                result = message.get("result", {})
                if not _is_mapping(result):
                    raise _stdio_error("Invalid response payload.", details={"result": result})
                return dict(result)

    def last_stderr(self) -> str:
        return "\n".join(self._stderr_lines)

    def _require_process(self) -> subprocess.Popen[bytes]:
        process = self._process
        if process is None or process.poll() is not None:
            detail = self.last_stderr().strip()
            suffix = f" Stderr: {detail}" if detail else ""
            raise RuntimeError(f"MCP server '{self.definition.server_id}' is not running.{suffix}")
        return process

    def _consume_stderr(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return
        while True:
            line = process.stderr.readline()
            if not line:
                break
            self._stderr_lines.append(line.decode("utf-8", errors="replace").strip())

    def _send(self, payload: Mapping[str, Any]) -> None:
        process = self._require_process()
        if process.stdin is None:
            raise RuntimeError(f"MCP server '{self.definition.server_id}' stdin is not available.")
        body = json.dumps(dict(payload), separators=(",", ":")).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
        process.stdin.write(header + body)
        process.stdin.flush()

    def _read_message(self, process: subprocess.Popen[bytes]) -> dict[str, Any]:
        if process.stdout is None:
            raise RuntimeError(f"MCP server '{self.definition.server_id}' stdout is not available.")
        headers: dict[str, str] = {}
        while True:
            line = process.stdout.readline()
            if not line:
                detail = self.last_stderr().strip()
                suffix = f" Stderr: {detail}" if detail else ""
                raise RuntimeError(f"MCP server '{self.definition.server_id}' closed its stdout stream.{suffix}")
            if line in {b"\r\n", b"\n"}:
                break
            key, _, value = line.decode("ascii", errors="replace").partition(":")
            headers[key.strip().lower()] = value.strip()
        content_length = int(headers.get("content-length", "0") or "0")
        if content_length <= 0:
            raise _stdio_error("Missing Content-Length header.", details=headers)
        payload = process.stdout.read(content_length)
        if len(payload) != content_length:
            raise RuntimeError(f"MCP server '{self.definition.server_id}' returned a truncated payload.")
        message = json.loads(payload.decode("utf-8"))
        if not _is_mapping(message):
            raise _stdio_error("Invalid JSON-RPC message.", details={"message": message})
        return dict(message)


def _normalize_content_blocks(content: list[Any]) -> Any:
    text_parts: list[str] = []
    json_parts: list[Any] = []
    for block in content:
        if not _is_mapping(block):
            continue
        block_type = block.get("type")
        if block_type == "text" and isinstance(block.get("text"), str):
            text_parts.append(block["text"])
        elif block_type == "json" and "json" in block:
            json_parts.append(block["json"])
    if json_parts:
        return json_parts[0] if len(json_parts) == 1 else json_parts
    return "\n".join(part for part in text_parts if part)


class McpServerManager:
    def __init__(self, tool_registry: ToolRegistry) -> None:
        self._tool_registry = tool_registry
        self._definitions: dict[str, McpServerDefinition] = {}
        self._states: dict[str, McpServerState] = {}
        self._sessions: dict[str, _McpStdioSession] = {}
        self._lock = Lock()

    def register_server(self, definition: McpServerDefinition) -> None:
        if definition.server_id in self._definitions:
            raise ValueError(f"MCP server '{definition.server_id}' is already registered.")
        self._definitions[definition.server_id] = definition
        self._states[definition.server_id] = McpServerState(
            server_id=definition.server_id,
            display_name=definition.display_name,
            description=definition.description,
            auto_boot=definition.auto_boot,
            persistent=definition.persistent,
        )

    def register_tool(self, tool: ToolDefinition) -> None:
        if tool.source_type != "mcp" or not tool.server_id:
            raise ValueError("MCP-managed tools must declare source_type='mcp' and a server_id.")
        self._tool_registry.register(tool)
        state = self._states.get(tool.server_id)
        if state is not None and tool.name not in state.tool_names:
            state.tool_names.append(tool.name)
            state.tool_names.sort()

    def list_servers(self) -> list[dict[str, Any]]:
        return [self._states[server_id].to_dict() for server_id in sorted(self._states)]

    def get_server(self, server_id: str) -> dict[str, Any]:
        if server_id not in self._states:
            raise KeyError(server_id)
        return self._states[server_id].to_dict()

    def start_auto_boot(self) -> None:
        for definition in self._definitions.values():
            if definition.auto_boot:
                self.boot_server(definition.server_id)

    def boot_server(self, server_id: str) -> dict[str, Any]:
        with self._lock:
            definition = self._definitions.get(server_id)
            if definition is None:
                raise KeyError(server_id)
            state = self._states[server_id]
            session = self._sessions.get(server_id)
            if session is None:
                session = _McpStdioSession(definition)
                self._sessions[server_id] = session
            try:
                session.start()
                self._sync_server_tools(server_id, session.list_tools())
                state.running = True
                state.pid = session.pid
                state.error = ""
                state.booted_at = state.booted_at or _utc_now_iso()
            except Exception as exc:
                state.running = False
                state.pid = None
                state.error = str(exc)
                self._tool_registry.mark_server_tools_unavailable(server_id, str(exc))
                session.close()
                self._sessions.pop(server_id, None)
                raise
            state.tool_names = self._tool_registry.list_server_tool_names(server_id)
            return state.to_dict()

    def stop_server(self, server_id: str) -> dict[str, Any]:
        with self._lock:
            if server_id not in self._states:
                raise KeyError(server_id)
            state = self._states[server_id]
            session = self._sessions.pop(server_id, None)
            if session is not None:
                session.close()
            state.running = False
            state.pid = None
            self._tool_registry.mark_server_tools_unavailable(server_id, "MCP server is offline.")
            return state.to_dict()

    def refresh_server(self, server_id: str) -> dict[str, Any]:
        with self._lock:
            if server_id not in self._states:
                raise KeyError(server_id)
            session = self._sessions.get(server_id)
            if session is None:
                raise RuntimeError(f"MCP server '{server_id}' is not running.")
            state = self._states[server_id]
            self._sync_server_tools(server_id, session.list_tools())
            state.running = True
            state.pid = session.pid
            state.error = ""
            state.tool_names = self._tool_registry.list_server_tool_names(server_id)
            return state.to_dict()

    def shutdown_all(self) -> None:
        for server_id in list(self._states):
            self.stop_server(server_id)

    def set_tool_enabled(self, tool_name: str, enabled: bool) -> dict[str, Any]:
        tool = self._tool_registry.set_tool_enabled(tool_name, enabled)
        return tool.to_dict()

    def _sync_server_tools(self, server_id: str, discovered_tools: Sequence[Mapping[str, Any]]) -> None:
        discovered_names: set[str] = set()
        for discovered in discovered_tools:
            name = str(discovered.get("name", "")).strip()
            if not name:
                continue
            discovered_names.add(name)
            existing = self._tool_registry.get_optional(name)
            schema_warning = ""
            if existing is not None and (
                dict(existing.input_schema) != dict(discovered.get("input_schema", {}))
                or str(existing.description) != str(discovered.get("description", ""))
            ):
                schema_warning = "Live MCP schema differs from the preregistered tool metadata."
            self._tool_registry.upsert(
                ToolDefinition(
                    name=name,
                    description=str(discovered.get("description", "")),
                    input_schema=dict(discovered.get("input_schema", {})),
                    executor=self._executor_for(server_id, name),
                    source_type="mcp",
                    server_id=server_id,
                    enabled=existing.enabled if existing is not None else False,
                    available=True,
                    availability_error="",
                    schema_origin="discovered",
                    schema_warning=schema_warning,
                    managed=True,
                )
            )
        for tool_name in self._tool_registry.list_server_tool_names(server_id):
            if tool_name not in discovered_names:
                self._tool_registry.mark_tool_unavailable(tool_name, "Tool was not reported by the running MCP server.")

    def _executor_for(self, server_id: str, tool_name: str):
        def _execute(payload: Mapping[str, Any], _context: ToolContext) -> ToolResult:
            session = self._sessions.get(server_id)
            if session is None:
                return ToolResult(
                    status="unavailable",
                    error={"message": f"MCP server '{server_id}' is not running."},
                    summary=f"MCP tool '{tool_name}' is unavailable.",
                )
            return session.call_tool(tool_name, payload)

        return _execute
