from __future__ import annotations

from abc import ABC, abstractmethod
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
import urllib.error
import urllib.request
from urllib.parse import urlparse

from graph_agent.tools.base import ToolContext, ToolDefinition, ToolRegistry, ToolResult


MCP_PROTOCOL_VERSION = "2024-11-05"
SUPPORTED_MCP_TRANSPORTS = {"stdio", "http"}
USER_MCP_SOURCE = "user"
BUILTIN_MCP_SOURCE = "builtin"


def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _is_mapping(value: Any) -> bool:
    return isinstance(value, Mapping)


def _normalize_string_map(value: Mapping[str, Any] | None) -> dict[str, str]:
    if not _is_mapping(value):
        return {}
    normalized: dict[str, str] = {}
    for key, item in value.items():
        normalized[str(key)] = str(item)
    return normalized


def _merge_env(overrides: Mapping[str, str]) -> dict[str, str]:
    env = dict(os.environ)
    for key, value in overrides.items():
        env[str(key)] = str(value)
    return env


def _mcp_error(message: str, *, details: Mapping[str, Any] | None = None) -> RuntimeError:
    suffix = f" Details: {json.dumps(details, sort_keys=True)}" if details else ""
    return RuntimeError(f"MCP server error: {message}{suffix}")


@dataclass(frozen=True)
class McpServerDefinition:
    server_id: str
    display_name: str
    description: str
    transport: str = "stdio"
    command: list[str] = field(default_factory=list)
    cwd: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    base_url: str | None = None
    timeout_seconds: int = 15
    auto_boot: bool = False
    persistent: bool = True
    source: str = USER_MCP_SOURCE

    def __post_init__(self) -> None:
        server_id = self.server_id.strip()
        display_name = self.display_name.strip()
        transport = self.transport.strip().lower()
        description = self.description.strip()
        source = self.source.strip().lower()
        command = [str(part).strip() for part in self.command if str(part).strip()]
        env = _normalize_string_map(self.env)
        base_url = str(self.base_url).strip() if self.base_url is not None else None
        cwd = str(self.cwd).strip() if self.cwd is not None and str(self.cwd).strip() else None

        object.__setattr__(self, "server_id", server_id)
        object.__setattr__(self, "display_name", display_name)
        object.__setattr__(self, "description", description)
        object.__setattr__(self, "transport", transport)
        object.__setattr__(self, "command", command)
        object.__setattr__(self, "env", env)
        object.__setattr__(self, "base_url", base_url)
        object.__setattr__(self, "cwd", cwd)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "timeout_seconds", max(1, int(self.timeout_seconds)))

        if not server_id:
            raise ValueError("MCP server definitions require a server_id.")
        if not display_name:
            raise ValueError(f"MCP server '{server_id}' requires a display_name.")
        if transport not in SUPPORTED_MCP_TRANSPORTS:
            raise ValueError(
                f"MCP server '{server_id}' uses unsupported transport '{transport}'. "
                f"Supported transports: {', '.join(sorted(SUPPORTED_MCP_TRANSPORTS))}."
            )
        if source not in {USER_MCP_SOURCE, BUILTIN_MCP_SOURCE}:
            raise ValueError(f"MCP server '{server_id}' has unsupported source '{source}'.")
        if transport == "stdio":
            if not command:
                raise ValueError(f"MCP stdio server '{server_id}' is missing a launch command.")
        if transport == "http":
            if not base_url:
                raise ValueError(f"MCP HTTP server '{server_id}' is missing a base_url.")
            parsed = urlparse(base_url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError(f"MCP HTTP server '{server_id}' must use a valid http or https URL.")

    @property
    def editable(self) -> bool:
        return self.source != BUILTIN_MCP_SOURCE

    def config_payload(self) -> dict[str, Any]:
        return {
            "command": list(self.command),
            "cwd": self.cwd,
            "env": dict(self.env),
            "base_url": self.base_url,
            "timeout_seconds": self.timeout_seconds,
        }

    def config_summary(self) -> str:
        if self.transport == "stdio":
            command = " ".join(self.command[:3]).strip()
            return command or "stdio subprocess"
        if self.transport == "http":
            return self.base_url or ""
        return self.transport

    def to_dict(self) -> dict[str, Any]:
        return {
            "server_id": self.server_id,
            "display_name": self.display_name,
            "description": self.description,
            "transport": self.transport,
            "auto_boot": self.auto_boot,
            "persistent": self.persistent,
            "source": self.source,
            "editable": self.editable,
            "config": self.config_payload(),
            "config_summary": self.config_summary(),
        }

    def to_persisted_dict(self) -> dict[str, Any]:
        return {
            "server_id": self.server_id,
            "display_name": self.display_name,
            "description": self.description,
            "transport": self.transport,
            "command": list(self.command),
            "cwd": self.cwd,
            "env": dict(self.env),
            "base_url": self.base_url,
            "timeout_seconds": self.timeout_seconds,
            "auto_boot": self.auto_boot,
            "persistent": self.persistent,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], *, source: str = USER_MCP_SOURCE) -> McpServerDefinition:
        if not _is_mapping(payload):
            raise ValueError("MCP server payload must be an object.")
        command = payload.get("command", [])
        env = payload.get("env", {})
        return cls(
            server_id=str(payload.get("server_id", "")).strip(),
            display_name=str(payload.get("display_name", "")).strip(),
            description=str(payload.get("description", "") or ""),
            transport=str(payload.get("transport", "stdio") or "stdio"),
            command=list(command) if isinstance(command, Sequence) and not isinstance(command, (str, bytes)) else [],
            cwd=(None if payload.get("cwd") in {"", None} else str(payload.get("cwd"))),
            env=_normalize_string_map(env if _is_mapping(env) else {}),
            base_url=(None if payload.get("base_url") in {"", None} else str(payload.get("base_url"))),
            timeout_seconds=int(payload.get("timeout_seconds", 15) or 15),
            auto_boot=bool(payload.get("auto_boot", False)),
            persistent=bool(payload.get("persistent", True)),
            source=source,
        )


@dataclass
class McpServerState:
    server_id: str
    display_name: str
    description: str
    transport: str
    auto_boot: bool = False
    persistent: bool = True
    source: str = USER_MCP_SOURCE
    config: dict[str, Any] = field(default_factory=dict)
    config_summary: str = ""
    running: bool = False
    tool_names: list[str] = field(default_factory=list)
    error: str = ""
    pid: int | None = None
    booted_at: str | None = None

    @classmethod
    def from_definition(cls, definition: McpServerDefinition) -> McpServerState:
        return cls(
            server_id=definition.server_id,
            display_name=definition.display_name,
            description=definition.description,
            transport=definition.transport,
            auto_boot=definition.auto_boot,
            persistent=definition.persistent,
            source=definition.source,
            config=definition.config_payload(),
            config_summary=definition.config_summary(),
        )

    def apply_definition(self, definition: McpServerDefinition) -> None:
        self.display_name = definition.display_name
        self.description = definition.description
        self.transport = definition.transport
        self.auto_boot = definition.auto_boot
        self.persistent = definition.persistent
        self.source = definition.source
        self.config = definition.config_payload()
        self.config_summary = definition.config_summary()

    def to_dict(self) -> dict[str, Any]:
        return {
            "server_id": self.server_id,
            "display_name": self.display_name,
            "description": self.description,
            "transport": self.transport,
            "auto_boot": self.auto_boot,
            "persistent": self.persistent,
            "source": self.source,
            "editable": self.source != BUILTIN_MCP_SOURCE,
            "config": dict(self.config),
            "config_summary": self.config_summary,
            "running": self.running,
            "tool_names": list(self.tool_names),
            "error": self.error,
            "pid": self.pid,
            "booted_at": self.booted_at,
        }


class _BaseMcpSession(ABC):
    def __init__(self, definition: McpServerDefinition) -> None:
        self.definition = definition
        self._lock = Lock()
        self._request_id = 0
        self._started = False

    @property
    @abstractmethod
    def pid(self) -> int | None:
        raise NotImplementedError

    def start(self) -> None:
        if self._started:
            return
        self._open()
        try:
            self.request(
                "initialize",
                {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "graph-agent", "version": "0.1.0"},
                },
            )
            try:
                self.notify("notifications/initialized", {})
            except RuntimeError:
                pass
        except Exception:
            self.close()
            raise
        self._started = True

    def close(self) -> None:
        self._started = False
        self._close()

    def list_tools(self) -> list[dict[str, Any]]:
        response = self.request("tools/list", {})
        tools = response.get("tools", [])
        if not isinstance(tools, list):
            raise _mcp_error("Invalid tools/list response.", details={"response": response})
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
        diagnostics_cursor = self._before_tool_call()
        response = self.request("tools/call", {"name": tool_name, "arguments": dict(arguments)})
        metadata = self._after_tool_call(tool_name, diagnostics_cursor)
        if bool(response.get("isError")):
            details = response.get("structuredContent")
            error = details if isinstance(details, dict) else {"message": "MCP tool execution failed."}
            return ToolResult(
                status="error",
                error=error,
                summary=f"MCP tool '{tool_name}' failed.",
                metadata=metadata,
            )
        structured = response.get("structuredContent")
        if structured is not None:
            output = structured
        else:
            content = response.get("content", [])
            output = response if not isinstance(content, list) else _normalize_content_blocks(content)
        return ToolResult(
            status="success",
            output=output,
            summary=f"MCP tool '{tool_name}' completed.",
            metadata=metadata,
        )

    def notify(self, method: str, params: Mapping[str, Any]) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": dict(params)}, expect_response=False)

    def request(self, method: str, params: Mapping[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._request_id += 1
            request_id = self._request_id
            response = self._send(
                {"jsonrpc": "2.0", "id": request_id, "method": method, "params": dict(params)},
                expect_response=True,
            )
        if "error" in response and response["error"]:
            error = response["error"]
            if _is_mapping(error):
                raise _mcp_error(str(error.get("message", "Unknown error")), details=error)
            raise _mcp_error(str(error))
        result = response.get("result", response)
        if not _is_mapping(result):
            raise _mcp_error("Invalid response payload.", details={"result": result})
        return dict(result)

    def _before_tool_call(self) -> Any:
        return None

    def _after_tool_call(self, _tool_name: str, _cursor: Any) -> dict[str, Any]:
        return {}

    @abstractmethod
    def _open(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _close(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _send(self, payload: Mapping[str, Any], *, expect_response: bool) -> dict[str, Any]:
        raise NotImplementedError


class _McpStdioSession(_BaseMcpSession):
    def __init__(self, definition: McpServerDefinition) -> None:
        super().__init__(definition)
        self._stderr_lock = Lock()
        self._stderr_lines: deque[str] = deque(maxlen=20)
        self._stderr_events: deque[tuple[int, str]] = deque(maxlen=200)
        self._stderr_cursor = 0
        self._stderr_thread: Thread | None = None
        self._process: subprocess.Popen[bytes] | None = None

    @property
    def pid(self) -> int | None:
        return self._process.pid if self._process is not None else None

    def _open(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        if not self.definition.command:
            raise RuntimeError(f"MCP server '{self.definition.server_id}' is missing a launch command.")
        if self.definition.cwd:
            Path(self.definition.cwd).mkdir(parents=True, exist_ok=True)
        self._process = subprocess.Popen(
            list(self.definition.command),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.definition.cwd,
            env=_merge_env(self.definition.env),
        )
        self._stderr_thread = Thread(target=self._consume_stderr, daemon=True)
        self._stderr_thread.start()

    def close(self) -> None:
        process = self._process
        if process is None:
            self._started = False
            return
        try:
            if process.poll() is None:
                try:
                    self.request("shutdown", {})
                except RuntimeError:
                    pass
                try:
                    self.notify("exit", {})
                except RuntimeError:
                    pass
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
        finally:
            self._started = False
            self._process = None
            if process.stdin is not None:
                process.stdin.close()
            if process.stdout is not None:
                process.stdout.close()
            if process.stderr is not None:
                process.stderr.close()

    def last_stderr(self) -> str:
        with self._stderr_lock:
            return "\n".join(self._stderr_lines)

    def stderr_cursor(self) -> int:
        with self._stderr_lock:
            return self._stderr_cursor

    def stderr_since(self, cursor: int) -> list[str]:
        with self._stderr_lock:
            return [line for event_cursor, line in self._stderr_events if event_cursor > cursor]

    def _before_tool_call(self) -> int:
        return self.stderr_cursor()

    def _after_tool_call(self, tool_name: str, cursor: int) -> dict[str, Any]:
        terminal_output = self._terminal_output_payload(tool_name, cursor)
        return {"terminal_output": terminal_output} if terminal_output else {}

    def _send(self, payload: Mapping[str, Any], *, expect_response: bool) -> dict[str, Any]:
        process = self._require_process()
        if process.stdin is None:
            raise RuntimeError(f"MCP server '{self.definition.server_id}' stdin is not available.")
        body = json.dumps(dict(payload), separators=(",", ":")).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
        process.stdin.write(header + body)
        process.stdin.flush()
        if not expect_response:
            return {}
        request_id = payload.get("id")
        while True:
            message = self._read_message(process)
            if "id" not in message:
                continue
            if message.get("id") != request_id:
                continue
            return message

    def _close(self) -> None:
        self.close()

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
            decoded_line = line.decode("utf-8", errors="replace").strip()
            with self._stderr_lock:
                self._stderr_cursor += 1
                self._stderr_lines.append(decoded_line)
                self._stderr_events.append((self._stderr_cursor, decoded_line))

    def _terminal_output_payload(self, tool_name: str, cursor: int) -> dict[str, Any] | None:
        stderr_lines = self.stderr_since(cursor)
        if not stderr_lines:
            return None
        return {
            "server_id": self.definition.server_id,
            "display_name": self.definition.display_name,
            "tool_name": tool_name,
            "pid": self.pid,
            "stderr": "\n".join(stderr_lines),
            "stderr_lines": stderr_lines,
        }

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
            raise _mcp_error("Missing Content-Length header.", details=headers)
        payload = process.stdout.read(content_length)
        if len(payload) != content_length:
            raise RuntimeError(f"MCP server '{self.definition.server_id}' returned a truncated payload.")
        message = json.loads(payload.decode("utf-8"))
        if not _is_mapping(message):
            raise _mcp_error("Invalid JSON-RPC message.", details={"message": message})
        return dict(message)


class _McpHttpSession(_BaseMcpSession):
    @property
    def pid(self) -> int | None:
        return None

    def _open(self) -> None:
        return None

    def _close(self) -> None:
        return None

    def _send(self, payload: Mapping[str, Any], *, expect_response: bool) -> dict[str, Any]:
        if not self.definition.base_url:
            raise RuntimeError(f"MCP HTTP server '{self.definition.server_id}' is missing a base_url.")
        body = json.dumps(dict(payload), separators=(",", ":")).encode("utf-8")
        request = urllib.request.Request(
            self.definition.base_url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "graph-agent-mcp/0.1",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.definition.timeout_seconds) as response:
                raw_payload = response.read().decode("utf-8", errors="replace").strip()
        except urllib.error.HTTPError as exc:
            body_text = exc.read().decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"MCP HTTP request failed with {exc.code}: {body_text}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"MCP HTTP request failed: {exc.reason}") from exc
        if not expect_response:
            return {}
        if not raw_payload:
            raise RuntimeError(f"MCP HTTP server '{self.definition.server_id}' returned an empty response.")
        decoded = json.loads(raw_payload)
        if not _is_mapping(decoded):
            raise _mcp_error("Invalid JSON-RPC message.", details={"message": decoded})
        return dict(decoded)


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
    def __init__(self, tool_registry: ToolRegistry, *, state_path: Path | None = None) -> None:
        self._tool_registry = tool_registry
        self._definitions: dict[str, McpServerDefinition] = {}
        self._states: dict[str, McpServerState] = {}
        self._sessions: dict[str, _BaseMcpSession] = {}
        self._lock = Lock()
        self._state_path = state_path or Path(__file__).resolve().parents[3] / ".graph-agent" / "mcp_servers_state.json"
        persisted_state = self._load_store_payload()
        desired_running = persisted_state.get("desired_running", {})
        self._desired_running = {
            str(server_id): bool(should_run)
            for server_id, should_run in desired_running.items()
            if isinstance(server_id, str)
        }
        persisted_servers = persisted_state.get("servers", [])
        self._persisted_user_servers: list[McpServerDefinition] = []
        for payload in persisted_servers:
            try:
                self._persisted_user_servers.append(McpServerDefinition.from_dict(payload, source=USER_MCP_SOURCE))
            except ValueError:
                continue

    def load_user_servers(self) -> None:
        for definition in self._persisted_user_servers:
            if definition.server_id in self._definitions:
                continue
            self.register_server(definition)

    def register_server(self, definition: McpServerDefinition) -> None:
        if definition.server_id in self._definitions:
            raise ValueError(f"MCP server '{definition.server_id}' is already registered.")
        self._definitions[definition.server_id] = definition
        self._states[definition.server_id] = McpServerState.from_definition(definition)
        if definition.source == USER_MCP_SOURCE:
            self._write_store()

    def create_server(self, definition: McpServerDefinition) -> dict[str, Any]:
        if definition.source != USER_MCP_SOURCE:
            raise ValueError("Only user-managed MCP servers can be created via the editor.")
        self.register_server(definition)
        return self.get_server(definition.server_id)

    def update_server(self, server_id: str, definition: McpServerDefinition) -> dict[str, Any]:
        with self._lock:
            if server_id not in self._definitions:
                raise KeyError(server_id)
            existing = self._definitions[server_id]
            if existing.source != USER_MCP_SOURCE:
                raise ValueError(f"Built-in MCP server '{server_id}' cannot be edited.")
            if server_id != definition.server_id:
                raise ValueError("MCP server IDs cannot be changed once created.")
            should_restart = server_id in self._sessions
            if should_restart:
                self.stop_server(server_id, preserve_desired_running=True)
            self._tool_registry.remove_server_tools(server_id)
            self._definitions[server_id] = definition
            state = self._states[server_id]
            state.apply_definition(definition)
            state.running = False
            state.pid = None
            state.error = ""
            state.tool_names = []
            self._write_store()
        if should_restart:
            return self.boot_server(server_id)
        return self.get_server(server_id)

    def delete_server(self, server_id: str) -> None:
        with self._lock:
            definition = self._definitions.get(server_id)
            if definition is None:
                raise KeyError(server_id)
            if definition.source != USER_MCP_SOURCE:
                raise ValueError(f"Built-in MCP server '{server_id}' cannot be deleted.")
        self.stop_server(server_id, preserve_desired_running=False)
        with self._lock:
            self._tool_registry.remove_server_tools(server_id)
            self._definitions.pop(server_id, None)
            self._states.pop(server_id, None)
            self._sessions.pop(server_id, None)
            self._desired_running.pop(server_id, None)
            self._write_store()

    def validate_server(self, definition: McpServerDefinition) -> dict[str, Any]:
        session = self._create_session(definition)
        try:
            session.start()
            tools = session.list_tools()
            return {
                "ok": True,
                "server": definition.to_dict(),
                "tool_names": [tool["name"] for tool in tools],
                "tools": tools,
                "message": f"Connected to MCP server '{definition.display_name}'.",
            }
        finally:
            session.close()

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
            if definition.auto_boot or (definition.persistent and self._desired_running.get(definition.server_id, False)):
                self.boot_server(definition.server_id)

    def boot_server(self, server_id: str) -> dict[str, Any]:
        with self._lock:
            definition = self._definitions.get(server_id)
            if definition is None:
                raise KeyError(server_id)
            state = self._states[server_id]
            session = self._sessions.get(server_id)
            if session is None:
                session = self._create_session(definition)
                self._sessions[server_id] = session
            try:
                session.start()
                self._sync_server_tools(server_id, session.list_tools())
                state.running = True
                state.pid = session.pid
                state.error = ""
                state.booted_at = state.booted_at or _utc_now_iso()
                self._set_desired_running(server_id, True)
            except Exception as exc:
                state.running = False
                state.pid = None
                state.error = str(exc)
                self._tool_registry.mark_server_tools_unavailable(server_id, str(exc))
                self._set_desired_running(server_id, False)
                session.close()
                self._sessions.pop(server_id, None)
                raise
            state.tool_names = self._tool_registry.list_server_tool_names(server_id)
            return state.to_dict()

    def stop_server(self, server_id: str, *, preserve_desired_running: bool = False) -> dict[str, Any]:
        with self._lock:
            if server_id not in self._states:
                raise KeyError(server_id)
            state = self._states[server_id]
            session = self._sessions.pop(server_id, None)
            if session is not None:
                session.close()
            state.running = False
            state.pid = None
            state.error = ""
            self._tool_registry.mark_server_tools_unavailable(server_id, "MCP server is offline.")
            if not preserve_desired_running:
                self._set_desired_running(server_id, False)
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

    def shutdown_all(self, *, preserve_desired_running: bool = False) -> None:
        for server_id in list(self._states):
            self.stop_server(server_id, preserve_desired_running=preserve_desired_running)

    def set_tool_enabled(self, tool_name: str, enabled: bool) -> dict[str, Any]:
        tool = self._tool_registry.set_tool_enabled(tool_name, enabled)
        return tool.to_dict()

    def _create_session(self, definition: McpServerDefinition) -> _BaseMcpSession:
        if definition.transport == "stdio":
            return _McpStdioSession(definition)
        if definition.transport == "http":
            return _McpHttpSession(definition)
        raise ValueError(f"Unsupported MCP transport '{definition.transport}'.")

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

    def _load_store_payload(self) -> dict[str, Any]:
        try:
            payload = json.loads(self._state_path.read_text())
        except FileNotFoundError:
            return {"desired_running": {}, "servers": []}
        except (OSError, json.JSONDecodeError):
            return {"desired_running": {}, "servers": []}
        if not _is_mapping(payload):
            return {"desired_running": {}, "servers": []}
        return {
            "desired_running": payload.get("desired_running", {}),
            "servers": payload.get("servers", []),
        }

    def _write_store(self) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "desired_running": {
                server_id: should_run
                for server_id, should_run in sorted(self._desired_running.items())
                if self._definitions.get(server_id, None) is None or self._definitions[server_id].persistent
            },
            "servers": [
                definition.to_persisted_dict()
                for definition in sorted(self._definitions.values(), key=lambda item: item.server_id)
                if definition.source == USER_MCP_SOURCE
            ],
        }
        self._state_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    def _set_desired_running(self, server_id: str, should_run: bool) -> None:
        definition = self._definitions.get(server_id)
        if definition is None or not definition.persistent:
            return
        self._desired_running[server_id] = should_run
        self._write_store()
