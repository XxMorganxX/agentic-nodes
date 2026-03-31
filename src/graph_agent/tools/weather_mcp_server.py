from __future__ import annotations

import json
from typing import Any
from urllib.parse import quote
import urllib.error
import urllib.request
import os
import sys


TOOLS = [
    {
        "name": "weather_current",
        "description": "Fetch the current weather conditions for a city or location string.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
            },
            "required": ["location"],
        },
    }
]


def _send(payload: dict[str, Any]) -> None:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    sys.stdout.buffer.write(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii"))
    sys.stdout.buffer.write(body)
    sys.stdout.buffer.flush()


def _read() -> dict[str, Any] | None:
    headers: dict[str, str] = {}
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        if line in {b"\r\n", b"\n"}:
            break
        key, _, value = line.decode("ascii", errors="replace").partition(":")
        headers[key.strip().lower()] = value.strip()
    content_length = int(headers.get("content-length", "0") or "0")
    if content_length <= 0:
        return None
    payload = sys.stdin.buffer.read(content_length)
    if len(payload) != content_length:
        return None
    message = json.loads(payload.decode("utf-8"))
    return message if isinstance(message, dict) else None


def _response(message_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": message_id, "result": result}


def _error(message_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": message_id, "error": {"code": code, "message": message}}


def _base_url() -> str:
    configured = os.environ.get("GRAPH_AGENT_WEATHER_API_BASE", "").strip()
    return configured.rstrip("/") if configured else "https://wttr.in"


def _extract_description(current: dict[str, Any]) -> str:
    weather_desc = current.get("weatherDesc", [])
    if isinstance(weather_desc, list) and weather_desc:
        first = weather_desc[0]
        if isinstance(first, dict) and isinstance(first.get("value"), str):
            return first["value"]
    return "Unknown"


def _fetch_weather(location: str) -> dict[str, Any]:
    url = f"{_base_url()}/{quote(location)}?format=j1"
    request = urllib.request.Request(url, headers={"User-Agent": "graph-agent-weather-mcp/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Weather lookup failed with {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Weather lookup failed: {exc.reason}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Weather API returned an unexpected payload.")
    current_list = payload.get("current_condition", [])
    nearest = payload.get("nearest_area", [])
    if not isinstance(current_list, list) or not current_list:
        raise RuntimeError("Weather API returned no current conditions.")
    current = current_list[0]
    if not isinstance(current, dict):
        raise RuntimeError("Weather API returned malformed current conditions.")
    resolved_area = location
    if isinstance(nearest, list) and nearest:
        first_area = nearest[0]
        if isinstance(first_area, dict):
            area_names = first_area.get("areaName", [])
            if isinstance(area_names, list) and area_names:
                first_name = area_names[0]
                if isinstance(first_name, dict) and isinstance(first_name.get("value"), str):
                    resolved_area = first_name["value"]
    description = _extract_description(current)
    return {
        "location": location,
        "resolved_location": resolved_area,
        "temperature_c": current.get("temp_C"),
        "temperature_f": current.get("temp_F"),
        "feels_like_c": current.get("FeelsLikeC"),
        "feels_like_f": current.get("FeelsLikeF"),
        "condition": description,
        "humidity": current.get("humidity"),
        "wind_kmph": current.get("windspeedKmph"),
        "wind_direction": current.get("winddir16Point"),
        "observation_time": current.get("observation_time"),
        "source": url,
    }


def _call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if name != "weather_current":
        return {
            "content": [{"type": "text", "text": f"Unknown tool '{name}'."}],
            "structuredContent": {"message": f"Unknown tool '{name}'."},
            "isError": True,
        }
    location = arguments.get("location")
    if not isinstance(location, str) or not location.strip():
        return {
            "content": [{"type": "text", "text": "The 'location' field is required."}],
            "structuredContent": {"message": "The 'location' field is required."},
            "isError": True,
        }
    try:
        forecast = _fetch_weather(location.strip())
    except RuntimeError as exc:
        return {
            "content": [{"type": "text", "text": str(exc)}],
            "structuredContent": {"message": str(exc), "location": location.strip()},
            "isError": True,
        }
    summary = (
        f"{forecast['resolved_location']}: {forecast['condition']}, "
        f"{forecast['temperature_c']}C ({forecast['temperature_f']}F)"
    )
    return {
        "content": [{"type": "text", "text": summary}],
        "structuredContent": forecast,
        "isError": False,
    }


def main() -> int:
    while True:
        message = _read()
        if message is None:
            return 0
        message_id = message.get("id")
        method = message.get("method")
        params = message.get("params", {})
        if not isinstance(method, str):
            if message_id is not None:
                _send(_error(message_id, -32600, "Invalid request."))
            continue
        if method == "initialize":
            if message_id is not None:
                _send(
                    _response(
                        message_id,
                        {
                            "protocolVersion": "2024-11-05",
                            "capabilities": {"tools": {"listChanged": False}},
                            "serverInfo": {"name": "weather-mcp-server", "version": "0.1.0"},
                        },
                    )
                )
            continue
        if method == "notifications/initialized":
            continue
        if method == "shutdown":
            return 0
        if method == "tools/list":
            if message_id is not None:
                _send(_response(message_id, {"tools": TOOLS}))
            continue
        if method == "tools/call":
            if not isinstance(params, dict):
                if message_id is not None:
                    _send(_error(message_id, -32602, "Invalid params."))
                continue
            name = params.get("name")
            arguments = params.get("arguments", {})
            if not isinstance(name, str) or not isinstance(arguments, dict):
                if message_id is not None:
                    _send(_error(message_id, -32602, "Invalid tool call params."))
                continue
            if message_id is not None:
                _send(_response(message_id, _call_tool(name, arguments)))
            continue
        if message_id is not None:
            _send(_error(message_id, -32601, f"Method '{method}' not found."))


if __name__ == "__main__":
    raise SystemExit(main())
