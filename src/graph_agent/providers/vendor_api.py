from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from typing import Any

from graph_agent.providers.base import ModelMessage, ModelProvider, ModelRequest, ModelResponse


def _is_mapping(value: Any) -> bool:
    return isinstance(value, Mapping)


def _string_config(config: Mapping[str, Any], key: str, default: str) -> str:
    value = config.get(key, default)
    return value if isinstance(value, str) and value.strip() else default


def _number_config(config: Mapping[str, Any], key: str) -> float | int | None:
    value = config.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    return None


class VendorAPIModelProvider(ModelProvider):
    name = "vendor"
    api_key_env_var = ""
    default_endpoint = ""
    default_model = ""

    def generate(self, request: ModelRequest) -> ModelResponse:
        started_at = time.perf_counter()
        provider_config = self._provider_config(request)
        response_schema = self._resolve_response_schema(request)
        payload = self._build_payload(request, provider_config, response_schema)
        response_payload = self._post_json(
            url=_string_config(provider_config, "api_base", self.default_endpoint),
            payload=payload,
            headers=self._headers(provider_config),
            timeout_seconds=float(_number_config(provider_config, "timeout_seconds") or 60),
        )
        latency_ms = int((time.perf_counter() - started_at) * 1000)
        return self._parse_response(request, response_payload, response_schema, latency_ms)

    def _provider_config(self, request: ModelRequest) -> Mapping[str, Any]:
        return request.provider_config if _is_mapping(request.provider_config) else {}

    def _headers(self, provider_config: Mapping[str, Any]) -> dict[str, str]:
        api_key_env_var = _string_config(provider_config, "api_key_env_var", self.api_key_env_var)
        api_key = os.environ.get(api_key_env_var, "").strip()
        if not api_key:
            raise RuntimeError(
                f"{self.name} provider requires the '{api_key_env_var}' environment variable to be set."
            )
        return self._build_headers(api_key)

    def _resolve_response_schema(self, request: ModelRequest) -> Mapping[str, Any] | None:
        if _is_mapping(request.response_schema):
            return request.response_schema

        if str(request.metadata.get("response_mode", "message")) != "tool_call":
            return None

        available_tools = request.metadata.get("available_tools", [])
        if not isinstance(available_tools, list):
            return None

        preferred_name = request.metadata.get("preferred_tool_name")
        preferred_tool = None
        if isinstance(preferred_name, str) and preferred_name:
            preferred_tool = next(
                (
                    tool
                    for tool in available_tools
                    if _is_mapping(tool) and tool.get("name") == preferred_name and _is_mapping(tool.get("input_schema"))
                ),
                None,
            )
        if preferred_tool is None:
            preferred_tool = next(
                (tool for tool in available_tools if _is_mapping(tool) and _is_mapping(tool.get("input_schema"))),
                None,
            )
        if preferred_tool is None:
            return None
        return preferred_tool["input_schema"]

    def _message_dicts(self, messages: list[ModelMessage]) -> list[dict[str, str]]:
        return [{"role": message.role, "content": message.content} for message in messages]

    def _build_payload(
        self,
        request: ModelRequest,
        provider_config: Mapping[str, Any],
        response_schema: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def _build_headers(self, api_key: str) -> dict[str, str]:
        raise NotImplementedError

    def _parse_response(
        self,
        request: ModelRequest,
        payload: Mapping[str, Any],
        response_schema: Mapping[str, Any] | None,
        latency_ms: int,
    ) -> ModelResponse:
        raise NotImplementedError

    def _post_json(
        self,
        url: str,
        payload: Mapping[str, Any],
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> Mapping[str, Any]:
        request = urllib.request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", **dict(headers)},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{self.name} provider request failed with {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"{self.name} provider request failed: {exc.reason}") from exc


class OpenAIChatModelProvider(VendorAPIModelProvider):
    name = "openai"
    api_key_env_var = "OPENAI_API_KEY"
    default_endpoint = "https://api.openai.com/v1/chat/completions"
    default_model = "gpt-4.1-mini"

    def _build_headers(self, api_key: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {api_key}"}

    def _build_payload(
        self,
        request: ModelRequest,
        provider_config: Mapping[str, Any],
        response_schema: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": _string_config(provider_config, "model", self.default_model),
            "messages": self._message_dicts(request.messages),
        }
        temperature = _number_config(provider_config, "temperature")
        max_tokens = _number_config(provider_config, "max_tokens")
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)
        if response_schema is not None:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": "emit_structured_output",
                        "description": "Return the structured payload for this graph node.",
                        "parameters": dict(response_schema),
                    },
                }
            ]
            payload["tool_choice"] = {
                "type": "function",
                "function": {"name": "emit_structured_output"},
            }
        return payload

    def _parse_response(
        self,
        request: ModelRequest,
        payload: Mapping[str, Any],
        response_schema: Mapping[str, Any] | None,
        latency_ms: int,
    ) -> ModelResponse:
        choices = payload.get("choices", [])
        first_choice = choices[0] if isinstance(choices, list) and choices else {}
        if not _is_mapping(first_choice):
            raise RuntimeError("openai provider returned an unexpected response shape.")

        message = first_choice.get("message", {})
        if not _is_mapping(message):
            raise RuntimeError("openai provider did not return a message payload.")

        content = message.get("content", "")
        if isinstance(content, list):
            content_text = "".join(
                block.get("text", "")
                for block in content
                if _is_mapping(block) and block.get("type") in {"text", "output_text"}
            )
        else:
            content_text = content if isinstance(content, str) else ""

        structured_output = None
        tool_calls = message.get("tool_calls", [])
        if isinstance(tool_calls, list) and tool_calls:
            first_tool_call = tool_calls[0]
            if _is_mapping(first_tool_call):
                function_call = first_tool_call.get("function", {})
                if _is_mapping(function_call):
                    arguments = function_call.get("arguments", "{}")
                    if isinstance(arguments, str):
                        structured_output = json.loads(arguments)
        elif response_schema is not None:
            structured_output = json.loads(content_text)

        return ModelResponse(
            content=content_text,
            structured_output=structured_output,
            metadata={
                "latency_ms": latency_ms,
                "vendor_model": payload.get("model"),
                "finish_reason": first_choice.get("finish_reason"),
                "usage": payload.get("usage"),
            },
        )


class ClaudeMessagesModelProvider(VendorAPIModelProvider):
    name = "claude"
    api_key_env_var = "ANTHROPIC_API_KEY"
    default_endpoint = "https://api.anthropic.com/v1/messages"
    default_model = "claude-3-5-haiku-latest"

    def _build_headers(self, api_key: str) -> dict[str, str]:
        return {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }

    def _build_payload(
        self,
        request: ModelRequest,
        provider_config: Mapping[str, Any],
        response_schema: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        system_messages = [message.content for message in request.messages if message.role == "system" and message.content]
        conversation = [
            {"role": message.role, "content": message.content}
            for message in request.messages
            if message.role in {"user", "assistant"} and message.content
        ]
        if not conversation:
            conversation = [{"role": "user", "content": ""}]

        payload: dict[str, Any] = {
            "model": _string_config(provider_config, "model", self.default_model),
            "messages": conversation,
            "max_tokens": int(_number_config(provider_config, "max_tokens") or 1024),
        }
        if system_messages:
            payload["system"] = "\n\n".join(system_messages)
        temperature = _number_config(provider_config, "temperature")
        if temperature is not None:
            payload["temperature"] = temperature
        if response_schema is not None:
            payload["tools"] = [
                {
                    "name": "emit_structured_output",
                    "description": "Return the structured payload for this graph node.",
                    "input_schema": dict(response_schema),
                }
            ]
            payload["tool_choice"] = {"type": "tool", "name": "emit_structured_output"}
        return payload

    def _parse_response(
        self,
        request: ModelRequest,
        payload: Mapping[str, Any],
        response_schema: Mapping[str, Any] | None,
        latency_ms: int,
    ) -> ModelResponse:
        blocks = payload.get("content", [])
        if not isinstance(blocks, list):
            raise RuntimeError("claude provider returned an unexpected response shape.")

        text_parts: list[str] = []
        structured_output = None
        for block in blocks:
            if not _is_mapping(block):
                continue
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                text_parts.append(block["text"])
            if block.get("type") == "tool_use" and _is_mapping(block.get("input")):
                structured_output = block["input"]

        content_text = "\n".join(part for part in text_parts if part)
        if structured_output is None and response_schema is not None and content_text.strip():
            structured_output = json.loads(content_text)

        return ModelResponse(
            content=content_text,
            structured_output=structured_output,
            metadata={
                "latency_ms": latency_ms,
                "vendor_model": payload.get("model"),
                "stop_reason": payload.get("stop_reason"),
                "usage": payload.get("usage"),
            },
        )
