from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from typing import Any

from graph_agent.providers.base import (
    ModelMessage,
    ModelProvider,
    ModelRequest,
    ModelResponse,
    ModelToolCall,
    ModelToolDefinition,
    ProviderPreflightResult,
)


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
        tools = self._tool_definitions(request)
        response_schema = request.response_schema if _is_mapping(request.response_schema) else None
        payload = self._build_payload(request, provider_config, tools, response_schema)
        response_payload = self._post_json(
            url=_string_config(provider_config, "api_base", self.default_endpoint),
            payload=payload,
            headers=self._headers(provider_config),
            timeout_seconds=float(_number_config(provider_config, "timeout_seconds") or 60),
        )
        latency_ms = int((time.perf_counter() - started_at) * 1000)
        return self._parse_response(request, response_payload, tools, response_schema, latency_ms)

    def _provider_config(self, request: ModelRequest) -> Mapping[str, Any]:
        return request.provider_config if _is_mapping(request.provider_config) else {}

    def preflight(self, provider_config: Mapping[str, Any] | None = None) -> ProviderPreflightResult:
        config = provider_config if _is_mapping(provider_config) else {}
        api_key_env_var = _string_config(config, "api_key_env_var", self.api_key_env_var)
        if not api_key_env_var:
            return ProviderPreflightResult(
                status="missing_config",
                ok=False,
                message=f"{self.name} provider is missing an API key env var setting.",
            )
        api_key = os.environ.get(api_key_env_var, "").strip()
        if not api_key:
            return ProviderPreflightResult(
                status="missing_credentials",
                ok=False,
                message=f"Set the '{api_key_env_var}' environment variable to use the {self.name} provider.",
                warnings=self._preflight_warnings(config, api_key_present=False),
                details={"api_key_env_var": api_key_env_var},
            )
        return ProviderPreflightResult(
            status="available",
            ok=True,
            message=f"{self.name} provider credentials are available.",
            warnings=self._preflight_warnings(config, api_key_present=True),
            details={
                "backend_type": self._backend_type(),
                "auth_mode": self._auth_mode(),
                "api_key_env_var": api_key_env_var,
                "api_key_present": True,
                "api_base": _string_config(config, "api_base", self.default_endpoint),
            },
        )

    def _backend_type(self) -> str:
        return "vendor_api"

    def _auth_mode(self) -> str:
        return "api_key"

    def _preflight_warnings(self, provider_config: Mapping[str, Any], *, api_key_present: bool) -> list[str]:
        return []

    def _headers(self, provider_config: Mapping[str, Any]) -> dict[str, str]:
        api_key_env_var = _string_config(provider_config, "api_key_env_var", self.api_key_env_var)
        api_key = os.environ.get(api_key_env_var, "").strip()
        if not api_key:
            raise RuntimeError(
                f"{self.name} provider requires the '{api_key_env_var}' environment variable to be set."
            )
        return self._build_headers(api_key)

    def _tool_definitions(self, request: ModelRequest) -> list[ModelToolDefinition]:
        if request.available_tools:
            return list(request.available_tools)
        if request.response_mode != "tool_call" or not _is_mapping(request.response_schema):
            return []
        tool_name = request.preferred_tool_name or "emit_structured_output"
        return [
            ModelToolDefinition(
                name=tool_name,
                description="Return the structured payload for this graph node.",
                input_schema=request.response_schema,
            )
        ]

    def _preferred_tool_name(
        self,
        request: ModelRequest,
        tools: list[ModelToolDefinition],
    ) -> str | None:
        preferred_name = str(request.preferred_tool_name or "").strip()
        if not preferred_name:
            return None
        if any(tool.name == preferred_name for tool in tools):
            return preferred_name
        return None

    def _message_dicts(self, messages: list[ModelMessage]) -> list[dict[str, str]]:
        return [{"role": message.role, "content": message.content} for message in messages]

    def _build_payload(
        self,
        request: ModelRequest,
        provider_config: Mapping[str, Any],
        tools: list[ModelToolDefinition],
        response_schema: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def _build_headers(self, api_key: str) -> dict[str, str]:
        raise NotImplementedError

    def _parse_response(
        self,
        request: ModelRequest,
        payload: Mapping[str, Any],
        tools: list[ModelToolDefinition],
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
        tools: list[ModelToolDefinition],
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
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": dict(tool.input_schema),
                    },
                }
                for tool in tools
            ]
            preferred_tool_name = self._preferred_tool_name(request, tools)
            if preferred_tool_name:
                payload["tool_choice"] = {
                    "type": "function",
                    "function": {"name": preferred_tool_name},
                }
            elif request.response_mode == "tool_call":
                payload["tool_choice"] = "required"
        return payload

    def _parse_response(
        self,
        request: ModelRequest,
        payload: Mapping[str, Any],
        tools: list[ModelToolDefinition],
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

        normalized_tool_calls: list[ModelToolCall] = []
        raw_tool_calls = message.get("tool_calls", [])
        if isinstance(raw_tool_calls, list):
            for raw_tool_call in raw_tool_calls:
                if not _is_mapping(raw_tool_call):
                    continue
                function_call = raw_tool_call.get("function", {})
                if not _is_mapping(function_call):
                    continue
                tool_name = function_call.get("name")
                if not isinstance(tool_name, str) or not tool_name.strip():
                    continue
                arguments: Any = {}
                raw_arguments = function_call.get("arguments", "{}")
                if isinstance(raw_arguments, str):
                    arguments = json.loads(raw_arguments)
                elif _is_mapping(raw_arguments):
                    arguments = dict(raw_arguments)
                normalized_tool_calls.append(
                    ModelToolCall(
                        tool_name=tool_name,
                        arguments=arguments,
                        provider_tool_id=str(raw_tool_call.get("id")) if raw_tool_call.get("id") else None,
                        metadata={"provider_type": raw_tool_call.get("type")},
                    )
                )

        structured_output = normalized_tool_calls[0].arguments if normalized_tool_calls else None
        if structured_output is None and response_schema is not None and content_text.strip():
            structured_output = json.loads(content_text)

        return ModelResponse(
            content=content_text,
            structured_output=structured_output,
            tool_calls=normalized_tool_calls,
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

    def _backend_type(self) -> str:
        return "anthropic_api"

    def _build_headers(self, api_key: str) -> dict[str, str]:
        return {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }

    def _build_payload(
        self,
        request: ModelRequest,
        provider_config: Mapping[str, Any],
        tools: list[ModelToolDefinition],
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
        if tools:
            payload["tools"] = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": dict(tool.input_schema),
                }
                for tool in tools
            ]
            preferred_tool_name = self._preferred_tool_name(request, tools)
            if preferred_tool_name:
                payload["tool_choice"] = {"type": "tool", "name": preferred_tool_name}
            elif request.response_mode == "tool_call":
                payload["tool_choice"] = {"type": "any"}
        return payload

    def _parse_response(
        self,
        request: ModelRequest,
        payload: Mapping[str, Any],
        tools: list[ModelToolDefinition],
        response_schema: Mapping[str, Any] | None,
        latency_ms: int,
    ) -> ModelResponse:
        blocks = payload.get("content", [])
        if not isinstance(blocks, list):
            raise RuntimeError("claude provider returned an unexpected response shape.")

        text_parts: list[str] = []
        normalized_tool_calls: list[ModelToolCall] = []
        for block in blocks:
            if not _is_mapping(block):
                continue
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                text_parts.append(block["text"])
            if block.get("type") == "tool_use" and _is_mapping(block.get("input")):
                tool_name = block.get("name")
                if isinstance(tool_name, str) and tool_name.strip():
                    normalized_tool_calls.append(
                        ModelToolCall(
                            tool_name=tool_name,
                            arguments=dict(block["input"]),
                            provider_tool_id=str(block.get("id")) if block.get("id") else None,
                        )
                    )

        content_text = "\n".join(part for part in text_parts if part)
        structured_output = normalized_tool_calls[0].arguments if normalized_tool_calls else None
        if structured_output is None and response_schema is not None and content_text.strip():
            structured_output = json.loads(content_text)

        return ModelResponse(
            content=content_text,
            structured_output=structured_output,
            tool_calls=normalized_tool_calls,
            metadata={
                "latency_ms": latency_ms,
                "vendor_model": payload.get("model"),
                "stop_reason": payload.get("stop_reason"),
                "usage": payload.get("usage"),
            },
        )
