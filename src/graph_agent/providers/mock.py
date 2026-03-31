from __future__ import annotations

from typing import Any, Mapping

from graph_agent.providers.base import ModelProvider, ModelRequest, ModelResponse, ProviderPreflightResult


class MockModelProvider(ModelProvider):
    name = "mock"

    def generate(self, request: ModelRequest) -> ModelResponse:
        mode = request.metadata.get("mode")

        if mode == "schema_proposal":
            return ModelResponse(
                content="Generated an initial tool payload.",
                structured_output={
                    "query": request.metadata.get("user_request", ""),
                    "limit": "three",
                },
                metadata={"mode": mode},
            )

        if mode == "schema_repair":
            return ModelResponse(
                content="Generated a repaired tool payload.",
                structured_output={
                    "query": request.metadata.get("user_request", ""),
                    "limit": 3,
                },
                metadata={"mode": mode},
            )

        if mode == "final_response":
            tool_result = request.metadata.get("tool_result", {})
            item_count = len(tool_result.get("items", []))
            return ModelResponse(
                content=f"I found {item_count} matching items and formatted the response.",
                structured_output={
                    "message": f"Found {item_count} results for '{request.metadata.get('user_request', '')}'.",
                    "items": tool_result.get("items", []),
                },
                metadata={"mode": mode},
            )

        return ModelResponse(content="Mock provider produced no structured output.", metadata={"mode": mode})

    def preflight(self, provider_config: Mapping[str, Any] | None = None) -> ProviderPreflightResult:
        check_auth = bool(provider_config and provider_config.get("check_auth"))
        return ProviderPreflightResult(
            status="available",
            ok=True,
            message=(
                "Mock provider is always available in local development."
                if not check_auth
                else "Mock provider does not require live verification."
            ),
            details={
                "backend_type": "mock",
                "auth_mode": "not_applicable",
                "supports_live_verification": False,
            },
        )
