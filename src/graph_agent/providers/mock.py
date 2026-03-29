from __future__ import annotations

from graph_agent.providers.base import ModelProvider, ModelRequest, ModelResponse


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
