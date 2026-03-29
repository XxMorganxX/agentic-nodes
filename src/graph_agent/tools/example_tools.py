from __future__ import annotations

from graph_agent.tools.base import ToolContext, ToolDefinition, ToolResult


def build_search_catalog_tool() -> ToolDefinition:
    def executor(payload: dict[str, object], context: ToolContext) -> ToolResult:
        query = str(payload["query"])
        limit = int(payload["limit"])
        items = [
            {"id": "item-1", "title": f"{query} overview"},
            {"id": "item-2", "title": f"{query} schema guide"},
            {"id": "item-3", "title": f"{query} API example"},
        ][:limit]
        return ToolResult(
            status="success",
            output={"items": items, "tool_context_run_id": context.run_id},
            summary="Catalog lookup completed.",
        )

    return ToolDefinition(
        name="search_catalog",
        description="Searches a mock catalog and returns matching reference items.",
        input_schema={
            "type": "object",
            "required": ["query", "limit"],
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer"},
            },
        },
        executor=executor,
    )
