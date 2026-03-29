"""Configuration constants and prompt templates for the graph agent framework."""

GRAPH_AGENT_NAME = "graph-agent"
GRAPH_AGENT_VERSION = "0.1.0"

DEFAULT_GRAPH_ID = "tool-schema-repair"
DEFAULT_RUN_MAX_STEPS = 20
DEFAULT_MAX_VISITS_PER_NODE = 4
DEFAULT_MAX_REPAIR_ATTEMPTS = 2

DEFAULT_MODEL_PROVIDER = "mock"
DEFAULT_MODEL_TIMEOUT_SECONDS = 15
DEFAULT_MODEL_TEMPERATURE = 0.1

DEFAULT_UI_STREAM_MODE = "sse"

SCHEMA_PROPOSAL_PROMPT = """
You are designing input for a tool call.
Given a user request, produce a JSON object for the requested tool.
Return only fields required by the schema and prefer the smallest valid payload.
""".strip()

SCHEMA_REPAIR_PROMPT = """
The previous tool payload failed validation.
Repair the tool input using the original request and the validation error details.
Return a corrected JSON object and avoid repeating the same mistake.
""".strip()

FINAL_RESPONSE_PROMPT = """
Given the original request, the intermediate tool results, and the graph history,
produce a concise final response for the user.
""".strip()
