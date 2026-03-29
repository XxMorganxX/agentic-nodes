# Graph Agent Framework

This directory contains the complete graph-agent implementation and its supporting docs in one place.

The framework provides a reusable graph-based runtime for agentic systems where execution moves between typed nodes connected by standard or conditional edges.

The first version is designed around these goals:

- graph authoring through Python definitions and the browser-based Studio
- provider-agnostic model and tool interfaces
- explicit retry and repair loops
- live execution visualization and editing through a FastAPI backend and React UI
- a concrete example flow that demonstrates tool-schema repair

## First-Version Capabilities

The framework supports these node roles:

- `input` nodes for starting execution from an external payload
- `model` nodes for LLM or planner calls
- `tool` nodes for schema-aware tool execution
- `data` nodes for deterministic data collection or transformation
- `output` nodes for final user-facing responses

Those concrete node kinds are grouped into higher-level categories:

- `start`
- `api`
- `tool`
- `data`
- `end`

This separation matters because future providers should plug into a standard classification model even when their implementations differ.

## Shared Terminology

To keep implementation and product conversations aligned, this project uses a few terms consistently:

- `Category`: the graph-level contract layer that determines which node connections are allowed. Current categories are `start`, `api`, `tool`, `data`, and `end`.
- `Kind` or `node kind`: the runtime behavior a node implements, such as `input`, `model`, `tool`, `data`, or `output`.
- `Provider`: the concrete implementation behind a node kind, identified by `provider_id`, such as `core.input`, `mock.model`, `tool.registry`, or a future `discord.webhook`.
- `Node instance`: a specific node placed on the canvas inside a graph.
- `Provider card`: the draggable Studio item that creates a node instance from a provider definition.
- `Tooltip`: the hover or focus summary shown for a node card on the canvas.
- `Tool details modal`: the deeper tool-specific modal opened from a tool node's `Learn More` action.
- `Inspector`: the right-hand editing surface for graph, node, and edge configuration.

The important distinction is:

- `category` answers "what can this connect to?"
- `kind` answers "what does this do at runtime?"
- `provider` answers "which concrete implementation powers it?"

## Standardized Node Providers

Each node now carries three important identifiers:

- a higher-level `category` such as `start`, `api`, `tool`, `data`, or `end`
- a concrete node kind such as `input`, `model`, `tool`, or `output`
- a pluggable `provider_id` such as `mock.model`, `tool.registry`, or a future `openai.chat`, `claude.messages`, or `gcal.webhook`

The runtime supports all of those layers generically, while the stock example catalog currently focuses on input, model, tool, and output providers.

The design goal is to make provider implementations easy to add without changing graph semantics. For example:

- an OpenAI node provider and a Claude node provider should both map to the `api` category
- a Google Calendar webhook receiver could map to the `start` category
- a registry-dispatched function tool could map to the `tool` category

In other words, the graph should branch on stable node meaning, not on vendor-specific class names.

Edges come in two forms:

- standard flow edges that always continue to the next node
- conditional edges that evaluate the latest node result and/or run state before continuing

The runtime allows short retry loops and larger graph loopbacks. A typical repair cycle is:

1. A model node proposes a tool payload.
2. A tool node validates the payload and rejects it.
3. The runtime records the error in run state.
4. A conditional edge routes execution back to a repair model node.
5. The repair node sees the original prompt plus the validation error and produces a corrected tool payload.

## How It Works

The framework separates graph authoring from graph execution.

### 1. Define a graph in Python or the Studio

A graph is built from:

- typed nodes such as `InputNode`, `ModelNode`, `ToolNode`, and `OutputNode`
- edges that connect those nodes
- optional conditions attached to conditional edges
- standardized node-provider metadata that identifies the provider plugin and higher-level category

In practice, the runtime works with serializable graph definitions that can be seeded from Python examples and edited through the browser-based Studio.

Each node returns a normalized `NodeExecutionResult` with:

- `status`
- `output`
- `error`
- `summary`
- `metadata`

That normalized result is what makes it possible for the runtime, the event log, and the UI to all reason about execution the same way.

### 2. Run the graph through the runtime

`GraphRuntime` starts at the configured start node, executes one node at a time, records state, evaluates outgoing edges, and either continues or stops.

For every step it emits structured runtime events such as:

- `run.started`
- `node.started`
- `node.completed`
- `condition.evaluated`
- `retry.triggered`
- `edge.selected`
- `run.completed`
- `run.failed`

Those events are the backbone of both debugging and visualization. In particular, `retry.triggered` is emitted when a matching conditional edge is taken after a non-success result.

### 3. Stream events to the GUI

The FastAPI backend exposes graph definitions, starts runs, and streams execution events over Server-Sent Events.

The React UI:

- draws the graph
- lets you create, save, delete, and run graphs
- lets you drag provider cards into the canvas
- labels conditional edges
- highlights the active node
- provides graph, node, and edge inspection and editing
- shows the execution timeline
- displays final output and recorded errors

## Runtime Model

The first version uses a lightweight local runtime model:

- graph definitions are represented as serializable graph payloads
- example graphs can be seeded from Python
- Studio-authored graphs are stored in a local JSON graph store
- runs are executed in-process
- run state is kept in memory
- event history is stored with each run
- no durable memory or persisted run history is included yet

This keeps the MVP simple while preserving the clean interfaces needed for future persistence and richer authoring workflows.

## Directory Layout

- `model.md` documents provider contracts and model-node behavior
- `tool-registry.md` documents tool registration and validation
- `control-loop.md` documents traversal, retries, loop guards, and termination
- `state.md` documents in-memory run state
- `memory.md` documents the intentionally minimal v1 memory strategy
- `logging.md` documents structured execution events
- `config.py` contains prompts and tuneable constants
- `pyproject.toml` packages the backend
- `src/` contains the runtime, example graph, and API implementation
- `frontend/` contains Graph Agent Studio, the React editor and live run viewer

Important implementation areas under `src/graph_agent/`:

- `runtime/` contains node abstractions, graph definitions, and the execution engine
- `providers/` contains the model provider contract and the mock provider
- `tools/` contains the tool registry and example tool
- `examples/` contains the concrete repair-loop graph
- `api/` contains the FastAPI app, graph store, and in-memory run manager

## Example Flow

The included example graph demonstrates the exact repair-loop pattern that motivated this project:

1. `start` captures the initial input.
2. `propose_tool` uses the `mock.model` API-node provider to ask for a tool payload.
3. `run_tool` uses the `tool.registry` tool-node provider to validate and execute the tool.
4. If validation fails, a conditional edge routes to `repair_tool`.
5. `repair_tool` uses the same API provider classification to repair the payload.
6. `run_tool` runs again.
7. On success, `compose_response` builds the final response payload.
8. `finish` uses the `core.output` end-node provider to return the result.

## Local Development

One-time setup from this directory:

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
cd frontend && npm install && cd ..
```

Single run command:

```bash
python3 run.py
```

That command:

- starts the backend and frontend together
- automatically wires the frontend to the backend
- picks free local ports starting from `8000` for the backend and `5173` for the frontend

Then open the frontend URL printed by the runner and use Graph Agent Studio.

## How To Extend

You can extend the framework in five primary ways:

1. Add a new node provider definition and reuse an existing category when possible.
2. Add a new node class only when the execution semantics are meaningfully different.
3. Register a new model provider behind the provider interface.
4. Register tools with schemas and runtime implementations.
5. Author a new graph definition and expose it through the API.

### Add a node provider

Register a `NodeProviderDefinition` with:

- a stable `provider_id`
- a human-readable name
- a higher-level category such as `start`, `api`, `tool`, `data`, or `end`
- the concrete node kind it belongs to
- a description and optional capabilities

This is the preferred extension point for vendor-specific or source-specific node implementations.

### Add a new node type

Create a class that inherits from `BaseNode`, implement `execute()`, and return a normalized `NodeExecutionResult`.

Use a new node type when the step has distinct semantics, not just different business logic or provider differences.

### Add a new model provider

Implement the `ModelProvider` contract in `providers/` and wire it into `RuntimeServices`.

The core runtime should never need to know which vendor is behind the provider.

### Add a new tool

Register a `ToolDefinition` with:

- a name
- a description
- an input schema
- an executor function

Validation failures should come back as structured `validation_error` results so the graph can route them intentionally.

### Add a new graph

Build a new `GraphDefinition` payload, save it through the API or Studio, and the UI will be able to load, edit, run, and visualize it through the same backend shape.

## Natural Next Extensions

The first version leaves room for these future upgrades:

- persisted run history
- durable memory backends
- richer graph editing workflows such as versioning, import and export, and collaboration
- background workers for long-running nodes
- multiple model providers and provider selection per graph
- declarative graph import and export formats

The supporting docs in this directory explain the core decisions behind those extension points in more detail.
