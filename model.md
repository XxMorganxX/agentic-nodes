# Model Component

## Purpose

Model nodes perform probabilistic planning steps inside the graph. They consume the current run context, optionally include prior tool errors or intermediate results, and return structured output that downstream nodes can act on.

## First-Version Decision

- The framework uses a provider-agnostic `ModelProvider` interface.
- The included example runs against a mock provider by default.
- Real provider integrations can be added later without changing graph definitions.
- Model-facing nodes live under the higher-level `api` node category and are identified by pluggable node-provider IDs such as `mock.model`, `openai.chat`, or `claude.messages`.

## Responsibilities

- accept a prompt payload derived from the current `RunContext`
- produce structured output for downstream graph steps
- return trace metadata such as provider name and latency
- avoid embedding provider-specific details into the core runtime

## Interface Expectations

The model layer must expose a single execution contract that accepts:

- provider configuration
- prompt messages or structured prompt fields
- optional response schema hints
- the current run context for observability

The response envelope must include:

- a structured result payload
- raw content for inspection when helpful
- usage or latency metadata if available
- a recoverable error shape when generation fails

## Prompt Strategy

All prompts live in `config.py`, not inline in runtime code.

The first example uses:

- a schema proposal prompt
- a schema repair prompt
- a final response synthesis prompt

## Extensibility

Future providers should plug in behind the same interface. Swapping from a mock provider to OpenAI-compatible or Anthropic-compatible logic should only require a wiring change, not runtime changes.

At the graph layer, those implementations should also register standardized node-provider definitions so the system can treat them uniformly as `api` nodes while still preserving the concrete provider identity.
