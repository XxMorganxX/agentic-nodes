# Logging Component

## Purpose

The logging layer provides observability for every meaningful graph execution step so branching, retries, and failures can be debugged without guessing.

## First-Version Decision

- Logging is event-based and structured.
- The same event model powers both debugging output and the live visualization UI.
- Events are stored in memory with the run and exposed through the API.

## Required Events

The first version records:

- run lifecycle events
- node lifecycle events
- condition evaluation results
- selected edge transitions
- model provider metadata
- tool validation and execution outcomes
- retry and loop-guard failures
- final output or terminal error

## Event Shape

Each event should contain:

- event type
- timestamp
- run identifier
- node or edge identifier when relevant
- a human-readable summary
- a machine-readable payload for the UI

## Why This Matters

Graph systems can be non-deterministic even when the graph structure is static. A stable event model makes it possible to reason about the exact path a run took and why it took it.

## Extensibility

Future versions can fan out the same structured events to:

- stdout for local debugging
- file sinks
- external observability services
- persisted run history stores
