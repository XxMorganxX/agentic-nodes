# State Store Component

## Purpose

The state store tracks all mutable information for a single graph run. It is scoped to one execution and is reset between runs.

## First-Version Decision

- State is stored in memory only.
- No run state survives process restarts in v1.
- Nodes read from the current run context and write results through normalized runtime updates.

## Stored Data

The run state includes:

- run identifier
- graph identifier
- initial input payload
- current node identifier
- ordered event history
- per-node outputs
- per-node errors
- transition history
- visit counts
- final output payload
- terminal status

## Access Patterns

- The runtime is the main owner of state mutation.
- Nodes return execution results; they do not mutate shared state directly.
- Edge conditions can inspect the current state snapshot.
- The API can read state for visualization after or during execution.

## Why This Shape

Keeping runtime state centralized avoids hidden mutations inside nodes and makes live visualization, replay, and debugging simpler.

## Extensibility

Future versions can swap the in-memory store for:

- persisted run storage
- distributed execution state
- replayable event sourcing
