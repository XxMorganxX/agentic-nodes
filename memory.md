# Memory Component

## Purpose

The memory component would hold context that survives beyond a single run, such as prior conversations, durable summaries, or retrieved knowledge.

## First-Version Decision

- The first version does not implement durable memory.
- A minimal placeholder interface exists so future work can add persistent memory without changing the control loop contract.
- The example graph relies only on in-run state.

## Rationale

The immediate goal is a strong graph runtime with clear node semantics, repair loops, and observability. Durable memory would add storage and retrieval design complexity before the core execution model is proven.

## Future Expansion

When memory is added later, document:

- what data is stored
- when memories are written
- how memories are retrieved
- how retrieved memory enters model prompts
- how memory backends can be swapped cleanly
