# Control Loop Component

## Purpose

The control loop moves execution through the graph one node at a time, records events, evaluates edge conditions, enforces loop guards, and stops on explicit termination or safety limits.

## First-Version Decision

- Execution is single-run, in-process, and synchronous from the runtime's perspective.
- The backend exposes live events as the loop progresses.
- The loop supports standard transitions, conditional branches, retry loops, and broader graph loopbacks.

## Traversal Rules

1. Start at the graph's declared start node.
2. Execute the current node with the current `RunContext`.
3. Persist node output, errors, and timing into run state.
4. Evaluate outgoing edges in priority order.
5. Take the first matching conditional edge, otherwise the first standard edge.
6. Stop when an output node finishes or a termination condition is hit.

## Safety Limits

The runtime must enforce:

- maximum total steps per run
- maximum visits per node
- optional maximum retries for repair loops

When a guard is hit, the run ends with a structured failure result and a terminal event.

## Error Handling

- Tool validation errors are treated as routable results.
- Recoverable model errors can be routed by conditional edges if a graph defines that path.
- Unhandled runtime errors produce a terminal failure event.

## Termination

The first version supports both:

- explicit stop via successful output node completion
- forced stop via guard limit or unrecoverable runtime error

## Event Emission

The control loop emits events for:

- run started
- node started
- node completed
- edge selected
- condition evaluated
- retry triggered
- run completed
- run failed
