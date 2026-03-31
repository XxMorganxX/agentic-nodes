import type {
  GraphDefinition,
  GraphDocument,
  GraphNode,
  RunState,
  RuntimeEvent,
  TestEnvironmentDefinition,
} from "./types";

export type EnvironmentRunSummary = {
  runId: string | null;
  status: string;
  totalAgents: number;
  completedAgents: number;
  runningAgents: number;
  failedAgents: number;
  queuedAgents: number;
  activeAgentNames: string[];
  focusedAgentId: string | null;
  focusedAgentName: string | null;
  elapsedLabel: string;
};

export type AgentRunMilestone = {
  id: string;
  label: string;
  eventType: string;
  timestamp: string;
  timestampLabel: string;
  timestampDetail: string;
  relativeTimestampLabel: string | null;
  deltaLabel: string | null;
  tone: "idle" | "info" | "running" | "success" | "danger";
  nodeId: string | null;
  details: AgentRunMilestoneDetail[];
  dataSections: AgentRunMilestoneDataSection[];
};

export type AgentRunMilestoneDetail = {
  label: string;
  value: string;
};

export type AgentRunMilestoneDataSection = {
  label: string;
  value: unknown;
};

export type AgentRunLane = {
  agentId: string;
  agentName: string;
  status: string;
  runId: string | null;
  currentNodeId: string | null;
  currentNodeLabel: string;
  completedNodes: number;
  totalNodes: number;
  transitionCount: number;
  errorCount: number;
  retryCount: number;
  elapsedLabel: string;
  milestones: AgentRunMilestone[];
};

export type FocusedEventGroup = {
  id: string;
  title: string;
  subtitle: string;
  tone: "info" | "running" | "success" | "danger";
  eventCount: number;
  startedAt: string | null;
  endedAt: string | null;
  nodeId: string | null;
  lines: string[];
};

export type FocusedRunSummary = {
  runId: string | null;
  status: string;
  currentNodeId: string | null;
  currentNodeLabel: string;
  completedNodes: number;
  totalNodes: number;
  transitionCount: number;
  errorCount: number;
  retryCount: number;
  elapsedLabel: string;
  finalOutput: unknown;
  nodeErrors: Record<string, unknown>;
};

export function formatRunStatusLabel(status: string | null | undefined): string {
  const normalized = (status ?? "idle").trim().toLowerCase();
  if (normalized === "idle") {
    return "Not started";
  }
  return normalized
    .split(/[\s_-]+/)
    .filter(Boolean)
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

function normalizeEventType(eventType: string): string {
  return eventType.replace(/^agent\./, "");
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function nodeLabelMap(graph: GraphDefinition | null): Map<string, string> {
  return new Map((graph?.nodes ?? []).map((node) => [node.id, node.label]));
}

function graphByAgent(environment: TestEnvironmentDefinition): Map<string, GraphDefinition> {
  return new Map(
    environment.agents.map((agent) => [
      agent.agent_id,
      {
        graph_id: environment.graph_id,
        name: agent.name,
        description: agent.description,
        version: agent.version,
        graph_type: "graph",
        start_node_id: agent.start_node_id,
        env_vars: { ...environment.env_vars, ...agent.env_vars },
        nodes: agent.nodes,
        edges: agent.edges,
        node_providers: environment.node_providers,
      },
    ]),
  );
}

function formatElapsed(startedAt: string | null | undefined, endedAt: string | null | undefined): string {
  if (!startedAt) {
    return "Not started";
  }
  const startMs = Date.parse(startedAt);
  const endMs = endedAt ? Date.parse(endedAt) : Date.now();
  if (!Number.isFinite(startMs) || !Number.isFinite(endMs)) {
    return "n/a";
  }
  const durationMs = Math.max(0, endMs - startMs);
  if (durationMs < 1000) {
    return "<1s";
  }
  const seconds = Math.round(durationMs / 1000);
  if (seconds < 60) {
    return `${seconds}s`;
  }
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;
  return remainingSeconds > 0 ? `${minutes}m ${remainingSeconds}s` : `${minutes}m`;
}

function formatDurationMs(durationMs: number): string {
  if (!Number.isFinite(durationMs) || durationMs < 0) {
    return "n/a";
  }
  if (durationMs < 1000) {
    return `${Math.round(durationMs)}ms`;
  }
  if (durationMs < 10_000) {
    return `${(durationMs / 1000).toFixed(1)}s`;
  }
  const seconds = Math.round(durationMs / 1000);
  if (seconds < 60) {
    return `${seconds}s`;
  }
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;
  return remainingSeconds > 0 ? `${minutes}m ${remainingSeconds}s` : `${minutes}m`;
}

function formatTimestamp(timestamp: string, detail = false): string {
  const parsed = Date.parse(timestamp);
  if (!Number.isFinite(parsed)) {
    return timestamp;
  }
  const date = new Date(parsed);
  const dateLabel = detail ? `${date.toLocaleDateString(undefined, { month: "short", day: "numeric" })} ` : "";
  const timeLabel = date.toLocaleTimeString(undefined, {
    hour: "numeric",
    minute: "2-digit",
    second: "2-digit",
  });
  return `${dateLabel}${timeLabel}.${String(date.getMilliseconds()).padStart(3, "0")}`;
}

function formatRelativeTimestamp(timestamp: string, startedAt: string | null | undefined): string | null {
  if (!startedAt) {
    return null;
  }
  const timestampMs = Date.parse(timestamp);
  const startMs = Date.parse(startedAt);
  if (!Number.isFinite(timestampMs) || !Number.isFinite(startMs)) {
    return null;
  }
  return `T+${formatDurationMs(Math.max(0, timestampMs - startMs))}`;
}

function formatDeltaLabel(timestamp: string, previousTimestamp: string | null): string | null {
  if (!previousTimestamp) {
    return null;
  }
  const timestampMs = Date.parse(timestamp);
  const previousMs = Date.parse(previousTimestamp);
  if (!Number.isFinite(timestampMs) || !Number.isFinite(previousMs)) {
    return null;
  }
  return `+${formatDurationMs(Math.max(0, timestampMs - previousMs))}`;
}

function nodeIdFromEvent(event: RuntimeEvent): string | null {
  const payloadNodeId = event.payload.node_id;
  return typeof payloadNodeId === "string" && payloadNodeId.length > 0 ? payloadNodeId : null;
}

function eventTone(eventType: string): AgentRunMilestone["tone"] {
  if (eventType === "run.completed" || eventType === "node.completed") {
    return "success";
  }
  if (eventType === "run.failed") {
    return "danger";
  }
  if (eventType === "run.started" || eventType === "node.started" || eventType === "retry.triggered") {
    return "running";
  }
  return "info";
}

function milestoneLabel(event: RuntimeEvent, graph: GraphDefinition | null): string {
  const eventType = normalizeEventType(event.event_type);
  const labels = nodeLabelMap(graph);
  const nodeId = nodeIdFromEvent(event);
  const nodeLabel = nodeId ? (labels.get(nodeId) ?? nodeId) : null;
  if (eventType === "run.started") {
    return "Run started";
  }
  if (eventType === "run.completed") {
    return "Run completed";
  }
  if (eventType === "run.failed") {
    return "Run failed";
  }
  if (eventType === "node.started") {
    return nodeLabel ? `${nodeLabel} started` : "Node started";
  }
  if (eventType === "node.completed") {
    return nodeLabel ? `${nodeLabel} completed` : "Node completed";
  }
  if (eventType === "retry.triggered") {
    return nodeLabel ? `Retry from ${nodeLabel}` : "Retry triggered";
  }
  if (eventType === "edge.selected") {
    return "Transition";
  }
  return event.summary;
}

function formatEventTypeLabel(eventType: string): string {
  return eventType
    .split(".")
    .flatMap((segment) => segment.split(/[\s_-]+/))
    .filter(Boolean)
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

function nodeInputPreview(
  nodeId: string | null,
  graph: GraphDefinition | null,
  inputPayload: unknown,
  knownNodeOutputs: Record<string, unknown>,
  labels: Map<string, string>,
): unknown {
  if (!nodeId) {
    return undefined;
  }
  const node = graph?.nodes.find((candidate) => candidate.id === nodeId) ?? null;
  if (node?.kind === "input") {
    return inputPayload;
  }
  const incomingEdges = (graph?.edges ?? []).filter((edge) => edge.target_id === nodeId);
  const sources = incomingEdges
    .filter((edge, index, edges) => edges.findIndex((candidate) => candidate.source_id === edge.source_id) === index)
    .map((edge) => ({
      sourceLabel: labels.get(edge.source_id) ?? edge.source_id,
      value: knownNodeOutputs[edge.source_id],
    }))
    .filter((entry) => entry.value !== undefined);
  if (sources.length === 0) {
    return undefined;
  }
  if (sources.length === 1) {
    return {
      source: sources[0].sourceLabel,
      value: sources[0].value,
    };
  }
  return Object.fromEntries(sources.map((entry) => [entry.sourceLabel, entry.value]));
}

function appendSection(
  sections: AgentRunMilestoneDataSection[],
  label: string,
  value: unknown,
  options: { allowNull?: boolean } = {},
): void {
  if (value === undefined) {
    return;
  }
  if (value === null && !options.allowNull) {
    return;
  }
  sections.push({ label, value });
}

function appendOutputSections(sections: AgentRunMilestoneDataSection[], output: unknown): void {
  if (!isRecord(output)) {
    appendSection(sections, "Output", output, { allowNull: true });
    return;
  }
  let added = false;
  if ("payload" in output && output.payload !== undefined) {
    sections.push({ label: "Output payload", value: output.payload });
    added = true;
  }
  if (Array.isArray(output.tool_calls) && output.tool_calls.length > 0) {
    sections.push({ label: "Tool calls", value: output.tool_calls });
    added = true;
  }
  if (Array.isArray(output.errors) && output.errors.length > 0) {
    sections.push({ label: "Output errors", value: output.errors });
    added = true;
  }
  if (isRecord(output.artifacts) && "display_envelope" in output.artifacts) {
    sections.push({ label: "Display envelope", value: output.artifacts.display_envelope });
    added = true;
  } else if ("artifacts" in output && output.artifacts !== undefined && !isRecord(output.artifacts)) {
    sections.push({ label: "Output artifacts", value: output.artifacts });
    added = true;
  }
  if ("metadata" in output && output.metadata !== undefined) {
    sections.push({ label: "Output metadata", value: output.metadata });
    added = true;
  }
  if (!added) {
    sections.push({ label: "Output", value: output });
  }
}

function buildMilestoneDetails(
  event: RuntimeEvent,
  graph: GraphDefinition | null,
  labels: Map<string, string>,
): AgentRunMilestoneDetail[] {
  const payload = event.payload;
  const nodeId = nodeIdFromEvent(event);
  const details: AgentRunMilestoneDetail[] = [{ label: "State", value: formatEventTypeLabel(event.event_type) }];
  if (nodeId) {
    details.push({ label: "Node", value: labels.get(nodeId) ?? nodeId });
  }
  if (typeof payload.node_kind === "string") {
    details.push({ label: "Kind", value: payload.node_kind });
  }
  if (typeof payload.node_category === "string") {
    details.push({ label: "Category", value: payload.node_category });
  }
  if (typeof payload.node_provider_label === "string") {
    details.push({ label: "Provider", value: payload.node_provider_label });
  }
  if (typeof payload.status === "string") {
    details.push({ label: "Status", value: payload.status });
  }
  if (typeof payload.visit_count === "number") {
    details.push({ label: "Visit", value: `#${payload.visit_count}` });
  }
  if (typeof payload.source_id === "string" && typeof payload.target_id === "string") {
    const sourceLabel = labels.get(payload.source_id) ?? payload.source_id;
    const targetLabel = labels.get(payload.target_id) ?? payload.target_id;
    details.push({ label: "Route", value: `${sourceLabel} -> ${targetLabel}` });
  }
  if (typeof payload.matched === "boolean") {
    details.push({ label: "Matched", value: payload.matched ? "yes" : "no" });
  }
  if (typeof payload.result_status === "string") {
    details.push({ label: "Result", value: payload.result_status });
  }
  if (typeof payload.agent_name === "string") {
    details.push({ label: "Agent", value: payload.agent_name });
  }
  return details;
}

function buildMilestoneDataSections(
  event: RuntimeEvent,
  graph: GraphDefinition | null,
  inputPayload: unknown,
  knownNodeOutputs: Record<string, unknown>,
  labels: Map<string, string>,
): AgentRunMilestoneDataSection[] {
  const payload = event.payload;
  const nodeId = nodeIdFromEvent(event);
  const sections: AgentRunMilestoneDataSection[] = [];
  if (event.event_type === "run.started") {
    appendSection(sections, "Input", inputPayload, { allowNull: true });
    return sections;
  }
  if (event.event_type === "node.started") {
    appendSection(sections, "Structured input", nodeInputPreview(nodeId, graph, inputPayload, knownNodeOutputs, labels), {
      allowNull: true,
    });
    return sections;
  }
  if (event.event_type === "node.completed") {
    if ("output" in payload) {
      appendOutputSections(sections, payload.output);
    }
    if ("route_outputs" in payload) {
      appendSection(sections, "Route outputs", payload.route_outputs, { allowNull: true });
    }
    if ("error" in payload) {
      appendSection(sections, "Error", payload.error, { allowNull: true });
    }
    if ("metadata" in payload) {
      appendSection(sections, "Execution metadata", payload.metadata);
    }
    return sections;
  }
  if (event.event_type === "run.completed") {
    appendSection(sections, "Final output", payload.final_output, { allowNull: true });
    return sections;
  }
  if (event.event_type === "run.failed") {
    appendSection(sections, "Failure", payload.error, { allowNull: true });
    if ("final_output" in payload) {
      appendSection(sections, "Final output", payload.final_output, { allowNull: true });
    }
    return sections;
  }
  if (event.event_type === "condition.evaluated") {
    appendSection(sections, "Condition payload", payload);
    return sections;
  }
  if (event.event_type === "retry.triggered") {
    appendSection(sections, "Retry context", payload);
    return sections;
  }
  if (event.event_type === "edge.selected") {
    appendSection(sections, "Transition payload", payload);
    return sections;
  }
  return sections;
}

function isEnvironmentGraph(graph: GraphDocument | null): graph is TestEnvironmentDefinition {
  return Boolean(graph && "agents" in graph && Array.isArray(graph.agents));
}

function focusedGraphName(graph: GraphDocument | null, selectedAgentId: string | null): string | null {
  if (!isEnvironmentGraph(graph) || !selectedAgentId) {
    return null;
  }
  return graph.agents.find((agent) => agent.agent_id === selectedAgentId)?.name ?? null;
}

export function buildEnvironmentRunSummary(
  graph: GraphDocument | null,
  runState: RunState | null,
  selectedAgentId: string | null,
): EnvironmentRunSummary | null {
  if (!graph || !isEnvironmentGraph(graph)) {
    return null;
  }
  const agentStates = Object.values(runState?.agent_runs ?? {});
  const runningAgents = agentStates.filter((state) => state.status === "running").length;
  const completedAgents = agentStates.filter((state) => state.status === "completed").length;
  const failedAgents = agentStates.filter((state) => state.status === "failed").length;
  const queuedAgents = Math.max(0, graph.agents.length - runningAgents - completedAgents - failedAgents);
  return {
    runId: runState?.run_id ?? null,
    status: runState?.status ?? "idle",
    totalAgents: graph.agents.length,
    completedAgents,
    runningAgents,
    failedAgents,
    queuedAgents,
    activeAgentNames: graph.agents
      .filter((agent) => runState?.agent_runs?.[agent.agent_id]?.status === "running")
      .map((agent) => agent.name),
    focusedAgentId: selectedAgentId,
    focusedAgentName: focusedGraphName(graph, selectedAgentId),
    elapsedLabel: formatElapsed(runState?.started_at, runState?.ended_at),
  };
}

export function buildAgentRunLanes(
  graph: GraphDocument | null,
  runState: RunState | null,
  events: RuntimeEvent[],
): AgentRunLane[] {
  if (!graph || !isEnvironmentGraph(graph)) {
    return [];
  }
  const graphsByAgent = graphByAgent(graph);
  return graph.agents.map((agent) => {
    const agentState = runState?.agent_runs?.[agent.agent_id] ?? null;
    const currentGraph = graphsByAgent.get(agent.agent_id) ?? null;
    const labels = nodeLabelMap(currentGraph);
    const agentEvents = events
      .filter((event) => event.agent_id === agent.agent_id)
      .map((event) => ({ ...event, event_type: normalizeEventType(event.event_type) }));
    const knownNodeOutputs: Record<string, unknown> = {};
    let previousTimestamp: string | null = null;
    return {
      agentId: agent.agent_id,
      agentName: agent.name,
      status: agentState?.status ?? "idle",
      runId: agentState?.run_id ?? null,
      currentNodeId: agentState?.current_node_id ?? null,
      currentNodeLabel:
        (agentState?.current_node_id ? labels.get(agentState.current_node_id) : null) ??
        agentState?.current_node_id ??
        "n/a",
      completedNodes: Object.keys(agentState?.visit_counts ?? {}).length,
      totalNodes: currentGraph?.nodes.length ?? 0,
      transitionCount: agentState?.transition_history.length ?? 0,
      errorCount: Object.keys(agentState?.node_errors ?? {}).length,
      retryCount: agentEvents.filter((event) => event.event_type === "retry.triggered").length,
      elapsedLabel: formatElapsed(agentState?.started_at, agentState?.ended_at),
      milestones: agentEvents.map((event, index) => {
        const milestone = {
          id: `${agent.agent_id}-${event.timestamp}-${index}`,
          label: milestoneLabel(event, currentGraph),
          eventType: event.event_type,
          timestamp: event.timestamp,
          timestampLabel: formatTimestamp(event.timestamp),
          timestampDetail: formatTimestamp(event.timestamp, true),
          relativeTimestampLabel: formatRelativeTimestamp(event.timestamp, agentState?.started_at),
          deltaLabel: formatDeltaLabel(event.timestamp, previousTimestamp),
          tone: eventTone(event.event_type),
          nodeId: nodeIdFromEvent(event),
          details: buildMilestoneDetails(event, currentGraph, labels),
          dataSections: buildMilestoneDataSections(
            event,
            currentGraph,
            agentState?.input_payload ?? runState?.input_payload ?? null,
            knownNodeOutputs,
            labels,
          ),
        };
        const completedNodeId = event.event_type === "node.completed" ? nodeIdFromEvent(event) : null;
        if (completedNodeId && "output" in event.payload) {
          knownNodeOutputs[completedNodeId] = event.payload.output;
        }
        previousTimestamp = event.timestamp;
        return milestone;
      }),
    };
  });
}

function normalizeFocusedEvents(events: RuntimeEvent[]): RuntimeEvent[] {
  return events.map((event) => ({ ...event, event_type: normalizeEventType(event.event_type) }));
}

export function buildFocusedRunSummary(
  graph: GraphDefinition | null,
  runState: RunState | null,
  events: RuntimeEvent[],
): FocusedRunSummary {
  const labels = nodeLabelMap(graph);
  const normalizedEvents = normalizeFocusedEvents(events);
  return {
    runId: runState?.run_id ?? null,
    status: runState?.status ?? "idle",
    currentNodeId: runState?.current_node_id ?? null,
    currentNodeLabel:
      (runState?.current_node_id ? labels.get(runState.current_node_id) : null) ?? runState?.current_node_id ?? "n/a",
    completedNodes: Object.keys(runState?.visit_counts ?? {}).length,
    totalNodes: graph?.nodes.length ?? 0,
    transitionCount: runState?.transition_history.length ?? 0,
    errorCount: Object.keys(runState?.node_errors ?? {}).length,
    retryCount: normalizedEvents.filter((event) => event.event_type === "retry.triggered").length,
    elapsedLabel: formatElapsed(runState?.started_at, runState?.ended_at),
    finalOutput: runState?.final_output ?? null,
    nodeErrors: runState?.node_errors ?? {},
  };
}

function buildExecutionGroup(
  id: string,
  graph: GraphDefinition | null,
  startedEvent: RuntimeEvent,
  completedEvent: RuntimeEvent | null,
): FocusedEventGroup {
  const labels = nodeLabelMap(graph);
  const nodeId = nodeIdFromEvent(startedEvent);
  const nodeLabel = nodeId ? (labels.get(nodeId) ?? nodeId) : "Node";
  const completePayloadError = completedEvent?.payload.error;
  const tone =
    completePayloadError != null ? "danger" : completedEvent ? "success" : "running";
  return {
    id,
    title: nodeLabel,
    subtitle: completePayloadError != null ? "Execution failed" : completedEvent ? "Execution completed" : "Execution in progress",
    tone,
    eventCount: completedEvent ? 2 : 1,
    startedAt: startedEvent.timestamp,
    endedAt: completedEvent?.timestamp ?? null,
    nodeId,
    lines: [
      startedEvent.summary,
      ...(completedEvent ? [completedEvent.summary] : []),
    ],
  };
}

function buildSingleEventGroup(id: string, event: RuntimeEvent): FocusedEventGroup {
  const eventType = normalizeEventType(event.event_type);
  const tone = eventTone(eventType);
  return {
    id,
    title: milestoneLabel(event, null),
    subtitle: eventType,
    tone: tone === "idle" ? "info" : tone,
    eventCount: 1,
    startedAt: event.timestamp,
    endedAt: null,
    nodeId: nodeIdFromEvent(event),
    lines: [event.summary],
  };
}

export function buildFocusedEventGroups(
  graph: GraphDefinition | null,
  events: RuntimeEvent[],
): FocusedEventGroup[] {
  const normalizedEvents = normalizeFocusedEvents(events);
  const groups: FocusedEventGroup[] = [];
  for (let index = 0; index < normalizedEvents.length; index += 1) {
    const event = normalizedEvents[index];
    if (event.event_type === "node.started") {
      const nextEvent = normalizedEvents[index + 1];
      const sameNodeCompleted =
        nextEvent &&
        nextEvent.event_type === "node.completed" &&
        nodeIdFromEvent(nextEvent) === nodeIdFromEvent(event);
      groups.push(
        buildExecutionGroup(
          `node-${event.timestamp}-${index}`,
          graph,
          event,
          sameNodeCompleted ? nextEvent : null,
        ),
      );
      if (sameNodeCompleted) {
        index += 1;
      }
      continue;
    }
    groups.push(buildSingleEventGroup(`event-${event.timestamp}-${index}`, event));
  }
  return groups.reverse();
}

export function nodeById(graph: GraphDefinition | null, nodeId: string | null): GraphNode | null {
  if (!graph || !nodeId) {
    return null;
  }
  return graph.nodes.find((node) => node.id === nodeId) ?? null;
}
