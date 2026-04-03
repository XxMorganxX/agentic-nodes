import type {
  GraphDefinition,
  GraphDocument,
  GraphNode,
  RunState,
  RuntimeEvent,
  TestEnvironmentDefinition,
} from "./types";
import { buildNodeInstanceLabelMap } from "./nodeInstanceLabels";

export type EnvironmentRunSummary = {
  runId: string | null;
  status: string;
  totalAgents: number;
  completedAgents: number;
  runningAgents: number;
  failedAgents: number;
  cancelledAgents: number;
  interruptedAgents: number;
  queuedAgents: number;
  activeAgentNames: string[];
  focusedAgentId: string | null;
  focusedAgentName: string | null;
  elapsedLabel: string;
  lastHeartbeatLabel: string;
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

export type AgentRunErrorSummary = {
  id: string;
  nodeId: string | null;
  nodeLabel: string;
  errorTypeLabel: string | null;
  message: string;
  metadata: string[];
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
  errorSummaries: AgentRunErrorSummary[];
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
  lastHeartbeatLabel: string;
  finalOutput: unknown;
  nodeErrors: Record<string, unknown>;
};

export type FocusedRunNodeState = {
  nodeId: string;
  isActive: boolean;
  wasVisited: boolean;
  hasError: boolean;
  latestOutput: unknown;
  latestError: unknown;
  visitCount: number;
};

export type FocusedRunProjection = {
  normalizedEvents: RuntimeEvent[];
  completedNodeIds: Set<string>;
  nodeStates: Record<string, FocusedRunNodeState>;
  errorSummaries: AgentRunErrorSummary[];
  runSummary: FocusedRunSummary;
  eventGroups: FocusedEventGroup[];
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

function cleanInlineText(value: string): string {
  return value.replace(/\s+/g, " ").trim();
}

function truncateText(value: string, limit = 220): string {
  return value.length > limit ? `${value.slice(0, limit - 1)}...` : value;
}

function firstNonEmptyString(...values: unknown[]): string | null {
  for (const value of values) {
    if (typeof value === "string") {
      const cleaned = cleanInlineText(value);
      if (cleaned.length > 0) {
        return cleaned;
      }
    }
  }
  return null;
}

function formatIdentifierLabel(value: string): string {
  const ACRONYMS = new Set(["api", "http", "https", "id", "json", "mcp", "sse", "stderr", "stdout", "ui", "url"]);
  return value
    .split(/[.\s_-]+/)
    .filter(Boolean)
    .map((word) => {
      const normalized = word.toLowerCase();
      if (ACRONYMS.has(normalized)) {
        return normalized.toUpperCase();
      }
      return normalized.charAt(0).toUpperCase() + normalized.slice(1);
    })
    .join(" ");
}

function stringifyCompactValue(value: unknown): string {
  if (typeof value === "string") {
    return value;
  }
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function extractErrorMessage(value: unknown): string | null {
  if (typeof value === "string") {
    return firstNonEmptyString(value);
  }
  if (Array.isArray(value)) {
    return value.map((entry) => extractErrorMessage(entry)).find((entry): entry is string => Boolean(entry)) ?? null;
  }
  if (isRecord(value)) {
    return (
      firstNonEmptyString(value.message, value.detail, value.reason, value.stderr, value.stdout, value.summary) ??
      extractErrorMessage(value.error) ??
      extractErrorMessage(value.errors)
    );
  }
  return null;
}

function buildErrorMetadata(error: Record<string, unknown>): string[] {
  const metadata = [
    typeof error.tool_name === "string" ? `Tool ${error.tool_name}` : null,
    typeof error.result_status === "string" ? `Status ${formatIdentifierLabel(error.result_status)}` : null,
  ].filter((entry): entry is string => Boolean(entry));
  return [...new Set(metadata)];
}

function summarizeErrorEntry(nodeId: string, nodeLabel: string, value: unknown): AgentRunErrorSummary {
  if (isRecord(value)) {
    const message =
      extractErrorMessage(value) ?? truncateText(cleanInlineText(stringifyCompactValue(value)), 220);
    return {
      id: `${nodeId}-${message}`,
      nodeId,
      nodeLabel,
      errorTypeLabel: typeof value.type === "string" ? formatIdentifierLabel(value.type) : null,
      message,
      metadata: buildErrorMetadata(value),
    };
  }
  const message = extractErrorMessage(value) ?? truncateText(cleanInlineText(stringifyCompactValue(value)), 220);
  return {
    id: `${nodeId}-${message}`,
    nodeId,
    nodeLabel,
    errorTypeLabel: null,
    message,
    metadata: [],
  };
}

function summarizeNodeErrors(nodeErrors: Record<string, unknown>, labels: Map<string, string>): AgentRunErrorSummary[] {
  const seen = new Set<string>();
  return Object.entries(nodeErrors).flatMap(([nodeId, value]) => {
    const nodeLabel = labels.get(nodeId) ?? nodeId;
    const entries = Array.isArray(value) ? value : [value];
    return entries.flatMap((entry, index) => {
      if (entry == null) {
        return [];
      }
      const summary = summarizeErrorEntry(nodeId, nodeLabel, entry);
      const key = `${nodeId}:${summary.errorTypeLabel ?? "none"}:${summary.message}`;
      if (seen.has(key)) {
        return [];
      }
      seen.add(key);
      return [{ ...summary, id: `${nodeId}-${index}-${summary.message}` }];
    });
  });
}

function nodeLabelMap(graph: GraphDefinition | null): Map<string, string> {
  return buildNodeInstanceLabelMap(graph);
}

function completedNodeIds(events: RuntimeEvent[]): Set<string> {
  return new Set(
    events
      .filter((event) => normalizeEventType(event.event_type) === "node.completed")
      .map((event) => (typeof event.payload.node_id === "string" ? event.payload.node_id : ""))
      .filter((nodeId) => nodeId.length > 0),
  );
}

function hasOwnRecordValue(record: Record<string, unknown> | null | undefined, key: string): boolean {
  return Boolean(record && Object.prototype.hasOwnProperty.call(record, key));
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

function formatHeartbeat(timestamp: string | null | undefined): string {
  if (!timestamp) {
    return "n/a";
  }
  return formatTimestamp(timestamp, true);
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

function incomingSourceNodeLabels(nodeId: string | null, graph: GraphDefinition | null, labels: Map<string, string>): string[] {
  if (!nodeId || !graph) {
    return [];
  }
  const sourceNodeIds = graph.edges
    .filter((edge) => edge.target_id === nodeId)
    .map((edge) => edge.source_id)
    .filter((sourceId, index, values) => values.indexOf(sourceId) === index);
  return sourceNodeIds.map((sourceId) => {
    const sourceLabel = labels.get(sourceId) ?? sourceId;
    return sourceLabel === sourceId ? sourceId : `${sourceLabel} (${sourceId})`;
  });
}

function eventTone(eventType: string): AgentRunMilestone["tone"] {
  if (eventType === "run.completed" || eventType === "node.completed") {
    return "success";
  }
  if (eventType === "run.failed" || eventType === "run.cancelled" || eventType === "run.interrupted") {
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
  if (eventType === "run.cancelled") {
    return "Run cancelled";
  }
  if (eventType === "run.interrupted") {
    return "Run interrupted";
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
  if (event.event_type === "node.started") {
    const sourceLabels = incomingSourceNodeLabels(nodeId, graph, labels);
    if (sourceLabels.length === 1) {
      details.push({ label: "Input source", value: sourceLabels[0] });
    } else if (sourceLabels.length > 1) {
      details.push({ label: "Input sources", value: sourceLabels.join(", ") });
    }
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
    if ("received_input" in payload) {
      appendSection(sections, "Received input", payload.received_input, { allowNull: true });
    } else {
      appendSection(sections, "Received input", nodeInputPreview(nodeId, graph, inputPayload, knownNodeOutputs, labels), {
        allowNull: true,
      });
    }
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
  if (event.event_type === "run.cancelled") {
    appendSection(sections, "Cancellation", payload.error, { allowNull: true });
    if ("final_output" in payload) {
      appendSection(sections, "Final output", payload.final_output, { allowNull: true });
    }
    return sections;
  }
  if (event.event_type === "run.interrupted") {
    appendSection(sections, "Interruption", payload.error, { allowNull: true });
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
  const totalAgents = runState?.agent_runs ? agentStates.length : graph.agents.length;
  const runningAgents = agentStates.filter((state) => state.status === "running").length;
  const completedAgents = agentStates.filter((state) => state.status === "completed").length;
  const failedAgents = agentStates.filter((state) => state.status === "failed").length;
  const cancelledAgents = agentStates.filter((state) => state.status === "cancelled").length;
  const interruptedAgents = agentStates.filter((state) => state.status === "interrupted").length;
  const queuedAgents = Math.max(
    0,
    totalAgents - runningAgents - completedAgents - failedAgents - cancelledAgents - interruptedAgents,
  );
  return {
    runId: runState?.run_id ?? null,
    status: runState?.status ?? "idle",
    totalAgents,
    completedAgents,
    runningAgents,
    failedAgents,
    cancelledAgents,
    interruptedAgents,
    queuedAgents,
    activeAgentNames: graph.agents
      .filter((agent) => runState?.agent_runs?.[agent.agent_id]?.status === "running")
      .map((agent) => agent.name),
    focusedAgentId: selectedAgentId,
    focusedAgentName: focusedGraphName(graph, selectedAgentId),
    elapsedLabel: formatElapsed(runState?.started_at, runState?.ended_at),
    lastHeartbeatLabel: formatHeartbeat(runState?.last_heartbeat_at),
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
    const projection = buildFocusedRunProjection(currentGraph, agentState, agentEvents);
    const errorSummaries = projection.errorSummaries;
    const knownNodeOutputs: Record<string, unknown> = {};
    let previousTimestamp: string | null = null;
    return {
      agentId: agent.agent_id,
      agentName: agent.name,
      status: projection.runSummary.status,
      runId: projection.runSummary.runId,
      currentNodeId: projection.runSummary.currentNodeId,
      currentNodeLabel: projection.runSummary.currentNodeLabel,
      completedNodes: projection.runSummary.completedNodes,
      totalNodes: projection.runSummary.totalNodes,
      transitionCount: projection.runSummary.transitionCount,
      errorCount: errorSummaries.length,
      errorSummaries,
      retryCount: projection.runSummary.retryCount,
      elapsedLabel: projection.runSummary.elapsedLabel,
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

const syncInvariantWarnings = new Set<string>();

function warnSyncInvariantOnce(key: string, details: Record<string, unknown>): void {
  if (typeof window === "undefined" || syncInvariantWarnings.has(key)) {
    return;
  }
  syncInvariantWarnings.add(key);
  console.warn("[runVisualization] runtime projection sync invariant", details);
}

function setDifference(left: Set<string>, right: Set<string>): string[] {
  return [...left].filter((value) => !right.has(value));
}

function deriveRunStatus(runState: RunState | null, normalizedEvents: RuntimeEvent[]): string {
  const currentStatus = runState?.status?.trim();
  if (currentStatus) {
    return currentStatus;
  }
  const lastEventType = normalizedEvents[normalizedEvents.length - 1]?.event_type ?? null;
  if (lastEventType === "run.completed") {
    return "completed";
  }
  if (lastEventType === "run.failed") {
    return "failed";
  }
  if (lastEventType === "run.interrupted") {
    return "interrupted";
  }
  if (normalizedEvents.some((event) => event.event_type === "run.started")) {
    return "running";
  }
  return "idle";
}

function deriveCurrentNodeId(runState: RunState | null, normalizedEvents: RuntimeEvent[]): string | null {
  let currentNodeId: string | null = null;
  for (const event of normalizedEvents) {
    if (event.event_type === "node.started") {
      currentNodeId = nodeIdFromEvent(event);
      continue;
    }
    if (event.event_type === "node.completed" && nodeIdFromEvent(event) === currentNodeId) {
      currentNodeId = null;
      continue;
    }
    if (event.event_type === "run.completed" || event.event_type === "run.failed" || event.event_type === "run.interrupted") {
      currentNodeId = null;
    }
  }
  if (currentNodeId) {
    return currentNodeId;
  }
  return runState?.current_node_id ?? null;
}

function wasNodeVisited(nodeId: string, runState: RunState | null, completedNodeIdSet: Set<string>): boolean {
  if (!runState) {
    return completedNodeIdSet.has(nodeId);
  }
  return (
    (runState.visit_counts?.[nodeId] ?? 0) > 0 ||
    hasOwnRecordValue(runState.node_outputs, nodeId) ||
    hasOwnRecordValue(runState.node_errors, nodeId)
  );
}

/** Latest `node.completed` output per node id (successive completions overwrite; supports progressive nodes). */
export function latestOutputsFromCompletedNodeEvents(normalizedEvents: RuntimeEvent[]): Record<string, unknown> {
  const outputs: Record<string, unknown> = {};
  for (const event of normalizedEvents) {
    if (event.event_type !== "node.completed") {
      continue;
    }
    const nodeId = nodeIdFromEvent(event);
    if (!nodeId || !Object.prototype.hasOwnProperty.call(event.payload, "output")) {
      continue;
    }
    outputs[nodeId] = event.payload.output;
  }
  return outputs;
}

function latestNodeOutputsByEvent(normalizedEvents: RuntimeEvent[]): Record<string, unknown> {
  return latestOutputsFromCompletedNodeEvents(normalizedEvents);
}

function buildFocusedNodeStates(
  graph: GraphDefinition | null,
  runState: RunState | null,
  normalizedEvents: RuntimeEvent[],
  completedNodeIdSet: Set<string>,
  currentNodeId: string | null,
): Record<string, FocusedRunNodeState> {
  const latestEventOutputs = latestNodeOutputsByEvent(normalizedEvents);
  return Object.fromEntries(
    (graph?.nodes ?? []).map((node) => {
      const latestError = runState?.node_errors?.[node.id];
      const latestOutput = Object.prototype.hasOwnProperty.call(latestEventOutputs, node.id)
        ? latestEventOutputs[node.id]
        : runState?.node_outputs?.[node.id];
      return [
        node.id,
        {
          nodeId: node.id,
          isActive: currentNodeId === node.id,
          wasVisited: wasNodeVisited(node.id, runState, completedNodeIdSet),
          hasError: latestError != null,
          latestOutput,
          latestError,
          visitCount: runState?.visit_counts?.[node.id] ?? 0,
        },
      ] satisfies [string, FocusedRunNodeState];
    }),
  );
}

function validateFocusedRunProjection(
  runState: RunState | null,
  normalizedEvents: RuntimeEvent[],
  completedNodeIdSet: Set<string>,
  currentNodeId: string | null,
): void {
  if (!runState) {
    return;
  }
  const historyCompletedNodeIds = completedNodeIds(normalizeFocusedEvents(runState.event_history ?? []));
  const missingFromProjection = setDifference(historyCompletedNodeIds, completedNodeIdSet);
  const missingFromHistory = setDifference(completedNodeIdSet, historyCompletedNodeIds);
  if (missingFromProjection.length > 0 || missingFromHistory.length > 0) {
    warnSyncInvariantOnce(
      `completed:${runState.run_id}:${missingFromProjection.join(",")}:${missingFromHistory.join(",")}`,
      {
        runId: runState.run_id,
        projectionCompletedNodeIds: [...completedNodeIdSet],
        historyCompletedNodeIds: [...historyCompletedNodeIds],
        missingFromProjection,
        missingFromHistory,
      },
    );
  }
  if ((runState.current_node_id ?? null) !== currentNodeId) {
    warnSyncInvariantOnce(`current-node:${runState.run_id}:${runState.current_node_id ?? "none"}:${currentNodeId ?? "none"}`, {
      runId: runState.run_id,
      projectedCurrentNodeId: currentNodeId,
      runStateCurrentNodeId: runState.current_node_id ?? null,
      lastEventType: normalizedEvents[normalizedEvents.length - 1]?.event_type ?? null,
    });
  }
}

function buildFocusedEventGroupsFromNormalizedEvents(
  graph: GraphDefinition | null,
  normalizedEvents: RuntimeEvent[],
): FocusedEventGroup[] {
  return normalizedEvents
    .filter((event) => event.event_type !== "edge.selected")
    .map((event, index) => buildSingleEventGroup(`event-${event.timestamp}-${index}`, event, graph))
    .reverse();
}

export function buildFocusedRunProjection(
  graph: GraphDefinition | null,
  runState: RunState | null,
  events: RuntimeEvent[],
): FocusedRunProjection {
  const labels = nodeLabelMap(graph);
  const normalizedEvents = normalizeFocusedEvents(events);
  const completedNodeIdSet = completedNodeIds(normalizedEvents);
  const currentNodeId = deriveCurrentNodeId(runState, normalizedEvents);
  const errorSummaries = summarizeNodeErrors(runState?.node_errors ?? {}, labels);
  const nodeStates = buildFocusedNodeStates(graph, runState, normalizedEvents, completedNodeIdSet, currentNodeId);
  validateFocusedRunProjection(runState, normalizedEvents, completedNodeIdSet, currentNodeId);
  return {
    normalizedEvents,
    completedNodeIds: completedNodeIdSet,
    nodeStates,
    errorSummaries,
    runSummary: {
      runId: runState?.run_id ?? null,
      status: deriveRunStatus(runState, normalizedEvents),
      currentNodeId,
      currentNodeLabel: (currentNodeId ? labels.get(currentNodeId) : null) ?? currentNodeId ?? "n/a",
      completedNodes: Object.values(nodeStates).filter((nodeState) => nodeState.wasVisited).length,
      totalNodes: graph?.nodes.length ?? 0,
      transitionCount: runState?.transition_history.length ?? 0,
      errorCount: errorSummaries.length,
      retryCount: normalizedEvents.filter((event) => event.event_type === "retry.triggered").length,
      elapsedLabel: formatElapsed(runState?.started_at, runState?.ended_at),
      lastHeartbeatLabel: formatHeartbeat(runState?.last_heartbeat_at),
      finalOutput: runState?.final_output ?? null,
      nodeErrors: runState?.node_errors ?? {},
    },
    eventGroups: buildFocusedEventGroupsFromNormalizedEvents(graph, normalizedEvents),
  };
}

export function buildFocusedRunSummary(
  graph: GraphDefinition | null,
  runState: RunState | null,
  events: RuntimeEvent[],
): FocusedRunSummary {
  return buildFocusedRunProjection(graph, runState, events).runSummary;
}

function buildSingleEventGroup(id: string, event: RuntimeEvent, graph: GraphDefinition | null): FocusedEventGroup {
  const eventType = normalizeEventType(event.event_type);
  const tone = eventTone(eventType);
  return {
    id,
    title: milestoneLabel(event, graph),
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
  return buildFocusedEventGroupsFromNormalizedEvents(graph, normalizedEvents);
}

export function nodeById(graph: GraphDefinition | null, nodeId: string | null): GraphNode | null {
  if (!graph || !nodeId) {
    return null;
  }
  return graph.nodes.find((node) => node.id === nodeId) ?? null;
}
