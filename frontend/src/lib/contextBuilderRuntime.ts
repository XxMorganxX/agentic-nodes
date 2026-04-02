import { getContextBuilderBindings } from "./contextBuilderBindings";
import { latestOutputsFromCompletedNodeEvents } from "./runVisualization";
import type { GraphDefinition, GraphNode, RunState, RuntimeEvent } from "./types";

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function summarizeNodeError(error: unknown): string {
  if (typeof error === "string") {
    return error.trim() || "Error";
  }
  if (isRecord(error)) {
    const msg = error.message ?? error.detail ?? error.reason ?? error.summary;
    if (typeof msg === "string" && msg.trim()) {
      return msg.trim();
    }
    try {
      return JSON.stringify(error);
    } catch {
      return "Error";
    }
  }
  return String(error);
}

export type ContextBuilderSourceSlot = {
  sourceNodeId: string;
  sourceLabel: string;
  placeholder: string;
  status: "pending" | "fulfilled" | "error";
  errorSummary?: string;
};

export type ContextBuilderRuntimeView = {
  sources: ContextBuilderSourceSlot[];
  fulfilledCount: number;
  errorCount: number;
  totalCount: number;
  payloadPreview: unknown;
  contextBuilderComplete: boolean | null;
  holdingOutgoing: boolean;
  /** True when the builder has produced output but is not allowed to forward (partial merge). */
  isWaitingToForward: boolean;
};

export function buildContextBuilderRuntimeView(
  graph: GraphDefinition,
  node: GraphNode,
  runState: RunState | null,
  normalizedEvents: RuntimeEvent[],
): ContextBuilderRuntimeView | null {
  if (node.provider_id !== "core.context_builder") {
    return null;
  }

  const latestByNode = latestOutputsFromCompletedNodeEvents(normalizedEvents);
  const bindings = getContextBuilderBindings(node, graph);

  const sources: ContextBuilderSourceSlot[] = bindings.map((binding) => {
    const err = runState?.node_errors?.[binding.sourceNodeId];
    if (err != null) {
      return {
        sourceNodeId: binding.sourceNodeId,
        sourceLabel: binding.sourceLabel,
        placeholder: binding.placeholder,
        status: "error" as const,
        errorSummary: summarizeNodeError(err),
      };
    }
    const hasOutput =
      Object.prototype.hasOwnProperty.call(latestByNode, binding.sourceNodeId) ||
      (runState?.node_outputs && Object.prototype.hasOwnProperty.call(runState.node_outputs, binding.sourceNodeId));
    return {
      sourceNodeId: binding.sourceNodeId,
      sourceLabel: binding.sourceLabel,
      placeholder: binding.placeholder,
      status: hasOutput ? ("fulfilled" as const) : ("pending" as const),
    };
  });

  const fulfilledCount = sources.filter((s) => s.status === "fulfilled").length;
  const errorCount = sources.filter((s) => s.status === "error").length;
  const totalCount = sources.length;

  const composeOutput =
    (latestByNode[node.id] as unknown) ?? (runState?.node_outputs?.[node.id] as unknown) ?? null;

  let contextBuilderComplete: boolean | null = null;
  let holdingOutgoing = false;

  const composeCompleted = normalizedEvents.filter(
    (e) => e.event_type === "node.completed" && e.payload.node_id === node.id,
  );
  const lastCompose = composeCompleted[composeCompleted.length - 1];
  if (lastCompose) {
    const meta = lastCompose.payload.metadata;
    if (isRecord(meta)) {
      holdingOutgoing = Boolean(meta.hold_outgoing_edges);
    }
  }

  if (isRecord(composeOutput) && isRecord(composeOutput.metadata)) {
    const c = composeOutput.metadata.context_builder_complete;
    contextBuilderComplete = typeof c === "boolean" ? c : null;
  }

  const payloadPreview = isRecord(composeOutput) && Object.prototype.hasOwnProperty.call(composeOutput, "payload")
    ? composeOutput.payload
    : composeOutput;

  const composeSelfError = runState?.node_errors?.[node.id];
  const isWaitingToForward =
    composeSelfError == null &&
    composeOutput != null &&
    (holdingOutgoing || contextBuilderComplete === false);

  return {
    sources,
    fulfilledCount,
    errorCount,
    totalCount,
    payloadPreview,
    contextBuilderComplete,
    holdingOutgoing,
    isWaitingToForward,
  };
}
