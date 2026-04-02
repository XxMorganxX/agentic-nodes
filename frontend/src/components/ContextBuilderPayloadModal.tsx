import { useEffect, useMemo, useState } from "react";
import type { MouseEvent } from "react";

import type { ContextBuilderRuntimeView } from "../lib/contextBuilderRuntime";
import { getNodeInstanceLabel } from "../lib/nodeInstanceLabels";
import type { GraphDefinition, GraphNode, RunState } from "../lib/types";

function resolveContextBuilderPayload(node: GraphNode, runState: RunState | null): unknown {
  const nodeOutput = runState?.node_outputs?.[node.id];
  if (
    nodeOutput &&
    typeof nodeOutput === "object" &&
    !Array.isArray(nodeOutput) &&
    "payload" in nodeOutput
  ) {
    return nodeOutput.payload;
  }
  return nodeOutput ?? null;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function formatContextBuilderSourceValue(value: unknown): string {
  if (typeof value === "string") {
    return value;
  }
  if (value === undefined) {
    return "No context captured yet.";
  }
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function resolveContextBuilderSourceValue(runState: RunState | null, sourceNodeId: string): unknown {
  const sourceError = runState?.node_errors?.[sourceNodeId];
  if (sourceError !== undefined) {
    return sourceError;
  }
  const sourceOutput = runState?.node_outputs?.[sourceNodeId];
  if (isRecord(sourceOutput) && Object.prototype.hasOwnProperty.call(sourceOutput, "payload")) {
    return sourceOutput.payload;
  }
  return sourceOutput;
}

type ContextBuilderPayloadModalProps = {
  graph: GraphDefinition;
  node: GraphNode;
  runState: RunState | null;
  runtimeView: ContextBuilderRuntimeView | null;
  onClose: () => void;
};

export function ContextBuilderPayloadModal({ graph, node, runState, runtimeView, onClose }: ContextBuilderPayloadModalProps) {
  const nodeLabel = getNodeInstanceLabel(graph, node);
  const [expandedSourceNodeId, setExpandedSourceNodeId] = useState<string | null>(null);

  useEffect(() => {
    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        onClose();
      }
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  const payload = useMemo(() => resolveContextBuilderPayload(node, runState), [node, runState]);
  const formattedPayload = useMemo(() => JSON.stringify(payload, null, 2), [payload]);

  useEffect(() => {
    if (!runtimeView || runtimeView.sources.length === 0) {
      setExpandedSourceNodeId(null);
      return;
    }
    if (expandedSourceNodeId && runtimeView.sources.some((slot) => slot.sourceNodeId === expandedSourceNodeId)) {
      return;
    }
    const firstSettledSource = runtimeView.sources.find((slot) => slot.status !== "pending");
    setExpandedSourceNodeId(firstSettledSource?.sourceNodeId ?? runtimeView.sources[0]?.sourceNodeId ?? null);
  }, [expandedSourceNodeId, runtimeView]);

  function handleOverlayClick(event: MouseEvent<HTMLDivElement>) {
    if (event.target === event.currentTarget) {
      onClose();
    }
  }

  const forwardingLabel = runtimeView?.isWaitingToForward
    ? "Holding downstream until every input is settled and the merge is complete."
    : runtimeView?.contextBuilderComplete === true
      ? "Merge complete — the run will forward this payload when execution reaches the next node."
      : runtimeView?.totalCount
        ? "Waiting for upstream nodes to produce output or errors for each bound input."
        : "Connect inputs to this context builder to track them here.";

  return (
    <div className="tool-details-modal-backdrop" onClick={handleOverlayClick} role="presentation">
      <section
        className="tool-details-modal display-response-modal context-builder-payload-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="context-builder-payload-modal-title"
      >
        <div className="tool-details-modal-header">
          <div>
            <div className="tool-details-modal-eyebrow">Context Builder Payload</div>
            <h3 id="context-builder-payload-modal-title">{nodeLabel}</h3>
            <p>
              Each upstream input behaves like a promise: when it completes with a message or an error, it is reflected here. The builder only forwards once the
              runtime merge is complete.
            </p>
          </div>
          <button type="button" className="secondary-button" onClick={onClose}>
            Close
          </button>
        </div>

        <div className="tool-details-modal-body">
          <section className="context-builder-modal-section">
            <div className="tool-details-modal-preview-header">
              <strong>Forwarding</strong>
              <span className={runtimeView?.isWaitingToForward ? "context-builder-modal-flag context-builder-modal-flag--hold" : ""}>
                {forwardingLabel}
              </span>
            </div>
          </section>

          {runtimeView && runtimeView.sources.length > 0 ? (
            <section className="context-builder-modal-section">
              <div className="tool-details-modal-preview-header">
                <strong>Inputs</strong>
                <span>
                  {runtimeView.fulfilledCount + runtimeView.errorCount}/{runtimeView.totalCount} settled
                  {runtimeView.errorCount > 0 ? ` (${runtimeView.errorCount} with errors)` : ""}
                </span>
              </div>
              <ul className="context-builder-input-list">
                {runtimeView.sources.map((slot) => {
                  const isExpanded = expandedSourceNodeId === slot.sourceNodeId;
                  const contextValue = resolveContextBuilderSourceValue(runState, slot.sourceNodeId);
                  const detailId = `context-builder-input-detail-${slot.sourceNodeId}`;
                  return (
                    <li
                      key={slot.sourceNodeId}
                      className={`context-builder-input-row context-builder-input-row--${slot.status}${isExpanded ? " is-expanded" : ""}`}
                    >
                      <button
                        type="button"
                        className="context-builder-input-row-button"
                        aria-expanded={isExpanded}
                        aria-controls={detailId}
                        onClick={() => setExpandedSourceNodeId(isExpanded ? null : slot.sourceNodeId)}
                      >
                        <div className="context-builder-input-row-main">
                          <span className="context-builder-input-label">{slot.sourceLabel}</span>
                          <span className="context-builder-input-placeholder">{`{${slot.placeholder}}`}</span>
                        </div>
                        <div className="context-builder-input-status">
                          {slot.status === "pending" ? "Waiting…" : null}
                          {slot.status === "fulfilled" ? "Ready" : null}
                          {slot.status === "error" ? <span className="context-builder-input-error">{slot.errorSummary ?? "Error"}</span> : null}
                          <span className="context-builder-input-toggle">{isExpanded ? "Hide context" : "Show context"}</span>
                        </div>
                      </button>
                      {isExpanded ? (
                        <div id={detailId} className="context-builder-input-detail">
                          <div className="context-builder-input-detail-label">
                            {slot.status === "error" ? "Captured error" : "Captured context"}
                          </div>
                          <pre>{formatContextBuilderSourceValue(contextValue)}</pre>
                        </div>
                      ) : null}
                    </li>
                  );
                })}
              </ul>
            </section>
          ) : null}

          <section className="tool-details-modal-preview display-response-modal-preview">
            <div className="tool-details-modal-preview-header">
              <strong>Current merged payload</strong>
              <span>
                {runState?.current_node_id === node.id
                  ? "This node is executing."
                  : payload != null
                    ? "Latest snapshot from the active or most recent run."
                    : "No payload yet for this run."}
              </span>
            </div>
            <pre>{formattedPayload}</pre>
          </section>
        </div>
      </section>
    </div>
  );
}
