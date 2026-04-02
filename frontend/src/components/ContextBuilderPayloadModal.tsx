import { useEffect, useMemo } from "react";
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

type ContextBuilderPayloadModalProps = {
  graph: GraphDefinition;
  node: GraphNode;
  runState: RunState | null;
  runtimeView: ContextBuilderRuntimeView | null;
  onClose: () => void;
};

export function ContextBuilderPayloadModal({ graph, node, runState, runtimeView, onClose }: ContextBuilderPayloadModalProps) {
  const nodeLabel = getNodeInstanceLabel(graph, node);
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
                {runtimeView.sources.map((slot) => (
                  <li
                    key={slot.sourceNodeId}
                    className={`context-builder-input-row context-builder-input-row--${slot.status}`}
                  >
                    <div className="context-builder-input-row-main">
                      <span className="context-builder-input-label">{slot.sourceLabel}</span>
                      <span className="context-builder-input-placeholder">{`{${slot.placeholder}}`}</span>
                    </div>
                    <div className="context-builder-input-status">
                      {slot.status === "pending" ? "Waiting…" : null}
                      {slot.status === "fulfilled" ? "Ready" : null}
                      {slot.status === "error" ? <span className="context-builder-input-error">{slot.errorSummary ?? "Error"}</span> : null}
                    </div>
                  </li>
                ))}
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
