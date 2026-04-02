import { useEffect, useMemo } from "react";
import type { MouseEvent } from "react";

import { getNodeInstanceLabel } from "../lib/nodeInstanceLabels";
import type { GraphDefinition, GraphNode, RunState } from "../lib/types";

type DisplayResponseModalProps = {
  graph: GraphDefinition;
  node: GraphNode;
  runState: RunState | null;
  onClose: () => void;
};

function resolveDisplayEnvelope(node: GraphNode, runState: RunState | null): unknown {
  const nodeOutput = runState?.node_outputs?.[node.id];
  if (
    nodeOutput &&
    typeof nodeOutput === "object" &&
    !Array.isArray(nodeOutput) &&
    "artifacts" in nodeOutput &&
    nodeOutput.artifacts &&
    typeof nodeOutput.artifacts === "object" &&
    !Array.isArray(nodeOutput.artifacts) &&
    "display_envelope" in nodeOutput.artifacts
  ) {
    return nodeOutput.artifacts.display_envelope;
  }
  return nodeOutput ?? null;
}

export function DisplayResponseModal({ graph, node, runState, onClose }: DisplayResponseModalProps) {
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

  const envelope = useMemo(() => resolveDisplayEnvelope(node, runState), [node, runState]);
  const formattedEnvelope = useMemo(() => JSON.stringify(envelope, null, 2), [envelope]);

  function handleOverlayClick(event: MouseEvent<HTMLDivElement>) {
    if (event.target === event.currentTarget) {
      onClose();
    }
  }

  return (
    <div className="tool-details-modal-backdrop" onClick={handleOverlayClick} role="presentation">
      <section
        className="tool-details-modal display-response-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="display-response-modal-title"
      >
        <div className="tool-details-modal-header">
          <div>
            <div className="tool-details-modal-eyebrow">Data Display Response</div>
            <h3 id="display-response-modal-title">{nodeLabel}</h3>
            <p>Inspect the full resolved envelope captured by this display node during the current or latest run.</p>
          </div>
          <button type="button" className="secondary-button" onClick={onClose}>
            Close
          </button>
        </div>

        <div className="tool-details-modal-body">
          <section className="tool-details-modal-preview display-response-modal-preview">
            <div className="tool-details-modal-preview-header">
              <strong>Full Response</strong>
              <span>{runState?.current_node_id === node.id ? "This node is still running." : "Snapshot from the latest available output."}</span>
            </div>
            <pre>{formattedEnvelope}</pre>
          </section>
        </div>
      </section>
    </div>
  );
}
