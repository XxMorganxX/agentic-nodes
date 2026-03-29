import { memo } from "react";
import { Handle, Position } from "reactflow";
import type { NodeProps } from "reactflow";

import { buildNodeTooltip } from "../lib/nodeTooltip";
import type { NodeTooltipData } from "../lib/nodeTooltip";
import type { EditorCatalog, GraphDefinition, GraphNode, RunState } from "../lib/types";

export type GraphCanvasNodeData = {
  node: GraphNode;
  graph: GraphDefinition | null;
  catalog: EditorCatalog | null;
  runState: RunState | null;
  kindColor: string;
  status: "idle" | "active" | "completed" | "error";
  tooltipVisible: boolean;
  onToggleTooltip: (nodeId: string) => void;
  onOpenToolDetails: (nodeId: string) => void;
  onOpenProviderDetails: (nodeId: string) => void;
};

const KIND_LABELS: Record<string, string> = {
  input: "IN",
  model: "AI",
  tool: "FX",
  data: "DB",
  output: "OUT",
};

const FALLBACK_TOOLTIP: NodeTooltipData = {
  title: "Node details unavailable",
  eyebrow: "graph / node",
  description: "This node could not render its tooltip data.",
  sections: [],
  parameters: [],
  emptyState: "The node is still available in the canvas.",
};

function GraphCanvasNodeComponent({
  data,
  selected,
}: NodeProps<GraphCanvasNodeData>) {
  const { node, graph, catalog, runState, kindColor, status, tooltipVisible, onToggleTooltip, onOpenToolDetails, onOpenProviderDetails } = data;
  let tooltip: NodeTooltipData = FALLBACK_TOOLTIP;
  if (tooltipVisible) {
    try {
      tooltip = buildNodeTooltip(node, graph, catalog, runState);
    } catch {
      tooltip = FALLBACK_TOOLTIP;
    }
  }
  const showTargetHandle = node.category !== "start";
  const showSourceHandle = node.category !== "end";
  const iconLabel = KIND_LABELS[node.kind] ?? node.kind.slice(0, 2).toUpperCase();
  const subtitle =
    node.kind === "model"
      ? String(node.config.provider_name ?? node.model_provider_name ?? node.provider_label ?? node.provider_id)
      : node.provider_label ?? node.provider_id;

  return (
    <div
      className={`graph-node-card graph-node-card--${status} ${selected ? "is-selected" : ""} ${tooltipVisible ? "is-tooltip-visible" : ""}`}
      style={
        {
          "--node-kind-color": kindColor,
        } as React.CSSProperties
      }
      tabIndex={0}
      aria-label={`${node.label} ${node.kind} node`}
      onContextMenu={(event) => {
        event.preventDefault();
        event.stopPropagation();
        onToggleTooltip(node.id);
      }}
    >
      {showTargetHandle ? (
        <Handle type="target" position={Position.Left} className="graph-node-handle graph-node-handle-target" />
      ) : null}
      <div className="graph-node-card-inner">
        <div className="graph-node-header">
          <div className="graph-node-icon" aria-hidden="true">
            {iconLabel}
          </div>
          <div className="graph-node-body">
            <strong className="graph-node-title">{node.label}</strong>
            <div className="graph-node-subtitle">{subtitle}</div>
          </div>
          <div className="graph-node-badge" aria-hidden="true">
            <span className="graph-node-badge-dot" />
          </div>
        </div>
        <div className="graph-node-meta">
          <span className="graph-node-chip">{node.category}</span>
          <span className="graph-node-meta-text">{node.kind}</span>
        </div>
        {node.kind === "tool" || node.kind === "model" ? (
          <div className="graph-node-card-actions">
            <button
              type="button"
              className="secondary-button graph-node-card-button"
              onMouseDown={(event) => event.stopPropagation()}
              onClick={(event) => {
                event.stopPropagation();
                if (node.kind === "tool") {
                  onOpenToolDetails(node.id);
                  return;
                }
                onOpenProviderDetails(node.id);
              }}
            >
              {node.kind === "tool" ? "Learn More" : "Provider Info"}
            </button>
          </div>
        ) : null}
      </div>
      <div className="graph-node-tooltip" role="tooltip">
        <div className="graph-node-tooltip-eyebrow">{tooltip.eyebrow}</div>
        <strong className="graph-node-tooltip-title">{tooltip.title}</strong>
        {tooltip.description ? <p className="graph-node-tooltip-description">{tooltip.description}</p> : null}
        {tooltip.sections.map((section) => (
          <section key={section.title} className="graph-node-tooltip-section">
            <div className="graph-node-tooltip-section-title">{section.title}</div>
            <div className="graph-node-tooltip-grid">
              {section.rows.map((row) => (
                <div key={`${section.title}-${row.label}`} className="graph-node-tooltip-row">
                  <span className="graph-node-tooltip-label">{row.label}</span>
                  <span className="graph-node-tooltip-value">{row.value}</span>
                </div>
              ))}
            </div>
          </section>
        ))}
        {tooltip.parameters.length > 0 ? (
          <section className="graph-node-tooltip-section">
            <div className="graph-node-tooltip-section-title">Parameters</div>
            <div className="graph-node-parameter-list">
              {tooltip.parameters.map((parameter) => (
                <div key={parameter.name} className="graph-node-parameter">
                  <div className="graph-node-parameter-header">
                    <code>{parameter.name}</code>
                    <span className="graph-node-parameter-type">{parameter.type}</span>
                    {parameter.required ? <span className="graph-node-parameter-required">required</span> : null}
                  </div>
                  {parameter.description ? (
                    <div className="graph-node-parameter-description">{parameter.description}</div>
                  ) : null}
                  {parameter.source ? <div className="graph-node-parameter-source">Source: {parameter.source}</div> : null}
                </div>
              ))}
            </div>
          </section>
        ) : null}
        {tooltip.emptyState ? <div className="graph-node-tooltip-empty">{tooltip.emptyState}</div> : null}
      </div>
      {showSourceHandle ? (
        <Handle type="source" position={Position.Right} className="graph-node-handle graph-node-handle-source" />
      ) : null}
    </div>
  );
}

export const GraphCanvasNode = memo(GraphCanvasNodeComponent);
