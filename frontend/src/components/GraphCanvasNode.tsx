import { memo } from "react";
import type { CSSProperties } from "react";
import { Handle, Position } from "reactflow";
import type { NodeProps } from "reactflow";

import {
  API_MESSAGE_HANDLE_ID,
  API_TOOL_CALL_HANDLE_ID,
  API_TOOL_CONTEXT_HANDLE_ID,
  getApiToolContextTargetAnchorRatio,
  getToolSourceHandleAnchorRatio,
  isApiModelNode,
  isMcpContextProviderNode,
  isRoutableToolNode,
  isWireJunctionNode,
  TOOL_CONTEXT_HANDLE_ID,
  TOOL_FAILURE_HANDLE_ID,
  TOOL_SUCCESS_HANDLE_ID,
} from "../lib/editor";
import { warnGraphDiagnostic } from "../lib/dragDiagnostics";
import { buildNodeTooltip } from "../lib/nodeTooltip";
import type { NodeTooltipData } from "../lib/nodeTooltip";
import { formatRunStatusLabel } from "../lib/runVisualization";
import type { EditorCatalog, GraphDefinition, GraphNode, RunState } from "../lib/types";

export type GraphCanvasNodeData = {
  node: GraphNode;
  graph: GraphDefinition | null;
  catalog: EditorCatalog | null;
  runState: RunState | null;
  kindColor: string;
  status: "idle" | "active" | "success" | "failed" | "unreached";
  isConnectionMagnetized?: boolean;
  preview?: boolean;
  tooltipVisible: boolean;
  onToggleTooltip: (nodeId: string) => void;
  onOpenToolDetails: (nodeId: string) => void;
  onOpenProviderDetails: (nodeId: string) => void;
  onOpenDisplayResponse: (nodeId: string) => void;
  onHandlePointerDown: (nodeId: string, handleType: "source" | "target", handleId: string | null) => boolean;
  onJunctionPointerDown: (nodeId: string, clientPosition: { x: number; y: number }) => void;
};

const KIND_LABELS: Record<string, string> = {
  input: "IN",
  model: "AI",
  tool: "FX",
  mcp_context_provider: "MC",
  mcp_tool_executor: "MX",
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

function hasContextConnection(graph: GraphDefinition | null, node: GraphNode): boolean {
  if (!graph || node.kind !== "mcp_context_provider") {
    return false;
  }
  const hasBindingEdge = graph.edges.some(
    (edge) => edge.kind === "binding" && edge.source_id === node.id && edge.target_id !== node.id,
  );
  if (hasBindingEdge) {
    return true;
  }
  return graph.nodes.some((candidate) => {
    if (candidate.kind !== "model") {
      return false;
    }
    const targetIds = Array.isArray(candidate.config.tool_target_node_ids) ? candidate.config.tool_target_node_ids : [];
    return targetIds.some((targetId) => String(targetId) === node.id);
  });
}

function isContextBooted(catalog: EditorCatalog | null, node: GraphNode): boolean {
  const toolNames = Array.isArray(node.config.tool_names) ? node.config.tool_names.map((toolName) => String(toolName)) : [];
  if (!catalog || toolNames.length === 0) {
    return false;
  }
  const selectedTools = catalog.tools.filter((tool) => toolNames.includes(tool.name));
  if (selectedTools.length === 0) {
    return false;
  }
  const serverIds = [...new Set(selectedTools.map((tool) => tool.server_id).filter((serverId): serverId is string => Boolean(serverId)))];
  if (serverIds.length === 0) {
    return selectedTools.every((tool) => tool.available !== false);
  }
  return serverIds.every((serverId) => catalog.mcp_servers?.some((server) => server.server_id === serverId && server.running));
}

function GraphCanvasNodeComponent({
  data,
  selected,
}: NodeProps<GraphCanvasNodeData>) {
  const {
    node,
    graph,
    catalog,
    runState,
    kindColor,
    status,
    isConnectionMagnetized = false,
    preview = false,
    tooltipVisible,
    onToggleTooltip,
    onOpenToolDetails,
    onOpenProviderDetails,
    onOpenDisplayResponse,
    onJunctionPointerDown,
  } = data;
  const isWireJunction = isWireJunctionNode(node);
  const isRoutableTool = isRoutableToolNode(node);
  const isContextProviderNode = isMcpContextProviderNode(node);
  const isModelNode = isApiModelNode(node);
  const isDisplayNode = node.provider_id === "core.data_display";
  const nodeOutput = runState?.node_outputs?.[node.id];
  const displayEnvelope =
    isDisplayNode &&
    nodeOutput &&
    typeof nodeOutput === "object" &&
    nodeOutput !== null &&
    "artifacts" in nodeOutput &&
    typeof nodeOutput.artifacts === "object" &&
    nodeOutput.artifacts !== null &&
    "display_envelope" in nodeOutput.artifacts
      ? nodeOutput.artifacts.display_envelope
      : nodeOutput;
  const displayText = isDisplayNode
    ? status === "active"
      ? "Running..."
      : displayEnvelope !== undefined
        ? JSON.stringify(displayEnvelope, null, 2)
        : "Run the graph to inspect the incoming envelope here."
    : null;
  let tooltip: NodeTooltipData = FALLBACK_TOOLTIP;
  if (tooltipVisible && !preview && !isWireJunction) {
    try {
      tooltip = buildNodeTooltip(node, graph, catalog, runState);
    } catch (error) {
      warnGraphDiagnostic("GraphCanvasNode", "tooltip fallback", error, {
        nodeId: node.id,
        nodeKind: node.kind,
        tooltipVisible,
      });
      tooltip = FALLBACK_TOOLTIP;
    }
  }
  const showTargetHandle = !preview && node.category !== "start";
  const showSourceHandle = !preview && node.category !== "end";
  const successHandleStyle = {
    top: `${getToolSourceHandleAnchorRatio(TOOL_SUCCESS_HANDLE_ID) * 100}%`,
  } satisfies CSSProperties;
  const failureHandleStyle = {
    top: `${getToolSourceHandleAnchorRatio(TOOL_FAILURE_HANDLE_ID) * 100}%`,
  } satisfies CSSProperties;
  const contextSourceHandleStyle = {
    top: `${getToolSourceHandleAnchorRatio(TOOL_CONTEXT_HANDLE_ID) * 100}%`,
  } satisfies CSSProperties;
  const apiToolCallHandleStyle = {
    top: `${getToolSourceHandleAnchorRatio(API_TOOL_CALL_HANDLE_ID) * 100}%`,
  } satisfies CSSProperties;
  const apiMessageHandleStyle = {
    top: `${getToolSourceHandleAnchorRatio(API_MESSAGE_HANDLE_ID) * 100}%`,
  } satisfies CSSProperties;
  const primaryTargetHandleStyle = {
    top: `${getApiToolContextTargetAnchorRatio(null) * 100}%`,
  } satisfies CSSProperties;
  const contextTargetHandleStyle = {
    top: `${getApiToolContextTargetAnchorRatio(API_TOOL_CONTEXT_HANDLE_ID) * 100}%`,
  } satisfies CSSProperties;
  const iconLabel = KIND_LABELS[node.kind] ?? node.kind.slice(0, 2).toUpperCase();
  const subtitle =
    node.kind === "model"
      ? String(node.config.provider_name ?? node.model_provider_name ?? node.provider_label ?? node.provider_id)
      : node.provider_label ?? node.provider_id;
  const isContextConnected = isContextProviderNode ? hasContextConnection(graph, node) : false;
  const contextBooted = isContextProviderNode ? isContextBooted(catalog, node) : false;
  const displayStatus = isContextProviderNode
    ? contextBooted && isContextConnected
      ? "success"
      : isContextConnected
        ? "failed"
        : "idle"
    : status;
  const isActive = displayStatus === "active";
  const statusLabel = isContextProviderNode
    ? contextBooted && isContextConnected
      ? "Connected and MCP booted"
      : isContextConnected
        ? "Connected but MCP not booted"
        : contextBooted
          ? "MCP booted but not connected"
          : "MCP not booted and not connected"
    : formatRunStatusLabel(displayStatus);
  const nodeCardClassName = `graph-node-card graph-node-card--${displayStatus} ${isRoutableTool ? "graph-node-card--tool-outputs" : ""} ${
    isContextProviderNode ? "graph-node-card--tool-context-provider" : ""
  } ${
    isModelNode ? "graph-node-card--model-inputs" : ""
  } ${
    isModelNode ? "graph-node-card--model-outputs" : ""
  } ${
    isDisplayNode ? "graph-node-card--display-node" : ""
  } ${selected ? "is-selected" : ""} ${tooltipVisible ? "is-tooltip-visible" : ""} ${preview ? "is-preview" : ""} ${
    isConnectionMagnetized ? "is-connection-magnetized" : ""
  }`;

  if (isWireJunction) {
    return (
      <div
        className={`graph-junction-node graph-junction-node--${status} ${selected ? "is-selected" : ""} ${isConnectionMagnetized ? "is-connection-magnetized" : ""}`}
        tabIndex={preview ? -1 : 0}
        aria-label="Wire junction"
        onMouseDown={(event) => {
          if (preview || event.button !== 0) {
            return;
          }
          event.preventDefault();
          event.stopPropagation();
          onJunctionPointerDown(node.id, { x: event.clientX, y: event.clientY });
        }}
      >
        {showTargetHandle ? (
          <Handle
            type="target"
            position={Position.Left}
            className={`graph-node-handle graph-node-handle-target graph-junction-handle ${isConnectionMagnetized ? "graph-node-handle-valid is-magnetized" : ""}`}
          />
        ) : null}
        {showSourceHandle ? (
          <Handle
            type="source"
            position={Position.Right}
            className="graph-node-handle graph-node-handle-source graph-junction-handle"
          />
        ) : null}
      </div>
    );
  }

  return (
    <div
      className={nodeCardClassName}
      style={
        {
          "--node-kind-color": kindColor,
        } as CSSProperties
      }
      tabIndex={preview ? -1 : 0}
      aria-label={`${node.label} ${node.kind} node`}
      onContextMenu={(event) => {
        if (preview) {
          return;
        }
        event.preventDefault();
        event.stopPropagation();
        onToggleTooltip(node.id);
      }}
    >
      {showTargetHandle && !isContextProviderNode ? (
        <Handle
          type="target"
          position={Position.Left}
          className={`graph-node-handle graph-node-handle-target ${isConnectionMagnetized ? "graph-node-handle-valid is-magnetized" : ""}`}
          style={isModelNode ? primaryTargetHandleStyle : undefined}
        />
      ) : null}
      {showTargetHandle && isModelNode ? (
        <>
          <div className="graph-node-input-port graph-node-input-port--context" style={contextTargetHandleStyle} aria-hidden="true">
            <span className="graph-node-output-port-label">Tool Context</span>
          </div>
          <Handle
            id={API_TOOL_CONTEXT_HANDLE_ID}
            type="target"
            position={Position.Left}
            className={`graph-node-handle graph-node-handle-target graph-node-handle-target--context ${isConnectionMagnetized ? "graph-node-handle-valid is-magnetized" : ""}`}
            style={contextTargetHandleStyle}
          />
        </>
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
          <div className="graph-node-badge" aria-label={`Node status: ${statusLabel}`}>
            {isActive ? (
              <span className="graph-node-badge-spinner">
                <span className="graph-node-badge-spinner-core" />
              </span>
            ) : (
              <span className="graph-node-badge-dot" />
            )}
            <span className="graph-node-status-tooltip" role="status">
              {statusLabel}
            </span>
          </div>
        </div>
        <div className="graph-node-meta">
          <span className="graph-node-chip">{node.category}</span>
          <span className="graph-node-meta-text">{node.kind}</span>
        </div>
        {isDisplayNode ? (
          <div
            role="button"
            tabIndex={preview ? -1 : 0}
            className="graph-node-inline-display"
            onMouseDown={(event) => event.stopPropagation()}
            onClick={(event) => {
              event.stopPropagation();
              onOpenDisplayResponse(node.id);
            }}
            onKeyDown={(event) => {
              if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                event.stopPropagation();
                onOpenDisplayResponse(node.id);
              }
            }}
            aria-label={`Open full response for ${node.label}`}
          >
            <div className="graph-node-inline-display-header">Run Envelope</div>
            <pre className="graph-node-inline-display-body">{displayText}</pre>
            <span className="graph-node-inline-display-hint">Click to expand</span>
          </div>
        ) : null}
        {!preview && (node.kind === "tool" || node.kind === "mcp_context_provider" || node.kind === "model") ? (
          <div className="graph-node-card-actions">
            <button
              type="button"
              className="secondary-button graph-node-card-button"
              onMouseDown={(event) => event.stopPropagation()}
              onClick={(event) => {
                event.stopPropagation();
                if (node.kind === "tool" || node.kind === "mcp_context_provider") {
                  onOpenToolDetails(node.id);
                  return;
                }
                onOpenProviderDetails(node.id);
              }}
            >
              {node.kind === "tool" || node.kind === "mcp_context_provider" ? "Learn More" : "Provider Info"}
            </button>
          </div>
        ) : null}
      </div>
      {!preview ? (
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
      ) : null}
      {showSourceHandle && isRoutableTool ? (
        <>
          <div className="graph-node-output-port graph-node-output-port--success" style={successHandleStyle} aria-hidden="true">
            <span className="graph-node-output-port-label">On Success</span>
          </div>
          <Handle
            id={TOOL_SUCCESS_HANDLE_ID}
            type="source"
            position={Position.Right}
            className="graph-node-handle graph-node-handle-source graph-node-handle-source--success"
            style={successHandleStyle}
          />
          <div className="graph-node-output-port graph-node-output-port--failure" style={failureHandleStyle} aria-hidden="true">
            <span className="graph-node-output-port-label">On Failure</span>
          </div>
          <Handle
            id={TOOL_FAILURE_HANDLE_ID}
            type="source"
            position={Position.Right}
            className="graph-node-handle graph-node-handle-source graph-node-handle-source--failure"
            style={failureHandleStyle}
          />
        </>
      ) : null}
      {showSourceHandle && isContextProviderNode ? (
        <>
          <div className="graph-node-output-port graph-node-output-port--context" style={contextSourceHandleStyle} aria-hidden="true">
            <span className="graph-node-output-port-label">Context</span>
          </div>
          <Handle
            id={TOOL_CONTEXT_HANDLE_ID}
            type="source"
            position={Position.Right}
            className="graph-node-handle graph-node-handle-source graph-node-handle-source--context"
            style={contextSourceHandleStyle}
          />
        </>
      ) : null}
      {showSourceHandle && isModelNode ? (
        <>
          <div className="graph-node-output-port graph-node-output-port--tool-call" style={apiToolCallHandleStyle} aria-hidden="true">
            <span className="graph-node-output-port-label">Tool Call</span>
          </div>
          <Handle
            id={API_TOOL_CALL_HANDLE_ID}
            type="source"
            position={Position.Right}
            className="graph-node-handle graph-node-handle-source graph-node-handle-source--tool-call"
            style={apiToolCallHandleStyle}
          />
          <div className="graph-node-output-port graph-node-output-port--message" style={apiMessageHandleStyle} aria-hidden="true">
            <span className="graph-node-output-port-label">Message</span>
          </div>
          <Handle
            id={API_MESSAGE_HANDLE_ID}
            type="source"
            position={Position.Right}
            className="graph-node-handle graph-node-handle-source graph-node-handle-source--message"
            style={apiMessageHandleStyle}
          />
        </>
      ) : null}
      {showSourceHandle && !isRoutableTool && !isContextProviderNode && !isModelNode ? (
        <Handle
          type="source"
          position={Position.Right}
          className="graph-node-handle graph-node-handle-source"
        />
      ) : null}
    </div>
  );
}

export const GraphCanvasNode = memo(GraphCanvasNodeComponent);
