import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Connection, Edge as FlowEdge, EdgeChange, Node as FlowNode, NodeChange, Position, ReactFlowInstance } from "reactflow";
import { Background, BackgroundVariant, ConnectionLineType, Controls, MarkerType, MiniMap, ReactFlow } from "reactflow";
import "reactflow/dist/style.css";

import { EventTimeline } from "./EventTimeline";
import { GraphCanvasEdge } from "./GraphCanvasEdge";
import { GraphInspector } from "./GraphInspector";
import { GraphCanvasNode } from "./GraphCanvasNode";
import type { GraphCanvasNodeData } from "./GraphCanvasNode";
import { ProviderSummary } from "./ProviderSummary";
import { ProviderDetailsModal } from "./ProviderDetailsModal";
import { ToolDetailsModal } from "./ToolDetailsModal";
import { canConnectNodes, createNodeFromProvider, defaultConditionalCondition } from "../lib/editor";
import type { EditorCatalog, GraphDefinition, NodeProviderDefinition, RunState, RuntimeEvent } from "../lib/types";

type GraphCanvasProps = {
  graph: GraphDefinition | null;
  runState: RunState | null;
  events: RuntimeEvent[];
  activeRunId: string | null;
  isRunning: boolean;
  catalog: EditorCatalog | null;
  selectedNodeId: string | null;
  selectedEdgeId: string | null;
  onGraphChange: (graph: GraphDefinition) => void;
  onGraphDrag: (graph: GraphDefinition) => void;
  onRunGraph: () => void;
  onScrollToTop: () => void;
  onSelectionChange: (nodeId: string | null, edgeId: string | null) => void;
};

type DrawerTab = "add" | "inspect" | "run";

const KIND_COLORS: Record<string, string> = {
  input: "#8486a5",
  model: "#6c5ce7",
  provider: "#f59e0b",
  tool: "#a78bfa",
  data: "#2dd4bf",
  output: "#4ade80",
};

const nodeTypes = { graphNode: GraphCanvasNode };
const edgeTypes = { graphEdge: GraphCanvasEdge };
const NODE_STYLE = {
  width: 280,
  background: "transparent",
  border: "none",
  padding: 0,
  boxShadow: "none",
} as const;

type DrawerActionButtonProps = {
  tab: DrawerTab;
  activeTab: DrawerTab;
  drawerOpen: boolean;
  label: string;
  onClick: () => void;
  children: React.ReactNode;
};

function DrawerActionButton({ tab, activeTab, drawerOpen, label, onClick, children }: DrawerActionButtonProps) {
  return (
    <button
      type="button"
      className={`graph-toolbar-button${drawerOpen && activeTab === tab ? " is-active" : ""}`}
      onClick={onClick}
      aria-label={label}
      title={label}
    >
      {children}
    </button>
  );
}

export function GraphCanvas({
  graph,
  runState,
  events,
  activeRunId,
  isRunning,
  catalog,
  selectedNodeId,
  selectedEdgeId,
  onGraphChange,
  onGraphDrag,
  onRunGraph,
  onScrollToTop,
  onSelectionChange,
}: GraphCanvasProps) {
  const [flowInstance, setFlowInstance] = useState<ReactFlowInstance | null>(null);
  const [editorMessage, setEditorMessage] = useState<string | null>(null);
  const [toolDetailsNodeId, setToolDetailsNodeId] = useState<string | null>(null);
  const [providerDetailsNodeId, setProviderDetailsNodeId] = useState<string | null>(null);
  const [tooltipNodeId, setTooltipNodeId] = useState<string | null>(null);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [drawerTab, setDrawerTab] = useState<DrawerTab>("add");
  const [providerQuery, setProviderQuery] = useState("");
  const [isProviderDragActive, setIsProviderDragActive] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [panLocked, setPanLocked] = useState(false);
  const canvasRef = useRef<HTMLDivElement | null>(null);
  const nodeDataCacheRef = useRef(new Map<string, GraphCanvasNodeData>());
  const dragFrameRef = useRef<number | null>(null);
  const pendingDragGraphRef = useRef<GraphDefinition | null>(null);
  const zoomAnimationDuration = 120;

  const setCanvasZoom = useCallback((zoom: number) => {
    const nextZoom = Number.isFinite(zoom) && zoom > 0 ? zoom : 1;
    canvasRef.current?.style.setProperty("--graph-viewport-zoom", String(nextZoom));
  }, []);

  const openDrawerTab = useCallback((tab: DrawerTab) => {
    setDrawerOpen((current) => {
      const isSameTab = current && tab === drawerTab;
      if (!isSameTab) {
        setDrawerTab(tab);
      }
      return !isSameTab;
    });
  }, [drawerTab]);

  const handleToggleTooltip = useCallback(
    (nodeId: string) => {
      onSelectionChange(nodeId, null);
      setTooltipNodeId((currentNodeId) => (currentNodeId === nodeId ? null : nodeId));
    },
    [onSelectionChange],
  );

  const handleOpenToolDetails = useCallback(
    (nodeId: string) => {
      onSelectionChange(nodeId, null);
      setToolDetailsNodeId(nodeId);
    },
    [onSelectionChange],
  );

  const handleFlowInit = useCallback((instance: ReactFlowInstance) => {
    setFlowInstance(instance);
    setCanvasZoom(instance.getZoom());
  }, [setCanvasZoom]);

  const handleViewportMove = useCallback((_: MouseEvent | TouchEvent | null, viewport: { zoom: number }) => {
    setCanvasZoom(viewport.zoom);
  }, [setCanvasZoom]);

  const handleOpenProviderDetails = useCallback(
    (nodeId: string) => {
      onSelectionChange(nodeId, null);
      setProviderDetailsNodeId(nodeId);
    },
    [onSelectionChange],
  );

  const handleZoomIn = useCallback(() => {
    void flowInstance?.zoomIn({ duration: zoomAnimationDuration });
  }, [flowInstance]);

  const handleZoomOut = useCallback(() => {
    void flowInstance?.zoomOut({ duration: zoomAnimationDuration });
  }, [flowInstance]);

  const handleFitView = useCallback(() => {
    void flowInstance?.fitView({ duration: zoomAnimationDuration, padding: 0.16 });
  }, [flowInstance]);

  const cancelPendingDragFrame = useCallback(() => {
    if (dragFrameRef.current !== null) {
      cancelAnimationFrame(dragFrameRef.current);
      dragFrameRef.current = null;
    }
    pendingDragGraphRef.current = null;
  }, []);

  const scheduleDragGraphUpdate = useCallback(
    (nextGraph: GraphDefinition) => {
      pendingDragGraphRef.current = nextGraph;
      if (dragFrameRef.current !== null) {
        return;
      }
      dragFrameRef.current = requestAnimationFrame(() => {
        dragFrameRef.current = null;
        const pendingGraph = pendingDragGraphRef.current;
        pendingDragGraphRef.current = null;
        if (pendingGraph) {
          onGraphDrag(pendingGraph);
        }
      });
    },
    [onGraphDrag],
  );

  useEffect(() => cancelPendingDragFrame, [cancelPendingDragFrame]);
  useEffect(() => {
    setCanvasZoom(flowInstance?.getZoom() ?? 1);
  }, [flowInstance, setCanvasZoom]);

  const wasdKeysRef = useRef(new Set<string>());
  const wasdFrameRef = useRef<number | null>(null);

  useEffect(() => {
    const PAN_SPEED = 8;
    const isTextInput = (target: EventTarget | null): boolean => {
      if (!(target instanceof HTMLElement)) return false;
      const tag = target.tagName;
      return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || target.isContentEditable;
    };

    const tick = () => {
      const keys = wasdKeysRef.current;
      if (keys.size === 0 || !flowInstance) {
        wasdFrameRef.current = null;
        return;
      }
      const viewport = flowInstance.getViewport();
      let dx = 0;
      let dy = 0;
      if (keys.has("a") || keys.has("arrowleft")) dx += PAN_SPEED;
      if (keys.has("d") || keys.has("arrowright")) dx -= PAN_SPEED;
      if (keys.has("w") || keys.has("arrowup")) dy += PAN_SPEED;
      if (keys.has("s") || keys.has("arrowdown")) dy -= PAN_SPEED;
      if (dx !== 0 || dy !== 0) {
        flowInstance.setViewport({ x: viewport.x + dx, y: viewport.y + dy, zoom: viewport.zoom });
      }
      wasdFrameRef.current = requestAnimationFrame(tick);
    };

    const onKeyDown = (event: KeyboardEvent) => {
      if (isTextInput(event.target)) return;
      const key = event.key.toLowerCase();
      if (key === "w" || key === "a" || key === "s" || key === "d" || key === "arrowup" || key === "arrowdown" || key === "arrowleft" || key === "arrowright") {
        event.preventDefault();
        if (!wasdKeysRef.current.has(key)) {
          wasdKeysRef.current.add(key);
          if (wasdFrameRef.current === null) {
            wasdFrameRef.current = requestAnimationFrame(tick);
          }
        }
      }
    };

    const onKeyUp = (event: KeyboardEvent) => {
      const key = event.key.toLowerCase();
      wasdKeysRef.current.delete(key);
      if (wasdKeysRef.current.size === 0 && wasdFrameRef.current !== null) {
        cancelAnimationFrame(wasdFrameRef.current);
        wasdFrameRef.current = null;
      }
    };

    const onBlur = () => {
      wasdKeysRef.current.clear();
      if (wasdFrameRef.current !== null) {
        cancelAnimationFrame(wasdFrameRef.current);
        wasdFrameRef.current = null;
      }
    };

    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    window.addEventListener("blur", onBlur);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
      window.removeEventListener("blur", onBlur);
      if (wasdFrameRef.current !== null) {
        cancelAnimationFrame(wasdFrameRef.current);
      }
    };
  }, [flowInstance]);

  const addProviderNode = useCallback(
    (provider: NodeProviderDefinition, screenPosition?: { x: number; y: number }) => {
      if (!graph || !catalog || !flowInstance) {
        return;
      }
      const bounds = canvasRef.current?.getBoundingClientRect();
      const fallbackPosition = bounds
        ? {
            x: bounds.left + (drawerOpen ? Math.max(140, (bounds.width - 360) * 0.5) : bounds.width * 0.5),
            y: bounds.top + bounds.height * 0.5,
          }
        : { x: window.innerWidth * 0.45, y: window.innerHeight * 0.5 };
      const position = flowInstance.screenToFlowPosition(screenPosition ?? fallbackPosition);
      const nextNode = createNodeFromProvider(graph, provider, catalog, position);
      onGraphChange({
        ...graph,
        start_node_id: !graph.start_node_id && nextNode.category === "start" ? nextNode.id : graph.start_node_id,
        nodes: [...graph.nodes, nextNode],
      });
      setDrawerOpen(true);
      setDrawerTab("inspect");
      onSelectionChange(nextNode.id, null);
      setEditorMessage(`Added ${provider.display_name}.`);
    },
    [catalog, drawerOpen, flowInstance, graph, onGraphChange, onSelectionChange],
  );

  const nodes = useMemo<FlowNode[]>(() => {
    if (!graph) {
      return [];
    }
    const nextNodeDataCache = new Map<string, GraphCanvasNodeData>();
    const nextNodes = graph.nodes.map((node) => {
      const isActive = runState?.current_node_id === node.id;
      const hasError = Boolean(runState?.node_errors?.[node.id]);
      const hasOutput = Boolean(runState?.node_outputs?.[node.id]);
      const kindColor = KIND_COLORS[node.kind] ?? "#8486a5";
      const status: GraphCanvasNodeData["status"] = hasError ? "error" : isActive ? "active" : hasOutput ? "completed" : "idle";
      const tooltipVisible = tooltipNodeId === node.id;
      const tooltipGraph = tooltipVisible ? graph : null;
      const previousData = nodeDataCacheRef.current.get(node.id);
      const nextData =
        previousData &&
        previousData.node === node &&
        previousData.graph === tooltipGraph &&
        previousData.catalog === catalog &&
        previousData.runState === runState &&
        previousData.kindColor === kindColor &&
        previousData.status === status &&
        previousData.tooltipVisible === tooltipVisible &&
        previousData.onToggleTooltip === handleToggleTooltip &&
        previousData.onOpenToolDetails === handleOpenToolDetails &&
        previousData.onOpenProviderDetails === handleOpenProviderDetails
          ? previousData
          : {
              node,
              graph: tooltipGraph,
              catalog,
              runState,
              kindColor,
              status,
              tooltipVisible,
              onToggleTooltip: handleToggleTooltip,
              onOpenToolDetails: handleOpenToolDetails,
              onOpenProviderDetails: handleOpenProviderDetails,
            };
      nextNodeDataCache.set(node.id, nextData);

      return {
        id: node.id,
        type: "graphNode",
        position: node.position,
        selected: node.id === selectedNodeId,
        sourcePosition: "right" as Position,
        targetPosition: "left" as Position,
        data: nextData,
        style: NODE_STYLE,
      };
    });
    nodeDataCacheRef.current = nextNodeDataCache;
    return nextNodes;
  }, [catalog, graph, handleOpenProviderDetails, handleOpenToolDetails, handleToggleTooltip, runState, selectedNodeId, tooltipNodeId]);

  const edges = useMemo<FlowEdge[]>(() => {
    if (!graph) {
      return [];
    }
    const nodeLookup = new Map(graph.nodes.map((node) => [node.id, node]));
    return graph.edges.map((edge) => ({
      id: edge.id,
      type: "graphEdge",
      source: edge.source_id,
      target: edge.target_id,
      selected: edge.id === selectedEdgeId,
      markerEnd: {
        type: MarkerType.ArrowClosed,
        width: 14,
        height: 14,
        color: edge.kind === "conditional" ? "#b08cff" : "#6ea8ff",
      },
      label: edge.condition?.label ?? edge.label,
      animated: false,
      data: {
        kind: edge.kind,
        labelOffset:
          (() => {
            const siblingEdges = graph.edges
              .filter((candidate) => candidate.target_id === edge.target_id)
              .sort((left, right) => {
                const leftSource = nodeLookup.get(left.source_id);
                const rightSource = nodeLookup.get(right.source_id);
                const verticalDelta = (leftSource?.position.y ?? 0) - (rightSource?.position.y ?? 0);
                if (verticalDelta !== 0) {
                  return verticalDelta;
                }
                const horizontalDelta = (leftSource?.position.x ?? 0) - (rightSource?.position.x ?? 0);
                if (horizontalDelta !== 0) {
                  return horizontalDelta;
                }
                return left.id.localeCompare(right.id);
              });
            const index = siblingEdges.findIndex((candidate) => candidate.id === edge.id);
            return (index - (siblingEdges.length - 1) / 2) * 28;
          })(),
      },
      style: {
        stroke: edge.kind === "conditional" ? "#b08cff" : "#6ea8ff",
        strokeWidth: edge.id === selectedEdgeId ? 1.9 : 1.5,
        strokeDasharray: edge.kind === "conditional" ? "4 5" : undefined,
        strokeLinecap: "round",
        filter:
          edge.id === selectedEdgeId
            ? "drop-shadow(0 0 7px rgba(255, 99, 196, 0.25)) drop-shadow(0 0 10px rgba(92, 146, 255, 0.22))"
            : "drop-shadow(0 0 7px rgba(111, 133, 255, 0.18))",
      },
    }));
  }, [graph, selectedEdgeId]);

  const onNodesChange = useCallback(
    (changes: NodeChange[]) => {
      if (!graph) {
        return;
      }
      let nextGraph = graph;
      let hasStructuralChange = false;
      let isDragging = false;
      let dragEnded = false;

      changes.forEach((change) => {
        if (change.type === "position" && change.position) {
          const dragging = (change as { dragging?: boolean }).dragging;
          if (dragging === true) {
            isDragging = true;
          }
          if (dragging === false) {
            dragEnded = true;
          }
          nextGraph = {
            ...nextGraph,
            nodes: nextGraph.nodes.map((node) =>
              node.id === change.id ? { ...node, position: change.position ?? node.position } : node,
            ),
          };
        }
        if (change.type === "remove") {
          hasStructuralChange = true;
          nextGraph = {
            ...nextGraph,
            nodes: nextGraph.nodes.filter((node) => node.id !== change.id),
            edges: nextGraph.edges.filter((edge) => edge.source_id !== change.id && edge.target_id !== change.id),
            start_node_id: nextGraph.start_node_id === change.id ? "" : nextGraph.start_node_id,
          };
          if (selectedNodeId === change.id) {
            onSelectionChange(null, null);
          }
        }
        if (change.type === "select" && change.selected) {
          onSelectionChange(change.id, null);
        }
      });

      if (hasStructuralChange || dragEnded || !isDragging) {
        cancelPendingDragFrame();
        onGraphChange(nextGraph);
      } else {
        scheduleDragGraphUpdate(nextGraph);
      }
    },
    [cancelPendingDragFrame, graph, onGraphChange, onSelectionChange, scheduleDragGraphUpdate, selectedNodeId],
  );

  const onEdgesChange = useCallback(
    (changes: EdgeChange[]) => {
      if (!graph) {
        return;
      }
      let nextGraph = graph;
      changes.forEach((change) => {
        if (change.type === "remove") {
          nextGraph = {
            ...nextGraph,
            edges: nextGraph.edges.filter((edge) => edge.id !== change.id),
          };
          if (selectedEdgeId === change.id) {
            onSelectionChange(null, null);
          }
        }
        if (change.type === "select" && change.selected) {
          onSelectionChange(null, change.id);
        }
      });
      onGraphChange(nextGraph);
    },
    [graph, onGraphChange, onSelectionChange, selectedEdgeId],
  );

  const isValidConnection = useCallback(
    (connection: Connection) => {
      if (!graph || !connection.source || !connection.target) {
        return false;
      }
      const sourceNode = graph.nodes.find((node) => node.id === connection.source);
      const targetNode = graph.nodes.find((node) => node.id === connection.target);
      return canConnectNodes(sourceNode, targetNode, catalog);
    },
    [catalog, graph],
  );

  const onConnect = useCallback(
    (connection: Connection) => {
      if (!graph || !connection.source || !connection.target) {
        return;
      }
      const sourceNode = graph.nodes.find((node) => node.id === connection.source);
      const targetNode = graph.nodes.find((node) => node.id === connection.target);
      if (!canConnectNodes(sourceNode, targetNode, catalog)) {
        setEditorMessage("That node connection is not allowed by the category contract matrix.");
        return;
      }
      const sourceEdges = graph.edges.filter((edge) => edge.source_id === connection.source);
      const hasStandardOutgoing = sourceEdges.some((edge) => edge.kind === "standard");
      const nextEdgeId = `edge-${connection.source}-${connection.target}-${Date.now()}`;
      onGraphChange({
        ...graph,
        edges: [
          ...graph.edges,
          {
            id: nextEdgeId,
            source_id: connection.source,
            target_id: connection.target,
            label: hasStandardOutgoing ? "conditional route" : "next",
            kind: hasStandardOutgoing ? "conditional" : "standard",
            priority: hasStandardOutgoing ? 10 : 100,
            condition: hasStandardOutgoing ? defaultConditionalCondition(nextEdgeId) : null,
          },
        ],
      });
      setEditorMessage(null);
    },
    [catalog, graph, onGraphChange],
  );

  const onDrop = useCallback(
    (event: React.DragEvent<HTMLDivElement>) => {
      event.preventDefault();
      setIsProviderDragActive(false);
      const rawProvider = event.dataTransfer.getData("application/graph-node-provider");
      if (!rawProvider) {
        return;
      }
      const provider = JSON.parse(rawProvider) as NodeProviderDefinition;
      addProviderNode(provider, { x: event.clientX, y: event.clientY });
    },
    [addProviderNode],
  );

  const onDragOver = useCallback((event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    if (Array.from(event.dataTransfer.types).includes("application/graph-node-provider")) {
      setIsProviderDragActive(true);
    }
    event.dataTransfer.dropEffect = "copy";
  }, []);

  const onDragLeave = useCallback((event: React.DragEvent<HTMLDivElement>) => {
    if (event.currentTarget === event.target) {
      setIsProviderDragActive(false);
    }
  }, []);

  const toolDetailsNode =
    toolDetailsNodeId && graph ? graph.nodes.find((node) => node.id === toolDetailsNodeId && node.kind === "tool") ?? null : null;
  const providerDetailsNode =
    providerDetailsNodeId && graph ? graph.nodes.find((node) => node.id === providerDetailsNodeId && node.kind === "model") ?? null : null;

  if (!graph) {
    return <div className="panel empty-panel">No graph selected.</div>;
  }

  return (
    <div className="graph-shell panel">
      <div className="graph-shell-header panel-header">
        <div>
          <h2>{graph.name}</h2>
          <p>
            Drag node cards into the canvas and use the inspector to configure prompts, providers, tools, and routing.
          </p>
          {editorMessage ? <p className="editor-message">{editorMessage}</p> : null}
        </div>
        <div className="graph-scroll-nav">
          <button type="button" className="secondary-button graph-scroll-nav-button" onClick={onScrollToTop}>
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <path d="M12 5 5 12M12 5l7 7M12 5v14" />
            </svg>
            <span>To Controls</span>
          </button>
        </div>
      </div>
      <div className="graph-workspace">
        <div
          ref={canvasRef}
          className={`graph-canvas${drawerOpen ? " graph-canvas--drawer-open" : ""}${isProviderDragActive ? " is-drop-target" : ""}${isConnecting ? " is-connecting" : ""}`}
          onDrop={onDrop}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
        >
          <div className="graph-toolbar" aria-label="Graph visualizer actions">
            <DrawerActionButton tab="add" activeTab={drawerTab} drawerOpen={drawerOpen} label="Open add node drawer" onClick={() => openDrawerTab("add")}>
              <svg viewBox="0 0 24 24" aria-hidden="true">
                <path d="M12 5v14M5 12h14" />
              </svg>
              <span>Add</span>
            </DrawerActionButton>
            <DrawerActionButton
              tab="inspect"
              activeTab={drawerTab}
              drawerOpen={drawerOpen}
              label="Open details drawer"
              onClick={() => openDrawerTab("inspect")}
            >
              <svg viewBox="0 0 24 24" aria-hidden="true">
                <path d="M4 7h16M7 12h10M10 17h4" />
              </svg>
              <span>Details</span>
            </DrawerActionButton>
            <DrawerActionButton tab="run" activeTab={drawerTab} drawerOpen={drawerOpen} label="Open run drawer" onClick={() => openDrawerTab("run")}>
              <svg viewBox="0 0 24 24" aria-hidden="true">
                <path d="M8 6.5v11l9-5.5-9-5.5Z" />
              </svg>
              <span>Run</span>
            </DrawerActionButton>
            <button
              type="button"
              className={`graph-toolbar-button${panLocked ? " is-active" : ""}`}
              onClick={() => setPanLocked((v) => !v)}
              aria-label={panLocked ? "Unlock scroll panning" : "Lock to WASD only"}
              title={panLocked ? "Unlock scroll panning" : "Lock to WASD only"}
            >
              <svg viewBox="0 0 24 24" aria-hidden="true">
                {panLocked ? (
                  <path d="M17 11V8A5 5 0 0 0 7 8v3M5 11h14a1 1 0 0 1 1 1v8a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1v-8a1 1 0 0 1 1-1Zm7 4v3" />
                ) : (
                  <path d="M7 11V8a5 5 0 0 1 9.9-1M5 11h14a1 1 0 0 1 1 1v8a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1v-8a1 1 0 0 1 1-1Zm7 4v3" />
                )}
              </svg>
              <span>{panLocked ? "Locked" : "Lock"}</span>
            </button>
          </div>
          {graph.nodes.length === 0 ? (
            <div className="graph-empty-state">
              <strong>Start building your graph</strong>
              <p>Open the add menu and drop nodes directly into the canvas.</p>
              <button type="button" onClick={() => openDrawerTab("add")}>
                Add Your First Node
              </button>
            </div>
          ) : null}
          {isProviderDragActive ? (
            <div className="graph-drop-hint">
              <strong>Drop to add node</strong>
              <span>Release anywhere in the visualizer to place the node.</span>
            </div>
          ) : null}
          <ReactFlow
            nodes={nodes}
            edges={edges}
            nodeTypes={nodeTypes}
            edgeTypes={edgeTypes}
            connectionLineType={ConnectionLineType.SmoothStep}
            connectionLineStyle={{
              stroke: "#9ab7ff",
              strokeWidth: 2.5,
              strokeDasharray: "6 6",
              filter: "drop-shadow(0 0 10px rgba(111, 130, 255, 0.45))",
            }}
            defaultViewport={{ x: 0, y: 0, zoom: 1 }}
            panOnDrag={!panLocked}
            panOnScroll={false}
            zoomOnScroll={!panLocked}
            zoomOnPinch={!panLocked}
            zoomOnDoubleClick={!panLocked}
            preventScrolling={!panLocked}
            onInit={handleFlowInit}
            onMove={handleViewportMove}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onConnectStart={() => setIsConnecting(true)}
            onConnectEnd={() => setIsConnecting(false)}
            isValidConnection={isValidConnection}
            onNodeClick={(_, node) => {
              onSelectionChange(node.id, null);
              setTooltipNodeId(null);
            }}
            onEdgeClick={(_, edge) => {
              onSelectionChange(null, edge.id);
              setTooltipNodeId(null);
            }}
            onPaneClick={() => {
              onSelectionChange(null, null);
              setTooltipNodeId(null);
              setToolDetailsNodeId(null);
              setProviderDetailsNodeId(null);
              setDrawerOpen(false);
            }}
            fitView
          >
            <MiniMap
              zoomable
              pannable
              nodeColor={(node) => {
                const kindMatch = graph?.nodes.find((n) => n.id === node.id);
                return kindMatch ? (KIND_COLORS[kindMatch.kind] ?? "#8486a5") : "#8486a5";
              }}
              maskColor="rgba(13, 14, 24, 0.7)"
            />
            <Controls onZoomIn={handleZoomIn} onZoomOut={handleZoomOut} onFitView={handleFitView} />
            <Background variant={BackgroundVariant.Dots} color="#2e2f48" gap={20} size={1} />
          </ReactFlow>
        </div>
        <aside className={`graph-drawer${drawerOpen ? " is-open" : ""}`}>
          <div className="graph-drawer-panel">
            <div className="graph-drawer-panel-header">
              <div>
                <h3>
                  {drawerTab === "add" ? "Add Nodes" : drawerTab === "inspect" ? "Details" : "Run Details"}
                </h3>
                <p>
                  {drawerTab === "add"
                    ? "Search nodes, drag them into the canvas, or click to insert near the center."
                    : drawerTab === "inspect"
                      ? "Configure the selected graph object without leaving the visual workspace."
                      : "Review the latest runtime snapshot and execution timeline."}
                </p>
              </div>
              <button type="button" className="secondary-button graph-drawer-close" onClick={() => setDrawerOpen(false)}>
                Close
              </button>
            </div>
            <div className="graph-drawer-panel-body">
              {drawerTab === "add" ? (
                <ProviderSummary
                  catalog={catalog}
                  variant="drawer"
                  query={providerQuery}
                  onQueryChange={setProviderQuery}
                  onProviderClick={addProviderNode}
                />
              ) : null}
              {drawerTab === "inspect" ? (
                <GraphInspector
                  graph={graph}
                  catalog={catalog}
                  selectedNodeId={selectedNodeId}
                  selectedEdgeId={selectedEdgeId}
                  onGraphChange={onGraphChange}
                  onOpenProviderDetails={handleOpenProviderDetails}
                />
              ) : null}
              {drawerTab === "run" ? (
                <div className="graph-run-drawer">
                  <section className="graph-run-launch">
                    <div className="graph-run-launch-copy">
                      <strong>Launch the current draft</strong>
                      <span>Run the graph from this drawer and watch events stream below.</span>
                    </div>
                    <button type="button" onClick={onRunGraph} disabled={isRunning}>
                      {isRunning ? "Running..." : "Run Graph"}
                    </button>
                  </section>
                  <section className="panel graph-run-summary">
                    <div className="panel-header">
                      <h2>Run State</h2>
                      <p>{activeRunId ? `Latest run: ${activeRunId}` : "No active run."}</p>
                    </div>
                    <dl className="state-grid">
                      <div>
                        <dt>Status</dt>
                        <dd>{runState?.status ?? "idle"}</dd>
                      </div>
                      <div>
                        <dt>Current Node</dt>
                        <dd>{runState?.current_node_id ?? "n/a"}</dd>
                      </div>
                      <div>
                        <dt>Visited Nodes</dt>
                        <dd>{Object.keys(runState?.visit_counts ?? {}).length}</dd>
                      </div>
                      <div>
                        <dt>Transitions</dt>
                        <dd>{runState?.transition_history.length ?? 0}</dd>
                      </div>
                    </dl>
                    <div className="json-block">
                      <strong>Final Output</strong>
                      <pre>{JSON.stringify(runState?.final_output ?? null, null, 2)}</pre>
                    </div>
                    <div className="json-block">
                      <strong>Errors</strong>
                      <pre>{JSON.stringify(runState?.node_errors ?? {}, null, 2)}</pre>
                    </div>
                  </section>
                  <EventTimeline events={events} embedded />
                </div>
              ) : null}
            </div>
          </div>
        </aside>
      </div>
      {toolDetailsNode ? (
        <ToolDetailsModal
          graph={graph}
          node={toolDetailsNode}
          catalog={catalog}
          onGraphChange={onGraphChange}
          onClose={() => setToolDetailsNodeId(null)}
        />
      ) : null}
      {providerDetailsNode ? (
        <ProviderDetailsModal
          graph={graph}
          node={providerDetailsNode}
          catalog={catalog}
          onGraphChange={onGraphChange}
          onClose={() => setProviderDetailsNodeId(null)}
        />
      ) : null}
    </div>
  );
}
