import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { CSSProperties, MouseEvent as ReactMouseEvent, TouchEvent as ReactTouchEvent } from "react";
import type { Connection, Edge as FlowEdge, EdgeChange, Node as FlowNode, NodeChange, OnConnectStartParams, Position, ReactFlowInstance } from "reactflow";
import { Background, BackgroundVariant, ConnectionLineType, Controls, MarkerType, MiniMap, ReactFlow } from "reactflow";
import "reactflow/dist/style.css";

import { EventTimeline } from "./EventTimeline";
import {
  buildOrthogonalPolylinePoints,
  buildPolylinePath,
  GraphCanvasConnectionLine,
  GraphCanvasEdge,
  getEdgeLabelPlacement,
} from "./GraphCanvasEdge";
import type { GraphCanvasEdgeData } from "./GraphCanvasEdge";
import { GraphInspector } from "./GraphInspector";
import { GraphCanvasNode } from "./GraphCanvasNode";
import type { GraphCanvasNodeData } from "./GraphCanvasNode";
import { ProviderSummary } from "./ProviderSummary";
import { ProviderDetailsModal } from "./ProviderDetailsModal";
import { ToolDetailsModal } from "./ToolDetailsModal";
import {
  canConnectNodes,
  createNodeFromProvider,
  createNodeFromSaved,
  createWireJunctionNode,
  defaultConditionalCondition,
  defaultToolFailureCondition,
  getToolSourceHandleAnchorRatio,
  inferToolEdgeSourceHandle,
  isWireJunctionNode,
  TOOL_FAILURE_HANDLE_ID,
  TOOL_SUCCESS_HANDLE_ID,
} from "../lib/editor";
import { deleteSavedNode, getSavedNodes, saveNodeToLibrary } from "../lib/savedNodes";
import type { SavedNode } from "../lib/savedNodes";
import type { EditorCatalog, GraphDefinition, GraphEdge, GraphNode, GraphPosition, NodeProviderDefinition, RunState, RuntimeEvent } from "../lib/types";

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

type AddCanvasNodeOptions = {
  screenPosition?: { x: number; y: number };
  openDrawer?: boolean;
  nextDrawerTab?: DrawerTab;
};

type QuickAddSlot = {
  hotkey: string;
  label: string;
  description: string;
  category: string;
  preferredProviderIds: string[];
};

type ViewportState = {
  x: number;
  y: number;
  zoom: number;
};

type DraftConnectionState = {
  sourceNodeId: string;
  sourceHandleId: string | null;
  waypoints: GraphPosition[];
  pointerPosition: GraphPosition;
};

type HandleClickType = "source" | "target";

type NodeRegionBounds = {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
};

type PlacementState =
  | {
      kind: "provider";
      provider: NodeProviderDefinition;
      position: GraphPosition;
    }
  | {
      kind: "saved";
      saved: SavedNode;
      position: GraphPosition;
    };

const KIND_COLORS: Record<string, string> = {
  input: "#8486a5",
  model: "#6c5ce7",
  provider: "#f59e0b",
  tool: "#a78bfa",
  data: "#2dd4bf",
  output: "#4ade80",
};

const KIND_LABELS: Record<string, string> = {
  input: "IN",
  model: "AI",
  provider: "PR",
  tool: "FX",
  data: "DB",
  output: "OUT",
};

const nodeTypes = { graphNode: GraphCanvasNode };
const edgeTypes = { graphEdge: GraphCanvasEdge };
const NODE_STYLE = {
  width: "max-content",
  minWidth: 280,
  background: "transparent",
  border: "none",
  padding: 0,
  boxShadow: "none",
} as const;
const NODE_WIDTH = 280;
const NODE_HEIGHT = 150;
const NODE_REGION_HEIGHT = 168;
const JUNCTION_NODE_SIZE = 24;
const MIN_POINTER_PAN_DAMPING = 0.08;
const VIEWPORT_SYNC_EPSILON = 0.25;
const EDGE_STROKE_WIDTH = 3.6;
const SELECTED_EDGE_STROKE_WIDTH = 4.3;
const DRAFT_WIRE_STROKE_WIDTH = 4.6;
const EDGE_LANE_SPACING = 12;
const EDGE_SIBLING_SPACING = 8;
const EDGE_LABEL_HEIGHT = 26;
const EDGE_LABEL_HORIZONTAL_PADDING = 24;
const EDGE_LABEL_CHARACTER_WIDTH = 7.1;
const EDGE_LABEL_COLLISION_GAP = 10;
const EDGE_LABEL_COLLISION_STEP = 24;
const EDGE_LABEL_COLLISION_ATTEMPTS = 8;
const EDGE_LABEL_EDGE_GAP = 12;
const EDGE_PATH_SAMPLE_STEP = 18;
const CONNECTION_HANDLE_SNAP_RADIUS_PX = 44;
const DRAFT_WIRE_MIN_SEGMENT_PX = 42;
const TOOL_EDGE_TONES = {
  success: {
    sourceColor: "#74f0b8",
    targetColor: "#3fd39e",
    markerColor: "#74f0b8",
    routeTone: "tool-success" as const,
  },
  failure: {
    sourceColor: "#ff8a8a",
    targetColor: "#ff6b9f",
    markerColor: "#ff8a8a",
    routeTone: "tool-failure" as const,
  },
};
const JUNCTION_NODE_STYLE = {
  width: JUNCTION_NODE_SIZE,
  height: JUNCTION_NODE_SIZE,
  background: "transparent",
  border: "none",
  padding: 0,
  boxShadow: "none",
} as const;

type LabelCollisionCandidate = {
  edgeId: string;
  center: GraphPosition;
  tangent: GraphPosition;
  width: number;
  height: number;
};

type LabelBounds = {
  left: number;
  right: number;
  top: number;
  bottom: number;
};

type DraftConnectionSnapTarget = {
  nodeId: string;
  handleId: string | null;
  anchor: GraphPosition;
  distance: number;
};

type EdgePathSamplePoint = GraphPosition & {
  edgeId: string;
};

type JunctionDragState = {
  nodeId: string;
  pointerOffset: GraphPosition;
};

type GraphCanvasRuntimeNodeData = GraphCanvasNodeData & {
  isConnectionMagnetized?: boolean;
};

function estimateEdgeLabelWidth(label: string) {
  const trimmedLabel = label.trim();
  const textWidth = Math.max(trimmedLabel.length, 1) * EDGE_LABEL_CHARACTER_WIDTH;
  return Math.max(56, Math.round(textWidth + EDGE_LABEL_HORIZONTAL_PADDING));
}

function getLabelBounds(center: GraphPosition, width: number, height: number): LabelBounds {
  return {
    left: center.x - width / 2,
    right: center.x + width / 2,
    top: center.y - height / 2,
    bottom: center.y + height / 2,
  };
}

function labelBoundsOverlap(left: LabelBounds, right: LabelBounds, gap: number) {
  return !(
    left.right + gap <= right.left ||
    left.left >= right.right + gap ||
    left.bottom + gap <= right.top ||
    left.top >= right.bottom + gap
  );
}

function sampleEdgePathPoints(edgeId: string, edgePath: string, shiftX = 0, shiftY = 0): EdgePathSamplePoint[] {
  if (typeof document === "undefined") {
    return [];
  }

  try {
    const measurementPath = document.createElementNS("http://www.w3.org/2000/svg", "path");
    measurementPath.setAttribute("d", edgePath);
    const totalLength = measurementPath.getTotalLength();
    if (!Number.isFinite(totalLength) || totalLength < 0.001) {
      const point = measurementPath.getPointAtLength(0);
      return [{ edgeId, x: point.x + shiftX, y: point.y + shiftY }];
    }

    const stepCount = Math.max(2, Math.ceil(totalLength / EDGE_PATH_SAMPLE_STEP));
    const points: EdgePathSamplePoint[] = [];
    for (let index = 0; index <= stepCount; index += 1) {
      const point = measurementPath.getPointAtLength((totalLength * index) / stepCount);
      points.push({
        edgeId,
        x: point.x + shiftX,
        y: point.y + shiftY,
      });
    }
    return points;
  } catch {
    return [];
  }
}

function labelBoundsOverlapEdgePath(bounds: LabelBounds, points: EdgePathSamplePoint[], edgeId: string, gap: number) {
  return points.some((point) => {
    if (point.edgeId === edgeId) {
      return false;
    }
    return (
      point.x >= bounds.left - gap &&
      point.x <= bounds.right + gap &&
      point.y >= bounds.top - gap &&
      point.y <= bounds.bottom + gap
    );
  });
}

function resolveEdgeLabelShifts(candidates: LabelCollisionCandidate[], edgePathPoints: EdgePathSamplePoint[]) {
  const placedBounds: Array<LabelBounds & { edgeId: string }> = [];
  const shifts = new Map<string, { x: number; y: number }>();
  const orderedCandidates = [...candidates].sort((left, right) => {
    const verticalDelta = left.center.y - right.center.y;
    if (Math.abs(verticalDelta) > 0.01) {
      return verticalDelta;
    }
    const horizontalDelta = left.center.x - right.center.x;
    if (Math.abs(horizontalDelta) > 0.01) {
      return horizontalDelta;
    }
    return left.edgeId.localeCompare(right.edgeId);
  });

  orderedCandidates.forEach((candidate) => {
    const tangentLength = Math.hypot(candidate.tangent.x, candidate.tangent.y);
    const normal =
      tangentLength < 0.001
        ? { x: 0, y: -1 }
        : {
            x: -candidate.tangent.y / tangentLength,
            y: candidate.tangent.x / tangentLength,
          };

    let acceptedCenter = candidate.center;
    let acceptedBounds = getLabelBounds(candidate.center, candidate.width, candidate.height);

    for (let attempt = 0; attempt <= EDGE_LABEL_COLLISION_ATTEMPTS; attempt += 1) {
      const laneIndex = attempt === 0 ? 0 : Math.ceil(attempt / 2) * (attempt % 2 === 1 ? 1 : -1);
      const candidateCenter =
        laneIndex === 0
          ? candidate.center
          : {
              x: candidate.center.x + normal.x * EDGE_LABEL_COLLISION_STEP * laneIndex,
              y: candidate.center.y + normal.y * EDGE_LABEL_COLLISION_STEP * laneIndex,
            };
      const candidateBounds = getLabelBounds(candidateCenter, candidate.width, candidate.height);

      if (
        !placedBounds.some((placed) => labelBoundsOverlap(candidateBounds, placed, EDGE_LABEL_COLLISION_GAP)) &&
        !labelBoundsOverlapEdgePath(candidateBounds, edgePathPoints, candidate.edgeId, EDGE_LABEL_EDGE_GAP)
      ) {
        acceptedCenter = candidateCenter;
        acceptedBounds = candidateBounds;
        break;
      }
    }

    shifts.set(candidate.edgeId, {
      x: acceptedCenter.x - candidate.center.x,
      y: acceptedCenter.y - candidate.center.y,
    });
    placedBounds.push({
      edgeId: candidate.edgeId,
      ...acceptedBounds,
    });
  });

  return shifts;
}

const PLACEMENT_UI_SELECTOR = ".graph-toolbar, .graph-quick-add, .graph-hotkey-guide, .react-flow__controls, .react-flow__minimap";

const QUICK_ADD_SLOTS: QuickAddSlot[] = [
  {
    hotkey: "1",
    label: "Input",
    description: "Create a start/input node.",
    category: "start",
    preferredProviderIds: ["core.input"],
  },
  {
    hotkey: "2",
    label: "API",
    description: "Create an API/model node.",
    category: "api",
    preferredProviderIds: ["core.api"],
  },
  {
    hotkey: "3",
    label: "Tool",
    description: "Create a tool node.",
    category: "tool",
    preferredProviderIds: ["tool.registry"],
  },
  {
    hotkey: "4",
    label: "Data",
    description: "Create a data node.",
    category: "data",
    preferredProviderIds: [],
  },
  {
    hotkey: "5",
    label: "Output",
    description: "Create an end/output node.",
    category: "end",
    preferredProviderIds: ["core.output"],
  },
];

function resolveQuickAddProvider(providers: NodeProviderDefinition[], slot: QuickAddSlot): NodeProviderDefinition | null {
  for (const providerId of slot.preferredProviderIds) {
    const preferred = providers.find((provider) => provider.provider_id === providerId);
    if (preferred) {
      return preferred;
    }
  }
  return providers.find((provider) => provider.category === slot.category) ?? null;
}

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
  const [isSavedNodeDragActive, setIsSavedNodeDragActive] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [savedNodes, setSavedNodes] = useState<SavedNode[]>(() => getSavedNodes());
  const [panLocked, setPanLocked] = useState(false);
  const [isCommandHeld, setIsCommandHeld] = useState(false);
  const [showHotkeys, setShowHotkeys] = useState(false);
  const [quickAddMinimized, setQuickAddMinimized] = useState(false);
  const [junctionDrag, setJunctionDrag] = useState<JunctionDragState | null>(null);
  const [isNodeDragActive, setIsNodeDragActive] = useState(false);
  const [pendingPlacement, setPendingPlacement] = useState<PlacementState | null>(null);
  const [draftConnection, setDraftConnection] = useState<DraftConnectionState | null>(null);
  const [draftConnectionSnapTargetNodeId, setDraftConnectionSnapTargetNodeId] = useState<string | null>(null);
  const [viewportState, setViewportState] = useState<ViewportState>({ x: 0, y: 0, zoom: 1 });
  const canvasRef = useRef<HTMLDivElement | null>(null);
  const canvasZoomRef = useRef(1);
  const graphRef = useRef<GraphDefinition | null>(graph);
  const draftConnectionRef = useRef<DraftConnectionState | null>(draftConnection);
  const draftWirePathRef = useRef<SVGPathElement | null>(null);
  const draftPreviewPointerPositionRef = useRef<GraphPosition | null>(null);
  const draftSnapTargetNodeIdRef = useRef<string | null>(null);
  const isConnectingRef = useRef(isConnecting);
  const nodeDataCacheRef = useRef(new Map<string, GraphCanvasRuntimeNodeData>());
  const dragFrameRef = useRef<number | null>(null);
  const pendingDragGraphRef = useRef<GraphDefinition | null>(null);
  const draftPointerFrameRef = useRef<number | null>(null);
  const pendingDraftPointerPositionRef = useRef<GraphPosition | null>(null);
  const pointerPanStateRef = useRef<{
    lastPointerViewport: ViewportState | null;
    lastAppliedViewport: ViewportState | null;
    isSyncingViewport: boolean;
  }>({
    lastPointerViewport: null,
    lastAppliedViewport: null,
    isSyncingViewport: false,
  });
  const didCreateConnectionRef = useRef(false);
  const suppressNextPaneClickRef = useRef(false);
  const zoomAnimationDuration = 120;
  const vizLocked = panLocked || isCommandHeld;

  useEffect(() => {
    graphRef.current = graph;
  }, [graph]);

  useEffect(() => {
    draftConnectionRef.current = draftConnection;
    draftPreviewPointerPositionRef.current = draftConnection?.pointerPosition ?? null;
    if (!draftConnection) {
      draftSnapTargetNodeIdRef.current = null;
      setDraftConnectionSnapTargetNodeId(null);
    }
  }, [draftConnection]);

  useEffect(() => {
    isConnectingRef.current = isConnecting;
    if (isConnecting) {
      draftSnapTargetNodeIdRef.current = null;
      setDraftConnectionSnapTargetNodeId(null);
    }
  }, [isConnecting]);

  const isEditableTarget = useCallback((target: EventTarget | null): boolean => {
    if (!(target instanceof HTMLElement)) {
      return false;
    }
    const tag = target.tagName;
    return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || target.isContentEditable;
  }, []);

  const setCanvasZoom = useCallback((zoom: number) => {
    const nextZoom = Number.isFinite(zoom) && zoom > 0 ? zoom : 1;
    if (Math.abs(canvasZoomRef.current - nextZoom) < 0.0001) {
      return;
    }
    canvasZoomRef.current = nextZoom;
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
    const viewport = instance.getViewport();
    setFlowInstance(instance);
    setCanvasZoom(viewport.zoom);
    setViewportState(viewport);
    pointerPanStateRef.current = {
      lastPointerViewport: viewport,
      lastAppliedViewport: viewport,
      isSyncingViewport: false,
    };
  }, [setCanvasZoom]);

  const getNodeDimensions = useCallback((node: GraphNode) => {
    if (isWireJunctionNode(node)) {
      return { width: JUNCTION_NODE_SIZE, height: JUNCTION_NODE_SIZE, regionHeight: JUNCTION_NODE_SIZE };
    }
    return { width: NODE_WIDTH, height: NODE_HEIGHT, regionHeight: NODE_REGION_HEIGHT };
  }, []);

  const nodeRegionBounds = useMemo<NodeRegionBounds | null>(() => {
    if (!graph || graph.nodes.length === 0) {
      return null;
    }
    return graph.nodes.reduce<NodeRegionBounds>(
      (bounds, node) => {
        const dimensions = getNodeDimensions(node);
        return {
          minX: Math.min(bounds.minX, node.position.x),
          maxX: Math.max(bounds.maxX, node.position.x + dimensions.width),
          minY: Math.min(bounds.minY, node.position.y),
          maxY: Math.max(bounds.maxY, node.position.y + dimensions.regionHeight),
        };
      },
      {
        minX: Number.POSITIVE_INFINITY,
        maxX: Number.NEGATIVE_INFINITY,
        minY: Number.POSITIVE_INFINITY,
        maxY: Number.NEGATIVE_INFINITY,
      },
    );
  }, [getNodeDimensions, graph]);

  const getPointerPanDamping = useCallback(
    (viewport: ViewportState) => {
      if (!nodeRegionBounds) {
        return 1;
      }
      const bounds = canvasRef.current?.getBoundingClientRect();
      if (!bounds || viewport.zoom <= 0) {
        return 1;
      }
      const visibleWidth = bounds.width / viewport.zoom;
      const visibleHeight = bounds.height / viewport.zoom;
      const centerX = (bounds.width * 0.5 - viewport.x) / viewport.zoom;
      const centerY = (bounds.height * 0.5 - viewport.y) / viewport.zoom;
      const paddingX = Math.max(180, visibleWidth * 0.18);
      const paddingY = Math.max(140, visibleHeight * 0.18);
      const outsideX = Math.max(nodeRegionBounds.minX - paddingX - centerX, 0, centerX - (nodeRegionBounds.maxX + paddingX));
      const outsideY = Math.max(nodeRegionBounds.minY - paddingY - centerY, 0, centerY - (nodeRegionBounds.maxY + paddingY));
      const distance = Math.hypot(outsideX, outsideY);
      if (distance === 0) {
        return 1;
      }
      const falloffDistance = Math.max(180, Math.min(360, Math.max(visibleWidth, visibleHeight) * 0.22));
      const normalizedDistance = distance / falloffDistance;
      return MIN_POINTER_PAN_DAMPING + (1 - MIN_POINTER_PAN_DAMPING) * Math.exp(-(normalizedDistance * normalizedDistance));
    },
    [nodeRegionBounds],
  );

  const handleViewportMove = useCallback(
    (event: MouseEvent | TouchEvent | null, viewport: ViewportState) => {
      setCanvasZoom(viewport.zoom);
      setViewportState(viewport);

      const panState = pointerPanStateRef.current;
      if (panState.isSyncingViewport) {
        panState.isSyncingViewport = false;
        panState.lastPointerViewport = viewport;
        panState.lastAppliedViewport = viewport;
        return;
      }

      const previousPointerViewport = panState.lastPointerViewport;
      panState.lastPointerViewport = viewport;

      const isPointerPan = (event instanceof MouseEvent || event instanceof TouchEvent) && flowInstance !== null;
      if (!isPointerPan) {
        panState.lastAppliedViewport = viewport;
        return;
      }
      if (!previousPointerViewport || previousPointerViewport.zoom !== viewport.zoom) {
        panState.lastAppliedViewport = viewport;
        return;
      }

      const previousAppliedViewport = panState.lastAppliedViewport ?? previousPointerViewport;
      const rawDx = viewport.x - previousPointerViewport.x;
      const rawDy = viewport.y - previousPointerViewport.y;
      if (Math.abs(rawDx) < 0.01 && Math.abs(rawDy) < 0.01) {
        panState.lastAppliedViewport = viewport;
        return;
      }

      const damping = getPointerPanDamping(previousAppliedViewport);
      if (damping >= 0.999) {
        panState.lastAppliedViewport = viewport;
        return;
      }

      const adjustedViewport: ViewportState = {
        x: previousAppliedViewport.x + rawDx * damping,
        y: previousAppliedViewport.y + rawDy * damping,
        zoom: viewport.zoom,
      };
      panState.lastAppliedViewport = adjustedViewport;

      if (
        Math.abs(adjustedViewport.x - viewport.x) <= VIEWPORT_SYNC_EPSILON &&
        Math.abs(adjustedViewport.y - viewport.y) <= VIEWPORT_SYNC_EPSILON
      ) {
        return;
      }

      panState.isSyncingViewport = true;
      setViewportState(adjustedViewport);
      void flowInstance.setViewport(adjustedViewport);
    },
    [flowInstance, getPointerPanDamping, setCanvasZoom],
  );

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

  const clearCanvasChrome = useCallback(() => {
    onSelectionChange(null, null);
    setTooltipNodeId(null);
    setToolDetailsNodeId(null);
    setProviderDetailsNodeId(null);
    setDrawerOpen(false);
    setShowHotkeys(false);
  }, [onSelectionChange]);

  const getFallbackScreenPosition = useCallback(() => {
    const bounds = canvasRef.current?.getBoundingClientRect();
    if (bounds) {
      return {
        x: bounds.left + (drawerOpen ? Math.max(140, (bounds.width - 360) * 0.5) : bounds.width * 0.5),
        y: bounds.top + bounds.height * 0.5,
      };
    }
    return { x: window.innerWidth * 0.45, y: window.innerHeight * 0.5 };
  }, [drawerOpen]);

  const getFlowPositionFromScreen = useCallback(
    (screenPosition?: { x: number; y: number }) => {
      if (!flowInstance) {
        return null;
      }
      return flowInstance.screenToFlowPosition(screenPosition ?? getFallbackScreenPosition());
    },
    [flowInstance, getFallbackScreenPosition],
  );

  const buildPlacementNode = useCallback(
    (placement: PlacementState, nextGraph: GraphDefinition) => {
      if (placement.kind === "provider") {
        if (!catalog) {
          return null;
        }
        return createNodeFromProvider(nextGraph, placement.provider, catalog, placement.position);
      }
      return createNodeFromSaved(nextGraph, placement.saved, placement.position);
    },
    [catalog],
  );

  const cancelPlacement = useCallback((message?: string) => {
    setPendingPlacement(null);
    if (message) {
      setEditorMessage(message);
    }
  }, []);

  const cancelPendingDraftPointerFrame = useCallback(() => {
    if (draftPointerFrameRef.current !== null) {
      cancelAnimationFrame(draftPointerFrameRef.current);
      draftPointerFrameRef.current = null;
    }
    pendingDraftPointerPositionRef.current = null;
  }, []);

  const cancelDraftConnection = useCallback((message?: string) => {
    cancelPendingDraftPointerFrame();
    setDraftConnection(null);
    draftSnapTargetNodeIdRef.current = null;
    setDraftConnectionSnapTargetNodeId(null);
    didCreateConnectionRef.current = false;
    suppressNextPaneClickRef.current = false;
    setIsConnecting(false);
    if (message) {
      setEditorMessage(message);
    }
  }, [cancelPendingDraftPointerFrame]);

  const getEventClientPosition = useCallback((event: MouseEvent | TouchEvent | ReactMouseEvent | ReactTouchEvent) => {
    const nativeEvent = "nativeEvent" in event ? event.nativeEvent : event;
    if ("clientX" in nativeEvent) {
      return { x: nativeEvent.clientX, y: nativeEvent.clientY };
    }
    const touch = nativeEvent.touches[0] ?? nativeEvent.changedTouches[0];
    return touch ? { x: touch.clientX, y: touch.clientY } : null;
  }, []);

  const getSourceAnchorPosition = useCallback(
    (nodeId: string, sourceHandleId: string | null = null): GraphPosition | null => {
      const sourceNode = graph?.nodes.find((node) => node.id === nodeId);
      if (!sourceNode) {
        return null;
      }
      const dimensions = getNodeDimensions(sourceNode);
      const verticalRatio =
        sourceNode.kind === "tool" ? getToolSourceHandleAnchorRatio(sourceHandleId ?? TOOL_SUCCESS_HANDLE_ID) : 0.5;
      return {
        x: sourceNode.position.x + dimensions.width,
        y: sourceNode.position.y + dimensions.height * verticalRatio,
      };
    },
    [getNodeDimensions, graph],
  );

  const getTargetAnchorPosition = useCallback(
    (nodeId: string): GraphPosition | null => {
      const targetNode = graph?.nodes.find((node) => node.id === nodeId);
      if (!targetNode) {
        return null;
      }
      const dimensions = getNodeDimensions(targetNode);
      return {
        x: targetNode.position.x,
        y: targetNode.position.y + dimensions.height * 0.5,
      };
    },
    [getNodeDimensions, graph],
  );

  const findDraftConnectionSnapTarget = useCallback(
    (sourceNodeId: string, pointerPosition: GraphPosition): DraftConnectionSnapTarget | null => {
      if (!graph) {
        return null;
      }
      const sourceNode = graph.nodes.find((node) => node.id === sourceNodeId);
      if (!sourceNode) {
        return null;
      }

      const snapRadius = CONNECTION_HANDLE_SNAP_RADIUS_PX / Math.max(viewportState.zoom, 0.35);
      let bestTarget: DraftConnectionSnapTarget | null = null;

      graph.nodes.forEach((node) => {
        if (!canConnectNodes(sourceNode, node, catalog)) {
          return;
        }
        const anchor = getTargetAnchorPosition(node.id);
        if (!anchor) {
          return;
        }
        const distance = Math.hypot(anchor.x - pointerPosition.x, anchor.y - pointerPosition.y);
        if (distance > snapRadius) {
          return;
        }
        if (
          !bestTarget ||
          distance < bestTarget.distance ||
          (Math.abs(distance - bestTarget.distance) < 0.01 && node.id.localeCompare(bestTarget.nodeId) < 0)
        ) {
          bestTarget = {
            nodeId: node.id,
            handleId: null,
            anchor,
            distance,
          };
        }
      });

      return bestTarget;
    },
    [catalog, getTargetAnchorPosition, graph, viewportState.zoom],
  );

  const getEdgeRouteSignature = useCallback(
    (edge: GraphEdge, sourceNode: GraphNode | undefined) => {
      const sourceHandleId = inferToolEdgeSourceHandle(edge, sourceNode);
      const sourceAnchor = getSourceAnchorPosition(edge.source_id, sourceHandleId);
      const targetAnchor = getTargetAnchorPosition(edge.target_id);
      if (!sourceAnchor || !targetAnchor) {
        return null;
      }
      const routePoints = edge.waypoints?.length
        ? buildOrthogonalPolylinePoints([sourceAnchor, ...edge.waypoints, targetAnchor], { endWithHorizontal: true })
        : [sourceAnchor, targetAnchor];
      return routePoints
        .map((point) => `${Math.round(point.x / 12)}:${Math.round(point.y / 12)}`)
        .join("|");
    },
    [getSourceAnchorPosition, getTargetAnchorPosition],
  );

  const pruneDisconnectedWireJunctions = useCallback((baseGraph: GraphDefinition): GraphDefinition => {
    let nextGraph = baseGraph;
    let didPrune = true;
    while (didPrune) {
      didPrune = false;
      const removableJunctionIds = nextGraph.nodes
        .filter((node) => isWireJunctionNode(node))
        .filter((node) => {
          const incomingCount = nextGraph.edges.filter((edge) => edge.target_id === node.id).length;
          const outgoingCount = nextGraph.edges.filter((edge) => edge.source_id === node.id).length;
          return incomingCount === 0 || outgoingCount === 0;
        })
        .map((node) => node.id);
      if (removableJunctionIds.length === 0) {
        continue;
      }
      didPrune = true;
      const removableSet = new Set(removableJunctionIds);
      nextGraph = {
        ...nextGraph,
        nodes: nextGraph.nodes.filter((node) => !removableSet.has(node.id)),
        edges: nextGraph.edges.filter((edge) => !removableSet.has(edge.source_id) && !removableSet.has(edge.target_id)),
      };
    }
    return nextGraph;
  }, []);

  const removeEdgeAndPruneJunctions = useCallback(
    (baseGraph: GraphDefinition, edgeId: string) =>
      pruneDisconnectedWireJunctions({
        ...baseGraph,
        edges: baseGraph.edges.filter((edge) => edge.id !== edgeId),
      }),
    [pruneDisconnectedWireJunctions],
  );

  const flowToViewportPosition = useCallback(
    (position: GraphPosition): GraphPosition => ({
      x: viewportState.x + position.x * viewportState.zoom,
      y: viewportState.y + position.y * viewportState.zoom,
    }),
    [viewportState],
  );

  const renderDraftConnectionPreview = useCallback(
    (pointerPosition: GraphPosition | null, connectionOverride: DraftConnectionState | null = draftConnectionRef.current) => {
      const draftWirePathElement = draftWirePathRef.current;
      if (!draftWirePathElement || !pointerPosition || !connectionOverride || isConnectingRef.current) {
        if (draftWirePathElement) {
          draftWirePathElement.setAttribute("d", "");
        }
        if (draftSnapTargetNodeIdRef.current !== null) {
          draftSnapTargetNodeIdRef.current = null;
          setDraftConnectionSnapTargetNodeId(null);
        }
        return;
      }

      const snapTarget = findDraftConnectionSnapTarget(connectionOverride.sourceNodeId, pointerPosition);
      const sourceAnchor = getSourceAnchorPosition(connectionOverride.sourceNodeId, connectionOverride.sourceHandleId);
      if (!sourceAnchor) {
        draftWirePathElement.setAttribute("d", "");
        if (draftSnapTargetNodeIdRef.current !== null) {
          draftSnapTargetNodeIdRef.current = null;
          setDraftConnectionSnapTargetNodeId(null);
        }
        return;
      }

      const endpoint = snapTarget?.anchor ?? pointerPosition;
      const nextPath = buildPolylinePath(
        [sourceAnchor, ...connectionOverride.waypoints, endpoint].map((point) => flowToViewportPosition(point)),
        { endWithHorizontal: false },
      );
      if (draftWirePathElement.getAttribute("d") !== nextPath) {
        draftWirePathElement.setAttribute("d", nextPath);
      }

      const nextSnapTargetNodeId = snapTarget?.nodeId ?? null;
      if (draftSnapTargetNodeIdRef.current !== nextSnapTargetNodeId) {
        draftSnapTargetNodeIdRef.current = nextSnapTargetNodeId;
        setDraftConnectionSnapTargetNodeId(nextSnapTargetNodeId);
      }
    },
    [findDraftConnectionSnapTarget, flowToViewportPosition, getSourceAnchorPosition],
  );

  const scheduleDraftPointerPositionUpdate = useCallback(
    (pointerPosition: GraphPosition) => {
      pendingDraftPointerPositionRef.current = pointerPosition;
      if (draftPointerFrameRef.current !== null) {
        return;
      }
      draftPointerFrameRef.current = requestAnimationFrame(() => {
        draftPointerFrameRef.current = null;
        const nextPointerPosition = pendingDraftPointerPositionRef.current;
        pendingDraftPointerPositionRef.current = null;
        if (!nextPointerPosition) {
          return;
        }
        draftPreviewPointerPositionRef.current = nextPointerPosition;
        renderDraftConnectionPreview(nextPointerPosition);
      });
    },
    [renderDraftConnectionPreview],
  );

  const resolveDraftConnectionSnapTargetNodeId = useCallback(
    (sourceNodeId: string, fallbackPointerPosition: GraphPosition | null = null) => {
      if (draftSnapTargetNodeIdRef.current) {
        return draftSnapTargetNodeIdRef.current;
      }
      const pointerPosition =
        pendingDraftPointerPositionRef.current ?? draftPreviewPointerPositionRef.current ?? fallbackPointerPosition;
      if (!pointerPosition) {
        return null;
      }
      return findDraftConnectionSnapTarget(sourceNodeId, pointerPosition)?.nodeId ?? null;
    },
    [findDraftConnectionSnapTarget],
  );

  const buildCommittedEdge = useCallback(
    (
      baseGraph: GraphDefinition,
      sourceId: string,
      targetId: string,
      waypoints: GraphPosition[] = [],
      sourceHandleId: string | null = null,
    ): GraphEdge | null => {
      const sourceNode = baseGraph.nodes.find((node) => node.id === sourceId);
      const targetNode = baseGraph.nodes.find((node) => node.id === targetId);
      if (!canConnectNodes(sourceNode, targetNode, catalog)) {
        setEditorMessage("That node connection is not allowed by the category contract matrix.");
        return null;
      }
      const sourceEdges = baseGraph.edges.filter((edge) => edge.source_id === sourceId);
      const hasStandardOutgoing = sourceEdges.some((edge) => edge.kind === "standard");
      const nextEdgeId = `edge-${sourceId}-${targetId}-${Date.now()}`;
      if (sourceNode?.kind === "tool") {
        const effectiveHandleId = sourceHandleId === TOOL_FAILURE_HANDLE_ID ? TOOL_FAILURE_HANDLE_ID : TOOL_SUCCESS_HANDLE_ID;
        const hasSuccessOutgoing = sourceEdges.some(
          (edge) => inferToolEdgeSourceHandle(edge, sourceNode) === TOOL_SUCCESS_HANDLE_ID,
        );
        const hasFailureOutgoing = sourceEdges.some(
          (edge) => inferToolEdgeSourceHandle(edge, sourceNode) === TOOL_FAILURE_HANDLE_ID,
        );
        if (effectiveHandleId === TOOL_SUCCESS_HANDLE_ID && hasSuccessOutgoing) {
          setEditorMessage("Run Tool already has an on success route.");
          return null;
        }
        if (effectiveHandleId === TOOL_FAILURE_HANDLE_ID && hasFailureOutgoing) {
          setEditorMessage("Run Tool already has an on failure route.");
          return null;
        }
        return {
          id: nextEdgeId,
          source_id: sourceId,
          target_id: targetId,
          source_handle_id: effectiveHandleId,
          label: effectiveHandleId === TOOL_FAILURE_HANDLE_ID ? "on failure" : "on success",
          kind: effectiveHandleId === TOOL_FAILURE_HANDLE_ID ? "conditional" : "standard",
          priority: effectiveHandleId === TOOL_FAILURE_HANDLE_ID ? 10 : 100,
          waypoints,
          condition: effectiveHandleId === TOOL_FAILURE_HANDLE_ID ? defaultToolFailureCondition(nextEdgeId) : null,
        };
      }
      return {
        id: nextEdgeId,
        source_id: sourceId,
        target_id: targetId,
        source_handle_id: sourceHandleId,
        label: hasStandardOutgoing ? "conditional route" : "next",
        kind: hasStandardOutgoing ? "conditional" : "standard",
        priority: hasStandardOutgoing ? 10 : 100,
        waypoints,
        condition: hasStandardOutgoing ? defaultConditionalCondition(nextEdgeId) : null,
      };
    },
    [catalog],
  );

  const commitEdge = useCallback(
    (
      sourceId: string,
      targetId: string,
      waypoints: GraphPosition[] = [],
      baseGraph: GraphDefinition | null = graph,
      sourceHandleId: string | null = null,
    ) => {
      if (!baseGraph) {
        return false;
      }
      const nextEdge = buildCommittedEdge(baseGraph, sourceId, targetId, waypoints, sourceHandleId);
      if (!nextEdge) {
        return false;
      }
      onGraphChange({
        ...baseGraph,
        edges: [...baseGraph.edges, nextEdge],
      });
      setEditorMessage(null);
      return true;
    },
    [buildCommittedEdge, graph, onGraphChange],
  );

  const handleNodeHandlePointerDown = useCallback(
    (nodeId: string, handleType: HandleClickType, handleId: string | null) => {
      if (!graph || pendingPlacement) {
        return false;
      }

      const sourceNode = graph.nodes.find((node) => node.id === nodeId);
      const sourceAnchor = getSourceAnchorPosition(nodeId, handleId);
      const targetAnchor = getTargetAnchorPosition(nodeId);
      if (!sourceNode) {
        return false;
      }

      const candidateEdges =
        handleType === "source"
          ? graph.edges.filter((edge) => {
              if (edge.source_id !== nodeId) {
                return false;
              }
              if (sourceNode.kind === "tool") {
                return inferToolEdgeSourceHandle(edge, sourceNode) === (handleId ?? TOOL_SUCCESS_HANDLE_ID);
              }
              return true;
            })
          : graph.edges.filter((edge) => edge.target_id === nodeId);

      let edgeToRedraw: GraphEdge | null = null;
      if (candidateEdges.length === 1) {
        edgeToRedraw = candidateEdges[0];
      } else if (candidateEdges.length > 1 && selectedEdgeId) {
        edgeToRedraw = candidateEdges.find((edge) => edge.id === selectedEdgeId) ?? null;
      }

      if (candidateEdges.length > 1 && !edgeToRedraw) {
        setEditorMessage("This handle has multiple lines. Select the line you want to redraw, then click the handle again.");
        return true;
      }

      if (!edgeToRedraw) {
        if (handleType !== "source" || !sourceAnchor) {
          return false;
        }
        return false;
      }

      const redrawSourceNode = graph.nodes.find((node) => node.id === edgeToRedraw.source_id);
      const redrawSourceHandleId = inferToolEdgeSourceHandle(edgeToRedraw, redrawSourceNode);
      const redrawSourceAnchor = getSourceAnchorPosition(edgeToRedraw.source_id, redrawSourceHandleId);
      if (!redrawSourceAnchor) {
        return false;
      }
      const redrawTargetAnchor = getTargetAnchorPosition(edgeToRedraw.target_id);

      const nextGraph = removeEdgeAndPruneJunctions(graph, edgeToRedraw.id);
      onGraphChange(nextGraph);
      onSelectionChange(null, null);
      setTooltipNodeId(null);
      setDraftConnection({
        sourceNodeId: edgeToRedraw.source_id,
        sourceHandleId: redrawSourceHandleId,
        waypoints: [],
        pointerPosition: redrawTargetAnchor ?? targetAnchor ?? sourceAnchor ?? redrawSourceAnchor,
      });
      setIsConnecting(false);
      setEditorMessage("Line detached. Click blank space to add bends, then click a node to redraw it.");
      return true;
    },
    [
      getSourceAnchorPosition,
      getTargetAnchorPosition,
      graph,
      onGraphChange,
      onSelectionChange,
      pendingPlacement,
      removeEdgeAndPruneJunctions,
      selectedEdgeId,
    ],
  );

  const handleJunctionPointerDown = useCallback(
    (nodeId: string, clientPosition: { x: number; y: number }) => {
      if (!graph || pendingPlacement || draftConnection) {
        return;
      }
      const node = graph.nodes.find((candidate) => candidate.id === nodeId);
      if (!node || !isWireJunctionNode(node)) {
        return;
      }
      const flowPosition = getFlowPositionFromScreen(clientPosition);
      if (!flowPosition) {
        return;
      }
      setTooltipNodeId(null);
      onSelectionChange(nodeId, null);
      setJunctionDrag({
        nodeId,
        pointerOffset: {
          x: flowPosition.x - node.position.x,
          y: flowPosition.y - node.position.y,
        },
      });
    },
    [draftConnection, getFlowPositionFromScreen, graph, onSelectionChange, pendingPlacement],
  );

  const extendDraftConnectionAt = useCallback(
    (flowPosition: GraphPosition) => {
      if (!graph || !draftConnection) {
        return false;
      }

      const sourceNode = graph.nodes.find((node) => node.id === draftConnection.sourceNodeId);
      if (!sourceNode) {
        setDraftConnection(null);
        return false;
      }
      const sourceAnchor = getSourceAnchorPosition(draftConnection.sourceNodeId, draftConnection.sourceHandleId);
      const segmentStart = draftConnection.waypoints[draftConnection.waypoints.length - 1] ?? sourceAnchor;
      const minimumSegmentLength = DRAFT_WIRE_MIN_SEGMENT_PX / Math.max(viewportState.zoom, 0.35);
      if (segmentStart && Math.hypot(flowPosition.x - segmentStart.x, flowPosition.y - segmentStart.y) < minimumSegmentLength) {
        setEditorMessage("Place the next wire point a bit farther from where this segment starts.");
        return true;
      }

      if (sourceNode.category === "provider") {
        setDraftConnection((current) =>
          current
            ? {
                ...current,
                waypoints: [...current.waypoints, flowPosition],
                pointerPosition: flowPosition,
              }
            : current,
        );
        setEditorMessage("Wire routing started. Click blank space to add bends, then click a target node to finish.");
        return true;
      }

      const junctionNode = createWireJunctionNode(graph, {
        x: flowPosition.x - JUNCTION_NODE_SIZE * 0.5,
        y: flowPosition.y - JUNCTION_NODE_SIZE * 0.5,
      });
      const graphWithJunction: GraphDefinition = {
        ...graph,
        nodes: [...graph.nodes, junctionNode],
      };
      const nextEdge = buildCommittedEdge(
        graphWithJunction,
        draftConnection.sourceNodeId,
        junctionNode.id,
        draftConnection.waypoints,
        draftConnection.sourceHandleId,
      );
      if (!nextEdge) {
        return false;
      }

      const junctionAnchor = {
        x: junctionNode.position.x + JUNCTION_NODE_SIZE,
        y: junctionNode.position.y + JUNCTION_NODE_SIZE * 0.5,
      };
      onGraphChange({
        ...graphWithJunction,
        edges: [...graphWithJunction.edges, nextEdge],
      });
      onSelectionChange(null, null);
      setDraftConnection({
        sourceNodeId: junctionNode.id,
        sourceHandleId: null,
        waypoints: [],
        pointerPosition: junctionAnchor,
      });
      setEditorMessage("Wire point added. Continue routing from the floating handle, or click a node to finish.");
      return true;
    },
    [buildCommittedEdge, draftConnection, getSourceAnchorPosition, graph, onGraphChange, onSelectionChange, viewportState.zoom],
  );

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
  useEffect(() => cancelPendingDraftPointerFrame, [cancelPendingDraftPointerFrame]);
  const updateNodePositionInGraph = useCallback((baseGraph: GraphDefinition, nodeId: string, position: GraphPosition) => {
    let didChange = false;
    const nextNodes = baseGraph.nodes.map((node) => {
      if (node.id !== nodeId) {
        return node;
      }
      if (Math.abs(node.position.x - position.x) < 0.01 && Math.abs(node.position.y - position.y) < 0.01) {
        return node;
      }
      didChange = true;
      return { ...node, position };
    });
    return didChange ? { ...baseGraph, nodes: nextNodes } : baseGraph;
  }, []);
  useEffect(() => {
    if (!junctionDrag) {
      return;
    }

    const updateDraggedJunction = (clientPosition: { x: number; y: number }, commit: boolean) => {
      const currentGraph = graphRef.current;
      if (!currentGraph) {
        return;
      }
      const flowPosition = getFlowPositionFromScreen(clientPosition);
      if (!flowPosition) {
        return;
      }
      const nextGraph = updateNodePositionInGraph(currentGraph, junctionDrag.nodeId, {
        x: flowPosition.x - junctionDrag.pointerOffset.x,
        y: flowPosition.y - junctionDrag.pointerOffset.y,
      });
      if (commit) {
        cancelPendingDragFrame();
        onGraphChange(nextGraph);
        return;
      }
      scheduleDragGraphUpdate(nextGraph);
    };

    const handleMouseMove = (event: MouseEvent) => {
      event.preventDefault();
      updateDraggedJunction({ x: event.clientX, y: event.clientY }, false);
    };

    const handleMouseUp = (event: MouseEvent) => {
      updateDraggedJunction({ x: event.clientX, y: event.clientY }, true);
      setJunctionDrag(null);
    };

    const handleWindowBlur = () => {
      cancelPendingDragFrame();
      setJunctionDrag(null);
    };

    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleMouseUp);
    window.addEventListener("blur", handleWindowBlur);
    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
      window.removeEventListener("blur", handleWindowBlur);
    };
  }, [cancelPendingDragFrame, getFlowPositionFromScreen, junctionDrag, onGraphChange, scheduleDragGraphUpdate, updateNodePositionInGraph]);

  useEffect(() => {
    setCanvasZoom(flowInstance?.getZoom() ?? 1);
  }, [flowInstance, setCanvasZoom]);
  useEffect(() => {
    const viewport = flowInstance?.getViewport() ?? null;
    pointerPanStateRef.current = {
      lastPointerViewport: viewport,
      lastAppliedViewport: viewport,
      isSyncingViewport: false,
    };
  }, [flowInstance, graph?.graph_id, graph?.nodes]);

  const wasdKeysRef = useRef(new Set<string>());
  const wasdFrameRef = useRef<number | null>(null);

  useEffect(() => {
    const PAN_SPEED = 8;

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
      if (event.key === "Meta" || event.metaKey) {
        setIsCommandHeld(true);
      }
      if (isEditableTarget(event.target)) return;
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
      if (event.key === "Meta" || !event.metaKey) {
        setIsCommandHeld(false);
      }
      const key = event.key.toLowerCase();
      wasdKeysRef.current.delete(key);
      if (wasdKeysRef.current.size === 0 && wasdFrameRef.current !== null) {
        cancelAnimationFrame(wasdFrameRef.current);
        wasdFrameRef.current = null;
      }
    };

    const onBlur = () => {
      setIsCommandHeld(false);
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
  }, [flowInstance, isEditableTarget]);

  const addProviderNode = useCallback(
    (provider: NodeProviderDefinition, options: AddCanvasNodeOptions = {}) => {
      if (!graph || !catalog) {
        return;
      }
      const { screenPosition, openDrawer = true, nextDrawerTab = "inspect" } = options;
      const position = getFlowPositionFromScreen(screenPosition);
      if (!position) {
        return;
      }
      const nextNode = createNodeFromProvider(graph, provider, catalog, position);
      onGraphChange({
        ...graph,
        start_node_id: !graph.start_node_id && nextNode.category === "start" ? nextNode.id : graph.start_node_id,
        nodes: [...graph.nodes, nextNode],
      });
      if (openDrawer) {
        setDrawerOpen(true);
        setDrawerTab(nextDrawerTab);
      }
      onSelectionChange(nextNode.id, null);
      setTooltipNodeId(null);
      setToolDetailsNodeId(null);
      setProviderDetailsNodeId(null);
      setEditorMessage(`Added ${provider.display_name}.`);
    },
    [catalog, getFlowPositionFromScreen, graph, onGraphChange, onSelectionChange],
  );

  const addSavedNode = useCallback(
    (saved: SavedNode, options: AddCanvasNodeOptions = {}) => {
      if (!graph) {
        return;
      }
      const { screenPosition, openDrawer = true, nextDrawerTab = "inspect" } = options;
      const position = getFlowPositionFromScreen(screenPosition);
      if (!position) {
        return;
      }
      const nextNode = createNodeFromSaved(graph, saved, position);
      onGraphChange({
        ...graph,
        start_node_id: !graph.start_node_id && nextNode.category === "start" ? nextNode.id : graph.start_node_id,
        nodes: [...graph.nodes, nextNode],
      });
      if (openDrawer) {
        setDrawerOpen(true);
        setDrawerTab(nextDrawerTab);
      }
      onSelectionChange(nextNode.id, null);
      setTooltipNodeId(null);
      setToolDetailsNodeId(null);
      setProviderDetailsNodeId(null);
      setEditorMessage(`Added saved node "${saved.name}".`);
    },
    [getFlowPositionFromScreen, graph, onGraphChange, onSelectionChange],
  );

  const beginProviderPlacement = useCallback(
    (provider: NodeProviderDefinition) => {
      if (!graph || !catalog) {
        return;
      }
      const position = getFlowPositionFromScreen();
      if (!position) {
        return;
      }
      onSelectionChange(null, null);
      setTooltipNodeId(null);
      setToolDetailsNodeId(null);
      setProviderDetailsNodeId(null);
      setDrawerOpen(false);
      setPendingPlacement({
        kind: "provider",
        provider,
        position,
      });
      setEditorMessage(`Placing ${provider.display_name}. Click anywhere in the canvas to confirm, or press Escape to cancel.`);
    },
    [catalog, getFlowPositionFromScreen, graph, onSelectionChange],
  );

  const beginSavedPlacement = useCallback(
    (saved: SavedNode) => {
      if (!graph) {
        return;
      }
      const position = getFlowPositionFromScreen();
      if (!position) {
        return;
      }
      onSelectionChange(null, null);
      setTooltipNodeId(null);
      setToolDetailsNodeId(null);
      setProviderDetailsNodeId(null);
      setDrawerOpen(false);
      setPendingPlacement({
        kind: "saved",
        saved,
        position,
      });
      setEditorMessage(`Placing saved node "${saved.name}". Click anywhere in the canvas to confirm, or press Escape to cancel.`);
    },
    [getFlowPositionFromScreen, graph, onSelectionChange],
  );

  const updatePendingPlacementFromScreen = useCallback(
    (screenPosition: { x: number; y: number }) => {
      const position = getFlowPositionFromScreen(screenPosition);
      if (!position) {
        return;
      }
      setPendingPlacement((current) => (current ? { ...current, position } : current));
    },
    [getFlowPositionFromScreen],
  );

  const confirmPendingPlacement = useCallback(
    (screenPosition?: { x: number; y: number }) => {
      if (!graph || !pendingPlacement) {
        return;
      }
      const placement = screenPosition
        ? (() => {
            const position = getFlowPositionFromScreen(screenPosition);
            return position ? { ...pendingPlacement, position } : pendingPlacement;
          })()
        : pendingPlacement;
      const nextNode = buildPlacementNode(placement, graph);
      if (!nextNode) {
        return;
      }
      setPendingPlacement(null);
      onGraphChange({
        ...graph,
        start_node_id: !graph.start_node_id && nextNode.category === "start" ? nextNode.id : graph.start_node_id,
        nodes: [...graph.nodes, nextNode],
      });
      onSelectionChange(nextNode.id, null);
      setEditorMessage(`Added ${nextNode.label}.`);
    },
    [buildPlacementNode, getFlowPositionFromScreen, graph, onGraphChange, onSelectionChange, pendingPlacement],
  );

  const handleSaveNode = useCallback(
    (node: GraphNode) => {
      const saved = saveNodeToLibrary(node);
      setSavedNodes(getSavedNodes());
      setEditorMessage(`Saved "${saved.name}" to library.`);
    },
    [],
  );

  const handleDeleteSavedNode = useCallback(
    (id: string) => {
      deleteSavedNode(id);
      setSavedNodes(getSavedNodes());
      setEditorMessage("Removed saved node from library.");
    },
    [],
  );

  const quickAddItems = useMemo(() => {
    const providers = (catalog?.node_providers ?? []).filter((provider) => provider.category !== "provider");
    return QUICK_ADD_SLOTS.map((slot) => ({
      ...slot,
      provider: resolveQuickAddProvider(providers, slot),
    }));
  }, [catalog]);

  const pendingPlacementPreviewNode = useMemo(() => {
    if (!graph || !pendingPlacement) {
      return null;
    }
    return buildPlacementNode(pendingPlacement, graph);
  }, [buildPlacementNode, graph, pendingPlacement]);

  const pendingPlacementPreviewStyle = useMemo<CSSProperties | null>(() => {
    if (!pendingPlacement) {
      return null;
    }
    return {
      transform: `translate(${viewportState.x + pendingPlacement.position.x * viewportState.zoom}px, ${viewportState.y + pendingPlacement.position.y * viewportState.zoom}px) scale(${viewportState.zoom})`,
    };
  }, [pendingPlacement, viewportState]);

  const draftConnectionToneClass = useMemo(() => {
    if (!draftConnection || !graph) {
      return "";
    }
    const sourceNode = graph.nodes.find((node) => node.id === draftConnection.sourceNodeId);
    if (sourceNode?.kind !== "tool") {
      return "";
    }
    return draftConnection.sourceHandleId === TOOL_FAILURE_HANDLE_ID
      ? " graph-draft-wire-path--tool-failure"
      : " graph-draft-wire-path--tool-success";
  }, [draftConnection, graph]);

  useEffect(() => {
    renderDraftConnectionPreview(draftPreviewPointerPositionRef.current);
  }, [draftConnection, isConnecting, renderDraftConnectionPreview, viewportState]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (isEditableTarget(event.target)) {
        return;
      }
      if (event.metaKey || event.ctrlKey || event.altKey) {
        return;
      }

      if (event.key === "?") {
        event.preventDefault();
        setShowHotkeys((current) => !current);
        return;
      }

      if (event.key === "Escape") {
        event.preventDefault();
        if (pendingPlacement) {
          cancelPlacement("Placement cancelled.");
        } else if (draftConnection) {
          cancelDraftConnection("Wire routing cancelled.");
        } else {
          clearCanvasChrome();
        }
        return;
      }

      const quickAddItem = quickAddItems.find((item) => item.hotkey === event.key);
      if (quickAddItem?.provider) {
        event.preventDefault();
        beginProviderPlacement(quickAddItem.provider);
        return;
      }

      if (pendingPlacement || draftConnection) {
        return;
      }

      if (event.code === "Space") {
        event.preventDefault();
        handleFitView();
        return;
      }

      const key = event.key.toLowerCase();
      if (key === "g") {
        event.preventDefault();
        openDrawerTab("add");
        return;
      }
      if (key === "i") {
        event.preventDefault();
        openDrawerTab("inspect");
        return;
      }
      if (key === "r") {
        event.preventDefault();
        openDrawerTab("run");
        return;
      }
      if (key === "f") {
        event.preventDefault();
        handleFitView();
        return;
      }
      if (key === "l") {
        event.preventDefault();
        setPanLocked((current) => !current);
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [beginProviderPlacement, cancelDraftConnection, cancelPlacement, clearCanvasChrome, draftConnection, handleFitView, isEditableTarget, openDrawerTab, pendingPlacement, quickAddItems]);

  const nodes = useMemo<FlowNode[]>(() => {
    if (!graph) {
      return [];
    }
    const isRoutingDraftWire = draftConnection !== null && !isConnecting;
    const snapTargetNodeId = isRoutingDraftWire ? draftConnectionSnapTargetNodeId : null;
    const nextNodeDataCache = new Map<string, GraphCanvasRuntimeNodeData>();
    const nextNodes = graph.nodes.map((node): FlowNode<GraphCanvasRuntimeNodeData> => {
      const isActive = runState?.current_node_id === node.id;
      const hasError = Boolean(runState?.node_errors?.[node.id]);
      const wasVisited = (runState?.visit_counts?.[node.id] ?? 0) > 0;
      const didLastRunFinish = runState?.status === "completed" || runState?.status === "failed";
      const isConnectionMagnetized = isRoutingDraftWire && snapTargetNodeId === node.id;
      const kindColor = KIND_COLORS[node.kind] ?? "#8486a5";
      const status: GraphCanvasNodeData["status"] = hasError
        ? "failed"
        : isActive
          ? "active"
          : wasVisited
            ? "success"
            : didLastRunFinish
              ? "unreached"
              : "idle";
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
        previousData.isConnectionMagnetized === isConnectionMagnetized &&
        (previousData.preview === false || previousData.preview === undefined) &&
        previousData.tooltipVisible === tooltipVisible &&
        previousData.onToggleTooltip === handleToggleTooltip &&
        previousData.onOpenToolDetails === handleOpenToolDetails &&
        previousData.onOpenProviderDetails === handleOpenProviderDetails &&
        previousData.onHandlePointerDown === handleNodeHandlePointerDown &&
        previousData.onJunctionPointerDown === handleJunctionPointerDown
          ? previousData
          : {
              node,
              graph: tooltipGraph,
              catalog,
              runState,
              kindColor,
              status,
              isConnectionMagnetized,
              preview: false,
              tooltipVisible,
              onToggleTooltip: handleToggleTooltip,
              onOpenToolDetails: handleOpenToolDetails,
              onOpenProviderDetails: handleOpenProviderDetails,
              onHandlePointerDown: handleNodeHandlePointerDown,
              onJunctionPointerDown: handleJunctionPointerDown,
            };
      nextNodeDataCache.set(node.id, nextData);
      const isJunction = isWireJunctionNode(node);

      return {
        id: node.id,
        type: "graphNode",
        position: node.position,
        selected: node.id === selectedNodeId,
        sourcePosition: "right" as Position,
        targetPosition: "left" as Position,
        data: nextData,
        style: isJunction ? JUNCTION_NODE_STYLE : NODE_STYLE,
      };
    });
    nodeDataCacheRef.current = nextNodeDataCache;
    return nextNodes;
  }, [catalog, draftConnection?.sourceNodeId, draftConnectionSnapTargetNodeId, graph, handleJunctionPointerDown, handleNodeHandlePointerDown, handleOpenProviderDetails, handleOpenToolDetails, handleToggleTooltip, isConnecting, runState, selectedNodeId, tooltipNodeId]);

  const edges = useMemo<FlowEdge<GraphCanvasEdgeData>[]>(() => {
    if (!graph) {
      return [];
    }
    const nodeLookup = new Map(graph.nodes.map((node) => [node.id, node]));
    const siblingEdgesByTarget = new Map<string, GraphEdge[]>();
    const routeSignatureGroups = new Map<string, GraphEdge[]>();
    graph.edges.forEach((edge) => {
      const siblingEdges = siblingEdgesByTarget.get(edge.target_id);
      if (siblingEdges) {
        siblingEdges.push(edge);
      } else {
        siblingEdgesByTarget.set(edge.target_id, [edge]);
      }
      const sourceNode = nodeLookup.get(edge.source_id);
      const signature = getEdgeRouteSignature(edge, sourceNode);
      if (!signature) {
        return;
      }
      const existing = routeSignatureGroups.get(signature);
      if (existing) {
        existing.push(edge);
      } else {
        routeSignatureGroups.set(signature, [edge]);
      }
    });
    siblingEdgesByTarget.forEach((siblingEdges) => {
      siblingEdges.sort((left, right) => {
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
    });
    const shouldResolveLabelCollisions = !isNodeDragActive && !junctionDrag;

    const edgeLayouts = graph.edges.map((edge) => {
      const sourceNode = nodeLookup.get(edge.source_id);
      const targetNode = nodeLookup.get(edge.target_id);
      const touchesWireJunction = isWireJunctionNode(sourceNode) || isWireJunctionNode(targetNode);
      const labelText = touchesWireJunction ? "" : String(edge.condition?.label ?? edge.label ?? "");
      const sourceHandleId = inferToolEdgeSourceHandle(edge, sourceNode);
      const sourceAnchor = getSourceAnchorPosition(edge.source_id, sourceHandleId);
      const targetAnchor = getTargetAnchorPosition(edge.target_id);
      const routeSignature = getEdgeRouteSignature(edge, sourceNode);
      const overlappingEdges = routeSignature ? routeSignatureGroups.get(routeSignature) ?? [edge] : [edge];
      const siblingEdges = siblingEdgesByTarget.get(edge.target_id) ?? [edge];
      const toolEdgeTone =
        sourceNode?.kind === "tool"
          ? sourceHandleId === TOOL_FAILURE_HANDLE_ID
            ? TOOL_EDGE_TONES.failure
            : TOOL_EDGE_TONES.success
          : null;
      const siblingIndex = siblingEdges.findIndex((candidate) => candidate.id === edge.id);
      const overlappingIndex = overlappingEdges.findIndex((candidate) => candidate.id === edge.id);
      const siblingLaneOffset = (siblingIndex - (siblingEdges.length - 1) / 2) * EDGE_SIBLING_SPACING;
      const overlapLaneOffset = (overlappingIndex - (overlappingEdges.length - 1) / 2) * EDGE_LANE_SPACING;
      const laneOffset = overlappingEdges.length > 1 ? overlapLaneOffset : siblingEdges.length > 1 ? siblingLaneOffset : 0;
      const routeShiftX =
        laneOffset !== 0 && sourceAnchor && targetAnchor && Math.abs(targetAnchor.x - sourceAnchor.x) < Math.abs(targetAnchor.y - sourceAnchor.y)
          ? laneOffset
          : 0;
      const routeShiftY =
        laneOffset !== 0 && sourceAnchor && targetAnchor && Math.abs(targetAnchor.x - sourceAnchor.x) >= Math.abs(targetAnchor.y - sourceAnchor.y)
          ? laneOffset
          : 0;
      const labelOffset = (siblingIndex - (siblingEdges.length - 1) / 2) * 28;

      const edgeGeometry =
        sourceAnchor && targetAnchor
          ? getEdgeLabelPlacement({
              sourceX: sourceAnchor.x,
              sourceY: sourceAnchor.y,
              targetX: targetAnchor.x,
              targetY: targetAnchor.y,
              sourcePosition: "right" as Position,
              targetPosition: "left" as Position,
              waypoints: edge.waypoints ?? [],
              labelOffset,
            })
          : null;

      return {
        id: edge.id,
        labelText,
        labelPlacement:
          edgeGeometry && labelText
            ? {
                center: {
                  x: edgeGeometry.point.x + routeShiftX,
                  y: edgeGeometry.point.y + routeShiftY,
                },
                tangent: edgeGeometry.tangent,
                width: estimateEdgeLabelWidth(labelText),
                height: EDGE_LABEL_HEIGHT,
              }
            : null,
        pathSamples:
          shouldResolveLabelCollisions && edgeGeometry
            ? sampleEdgePathPoints(edge.id, edgeGeometry.edgePath, routeShiftX, routeShiftY)
            : [],
        type: "graphEdge",
        source: edge.source_id,
        target: edge.target_id,
        sourceHandle: edge.source_handle_id ?? sourceHandleId ?? undefined,
        selected: edge.id === selectedEdgeId,
        markerEnd: {
          type: MarkerType.ArrowClosed,
          width: 14,
          height: 14,
          color: toolEdgeTone?.markerColor ?? KIND_COLORS[targetNode?.kind ?? ""] ?? "#6ea8ff",
        },
        label: labelText,
        animated: false,
        data: {
          kind: edge.kind,
          waypoints: edge.waypoints ?? [],
          sourceColor: toolEdgeTone?.sourceColor ?? KIND_COLORS[sourceNode?.kind ?? ""] ?? "#6ea8ff",
          targetColor: toolEdgeTone?.targetColor ?? KIND_COLORS[targetNode?.kind ?? ""] ?? "#6ea8ff",
          routeTone: toolEdgeTone?.routeTone,
          routeShiftX,
          routeShiftY,
          labelOffset,
        },
        style: {
          strokeWidth: edge.id === selectedEdgeId ? SELECTED_EDGE_STROKE_WIDTH : EDGE_STROKE_WIDTH,
          strokeLinecap: "round" as const,
          filter:
            edge.id === selectedEdgeId
              ? "drop-shadow(0 0 7px rgba(255, 99, 196, 0.25)) drop-shadow(0 0 10px rgba(92, 146, 255, 0.22))"
              : "drop-shadow(0 0 7px rgba(111, 133, 255, 0.18))",
        },
      };
    });

    const labelShifts = shouldResolveLabelCollisions
      ? resolveEdgeLabelShifts(
          edgeLayouts.flatMap((edge) =>
            edge.labelPlacement
              ? [
                  {
                    edgeId: edge.id,
                    center: edge.labelPlacement.center,
                    tangent: edge.labelPlacement.tangent,
                    width: edge.labelPlacement.width,
                    height: edge.labelPlacement.height,
                  },
                ]
              : [],
          ),
          edgeLayouts.flatMap((edge) => edge.pathSamples),
        )
      : new Map<string, { x: number; y: number }>();

    return edgeLayouts.map(({ labelPlacement, labelText: _labelText, pathSamples: _pathSamples, data, ...edge }) => {
      const labelShift = labelShifts.get(edge.id);
      return {
        ...edge,
        data: {
          ...data,
          labelShiftX: labelShift?.x ?? 0,
          labelShiftY: labelShift?.y ?? 0,
        },
      };
    });
  }, [getEdgeRouteSignature, getSourceAnchorPosition, getTargetAnchorPosition, graph, isNodeDragActive, junctionDrag, selectedEdgeId]);

  const onNodesChange = useCallback(
    (changes: NodeChange[]) => {
      if (!graph) {
        return;
      }
      let nextGraph = graph;
      let hasStructuralChange = false;
      let isDragging = false;
      let dragEnded = false;
      let nextIsNodeDragActive = isNodeDragActive;
      let didSeeDragState = false;

      changes.forEach((change) => {
        if (change.type === "position" && change.position) {
          const dragging = (change as { dragging?: boolean }).dragging;
          if (dragging === true) {
            isDragging = true;
            nextIsNodeDragActive = true;
            didSeeDragState = true;
          }
          if (dragging === false) {
            dragEnded = true;
            nextIsNodeDragActive = false;
            didSeeDragState = true;
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

      if (didSeeDragState && nextIsNodeDragActive !== isNodeDragActive) {
        setIsNodeDragActive(nextIsNodeDragActive);
      }

      if (hasStructuralChange || dragEnded || !isDragging) {
        cancelPendingDragFrame();
        onGraphChange(nextGraph);
      } else {
        scheduleDragGraphUpdate(nextGraph);
      }
    },
    [cancelPendingDragFrame, graph, isNodeDragActive, onGraphChange, onSelectionChange, scheduleDragGraphUpdate, selectedNodeId],
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
      if (!connection.source || !connection.target) {
        return;
      }
      didCreateConnectionRef.current = true;
      commitEdge(
        connection.source,
        connection.target,
        draftConnection?.waypoints ?? [],
        graph,
        connection.sourceHandle ?? draftConnection?.sourceHandleId ?? null,
      );
      setDraftConnection(null);
    },
    [commitEdge, draftConnection, graph],
  );

  const onConnectStart = useCallback(
    (_event: ReactMouseEvent | ReactTouchEvent, params: OnConnectStartParams) => {
      if (!params.nodeId) {
        return;
      }
      const sourceHandleId = ((params as OnConnectStartParams & { handleId?: string | null }).handleId ?? null) as string | null;
      const sourceAnchor = getSourceAnchorPosition(params.nodeId, sourceHandleId);
      if (!sourceAnchor) {
        return;
      }
      didCreateConnectionRef.current = false;
      setIsConnecting(true);
      setDraftConnection({
        sourceNodeId: params.nodeId,
        sourceHandleId,
        waypoints: [],
        pointerPosition: sourceAnchor,
      });
    },
    [getSourceAnchorPosition],
  );

  const onConnectEnd = useCallback(
    (event: MouseEvent | TouchEvent) => {
      setIsConnecting(false);
      const clientPosition = getEventClientPosition(event);
      if (didCreateConnectionRef.current) {
        didCreateConnectionRef.current = false;
        return;
      }
      if (!draftConnection || !clientPosition) {
        setDraftConnection(null);
        return;
      }
      const flowPosition = getFlowPositionFromScreen(clientPosition);
      if (!flowPosition) {
        setDraftConnection(null);
        return;
      }
      const snapTargetNodeId = resolveDraftConnectionSnapTargetNodeId(draftConnection.sourceNodeId, flowPosition);
      if (snapTargetNodeId) {
        const didCommit = commitEdge(
          draftConnection.sourceNodeId,
          snapTargetNodeId,
          draftConnection.waypoints,
          graph,
          draftConnection.sourceHandleId,
        );
        if (didCommit) {
          didCreateConnectionRef.current = true;
          setDraftConnection(null);
          return;
        }
      }
      suppressNextPaneClickRef.current = true;
      void extendDraftConnectionAt(flowPosition);
    },
    [commitEdge, draftConnection, extendDraftConnectionAt, getEventClientPosition, getFlowPositionFromScreen, graph, resolveDraftConnectionSnapTargetNodeId],
  );

  const onDrop = useCallback(
    (event: React.DragEvent<HTMLDivElement>) => {
      event.preventDefault();
      setIsProviderDragActive(false);
      setIsSavedNodeDragActive(false);
      const rawSaved = event.dataTransfer.getData("application/graph-saved-node");
      if (rawSaved) {
        const saved = JSON.parse(rawSaved) as SavedNode;
        addSavedNode(saved, {
          screenPosition: { x: event.clientX, y: event.clientY },
          openDrawer: true,
          nextDrawerTab: drawerTab,
        });
        return;
      }
      const rawProvider = event.dataTransfer.getData("application/graph-node-provider");
      if (!rawProvider) {
        return;
      }
      const provider = JSON.parse(rawProvider) as NodeProviderDefinition;
      addProviderNode(provider, {
        screenPosition: { x: event.clientX, y: event.clientY },
        openDrawer: true,
        nextDrawerTab: drawerTab,
      });
    },
    [addProviderNode, addSavedNode, drawerTab],
  );

  const onDragOver = useCallback((event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    const types = Array.from(event.dataTransfer.types);
    if (types.includes("application/graph-node-provider")) {
      setIsProviderDragActive(true);
    }
    if (types.includes("application/graph-saved-node")) {
      setIsSavedNodeDragActive(true);
    }
    event.dataTransfer.dropEffect = "copy";
  }, []);

  const onDragLeave = useCallback((event: React.DragEvent<HTMLDivElement>) => {
    if (event.currentTarget === event.target) {
      setIsProviderDragActive(false);
      setIsSavedNodeDragActive(false);
    }
  }, []);

  const shouldIgnorePlacementTarget = useCallback((target: EventTarget | null) => {
    return target instanceof HTMLElement && Boolean(target.closest(PLACEMENT_UI_SELECTOR));
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
          className={`graph-canvas${drawerOpen ? " graph-canvas--drawer-open" : ""}${isProviderDragActive || isSavedNodeDragActive ? " is-drop-target" : ""}${isConnecting ? " is-connecting" : ""}${pendingPlacement ? " is-placing-node" : ""}${draftConnection ? " is-routing-wire" : ""}${junctionDrag ? " is-dragging-junction" : ""}`}
          onDrop={onDrop}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          onMouseMove={(event) => {
            if (draftConnection && !isConnecting && !shouldIgnorePlacementTarget(event.target)) {
              const flowPosition = getFlowPositionFromScreen({ x: event.clientX, y: event.clientY });
              if (flowPosition) {
                scheduleDraftPointerPositionUpdate(flowPosition);
              }
            }
            if (!pendingPlacement || shouldIgnorePlacementTarget(event.target)) {
              return;
            }
            updatePendingPlacementFromScreen({ x: event.clientX, y: event.clientY });
          }}
          onClickCapture={(event) => {
            if (!pendingPlacement || shouldIgnorePlacementTarget(event.target)) {
              return;
            }
            event.preventDefault();
            event.stopPropagation();
            confirmPendingPlacement({ x: event.clientX, y: event.clientY });
          }}
        >
          <div className="graph-toolbar" aria-label="Graph visualizer actions">
            <DrawerActionButton tab="add" activeTab={drawerTab} drawerOpen={drawerOpen} label="Open add node drawer (G)" onClick={() => openDrawerTab("add")}>
              <svg viewBox="0 0 24 24" aria-hidden="true">
                <path d="M12 5v14M5 12h14" />
              </svg>
              <span>Add</span>
            </DrawerActionButton>
            <DrawerActionButton
              tab="inspect"
              activeTab={drawerTab}
              drawerOpen={drawerOpen}
              label="Open details drawer (I)"
              onClick={() => openDrawerTab("inspect")}
            >
              <svg viewBox="0 0 24 24" aria-hidden="true">
                <path d="M4 7h16M7 12h10M10 17h4" />
              </svg>
              <span>Details</span>
            </DrawerActionButton>
            <DrawerActionButton tab="run" activeTab={drawerTab} drawerOpen={drawerOpen} label="Open run drawer (R)" onClick={() => openDrawerTab("run")}>
              <svg viewBox="0 0 24 24" aria-hidden="true">
                <path d="M8 6.5v11l9-5.5-9-5.5Z" />
              </svg>
              <span>Run</span>
            </DrawerActionButton>
            <button
              type="button"
              className={`graph-toolbar-button${vizLocked ? " is-active" : ""}`}
              onClick={() => setPanLocked((v) => !v)}
              aria-label={panLocked ? "Unlock scroll panning (L)" : vizLocked ? "Visualizer locked while Command is held" : "Lock to WASD only (L)"}
              title={panLocked ? "Unlock scroll panning (L)" : vizLocked ? "Visualizer locked while Command is held" : "Lock to WASD only (L)"}
            >
              <svg viewBox="0 0 24 24" aria-hidden="true">
                {vizLocked ? (
                  <path d="M17 11V8A5 5 0 0 0 7 8v3M5 11h14a1 1 0 0 1 1 1v8a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1v-8a1 1 0 0 1 1-1Zm7 4v3" />
                ) : (
                  <path d="M7 11V8a5 5 0 0 1 9.9-1M5 11h14a1 1 0 0 1 1 1v8a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1v-8a1 1 0 0 1 1-1Zm7 4v3" />
                )}
              </svg>
              <span>{vizLocked ? "Locked" : "Lock"}</span>
            </button>
          </div>
          {graph.nodes.length === 0 ? (
            <div className="graph-empty-state">
              <strong>Start building your graph</strong>
              <p>Use the quick-add bar, press `1-5`, or drag nodes directly into the canvas.</p>
              <button type="button" onClick={() => openDrawerTab("add")}>
                Add Your First Node
              </button>
            </div>
          ) : null}
          {isProviderDragActive || isSavedNodeDragActive ? (
            <div className="graph-drop-hint">
              <strong>Drop to add node</strong>
              <span>Release anywhere in the visualizer to place the node.</span>
            </div>
          ) : null}
          {pendingPlacement ? (
            <div className="graph-placement-hint">
              <strong>Click to place node</strong>
              <span>Move the cursor to position it, then click to confirm or press Escape to cancel.</span>
            </div>
          ) : null}
          {draftConnection && !isConnecting ? (
            <div className="graph-placement-hint graph-placement-hint--wire">
              <strong>Route the wire</strong>
              <span>Click blank space to drop a reusable wire point, click a node to connect, or press Escape to cancel.</span>
            </div>
          ) : null}
          {pendingPlacementPreviewNode && pendingPlacementPreviewStyle ? (
            <div className="graph-placement-preview" style={pendingPlacementPreviewStyle} aria-hidden="true">
              <div
                className="graph-node-card is-preview"
                style={
                  {
                    "--node-kind-color": KIND_COLORS[pendingPlacementPreviewNode.kind] ?? "#8486a5",
                  } as CSSProperties
                }
              >
                <div className="graph-node-card-inner">
                  <div className="graph-node-header">
                    <div className="graph-node-icon">{KIND_LABELS[pendingPlacementPreviewNode.kind] ?? pendingPlacementPreviewNode.kind.slice(0, 2).toUpperCase()}</div>
                    <div className="graph-node-body">
                      <strong className="graph-node-title">{pendingPlacementPreviewNode.label}</strong>
                      <div className="graph-node-subtitle">
                        {pendingPlacementPreviewNode.kind === "model"
                          ? String(
                              pendingPlacementPreviewNode.config.provider_name ??
                                pendingPlacementPreviewNode.model_provider_name ??
                                pendingPlacementPreviewNode.provider_label ??
                                pendingPlacementPreviewNode.provider_id,
                            )
                          : pendingPlacementPreviewNode.provider_label ?? pendingPlacementPreviewNode.provider_id}
                      </div>
                    </div>
                    <div className="graph-node-badge" aria-hidden="true">
                      <span className="graph-node-badge-dot" />
                    </div>
                  </div>
                  <div className="graph-node-meta">
                    <span className="graph-node-chip">{pendingPlacementPreviewNode.category}</span>
                    <span className="graph-node-meta-text">{pendingPlacementPreviewNode.kind}</span>
                  </div>
                </div>
              </div>
            </div>
          ) : null}
          {draftConnection && !isConnecting ? (
            <svg className="graph-draft-wire" aria-hidden="true">
              <path ref={draftWirePathRef} d="" className={`graph-draft-wire-path${draftConnectionToneClass}`} />
            </svg>
          ) : null}
          <ReactFlow
            nodes={nodes}
            edges={edges}
            nodeTypes={nodeTypes}
            edgeTypes={edgeTypes}
            connectionLineComponent={GraphCanvasConnectionLine}
            connectionLineType={ConnectionLineType.SmoothStep}
            connectionLineStyle={{
              stroke: "#9ab7ff",
              strokeWidth: DRAFT_WIRE_STROKE_WIDTH,
              strokeDasharray: "6 6",
              filter: "drop-shadow(0 0 10px rgba(111, 130, 255, 0.45))",
            }}
            defaultViewport={{ x: 0, y: 0, zoom: 1 }}
            panOnDrag={!vizLocked && !pendingPlacement && !junctionDrag}
            panOnScroll={false}
            zoomOnScroll={!vizLocked}
            zoomOnPinch={!vizLocked}
            zoomOnDoubleClick={!vizLocked}
            preventScrolling={!vizLocked}
            onInit={handleFlowInit}
            onMove={handleViewportMove}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onConnectStart={onConnectStart}
            onConnectEnd={onConnectEnd}
            isValidConnection={isValidConnection}
            onNodeClick={(_, node) => {
              if (draftConnection && !isConnecting) {
                const didCommit = commitEdge(
                  draftConnection.sourceNodeId,
                  node.id,
                  draftConnection.waypoints,
                  graph,
                  draftConnection.sourceHandleId,
                );
                if (didCommit) {
                  cancelPendingDraftPointerFrame();
                  setDraftConnection(null);
                }
                return;
              }
              if (pendingPlacement) {
                return;
              }
              onSelectionChange(node.id, null);
              setTooltipNodeId(null);
            }}
            onEdgeClick={(_, edge) => {
              if (draftConnection) {
                return;
              }
              if (pendingPlacement) {
                return;
              }
              onSelectionChange(null, edge.id);
              setTooltipNodeId(null);
            }}
            onPaneClick={(event) => {
              if (suppressNextPaneClickRef.current) {
                suppressNextPaneClickRef.current = false;
                return;
              }
              if (draftConnection && !isConnecting) {
                const flowPosition = getFlowPositionFromScreen({ x: event.clientX, y: event.clientY });
                if (flowPosition) {
                  const snapTargetNodeId = resolveDraftConnectionSnapTargetNodeId(draftConnection.sourceNodeId, flowPosition);
                  if (
                    snapTargetNodeId &&
                    commitEdge(
                      draftConnection.sourceNodeId,
                      snapTargetNodeId,
                      draftConnection.waypoints,
                      graph,
                      draftConnection.sourceHandleId,
                    )
                  ) {
                    cancelPendingDraftPointerFrame();
                    setDraftConnection(null);
                    return;
                  }
                  void extendDraftConnectionAt(flowPosition);
                }
                return;
              }
              if (pendingPlacement) {
                confirmPendingPlacement({ x: event.clientX, y: event.clientY });
                return;
              }
              clearCanvasChrome();
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
          <div className={`graph-quick-add${quickAddMinimized ? " is-minimized" : ""}`} role="toolbar" aria-label="Quick add nodes">
            <div className="graph-quick-add-main">
              <div className="graph-quick-add-copy">
                <strong>Quick Add</strong>
                <span>{quickAddMinimized ? "1-5" : "Press 1-5 or click a node to insert it near center."}</span>
              </div>
              <div id="graph-quick-add-list" className="graph-quick-add-list">
                {quickAddItems.map((item) => (
                  <button
                    key={item.hotkey}
                    type="button"
                    className="graph-quick-add-button"
                    onClick={() => item.provider && beginProviderPlacement(item.provider)}
                    disabled={!item.provider}
                    title={item.provider ? `${item.label} (${item.hotkey})` : `No ${item.label.toLowerCase()} node available`}
                  >
                    <span className="graph-quick-add-hotkey">{item.hotkey}</span>
                    <span className="graph-quick-add-label">{item.label}</span>
                  </button>
                ))}
              </div>
            </div>
            <div className="graph-quick-add-actions">
              <button
                type="button"
                className={`graph-hotkey-toggle${showHotkeys ? " is-active" : ""}`}
                onClick={() => setShowHotkeys((current) => !current)}
                aria-pressed={showHotkeys}
                title="Toggle keyboard shortcuts (?)"
              >
                <span className="graph-hotkey-chip">?</span>
                <span>Hotkeys</span>
              </button>
              <button
                type="button"
                className={`graph-quick-add-toggle${quickAddMinimized ? " is-active" : ""}`}
                onClick={() => setQuickAddMinimized((current) => !current)}
                aria-expanded={!quickAddMinimized}
                aria-controls="graph-quick-add-list"
                aria-label={quickAddMinimized ? "Expand quick add bar" : "Collapse quick add bar"}
                title={quickAddMinimized ? "Expand quick add bar" : "Collapse quick add bar"}
              >
                <span className="graph-quick-add-toggle-icon" aria-hidden="true">
                  {quickAddMinimized ? "+" : "-"}
                </span>
              </button>
            </div>
          </div>
          {showHotkeys ? (
            <div className="graph-hotkey-guide" role="note" aria-label="Canvas keyboard shortcuts">
              <span>
                <strong>1-5</strong> quick add
              </span>
              <span>
                <strong>G</strong> add drawer
              </span>
              <span>
                <strong>I</strong> inspector
              </span>
              <span>
                <strong>R</strong> run drawer
              </span>
              <span>
                <strong>Space / F</strong> recenter camera
              </span>
              <span>
                <strong>L</strong> pan lock
              </span>
              <span>
                <strong>Esc</strong> close panels
              </span>
            </div>
          ) : null}
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
                  savedNodes={savedNodes}
                  onSavedNodeClick={addSavedNode}
                  onDeleteSavedNode={handleDeleteSavedNode}
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
                  onSaveNode={handleSaveNode}
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
