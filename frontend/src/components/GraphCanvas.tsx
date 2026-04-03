import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { CSSProperties, MouseEvent as ReactMouseEvent, TouchEvent as ReactTouchEvent } from "react";
import type { Connection, Edge as FlowEdge, EdgeChange, Node as FlowNode, NodeChange, OnConnectStartParams, Position, ReactFlowInstance } from "reactflow";
import { Background, BackgroundVariant, ConnectionLineType, Controls, MarkerType, MiniMap, ReactFlow } from "reactflow";
import "reactflow/dist/style.css";

import { EventTimeline } from "./EventTimeline";
import { EnvironmentAgentMenu } from "./EnvironmentAgentMenu";
import {
  buildOrthogonalPolylinePoints,
  buildPolylinePath,
  GraphCanvasConnectionLine,
  GraphCanvasEdge,
  getEdgeLabelPlacement,
  resolveEdgeRoutePoints,
} from "./GraphCanvasEdge";
import type { GraphCanvasEdgeData } from "./GraphCanvasEdge";
import { GraphInspector } from "./GraphInspector";
import { GraphCanvasNode } from "./GraphCanvasNode";
import type { GraphCanvasNodeData } from "./GraphCanvasNode";
import { ContextBuilderPayloadModal } from "./ContextBuilderPayloadModal";
import { DisplayResponseModal } from "./DisplayResponseModal";
import { PromptBlockDetailsModal } from "./PromptBlockDetailsModal";
import { ProviderSummary } from "./ProviderSummary";
import { ProviderDetailsModal } from "./ProviderDetailsModal";
import { ToolDetailsModal } from "./ToolDetailsModal";
import {
  API_FINAL_MESSAGE_HANDLE_ID,
  API_TOOL_CALL_HANDLE_ID,
  API_TOOL_CONTEXT_HANDLE_ID,
  canConnectNodes,
  createNodeFromProvider,
  createNodeFromSaved,
  createWireJunctionNode,
  defaultApiMessageCondition,
  defaultApiToolCallCondition,
  defaultConditionalCondition,
  defaultMcpTerminalOutputCondition,
  defaultToolFailureCondition,
  duplicateGraphNode,
  getApiToolContextTargetAnchorRatio,
  getToolSourceHandleAnchorRatio,
  inferToolEdgeSourceHandle,
  isApiModelNode,
  isApiOutputHandleId,
  isMcpContextProviderNode,
  isPromptBlockNode,
  isRoutableToolNode,
  isWireJunctionNode,
  MCP_TERMINAL_OUTPUT_HANDLE_ID,
  TOOL_CONTEXT_HANDLE_ID,
  TOOL_FAILURE_HANDLE_ID,
  TOOL_SUCCESS_HANDLE_ID,
} from "../lib/editor";
import type { GraphLayoutNodeDimensions } from "../lib/editor";
import { logGraphDiagnostic, useGraphDiagnosticsEnabled, useRenderDiagnostics, warnGraphDiagnostic } from "../lib/dragDiagnostics";
import { clearHotbarFavorite, getHotbarFavorites, setHotbarFavorite } from "../lib/hotbarFavorites";
import type { HotbarFavorites } from "../lib/hotbarFavorites";
import { deleteSavedNode, getSavedNodes, saveNodeToLibrary } from "../lib/savedNodes";
import type { SavedNode } from "../lib/savedNodes";
import { buildContextBuilderRuntimeView } from "../lib/contextBuilderRuntime";
import {
  formatRunStatusLabel,
  type AgentRunLane,
  type FocusedEventGroup,
  type FocusedRunProjection,
  type FocusedRunSummary,
} from "../lib/runVisualization";
import type { EditorCatalog, GraphDefinition, GraphEdge, GraphNode, GraphPosition, NodeProviderDefinition, RunState, RuntimeEvent } from "../lib/types";

type GraphCanvasProps = {
  graph: GraphDefinition | null;
  runState: RunState | null;
  events: RuntimeEvent[];
  activeRunId: string | null;
  isRunning: boolean;
  runButtonLabel?: string;
  focusedAgentName?: string | null;
  focusedAgentStatus?: string | null;
  environmentAgents?: AgentRunLane[];
  selectedAgentId?: string | null;
  onSelectAgent?: (agentId: string) => void;
  runProjection?: FocusedRunProjection | null;
  runSummary?: FocusedRunSummary | null;
  eventGroups?: FocusedEventGroup[];
  catalog: EditorCatalog | null;
  selectedNodeId: string | null;
  selectedEdgeId: string | null;
  onGraphChange: (graph: GraphDefinition) => void;
  onGraphDrag: (graph: GraphDefinition) => void;
  onFormatGraph: (nodeDimensions: Record<string, GraphLayoutNodeDimensions>) => void;
  onRunGraph: () => void;
  onScrollToTop: () => void;
  isMcpPanelOpen?: boolean;
  onToggleMcpPanel?: () => void;
  backgroundDragSensitivity?: number;
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

type NodeMeasuredDimensions = {
  width: number;
  height: number;
};

type WaypointDragState = {
  edgeId: string;
  waypointIndex: number;
  pointerOffset: GraphPosition;
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

type NodeClipboardState = {
  nodes: GraphNode[];
  edges: GraphEdge[];
  pasteCount: number;
};

type GraphSelectionState = {
  nodeIds: string[];
  edgeIds: string[];
};

type PrimarySelectionState = {
  nodeId: string | null;
  edgeId: string | null;
};

type SelectionRect = {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
};

type MarqueeSelectionState = {
  originClient: GraphPosition;
  currentClient: GraphPosition;
  originFlow: GraphPosition;
  currentFlow: GraphPosition;
};

const NODE_CLIPBOARD_PASTE_OFFSET = 48;

const KIND_COLORS: Record<string, string> = {
  input: "#8486a5",
  model: "#6c5ce7",
  provider: "#f59e0b",
  tool: "#a78bfa",
  mcp_context_provider: "#5aa7ff",
  mcp_tool_executor: "#a78bfa",
  data: "#2dd4bf",
  output: "#4ade80",
};

const KIND_LABELS: Record<string, string> = {
  input: "IN",
  model: "AI",
  provider: "PR",
  tool: "FX",
  mcp_context_provider: "MC",
  mcp_tool_executor: "MX",
  data: "DB",
  output: "OUT",
};

const nodeTypes = { graphNode: GraphCanvasNode };
const edgeTypes = { graphEdge: GraphCanvasEdge };
const NODE_STYLE = {
  width: "max-content",
  minWidth: 320,
  background: "transparent",
  border: "none",
  padding: 0,
  boxShadow: "none",
} as const;
const NODE_WIDTH = 320;
const NODE_HEIGHT = 178;
const NODE_REGION_HEIGHT = 198;
const MODEL_NODE_HEIGHT = 228;
const MODEL_NODE_REGION_HEIGHT = 252;
const JUNCTION_NODE_SIZE = 24;
const DEFAULT_BACKGROUND_DRAG_SENSITIVITY = 0.28;
const VIEWPORT_SYNC_EPSILON = 0.25;
const MIN_GRAPH_ZOOM = 0.1;
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
const AUTO_CONNECT_TARGET_VERTICAL_GAP = 40;
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
const API_EDGE_TONES = {
  toolCall: {
    sourceColor: "#f4a8ff",
    targetColor: "#d97bff",
    markerColor: "#f4a8ff",
    routeTone: "api-tool-call" as const,
  },
  message: {
    sourceColor: "#7eb8ff",
    targetColor: "#55a3ff",
    markerColor: "#7eb8ff",
    routeTone: "api-message" as const,
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

type DragDiagnosticSession = {
  active: boolean;
  sessionId: number;
  startedAt: number;
  dragMoveEvents: number;
  draggedNodeChanges: number;
  rafTicks: number;
  nodeBuilds: number;
  nodeBuildTotalMs: number;
  nodeBuildMaxMs: number;
  drawerOpen: boolean;
  drawerTab: DrawerTab | "closed";
  minimapVisible: boolean;
  nodeCount: number;
  edgeCount: number;
};

function createDragDiagnosticSession(): DragDiagnosticSession {
  return {
    active: false,
    sessionId: 0,
    startedAt: 0,
    dragMoveEvents: 0,
    draggedNodeChanges: 0,
    rafTicks: 0,
    nodeBuilds: 0,
    nodeBuildTotalMs: 0,
    nodeBuildMaxMs: 0,
    drawerOpen: false,
    drawerTab: "closed",
    minimapVisible: true,
    nodeCount: 0,
    edgeCount: 0,
  };
}

function roundDiagnosticValue(value: number) {
  return Number(value.toFixed(2));
}

function cloneClipboardNodes(nodes: GraphNode[]) {
  return JSON.parse(JSON.stringify(nodes)) as GraphNode[];
}

function cloneClipboardEdges(edges: GraphEdge[]) {
  return JSON.parse(JSON.stringify(edges)) as GraphEdge[];
}

function normalizeSelectionRect(start: GraphPosition, end: GraphPosition): SelectionRect {
  return {
    minX: Math.min(start.x, end.x),
    maxX: Math.max(start.x, end.x),
    minY: Math.min(start.y, end.y),
    maxY: Math.max(start.y, end.y),
  };
}

function selectionRectWidth(rect: SelectionRect) {
  return rect.maxX - rect.minX;
}

function selectionRectHeight(rect: SelectionRect) {
  return rect.maxY - rect.minY;
}

function pointInsideSelectionRect(point: GraphPosition, rect: SelectionRect) {
  return point.x >= rect.minX && point.x <= rect.maxX && point.y >= rect.minY && point.y <= rect.maxY;
}

function rectsIntersect(left: SelectionRect, right: SelectionRect) {
  return !(
    left.maxX < right.minX ||
    left.minX > right.maxX ||
    left.maxY < right.minY ||
    left.minY > right.maxY
  );
}

function segmentOrientation(start: GraphPosition, middle: GraphPosition, end: GraphPosition) {
  const cross = (middle.y - start.y) * (end.x - middle.x) - (middle.x - start.x) * (end.y - middle.y);
  if (Math.abs(cross) <= 0.0001) {
    return 0;
  }
  return cross > 0 ? 1 : 2;
}

function pointOnSegment(start: GraphPosition, point: GraphPosition, end: GraphPosition) {
  return (
    point.x <= Math.max(start.x, end.x) + 0.0001 &&
    point.x + 0.0001 >= Math.min(start.x, end.x) &&
    point.y <= Math.max(start.y, end.y) + 0.0001 &&
    point.y + 0.0001 >= Math.min(start.y, end.y)
  );
}

function segmentsIntersect(startA: GraphPosition, endA: GraphPosition, startB: GraphPosition, endB: GraphPosition) {
  const orientation1 = segmentOrientation(startA, endA, startB);
  const orientation2 = segmentOrientation(startA, endA, endB);
  const orientation3 = segmentOrientation(startB, endB, startA);
  const orientation4 = segmentOrientation(startB, endB, endA);

  if (orientation1 !== orientation2 && orientation3 !== orientation4) {
    return true;
  }
  if (orientation1 === 0 && pointOnSegment(startA, startB, endA)) {
    return true;
  }
  if (orientation2 === 0 && pointOnSegment(startA, endB, endA)) {
    return true;
  }
  if (orientation3 === 0 && pointOnSegment(startB, startA, endB)) {
    return true;
  }
  return orientation4 === 0 && pointOnSegment(startB, endA, endB);
}

function polylineIntersectsSelectionRect(points: GraphPosition[], rect: SelectionRect) {
  if (points.length === 0) {
    return false;
  }
  if (points.some((point) => pointInsideSelectionRect(point, rect))) {
    return true;
  }

  const corners = [
    { x: rect.minX, y: rect.minY },
    { x: rect.maxX, y: rect.minY },
    { x: rect.maxX, y: rect.maxY },
    { x: rect.minX, y: rect.maxY },
  ];
  const rectSegments: Array<[GraphPosition, GraphPosition]> = [
    [corners[0], corners[1]],
    [corners[1], corners[2]],
    [corners[2], corners[3]],
    [corners[3], corners[0]],
  ];

  for (let index = 1; index < points.length; index += 1) {
    const start = points[index - 1];
    const end = points[index];
    if (rectSegments.some(([rectStart, rectEnd]) => segmentsIntersect(start, end, rectStart, rectEnd))) {
      return true;
    }
  }
  return false;
}

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
  } catch (error) {
    warnGraphDiagnostic("GraphCanvas", "edge path sampling fallback", error, {
      edgeId,
      edgePath,
      shiftX,
      shiftY,
    });
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
    label: "Start",
    description: "Create a run-triggered start node.",
    category: "start",
    preferredProviderIds: ["start.manual_run", "core.input"],
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
    description: "Create a tool or MCP node.",
    category: "tool",
    preferredProviderIds: ["tool.mcp_context_provider", "tool.mcp_tool_executor", "tool.registry"],
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
    preferredProviderIds: ["core.output", "end.discord_message"],
  },
];

function resolveQuickAddProvider(
  providers: NodeProviderDefinition[],
  slot: QuickAddSlot,
  favoriteProviderId?: string | null,
): NodeProviderDefinition | null {
  if (favoriteProviderId) {
    const favorite = providers.find(
      (provider) => provider.provider_id === favoriteProviderId && provider.category === slot.category,
    );
    if (favorite) {
      return favorite;
    }
  }
  for (const providerId of slot.preferredProviderIds) {
    const preferred = providers.find((provider) => provider.provider_id === providerId);
    if (preferred) {
      return preferred;
    }
  }
  return providers.find((provider) => provider.category === slot.category) ?? null;
}

function participatesInExecutionRoute(graph: GraphDefinition, node: GraphNode): boolean {
  if (node.category === "start") {
    return true;
  }
  return graph.edges.some(
    (edge) => edge.kind !== "binding" && (edge.source_id === node.id || edge.target_id === node.id),
  );
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

function updateNode(graph: GraphDefinition, nodeId: string, updater: (node: GraphNode) => GraphNode): GraphDefinition {
  return {
    ...graph,
    nodes: graph.nodes.map((node) => (node.id === nodeId ? updater(node) : node)),
  };
}

const MILESTONE_CHAT_MAX_VISIBLE = 6;
/** Max age before a line is dropped; opacity eases over this window (not linear). */
const MILESTONE_CHAT_MAX_AGE_MS = 14_000;
/** Ticks + CSS transition interpolate smoothly between updates. */
const MILESTONE_CHAT_TICK_MS = 300;

export function GraphCanvas({
  graph,
  runState,
  events,
  activeRunId,
  isRunning,
  runButtonLabel = "Run Graph",
  focusedAgentName = null,
  focusedAgentStatus = null,
  environmentAgents = [],
  selectedAgentId = null,
  onSelectAgent,
  runProjection = null,
  runSummary,
  eventGroups = [],
  catalog,
  selectedNodeId,
  selectedEdgeId,
  onGraphChange,
  onGraphDrag,
  onFormatGraph,
  onRunGraph,
  onScrollToTop,
  isMcpPanelOpen = false,
  onToggleMcpPanel,
  backgroundDragSensitivity = DEFAULT_BACKGROUND_DRAG_SENSITIVITY,
  onSelectionChange,
}: GraphCanvasProps) {
  const [flowInstance, setFlowInstance] = useState<ReactFlowInstance | null>(null);
  const [editorMessage, setEditorMessage] = useState<string | null>(null);
  const [toolDetailsNodeId, setToolDetailsNodeId] = useState<string | null>(null);
  const [providerDetailsNodeId, setProviderDetailsNodeId] = useState<string | null>(null);
  const [promptBlockDetailsNodeId, setPromptBlockDetailsNodeId] = useState<string | null>(null);
  const [displayResponseNodeId, setDisplayResponseNodeId] = useState<string | null>(null);
  const [contextBuilderPayloadNodeId, setContextBuilderPayloadNodeId] = useState<string | null>(null);
  const [tooltipNodeId, setTooltipNodeId] = useState<string | null>(null);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [drawerTab, setDrawerTab] = useState<DrawerTab>("add");
  const [providerQuery, setProviderQuery] = useState("");
  const [isProviderDragActive, setIsProviderDragActive] = useState(false);
  const [isSavedNodeDragActive, setIsSavedNodeDragActive] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [savedNodes, setSavedNodes] = useState<SavedNode[]>(() => getSavedNodes());
  const [hotbarFavorites, setHotbarFavorites] = useState<HotbarFavorites>(() => getHotbarFavorites());
  const [panLocked, setPanLocked] = useState(false);
  const [isCommandHeld, setIsCommandHeld] = useState(false);
  const [showHotkeys, setShowHotkeys] = useState(false);
  const [quickAddMinimized, setQuickAddMinimized] = useState(false);
  const [milestoneChatNow, setMilestoneChatNow] = useState(() => Date.now());
  const [junctionDrag, setJunctionDrag] = useState<JunctionDragState | null>(null);
  const [waypointDrag, setWaypointDrag] = useState<WaypointDragState | null>(null);
  const [isNodeDragActive, setIsNodeDragActive] = useState(false);
  const [dragRenderTick, setDragRenderTick] = useState(0);
  const [pendingPlacement, setPendingPlacement] = useState<PlacementState | null>(null);
  const [draftConnection, setDraftConnection] = useState<DraftConnectionState | null>(null);
  const [draftConnectionSnapTargetNodeId, setDraftConnectionSnapTargetNodeId] = useState<string | null>(null);
  const [viewportState, setViewportState] = useState<ViewportState>({ x: 0, y: 0, zoom: 1 });
  const [selectedNodeIds, setSelectedNodeIds] = useState<string[]>(() => (selectedNodeId ? [selectedNodeId] : []));
  const [selectedEdgeIds, setSelectedEdgeIds] = useState<string[]>(() => (selectedEdgeId ? [selectedEdgeId] : []));
  const [marqueeSelection, setMarqueeSelection] = useState<MarqueeSelectionState | null>(null);
  const dragDiagnosticsEnabled = useGraphDiagnosticsEnabled();
  const canvasRef = useRef<HTMLDivElement | null>(null);
  const canvasZoomRef = useRef(1);
  const graphRef = useRef<GraphDefinition | null>(graph);
  const lastRequestedPrimarySelectionRef = useRef<PrimarySelectionState>({ nodeId: selectedNodeId, edgeId: selectedEdgeId });
  const draftConnectionRef = useRef<DraftConnectionState | null>(draftConnection);
  const nodeClipboardRef = useRef<NodeClipboardState | null>(null);
  const draftWirePathRef = useRef<SVGPathElement | null>(null);
  const draftPreviewPointerPositionRef = useRef<GraphPosition | null>(null);
  const draftSnapTargetNodeIdRef = useRef<string | null>(null);
  const isConnectingRef = useRef(isConnecting);
  const nodeDataCacheRef = useRef(new Map<string, GraphCanvasRuntimeNodeData>());
  const flowNodeCacheRef = useRef(new Map<string, FlowNode<GraphCanvasRuntimeNodeData>>());
  const measuredNodeDimensionsRef = useRef(new Map<string, NodeMeasuredDimensions>());
  const dragFrameRef = useRef<number | null>(null);
  const pendingDragGraphRef = useRef<GraphDefinition | null>(null);
  const dragPositionMapRef = useRef<Map<string, GraphPosition> | null>(null);
  const nodeDragRafRef = useRef<number | null>(null);
  const cachedEdgesRef = useRef<FlowEdge<GraphCanvasEdgeData>[]>([]);
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
  const dragDiagnosticSessionRef = useRef<DragDiagnosticSession>(createDragDiagnosticSession());
  const dragDiagnosticSessionIdRef = useRef(0);
  const zoomAnimationDuration = 120;
  const vizLocked = panLocked || isCommandHeld;
  const toolDetailsNode = useMemo(
    () =>
      graph?.nodes.find((node) => node.id === toolDetailsNodeId && node.category === "tool") ??
      null,
    [graph, toolDetailsNodeId],
  );
  const providerDetailsNode = useMemo(
    () => graph?.nodes.find((node) => node.id === providerDetailsNodeId && node.kind === "model") ?? null,
    [graph, providerDetailsNodeId],
  );
  const promptBlockDetailsNode = useMemo(
    () => graph?.nodes.find((node) => node.id === promptBlockDetailsNodeId && node.provider_id === "core.prompt_block") ?? null,
    [graph, promptBlockDetailsNodeId],
  );
  const displayResponseNode = useMemo(
    () => graph?.nodes.find((node) => node.id === displayResponseNodeId && node.provider_id === "core.data_display") ?? null,
    [displayResponseNodeId, graph],
  );
  const contextBuilderPayloadNode = useMemo(
    () => graph?.nodes.find((node) => node.id === contextBuilderPayloadNodeId && node.provider_id === "core.context_builder") ?? null,
    [contextBuilderPayloadNodeId, graph],
  );
  const contextBuilderModalRuntime = useMemo(() => {
    if (!contextBuilderPayloadNode || !graph || !runState) {
      return null;
    }
    const normalizedEvents = runProjection?.normalizedEvents ?? runState.event_history ?? [];
    return buildContextBuilderRuntimeView(graph, contextBuilderPayloadNode, runState, normalizedEvents);
  }, [contextBuilderPayloadNode, graph, runProjection?.normalizedEvents, runState]);
  const selectedNodeIdSet = useMemo(() => new Set(selectedNodeIds), [selectedNodeIds]);
  const selectedEdgeIdSet = useMemo(() => new Set(selectedEdgeIds), [selectedEdgeIds]);
  const milestoneChatEntries = useMemo(() => {
    const entries = environmentAgents
      .flatMap((agent) =>
        agent.milestones
          .filter((milestone) => milestone.eventType !== "edge.selected")
          .map((milestone) => {
          const parsedTimestampMs = Date.parse(milestone.timestamp);
          const timestampMs = Number.isFinite(parsedTimestampMs) ? parsedTimestampMs : milestoneChatNow;
          const ageMs = Math.max(0, milestoneChatNow - timestampMs);
          return {
            id: milestone.id,
            label: milestone.label,
            timestampMs,
            ageMs,
          };
          }),
      )
      .filter((entry) => entry.ageMs <= MILESTONE_CHAT_MAX_AGE_MS)
      .sort((left, right) => left.timestampMs - right.timestampMs)
      .slice(-MILESTONE_CHAT_MAX_VISIBLE);
    return entries.map((entry, index) => {
      const positionFraction = entries.length > 1 ? index / (entries.length - 1) : 1;
      const staggerBonusMs = (1 - positionFraction) * MILESTONE_CHAT_MAX_AGE_MS * 0.35;
      const effectiveAge = entry.ageMs + staggerBonusMs;
      const t = Math.min(1, effectiveAge / MILESTONE_CHAT_MAX_AGE_MS);
      // Ease-out curve: stays readable longer, then tapers smoothly (not a harsh linear cliff).
      const opacity = Math.max(0, Math.pow(1 - t, 1.35));
      return {
        ...entry,
        opacity,
      };
    });
  }, [environmentAgents, milestoneChatNow]);

  useRenderDiagnostics(
    "GraphCanvas",
    isNodeDragActive || drawerOpen || junctionDrag !== null || waypointDrag !== null,
    {
      drawerOpen,
      drawerTab,
      isNodeDragActive,
      isJunctionDragActive: junctionDrag !== null,
      isWaypointDragActive: waypointDrag !== null,
      nodeCount: graph?.nodes.length ?? 0,
      edgeCount: graph?.edges.length ?? 0,
    },
    12,
  );

  useEffect(() => {
    const hasMilestones = environmentAgents.some((agent) => agent.milestones.length > 0);
    if (!hasMilestones) {
      return;
    }
    setMilestoneChatNow(Date.now());
    const intervalId = window.setInterval(() => {
      setMilestoneChatNow(Date.now());
    }, MILESTONE_CHAT_TICK_MS);
    return () => {
      window.clearInterval(intervalId);
    };
  }, [environmentAgents]);

  const requestPrimarySelectionChange = useCallback(
    (nodeId: string | null, edgeId: string | null) => {
      lastRequestedPrimarySelectionRef.current = { nodeId, edgeId };
      onSelectionChange(nodeId, edgeId);
    },
    [onSelectionChange],
  );

  const clearFlowInstanceSelection = useCallback(() => {
    if (!flowInstance) {
      return;
    }
    flowInstance.setNodes((currentNodes) =>
      currentNodes.map((node) => (node.selected ? { ...node, selected: false } : node)),
    );
    flowInstance.setEdges((currentEdges) =>
      currentEdges.map((edge) => (edge.selected ? { ...edge, selected: false } : edge)),
    );
  }, [flowInstance]);

  const setCanvasSelection = useCallback(
    (selection: GraphSelectionState) => {
      const validNodeIds = graph
        ? graph.nodes.filter((node) => selection.nodeIds.includes(node.id)).map((node) => node.id)
        : selection.nodeIds;
      const validEdgeIds = graph
        ? graph.edges.filter((edge) => selection.edgeIds.includes(edge.id)).map((edge) => edge.id)
        : selection.edgeIds;
      setSelectedNodeIds(validNodeIds);
      setSelectedEdgeIds(validEdgeIds);
      requestPrimarySelectionChange(
        validNodeIds.length === 1 && validEdgeIds.length === 0 ? validNodeIds[0] : null,
        validEdgeIds.length === 1 && validNodeIds.length === 0 ? validEdgeIds[0] : null,
      );
    },
    [graph, requestPrimarySelectionChange],
  );

  const clearCanvasSelection = useCallback(() => {
    clearFlowInstanceSelection();
    setCanvasSelection({ nodeIds: [], edgeIds: [] });
  }, [clearFlowInstanceSelection, setCanvasSelection]);

  const beginDragDiagnosticSession = useCallback(() => {
    if (!dragDiagnosticsEnabled) {
      return;
    }

    const sessionId = dragDiagnosticSessionIdRef.current + 1;
    dragDiagnosticSessionIdRef.current = sessionId;
    dragDiagnosticSessionRef.current = {
      active: true,
      sessionId,
      startedAt: performance.now(),
      dragMoveEvents: 0,
      draggedNodeChanges: 0,
      rafTicks: 0,
      nodeBuilds: 0,
      nodeBuildTotalMs: 0,
      nodeBuildMaxMs: 0,
      drawerOpen,
      drawerTab: drawerOpen ? drawerTab : "closed",
      minimapVisible: true,
      nodeCount: graph?.nodes.length ?? 0,
      edgeCount: graph?.edges.length ?? 0,
    };

    logGraphDiagnostic("GraphCanvas", "node-drag-start", {
      sessionId,
      drawerOpen,
      drawerTab: drawerOpen ? drawerTab : "closed",
      nodeCount: graph?.nodes.length ?? 0,
      edgeCount: graph?.edges.length ?? 0,
      minimapVisible: true,
    });
  }, [dragDiagnosticsEnabled, drawerOpen, drawerTab, graph]);

  const flushDragDiagnosticSession = useCallback(
    (reason: string) => {
      if (!dragDiagnosticsEnabled) {
        return;
      }

      const session = dragDiagnosticSessionRef.current;
      if (!session.active) {
        return;
      }

      const elapsedMs = performance.now() - session.startedAt;
      logGraphDiagnostic("GraphCanvas", "node-drag-summary", {
        reason,
        sessionId: session.sessionId,
        elapsedMs: roundDiagnosticValue(elapsedMs),
        dragMoveEvents: session.dragMoveEvents,
        draggedNodeChanges: session.draggedNodeChanges,
        rafTicks: session.rafTicks,
        nodeBuilds: session.nodeBuilds,
        avgNodeBuildMs:
          session.nodeBuilds > 0 ? roundDiagnosticValue(session.nodeBuildTotalMs / session.nodeBuilds) : 0,
        maxNodeBuildMs: roundDiagnosticValue(session.nodeBuildMaxMs),
        drawerOpen: session.drawerOpen,
        drawerTab: session.drawerTab,
        minimapVisible: session.minimapVisible,
        nodeCount: session.nodeCount,
        edgeCount: session.edgeCount,
      });

      dragDiagnosticSessionRef.current = createDragDiagnosticSession();
    },
    [dragDiagnosticsEnabled],
  );

  const recordNodeBuildDiagnostic = useCallback(
    (durationMs: number) => {
      if (!dragDiagnosticsEnabled) {
        return;
      }

      const session = dragDiagnosticSessionRef.current;
      if (!session.active) {
        return;
      }

      session.nodeBuilds += 1;
      session.nodeBuildTotalMs += durationMs;
      session.nodeBuildMaxMs = Math.max(session.nodeBuildMaxMs, durationMs);

      if (durationMs >= 8) {
        logGraphDiagnostic("GraphCanvas", "slow-node-build", {
          sessionId: session.sessionId,
          durationMs: roundDiagnosticValue(durationMs),
          nodeCount: session.nodeCount,
          drawerOpen: session.drawerOpen,
          drawerTab: session.drawerTab,
        });
      }
    },
    [dragDiagnosticsEnabled],
  );

  useEffect(() => {
    graphRef.current = graph;
  }, [graph]);

  useEffect(() => {
    return () => flushDragDiagnosticSession("unmount");
  }, [flushDragDiagnosticSession]);

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

  useEffect(() => {
    const requestedSelection = lastRequestedPrimarySelectionRef.current;
    if (selectedNodeId === requestedSelection.nodeId && selectedEdgeId === requestedSelection.edgeId) {
      return;
    }
    setSelectedNodeIds(selectedNodeId ? [selectedNodeId] : []);
    setSelectedEdgeIds(selectedEdgeId ? [selectedEdgeId] : []);
    lastRequestedPrimarySelectionRef.current = { nodeId: selectedNodeId, edgeId: selectedEdgeId };
  }, [selectedEdgeId, selectedNodeId]);

  useEffect(() => {
    if (!graph) {
      if (selectedNodeIds.length > 0 || selectedEdgeIds.length > 0) {
        setSelectedNodeIds([]);
        setSelectedEdgeIds([]);
      }
      return;
    }
    const validNodeIds = selectedNodeIds.filter((nodeId) => graph.nodes.some((node) => node.id === nodeId));
    const validEdgeIds = selectedEdgeIds.filter((edgeId) => graph.edges.some((edge) => edge.id === edgeId));
    if (validNodeIds.length === selectedNodeIds.length && validEdgeIds.length === selectedEdgeIds.length) {
      return;
    }
    setSelectedNodeIds(validNodeIds);
    setSelectedEdgeIds(validEdgeIds);
    requestPrimarySelectionChange(
      validNodeIds.length === 1 && validEdgeIds.length === 0 ? validNodeIds[0] : null,
      validEdgeIds.length === 1 && validNodeIds.length === 0 ? validEdgeIds[0] : null,
    );
  }, [graph, requestPrimarySelectionChange, selectedEdgeIds, selectedNodeIds]);

  const isEditableTarget = useCallback((target: EventTarget | null): boolean => {
    if (!(target instanceof HTMLElement)) {
      return false;
    }
    const tag = target.tagName;
    return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || target.isContentEditable;
  }, []);

  const hasSelectedText = useCallback((): boolean => {
    const selection = window.getSelection();
    return Boolean(selection && !selection.isCollapsed && selection.toString().trim().length > 0);
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
      setPromptBlockDetailsNodeId(null);
      setToolDetailsNodeId(nodeId);
    },
    [onSelectionChange],
  );

  const handleToggleExecutorRetries = useCallback(
    (nodeId: string) => {
      if (!graph) {
        return;
      }
      onGraphChange(
        updateNode(graph, nodeId, (node) => ({
          ...node,
          config: {
            ...node.config,
            allow_retries: !(node.config.allow_retries ?? true),
          },
        })),
      );
    },
    [graph, onGraphChange],
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
    if (node.kind === "model") {
      return { width: NODE_WIDTH, height: MODEL_NODE_HEIGHT, regionHeight: MODEL_NODE_REGION_HEIGHT };
    }
    return { width: NODE_WIDTH, height: NODE_HEIGHT, regionHeight: NODE_REGION_HEIGHT };
  }, []);

  const rebalanceOutgoingTargets = useCallback(
    (baseGraph: GraphDefinition, sourceNodeId: string): GraphDefinition => {
      const sourceNode = baseGraph.nodes.find((node) => node.id === sourceNodeId);
      if (!sourceNode || isWireJunctionNode(sourceNode)) {
        return baseGraph;
      }

      const outgoingEdges = baseGraph.edges.filter((edge) => edge.source_id === sourceNodeId && edge.kind !== "binding");
      if (outgoingEdges.length === 0) {
        return baseGraph;
      }

      const sourceDimensions = getNodeDimensions(sourceNode);
      const repositionCandidates = outgoingEdges
        .map((edge) => {
          const targetNode = baseGraph.nodes.find((node) => node.id === edge.target_id);
          if (!targetNode || isWireJunctionNode(targetNode)) {
            return null;
          }
          const incomingEdgeCount = baseGraph.edges.filter(
            (candidate) => candidate.target_id === targetNode.id && candidate.kind !== "binding",
          ).length;
          if (incomingEdgeCount !== 1) {
            return null;
          }

          const sourceHandleId = inferToolEdgeSourceHandle(edge, sourceNode);
          const sourceAnchorRatio =
            isApiModelNode(sourceNode) || isRoutableToolNode(sourceNode) || isMcpContextProviderNode(sourceNode)
              ? getToolSourceHandleAnchorRatio(sourceHandleId ?? TOOL_SUCCESS_HANDLE_ID)
              : 0.5;
          const targetDimensions = getNodeDimensions(targetNode);
          const targetAnchorRatio =
            targetNode.kind === "model" ? getApiToolContextTargetAnchorRatio(edge.target_handle_id ?? null) : 0.5;
          const desiredTop = sourceNode.position.y + sourceDimensions.height * sourceAnchorRatio - targetDimensions.height * targetAnchorRatio;

          return {
            targetNode,
            desiredTop,
            currentTop: targetNode.position.y,
            height: targetDimensions.height,
          };
        })
        .filter(
          (
            candidate,
          ): candidate is {
            targetNode: GraphNode;
            desiredTop: number;
            currentTop: number;
            height: number;
          } => candidate !== null,
        )
        .sort((left, right) => {
          const desiredDelta = left.desiredTop - right.desiredTop;
          if (Math.abs(desiredDelta) > 0.5) {
            return desiredDelta;
          }
          const currentDelta = left.currentTop - right.currentTop;
          if (Math.abs(currentDelta) > 0.5) {
            return currentDelta;
          }
          return left.targetNode.id.localeCompare(right.targetNode.id);
        });

      if (repositionCandidates.length === 0) {
        return baseGraph;
      }

      const nextPositions = new Map<string, GraphPosition>();
      let previousBottom = Number.NEGATIVE_INFINITY;
      repositionCandidates.forEach((candidate) => {
        const nextTop =
          previousBottom === Number.NEGATIVE_INFINITY
            ? candidate.desiredTop
            : Math.max(candidate.desiredTop, previousBottom + AUTO_CONNECT_TARGET_VERTICAL_GAP);
        previousBottom = nextTop + candidate.height;
        nextPositions.set(candidate.targetNode.id, {
          ...candidate.targetNode.position,
          y: nextTop,
        });
      });

      return {
        ...baseGraph,
        nodes: baseGraph.nodes.map((node) => {
          const nextPosition = nextPositions.get(node.id);
          return nextPosition ? { ...node, position: nextPosition } : node;
        }),
      };
    },
    [getNodeDimensions],
  );

  const updateMeasuredNodeDimensions = useCallback((nodeId: string, dimensions?: { width?: number; height?: number } | null) => {
    const width = dimensions?.width;
    const height = dimensions?.height;
    if (typeof width === "number" && width > 0 && typeof height === "number" && height > 0) {
      measuredNodeDimensionsRef.current.set(nodeId, { width, height });
    }
  }, []);

  const getFlowNodeDimensions = useCallback(
    (node: GraphNode, cachedNode?: FlowNode<GraphCanvasRuntimeNodeData>): NodeMeasuredDimensions => {
      const fallbackDimensions = getNodeDimensions(node);
      const measuredDimensions = measuredNodeDimensionsRef.current.get(node.id);
      const width = cachedNode?.width ?? measuredDimensions?.width ?? fallbackDimensions.width;
      const height = cachedNode?.height ?? measuredDimensions?.height ?? fallbackDimensions.height;
      return { width, height };
    },
    [getNodeDimensions],
  );

  useEffect(() => {
    if (!graph) {
      measuredNodeDimensionsRef.current.clear();
      return;
    }
    const activeNodeIds = new Set(graph.nodes.map((node) => node.id));
    for (const nodeId of measuredNodeDimensionsRef.current.keys()) {
      if (!activeNodeIds.has(nodeId)) {
        measuredNodeDimensionsRef.current.delete(nodeId);
      }
    }
  }, [graph]);

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

  const getPointerPanOutsideDistance = useCallback(
    (viewport: ViewportState) => {
      if (!nodeRegionBounds) {
        return 0;
      }
      const bounds = canvasRef.current?.getBoundingClientRect();
      if (!bounds || viewport.zoom <= 0) {
        return 0;
      }
      const visibleWidth = bounds.width / viewport.zoom;
      const visibleHeight = bounds.height / viewport.zoom;
      const centerX = (bounds.width * 0.5 - viewport.x) / viewport.zoom;
      const centerY = (bounds.height * 0.5 - viewport.y) / viewport.zoom;
      const paddingX = Math.max(100, visibleWidth * 0.15);
      const paddingY = Math.max(80, visibleHeight * 0.15);
      const outsideX = Math.max(nodeRegionBounds.minX - paddingX - centerX, 0, centerX - (nodeRegionBounds.maxX + paddingX));
      const outsideY = Math.max(nodeRegionBounds.minY - paddingY - centerY, 0, centerY - (nodeRegionBounds.maxY + paddingY));
      return Math.hypot(outsideX, outsideY);
    },
    [nodeRegionBounds],
  );

  const getPointerPanDamping = useCallback(
    (previousViewport: ViewportState, nextViewport: ViewportState) => {
      const previousDistance = getPointerPanOutsideDistance(previousViewport);
      const nextDistance = getPointerPanOutsideDistance(nextViewport);
      if (nextDistance <= previousDistance + 0.01) {
        return 1;
      }
      const bounds = canvasRef.current?.getBoundingClientRect();
      if (!bounds || nextViewport.zoom <= 0) {
        return 1;
      }
      const visibleWidth = bounds.width / nextViewport.zoom;
      const visibleHeight = bounds.height / nextViewport.zoom;
      const falloffDistance = Math.max(100, Math.min(220, Math.max(visibleWidth, visibleHeight) * 0.14));
      const normalizedDistance = nextDistance / falloffDistance;
      return (
        backgroundDragSensitivity +
        (1 - backgroundDragSensitivity) * Math.exp(-1.9 * normalizedDistance * normalizedDistance)
      );
    },
    [backgroundDragSensitivity, getPointerPanOutsideDistance],
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

      const damping = getPointerPanDamping(previousAppliedViewport, viewport);
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
      setPromptBlockDetailsNodeId(null);
      setProviderDetailsNodeId(nodeId);
    },
    [onSelectionChange],
  );

  const handleOpenPromptBlockDetails = useCallback(
    (nodeId: string) => {
      onSelectionChange(nodeId, null);
      setTooltipNodeId(null);
      setToolDetailsNodeId(null);
      setProviderDetailsNodeId(null);
      setPromptBlockDetailsNodeId(nodeId);
      setDisplayResponseNodeId(null);
      setContextBuilderPayloadNodeId(null);
    },
    [onSelectionChange],
  );

  const handleOpenDisplayResponse = useCallback(
    (nodeId: string) => {
      onSelectionChange(nodeId, null);
      setPromptBlockDetailsNodeId(null);
      setContextBuilderPayloadNodeId(null);
      setDisplayResponseNodeId(nodeId);
    },
    [onSelectionChange],
  );

  const handleOpenContextBuilderPayload = useCallback(
    (nodeId: string) => {
      onSelectionChange(nodeId, null);
      setPromptBlockDetailsNodeId(null);
      setDisplayResponseNodeId(null);
      setContextBuilderPayloadNodeId(nodeId);
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
    clearCanvasSelection();
    setTooltipNodeId(null);
    setToolDetailsNodeId(null);
    setProviderDetailsNodeId(null);
    setPromptBlockDetailsNodeId(null);
    setDisplayResponseNodeId(null);
    setContextBuilderPayloadNodeId(null);
    setDrawerOpen(false);
    setShowHotkeys(false);
  }, [clearCanvasSelection]);

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
      const verticalRatio = isApiModelNode(sourceNode)
        ? sourceHandleId
          ? getToolSourceHandleAnchorRatio(sourceHandleId)
          : 0.5
        : isRoutableToolNode(sourceNode) || isMcpContextProviderNode(sourceNode)
          ? getToolSourceHandleAnchorRatio(sourceHandleId ?? TOOL_SUCCESS_HANDLE_ID)
          : 0.5;
      return {
        x: sourceNode.position.x + dimensions.width,
        y: sourceNode.position.y + dimensions.height * verticalRatio,
      };
    },
    [getNodeDimensions, graph],
  );

  const getTargetAnchorPosition = useCallback(
    (nodeId: string, targetHandleId: string | null = null): GraphPosition | null => {
      const targetNode = graph?.nodes.find((node) => node.id === nodeId);
      if (!targetNode) {
        return null;
      }
      const dimensions = getNodeDimensions(targetNode);
      const verticalRatio =
        targetNode.kind === "model" ? getApiToolContextTargetAnchorRatio(targetHandleId) : 0.5;
      return {
        x: targetNode.position.x,
        y: targetNode.position.y + dimensions.height * verticalRatio,
      };
    },
    [getNodeDimensions, graph],
  );

  const getSelectionForRect = useCallback(
    (rect: SelectionRect): GraphSelectionState => {
      if (!graph) {
        return { nodeIds: [], edgeIds: [] };
      }

      const selectedNodeIds = graph.nodes
        .filter((node) =>
          rectsIntersect(rect, {
            minX: node.position.x,
            maxX: node.position.x + getFlowNodeDimensions(node).width,
            minY: node.position.y,
            maxY: node.position.y + getFlowNodeDimensions(node).height,
          }),
        )
        .map((node) => node.id);
      const selectedNodeIdSet = new Set(selectedNodeIds);
      const selectedEdgeIds = graph.edges
        .filter((edge) => {
          if (selectedNodeIdSet.has(edge.source_id) && selectedNodeIdSet.has(edge.target_id)) {
            return true;
          }
          const sourceNode = graph.nodes.find((node) => node.id === edge.source_id);
          const targetNode = graph.nodes.find((node) => node.id === edge.target_id);
          if (!sourceNode || !targetNode) {
            return false;
          }
          const sourceHandleId = inferToolEdgeSourceHandle(edge, sourceNode);
          const sourceAnchor = getSourceAnchorPosition(edge.source_id, sourceHandleId);
          const targetAnchor = getTargetAnchorPosition(edge.target_id, edge.target_handle_id ?? null);
          if (!sourceAnchor || !targetAnchor) {
            return false;
          }
          const routePoints = resolveEdgeRoutePoints(sourceAnchor, targetAnchor, edge.waypoints ?? [], { endWithHorizontal: true });
          return polylineIntersectsSelectionRect(routePoints, rect);
        })
        .map((edge) => edge.id);

      return { nodeIds: selectedNodeIds, edgeIds: selectedEdgeIds };
    },
    [getFlowNodeDimensions, getSourceAnchorPosition, getTargetAnchorPosition, graph],
  );

  const isMarqueeSelectionTarget = useCallback((target: EventTarget | null) => {
    if (!(target instanceof HTMLElement)) {
      return false;
    }
    if (target.closest(PLACEMENT_UI_SELECTOR) || target.closest(".react-flow__node") || target.closest(".react-flow__edge")) {
      return false;
    }
    return Boolean(target.closest(".react-flow__pane"));
  }, []);

  const handleCanvasMouseDownCapture = useCallback(
    (event: ReactMouseEvent<HTMLDivElement>) => {
      if (
        event.button !== 0 ||
        !event.shiftKey ||
        pendingPlacement ||
        draftConnection ||
        junctionDrag ||
        waypointDrag ||
        !isMarqueeSelectionTarget(event.target)
      ) {
        return;
      }
      const flowPosition = getFlowPositionFromScreen({ x: event.clientX, y: event.clientY });
      if (!flowPosition) {
        return;
      }
      setTooltipNodeId(null);
      event.preventDefault();
      event.stopPropagation();
      setMarqueeSelection({
        originClient: { x: event.clientX, y: event.clientY },
        currentClient: { x: event.clientX, y: event.clientY },
        originFlow: flowPosition,
        currentFlow: flowPosition,
      });
    },
    [draftConnection, getFlowPositionFromScreen, isMarqueeSelectionTarget, junctionDrag, pendingPlacement, waypointDrag],
  );

  useEffect(() => {
    if (!marqueeSelection) {
      return;
    }

    const updateSelectionPointer = (clientPosition: GraphPosition) => {
      const flowPosition = getFlowPositionFromScreen(clientPosition);
      if (!flowPosition) {
        return;
      }
      setMarqueeSelection((current) =>
        current
          ? {
              ...current,
              currentClient: clientPosition,
              currentFlow: flowPosition,
            }
          : current,
      );
    };

    const handleMouseMove = (event: MouseEvent) => {
      updateSelectionPointer({ x: event.clientX, y: event.clientY });
    };

    const handleMouseUp = (event: MouseEvent) => {
      const flowPosition = getFlowPositionFromScreen({ x: event.clientX, y: event.clientY });
      const finalSelection = flowPosition
        ? {
            ...marqueeSelection,
            currentClient: { x: event.clientX, y: event.clientY },
            currentFlow: flowPosition,
          }
        : marqueeSelection;
      const flowRect = normalizeSelectionRect(finalSelection.originFlow, finalSelection.currentFlow);
      setMarqueeSelection(null);
      suppressNextPaneClickRef.current = true;
      requestAnimationFrame(() => {
        suppressNextPaneClickRef.current = false;
      });
      if (selectionRectWidth(flowRect) < 4 && selectionRectHeight(flowRect) < 4) {
        return;
      }
      setCanvasSelection(getSelectionForRect(flowRect));
    };

    const handleWindowBlur = () => {
      setMarqueeSelection(null);
    };

    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleMouseUp);
    window.addEventListener("blur", handleWindowBlur);
    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
      window.removeEventListener("blur", handleWindowBlur);
    };
  }, [getFlowPositionFromScreen, getSelectionForRect, marqueeSelection, setCanvasSelection]);

  const findDraftConnectionSnapTarget = useCallback(
    (sourceNodeId: string, sourceHandleId: string | null, pointerPosition: GraphPosition): DraftConnectionSnapTarget | null => {
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
        if (
          isApiModelNode(sourceNode) &&
          sourceHandleId === API_TOOL_CALL_HANDLE_ID &&
          node.category !== "tool" &&
          !(node.category === "data" && node.provider_id === "core.data_display")
        ) {
          return;
        }
        if (isApiModelNode(sourceNode) && sourceHandleId === API_FINAL_MESSAGE_HANDLE_ID && node.category === "tool") {
          return;
        }
        const targetHandleId =
          sourceHandleId === TOOL_CONTEXT_HANDLE_ID
            ? node.kind === "model"
              ? API_TOOL_CONTEXT_HANDLE_ID
              : null
            : null;
        if (
          sourceHandleId === TOOL_CONTEXT_HANDLE_ID &&
          targetHandleId === null
        ) {
          return;
        }
        const anchor = getTargetAnchorPosition(node.id, targetHandleId);
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
            handleId: targetHandleId,
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
      const targetAnchor = getTargetAnchorPosition(edge.target_id, edge.target_handle_id ?? null);
      if (!sourceAnchor || !targetAnchor) {
        return null;
      }
      const routePoints = resolveEdgeRoutePoints(sourceAnchor, targetAnchor, edge.waypoints ?? [], { endWithHorizontal: true });
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

  const removeEdgesAndPruneJunctions = useCallback(
    (baseGraph: GraphDefinition, edgeIds: string[]) => {
      if (edgeIds.length === 0) {
        return baseGraph;
      }
      const edgeIdSet = new Set(edgeIds);
      return pruneDisconnectedWireJunctions({
        ...baseGraph,
        edges: baseGraph.edges.filter((edge) => !edgeIdSet.has(edge.id)),
      });
    },
    [pruneDisconnectedWireJunctions],
  );

  const getEffectiveSourceHandleId = useCallback(
    (sourceNode: GraphNode | undefined, sourceHandleId: string | null) => {
      if (isMcpContextProviderNode(sourceNode)) {
        return TOOL_CONTEXT_HANDLE_ID;
      }
      if (isApiModelNode(sourceNode)) {
        if (sourceHandleId === API_FINAL_MESSAGE_HANDLE_ID) {
          return API_FINAL_MESSAGE_HANDLE_ID;
        }
        if (sourceHandleId === API_TOOL_CALL_HANDLE_ID) {
          return API_TOOL_CALL_HANDLE_ID;
        }
        return sourceHandleId ?? null;
      }
      if (sourceNode?.kind === "mcp_tool_executor" && sourceHandleId === MCP_TERMINAL_OUTPUT_HANDLE_ID) {
        return MCP_TERMINAL_OUTPUT_HANDLE_ID;
      }
      if (!isRoutableToolNode(sourceNode)) {
        return sourceHandleId ?? null;
      }
      if (sourceHandleId === TOOL_FAILURE_HANDLE_ID) {
        return TOOL_FAILURE_HANDLE_ID;
      }
      return TOOL_SUCCESS_HANDLE_ID;
    },
    [],
  );

  const getEffectiveTargetHandleId = useCallback(
    (targetNode: GraphNode | undefined, targetHandleId: string | null) =>
      targetNode?.kind === "model" && targetHandleId === API_TOOL_CONTEXT_HANDLE_ID
        ? API_TOOL_CONTEXT_HANDLE_ID
        : (targetHandleId ?? null),
    [],
  );

  const isToolContextBindingConnection = useCallback(
    (
      sourceNode: GraphNode | undefined,
      targetNode: GraphNode | undefined,
      sourceHandleId: string | null,
      targetHandleId: string | null,
    ) =>
      isMcpContextProviderNode(sourceNode) &&
      sourceHandleId === TOOL_CONTEXT_HANDLE_ID &&
      targetNode?.kind === "model" &&
      targetHandleId === API_TOOL_CONTEXT_HANDLE_ID,
    [],
  );

  const isPromptBlockBindingConnection = useCallback(
    (sourceNode: GraphNode | undefined, targetNode: GraphNode | undefined) =>
      isPromptBlockNode(sourceNode) && (targetNode?.kind === "model" || targetNode?.kind === "data"),
    [],
  );

  const isDisplayContextBuilderBindingConnection = useCallback(
    (sourceNode: GraphNode | undefined, targetNode: GraphNode | undefined) =>
      sourceNode?.provider_id === "core.data_display" && targetNode?.provider_id === "core.context_builder",
    [],
  );

  const getConnectionConflictState = useCallback(
    (
      baseGraph: GraphDefinition,
      sourceId: string,
      targetId: string,
      sourceHandleId: string | null = null,
      targetHandleId: string | null = null,
    ) => {
      const sourceNode = baseGraph.nodes.find((node) => node.id === sourceId);
      const targetNode = baseGraph.nodes.find((node) => node.id === targetId);
      const effectiveSourceHandleId = getEffectiveSourceHandleId(sourceNode, sourceHandleId);
      const effectiveTargetHandleId = getEffectiveTargetHandleId(targetNode, targetHandleId);
      const isBindingConnection =
        isToolContextBindingConnection(sourceNode, targetNode, effectiveSourceHandleId, effectiveTargetHandleId) ||
        isPromptBlockBindingConnection(sourceNode, targetNode) ||
        isDisplayContextBuilderBindingConnection(sourceNode, targetNode);
      const duplicateEdgeId =
        baseGraph.edges.find((edge) => {
          if (edge.source_id !== sourceId || edge.target_id !== targetId) {
            return false;
          }
          const existingSourceHandleId =
            isRoutableToolNode(sourceNode) || isMcpContextProviderNode(sourceNode)
              ? inferToolEdgeSourceHandle(edge, sourceNode)
              : (edge.source_handle_id ?? null);
          return existingSourceHandleId === effectiveSourceHandleId && (edge.target_handle_id ?? null) === effectiveTargetHandleId;
        })?.id ?? null;
      if (isBindingConnection) {
        return { effectiveSourceHandleId, effectiveTargetHandleId, duplicateEdgeId, conflictingEdgeIds: [] as string[] };
      }
      const allowParallelApiFanOut =
        isApiModelNode(sourceNode) &&
        (effectiveSourceHandleId === API_TOOL_CALL_HANDLE_ID || effectiveSourceHandleId === API_FINAL_MESSAGE_HANDLE_ID);
      if (allowParallelApiFanOut) {
        return { effectiveSourceHandleId, effectiveTargetHandleId, duplicateEdgeId, conflictingEdgeIds: [] as string[] };
      }
      const conflictingEdgeIds = Array.from(
        new Set(
          baseGraph.edges
            .filter((edge) => {
              if (edge.id === duplicateEdgeId) {
                return false;
              }
              const existingSourceHandleId =
                isRoutableToolNode(sourceNode) || isMcpContextProviderNode(sourceNode)
                  ? inferToolEdgeSourceHandle(edge, sourceNode)
                  : (edge.source_handle_id ?? null);
              return (
                (edge.source_id === sourceId && existingSourceHandleId === effectiveSourceHandleId) || edge.target_id === targetId
              );
            })
            .map((edge) => edge.id),
        ),
      );
      return { effectiveSourceHandleId, effectiveTargetHandleId, duplicateEdgeId, conflictingEdgeIds };
    },
    [
      getEffectiveSourceHandleId,
      getEffectiveTargetHandleId,
      isDisplayContextBuilderBindingConnection,
      isPromptBlockBindingConnection,
      isToolContextBindingConnection,
    ],
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

      const snapTarget = findDraftConnectionSnapTarget(
        connectionOverride.sourceNodeId,
        connectionOverride.sourceHandleId,
        pointerPosition,
      );
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
      return findDraftConnectionSnapTarget(sourceNodeId, draftConnectionRef.current?.sourceHandleId ?? null, pointerPosition)?.nodeId ?? null;
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
      targetHandleId: string | null = null,
    ): GraphEdge | null => {
      const sourceNode = baseGraph.nodes.find((node) => node.id === sourceId);
      const targetNode = baseGraph.nodes.find((node) => node.id === targetId);
      if (!canConnectNodes(sourceNode, targetNode, catalog)) {
        setEditorMessage("That node connection is not allowed by the category contract matrix.");
        return null;
      }
      if (isPromptBlockNode(targetNode)) {
        setEditorMessage("Prompt Block nodes are source-only and cannot receive incoming connections.");
        return null;
      }
      const { effectiveSourceHandleId, effectiveTargetHandleId, duplicateEdgeId, conflictingEdgeIds } = getConnectionConflictState(
        baseGraph,
        sourceId,
        targetId,
        sourceHandleId,
        targetHandleId,
      );
      if (duplicateEdgeId) {
        setEditorMessage("Those handles are already connected.");
        return null;
      }
      if (
        (isRoutableToolNode(sourceNode) || isMcpContextProviderNode(sourceNode)) &&
        (effectiveSourceHandleId === TOOL_CONTEXT_HANDLE_ID || effectiveTargetHandleId === API_TOOL_CONTEXT_HANDLE_ID) &&
        !isToolContextBindingConnection(sourceNode, targetNode, effectiveSourceHandleId, effectiveTargetHandleId)
      ) {
        setEditorMessage("Tool context wires can only connect from a tool context port into an API node.");
        return null;
      }
      if (isMcpContextProviderNode(sourceNode) && !isToolContextBindingConnection(sourceNode, targetNode, effectiveSourceHandleId, effectiveTargetHandleId)) {
        setEditorMessage("MCP context providers can only create tool-context bindings into API nodes.");
        return null;
      }
      if (isPromptBlockNode(sourceNode) && !isPromptBlockBindingConnection(sourceNode, targetNode)) {
        setEditorMessage("Prompt Block nodes can only create binding connections into model or data nodes.");
        return null;
      }
      const nextEdgeId = `edge-${sourceId}-${targetId}-${Date.now()}`;
      if (sourceNode?.provider_id === "core.data_display" && targetNode?.provider_id === "core.context_builder") {
        return {
          id: nextEdgeId,
          source_id: sourceId,
          target_id: targetId,
          source_handle_id: effectiveSourceHandleId,
          target_handle_id: effectiveTargetHandleId,
          label: "display input",
          kind: "binding",
          priority: 0,
          waypoints,
          condition: null,
        };
      }
      const remainingGraph = removeEdgesAndPruneJunctions(baseGraph, conflictingEdgeIds);
      const sourceEdges = remainingGraph.edges.filter((edge) => edge.source_id === sourceId && edge.kind !== "binding");
      const hasStandardOutgoing = sourceEdges.some((edge) => edge.kind === "standard");
      if (isToolContextBindingConnection(sourceNode, targetNode, effectiveSourceHandleId, effectiveTargetHandleId)) {
        return {
          id: nextEdgeId,
          source_id: sourceId,
          target_id: targetId,
          source_handle_id: effectiveSourceHandleId,
          target_handle_id: effectiveTargetHandleId,
          label: "tool context",
          kind: "binding",
          priority: 0,
          waypoints,
          condition: null,
        };
      }
      if (isPromptBlockBindingConnection(sourceNode, targetNode)) {
        return {
          id: nextEdgeId,
          source_id: sourceId,
          target_id: targetId,
          source_handle_id: effectiveSourceHandleId,
          target_handle_id: effectiveTargetHandleId,
          label: "prompt block",
          kind: "binding",
          priority: 0,
          waypoints,
          condition: null,
        };
      }
      if (isRoutableToolNode(sourceNode)) {
        if (sourceNode?.kind === "mcp_tool_executor" && effectiveSourceHandleId === MCP_TERMINAL_OUTPUT_HANDLE_ID) {
          const hasTerminalOutgoing = sourceEdges.some(
            (edge) => inferToolEdgeSourceHandle(edge, sourceNode) === MCP_TERMINAL_OUTPUT_HANDLE_ID,
          );
          if (hasTerminalOutgoing) {
            setEditorMessage("MCP Tool Executor already has a terminal output route.");
            return null;
          }
          return {
            id: nextEdgeId,
            source_id: sourceId,
            target_id: targetId,
            source_handle_id: MCP_TERMINAL_OUTPUT_HANDLE_ID,
            target_handle_id: effectiveTargetHandleId,
            label: "terminal output",
            kind: "conditional",
            priority: 30,
            waypoints,
            condition: defaultMcpTerminalOutputCondition(nextEdgeId),
          };
        }
        const effectiveHandleId = effectiveSourceHandleId === TOOL_FAILURE_HANDLE_ID ? TOOL_FAILURE_HANDLE_ID : TOOL_SUCCESS_HANDLE_ID;
        const hasSuccessOutgoing = sourceEdges.some(
          (edge) => inferToolEdgeSourceHandle(edge, sourceNode) === TOOL_SUCCESS_HANDLE_ID,
        );
        const hasFailureOutgoing = sourceEdges.some(
          (edge) => inferToolEdgeSourceHandle(edge, sourceNode) === TOOL_FAILURE_HANDLE_ID,
        );
        if (effectiveHandleId === TOOL_SUCCESS_HANDLE_ID && hasSuccessOutgoing) {
          setEditorMessage(
            sourceNode?.kind === "mcp_tool_executor"
              ? "MCP Tool Executor already has an on finish route."
              : "Run Tool already has an on success route.",
          );
          return null;
        }
        if (effectiveHandleId === TOOL_FAILURE_HANDLE_ID && hasFailureOutgoing) {
          setEditorMessage(
            sourceNode?.kind === "mcp_tool_executor"
              ? "MCP Tool Executor already has an on failure route."
              : "Run Tool already has an on failure route.",
          );
          return null;
        }
        return {
          id: nextEdgeId,
          source_id: sourceId,
          target_id: targetId,
          source_handle_id: effectiveHandleId,
          target_handle_id: effectiveTargetHandleId,
          label:
            effectiveHandleId === TOOL_FAILURE_HANDLE_ID
              ? "on failure"
              : sourceNode?.kind === "mcp_tool_executor"
                ? "on finish"
                : "on success",
          kind: effectiveHandleId === TOOL_FAILURE_HANDLE_ID ? "conditional" : "standard",
          priority: effectiveHandleId === TOOL_FAILURE_HANDLE_ID ? 10 : 100,
          waypoints,
          condition: effectiveHandleId === TOOL_FAILURE_HANDLE_ID ? defaultToolFailureCondition(nextEdgeId) : null,
        };
      }
      if (isApiModelNode(sourceNode) && isApiOutputHandleId(effectiveSourceHandleId)) {
        if (effectiveSourceHandleId === API_TOOL_CALL_HANDLE_ID) {
          if (
            targetNode?.category !== "tool" &&
            !(targetNode?.category === "data" && targetNode.provider_id === "core.data_display")
          ) {
            setEditorMessage("API tool-call output can only connect to tool nodes or Envelope Display nodes.");
            return null;
          }
          return {
            id: nextEdgeId,
            source_id: sourceId,
            target_id: targetId,
            source_handle_id: API_TOOL_CALL_HANDLE_ID,
            target_handle_id: effectiveTargetHandleId,
            label: "tool call",
            kind: "conditional",
            priority: 10,
            waypoints,
            condition: defaultApiToolCallCondition(nextEdgeId),
          };
        }
        if (targetNode?.category === "tool") {
          setEditorMessage("API message output cannot connect directly to tool nodes.");
          return null;
        }
        return {
          id: nextEdgeId,
          source_id: sourceId,
          target_id: targetId,
          source_handle_id: API_FINAL_MESSAGE_HANDLE_ID,
          target_handle_id: effectiveTargetHandleId,
          label: "message",
          kind: "conditional",
          priority: 20,
          waypoints,
          condition: defaultApiMessageCondition(nextEdgeId),
        };
      }
      return {
        id: nextEdgeId,
        source_id: sourceId,
        target_id: targetId,
          source_handle_id: effectiveSourceHandleId,
        target_handle_id: effectiveTargetHandleId,
        label: hasStandardOutgoing ? "conditional route" : "next",
        kind: hasStandardOutgoing ? "conditional" : "standard",
        priority: hasStandardOutgoing ? 10 : 100,
        waypoints,
        condition: hasStandardOutgoing ? defaultConditionalCondition(nextEdgeId) : null,
      };
    },
    [
      catalog,
      getConnectionConflictState,
      isDisplayContextBuilderBindingConnection,
      isPromptBlockBindingConnection,
      isToolContextBindingConnection,
      removeEdgesAndPruneJunctions,
    ],
  );

  const commitEdge = useCallback(
    (
      sourceId: string,
      targetId: string,
      waypoints: GraphPosition[] = [],
      baseGraph: GraphDefinition | null = graph,
      sourceHandleId: string | null = null,
      targetHandleId: string | null = null,
    ) => {
      if (!baseGraph) {
        return false;
      }
      const { conflictingEdgeIds } = getConnectionConflictState(baseGraph, sourceId, targetId, sourceHandleId, targetHandleId);
      const nextBaseGraph = removeEdgesAndPruneJunctions(baseGraph, conflictingEdgeIds);
      const nextEdge = buildCommittedEdge(baseGraph, sourceId, targetId, waypoints, sourceHandleId, targetHandleId);
      if (!nextEdge) {
        return false;
      }
      onGraphChange(rebalanceOutgoingTargets({
        ...nextBaseGraph,
        edges: [...nextBaseGraph.edges, nextEdge],
      }, sourceId));
      setEditorMessage(null);
      return true;
    },
    [buildCommittedEdge, getConnectionConflictState, graph, onGraphChange, rebalanceOutgoingTargets, removeEdgesAndPruneJunctions],
  );

  const handleNodeHandlePointerDown = useCallback(
    (_nodeId: string, _handleType: HandleClickType, _handleId: string | null) => {
      // Let React Flow own the default handle press gesture.
      // Existing-edge rerouting should not hijack a normal drag-to-connect action.
      return false;
    },
    [],
  );

  const handleJunctionPointerDown = useCallback(
    (nodeId: string, clientPosition: { x: number; y: number }) => {
      if (!graph || pendingPlacement || draftConnectionRef.current) {
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
    [getFlowPositionFromScreen, graph, onSelectionChange, pendingPlacement],
  );

  const handleWaypointPointerDown = useCallback(
    (edgeId: string, waypointIndex: number, clientPosition: { x: number; y: number }) => {
      if (!graph || pendingPlacement || draftConnectionRef.current) {
        return;
      }
      const edge = graph.edges.find((candidate) => candidate.id === edgeId);
      if (!edge) {
        return;
      }
      const sourceNode = graph.nodes.find((node) => node.id === edge.source_id);
      const targetNode = graph.nodes.find((node) => node.id === edge.target_id);
      if (!sourceNode || !targetNode) {
        return;
      }
      const sourceHandleId = inferToolEdgeSourceHandle(edge, sourceNode);
      const sourceAnchor = getSourceAnchorPosition(edge.source_id, sourceHandleId);
      const targetAnchor = getTargetAnchorPosition(edge.target_id, edge.target_handle_id ?? null);
      if (!sourceAnchor || !targetAnchor) {
        return;
      }
      const routePoints = resolveEdgeRoutePoints(sourceAnchor, targetAnchor, edge.waypoints ?? [], { endWithHorizontal: true });
      const waypoint = routePoints[waypointIndex + 1];
      if (!waypoint) {
        return;
      }
      const flowPosition = getFlowPositionFromScreen(clientPosition);
      if (!flowPosition) {
        return;
      }
      suppressNextPaneClickRef.current = true;
      setTooltipNodeId(null);
      onSelectionChange(null, edgeId);
      setWaypointDrag({
        edgeId,
        waypointIndex,
        pointerOffset: {
          x: flowPosition.x - waypoint.x,
          y: flowPosition.y - waypoint.y,
        },
      });
    },
    [getFlowPositionFromScreen, getSourceAnchorPosition, getTargetAnchorPosition, graph, onSelectionChange, pendingPlacement],
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
        null,
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

  const scheduleDragRender = useCallback(() => {
    if (nodeDragRafRef.current !== null) return;
    nodeDragRafRef.current = requestAnimationFrame(() => {
      nodeDragRafRef.current = null;
      if (dragDiagnosticsEnabled && dragDiagnosticSessionRef.current.active) {
        dragDiagnosticSessionRef.current.rafTicks += 1;
      }
      setDragRenderTick((t) => t + 1);
    });
  }, [dragDiagnosticsEnabled]);

  const cancelPendingNodeDragFrame = useCallback(() => {
    if (nodeDragRafRef.current !== null) {
      cancelAnimationFrame(nodeDragRafRef.current);
      nodeDragRafRef.current = null;
    }
  }, []);

  useEffect(() => cancelPendingDragFrame, [cancelPendingDragFrame]);
  useEffect(() => cancelPendingNodeDragFrame, [cancelPendingNodeDragFrame]);
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
  const updateEdgeWaypointInGraph = useCallback(
    (baseGraph: GraphDefinition, waypointEdgeId: string, waypointIndex: number, position: GraphPosition) => {
      let didChange = false;
      const nextEdges = baseGraph.edges.map((edge) => {
        if (edge.id !== waypointEdgeId) {
          return edge;
        }
        const sourceNode = baseGraph.nodes.find((node) => node.id === edge.source_id);
        const targetNode = baseGraph.nodes.find((node) => node.id === edge.target_id);
        if (!sourceNode || !targetNode) {
          return edge;
        }
        const sourceHandleId = inferToolEdgeSourceHandle(edge, sourceNode);
        const sourceDimensions = getNodeDimensions(sourceNode);
        const targetDimensions = getNodeDimensions(targetNode);
        const verticalRatio = getToolSourceHandleAnchorRatio(sourceHandleId);
        const sourceAnchor = {
          x: sourceNode.position.x + sourceDimensions.width,
          y: sourceNode.position.y + sourceDimensions.height * verticalRatio,
        };
        const targetVerticalRatio =
          targetNode.kind === "model" ? getApiToolContextTargetAnchorRatio(edge.target_handle_id ?? null) : 0.5;
        const targetAnchor = {
          x: targetNode.position.x,
          y: targetNode.position.y + targetDimensions.height * targetVerticalRatio,
        };
        const currentRoutePoints = resolveEdgeRoutePoints(sourceAnchor, targetAnchor, edge.waypoints ?? [], {
          endWithHorizontal: true,
        });
        const controlPointIndex = waypointIndex + 1;
        const currentWaypoint = currentRoutePoints[controlPointIndex];
        if (!currentWaypoint) {
          return edge;
        }
        if (Math.abs(currentWaypoint.x - position.x) < 0.01 && Math.abs(currentWaypoint.y - position.y) < 0.01) {
          return edge;
        }
        const nextRoutePoints = currentRoutePoints.map((waypoint, index) => (index === controlPointIndex ? position : waypoint));
        didChange = true;
        return {
          ...edge,
          waypoints: nextRoutePoints.slice(1, -1),
        };
      });
      return didChange ? { ...baseGraph, edges: nextEdges } : baseGraph;
    },
    [getNodeDimensions],
  );
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
    if (!waypointDrag) {
      return;
    }

    const updateDraggedWaypoint = (clientPosition: { x: number; y: number }, commit: boolean) => {
      const currentGraph = graphRef.current;
      if (!currentGraph) {
        return;
      }
      const flowPosition = getFlowPositionFromScreen(clientPosition);
      if (!flowPosition) {
        return;
      }
      const nextGraph = updateEdgeWaypointInGraph(currentGraph, waypointDrag.edgeId, waypointDrag.waypointIndex, {
        x: flowPosition.x - waypointDrag.pointerOffset.x,
        y: flowPosition.y - waypointDrag.pointerOffset.y,
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
      updateDraggedWaypoint({ x: event.clientX, y: event.clientY }, false);
    };

    const handleMouseUp = (event: MouseEvent) => {
      updateDraggedWaypoint({ x: event.clientX, y: event.clientY }, true);
      setWaypointDrag(null);
    };

    const handleWindowBlur = () => {
      cancelPendingDragFrame();
      setWaypointDrag(null);
    };

    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleMouseUp);
    window.addEventListener("blur", handleWindowBlur);
    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
      window.removeEventListener("blur", handleWindowBlur);
    };
  }, [cancelPendingDragFrame, getFlowPositionFromScreen, onGraphChange, scheduleDragGraphUpdate, updateEdgeWaypointInGraph, waypointDrag]);

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
      setPromptBlockDetailsNodeId(null);
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
      setPromptBlockDetailsNodeId(null);
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
      setPromptBlockDetailsNodeId(null);
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
      setPromptBlockDetailsNodeId(null);
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

  const copySelectedNodeToClipboard = useCallback(() => {
    if (!graph) {
      setEditorMessage("Select nodes to copy.");
      return;
    }

    const selectedNodeIdSet = new Set(selectedNodeIds);
    selectedEdgeIds.forEach((edgeId) => {
      const edge = graph.edges.find((candidate) => candidate.id === edgeId);
      if (!edge) {
        return;
      }
      selectedNodeIdSet.add(edge.source_id);
      selectedNodeIdSet.add(edge.target_id);
    });

    const copiedNodes = graph.nodes.filter((node) => selectedNodeIdSet.has(node.id));
    if (copiedNodes.length === 0) {
      setEditorMessage("Select nodes to copy.");
      return;
    }

    const copiedNodeIdSet = new Set(copiedNodes.map((node) => node.id));
    const selectedEdgeIdSet = new Set(selectedEdgeIds);
    const copiedEdges = graph.edges.filter(
      (edge) =>
        copiedNodeIdSet.has(edge.source_id) &&
        copiedNodeIdSet.has(edge.target_id) &&
        (selectedEdgeIdSet.size === 0 || selectedEdgeIdSet.has(edge.id) || (selectedNodeIdSet.has(edge.source_id) && selectedNodeIdSet.has(edge.target_id))),
    );

    nodeClipboardRef.current = {
      nodes: cloneClipboardNodes(copiedNodes),
      edges: cloneClipboardEdges(copiedEdges),
      pasteCount: 0,
    };
    if (copiedNodes.length === 1 && copiedEdges.length === 0) {
      setEditorMessage(`Copied ${copiedNodes[0].label}.`);
      return;
    }
    setEditorMessage(
      `Copied ${copiedNodes.length} node${copiedNodes.length === 1 ? "" : "s"} and ${copiedEdges.length} edge${copiedEdges.length === 1 ? "" : "s"}.`,
    );
  }, [graph, selectedEdgeIds, selectedNodeIds]);

  const pasteNodeFromClipboard = useCallback(() => {
    if (!graph) {
      return;
    }
    if (pendingPlacement || draftConnection) {
      setEditorMessage("Finish the current placement before pasting.");
      return;
    }
    const clipboard = nodeClipboardRef.current;
    if (!clipboard) {
      setEditorMessage("Copy a node first.");
      return;
    }
    const pasteCount = clipboard.pasteCount + 1;
    let nextGraph = graph;
    const nodeIdMap = new Map<string, string>();
    const pastedNodeIds: string[] = [];
    const pastedEdgeIds: string[] = [];

    clipboard.nodes.forEach((clipboardNode) => {
      const nextNode = duplicateGraphNode(nextGraph, clipboardNode, {
        position: {
          x: clipboardNode.position.x + NODE_CLIPBOARD_PASTE_OFFSET * pasteCount,
          y: clipboardNode.position.y + NODE_CLIPBOARD_PASTE_OFFSET * pasteCount,
        },
      });
      nodeIdMap.set(clipboardNode.id, nextNode.id);
      pastedNodeIds.push(nextNode.id);
      nextGraph = {
        ...nextGraph,
        start_node_id: !nextGraph.start_node_id && nextNode.category === "start" ? nextNode.id : nextGraph.start_node_id,
        nodes: [...nextGraph.nodes, nextNode],
      };
    });

    clipboard.edges.forEach((clipboardEdge, index) => {
      const nextSourceId = nodeIdMap.get(clipboardEdge.source_id);
      const nextTargetId = nodeIdMap.get(clipboardEdge.target_id);
      if (!nextSourceId || !nextTargetId) {
        return;
      }
      const nextEdgeId = `edge-${nextSourceId}-${nextTargetId}-${Date.now()}-${index}`;
      const nextEdge: GraphEdge = {
        ...cloneClipboardEdges([clipboardEdge])[0],
        id: nextEdgeId,
        source_id: nextSourceId,
        target_id: nextTargetId,
        waypoints: clipboardEdge.waypoints?.map((waypoint) => ({
          x: waypoint.x + NODE_CLIPBOARD_PASTE_OFFSET * pasteCount,
          y: waypoint.y + NODE_CLIPBOARD_PASTE_OFFSET * pasteCount,
        })),
        condition: clipboardEdge.condition
          ? {
              ...clipboardEdge.condition,
              id: `${nextEdgeId}-condition`,
            }
          : null,
      };
      pastedEdgeIds.push(nextEdgeId);
      nextGraph = {
        ...nextGraph,
        edges: [...nextGraph.edges, nextEdge],
      };
    });

    nodeClipboardRef.current = {
      ...clipboard,
      pasteCount,
    };
    onGraphChange(nextGraph);
    setCanvasSelection({ nodeIds: pastedNodeIds, edgeIds: pastedEdgeIds });
    setTooltipNodeId(null);
    setToolDetailsNodeId(null);
    setProviderDetailsNodeId(null);
    setPromptBlockDetailsNodeId(null);
    if (pastedNodeIds.length === 1 && pastedEdgeIds.length === 0) {
      const pastedNode = nextGraph.nodes.find((node) => node.id === pastedNodeIds[0]);
      setEditorMessage(`Pasted ${pastedNode?.label ?? "node"}.`);
      return;
    }
    setEditorMessage(
      `Pasted ${pastedNodeIds.length} node${pastedNodeIds.length === 1 ? "" : "s"} and ${pastedEdgeIds.length} edge${pastedEdgeIds.length === 1 ? "" : "s"}.`,
    );
  }, [draftConnection, graph, onGraphChange, pendingPlacement, setCanvasSelection]);

  const handleDeleteSavedNode = useCallback(
    (id: string) => {
      deleteSavedNode(id);
      setSavedNodes(getSavedNodes());
      setEditorMessage("Removed saved node from library.");
    },
    [],
  );

  const deleteSelectedElements = useCallback(() => {
    if (!graph || (selectedNodeIds.length === 0 && selectedEdgeIds.length === 0)) {
      return false;
    }

    const selectedNodeIdSet = new Set(selectedNodeIds);
    const connectedEdgeIds = graph.edges
      .filter((edge) => selectedNodeIdSet.has(edge.source_id) || selectedNodeIdSet.has(edge.target_id))
      .map((edge) => edge.id);
    const edgeIdsToRemove = [...new Set([...selectedEdgeIds, ...connectedEdgeIds])];

    let nextGraph: GraphDefinition = {
      ...graph,
      nodes: graph.nodes.filter((node) => !selectedNodeIdSet.has(node.id)),
      edges: graph.edges.filter(
        (edge) =>
          !selectedNodeIdSet.has(edge.source_id) &&
          !selectedNodeIdSet.has(edge.target_id) &&
          !edgeIdsToRemove.includes(edge.id),
      ),
      start_node_id: selectedNodeIdSet.has(graph.start_node_id) ? "" : graph.start_node_id,
    };
    nextGraph = pruneDisconnectedWireJunctions(nextGraph);
    nextGraph = removeEdgesAndPruneJunctions(nextGraph, edgeIdsToRemove);

    clearCanvasSelection();
    onGraphChange(nextGraph);
    setEditorMessage(
      `Deleted ${selectedNodeIds.length} node${selectedNodeIds.length === 1 ? "" : "s"} and ${edgeIdsToRemove.length} edge${edgeIdsToRemove.length === 1 ? "" : "s"}.`,
    );
    return true;
  }, [clearCanvasSelection, graph, onGraphChange, pruneDisconnectedWireJunctions, removeEdgesAndPruneJunctions, selectedEdgeIds, selectedNodeIds]);

  const handleToggleHotbarFavorite = useCallback(
    (provider: NodeProviderDefinition) => {
      if (hotbarFavorites[provider.category] === provider.provider_id) {
        setHotbarFavorites(clearHotbarFavorite(provider.category));
        setEditorMessage(`Removed ${provider.display_name} from the ${provider.category} hotbar.`);
        return;
      }
      setHotbarFavorites(setHotbarFavorite(provider.category, provider.provider_id));
      setEditorMessage(`Pinned ${provider.display_name} to the ${provider.category} hotbar.`);
    },
    [hotbarFavorites],
  );

  const quickAddItems = useMemo(() => {
    const providers = (catalog?.node_providers ?? []).filter((provider) => provider.category !== "provider");
    return QUICK_ADD_SLOTS.map((slot) => ({
      ...slot,
      provider: resolveQuickAddProvider(providers, slot, hotbarFavorites[slot.category]),
      isFavorite: hotbarFavorites[slot.category] !== undefined,
    }));
  }, [catalog, hotbarFavorites]);

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

  const marqueeSelectionStyle = useMemo<CSSProperties | null>(() => {
    if (!marqueeSelection) {
      return null;
    }
    const canvasBounds = canvasRef.current?.getBoundingClientRect();
    if (!canvasBounds) {
      return null;
    }
    const clientRect = normalizeSelectionRect(marqueeSelection.originClient, marqueeSelection.currentClient);
    const left = Math.max(0, clientRect.minX - canvasBounds.left);
    const top = Math.max(0, clientRect.minY - canvasBounds.top);
    const right = Math.min(canvasBounds.width, clientRect.maxX - canvasBounds.left);
    const bottom = Math.min(canvasBounds.height, clientRect.maxY - canvasBounds.top);
    return {
      left,
      top,
      width: Math.max(0, right - left),
      height: Math.max(0, bottom - top),
    };
  }, [marqueeSelection]);

  const draftConnectionToneClass = useMemo(() => {
    if (!draftConnection || !graph) {
      return "";
    }
    const sourceNode = graph.nodes.find((node) => node.id === draftConnection.sourceNodeId);
    if (!isRoutableToolNode(sourceNode) && !isMcpContextProviderNode(sourceNode) && !isApiModelNode(sourceNode)) {
      return "";
    }
    if (draftConnection.sourceHandleId === TOOL_CONTEXT_HANDLE_ID) {
      return " graph-draft-wire-path--tool-context";
    }
    if (draftConnection.sourceHandleId === API_TOOL_CALL_HANDLE_ID) {
      return " graph-draft-wire-path--api-tool-call";
    }
    if (draftConnection.sourceHandleId === API_FINAL_MESSAGE_HANDLE_ID) {
      return " graph-draft-wire-path--api-message";
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

      if ((event.key === "Backspace" || event.key === "Delete") && !pendingPlacement && !draftConnection) {
        if (deleteSelectedElements()) {
          event.preventDefault();
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
  }, [beginProviderPlacement, cancelDraftConnection, cancelPlacement, clearCanvasChrome, deleteSelectedElements, draftConnection, handleFitView, isEditableTarget, openDrawerTab, pendingPlacement, quickAddItems]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (isEditableTarget(event.target) || hasSelectedText()) {
        return;
      }
      if (!(event.metaKey || event.ctrlKey) || event.altKey) {
        return;
      }
      const key = event.key.toLowerCase();
      if (key === "c") {
        event.preventDefault();
        copySelectedNodeToClipboard();
        return;
      }
      if (key === "v") {
        event.preventDefault();
        pasteNodeFromClipboard();
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [copySelectedNodeToClipboard, hasSelectedText, isEditableTarget, pasteNodeFromClipboard]);

  const nodes = useMemo<FlowNode[]>(() => {
    if (!graph) {
      nodeDataCacheRef.current = new Map();
      flowNodeCacheRef.current = new Map();
      measuredNodeDimensionsRef.current.clear();
      return [];
    }
    const diagnosticsStart = dragDiagnosticsEnabled && isNodeDragActive ? performance.now() : null;
    if (isNodeDragActive && flowNodeCacheRef.current.size === graph.nodes.length) {
      const dragPositions = dragPositionMapRef.current;
      const nextFlowNodeCache = new Map<string, FlowNode<GraphCanvasRuntimeNodeData>>();
      const nextNodes = graph.nodes.map((node): FlowNode<GraphCanvasRuntimeNodeData> => {
        const cachedNode = flowNodeCacheRef.current.get(node.id);
        const nextPosition = dragPositions?.get(node.id) ?? node.position;
        const nextSelected = selectedNodeIdSet.has(node.id);
        const nextStyle = isWireJunctionNode(node) ? JUNCTION_NODE_STYLE : NODE_STYLE;
        const nextDimensions = getFlowNodeDimensions(node, cachedNode);
        if (
          cachedNode &&
          cachedNode.selected === nextSelected &&
          cachedNode.style === nextStyle &&
          cachedNode.position.x === nextPosition.x &&
          cachedNode.position.y === nextPosition.y &&
          cachedNode.width === nextDimensions.width &&
          cachedNode.height === nextDimensions.height
        ) {
          nextFlowNodeCache.set(node.id, cachedNode);
          return cachedNode;
        }
        const nextFlowNode = cachedNode
          ? {
              ...cachedNode,
              position: nextPosition,
              selected: nextSelected,
              style: nextStyle,
              width: nextDimensions.width,
              height: nextDimensions.height,
            }
          : {
              id: node.id,
              type: "graphNode",
              position: nextPosition,
              selected: nextSelected,
              sourcePosition: "right" as Position,
              targetPosition: "left" as Position,
              data: nodeDataCacheRef.current.get(node.id) ?? {
                node,
                graph: null,
                catalog,
                runState,
                kindColor: KIND_COLORS[node.kind] ?? "#8486a5",
                status: "idle",
                preview: false,
                tooltipVisible: false,
                onToggleTooltip: handleToggleTooltip,
                onOpenToolDetails: handleOpenToolDetails,
                onOpenProviderDetails: handleOpenProviderDetails,
                onToggleExecutorRetries: handleToggleExecutorRetries,
                onOpenPromptBlockDetails: handleOpenPromptBlockDetails,
                onOpenDisplayResponse: handleOpenDisplayResponse,
                onOpenContextBuilderPayload: handleOpenContextBuilderPayload,
                onHandlePointerDown: handleNodeHandlePointerDown,
                onJunctionPointerDown: handleJunctionPointerDown,
              },
              style: nextStyle,
              width: nextDimensions.width,
              height: nextDimensions.height,
            };
        nextFlowNodeCache.set(node.id, nextFlowNode);
        return nextFlowNode;
      });
      flowNodeCacheRef.current = nextFlowNodeCache;
      if (diagnosticsStart !== null) {
        recordNodeBuildDiagnostic(performance.now() - diagnosticsStart);
      }
      return nextNodes;
    }
    const isRoutingDraftWire = draftConnection !== null && !isConnecting;
    const snapTargetNodeId = isRoutingDraftWire ? draftConnectionSnapTargetNodeId : null;
    const nextNodeDataCache = new Map<string, GraphCanvasRuntimeNodeData>();
    const nextFlowNodeCache = new Map<string, FlowNode<GraphCanvasRuntimeNodeData>>();
    const normalizedEvents = runProjection?.normalizedEvents ?? runState?.event_history ?? [];
    const nextNodes = graph.nodes.map((node): FlowNode<GraphCanvasRuntimeNodeData> => {
      const runtimeNodeState = runProjection?.nodeStates[node.id];
      const isActive = runtimeNodeState?.isActive ?? (runState?.current_node_id === node.id);
      const hasError = runtimeNodeState?.hasError ?? Object.prototype.hasOwnProperty.call(runState?.node_errors ?? {}, node.id);
      const wasVisited =
        runtimeNodeState?.wasVisited ??
        ((runState?.visit_counts?.[node.id] ?? 0) > 0 ||
          Object.prototype.hasOwnProperty.call(runState?.node_outputs ?? {}, node.id) ||
          hasError);
      const didLastRunFinish =
        runState?.status === "completed" ||
        runState?.status === "failed" ||
        runState?.status === "cancelled" ||
        runState?.status === "interrupted";
      const expectsExecutionStatus = participatesInExecutionRoute(graph, node);
      const isConnectionMagnetized = isRoutingDraftWire && snapTargetNodeId === node.id;
      const kindColor = KIND_COLORS[node.kind] ?? "#8486a5";
      const contextBuilderRuntime =
        node.provider_id === "core.context_builder"
          ? buildContextBuilderRuntimeView(graph, node, runState, normalizedEvents)
          : null;
      const contextBuilderRuntimeKey = contextBuilderRuntime
        ? [
            contextBuilderRuntime.fulfilledCount,
            contextBuilderRuntime.errorCount,
            contextBuilderRuntime.totalCount,
            contextBuilderRuntime.isWaitingToForward ? 1 : 0,
            contextBuilderRuntime.holdingOutgoing ? 1 : 0,
            contextBuilderRuntime.contextBuilderComplete === null ? "n" : contextBuilderRuntime.contextBuilderComplete ? "y" : "f",
            ...contextBuilderRuntime.sources.map((s) => `${s.sourceNodeId}:${s.status}`),
          ].join("|")
        : "";
      let status: GraphCanvasNodeData["status"] = hasError
        ? "failed"
        : isActive
          ? "active"
          : wasVisited
            ? "success"
            : didLastRunFinish && expectsExecutionStatus
              ? "unreached"
              : "idle";
      if (contextBuilderRuntime?.isWaitingToForward && !hasError) {
        status = "active";
      }
      const tooltipVisible = tooltipNodeId === node.id;
      const tooltipGraph = tooltipVisible ? graph : null;
      const previousData = nodeDataCacheRef.current.get(node.id);
      const nextData =
        previousData &&
        previousData.node === node &&
        previousData.graph === graph &&
        previousData.tooltipGraph === tooltipGraph &&
        previousData.catalog === catalog &&
        previousData.runState === runState &&
        previousData.kindColor === kindColor &&
        previousData.status === status &&
        previousData.runtimeOutput === runtimeNodeState?.latestOutput &&
        previousData.contextBuilderRuntimeKey === contextBuilderRuntimeKey &&
        previousData.isConnectionMagnetized === isConnectionMagnetized &&
        (previousData.preview === false || previousData.preview === undefined) &&
        previousData.tooltipVisible === tooltipVisible &&
        previousData.onToggleTooltip === handleToggleTooltip &&
        previousData.onOpenToolDetails === handleOpenToolDetails &&
        previousData.onOpenProviderDetails === handleOpenProviderDetails &&
        previousData.onToggleExecutorRetries === handleToggleExecutorRetries &&
        previousData.onOpenPromptBlockDetails === handleOpenPromptBlockDetails &&
        previousData.onOpenDisplayResponse === handleOpenDisplayResponse &&
        previousData.onOpenContextBuilderPayload === handleOpenContextBuilderPayload &&
        previousData.onHandlePointerDown === handleNodeHandlePointerDown &&
        previousData.onJunctionPointerDown === handleJunctionPointerDown
          ? previousData
          : {
              node,
              graph,
              tooltipGraph,
              catalog,
              runState,
              kindColor,
              status,
              runtimeOutput: runtimeNodeState?.latestOutput,
              contextBuilderRuntime,
              contextBuilderRuntimeKey,
              isConnectionMagnetized,
              preview: false,
              tooltipVisible,
              onToggleTooltip: handleToggleTooltip,
              onOpenToolDetails: handleOpenToolDetails,
              onOpenProviderDetails: handleOpenProviderDetails,
              onToggleExecutorRetries: handleToggleExecutorRetries,
              onOpenPromptBlockDetails: handleOpenPromptBlockDetails,
              onOpenDisplayResponse: handleOpenDisplayResponse,
              onOpenContextBuilderPayload: handleOpenContextBuilderPayload,
              onHandlePointerDown: handleNodeHandlePointerDown,
              onJunctionPointerDown: handleJunctionPointerDown,
            };
      nextNodeDataCache.set(node.id, nextData);
      const isJunction = isWireJunctionNode(node);
      const nextPosition = dragPositionMapRef.current?.get(node.id) ?? node.position;
      const nextSelected = selectedNodeIdSet.has(node.id);
      const nextStyle = isJunction ? JUNCTION_NODE_STYLE : NODE_STYLE;
      const previousFlowNode = flowNodeCacheRef.current.get(node.id);
      const nextDimensions = getFlowNodeDimensions(node, previousFlowNode);
      const nextFlowNode =
        previousFlowNode &&
        previousFlowNode.data === nextData &&
        previousFlowNode.selected === nextSelected &&
        previousFlowNode.style === nextStyle &&
        previousFlowNode.position.x === nextPosition.x &&
        previousFlowNode.position.y === nextPosition.y &&
        previousFlowNode.width === nextDimensions.width &&
        previousFlowNode.height === nextDimensions.height
          ? previousFlowNode
          : {
              id: node.id,
              type: "graphNode",
              position: nextPosition,
              selected: nextSelected,
              sourcePosition: "right" as Position,
              targetPosition: "left" as Position,
              data: nextData,
              style: nextStyle,
              width: nextDimensions.width,
              height: nextDimensions.height,
            };
      nextFlowNodeCache.set(node.id, nextFlowNode);
      return nextFlowNode;
    });
    nodeDataCacheRef.current = nextNodeDataCache;
    flowNodeCacheRef.current = nextFlowNodeCache;
    if (diagnosticsStart !== null) {
      recordNodeBuildDiagnostic(performance.now() - diagnosticsStart);
    }
    return nextNodes;
  }, [
    catalog,
    dragDiagnosticsEnabled,
    dragRenderTick,
    draftConnection?.sourceNodeId,
    draftConnectionSnapTargetNodeId,
    getFlowNodeDimensions,
    graph,
    handleJunctionPointerDown,
    handleNodeHandlePointerDown,
    handleOpenContextBuilderPayload,
    handleOpenDisplayResponse,
    handleOpenPromptBlockDetails,
    handleOpenProviderDetails,
    handleToggleExecutorRetries,
    handleOpenToolDetails,
    handleToggleTooltip,
    isConnecting,
    isNodeDragActive,
    recordNodeBuildDiagnostic,
    runProjection,
    runState,
    selectedNodeIdSet,
    tooltipNodeId,
  ]);

  const handleFormatGraph = useCallback(() => {
    if (!graph) {
      onFormatGraph({});
      return;
    }
    const nodeDimensions: Record<string, GraphLayoutNodeDimensions> = {};
    graph.nodes.forEach((node) => {
      nodeDimensions[node.id] = getFlowNodeDimensions(node, flowNodeCacheRef.current.get(node.id));
    });
    onFormatGraph(nodeDimensions);
  }, [getFlowNodeDimensions, graph, onFormatGraph]);

  const edges = useMemo<FlowEdge<GraphCanvasEdgeData>[]>(() => {
    if (!graph) {
      cachedEdgesRef.current = [];
      return [];
    }
    const dragPositions = isNodeDragActive ? dragPositionMapRef.current : null;
    const nodeLookup = new Map(
      graph.nodes.map((node) => {
        const dragPosition = dragPositions?.get(node.id);
        return [
          node.id,
          dragPosition
            ? {
                ...node,
                position: dragPosition,
              }
            : node,
        ];
      }),
    );
    const getSourceAnchorForLookup = (nodeId: string, sourceHandleId: string | null = null): GraphPosition | null => {
      const sourceNode = nodeLookup.get(nodeId);
      if (!sourceNode) {
        return null;
      }
      const dimensions = getNodeDimensions(sourceNode);
      const verticalRatio = isApiModelNode(sourceNode)
        ? sourceHandleId
          ? getToolSourceHandleAnchorRatio(sourceHandleId)
          : 0.5
        : isRoutableToolNode(sourceNode) || isMcpContextProviderNode(sourceNode)
          ? getToolSourceHandleAnchorRatio(sourceHandleId ?? TOOL_SUCCESS_HANDLE_ID)
          : 0.5;
      return {
        x: sourceNode.position.x + dimensions.width,
        y: sourceNode.position.y + dimensions.height * verticalRatio,
      };
    };
    const getTargetAnchorForLookup = (nodeId: string, targetHandleId: string | null = null): GraphPosition | null => {
      const targetNode = nodeLookup.get(nodeId);
      if (!targetNode) {
        return null;
      }
      const dimensions = getNodeDimensions(targetNode);
      const verticalRatio =
        targetNode.kind === "model" ? getApiToolContextTargetAnchorRatio(targetHandleId) : 0.5;
      return {
        x: targetNode.position.x,
        y: targetNode.position.y + dimensions.height * verticalRatio,
      };
    };
    const getEdgeRouteSignatureForLookup = (edge: GraphEdge, sourceNode: GraphNode | undefined) => {
      const sourceHandleId = inferToolEdgeSourceHandle(edge, sourceNode);
      const sourceAnchor = getSourceAnchorForLookup(edge.source_id, sourceHandleId);
      const targetAnchor = getTargetAnchorForLookup(edge.target_id, edge.target_handle_id ?? null);
      if (!sourceAnchor || !targetAnchor) {
        return null;
      }
      const routePoints = resolveEdgeRoutePoints(sourceAnchor, targetAnchor, edge.waypoints ?? [], { endWithHorizontal: true });
      return routePoints
        .map((point) => `${Math.round(point.x / 12)}:${Math.round(point.y / 12)}`)
        .join("|");
    };
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
      const signature = getEdgeRouteSignatureForLookup(edge, sourceNode);
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
    const previousEdgeDataById = shouldResolveLabelCollisions
      ? null
      : new Map(cachedEdgesRef.current.map((edge) => [edge.id, edge.data]));

    const edgeLayouts = graph.edges.map((edge) => {
      const sourceNode = nodeLookup.get(edge.source_id);
      const targetNode = nodeLookup.get(edge.target_id);
      const touchesWireJunction = isWireJunctionNode(sourceNode) || isWireJunctionNode(targetNode);
      const labelText = touchesWireJunction ? "" : String(edge.condition?.label ?? edge.label ?? "");
      const sourceHandleId = inferToolEdgeSourceHandle(edge, sourceNode);
      const sourceAnchor = getSourceAnchorForLookup(edge.source_id, sourceHandleId);
      const targetAnchor = getTargetAnchorForLookup(edge.target_id, edge.target_handle_id ?? null);
      const routeSignature = getEdgeRouteSignatureForLookup(edge, sourceNode);
      const overlappingEdges = routeSignature ? routeSignatureGroups.get(routeSignature) ?? [edge] : [edge];
      const siblingEdges = siblingEdgesByTarget.get(edge.target_id) ?? [edge];
      const toolEdgeTone =
        isRoutableToolNode(sourceNode)
          ? sourceHandleId === TOOL_FAILURE_HANDLE_ID
            ? TOOL_EDGE_TONES.failure
            : sourceHandleId === TOOL_SUCCESS_HANDLE_ID
              ? TOOL_EDGE_TONES.success
              : null
          : isApiModelNode(sourceNode)
            ? sourceHandleId === API_TOOL_CALL_HANDLE_ID
              ? API_EDGE_TONES.toolCall
              : sourceHandleId === API_FINAL_MESSAGE_HANDLE_ID
                ? API_EDGE_TONES.message
                : null
          : null;
      const isActiveEdge = runState?.current_edge_id === edge.id;
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
      const previousEdgeData = previousEdgeDataById?.get(edge.id);

      const edgeGeometry =
        shouldResolveLabelCollisions && sourceAnchor && targetAnchor
          ? getEdgeLabelPlacement({
              sourceX: sourceAnchor.x,
              sourceY: sourceAnchor.y,
              targetX: targetAnchor.x,
              targetY: targetAnchor.y,
              sourcePosition: "right" as Position,
              targetPosition: "left" as Position,
              routePoints: edge.waypoints ?? [],
              labelOffset,
            })
          : null;
      const routePoints = sourceAnchor && targetAnchor
        ? resolveEdgeRoutePoints(sourceAnchor, targetAnchor, edge.waypoints ?? [], { endWithHorizontal: true }).slice(1, -1)
        : [];

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
        targetHandle: edge.target_handle_id ?? undefined,
        selected: selectedEdgeIdSet.has(edge.id),
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
          isActive: isActiveEdge,
          routePoints,
          sourceColor: toolEdgeTone?.sourceColor ?? KIND_COLORS[sourceNode?.kind ?? ""] ?? "#6ea8ff",
          targetColor: toolEdgeTone?.targetColor ?? KIND_COLORS[targetNode?.kind ?? ""] ?? "#6ea8ff",
          routeTone: toolEdgeTone?.routeTone,
          routeShiftX,
          routeShiftY,
          labelOffset,
          labelShiftX: previousEdgeData?.labelShiftX ?? 0,
          labelShiftY: previousEdgeData?.labelShiftY ?? 0,
          showWaypointHandles: (edge.waypoints?.length ?? 0) > 0,
          waypointSelected: selectedEdgeIdSet.has(edge.id),
          waypointDragActive: waypointDrag?.edgeId === edge.id,
          onWaypointPointerDown: handleWaypointPointerDown,
        },
        style: {
          strokeWidth: isActiveEdge
            ? SELECTED_EDGE_STROKE_WIDTH
            : selectedEdgeIdSet.has(edge.id)
              ? SELECTED_EDGE_STROKE_WIDTH
              : EDGE_STROKE_WIDTH,
          strokeLinecap: "round" as const,
          filter:
            isActiveEdge
              ? "drop-shadow(0 0 7px rgba(96, 165, 250, 0.45)) drop-shadow(0 0 12px rgba(125, 211, 252, 0.28))"
              : selectedEdgeIdSet.has(edge.id)
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

    const result = edgeLayouts.map(({ labelPlacement, labelText: _labelText, pathSamples: _pathSamples, data, ...edge }) => {
      const labelShift = labelShifts.get(edge.id);
      return {
        ...edge,
        data: {
          ...data,
          labelShiftX: labelShift?.x ?? data.labelShiftX ?? 0,
          labelShiftY: labelShift?.y ?? data.labelShiftY ?? 0,
        },
      };
    });
    if (shouldResolveLabelCollisions) {
      cachedEdgesRef.current = result;
    }
    return result;
  }, [dragRenderTick, getNodeDimensions, graph, handleWaypointPointerDown, isNodeDragActive, junctionDrag, runState?.current_edge_id, selectedEdgeIdSet, waypointDrag]);

  const onNodesChange = useCallback(
    (changes: NodeChange[]) => {
      if (!graph) {
        return;
      }

      let isDragMove = false;
      let dragEnded = false;
      let hasNonDragChanges = false;
      let dragChangeCount = 0;

      for (const change of changes) {
        if (change.type === "dimensions") {
          updateMeasuredNodeDimensions(change.id, change.dimensions);
        } else if (change.type === "position" && change.position) {
          const dragging = (change as { dragging?: boolean }).dragging;
          if (dragging === true) {
            isDragMove = true;
            dragChangeCount += 1;
            if (!dragPositionMapRef.current) {
              dragPositionMapRef.current = new Map();
            }
            dragPositionMapRef.current.set(change.id, change.position);
          } else if (dragging === false) {
            dragEnded = true;
            dragPositionMapRef.current?.set(change.id, change.position);
          } else {
            hasNonDragChanges = true;
          }
        } else if (change.type === "select" && change.selected) {
          requestPrimarySelectionChange(change.id, null);
        } else if (change.type !== "select") {
          hasNonDragChanges = true;
        }
      }

      if (isDragMove && !dragEnded && !hasNonDragChanges) {
        if (dragDiagnosticsEnabled) {
          if (!dragDiagnosticSessionRef.current.active) {
            beginDragDiagnosticSession();
          }
          dragDiagnosticSessionRef.current.dragMoveEvents += 1;
          dragDiagnosticSessionRef.current.draggedNodeChanges += dragChangeCount;
        }
        if (!isNodeDragActive) {
          setIsNodeDragActive(true);
          setDragRenderTick((t) => t + 1);
        } else {
          scheduleDragRender();
        }
        return;
      }

      if (dragEnded && isNodeDragActive) {
        setIsNodeDragActive(false);
        flushDragDiagnosticSession("drag-end");
      }

      let nextGraph = graph;
      const dragPositions = dragPositionMapRef.current;
      if (dragPositions && dragPositions.size > 0) {
        nextGraph = {
          ...nextGraph,
          nodes: nextGraph.nodes.map((node) => {
            const pos = dragPositions.get(node.id);
            return pos ? { ...node, position: pos } : node;
          }),
        };
      }
      dragPositionMapRef.current = null;

      for (const change of changes) {
        if (change.type === "position" && change.position && (change as { dragging?: boolean }).dragging === undefined) {
          nextGraph = {
            ...nextGraph,
            nodes: nextGraph.nodes.map((node) =>
              node.id === change.id ? { ...node, position: change.position ?? node.position } : node,
            ),
          };
        }
        if (change.type === "remove") {
          measuredNodeDimensionsRef.current.delete(change.id);
          nextGraph = {
            ...nextGraph,
            nodes: nextGraph.nodes.filter((node) => node.id !== change.id),
            edges: nextGraph.edges.filter((edge) => edge.source_id !== change.id && edge.target_id !== change.id),
            start_node_id: nextGraph.start_node_id === change.id ? "" : nextGraph.start_node_id,
          };
          if (selectedNodeIdSet.has(change.id)) {
            clearCanvasSelection();
          }
        }
      }

      cancelPendingNodeDragFrame();
      cancelPendingDragFrame();
      if (nextGraph === graph) {
        return;
      }
      onGraphChange(nextGraph);
    },
    [beginDragDiagnosticSession, cancelPendingDragFrame, cancelPendingNodeDragFrame, clearCanvasSelection, dragDiagnosticsEnabled, flushDragDiagnosticSession, graph, isNodeDragActive, onGraphChange, requestPrimarySelectionChange, scheduleDragRender, selectedNodeIdSet, updateMeasuredNodeDimensions],
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
          if (selectedEdgeIdSet.has(change.id)) {
            clearCanvasSelection();
          }
        }
        if (change.type === "select" && change.selected) {
          requestPrimarySelectionChange(null, change.id);
        }
      });
      onGraphChange(nextGraph);
    },
    [clearCanvasSelection, graph, onGraphChange, requestPrimarySelectionChange, selectedEdgeIdSet],
  );

  const isValidConnection = useCallback(
    (connection: Connection) => {
      if (!graph || !connection.source || !connection.target) {
        return false;
      }
      const sourceNode = graph.nodes.find((node) => node.id === connection.source);
      const targetNode = graph.nodes.find((node) => node.id === connection.target);
      if (!canConnectNodes(sourceNode, targetNode, catalog)) {
        return false;
      }
      const effectiveSourceHandleId = getEffectiveSourceHandleId(sourceNode, connection.sourceHandle ?? null);
      const effectiveTargetHandleId = getEffectiveTargetHandleId(targetNode, connection.targetHandle ?? null);
      if (
        (isRoutableToolNode(sourceNode) || isMcpContextProviderNode(sourceNode)) &&
        (effectiveSourceHandleId === TOOL_CONTEXT_HANDLE_ID || effectiveTargetHandleId === API_TOOL_CONTEXT_HANDLE_ID) &&
        !isToolContextBindingConnection(sourceNode, targetNode, effectiveSourceHandleId, effectiveTargetHandleId)
      ) {
        return false;
      }
      const { duplicateEdgeId } = getConnectionConflictState(
        graph,
        connection.source,
        connection.target,
        connection.sourceHandle ?? null,
        connection.targetHandle ?? null,
      );
      if (duplicateEdgeId) {
        return false;
      }
      return true;
    },
    [
      catalog,
      getConnectionConflictState,
      getEffectiveSourceHandleId,
      getEffectiveTargetHandleId,
      graph,
      isToolContextBindingConnection,
    ],
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
        connection.targetHandle ?? null,
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
    (_event: MouseEvent | TouchEvent) => {
      setIsConnecting(false);
      if (didCreateConnectionRef.current) {
        didCreateConnectionRef.current = false;
        return;
      }
      cancelDraftConnection();
    },
    [cancelDraftConnection],
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

  const [agentMenuOpen, setAgentMenuOpen] = useState(false);

  const mcpButtonStatus = useMemo(() => {
    const mcpTools = (catalog?.tools ?? []).filter((tool) => tool.source_type === "mcp");
    const enabledMcpTools = mcpTools.filter((tool) => tool.enabled !== false);
    const selectedServerIds = new Set(
      enabledMcpTools
        .map((tool) => (typeof tool.server_id === "string" ? tool.server_id : null))
        .filter((serverId): serverId is string => Boolean(serverId)),
    );
    const selectedServers = (catalog?.mcp_servers ?? []).filter((server) => selectedServerIds.has(server.server_id));
    const allSelectedServersRunning =
      selectedServerIds.size > 0 &&
      selectedServers.length === selectedServerIds.size &&
      selectedServers.every((server) => server.running && !server.error);
    const allEnabledToolsReady = enabledMcpTools.every((tool) => tool.available !== false);
    const health: "idle" | "healthy" | "degraded" =
      enabledMcpTools.length === 0 ? "idle" : allSelectedServersRunning && allEnabledToolsReady ? "healthy" : "degraded";
    return {
      enabledToolCount: enabledMcpTools.length,
      totalToolCount: mcpTools.length,
      health,
    };
  }, [catalog]);

  if (!graph) {
    return <div className="panel empty-panel">No graph selected.</div>;
  }

  return (
    <div
      className="graph-shell panel"
      onBlurCapture={(event) => {
        if (!(event.currentTarget instanceof HTMLElement)) {
          return;
        }
        const nextFocused = event.relatedTarget;
        if (nextFocused instanceof Node && event.currentTarget.contains(nextFocused)) {
          return;
        }
        setAgentMenuOpen(false);
      }}
    >
      <div className="graph-shell-header panel-header">
        <div>
          <h2
            className="graph-shell-name"
            contentEditable
            suppressContentEditableWarning
            spellCheck={false}
            onBlur={(event) => {
              const next = (event.currentTarget.textContent ?? "").trim();
              if (next && next !== graph.name) {
                onGraphChange({ ...graph, name: next });
              } else {
                event.currentTarget.textContent = graph.name;
              }
            }}
            onKeyDown={(event) => {
              if (event.key === "Enter") {
                event.preventDefault();
                event.currentTarget.blur();
              }
              if (event.key === "Escape") {
                event.currentTarget.textContent = graph.name;
                event.currentTarget.blur();
              }
            }}
          >
            {graph.name}
          </h2>
          <p>
            Drag node cards into the canvas and use the inspector to configure prompts, providers, tools, and routing.
          </p>
          {focusedAgentName ? (
            <div className="graph-run-context">
              <span className="graph-run-context-label">Focused agent</span>
              <span className="graph-run-context-name">{focusedAgentName}</span>
              <span className={`graph-run-context-status graph-run-context-status--${focusedAgentStatus ?? "idle"}`}>
                {formatRunStatusLabel(focusedAgentStatus)}
              </span>
            </div>
          ) : null}
          {editorMessage ? <p className="editor-message">{editorMessage}</p> : null}
        </div>
        <div className="graph-scroll-nav">
          <button
            type="button"
            className={`secondary-button graph-scroll-nav-button${isMcpPanelOpen ? " is-active" : ""}`}
            onClick={() => onToggleMcpPanel?.()}
            aria-label={isMcpPanelOpen ? "Hide project MCP modal" : "Show project MCP modal"}
            title={`${isMcpPanelOpen ? "Hide" : "Show"} project MCP modal (${mcpButtonStatus.enabledToolCount}/${mcpButtonStatus.totalToolCount} tools enabled)`}
          >
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <path d="M4 7.5h16M4 12h16M4 16.5h10" />
              <rect x="15" y="14.5" width="5" height="4" rx="1" />
              <path d="M17.5 10.5v4" />
            </svg>
            <span>MCP</span>
            <span className="graph-scroll-nav-button-meta" aria-hidden="true">
              <span className={`mcp-health-indicator mcp-health-indicator--${mcpButtonStatus.health}`} />
              <span className="graph-scroll-nav-button-fraction">
                {mcpButtonStatus.enabledToolCount}/{mcpButtonStatus.totalToolCount}
              </span>
            </span>
          </button>
          <button type="button" className="secondary-button graph-scroll-nav-button" onClick={onScrollToTop}>
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <path d="M12 5 5 12M12 5l7 7M12 5v14" />
            </svg>
            <span>To Controls</span>
          </button>
        </div>
      </div>
      <div className="graph-workspace">
        {environmentAgents.length > 0 ? (
          <EnvironmentAgentMenu
            agents={environmentAgents}
            selectedAgentId={selectedAgentId}
            open={agentMenuOpen}
            onToggle={() => setAgentMenuOpen((previous) => !previous)}
            onSelectAgent={(agentId) => {
              onSelectAgent?.(agentId);
              setAgentMenuOpen(false);
            }}
          />
        ) : null}
        <div
          ref={canvasRef}
          className={`graph-canvas${drawerOpen ? " graph-canvas--drawer-open" : ""}${isProviderDragActive || isSavedNodeDragActive ? " is-drop-target" : ""}${isConnecting ? " is-connecting" : ""}${pendingPlacement ? " is-placing-node" : ""}${draftConnection ? " is-routing-wire" : ""}${junctionDrag ? " is-dragging-junction" : ""}${waypointDrag ? " is-dragging-waypoint" : ""}${isNodeDragActive ? " is-dragging-node" : ""}${marqueeSelection ? " is-marquee-selecting" : ""}`}
          onDrop={onDrop}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          onMouseDownCapture={handleCanvasMouseDownCapture}
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
            <button
              type="button"
              className="graph-toolbar-button"
              onClick={handleFormatGraph}
              aria-label="Format graph layout"
              title="Format graph layout"
              disabled={graph.nodes.length === 0}
            >
              <svg viewBox="0 0 24 24" aria-hidden="true">
                <path d="M4 6h6M14 6h6M4 12h4M12 12h8M4 18h8M16 18h4" />
              </svg>
              <span>Format</span>
            </button>
          </div>
          {milestoneChatEntries.length > 0 ? (
            <div className="graph-milestone-chat" role="log" aria-live="polite" aria-label="Recent environment milestones">
              {milestoneChatEntries.map((entry) => (
                <div key={entry.id} className="graph-milestone-chat-entry" style={{ opacity: entry.opacity }}>
                  {entry.label}
                </div>
              ))}
            </div>
          ) : null}
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
          {marqueeSelectionStyle ? <div className="graph-selection-marquee" style={marqueeSelectionStyle} aria-hidden="true" /> : null}
          <ReactFlow
            nodes={nodes}
            edges={edges}
            nodeTypes={nodeTypes}
            edgeTypes={edgeTypes}
            minZoom={MIN_GRAPH_ZOOM}
            connectionLineComponent={GraphCanvasConnectionLine}
            connectionLineType={ConnectionLineType.SmoothStep}
            connectionLineStyle={{
              stroke: "#9ab7ff",
              strokeWidth: DRAFT_WIRE_STROKE_WIDTH,
              strokeDasharray: "6 6",
              filter: "drop-shadow(0 0 10px rgba(111, 130, 255, 0.45))",
            }}
            defaultViewport={{ x: 0, y: 0, zoom: 1 }}
            panOnDrag={!vizLocked && !pendingPlacement && !junctionDrag && !waypointDrag && !marqueeSelection}
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
                cancelDraftConnection();
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
                cancelDraftConnection();
                return;
              }
              if (pendingPlacement) {
                confirmPendingPlacement({ x: event.clientX, y: event.clientY });
                return;
              }
              clearCanvasChrome();
            }}
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
                    title={
                      item.provider
                        ? `${item.label}: ${item.provider.display_name} (${item.hotkey})${item.isFavorite ? " [pinned]" : ""}`
                        : `No ${item.label.toLowerCase()} node available`
                    }
                  >
                    <span className="graph-quick-add-hotkey">{item.hotkey}</span>
                    <span className="graph-quick-add-label-stack">
                      <span className="graph-quick-add-label">{item.label}</span>
                      <span className="graph-quick-add-provider">
                        {item.provider ? item.provider.display_name : "Unavailable"}
                      </span>
                    </span>
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
                <strong>Cmd/Ctrl+C, Cmd/Ctrl+V</strong> copy and paste selected node
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
                  hotbarItems={quickAddItems}
                  hotbarFavorites={hotbarFavorites}
                  onToggleHotbarFavorite={handleToggleHotbarFavorite}
                  savedNodes={savedNodes}
                  onSavedNodeClick={addSavedNode}
                  onDeleteSavedNode={handleDeleteSavedNode}
                />
              ) : null}
              {drawerTab === "inspect" ? (
                <GraphInspector
                  graph={graph}
                  catalog={catalog}
                  runState={runState}
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
                      <span>
                        {focusedAgentName
                          ? `Run the environment and follow ${focusedAgentName} below.`
                          : "Run the graph from this drawer and watch events stream below."}
                      </span>
                    </div>
                    <button type="button" onClick={onRunGraph} disabled={isRunning}>
                      {isRunning ? "Running..." : runButtonLabel}
                    </button>
                  </section>
                  <section className="panel graph-run-summary">
                    <div className="panel-header">
                      <h2>{focusedAgentName ? `${focusedAgentName} Run` : "Run State"}</h2>
                      <p>{activeRunId ? `Latest run: ${activeRunId}` : "No active run."}</p>
                    </div>
                    <dl className="state-grid">
                      <div>
                        <dt>Status</dt>
                        <dd>{runSummary?.status ?? runState?.status ?? "idle"}</dd>
                      </div>
                      <div>
                        <dt>Current Node</dt>
                        <dd>{runSummary?.currentNodeLabel ?? runState?.current_node_id ?? "n/a"}</dd>
                      </div>
                      <div>
                        <dt>Progress</dt>
                        <dd>{`${runSummary?.completedNodes ?? Object.keys(runState?.visit_counts ?? {}).length}/${runSummary?.totalNodes ?? graph.nodes.length}`}</dd>
                      </div>
                      <div>
                        <dt>Transitions</dt>
                        <dd>{runSummary?.transitionCount ?? runState?.transition_history.length ?? 0}</dd>
                      </div>
                      <div>
                        <dt>Retries</dt>
                        <dd>{runSummary?.retryCount ?? 0}</dd>
                      </div>
                      <div>
                        <dt>Errors</dt>
                        <dd>{runSummary?.errorCount ?? Object.keys(runState?.node_errors ?? {}).length}</dd>
                      </div>
                      <div>
                        <dt>Elapsed</dt>
                        <dd>{runSummary?.elapsedLabel ?? "Not started"}</dd>
                      </div>
                      <div>
                        <dt>Last Heartbeat</dt>
                        <dd>{runSummary?.lastHeartbeatLabel ?? runState?.last_heartbeat_at ?? "n/a"}</dd>
                      </div>
                    </dl>
                    <div className="json-block">
                      <strong>Final Output</strong>
                      <pre>{JSON.stringify(runSummary?.finalOutput ?? runState?.final_output ?? null, null, 2)}</pre>
                    </div>
                    <div className="json-block">
                      <strong>Errors</strong>
                      <pre>{JSON.stringify(runSummary?.nodeErrors ?? runState?.node_errors ?? {}, null, 2)}</pre>
                    </div>
                  </section>
                  <EventTimeline events={events} groups={eventGroups} embedded />
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
      {promptBlockDetailsNode ? (
        <PromptBlockDetailsModal
          graph={graph}
          node={promptBlockDetailsNode}
          runState={runState}
          onGraphChange={onGraphChange}
          onClose={() => setPromptBlockDetailsNodeId(null)}
        />
      ) : null}
      {displayResponseNode ? (
        <DisplayResponseModal graph={graph} node={displayResponseNode} runState={runState} onClose={() => setDisplayResponseNodeId(null)} />
      ) : null}
      {contextBuilderPayloadNode ? (
        <ContextBuilderPayloadModal
          graph={graph}
          node={contextBuilderPayloadNode}
          runState={runState}
          runtimeView={contextBuilderModalRuntime}
          onClose={() => setContextBuilderPayloadNodeId(null)}
        />
      ) : null}
    </div>
  );
}
