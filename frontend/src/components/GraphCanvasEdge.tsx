import { memo, useMemo } from "react";
import type { CSSProperties, PointerEvent as ReactPointerEvent } from "react";
import { EdgeLabelRenderer, getSmoothStepPath } from "reactflow";
import type { ConnectionLineComponentProps, EdgeProps } from "reactflow";

import { warnGraphDiagnostic } from "../lib/dragDiagnostics";

type GraphPosition = {
  x: number;
  y: number;
};

export type GraphCanvasEdgeData = {
  kind: string;
  labelOffset?: number;
  labelShiftX?: number;
  labelShiftY?: number;
  routePoints?: GraphPosition[];
  sourceColor?: string;
  targetColor?: string;
  routeTone?: "tool-success" | "tool-failure" | "api-tool-call" | "api-message";
  routeShiftX?: number;
  routeShiftY?: number;
  showWaypointHandles?: boolean;
  waypointSelected?: boolean;
  waypointDragActive?: boolean;
  onWaypointPointerDown?: (edgeId: string, waypointIndex: number, clientPosition: { x: number; y: number }) => void;
};

export function offsetPointAcrossEdge(point: { x: number; y: number }, tangent: { x: number; y: number }, offset: number) {
  if (!offset) {
    return point;
  }

  const tangentLength = Math.hypot(tangent.x, tangent.y);
  if (tangentLength < 0.001) {
    return { x: point.x, y: point.y - offset };
  }

  return {
    x: point.x + (-tangent.y / tangentLength) * offset,
    y: point.y + (tangent.x / tangentLength) * offset,
  };
}

function polylineLengths(points: GraphPosition[]) {
  const lengths: number[] = [];
  let totalLength = 0;
  for (let index = 1; index < points.length; index += 1) {
    const segmentLength = Math.hypot(points[index].x - points[index - 1].x, points[index].y - points[index - 1].y);
    lengths.push(segmentLength);
    totalLength += segmentLength;
  }
  return { lengths, totalLength };
}

const POINT_EPSILON = 0.5;
const ORTHOGONAL_CORNER_RADIUS = 20;
const TARGET_APPROACH_OFFSET = 36;

function isSamePoint(left: GraphPosition, right: GraphPosition) {
  return Math.abs(left.x - right.x) < POINT_EPSILON && Math.abs(left.y - right.y) < POINT_EPSILON;
}

function appendUniquePoint(points: GraphPosition[], point: GraphPosition) {
  if (!points.length || !isSamePoint(points[points.length - 1], point)) {
    points.push(point);
  }
}

function collapseCollinearPoints(points: GraphPosition[]) {
  if (points.length <= 2) {
    return points;
  }

  const collapsed: GraphPosition[] = [points[0]];
  for (let index = 1; index < points.length - 1; index += 1) {
    const previous = collapsed[collapsed.length - 1];
    const current = points[index];
    const next = points[index + 1];
    const isHorizontal = Math.abs(previous.y - current.y) < POINT_EPSILON && Math.abs(current.y - next.y) < POINT_EPSILON;
    const isVertical = Math.abs(previous.x - current.x) < POINT_EPSILON && Math.abs(current.x - next.x) < POINT_EPSILON;
    if (isHorizontal || isVertical) {
      continue;
    }
    collapsed.push(current);
  }
  collapsed.push(points[points.length - 1]);
  return collapsed;
}

function isAxisAlignedRoute(points: GraphPosition[]) {
  for (let index = 1; index < points.length; index += 1) {
    const previous = points[index - 1];
    const current = points[index];
    if (Math.abs(previous.x - current.x) >= POINT_EPSILON && Math.abs(previous.y - current.y) >= POINT_EPSILON) {
      return false;
    }
  }
  return true;
}

function roundedOrthogonalPath(points: GraphPosition[]) {
  if (points.length === 0) {
    return "";
  }
  if (points.length === 1) {
    return `M ${points[0].x} ${points[0].y}`;
  }

  let path = `M ${points[0].x} ${points[0].y}`;
  for (let index = 1; index < points.length; index += 1) {
    const current = points[index];
    if (index === points.length - 1) {
      path += ` L ${current.x} ${current.y}`;
      continue;
    }

    const previous = points[index - 1];
    const next = points[index + 1];
    const incomingX = current.x - previous.x;
    const incomingY = current.y - previous.y;
    const outgoingX = next.x - current.x;
    const outgoingY = next.y - current.y;
    const incomingLength = Math.hypot(incomingX, incomingY);
    const outgoingLength = Math.hypot(outgoingX, outgoingY);

    const isAxisAligned =
      (Math.abs(incomingX) < POINT_EPSILON || Math.abs(incomingY) < POINT_EPSILON) &&
      (Math.abs(outgoingX) < POINT_EPSILON || Math.abs(outgoingY) < POINT_EPSILON);
    if (!isAxisAligned || incomingLength < POINT_EPSILON || outgoingLength < POINT_EPSILON) {
      path += ` L ${current.x} ${current.y}`;
      continue;
    }

    const radius = Math.min(ORTHOGONAL_CORNER_RADIUS, incomingLength / 2, outgoingLength / 2);
    if (radius < POINT_EPSILON) {
      path += ` L ${current.x} ${current.y}`;
      continue;
    }

    const incomingPoint = {
      x: current.x - (incomingX / incomingLength) * radius,
      y: current.y - (incomingY / incomingLength) * radius,
    };
    const outgoingPoint = {
      x: current.x + (outgoingX / outgoingLength) * radius,
      y: current.y + (outgoingY / outgoingLength) * radius,
    };

    path += ` L ${incomingPoint.x} ${incomingPoint.y}`;
    path += ` Q ${current.x} ${current.y} ${outgoingPoint.x} ${outgoingPoint.y}`;
  }

  return path;
}

export function buildOrthogonalPolylinePoints(points: GraphPosition[], options?: { endWithHorizontal?: boolean }) {
  if (points.length <= 1) {
    return points;
  }

  const endWithHorizontal = options?.endWithHorizontal ?? false;
  const source = points[0];
  const target = points[points.length - 1];
  const intermediatePoints = points.slice(1, -1);
  const orthogonalPoints: GraphPosition[] = [source];
  let current = source;

  intermediatePoints.forEach((point) => {
    appendUniquePoint(orthogonalPoints, { x: point.x, y: current.y });
    appendUniquePoint(orthogonalPoints, { x: point.x, y: point.y });
    current = orthogonalPoints[orthogonalPoints.length - 1];
  });

  if (!endWithHorizontal || Math.abs(current.y - target.y) < POINT_EPSILON) {
    appendUniquePoint(orthogonalPoints, { x: target.x, y: current.y });
    appendUniquePoint(orthogonalPoints, target);
    return collapseCollinearPoints(orthogonalPoints);
  }

  const approachX = target.x - TARGET_APPROACH_OFFSET;
  appendUniquePoint(orthogonalPoints, { x: approachX, y: current.y });
  appendUniquePoint(orthogonalPoints, { x: approachX, y: target.y });
  appendUniquePoint(orthogonalPoints, target);
  return collapseCollinearPoints(orthogonalPoints);
}

export function buildPolylinePath(points: GraphPosition[], options?: { endWithHorizontal?: boolean }): string {
  return roundedOrthogonalPath(buildOrthogonalPolylinePoints(points, options));
}

export function resolveEdgeRoutePoints(
  source: GraphPosition,
  target: GraphPosition,
  routePoints: GraphPosition[] = [],
  options?: { endWithHorizontal?: boolean },
) {
  if (routePoints.length === 0) {
    return [source, target];
  }

  const candidatePoints = [source, ...routePoints, target];
  if (isAxisAlignedRoute(candidatePoints)) {
    return collapseCollinearPoints(candidatePoints);
  }

  return buildOrthogonalPolylinePoints(candidatePoints, options);
}

export function midpointOnPolyline(points: GraphPosition[]) {
  if (points.length === 0) {
    return {
      point: { x: 0, y: 0 },
      tangent: { x: 1, y: 0 },
    };
  }
  if (points.length === 1) {
    return {
      point: points[0],
      tangent: { x: 1, y: 0 },
    };
  }

  const { lengths, totalLength } = polylineLengths(points);
  if (totalLength < 0.001) {
    return {
      point: points[0],
      tangent: { x: points[points.length - 1].x - points[0].x, y: points[points.length - 1].y - points[0].y },
    };
  }

  const midpointLength = totalLength * 0.5;
  let traversed = 0;
  for (let index = 1; index < points.length; index += 1) {
    const segmentLength = lengths[index - 1];
    if (traversed + segmentLength >= midpointLength) {
      const start = points[index - 1];
      const end = points[index];
      const distanceIntoSegment = midpointLength - traversed;
      const ratio = segmentLength < 0.001 ? 0 : distanceIntoSegment / segmentLength;
      return {
        point: {
          x: start.x + (end.x - start.x) * ratio,
          y: start.y + (end.y - start.y) * ratio,
        },
        tangent: {
          x: end.x - start.x,
          y: end.y - start.y,
        },
      };
    }
    traversed += segmentLength;
  }

  const last = points[points.length - 1];
  const previous = points[points.length - 2];
  return {
    point: last,
    tangent: {
      x: last.x - previous.x,
      y: last.y - previous.y,
    },
  };
}

export function buildSmoothEdgePath({
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
}: {
  sourceX: number;
  sourceY: number;
  targetX: number;
  targetY: number;
  sourcePosition: EdgeProps<GraphCanvasEdgeData>["sourcePosition"] | ConnectionLineComponentProps["fromPosition"];
  targetPosition: EdgeProps<GraphCanvasEdgeData>["targetPosition"] | ConnectionLineComponentProps["toPosition"];
}) {
  return getSmoothStepPath({
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
    borderRadius: 28,
    offset: 18,
  });
}

function samplePathLabelPlacement(
  edgePath: string,
  labelOffset: number,
  fallbackPoint: GraphPosition,
  fallbackTangent: GraphPosition,
) {
  if (typeof document === "undefined") {
    return {
      point: offsetPointAcrossEdge(fallbackPoint, fallbackTangent, labelOffset),
      tangent: fallbackTangent,
    };
  }

  try {
    const measurementPath = document.createElementNS("http://www.w3.org/2000/svg", "path");
    measurementPath.setAttribute("d", edgePath);
    const totalLength = measurementPath.getTotalLength();
    const midpointLength = totalLength * 0.5;
    const tangentSampleLength = Math.min(Math.max(totalLength * 0.08, 8), 18);
    const point = measurementPath.getPointAtLength(midpointLength);
    const tangentStart = measurementPath.getPointAtLength(Math.max(midpointLength - tangentSampleLength, 0));
    const tangentEnd = measurementPath.getPointAtLength(Math.min(midpointLength + tangentSampleLength, totalLength));
    const tangent = {
      x: tangentEnd.x - tangentStart.x,
      y: tangentEnd.y - tangentStart.y,
    };
    return {
      point: offsetPointAcrossEdge(point, tangent, labelOffset),
      tangent,
    };
  } catch (error) {
    warnGraphDiagnostic("GraphCanvasEdge", "label placement fallback", error, {
      edgePath,
      labelOffset,
    });
    return {
      point: offsetPointAcrossEdge(fallbackPoint, fallbackTangent, labelOffset),
      tangent: fallbackTangent,
    };
  }
}

export function getEdgeLabelPlacement({
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  routePoints = [],
  labelOffset = 0,
}: {
  sourceX: number;
  sourceY: number;
  targetX: number;
  targetY: number;
  sourcePosition: EdgeProps<GraphCanvasEdgeData>["sourcePosition"] | ConnectionLineComponentProps["fromPosition"];
  targetPosition: EdgeProps<GraphCanvasEdgeData>["targetPosition"] | ConnectionLineComponentProps["toPosition"];
  routePoints?: GraphPosition[];
  labelOffset?: number;
}) {
  const hasRoutePoints = routePoints.length > 0;
  const [smoothEdgePath, fallbackLabelX, fallbackLabelY] = buildSmoothEdgePath({
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
  });

  if (hasRoutePoints) {
    const routedPolylinePoints = resolveEdgeRoutePoints(
      { x: sourceX, y: sourceY },
      { x: targetX, y: targetY },
      routePoints,
      { endWithHorizontal: true },
    );
    const midpoint = midpointOnPolyline(routedPolylinePoints);
    return {
      edgePath: roundedOrthogonalPath(routedPolylinePoints),
      point: offsetPointAcrossEdge(midpoint.point, midpoint.tangent, labelOffset),
      tangent: midpoint.tangent,
    };
  }

  const fallbackPoint = { x: fallbackLabelX, y: fallbackLabelY };
  const fallbackTangent = { x: targetX - sourceX, y: targetY - sourceY };
  const placement = samplePathLabelPlacement(smoothEdgePath, labelOffset, fallbackPoint, fallbackTangent);

  return {
    edgePath: smoothEdgePath,
    point: placement.point,
    tangent: placement.tangent,
  };
}

function GraphCanvasEdgeComponent({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  style,
  markerEnd,
  selected,
  data,
  label,
}: EdgeProps<GraphCanvasEdgeData>) {
  const routePoints = data?.routePoints ?? [];
  const sourceColor = data?.sourceColor ?? "#6ea8ff";
  const targetColor = data?.targetColor ?? sourceColor;
  const labelOffset = data?.labelOffset ?? 0;
  const routeShiftX = data?.routeShiftX ?? 0;
  const routeShiftY = data?.routeShiftY ?? 0;
  const labelShiftX = data?.labelShiftX ?? 0;
  const labelShiftY = data?.labelShiftY ?? 0;
  const showWaypointHandles = data?.showWaypointHandles ?? false;
  const waypointSelected = data?.waypointSelected ?? false;
  const waypointDragActive = data?.waypointDragActive ?? false;
  const onWaypointPointerDown = data?.onWaypointPointerDown;
  const { edgePath, point: labelPosition } = useMemo(
    () =>
      getEdgeLabelPlacement({
        sourceX,
        sourceY,
        targetX,
        targetY,
        sourcePosition,
        targetPosition,
        routePoints,
        labelOffset,
      }),
    [labelOffset, routePoints, sourcePosition, sourceX, sourceY, targetPosition, targetX, targetY],
  );
  const pathTransform = routeShiftX !== 0 || routeShiftY !== 0 ? `translate(${routeShiftX} ${routeShiftY})` : undefined;

  const labelStyle = {
    transform: `translate(-50%, -50%) translate(${labelPosition.x + routeShiftX + labelShiftX}px, ${labelPosition.y + routeShiftY + labelShiftY}px)`,
  } satisfies CSSProperties;
  const strokeWidth = typeof style?.strokeWidth === "number" ? style.strokeWidth : Number(style?.strokeWidth ?? 1.5);
  const interactionStrokeWidth = Math.max(strokeWidth + 18, 24);
  const stripeLength = data?.kind === "conditional" ? 7 : 10;
  const interactionEdgeStyle = {
    stroke: "transparent",
    strokeWidth: interactionStrokeWidth,
    strokeLinecap: "round",
    strokeLinejoin: "round",
    pointerEvents: "stroke",
  } satisfies CSSProperties;
  const baseEdgeStyle = {
    ...style,
    stroke: targetColor,
    strokeWidth,
    strokeLinecap: "round",
    strokeLinejoin: "round",
    strokeDasharray: undefined,
  } satisfies CSSProperties;
  const stripeEdgeStyle = {
    ...style,
    stroke: sourceColor,
    strokeWidth,
    strokeLinecap: "butt",
    strokeLinejoin: "round",
    strokeDasharray: `${stripeLength} ${stripeLength}`,
    filter: undefined,
  } satisfies CSSProperties;
  const handleWaypointPointerDown = (event: ReactPointerEvent<HTMLButtonElement>, waypointIndex: number) => {
    event.preventDefault();
    event.stopPropagation();
    onWaypointPointerDown?.(id, waypointIndex, { x: event.clientX, y: event.clientY });
  };

  return (
    <>
      <path d={edgePath} fill="none" className="react-flow__edge-interaction graph-edge-hitbox" style={interactionEdgeStyle} transform={pathTransform} />
      <path id={id} d={edgePath} fill="none" className="react-flow__edge-path" style={baseEdgeStyle} markerEnd={markerEnd} transform={pathTransform} />
      <path d={edgePath} fill="none" className="react-flow__edge-path" style={stripeEdgeStyle} transform={pathTransform} />
      {label ? (
        <EdgeLabelRenderer>
          <div
            className={`graph-edge-label ${data?.kind === "conditional" ? "graph-edge-label--conditional" : ""} ${
              data?.routeTone ? `graph-edge-label--${data.routeTone}` : ""
            } ${
              selected ? "is-selected" : ""
            }`}
            style={labelStyle}
          >
            {String(label)}
          </div>
        </EdgeLabelRenderer>
      ) : null}
      {routePoints.length > 0 && (showWaypointHandles || waypointDragActive) ? (
        <EdgeLabelRenderer>
          <>
            {routePoints.map((waypoint, waypointIndex) => {
              const waypointStyle = {
                transform: `translate(-50%, -50%) translate(${waypoint.x + routeShiftX}px, ${waypoint.y + routeShiftY}px)`,
                pointerEvents: "all",
              } satisfies CSSProperties;
              return (
                <button
                  key={`${id}-waypoint-${waypointIndex}`}
                  type="button"
                  className={`graph-edge-waypoint nopan nodrag${waypointSelected ? " is-selected" : ""}${waypointDragActive ? " is-dragging" : ""}`}
                  style={waypointStyle}
                  onPointerDown={(event) => handleWaypointPointerDown(event, waypointIndex)}
                  aria-label="Drag wire corner"
                  title="Drag wire corner"
                />
              );
            })}
          </>
        </EdgeLabelRenderer>
      ) : null}
    </>
  );
}

function GraphCanvasConnectionLineComponent({
  fromX,
  fromY,
  toX,
  toY,
  fromPosition,
  toPosition,
  connectionLineStyle,
}: ConnectionLineComponentProps) {
  const [edgePath] = buildSmoothEdgePath({
    sourceX: fromX,
    sourceY: fromY,
    targetX: toX,
    targetY: toY,
    sourcePosition: fromPosition,
    targetPosition: toPosition,
  });

  return <path d={edgePath} fill="none" className="react-flow__connection-path" style={connectionLineStyle} />;
}

export const GraphCanvasEdge = memo(GraphCanvasEdgeComponent);
export const GraphCanvasConnectionLine = memo(GraphCanvasConnectionLineComponent);
