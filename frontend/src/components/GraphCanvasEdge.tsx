import { memo, useMemo } from "react";
import type { CSSProperties } from "react";
import { EdgeLabelRenderer, getSmoothStepPath } from "reactflow";
import type { EdgeProps } from "reactflow";

export type GraphCanvasEdgeData = {
  kind: string;
  labelOffset?: number;
};

function offsetPointAcrossEdge(point: { x: number; y: number }, tangent: { x: number; y: number }, offset: number) {
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
  const [edgePath, fallbackLabelX, fallbackLabelY] = getSmoothStepPath({
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
    borderRadius: 28,
    offset: 18,
  });

  const labelOffset = data?.labelOffset ?? 0;
  const labelPosition = useMemo(() => {
    const fallbackPoint = { x: fallbackLabelX, y: fallbackLabelY };
    const fallbackTangent = { x: targetX - sourceX, y: targetY - sourceY };

    if (typeof document === "undefined") {
      return offsetPointAcrossEdge(fallbackPoint, fallbackTangent, labelOffset);
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
      return offsetPointAcrossEdge(
        point,
        {
          x: tangentEnd.x - tangentStart.x,
          y: tangentEnd.y - tangentStart.y,
        },
        labelOffset,
      );
    } catch {
      return offsetPointAcrossEdge(fallbackPoint, fallbackTangent, labelOffset);
    }
  }, [edgePath, fallbackLabelX, fallbackLabelY, labelOffset, sourceX, sourceY, targetX, targetY]);

  const labelStyle = {
    transform: `translate(-50%, -50%) translate(${labelPosition.x}px, ${labelPosition.y}px)`,
  } satisfies CSSProperties;

  return (
    <>
      <path id={id} d={edgePath} fill="none" className="react-flow__edge-path" style={style} markerEnd={markerEnd} />
      {label ? (
        <EdgeLabelRenderer>
          <div
            className={`graph-edge-label ${data?.kind === "conditional" ? "graph-edge-label--conditional" : ""} ${
              selected ? "is-selected" : ""
            }`}
            style={labelStyle}
          >
            {String(label)}
          </div>
        </EdgeLabelRenderer>
      ) : null}
    </>
  );
}

export const GraphCanvasEdge = memo(GraphCanvasEdgeComponent);
