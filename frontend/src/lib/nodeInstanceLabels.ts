import type { GraphDefinition, GraphNode } from "./types";

function baseNodeLabel(node: GraphNode): string {
  const explicitLabel = String(node.label ?? "").trim();
  if (explicitLabel.length > 0) {
    return explicitLabel;
  }
  const providerLabel = String(node.provider_label ?? "").trim();
  if (providerLabel.length > 0) {
    return providerLabel;
  }
  return node.id;
}

export function buildNodeInstanceLabelMap(graph: GraphDefinition | null): Map<string, string> {
  const labels = new Map<string, string>();
  if (!graph) {
    return labels;
  }
  const nodesByBaseLabel = new Map<string, GraphNode[]>();
  graph.nodes.forEach((node) => {
    const baseLabel = baseNodeLabel(node);
    const bucket = nodesByBaseLabel.get(baseLabel) ?? [];
    bucket.push(node);
    nodesByBaseLabel.set(baseLabel, bucket);
  });
  nodesByBaseLabel.forEach((nodes, baseLabel) => {
    if (nodes.length <= 1) {
      labels.set(nodes[0].id, baseLabel);
      return;
    }
    nodes.forEach((node, index) => {
      labels.set(node.id, `${baseLabel} ${index + 1}`);
    });
  });
  return labels;
}

export function getNodeInstanceLabelById(graph: GraphDefinition | null, nodeId: string): string {
  return buildNodeInstanceLabelMap(graph).get(nodeId) ?? nodeId;
}

export function getNodeInstanceLabel(graph: GraphDefinition | null, node: GraphNode): string {
  return buildNodeInstanceLabelMap(graph).get(node.id) ?? baseNodeLabel(node);
}
