import dagre from "dagre";

import { DEFAULT_GRAPH_ENV_VARS, getGraphEnvVars } from "./graphEnv";
import type { EditorCatalog, GraphDefinition, GraphEdge, GraphNode, GraphPosition, NodeProviderDefinition } from "./types";

function slugify(value: string): string {
  return value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

function uniqueNodeId(graph: GraphDefinition, base: string): string {
  const existingIds = new Set(graph.nodes.map((node) => node.id));
  let candidate = slugify(base) || "node";
  let suffix = 1;
  while (existingIds.has(candidate)) {
    candidate = `${slugify(base) || "node"}-${suffix}`;
    suffix += 1;
  }
  return candidate;
}

function defaultModelProviderName(provider: NodeProviderDefinition): string {
  if (provider.provider_id === "provider.openai") {
    return "openai";
  }
  if (provider.provider_id === "provider.claude") {
    return "claude";
  }
  return "mock";
}

function defaultModelName(providerName: string): string {
  if (providerName === "openai") {
    return "gpt-4.1-mini";
  }
  if (providerName === "claude") {
    return "claude-3-5-haiku-latest";
  }
  return "mock-default";
}

function defaultModelConfig(promptName: string): GraphNode["config"] {
  return {
    provider_name: "mock",
    prompt_name: promptName,
    mode: promptName,
    system_prompt: "You are a model node in an editable graph.",
    user_message_template: "{input_payload}",
    response_mode: "message",
    allowed_tool_names: [],
    model: "mock-default",
  };
}

function defaultProviderConfig(provider: NodeProviderDefinition): GraphNode["config"] {
  const providerName = defaultModelProviderName(provider);
  if (provider.provider_id === "provider.openai") {
    return {
      provider_name: providerName,
      model: "gpt-4.1-mini",
    };
  }
  if (provider.provider_id === "provider.claude") {
    return {
      provider_name: providerName,
      model: "claude-3-5-haiku-latest",
      max_tokens: 1024,
    };
  }
  return {
    provider_name: providerName,
    model: "mock-default",
  };
}

export function syncModelNodeWithProvider(modelNode: GraphNode, providerNode: GraphNode): GraphNode {
  if (modelNode.kind !== "model" || providerNode.kind !== "provider") {
    return modelNode;
  }

  const providerName = String(
    providerNode.config.provider_name ??
      providerNode.model_provider_name ??
      providerNode.provider_id.replace(/^provider\./, "") ??
      "mock",
  ).trim() || "mock";

  const nextConfig: GraphNode["config"] = {
    ...modelNode.config,
    provider_binding_node_id: providerNode.id,
    provider_name: providerName,
  };

  const providerModel = typeof providerNode.config.model === "string" ? providerNode.config.model.trim() : "";
  nextConfig.model = providerModel || String(modelNode.config.model ?? "").trim() || defaultModelName(providerName);

  if (providerNode.config.api_base !== undefined) {
    nextConfig.api_base = providerNode.config.api_base;
  }

  if (providerNode.config.max_tokens !== undefined) {
    nextConfig.max_tokens = providerNode.config.max_tokens;
  }

  return {
    ...modelNode,
    model_provider_name: providerName,
    config: nextConfig,
  };
}

export function createBlankGraph(): GraphDefinition {
  return {
    graph_id: `agent-${Date.now()}`,
    name: "Untitled Agent",
    description: "",
    version: "1.0",
    start_node_id: "",
    env_vars: { ...DEFAULT_GRAPH_ENV_VARS },
    nodes: [],
    edges: [],
    node_providers: [],
  };
}

export function createNodeFromProvider(
  graph: GraphDefinition,
  provider: NodeProviderDefinition,
  catalog: EditorCatalog,
  position: GraphPosition,
): GraphNode {
  const id = uniqueNodeId(graph, provider.display_name);
  const baseNode: GraphNode = {
    id,
    kind: provider.node_kind,
    category: provider.category,
    label: provider.display_name,
    description: provider.description,
    provider_id: provider.provider_id,
    provider_label: provider.display_name,
    position,
    config: {},
  };

  if (provider.node_kind === "input") {
    return {
      ...baseNode,
      config: {
        input_binding: { type: "input_payload" },
      },
    };
  }

  if (provider.node_kind === "model") {
    const promptName = `${id}_prompt`;
    return {
      ...baseNode,
      model_provider_name: "mock",
      prompt_name: promptName,
      config: defaultModelConfig(promptName),
    };
  }

  if (provider.node_kind === "tool") {
    const toolName = catalog.tools[0]?.name ?? "";
    return {
      ...baseNode,
      tool_name: toolName,
      config: {
        tool_name: toolName,
      },
    };
  }

  if (provider.node_kind === "data") {
    return {
      ...baseNode,
      config: {
        mode: "passthrough",
      },
    };
  }

  if (provider.node_kind === "provider") {
    const providerName = defaultModelProviderName(provider);
    return {
      ...baseNode,
      model_provider_name: providerName,
      config: defaultProviderConfig(provider),
    };
  }

  return {
    ...baseNode,
    config: {},
  };
}

export function normalizeGraph(graph: GraphDefinition): GraphDefinition {
  return {
    ...graph,
    env_vars: getGraphEnvVars(graph),
    nodes: graph.nodes.map((node) => {
      const nextNode: GraphNode = {
        ...node,
        config: { ...node.config },
      };
      if (node.kind === "model") {
        nextNode.model_provider_name = String(node.config.provider_name ?? node.model_provider_name ?? "");
        nextNode.prompt_name = String(node.config.prompt_name ?? node.prompt_name ?? "");
      }
      if (node.kind === "provider") {
        nextNode.model_provider_name = String(node.config.provider_name ?? node.model_provider_name ?? "");
      }
      if (node.kind === "tool") {
        nextNode.tool_name = String(node.config.tool_name ?? node.tool_name ?? "");
      }
      return nextNode;
    }),
    edges: graph.edges.map((edge) => ({
      ...edge,
      condition: edge.kind === "conditional" ? edge.condition ?? defaultConditionalCondition(edge.id) : null,
    })),
  };
}

export function defaultConditionalCondition(edgeId: string): GraphEdge["condition"] {
  return {
    id: `${edgeId}-condition`,
    label: "Validation error",
    type: "result_status_equals",
    value: "validation_error",
    path: null,
  };
}

export function canConnectNodes(
  sourceNode: GraphNode | undefined,
  targetNode: GraphNode | undefined,
  catalog: EditorCatalog | null,
): boolean {
  if (!sourceNode || !targetNode || !catalog) {
    return false;
  }
  if (sourceNode.id === targetNode.id) {
    return false;
  }
  return catalog.connection_rules.some(
    (rule) => rule.source_category === sourceNode.category && rule.target_category === targetNode.category,
  );
}

const NODE_WIDTH = 280;
const NODE_HEIGHT = 150;

export function layoutGraphLR(graph: GraphDefinition): GraphDefinition {
  if (graph.nodes.length === 0) {
    return graph;
  }

  const g = new dagre.graphlib.Graph();
  g.setGraph({ rankdir: "LR", nodesep: 60, ranksep: 120, marginx: 40, marginy: 40 });
  g.setDefaultEdgeLabel(() => ({}));

  for (const node of graph.nodes) {
    g.setNode(node.id, { width: NODE_WIDTH, height: NODE_HEIGHT });
  }
  for (const edge of graph.edges) {
    g.setEdge(edge.source_id, edge.target_id);
  }

  dagre.layout(g);

  const positionMap = new Map<string, GraphPosition>();
  g.nodes().forEach((id) => {
    const info = g.node(id);
    if (info) {
      positionMap.set(id, { x: info.x - NODE_WIDTH / 2, y: info.y - NODE_HEIGHT / 2 });
    }
  });

  return {
    ...graph,
    nodes: graph.nodes.map((node) => ({
      ...node,
      position: positionMap.get(node.id) ?? node.position,
    })),
  };
}
