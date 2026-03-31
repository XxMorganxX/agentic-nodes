import dagre from "dagre";

import { DEFAULT_GRAPH_ENV_VARS, getGraphEnvVars } from "./graphEnv";
import { isTestEnvironment } from "./graphDocuments";
import type { SavedNode } from "./savedNodes";
import type { AgentDefinition, EditorCatalog, GraphDefinition, GraphDocument, GraphEdge, GraphNode, GraphPosition, NodeProviderDefinition } from "./types";

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

export function providerModelName(provider: NodeProviderDefinition): string {
  const explicitName = typeof provider.model_provider_name === "string" ? provider.model_provider_name.trim() : "";
  if (explicitName) {
    return explicitName;
  }
  return provider.provider_id.replace(/^provider\./, "") || "mock";
}

export function modelProviderDefinitions(catalog: EditorCatalog | null): NodeProviderDefinition[] {
  return (catalog?.node_providers ?? []).filter((provider) => provider.category === "provider");
}

export function findProviderDefinition(catalog: EditorCatalog | null, providerName: string): NodeProviderDefinition | null {
  return modelProviderDefinitions(catalog).find((provider) => providerModelName(provider) === providerName) ?? null;
}

export function providerDefaultConfig(provider: NodeProviderDefinition): GraphNode["config"] {
  const defaultConfig = provider.default_config && typeof provider.default_config === "object" ? provider.default_config : {};
  return {
    provider_name: providerModelName(provider),
    ...defaultConfig,
  };
}

export function defaultModelName(providerName: string, catalog: EditorCatalog | null): string {
  const provider = findProviderDefinition(catalog, providerName);
  const defaultConfig = provider ? providerDefaultConfig(provider) : {};
  if (typeof defaultConfig.model === "string" && defaultConfig.model.trim()) {
    return defaultConfig.model.trim();
  }
  if (providerName === "mock") {
    return "mock-default";
  }
  return "";
}

function defaultModelConfig(promptName: string, catalog: EditorCatalog | null): GraphNode["config"] {
  const defaultProvider =
    findProviderDefinition(catalog, "mock") ?? modelProviderDefinitions(catalog)[0] ?? null;
  const providerName = defaultProvider ? providerModelName(defaultProvider) : "mock";
  const defaultConfig = defaultProvider ? providerDefaultConfig(defaultProvider) : { provider_name: "mock", model: "mock-default" };
  return {
    ...defaultConfig,
    provider_name: providerName,
    prompt_name: promptName,
    mode: promptName,
    system_prompt: "You are a model node in an editable graph.",
    user_message_template: "{input_payload}",
    response_mode: "message",
    allowed_tool_names: [],
    model: typeof defaultConfig.model === "string" ? defaultConfig.model : "mock-default",
  };
}

function defaultStartConfig(provider: NodeProviderDefinition): GraphNode["config"] {
  if (provider.provider_id === "start.discord_message") {
    return {
      trigger_mode: "discord_message",
      discord_bot_token_env_var: "{DISCORD_BOT_TOKEN}",
      discord_channel_id: "",
      ignore_bot_messages: true,
      ignore_self_messages: true,
      input_binding: { type: "input_payload" },
    };
  }
  return {
    trigger_mode: "manual_run",
    input_binding: { type: "input_payload" },
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
    ...providerNode.config,
    provider_binding_node_id: providerNode.id,
    provider_name: providerName,
  };

  const providerModel = typeof providerNode.config.model === "string" ? providerNode.config.model.trim() : "";
  nextConfig.model = providerModel || String(modelNode.config.model ?? "").trim() || "mock-default";

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
      config: defaultStartConfig(provider),
    };
  }

  if (provider.node_kind === "model") {
    const promptName = `${id}_prompt`;
    return {
      ...baseNode,
      model_provider_name: String(defaultModelConfig(promptName, catalog).provider_name ?? "mock"),
      prompt_name: promptName,
      config: defaultModelConfig(promptName, catalog),
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
    const defaultConfig = provider.default_config && typeof provider.default_config === "object" ? provider.default_config : {};
    return {
      ...baseNode,
      config: {
        mode: "passthrough",
        ...defaultConfig,
      },
    };
  }

  if (provider.node_kind === "provider") {
    const providerName = providerModelName(provider);
    return {
      ...baseNode,
      model_provider_name: providerName,
      config: providerDefaultConfig(provider),
    };
  }

  return {
    ...baseNode,
    config: {},
  };
}

export function createNodeFromSaved(
  graph: GraphDefinition,
  saved: SavedNode,
  position: GraphPosition,
): GraphNode {
  const id = uniqueNodeId(graph, saved.name);
  return {
    id,
    kind: saved.kind,
    category: saved.category,
    label: saved.name,
    description: saved.description || undefined,
    provider_id: saved.provider_id,
    provider_label: saved.provider_label,
    position,
    config: { ...saved.config },
    model_provider_name: saved.model_provider_name,
    prompt_name: saved.prompt_name,
    tool_name: saved.tool_name,
  };
}

export function createWireJunctionNode(graph: GraphDefinition, position: GraphPosition): GraphNode {
  const id = uniqueNodeId(graph, "wire-junction");
  return {
    id,
    kind: "data",
    category: "data",
    label: "Wire Point",
    description: "Floating wire routing point.",
    provider_id: "core.data",
    provider_label: "Core Data Node",
    position,
    config: {
      mode: "passthrough",
      is_wire_junction: true,
    },
  };
}

export function isWireJunctionNode(node: GraphNode | null | undefined): boolean {
  return Boolean(node && node.kind === "data" && node.config.is_wire_junction === true);
}

export const TOOL_SUCCESS_HANDLE_ID = "tool-success";
export const TOOL_FAILURE_HANDLE_ID = "tool-failure";

export function defaultToolFailureCondition(edgeId: string): GraphEdge["condition"] {
  return {
    id: `${edgeId}-condition`,
    label: "On failure",
    type: "result_has_error",
    value: true,
    path: null,
  };
}

export function inferToolEdgeSourceHandle(edge: GraphEdge, sourceNode: GraphNode | null | undefined): string | null {
  if (!sourceNode || sourceNode.kind !== "tool") {
    return edge.source_handle_id ?? null;
  }
  if (edge.source_handle_id === TOOL_SUCCESS_HANDLE_ID || edge.source_handle_id === TOOL_FAILURE_HANDLE_ID) {
    return edge.source_handle_id;
  }
  return edge.kind === "conditional" ? TOOL_FAILURE_HANDLE_ID : TOOL_SUCCESS_HANDLE_ID;
}

export function getToolSourceHandleAnchorRatio(handleId: string | null | undefined): number {
  return handleId === TOOL_FAILURE_HANDLE_ID ? 0.68 : 0.4;
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
      waypoints: edge.waypoints?.map((waypoint) => ({ ...waypoint })),
      condition: edge.kind === "conditional" ? edge.condition ?? defaultConditionalCondition(edge.id) : null,
    })),
  };
}

function normalizeAgent(agent: AgentDefinition): AgentDefinition {
  const normalizedGraph = normalizeGraph({
    graph_id: agent.agent_id,
    name: agent.name,
    description: agent.description,
    version: agent.version,
    graph_type: "graph",
    start_node_id: agent.start_node_id,
    env_vars: agent.env_vars,
    nodes: agent.nodes,
    edges: agent.edges,
  });
  return {
    agent_id: agent.agent_id,
    name: normalizedGraph.name,
    description: normalizedGraph.description,
    version: normalizedGraph.version,
    start_node_id: normalizedGraph.start_node_id,
    env_vars: normalizedGraph.env_vars,
    nodes: normalizedGraph.nodes,
    edges: normalizedGraph.edges,
  };
}

export function normalizeGraphDocument(graph: GraphDocument): GraphDocument {
  if (!isTestEnvironment(graph)) {
    return normalizeGraph(graph);
  }
  return {
    ...graph,
    agents: graph.agents.map((agent) => normalizeAgent(agent)),
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
  g.setGraph({ rankdir: "LR", nodesep: 120, ranksep: 240, marginx: 96, marginy: 96 });
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

export function layoutGraphDocument(graph: GraphDocument): GraphDocument {
  if (!isTestEnvironment(graph)) {
    return layoutGraphLR(graph);
  }
  return {
    ...graph,
    agents: graph.agents.map((agent) => {
      const laidOutGraph = layoutGraphLR({
        graph_id: agent.agent_id,
        name: agent.name,
        description: agent.description,
        version: agent.version,
        graph_type: "graph",
        start_node_id: agent.start_node_id,
        env_vars: agent.env_vars,
        nodes: agent.nodes,
        edges: agent.edges,
      });
      return {
        ...agent,
        start_node_id: laidOutGraph.start_node_id,
        nodes: laidOutGraph.nodes,
        edges: laidOutGraph.edges,
      };
    }),
  };
}
