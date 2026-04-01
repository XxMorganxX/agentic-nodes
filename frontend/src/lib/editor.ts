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

function uniquePromptName(graph: GraphDefinition, base: string): string {
  const existingPromptNames = new Set(
    graph.nodes
      .map((node) => (typeof node.prompt_name === "string" ? node.prompt_name.trim() : ""))
      .filter((promptName) => promptName.length > 0),
  );
  const trimmedBase = base.trim() || "prompt";
  let candidate = trimmedBase;
  let suffix = 1;
  while (existingPromptNames.has(candidate)) {
    candidate = `${trimmedBase}-${suffix}`;
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
    default_input: "",
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

  if (provider.node_kind === "mcp_context_provider") {
    const mcpToolNames = catalog.tools.filter((tool) => tool.source_type === "mcp").map((tool) => tool.name);
    return {
      ...baseNode,
      config: {
        tool_names: mcpToolNames.length > 0 ? [mcpToolNames[0]] : [],
        expose_mcp_tools: true,
        include_mcp_tool_context: false,
      },
    };
  }

  if (provider.node_kind === "mcp_tool_executor") {
    const defaultConfig = provider.default_config && typeof provider.default_config === "object" ? provider.default_config : {};
    return {
      ...baseNode,
      config: {
        ...defaultConfig,
      },
    };
  }

  if (provider.node_kind === "data") {
    const defaultConfig = provider.default_config && typeof provider.default_config === "object" ? provider.default_config : {};
    if (provider.provider_id === "core.context_builder") {
      return {
        ...baseNode,
        config: {
          template: "",
          input_bindings: [],
          joiner: "\n\n",
          ...defaultConfig,
        },
      };
    }
    if (provider.provider_id === "core.prompt_block") {
      return {
        ...baseNode,
        config: {
          mode: "prompt_block",
          role: "user",
          content: "",
          name: "",
          ...defaultConfig,
        },
      };
    }
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

export function duplicateGraphNode(
  graph: GraphDefinition,
  node: GraphNode,
  options: { position?: GraphPosition } = {},
): GraphNode {
  const id = uniqueNodeId(graph, node.label);
  const nextConfig = JSON.parse(JSON.stringify(node.config ?? {})) as GraphNode["config"];
  const nextNode: GraphNode = {
    ...node,
    id,
    position: options.position ?? { ...node.position },
    config: nextConfig,
  };

  if (node.kind === "model") {
    const promptName = uniquePromptName(graph, String(node.prompt_name ?? `${node.label}-prompt`));
    nextNode.prompt_name = promptName;
    nextNode.config = {
      ...nextConfig,
      prompt_name: promptName,
    };
  }

  return nextNode;
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

export function isRoutableToolNode(node: GraphNode | null | undefined): boolean {
  return Boolean(node && (node.kind === "tool" || node.kind === "mcp_tool_executor"));
}

export function isMcpContextProviderNode(node: GraphNode | null | undefined): boolean {
  return Boolean(node && node.kind === "mcp_context_provider");
}

export function isPromptBlockNode(node: GraphNode | null | undefined): boolean {
  return Boolean(node && node.kind === "data" && node.provider_id === "core.prompt_block");
}

export function isApiModelNode(node: GraphNode | null | undefined): boolean {
  return Boolean(node && node.kind === "model");
}

export const TOOL_SUCCESS_HANDLE_ID = "tool-success";
export const TOOL_FAILURE_HANDLE_ID = "tool-failure";
export const TOOL_CONTEXT_HANDLE_ID = "tool-context";
export const API_TOOL_CONTEXT_HANDLE_ID = "api-tool-context";
export const API_TOOL_CALL_HANDLE_ID = "api-tool-call";
export const API_FINAL_MESSAGE_HANDLE_ID = "api-message";
export const API_MESSAGE_HANDLE_ID = API_FINAL_MESSAGE_HANDLE_ID;
export const MCP_TERMINAL_OUTPUT_HANDLE_ID = "mcp-terminal-output";
export const PROMPT_BLOCK_PROVIDER_ID = "core.prompt_block";

export function defaultToolFailureCondition(edgeId: string): GraphEdge["condition"] {
  return {
    id: `${edgeId}-condition`,
    label: "On failure",
    type: "result_has_error",
    value: true,
    path: null,
  };
}

export function defaultApiToolCallCondition(edgeId: string): GraphEdge["condition"] {
  return {
    id: `${edgeId}-condition`,
    label: "Tool calls output",
    type: "result_payload_path_equals",
    value: "tool_call_envelope",
    path: "metadata.contract",
  };
}

export function defaultApiMessageCondition(edgeId: string): GraphEdge["condition"] {
  return {
    id: `${edgeId}-condition`,
    label: "Message output",
    type: "result_payload_path_equals",
    value: "message_envelope",
    path: "metadata.contract",
  };
}

export function defaultMcpTerminalOutputCondition(edgeId: string): GraphEdge["condition"] {
  return {
    id: `${edgeId}-condition`,
    label: "Terminal output",
    type: "result_payload_path_equals",
    value: "terminal_output_envelope",
    path: "metadata.contract",
  };
}

export function isApiOutputHandleId(handleId: string | null | undefined): boolean {
  return handleId === API_TOOL_CALL_HANDLE_ID || handleId === API_FINAL_MESSAGE_HANDLE_ID;
}

function hasExposedMcpToolContext(graph: GraphDefinition, node: GraphNode): boolean {
  const candidateNodeIds = new Set<string>();
  const configuredTargetIds = Array.isArray(node.config.tool_target_node_ids)
    ? node.config.tool_target_node_ids.map((nodeId) => String(nodeId))
    : [];
  configuredTargetIds.forEach((nodeId) => candidateNodeIds.add(nodeId));
  graph.edges
    .filter((edge) => edge.kind === "binding" && edge.target_id === node.id)
    .forEach((edge) => candidateNodeIds.add(edge.source_id));
  return [...candidateNodeIds]
    .map((nodeId) => graph.nodes.find((candidate) => candidate.id === nodeId) ?? null)
    .some(
      (candidate) =>
        candidate?.kind === "mcp_context_provider" &&
        candidate.config.expose_mcp_tools !== false &&
        Array.isArray(candidate.config.tool_names) &&
        candidate.config.tool_names.some((toolName) => String(toolName).trim().length > 0),
    );
}

function hasToolOutputRoute(graph: GraphDefinition, node: GraphNode): boolean {
  return graph.edges.some((edge) => {
    if (edge.kind === "binding" || edge.source_id !== node.id) {
      return false;
    }
    if (edge.source_handle_id === API_TOOL_CALL_HANDLE_ID) {
      return true;
    }
    if (edge.source_handle_id === API_FINAL_MESSAGE_HANDLE_ID) {
      return false;
    }
    const targetNode = graph.nodes.find((candidate) => candidate.id === edge.target_id);
    if (!targetNode) {
      return false;
    }
    return (
      targetNode.category === "tool" ||
      (edge.condition?.type === "result_payload_path_equals" &&
        edge.condition.path === "metadata.contract" &&
        edge.condition.value === "tool_call_envelope")
    );
  });
}

function hasMessageOutputRoute(graph: GraphDefinition, node: GraphNode): boolean {
  return graph.edges.some((edge) => {
    if (edge.kind === "binding" || edge.source_id !== node.id) {
      return false;
    }
    if (edge.source_handle_id === API_FINAL_MESSAGE_HANDLE_ID) {
      return true;
    }
    if (edge.source_handle_id === API_TOOL_CALL_HANDLE_ID) {
      return false;
    }
    const targetNode = graph.nodes.find((candidate) => candidate.id === edge.target_id);
    if (!targetNode) {
      return false;
    }
    if (
      edge.condition?.type === "result_payload_path_equals" &&
      edge.condition.path === "metadata.contract" &&
      edge.condition.value === "message_envelope"
    ) {
      return true;
    }
    return targetNode.category === "api" || targetNode.category === "data" || targetNode.category === "end";
  });
}

export function inferModelResponseMode(graph: GraphDefinition | null, node: GraphNode | null | undefined): "message" | "tool_call" | "auto" {
  if (!graph || !node || node.kind !== "model") {
    return "message";
  }
  const configuredMode = String(node.config.response_mode ?? "").trim();
  if (configuredMode === "message" || configuredMode === "tool_call" || configuredMode === "auto") {
    return configuredMode;
  }
  const toolOutputRoute = hasToolOutputRoute(graph, node);
  const messageOutputRoute = hasMessageOutputRoute(graph, node);
  if (toolOutputRoute && messageOutputRoute) {
    return "auto";
  }
  if (toolOutputRoute) {
    return "tool_call";
  }
  if (hasExposedMcpToolContext(graph, node) && !messageOutputRoute) {
    return "tool_call";
  }
  return "message";
}

export function inferToolEdgeSourceHandle(edge: GraphEdge, sourceNode: GraphNode | null | undefined): string | null {
  if (!sourceNode) {
    return edge.source_handle_id ?? null;
  }
  if (isApiModelNode(sourceNode)) {
    if (edge.source_handle_id === API_TOOL_CALL_HANDLE_ID || edge.source_handle_id === API_FINAL_MESSAGE_HANDLE_ID) {
      return edge.source_handle_id;
    }
    if (
      edge.condition?.type === "result_payload_path_equals" &&
      edge.condition.path === "metadata.contract" &&
      edge.condition.value === "tool_call_envelope"
    ) {
      return API_TOOL_CALL_HANDLE_ID;
    }
    if (
      edge.condition?.type === "result_payload_path_equals" &&
      edge.condition.path === "metadata.contract" &&
      edge.condition.value === "message_envelope"
    ) {
      return API_FINAL_MESSAGE_HANDLE_ID;
    }
    return edge.source_handle_id ?? null;
  }
  if (isMcpContextProviderNode(sourceNode)) {
    return TOOL_CONTEXT_HANDLE_ID;
  }
  if (!isRoutableToolNode(sourceNode)) {
    return edge.source_handle_id ?? null;
  }
  if (
    edge.source_handle_id === TOOL_SUCCESS_HANDLE_ID ||
    edge.source_handle_id === TOOL_FAILURE_HANDLE_ID ||
    (sourceNode.kind === "mcp_tool_executor" && edge.source_handle_id === MCP_TERMINAL_OUTPUT_HANDLE_ID)
  ) {
    return edge.source_handle_id;
  }
  return edge.kind === "conditional" ? TOOL_FAILURE_HANDLE_ID : TOOL_SUCCESS_HANDLE_ID;
}

export function getToolSourceHandleAnchorRatio(handleId: string | null | undefined): number {
  if (handleId === API_TOOL_CALL_HANDLE_ID) {
    return 0.4;
  }
  if (handleId === API_FINAL_MESSAGE_HANDLE_ID) {
    return 0.68;
  }
  if (handleId === TOOL_CONTEXT_HANDLE_ID) {
    return 0.86;
  }
  if (handleId === MCP_TERMINAL_OUTPUT_HANDLE_ID) {
    return 0.86;
  }
  return handleId === TOOL_FAILURE_HANDLE_ID ? 0.68 : 0.4;
}

export function getApiToolContextTargetAnchorRatio(handleId: string | null | undefined): number {
  return handleId === API_TOOL_CONTEXT_HANDLE_ID ? 0.82 : 0.26;
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
      if (node.kind === "tool" && nextNode.config.include_mcp_tool_context === true) {
        const legacyToolName = String(node.config.tool_name ?? node.tool_name ?? "").trim();
        nextNode.kind = "mcp_context_provider";
        nextNode.provider_id = "tool.mcp_context_provider";
        nextNode.provider_label = "MCP Context Provider";
        nextNode.config.tool_names = legacyToolName ? [legacyToolName] : [];
      }
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
      if (node.kind === "mcp_context_provider") {
        const toolNames = Array.isArray(node.config.tool_names) ? node.config.tool_names : [];
        nextNode.config.tool_names = toolNames.map((toolName) => String(toolName)).filter((toolName) => toolName.trim().length > 0);
        nextNode.config.expose_mcp_tools = nextNode.config.expose_mcp_tools !== false;
      }
      return nextNode;
    }),
    edges: graph.edges.map((edge) => ({
      ...edge,
      source_handle_id: edge.source_handle_id ?? null,
      target_handle_id: edge.target_handle_id ?? null,
      waypoints: edge.waypoints?.map((waypoint) => ({ ...waypoint })),
      kind:
        edge.source_handle_id === API_TOOL_CALL_HANDLE_ID || edge.source_handle_id === API_FINAL_MESSAGE_HANDLE_ID
          ? "conditional"
          : edge.kind,
      condition:
        edge.source_handle_id === API_TOOL_CALL_HANDLE_ID
          ? edge.condition ?? defaultApiToolCallCondition(edge.id)
          : edge.source_handle_id === API_FINAL_MESSAGE_HANDLE_ID
            ? edge.condition ?? defaultApiMessageCondition(edge.id)
            : edge.source_handle_id === MCP_TERMINAL_OUTPUT_HANDLE_ID
              ? edge.condition ?? defaultMcpTerminalOutputCondition(edge.id)
            : edge.kind === "conditional"
              ? edge.condition ?? defaultConditionalCondition(edge.id)
              : null,
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

const NODE_WIDTH = 320;
const NODE_HEIGHT = 178;
const MCP_TOOL_EXECUTOR_NODE_HEIGHT = 244;
const AUTO_LAYOUT_VERTICAL_GAP = NODE_HEIGHT + 120;
const AUTO_LAYOUT_DEPTH_SPREAD_MULTIPLIER = 8;  // ADD to user prefernce config
const AUTO_LAYOUT_COLUMN_TOLERANCE = 120;

function getNodeHeight(node: GraphNode): number {
  return node.kind === "mcp_tool_executor" ? MCP_TOOL_EXECUTOR_NODE_HEIGHT : NODE_HEIGHT;
}

function getNodeTargetAnchorRatio(node: GraphNode, targetHandleId: string | null | undefined): number {
  return node.kind === "model" ? getApiToolContextTargetAnchorRatio(targetHandleId) : 0.5;
}

function getNodeSourceAnchorRatio(node: GraphNode, sourceHandleId: string | null | undefined): number {
  if (isApiModelNode(node) || isRoutableToolNode(node) || isMcpContextProviderNode(node)) {
    return getToolSourceHandleAnchorRatio(sourceHandleId);
  }
  return 0.5;
}

function average(values: number[]): number {
  if (values.length === 0) {
    return 0;
  }
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function normalizedDepthSlot(index: number, count: number): number {
  if (count <= 1) {
    return 0.5;
  }
  return (index + 1) / (count + 1);
}

function groupNodeIdsByColumn(centerMap: Map<string, GraphPosition>) {
  const columns: Array<{ x: number; nodeIds: string[] }> = [];
  [...centerMap.entries()]
    .sort((left, right) => left[1].x - right[1].x)
    .forEach(([nodeId, center]) => {
      const column = columns.find((candidate) => Math.abs(candidate.x - center.x) <= AUTO_LAYOUT_COLUMN_TOLERANCE);
      if (column) {
        const nextCount = column.nodeIds.length + 1;
        column.x = (column.x * column.nodeIds.length + center.x) / nextCount;
        column.nodeIds.push(nodeId);
        return;
      }
      columns.push({ x: center.x, nodeIds: [nodeId] });
    });
  return columns;
}

function compareOrderedNodes(
  left: { nodeId: string; currentY: number; desiredY: number },
  right: { nodeId: string; currentY: number; desiredY: number },
): number {
  const desiredDelta = left.desiredY - right.desiredY;
  if (Math.abs(desiredDelta) > 0.5) {
    return desiredDelta;
  }
  const currentDelta = left.currentY - right.currentY;
  if (Math.abs(currentDelta) > 0.5) {
    return currentDelta;
  }
  return left.nodeId.localeCompare(right.nodeId);
}

export function layoutGraphLR(graph: GraphDefinition): GraphDefinition {
  if (graph.nodes.length === 0) {
    return graph;
  }

  const g = new dagre.graphlib.Graph();
  g.setGraph({ rankdir: "LR", nodesep: 120, ranksep: 240, marginx: 96, marginy: 96 });
  g.setDefaultEdgeLabel(() => ({}));

  for (const node of graph.nodes) {
    g.setNode(node.id, { width: NODE_WIDTH, height: getNodeHeight(node) });
  }
  for (const edge of graph.edges) {
    g.setEdge(edge.source_id, edge.target_id);
  }

  dagre.layout(g);

  const nodeMap = new Map(graph.nodes.map((node) => [node.id, node]));
  const incomingEdgesByTarget = new Map<string, GraphEdge[]>();
  const outgoingEdgesBySource = new Map<string, GraphEdge[]>();
  for (const edge of graph.edges) {
    const existing = incomingEdgesByTarget.get(edge.target_id);
    if (existing) {
      existing.push(edge);
    } else {
      incomingEdgesByTarget.set(edge.target_id, [edge]);
    }
    const outgoing = outgoingEdgesBySource.get(edge.source_id);
    if (outgoing) {
      outgoing.push(edge);
    } else {
      outgoingEdgesBySource.set(edge.source_id, [edge]);
    }
  }

  const centerMap = new Map<string, GraphPosition>();
  g.nodes().forEach((id) => {
    const info = g.node(id);
    if (!info) {
      return;
    }
    centerMap.set(id, { x: info.x, y: info.y });
  });

  const allCenterYs = [...centerMap.values()].map((center) => center.y);
  const graphMinCenterY = Math.min(...allCenterYs);
  const graphMaxCenterY = Math.max(...allCenterYs);
  const graphCenterY = average(allCenterYs);
  const columns = groupNodeIdsByColumn(centerMap);
  const layoutColumns = columns.slice(1);

  function desiredCenterYForNode(nodeId: string): { nodeId: string; currentY: number; desiredY: number } | null {
    const node = nodeMap.get(nodeId);
    const center = centerMap.get(nodeId);
    if (!node || !center) {
      return null;
    }

    const anchorTargets: number[] = [];
    const targetOffset = (getNodeTargetAnchorRatio(node, null) - 0.5) * getNodeHeight(node);

    (incomingEdgesByTarget.get(nodeId) ?? []).forEach((edge) => {
      const sourceNode = nodeMap.get(edge.source_id);
      const sourceCenter = centerMap.get(edge.source_id);
      if (!sourceNode || !sourceCenter || sourceCenter.x >= center.x) {
        return;
      }
      const sourceHandleId = inferToolEdgeSourceHandle(edge, sourceNode);
      const sourceOffset = (getNodeSourceAnchorRatio(sourceNode, sourceHandleId) - 0.5) * getNodeHeight(sourceNode);
      const edgeTargetOffset =
        (getNodeTargetAnchorRatio(node, edge.target_handle_id ?? null) - 0.5) * getNodeHeight(node);
      anchorTargets.push(sourceCenter.y + sourceOffset - edgeTargetOffset);
    });

    (outgoingEdgesBySource.get(nodeId) ?? []).forEach((edge) => {
      const targetNode = nodeMap.get(edge.target_id);
      const targetCenter = centerMap.get(edge.target_id);
      if (!targetNode || !targetCenter || targetCenter.x <= center.x) {
        return;
      }
      const sourceHandleId = inferToolEdgeSourceHandle(edge, node);
      const sourceOffset = (getNodeSourceAnchorRatio(node, sourceHandleId) - 0.5) * getNodeHeight(node);
      const edgeTargetOffset =
        (getNodeTargetAnchorRatio(targetNode, edge.target_handle_id ?? null) - 0.5) * getNodeHeight(targetNode);
      anchorTargets.push(targetCenter.y + edgeTargetOffset - sourceOffset);
    });

    const desiredY = anchorTargets.length > 0 ? average(anchorTargets) * 0.9 + center.y * 0.1 : center.y + targetOffset * 0;
    return { nodeId, currentY: center.y, desiredY };
  }

  function placeColumn(nodeIds: string[]): void {
    if (nodeIds.length === 0) {
      return;
    }

    const orderedNodes = nodeIds.map((nodeId) => desiredCenterYForNode(nodeId)).filter(
      (entry): entry is { nodeId: string; currentY: number; desiredY: number } => entry !== null,
    );
    orderedNodes.sort(compareOrderedNodes);

    if (orderedNodes.length === 0) {
      return;
    }

    const minimumSpan =
      AUTO_LAYOUT_VERTICAL_GAP * AUTO_LAYOUT_DEPTH_SPREAD_MULTIPLIER * Math.max(1, orderedNodes.length - 1);
    const preferredSpan = Math.max(graphMaxCenterY - graphMinCenterY, minimumSpan);
    const columnCenterY = average(orderedNodes.map((entry) => entry.desiredY)) * 0.9 + graphCenterY * 0.1;
    const topY = columnCenterY - preferredSpan / 2;
    orderedNodes.forEach((entry, index) => {
      const current = centerMap.get(entry.nodeId);
      if (!current) {
        return;
      }
      const slotRatio = normalizedDepthSlot(index, orderedNodes.length);
      centerMap.set(entry.nodeId, {
        x: current.x,
        y: topY + preferredSpan * slotRatio,
      });
    });
  }

  for (let iteration = 0; iteration < 3; iteration += 1) {
    layoutColumns.forEach((column) => placeColumn(column.nodeIds));
    [...layoutColumns].reverse().forEach((column) => placeColumn(column.nodeIds));
  }

  const positionMap = new Map<string, GraphPosition>();
  centerMap.forEach((center, id) => {
    const node = nodeMap.get(id);
    positionMap.set(id, {
      x: center.x - NODE_WIDTH / 2,
      y: center.y - (node ? getNodeHeight(node) : NODE_HEIGHT) / 2,
    });
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
