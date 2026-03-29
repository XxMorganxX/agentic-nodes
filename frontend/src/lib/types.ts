export type GraphPosition = {
  x: number;
  y: number;
};

export type GraphNodeConfig = Record<string, unknown>;

export type GraphNode = {
  id: string;
  kind: string;
  category: string;
  label: string;
  description?: string;
  provider_id: string;
  provider_label: string;
  position: GraphPosition;
  config: GraphNodeConfig;
  model_provider_name?: string;
  prompt_name?: string;
  tool_name?: string;
};

export type GraphEdgeCondition = {
  id: string;
  label: string;
  type: string;
  value?: unknown;
  path?: string | null;
};

export type GraphEdge = {
  id: string;
  source_id: string;
  target_id: string;
  label: string;
  kind: string;
  priority: number;
  condition?: GraphEdgeCondition | null;
};

export type GraphDefinition = {
  graph_id: string;
  name: string;
  description: string;
  version: string;
  start_node_id: string;
  env_vars?: Record<string, string>;
  nodes: GraphNode[];
  edges: GraphEdge[];
  node_providers?: NodeProviderDefinition[];
};

export type NodeProviderDefinition = {
  provider_id: string;
  display_name: string;
  category: string;
  node_kind: string;
  description: string;
  capabilities: string[];
};

export type ToolDefinition = {
  name: string;
  description: string;
  input_schema: Record<string, unknown>;
};

export type ConnectionRule = {
  source_category: string;
  target_category: string;
  rationale: string;
};

export type CategoryContract = {
  category: string;
  accepted_inputs: string[];
  produced_outputs: string[];
  description: string;
};

export type EditorCatalog = {
  node_providers: NodeProviderDefinition[];
  tools: ToolDefinition[];
  connection_rules: ConnectionRule[];
  contracts: Record<string, CategoryContract>;
};

export type RuntimeEvent = {
  event_type: string;
  summary: string;
  payload: Record<string, unknown>;
  run_id: string;
  timestamp: string;
};

export type RunState = {
  run_id: string;
  graph_id: string;
  current_node_id: string | null;
  status: string;
  started_at: string | null;
  ended_at: string | null;
  input_payload: unknown;
  node_outputs: Record<string, unknown>;
  node_errors: Record<string, unknown>;
  visit_counts: Record<string, number>;
  transition_history: Array<Record<string, unknown>>;
  event_history: RuntimeEvent[];
  final_output: unknown;
  terminal_error: Record<string, unknown> | null;
};

export function cloneGraphDefinition(graph: GraphDefinition): GraphDefinition {
  return JSON.parse(JSON.stringify(graph)) as GraphDefinition;
}
