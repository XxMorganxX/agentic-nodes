import type { EditorCatalog, GraphDefinition, GraphNode, RunState } from "./types";
import { resolveToolNodeDetails } from "./toolNodeDetails";

type TooltipRow = {
  label: string;
  value: string;
};

export type TooltipSection = {
  title: string;
  rows: TooltipRow[];
};

export type TooltipParameter = {
  name: string;
  type: string;
  description?: string;
  required: boolean;
  source?: string;
};

export type NodeTooltipData = {
  title: string;
  eyebrow: string;
  description?: string;
  sections: TooltipSection[];
  parameters: TooltipParameter[];
  emptyState?: string;
};

type SchemaProperty = {
  type?: string | string[];
  description?: string;
  enum?: unknown[];
  items?: {
    type?: string | string[];
  };
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function asString(value: unknown): string | null {
  return typeof value === "string" && value.trim().length > 0 ? value : null;
}

function asStringArray(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : [];
}

function formatList(values: string[]): string {
  return values.length > 0 ? values.join(", ") : "None";
}

function describeResponseMode(value: string | null | undefined): string {
  if (value === "tool_call") {
    return "tool_call (always emit tool-call envelope)";
  }
  if (value === "auto") {
    return "auto (emit tool-call or message envelope)";
  }
  return "message (always emit message envelope)";
}

function truncate(value: string, limit = 96): string {
  if (value.length <= limit) {
    return value;
  }
  return `${value.slice(0, limit - 1)}...`;
}

function inferSchemaType(property: SchemaProperty): string {
  if (Array.isArray(property.type) && property.type.length > 0) {
    return property.type.join(" | ");
  }
  if (typeof property.type === "string") {
    if (property.type === "array" && property.items) {
      const itemType = Array.isArray(property.items.type)
        ? property.items.type.join(" | ")
        : property.items.type ?? "unknown";
      return `array<${itemType}>`;
    }
    return property.type;
  }
  if (Array.isArray(property.enum) && property.enum.length > 0) {
    return `enum(${property.enum.map((value) => JSON.stringify(value)).join(", ")})`;
  }
  return "unknown";
}

function findNode(graph: GraphDefinition | null | undefined, nodeId: string): GraphNode | null {
  if (!graph || !Array.isArray(graph.nodes)) {
    return null;
  }
  return graph.nodes.find((candidate) => candidate.id === nodeId) ?? null;
}

function formatNodeLabel(graph: GraphDefinition | null | undefined, nodeId: string): string {
  const sourceNode = findNode(graph, nodeId);
  if (!sourceNode) {
    return nodeId;
  }
  if (sourceNode.kind === "input") {
    return "user input";
  }
  return sourceNode.label;
}

function formatSourceList(sourceIds: string[], graph: GraphDefinition | null | undefined): string {
  const labels = sourceIds.map((sourceId) => formatNodeLabel(graph, sourceId));
  if (labels.length === 0) {
    return "upstream nodes";
  }
  if (labels.length === 1) {
    return labels[0];
  }
  if (labels.length === 2) {
    return `${labels[0]} or ${labels[1]}`;
  }
  return `${labels.slice(0, -1).join(", ")}, or ${labels[labels.length - 1]}`;
}

function describeBindingSource(node: GraphNode, graph: GraphDefinition | null | undefined): string {
  const binding = isRecord(node.config.input_binding) ? node.config.input_binding : null;
  if (!binding) {
    const incomingSourceIds = graph?.edges.filter((edge) => edge.target_id === node.id).map((edge) => edge.source_id) ?? [];
    if (incomingSourceIds.length === 0) {
      return "user input";
    }
    return `upstream from ${formatSourceList(incomingSourceIds, graph)}`;
  }

  const bindingType = asString(binding.type) ?? "latest_output";
  if (bindingType === "input_payload") {
    return "user input";
  }
  if (bindingType === "latest_output") {
    return `latest output from ${formatNodeLabel(graph, String(binding.source ?? ""))}`;
  }
  if (bindingType === "latest_payload") {
    return `latest payload from ${formatNodeLabel(graph, String(binding.source ?? ""))}`;
  }
  if (bindingType === "latest_envelope") {
    return `latest envelope from ${formatNodeLabel(graph, String(binding.source ?? ""))}`;
  }
  if (bindingType === "latest_error") {
    return `latest error from ${formatNodeLabel(graph, String(binding.source ?? ""))}`;
  }
  if (bindingType === "first_available_payload") {
    return `first available payload from ${formatSourceList(asStringArray(binding.sources), graph)}`;
  }
  if (bindingType === "first_available_envelope") {
    return `first available envelope from ${formatSourceList(asStringArray(binding.sources), graph)}`;
  }
  return bindingType.replace(/_/g, " ");
}

function summarizeToolParameters(inputSchema: Record<string, unknown>): {
  requiredNames: string[];
  parameters: TooltipParameter[];
} {
  const requiredNames = asStringArray(inputSchema.required);
  const properties = isRecord(inputSchema.properties) ? inputSchema.properties : {};

  const parameters = Object.entries(properties).map(([name, value]) => {
    const property = isRecord(value) ? (value as SchemaProperty) : {};
    return {
      name,
      type: inferSchemaType(property),
      description: asString(property.description) ?? undefined,
      required: requiredNames.includes(name),
    };
  });

  return { requiredNames, parameters };
}

function buildRuntimeSection(node: GraphNode, runState: RunState | null | undefined): TooltipSection | null {
  if (!runState) {
    return null;
  }

  const rows: TooltipRow[] = [];
  const isCurrent = runState.current_node_id === node.id;
  const hasOutput = Object.prototype.hasOwnProperty.call(runState.node_outputs ?? {}, node.id);
  const hasError = Object.prototype.hasOwnProperty.call(runState.node_errors ?? {}, node.id);
  const visitCount = runState.visit_counts?.[node.id];

  if (isCurrent) {
    rows.push({ label: "Status", value: "Currently running" });
  } else if (hasError) {
    rows.push({ label: "Status", value: "Completed with error" });
  } else if (hasOutput) {
    rows.push({ label: "Status", value: "Completed with output" });
  }

  if (typeof visitCount === "number") {
    rows.push({ label: "Visits", value: String(visitCount) });
  }

  return rows.length > 0 ? { title: "Runtime", rows } : null;
}

export function buildNodeTooltip(
  node: GraphNode,
  graph: GraphDefinition | null,
  catalog: EditorCatalog | null,
  runState?: RunState | null,
): NodeTooltipData {
  const contract = catalog?.contracts[node.category];
  const baseSections: TooltipSection[] = [
    {
      title: "Node",
      rows: [
        { label: "Provider", value: node.provider_label ?? node.provider_id },
        { label: "Category", value: node.category },
        { label: "Kind", value: node.kind },
      ],
    },
  ];

  if (contract) {
    baseSections.push({
      title: "Contract",
      rows: [
        { label: "Accepts", value: formatList(contract.accepted_inputs) },
        { label: "Produces", value: formatList(contract.produced_outputs) },
      ],
    });
  }

  const runtimeSection = buildRuntimeSection(node, runState);
  if (runtimeSection) {
    baseSections.push(runtimeSection);
  }

  if (node.kind === "tool") {
    const details = resolveToolNodeDetails(node, catalog, graph);
    const summary = details.parsedSchema
      ? summarizeToolParameters(details.parsedSchema)
      : { requiredNames: [], parameters: [] };
    const parameterSource = describeBindingSource(node, graph);

    return {
      title: node.label,
      eyebrow: `${node.category} / ${node.kind}`,
      description: details.userDescriptionText || node.description,
      sections: [
        {
          title: "Tool",
          rows: [
            { label: "Selected", value: details.toolName || "Not configured" },
            { label: "Agent Description", value: details.agentDescriptionText ? "Configured" : "Not configured" },
            { label: "Schema", value: details.schemaError ? "Invalid JSON override" : "Ready" },
          ],
        },
        ...baseSections,
        {
          title: "Schema",
          rows: [
            { label: "Required", value: formatList(summary.requiredNames) },
            { label: "Parameters", value: String(summary.parameters.length) },
          ],
        },
      ],
      parameters: summary.parameters.map((parameter) => ({
        ...parameter,
        source: parameterSource,
      })),
      emptyState: details.tool
        ? undefined
        : details.toolName
          ? "Tool metadata is unavailable in the current editor catalog."
          : "Select a tool to preview its description and parameter schema.",
    };
  }

  if (node.kind === "mcp_context_provider") {
    const toolNames = Array.isArray(node.config.tool_names) ? node.config.tool_names.map((toolName) => String(toolName)) : [];
    return {
      title: node.label,
      eyebrow: `${node.category} / ${node.kind}`,
      description: node.description,
      sections: [
        {
          title: "MCP Context",
          rows: [
            { label: "Registered Tools", value: toolNames.length > 0 ? formatList(toolNames) : "None selected" },
            { label: "Expose Callable Tools", value: node.config.expose_mcp_tools === false ? "Disabled" : "Enabled" },
            { label: "Inject Prompt Context", value: node.config.include_mcp_tool_context ? "Enabled" : "Disabled" },
          ],
        },
        ...baseSections,
      ],
      parameters: [],
      emptyState: toolNames.length > 0 ? undefined : "Select one or more MCP tools to expose through this context provider.",
    };
  }

  if (node.kind === "mcp_tool_executor") {
    return {
      title: node.label,
      eyebrow: `${node.category} / ${node.kind}`,
      description: node.description,
      sections: [
        {
          title: "MCP Execution",
          rows: [
            { label: "Dispatch", value: "Single tool call from upstream API output" },
            {
              label: "Input Binding",
              value:
                node.config.input_binding && typeof node.config.input_binding === "object"
                  ? "Explicit binding configured"
                  : "Implicit latest incoming edge",
            },
            { label: "Routes", value: "On success / On failure" },
          ],
        },
        ...baseSections,
      ],
      parameters: [],
    };
  }

  if (node.kind === "model") {
    const allowedTools = asStringArray(node.config.allowed_tool_names);
    const preferredTool = asString(node.config.preferred_tool_name);
    return {
      title: node.label,
      eyebrow: `${node.category} / ${node.kind}`,
      description: node.description,
      sections: [
        {
          title: "Configuration",
          rows: [
            { label: "Provider", value: asString(node.config.provider_name) ?? node.model_provider_name ?? "Not set" },
            { label: "Model", value: asString(node.config.model) ?? "Default" },
            { label: "Prompt", value: asString(node.config.prompt_name) ?? node.prompt_name ?? "Not set" },
            { label: "Response", value: describeResponseMode(asString(node.config.response_mode)) },
            { label: "Tool Call Output", value: "Routes structured tool-call envelopes to tool nodes" },
            { label: "Message Output", value: "Routes normal assistant/message envelopes to api, data, or end nodes" },
            { label: "Allowed Tools", value: allowedTools.length > 0 ? formatList(allowedTools) : "None" },
            { label: "Preferred Tool", value: preferredTool ?? "None" },
          ],
        },
        ...baseSections,
      ],
      parameters: [],
    };
  }

  if (node.kind === "data") {
    const isDisplayNode = node.provider_id === "core.data_display";
    return {
      title: node.label,
      eyebrow: `${node.category} / ${node.kind}`,
      description: node.description,
      sections: [
        {
          title: "Configuration",
          rows: [
            { label: "Mode", value: isDisplayNode ? "display envelope passthrough" : (asString(node.config.mode) ?? "passthrough") },
            { label: "Template", value: isDisplayNode ? "Disabled for display-only provider" : truncate(asString(node.config.template) ?? "{input_payload}") },
          ],
        },
        ...baseSections,
      ],
      parameters: [],
    };
  }

  if (node.kind === "provider") {
    return {
      title: node.label,
      eyebrow: `${node.category} / ${node.kind}`,
      description: node.description,
      sections: [
        {
          title: "Provider",
          rows: [
            { label: "Provider", value: asString(node.config.provider_name) ?? node.model_provider_name ?? "Not set" },
            { label: "Model", value: asString(node.config.model) ?? "Default" },
            { label: "API Base", value: asString(node.config.api_base) ?? "Default" },
          ],
        },
        ...baseSections,
      ],
      parameters: [],
    };
  }

  if (node.kind === "input") {
    return {
      title: node.label,
      eyebrow: `${node.category} / ${node.kind}`,
      description: node.description,
      sections: [
        {
          title: "Configuration",
          rows: [{ label: "Binding", value: truncate(JSON.stringify(node.config.input_binding ?? "input_payload")) }],
        },
        ...baseSections,
      ],
      parameters: [],
    };
  }

  return {
    title: node.label,
    eyebrow: `${node.category} / ${node.kind}`,
    description: node.description,
    sections: [
      {
        title: "Configuration",
        rows: [{ label: "Description", value: node.description ?? "Final output node" }],
      },
      ...baseSections,
    ],
    parameters: [],
  };
}
