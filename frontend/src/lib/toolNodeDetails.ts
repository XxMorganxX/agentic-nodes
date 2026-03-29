import { getGraphEnvVars, resolveGraphEnvReferences } from "./graphEnv";
import type { EditorCatalog, GraphDefinition, GraphNode, ToolDefinition } from "./types";

export const TOOL_MODEL_TEMPLATE_DEFAULT = `Tool: {tool_name}
Description:
{tool_agent_description}

Schema:
{tool_schema}`;

export const TOOL_USER_DESCRIPTION_CONFIG_KEY = "tool_user_description";
export const TOOL_AGENT_DESCRIPTION_CONFIG_KEY = "tool_agent_description";
export const TOOL_SCHEMA_TEXT_CONFIG_KEY = "tool_model_schema_text";
export const TOOL_TEMPLATE_CONFIG_KEY = "tool_model_template";
export const TOOL_DESCRIPTION_CONFIG_KEY = TOOL_AGENT_DESCRIPTION_CONFIG_KEY;
const LEGACY_TOOL_DESCRIPTION_CONFIG_KEY = "tool_model_description";

type ToolNodeConfigValue = string | null;

export type ResolvedToolNodeDetails = {
  toolName: string;
  tool: ToolDefinition | null;
  userDescriptionText: string;
  agentDescriptionText: string;
  schemaText: string;
  templateText: string;
  renderedPromptText: string;
  parsedSchema: Record<string, unknown> | null;
  schemaError: string | null;
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function getConfigString(node: GraphNode, key: string): ToolNodeConfigValue {
  const value = node.config[key];
  return typeof value === "string" && value.trim().length > 0 ? value : null;
}

function safeJsonParse(value: string): { parsedSchema: Record<string, unknown> | null; schemaError: string | null } {
  try {
    const parsed = JSON.parse(value) as unknown;
    return {
      parsedSchema: isRecord(parsed) ? parsed : null,
      schemaError: isRecord(parsed) ? null : "Schema JSON must be an object.",
    };
  } catch (error) {
    return {
      parsedSchema: null,
      schemaError: error instanceof Error ? error.message : "Schema JSON is invalid.",
    };
  }
}

function renderTemplate(template: string, variables: Record<string, string>): string {
  return template.replace(/\{([A-Za-z_][A-Za-z0-9_]*)\}/g, (match, key: string) => variables[key] ?? match);
}

export function resolveToolNodeDetails(
  node: GraphNode,
  catalog: EditorCatalog | null,
  graph: GraphDefinition | null,
): ResolvedToolNodeDetails {
  const toolName =
    (typeof node.config.tool_name === "string" && node.config.tool_name) || node.tool_name || "";
  const tool = catalog?.tools.find((candidate) => candidate.name === toolName) ?? null;
  const fallbackDescription = tool?.description ?? node.description ?? "";
  const fallbackSchemaText = JSON.stringify(tool?.input_schema ?? {}, null, 2);
  const legacyDescriptionText = getConfigString(node, LEGACY_TOOL_DESCRIPTION_CONFIG_KEY);
  const envVars = getGraphEnvVars(graph);
  const userDescriptionText = resolveGraphEnvReferences(
    getConfigString(node, TOOL_USER_DESCRIPTION_CONFIG_KEY) ?? fallbackDescription,
    graph,
  );
  const agentDescriptionText = resolveGraphEnvReferences(
    getConfigString(node, TOOL_AGENT_DESCRIPTION_CONFIG_KEY) ?? legacyDescriptionText ?? fallbackDescription,
    graph,
  );
  const schemaText = resolveGraphEnvReferences(getConfigString(node, TOOL_SCHEMA_TEXT_CONFIG_KEY) ?? fallbackSchemaText, graph);
  const templateText = resolveGraphEnvReferences(getConfigString(node, TOOL_TEMPLATE_CONFIG_KEY) ?? TOOL_MODEL_TEMPLATE_DEFAULT, graph);
  const { parsedSchema, schemaError } = safeJsonParse(schemaText);
  const renderedPromptText = renderTemplate(templateText, {
    ...envVars,
    tool_name: toolName || "unnamed_tool",
    tool_user_description: userDescriptionText,
    tool_agent_description: agentDescriptionText,
    tool_description: agentDescriptionText,
    tool_schema: schemaText,
  });

  return {
    toolName,
    tool,
    userDescriptionText,
    agentDescriptionText,
    schemaText,
    templateText,
    renderedPromptText,
    parsedSchema,
    schemaError,
  };
}
