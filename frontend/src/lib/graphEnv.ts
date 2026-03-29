import type { GraphDefinition } from "./types";

export const DEFAULT_GRAPH_ENV_VARS: Record<string, string> = {
  OPENAI_API_KEY: "OPENAI_API_KEY",
  ANTHROPIC_API_KEY: "ANTHROPIC_API_KEY",
};

export const STANDARD_GRAPH_ENV_FIELDS = [
  {
    key: "OPENAI_API_KEY",
    label: "OpenAI API Key Reference",
    placeholder: "OPENAI_API_KEY",
  },
  {
    key: "ANTHROPIC_API_KEY",
    label: "Anthropic API Key Reference",
    placeholder: "ANTHROPIC_API_KEY",
  },
] as const;

const GRAPH_ENV_REFERENCE_PATTERN = /\{([A-Za-z_][A-Za-z0-9_]*)\}/g;

export function getGraphEnvVars(graph: GraphDefinition | null | undefined): Record<string, string> {
  const nextEnvVars: Record<string, string> = { ...DEFAULT_GRAPH_ENV_VARS };
  const rawEnvVars = graph?.env_vars;
  if (!rawEnvVars) {
    return nextEnvVars;
  }

  for (const [key, value] of Object.entries(rawEnvVars)) {
    const trimmedKey = key.trim();
    if (!trimmedKey) {
      continue;
    }
    nextEnvVars[trimmedKey] = typeof value === "string" ? value : String(value ?? "");
  }

  return nextEnvVars;
}

export function resolveGraphEnvReferences(
  value: string,
  graph: GraphDefinition | null | undefined,
  extraVariables: Record<string, string> = {},
): string {
  const variables = {
    ...getGraphEnvVars(graph),
    ...extraVariables,
  };

  return value.replace(GRAPH_ENV_REFERENCE_PATTERN, (match, key: string) => variables[key] ?? match);
}
