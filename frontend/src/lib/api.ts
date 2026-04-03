import type {
  EditorCatalog,
  GraphDocument,
  McpServerDraft,
  McpServerStatus,
  McpServerTestResult,
  ProviderDiagnosticsResult,
  ProviderPreflightResult,
  RunState,
  StartRunOptions,
  ToolDefinition,
} from "./types";
import { normalizeRunState } from "./runtimeEvents";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

export async function fetchGraphs(): Promise<GraphDocument[]> {
  const response = await fetch(`${API_BASE_URL}/api/graphs`);
  if (!response.ok) {
    throw new Error("Failed to load graph definitions.");
  }
  const payload = (await response.json()) as { graphs: GraphDocument[] };
  return payload.graphs;
}

export async function fetchGraph(graphId: string): Promise<GraphDocument> {
  const response = await fetch(`${API_BASE_URL}/api/graphs/${graphId}`);
  if (!response.ok) {
    throw new Error("Failed to load graph definition.");
  }
  return (await response.json()) as GraphDocument;
}

export async function createGraph(graph: GraphDocument): Promise<GraphDocument> {
  const response = await fetch(`${API_BASE_URL}/api/graphs`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(graph),
  });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return (await response.json()) as GraphDocument;
}

export async function updateGraph(graphId: string, graph: GraphDocument): Promise<GraphDocument> {
  const response = await fetch(`${API_BASE_URL}/api/graphs/${graphId}`, {
    method: "PUT",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(graph),
  });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return (await response.json()) as GraphDocument;
}

export async function deleteGraph(graphId: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/api/graphs/${graphId}`, {
    method: "DELETE",
  });
  if (!response.ok) {
    throw new Error("Failed to delete graph.");
  }
}

export async function fetchEditorCatalog(): Promise<EditorCatalog> {
  const response = await fetch(`${API_BASE_URL}/api/editor/catalog`, {
    cache: "no-store",
  });
  if (!response.ok) {
    throw new Error("Failed to load editor catalog.");
  }
  return (await response.json()) as EditorCatalog;
}

export async function preflightProvider(
  providerName: string,
  providerConfig: Record<string, unknown>,
  live = false,
): Promise<ProviderPreflightResult> {
  const response = await fetch(`${API_BASE_URL}/api/editor/providers/preflight`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      provider_name: providerName,
      provider_config: providerConfig,
      live,
    }),
  });
  if (!response.ok) {
    throw new Error("Failed to preflight provider.");
  }
  return (await response.json()) as ProviderPreflightResult;
}

export async function fetchProviderDiagnostics(
  providerName: string,
  providerConfig: Record<string, unknown>,
  live = false,
): Promise<ProviderDiagnosticsResult> {
  const response = await fetch(`${API_BASE_URL}/api/editor/providers/diagnostics`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      provider_name: providerName,
      provider_config: providerConfig,
      live,
    }),
  });
  if (!response.ok) {
    throw new Error("Failed to load provider diagnostics.");
  }
  return (await response.json()) as ProviderDiagnosticsResult;
}

export async function bootMcpServer(serverId: string): Promise<McpServerStatus> {
  const response = await fetch(`${API_BASE_URL}/api/editor/mcp/servers/${serverId}/boot`, {
    method: "POST",
  });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return (await response.json()) as McpServerStatus;
}

export async function stopMcpServer(serverId: string): Promise<McpServerStatus> {
  const response = await fetch(`${API_BASE_URL}/api/editor/mcp/servers/${serverId}/stop`, {
    method: "POST",
  });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return (await response.json()) as McpServerStatus;
}

export async function refreshMcpServer(serverId: string): Promise<McpServerStatus> {
  const response = await fetch(`${API_BASE_URL}/api/editor/mcp/servers/${serverId}/refresh`, {
    method: "POST",
  });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return (await response.json()) as McpServerStatus;
}

export async function createMcpServer(server: McpServerDraft): Promise<McpServerStatus> {
  const response = await fetch(`${API_BASE_URL}/api/editor/mcp/servers`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(server),
  });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return (await response.json()) as McpServerStatus;
}

export async function updateMcpServer(serverId: string, server: McpServerDraft): Promise<McpServerStatus> {
  const response = await fetch(`${API_BASE_URL}/api/editor/mcp/servers/${serverId}`, {
    method: "PUT",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(server),
  });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return (await response.json()) as McpServerStatus;
}

export async function deleteMcpServer(serverId: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/api/editor/mcp/servers/${serverId}`, {
    method: "DELETE",
  });
  if (!response.ok) {
    throw new Error(await response.text());
  }
}

export async function testMcpServer(server: McpServerDraft): Promise<McpServerTestResult> {
  const response = await fetch(`${API_BASE_URL}/api/editor/mcp/servers/test`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(server),
  });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return (await response.json()) as McpServerTestResult;
}

export async function setMcpToolEnabled(toolName: string, enabled: boolean): Promise<ToolDefinition> {
  const response = await fetch(`${API_BASE_URL}/api/editor/mcp/tools/${encodeURIComponent(toolName)}/toggle`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ enabled }),
  });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return (await response.json()) as ToolDefinition;
}

export async function startRun(graphId: string, input: string, options?: StartRunOptions): Promise<string> {
  const response = await fetch(`${API_BASE_URL}/api/graphs/${graphId}/runs`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      input,
      agent_ids: options?.agent_ids,
    }),
  });

  if (!response.ok) {
    throw new Error("Failed to start graph run.");
  }

  const payload = (await response.json()) as { run_id: string };
  return payload.run_id;
}

export async function resetRuntime(): Promise<{
  cancelled_run_ids: string[];
  cancelled_run_count: number;
  stopped_mcp_server_ids: string[];
  stopped_mcp_server_count: number;
  discord_stopped: boolean;
}> {
  const response = await fetch(`${API_BASE_URL}/api/runtime/reset`, {
    method: "POST",
  });
  if (!response.ok) {
    throw new Error("Failed to reset runtime.");
  }
  return (await response.json()) as {
    cancelled_run_ids: string[];
    cancelled_run_count: number;
    stopped_mcp_server_ids: string[];
    stopped_mcp_server_count: number;
    discord_stopped: boolean;
  };
}

export async function fetchRun(runId: string): Promise<RunState> {
  const response = await fetch(`${API_BASE_URL}/api/runs/${runId}`);
  if (!response.ok) {
    throw new Error("Failed to load run state.");
  }
  return normalizeRunState((await response.json()) as RunState) as RunState;
}

export function eventStreamUrl(runId: string): string {
  return `${API_BASE_URL}/api/runs/${runId}/events`;
}
