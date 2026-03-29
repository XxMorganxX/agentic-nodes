import type { EditorCatalog, GraphDefinition, RunState } from "./types";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

export async function fetchGraphs(): Promise<GraphDefinition[]> {
  const response = await fetch(`${API_BASE_URL}/api/graphs`);
  if (!response.ok) {
    throw new Error("Failed to load graph definitions.");
  }
  const payload = (await response.json()) as { graphs: GraphDefinition[] };
  return payload.graphs;
}

export async function fetchGraph(graphId: string): Promise<GraphDefinition> {
  const response = await fetch(`${API_BASE_URL}/api/graphs/${graphId}`);
  if (!response.ok) {
    throw new Error("Failed to load graph definition.");
  }
  return (await response.json()) as GraphDefinition;
}

export async function createGraph(graph: GraphDefinition): Promise<GraphDefinition> {
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
  return (await response.json()) as GraphDefinition;
}

export async function updateGraph(graphId: string, graph: GraphDefinition): Promise<GraphDefinition> {
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
  return (await response.json()) as GraphDefinition;
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
  const response = await fetch(`${API_BASE_URL}/api/editor/catalog`);
  if (!response.ok) {
    throw new Error("Failed to load editor catalog.");
  }
  return (await response.json()) as EditorCatalog;
}

export async function startRun(graphId: string, input: string): Promise<string> {
  const response = await fetch(`${API_BASE_URL}/api/graphs/${graphId}/runs`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ input }),
  });

  if (!response.ok) {
    throw new Error("Failed to start graph run.");
  }

  const payload = (await response.json()) as { run_id: string };
  return payload.run_id;
}

export async function fetchRun(runId: string): Promise<RunState> {
  const response = await fetch(`${API_BASE_URL}/api/runs/${runId}`);
  if (!response.ok) {
    throw new Error("Failed to load run state.");
  }
  return (await response.json()) as RunState;
}

export function eventStreamUrl(runId: string): string {
  return `${API_BASE_URL}/api/runs/${runId}/events`;
}
