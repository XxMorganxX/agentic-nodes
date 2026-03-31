import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { AgentRunSwimlanes } from "./components/AgentRunSwimlanes";
import { EnvironmentRunSummary } from "./components/EnvironmentRunSummary";
import { GraphCanvas } from "./components/GraphCanvas";
import { GraphEnvEditor } from "./components/GraphEnvEditor";
import { McpServerPanel } from "./components/McpServerPanel";
import {
  bootMcpServer,
  createGraph,
  deleteGraph,
  eventStreamUrl,
  fetchEditorCatalog,
  fetchGraph,
  fetchGraphs,
  refreshMcpServer,
  setMcpToolEnabled,
  startRun,
  stopMcpServer,
  updateGraph,
} from "./lib/api";
import { createBlankGraph, layoutGraphDocument, layoutGraphLR, normalizeGraphDocument } from "./lib/editor";
import { filterEventsForAgent, getCanvasGraph, getDefaultAgentId, getSelectedRunId, getSelectedRunState, isTestEnvironment, updateSelectedAgentGraph } from "./lib/graphDocuments";
import { buildAgentRunLanes, buildEnvironmentRunSummary, buildFocusedEventGroups, buildFocusedRunSummary } from "./lib/runVisualization";
import type { EditorCatalog, GraphDefinition, GraphDocument, RunState, RuntimeEvent } from "./lib/types";
import { useGraphHistory } from "./lib/useGraphHistory";

const DEFAULT_INPUT = "Find graph-agent references for a schema repair workflow.";
const DEFAULT_TEST_ENVIRONMENT_ID = "test-environment";

function createEmptyRunState(runId: string, graphId: string, input: string): RunState {
  return {
    run_id: runId,
    graph_id: graphId,
    agent_id: null,
    agent_name: null,
    parent_run_id: null,
    current_node_id: null,
    status: "queued",
    started_at: null,
    ended_at: null,
    input_payload: input,
    node_outputs: {},
    node_errors: {},
    visit_counts: {},
    transition_history: [],
    event_history: [],
    final_output: null,
    terminal_error: null,
    agent_runs: {},
  };
}

function applySingleRunEvent(previous: RunState, event: RuntimeEvent): RunState {
  const next: RunState = {
    ...previous,
    event_history: [...previous.event_history, event],
  };

  if (event.event_type === "run.started") {
    next.status = "running";
    next.started_at = event.timestamp;
  }

  if (event.event_type === "node.started") {
    const payload = event.payload as { node_id: string; visit_count: number };
    next.current_node_id = payload.node_id;
    next.visit_counts = {
      ...next.visit_counts,
      [payload.node_id]: payload.visit_count,
    };
  }

  if (event.event_type === "node.completed") {
    const payload = event.payload as { node_id: string; output?: unknown; error?: unknown };
    if (payload.output !== undefined) {
      next.node_outputs = {
        ...next.node_outputs,
        [payload.node_id]: payload.output,
      };
    }
    if (payload.error !== undefined) {
      next.node_errors = {
        ...next.node_errors,
        [payload.node_id]: payload.error,
      };
    }
  }

  if (event.event_type === "edge.selected") {
    next.transition_history = [
      ...next.transition_history,
      {
        edge_id: event.payload.id,
        source_id: event.payload.source_id,
        target_id: event.payload.target_id,
        timestamp: event.timestamp,
      },
    ];
  }

  if (event.event_type === "run.completed") {
    next.status = "completed";
    next.ended_at = event.timestamp;
    next.final_output = event.payload.final_output;
  }

  if (event.event_type === "run.failed") {
    next.status = "failed";
    next.ended_at = event.timestamp;
    next.terminal_error = (event.payload.error ?? null) as Record<string, unknown> | null;
    if ("final_output" in event.payload) {
      next.final_output = event.payload.final_output;
    }
  }

  return next;
}

function applyEvent(previous: RunState | null, event: RuntimeEvent, graphId: string, input: string): RunState {
  const next = previous ?? createEmptyRunState(event.run_id, graphId, input);
  if (event.agent_id) {
    const agentId = event.agent_id;
    const payload = event.payload as { child_run_id?: string; agent_name?: string };
    const priorAgentRun =
      next.agent_runs?.[agentId] ??
      {
        ...createEmptyRunState(payload.child_run_id ?? `${event.run_id}:${agentId}`, graphId, input),
        agent_id: agentId,
        agent_name: payload.agent_name ?? agentId,
        parent_run_id: event.run_id,
      };
    const normalizedEvent: RuntimeEvent = {
      ...event,
      event_type: event.event_type.replace(/^agent\./, ""),
      run_id: priorAgentRun.run_id,
    };
    return {
      ...next,
      status: next.status === "queued" ? "running" : next.status,
      event_history: [...next.event_history, event],
      agent_runs: {
        ...(next.agent_runs ?? {}),
        [agentId]: applySingleRunEvent(priorAgentRun, normalizedEvent),
      },
    };
  }
  return applySingleRunEvent(next, event);
}

function pickDefaultGraphId(graphs: GraphDocument[]): string {
  return graphs.find((graph) => graph.graph_id === DEFAULT_TEST_ENVIRONMENT_ID)?.graph_id ?? graphs[0]?.graph_id ?? "";
}

function serializeGraphSnapshot(graph: GraphDocument | null): string {
  return graph ? JSON.stringify(normalizeGraphDocument(graph)) : "";
}

export default function App() {
  const [graphs, setGraphs] = useState<GraphDocument[]>([]);
  const [selectedGraphId, setSelectedGraphId] = useState<string>("");
  const [selectedAgentId, setSelectedAgentId] = useState<string | null>(null);
  const history = useGraphHistory();
  const { graph: draftGraph, set: setDraftGraph, setQuiet: setDraftGraphQuiet, reset: resetHistory } = history;
  const [savedGraphSnapshot, setSavedGraphSnapshot] = useState("");
  const [catalog, setCatalog] = useState<EditorCatalog | null>(null);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [selectedEdgeId, setSelectedEdgeId] = useState<string | null>(null);
  const [input, setInput] = useState(DEFAULT_INPUT);
  const [events, setEvents] = useState<RuntimeEvent[]>([]);
  const [runState, setRunState] = useState<RunState | null>(null);
  const [activeRunId, setActiveRunId] = useState<string | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [mcpPendingKey, setMcpPendingKey] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const sourceRef = useRef<EventSource | null>(null);
  const executionBoxRef = useRef<HTMLDivElement | null>(null);

  const canvasGraph = useMemo(() => getCanvasGraph(draftGraph, selectedAgentId), [draftGraph, selectedAgentId]);
  const selectedRunState = useMemo(() => getSelectedRunState(runState, selectedAgentId), [runState, selectedAgentId]);
  const selectedRunId = useMemo(() => getSelectedRunId(runState, activeRunId, selectedAgentId), [runState, activeRunId, selectedAgentId]);
  const filteredEvents = useMemo(() => filterEventsForAgent(events, selectedAgentId), [events, selectedAgentId]);
  const persistedGraphIds = useMemo(() => new Set(graphs.map((graph) => graph.graph_id)), [graphs]);
  const isEnvironment = isTestEnvironment(draftGraph);
  const environmentRunSummary = useMemo(
    () => buildEnvironmentRunSummary(draftGraph, runState, selectedAgentId),
    [draftGraph, runState, selectedAgentId],
  );
  const agentRunLanes = useMemo(() => buildAgentRunLanes(draftGraph, runState, events), [draftGraph, runState, events]);
  const focusedRunSummary = useMemo(
    () => buildFocusedRunSummary(canvasGraph, selectedRunState, filteredEvents),
    [canvasGraph, selectedRunState, filteredEvents],
  );
  const focusedEventGroups = useMemo(
    () => buildFocusedEventGroups(canvasGraph, filteredEvents),
    [canvasGraph, filteredEvents],
  );
  const draftGraphSnapshot = useMemo(() => serializeGraphSnapshot(draftGraph), [draftGraph]);
  const hasUnsavedChanges = Boolean(draftGraph) && draftGraphSnapshot !== savedGraphSnapshot;

  const refreshCatalog = useCallback(async () => {
    const loadedCatalog = await fetchEditorCatalog();
    setCatalog(loadedCatalog);
    return loadedCatalog;
  }, []);

  useEffect(() => {
    Promise.all([fetchGraphs(), refreshCatalog()])
      .then(([loadedGraphs, loadedCatalog]) => {
        setGraphs(loadedGraphs);
        setCatalog(loadedCatalog);
        if (loadedGraphs.length > 0) {
          setSelectedGraphId(pickDefaultGraphId(loadedGraphs));
        } else {
          const blankGraph = createBlankGraph();
          resetHistory(blankGraph);
          setSavedGraphSnapshot(serializeGraphSnapshot(blankGraph));
        }
      })
      .catch((loadError: Error) => {
        setError(loadError.message);
      });
  }, [refreshCatalog, resetHistory]);

  useEffect(() => {
    return () => {
      sourceRef.current?.close();
    };
  }, []);

  useEffect(() => {
    if (!selectedGraphId) {
      return;
    }
    fetchGraph(selectedGraphId)
      .then((graph) => {
        const nextGraph = layoutGraphDocument(graph);
        resetHistory(nextGraph);
        setSavedGraphSnapshot(serializeGraphSnapshot(nextGraph));
        setSelectedAgentId(getDefaultAgentId(nextGraph));
        setSelectedNodeId(null);
        setSelectedEdgeId(null);
      })
      .catch((loadError: Error) => setError(loadError.message));
  }, [resetHistory, selectedGraphId]);

  useEffect(() => {
    setSelectedNodeId(null);
    setSelectedEdgeId(null);
  }, [selectedAgentId]);

  useEffect(() => {
    if (!hasUnsavedChanges || isSaving) {
      return;
    }
    const handleBeforeUnload = (event: BeforeUnloadEvent) => {
      event.preventDefault();
      event.returnValue = "";
    };
    window.addEventListener("beforeunload", handleBeforeUnload);
    return () => {
      window.removeEventListener("beforeunload", handleBeforeUnload);
    };
  }, [hasUnsavedChanges, isSaving]);

  async function refreshGraphs(nextSelectedGraphId?: string) {
    const loadedGraphs = await fetchGraphs();
    setGraphs(loadedGraphs);
    if (nextSelectedGraphId) {
      setSelectedGraphId(nextSelectedGraphId);
    } else if (loadedGraphs.length === 0) {
      setSelectedGraphId("");
      resetHistory(createBlankGraph());
      setSelectedAgentId(null);
    }
  }

  async function runMcpAction(actionKey: string, callback: () => Promise<void>) {
    setMcpPendingKey(actionKey);
    setError(null);
    try {
      await callback();
      await refreshCatalog();
    } catch (actionError) {
      const message = actionError instanceof Error ? actionError.message : "Unable to update MCP state.";
      setError(message);
    } finally {
      setMcpPendingKey(null);
    }
  }

  async function saveCurrentGraph(): Promise<GraphDocument | null> {
    if (!draftGraph) {
      return null;
    }
    setIsSaving(true);
    setError(null);
    try {
      const normalized = normalizeGraphDocument(draftGraph);
      const savedGraph =
        selectedGraphId && persistedGraphIds.has(selectedGraphId)
          ? await updateGraph(selectedGraphId, normalized)
          : await createGraph(normalized);
      await refreshGraphs(savedGraph.graph_id);
      setSavedGraphSnapshot(serializeGraphSnapshot(savedGraph));
      setDraftGraph(savedGraph);
      if (isTestEnvironment(savedGraph)) {
        setSelectedAgentId((current) => current ?? getDefaultAgentId(savedGraph));
      }
      return savedGraph;
    } catch (saveError) {
      const message = saveError instanceof Error ? saveError.message : "Unable to save graph.";
      setError(message);
      return null;
    } finally {
      setIsSaving(false);
    }
  }

  function handleCreateGraph() {
    const blankGraph = createBlankGraph();
    sourceRef.current?.close();
    setSelectedGraphId("");
    setSelectedAgentId(null);
    resetHistory(blankGraph);
    setSavedGraphSnapshot(serializeGraphSnapshot(blankGraph));
    setSelectedNodeId(null);
    setSelectedEdgeId(null);
    setActiveRunId(null);
    setEvents([]);
    setRunState(null);
    setError(null);
  }

  async function handleDeleteGraph() {
    if (!selectedGraphId) {
      handleCreateGraph();
      return;
    }
    try {
      await deleteGraph(selectedGraphId);
      const loadedGraphs = await fetchGraphs();
      setGraphs(loadedGraphs);
      if (loadedGraphs.length > 0) {
        setSelectedGraphId(pickDefaultGraphId(loadedGraphs));
      } else {
        const blankGraph = createBlankGraph();
        setSelectedGraphId("");
        setSelectedAgentId(null);
        resetHistory(blankGraph);
        setSavedGraphSnapshot(serializeGraphSnapshot(blankGraph));
      }
      setSelectedNodeId(null);
      setSelectedEdgeId(null);
      setError(null);
    } catch (deleteError) {
      const message = deleteError instanceof Error ? deleteError.message : "Unable to delete graph.";
      setError(message);
    }
  }

  async function handleRun() {
    if (!draftGraph) {
      return;
    }

    const savedGraph = await saveCurrentGraph();
    if (!savedGraph) {
      return;
    }

    sourceRef.current?.close();
    setError(null);
    setEvents([]);
    setRunState(null);
    setIsRunning(true);

    try {
      const runId = await startRun(savedGraph.graph_id, input);
      setActiveRunId(runId);

      const source = new EventSource(eventStreamUrl(runId));
      sourceRef.current = source;

      source.onmessage = (message) => {
        const event = JSON.parse(message.data) as RuntimeEvent;
        setEvents((previous) => [...previous, event]);
        setRunState((previous) => applyEvent(previous, event, savedGraph.graph_id, input));

        if (!event.agent_id && (event.event_type === "run.completed" || event.event_type === "run.failed")) {
          source.close();
          setIsRunning(false);
        }
      };

      source.onerror = () => {
        source.close();
        setIsRunning(false);
      };
    } catch (runError) {
      const message = runError instanceof Error ? runError.message : "Unable to start run.";
      setError(message);
      setIsRunning(false);
    }
  }

  function scrollToExecutionBox() {
    executionBoxRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function handleCanvasGraphChange(nextGraph: GraphDefinition) {
    if (!canvasGraph || !draftGraph) {
      return;
    }
    setDraftGraph(updateSelectedAgentGraph(draftGraph, selectedAgentId, nextGraph));
  }

  function handleCanvasGraphDrag(nextGraph: GraphDefinition) {
    if (!canvasGraph || !draftGraph) {
      return;
    }
    setDraftGraphQuiet(updateSelectedAgentGraph(draftGraph, selectedAgentId, nextGraph));
  }

  return (
    <main className="app-shell">
      <div ref={executionBoxRef} className="hero-section">
        <div className="hero-mosaic">
          <div className="hero-main-column">
            <div className="hero-main-row">
              <div className="mosaic-tile panel mosaic-title">
                <h1>Graph Agent Studio</h1>
                <p>{isEnvironment ? "Compose a test environment with isolated agents and drill into each run." : "Drag nodes into the canvas, wire edges, and launch your agent."}</p>
                <div className="mosaic-title-actions">
                  <button type="button" className="secondary-button" onClick={handleCreateGraph}>
                    New Agent
                  </button>
                  <button type="button" className="secondary-button" onClick={() => void saveCurrentGraph()} disabled={!draftGraph || isSaving}>
                    {isSaving ? "Saving..." : "Save"}
                  </button>
                  <button type="button" className="secondary-button" onClick={history.undo} disabled={!history.canUndo} title="Undo (⌘Z)">
                    Undo
                  </button>
                  <button type="button" className="secondary-button" onClick={history.redo} disabled={!history.canRedo} title="Redo (⌘⇧Z)">
                    Redo
                  </button>
                  <button
                    type="button"
                    className="secondary-button"
                    onClick={() => canvasGraph && handleCanvasGraphChange(layoutGraphLR(canvasGraph))}
                    disabled={!canvasGraph || canvasGraph.nodes.length === 0}
                  >
                    Auto Layout
                  </button>
                  <button type="button" className="danger-button" onClick={() => void handleDeleteGraph()} disabled={!draftGraph}>
                    Delete
                  </button>
                </div>
              </div>

              <div className="mosaic-tile panel mosaic-graph">
                <label>
                  Graph
                  <select
                    value={selectedGraphId || "__draft__"}
                    onChange={(event) => {
                      if (event.target.value === "__draft__") {
                        handleCreateGraph();
                        return;
                      }
                      setSelectedGraphId(event.target.value);
                    }}
                  >
                    {!selectedGraphId ? <option value="__draft__">Unsaved Draft</option> : null}
                    {graphs.map((graph) => (
                      <option key={graph.graph_id} value={graph.graph_id}>
                        {graph.name}
                      </option>
                    ))}
                  </select>
                </label>
              </div>
            </div>

            <div className="mosaic-tile panel mosaic-execution">
              <label className="mosaic-execution-input">
                Input
                <textarea value={input} onChange={(event) => setInput(event.target.value)} rows={10} />
              </label>
              <div className="mosaic-execution-run">
                <button type="button" onClick={() => void handleRun()} disabled={!draftGraph || isRunning || isSaving}>
                  {isRunning ? "Running..." : isEnvironment ? "Run Environment" : "Run Graph"}
                </button>
                {selectedRunId ? <code>Run ID: {selectedRunId}</code> : <p>Ready to launch the selected graph.</p>}
                {error ? <p className="error-text">{error}</p> : null}
              </div>
            </div>
          </div>

          <div className="hero-side-column">
            <div className="mosaic-tile panel mosaic-env">
              <h2>Environment</h2>
              <GraphEnvEditor graph={draftGraph} onGraphChange={setDraftGraph} />
            </div>

            {(catalog?.mcp_servers?.length ?? 0) > 0 ? (
              <div className="mosaic-tile panel mosaic-mcp">
                <McpServerPanel
                  catalog={catalog}
                  onBootMcpServer={(serverId) => void runMcpAction(`boot:${serverId}`, () => bootMcpServer(serverId))}
                  onStopMcpServer={(serverId) => void runMcpAction(`stop:${serverId}`, () => stopMcpServer(serverId))}
                  onRefreshMcpServer={(serverId) => void runMcpAction(`refresh:${serverId}`, () => refreshMcpServer(serverId))}
                  onToggleMcpTool={(toolName, enabled) => void runMcpAction(`tool:${toolName}`, () => setMcpToolEnabled(toolName, enabled))}
                  mcpPendingKey={mcpPendingKey}
                  title="Project MCP"
                  description="Manage project-level MCP servers. Tool and model nodes can consume these tools, but they do not own the server lifecycle."
                />
              </div>
            ) : null}
          </div>
        </div>

        {environmentRunSummary ? <EnvironmentRunSummary summary={environmentRunSummary} /> : null}
      </div>

      <section className="content-grid">
        <GraphCanvas
          graph={canvasGraph}
          runState={selectedRunState}
          events={filteredEvents}
          activeRunId={selectedRunId}
          isRunning={isRunning}
          runButtonLabel={isEnvironment ? "Run Environment" : "Run Graph"}
          focusedAgentName={isEnvironment ? (environmentRunSummary?.focusedAgentName ?? null) : null}
          focusedAgentStatus={isEnvironment ? (selectedRunState?.status ?? "idle") : null}
          environmentAgents={isEnvironment ? agentRunLanes : []}
          selectedAgentId={selectedAgentId}
          onSelectAgent={(agentId) => {
            setSelectedAgentId(agentId);
            setSelectedNodeId(null);
            setSelectedEdgeId(null);
          }}
          runSummary={focusedRunSummary}
          eventGroups={focusedEventGroups}
          catalog={catalog}
          selectedNodeId={selectedNodeId}
          selectedEdgeId={selectedEdgeId}
          onGraphChange={handleCanvasGraphChange}
          onGraphDrag={handleCanvasGraphDrag}
          onRunGraph={() => void handleRun()}
          onScrollToTop={scrollToExecutionBox}
          onSelectionChange={(nodeId, edgeId) => {
            setSelectedNodeId(nodeId);
            setSelectedEdgeId(edgeId);
          }}
        />
      </section>

      {isTestEnvironment(draftGraph) ? (
        <AgentRunSwimlanes
          lanes={agentRunLanes}
          selectedAgentId={selectedAgentId}
          onSelectAgent={(agentId) => setSelectedAgentId(agentId)}
          onSelectNode={(agentId, nodeId) => {
            setSelectedAgentId(agentId);
            setSelectedNodeId(nodeId);
            setSelectedEdgeId(null);
          }}
        />
      ) : null}
    </main>
  );
}
