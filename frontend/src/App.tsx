import { useEffect, useMemo, useRef, useState } from "react";

import { GraphCanvas } from "./components/GraphCanvas";
import { GraphEnvEditor } from "./components/GraphEnvEditor";
import {
  createGraph,
  deleteGraph,
  eventStreamUrl,
  fetchEditorCatalog,
  fetchGraph,
  fetchGraphs,
  fetchRun,
  startRun,
  updateGraph,
} from "./lib/api";
import { createBlankGraph, layoutGraphLR, normalizeGraph } from "./lib/editor";
import type { EditorCatalog, GraphDefinition, RunState, RuntimeEvent } from "./lib/types";
import { useGraphHistory } from "./lib/useGraphHistory";

const DEFAULT_INPUT = "Find graph-agent references for a schema repair workflow.";

function applyEvent(previous: RunState | null, event: RuntimeEvent, graphId: string, input: string): RunState {
  const next: RunState = previous ?? {
    run_id: event.run_id,
    graph_id: graphId,
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
  };

  next.event_history = [...next.event_history, event];

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
    next.terminal_error = event.payload.error as Record<string, unknown>;
  }

  return { ...next };
}

export default function App() {
  const [graphs, setGraphs] = useState<GraphDefinition[]>([]);
  const [selectedGraphId, setSelectedGraphId] = useState<string>("");
  const history = useGraphHistory();
  const draftGraph = history.graph;
  const setDraftGraph = history.set;
  const [catalog, setCatalog] = useState<EditorCatalog | null>(null);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [selectedEdgeId, setSelectedEdgeId] = useState<string | null>(null);
  const [input, setInput] = useState(DEFAULT_INPUT);
  const [events, setEvents] = useState<RuntimeEvent[]>([]);
  const [runState, setRunState] = useState<RunState | null>(null);
  const [activeRunId, setActiveRunId] = useState<string | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const sourceRef = useRef<EventSource | null>(null);
  const executionBoxRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    Promise.all([fetchGraphs(), fetchEditorCatalog()])
      .then(([loadedGraphs, loadedCatalog]) => {
        setGraphs(loadedGraphs);
        setCatalog(loadedCatalog);
        if (loadedGraphs.length > 0) {
          setSelectedGraphId(loadedGraphs[0].graph_id);
        } else {
          history.reset(createBlankGraph());
        }
      })
      .catch((loadError: Error) => {
        setError(loadError.message);
      });
  }, []);

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
        history.reset(layoutGraphLR(graph));
        setSelectedNodeId(null);
        setSelectedEdgeId(null);
      })
      .catch((loadError: Error) => setError(loadError.message));
  }, [selectedGraphId]);

  const persistedGraphIds = useMemo(() => new Set(graphs.map((graph) => graph.graph_id)), [graphs]);

  async function refreshGraphs(nextSelectedGraphId?: string) {
    const loadedGraphs = await fetchGraphs();
    setGraphs(loadedGraphs);
    if (nextSelectedGraphId) {
      setSelectedGraphId(nextSelectedGraphId);
    } else if (loadedGraphs.length === 0) {
      setSelectedGraphId("");
      history.reset(createBlankGraph());
    }
  }

  async function saveCurrentGraph(): Promise<GraphDefinition | null> {
    if (!draftGraph) {
      return null;
    }
    setIsSaving(true);
    setError(null);
    try {
      const normalized = normalizeGraph(draftGraph);
      const savedGraph =
        selectedGraphId && persistedGraphIds.has(selectedGraphId)
          ? await updateGraph(selectedGraphId, normalized)
          : await createGraph(normalized);
      await refreshGraphs(savedGraph.graph_id);
      setDraftGraph(savedGraph);
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
    sourceRef.current?.close();
    setSelectedGraphId("");
    history.reset(createBlankGraph());
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
        setSelectedGraphId(loadedGraphs[0].graph_id);
      } else {
        setSelectedGraphId("");
        history.reset(createBlankGraph());
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

        if (event.event_type === "run.completed" || event.event_type === "run.failed") {
          source.close();
          fetchRun(runId)
            .then((state) => setRunState(state))
            .catch((fetchError: Error) => setError(fetchError.message))
            .finally(() => setIsRunning(false));
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

  return (
    <main className="app-shell">
      <div ref={executionBoxRef} className="hero-mosaic">
        <div className="mosaic-tile panel mosaic-title">
          <h1>Graph Agent Studio</h1>
          <p>Drag nodes into the canvas, wire edges, and launch your agent.</p>
        </div>

        <div className="mosaic-tile panel mosaic-run">
          <button type="button" onClick={() => void handleRun()} disabled={!draftGraph || isRunning || isSaving}>
            {isRunning ? "Running..." : "Run Graph"}
          </button>
          {activeRunId ? <code>Run ID: {activeRunId}</code> : null}
          {error ? <p className="error-text">{error}</p> : null}
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

        <div className="mosaic-tile panel mosaic-input">
          <label>
            Input
            <textarea value={input} onChange={(event) => setInput(event.target.value)} rows={3} />
          </label>
        </div>

        <div className="mosaic-tile panel mosaic-env">
          <h2>Environment</h2>
          <GraphEnvEditor graph={draftGraph} onGraphChange={setDraftGraph} />
        </div>

        <div className="mosaic-tile panel mosaic-actions">
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
            onClick={() => draftGraph && setDraftGraph(layoutGraphLR(draftGraph))}
            disabled={!draftGraph || draftGraph.nodes.length === 0}
          >
            Auto Layout
          </button>
          <button
            type="button"
            className="danger-button"
            onClick={() => void handleDeleteGraph()}
            disabled={!draftGraph}
          >
            Delete
          </button>
        </div>
      </div>

      <section className="content-grid">
        <GraphCanvas
          graph={draftGraph}
          runState={runState}
          events={events}
          activeRunId={activeRunId}
          isRunning={isRunning}
          catalog={catalog}
          selectedNodeId={selectedNodeId}
          selectedEdgeId={selectedEdgeId}
          onGraphChange={setDraftGraph}
          onGraphDrag={history.setQuiet}
          onRunGraph={() => void handleRun()}
          onScrollToTop={scrollToExecutionBox}
          onSelectionChange={(nodeId, edgeId) => {
            setSelectedNodeId(nodeId);
            setSelectedEdgeId(edgeId);
          }}
        />
      </section>
    </main>
  );
}
