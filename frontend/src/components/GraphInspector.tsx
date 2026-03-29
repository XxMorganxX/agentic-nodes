import type { ChangeEvent } from "react";

import type { EditorCatalog, GraphDefinition, GraphEdge, GraphNode, NodeProviderDefinition } from "../lib/types";

type GraphInspectorProps = {
  graph: GraphDefinition | null;
  catalog: EditorCatalog | null;
  selectedNodeId: string | null;
  selectedEdgeId: string | null;
  onGraphChange: (graph: GraphDefinition) => void;
  onOpenProviderDetails?: (nodeId: string) => void;
};

function updateNode(graph: GraphDefinition, nodeId: string, updater: (node: GraphNode) => GraphNode): GraphDefinition {
  return {
    ...graph,
    nodes: graph.nodes.map((node) => (node.id === nodeId ? updater(node) : node)),
  };
}

function updateEdge(graph: GraphDefinition, edgeId: string, updater: (edge: GraphEdge) => GraphEdge): GraphDefinition {
  return {
    ...graph,
    edges: graph.edges.map((edge) => (edge.id === edgeId ? updater(edge) : edge)),
  };
}

function modelProviders(catalog: EditorCatalog | null): NodeProviderDefinition[] {
  return (catalog?.node_providers ?? []).filter((provider) => provider.category === "provider");
}

function defaultModelName(providerName: string): string {
  if (providerName === "openai") {
    return "gpt-4.1-mini";
  }
  if (providerName === "claude") {
    return "claude-3-5-haiku-latest";
  }
  return "mock-default";
}

export function GraphInspector({
  graph,
  catalog,
  selectedNodeId,
  selectedEdgeId,
  onGraphChange,
  onOpenProviderDetails,
}: GraphInspectorProps) {
  if (!graph) {
    return (
      <section className="panel inspector-panel">
        <div className="panel-header">
          <h2>Inspector</h2>
          <p>Select or create an agent to inspect its settings.</p>
        </div>
      </section>
    );
  }

  const selectedNode = selectedNodeId ? graph.nodes.find((node) => node.id === selectedNodeId) ?? null : null;
  const selectedEdge = selectedEdgeId ? graph.edges.find((edge) => edge.id === selectedEdgeId) ?? null : null;

  if (selectedNode) {
    const contract = catalog?.contracts[selectedNode.category];
    const allowedTools = Array.isArray(selectedNode.config.allowed_tool_names)
      ? (selectedNode.config.allowed_tool_names as string[])
      : [];
    const availableModelProviders = modelProviders(catalog);

    return (
      <section className="panel inspector-panel">
        <div className="panel-header">
          <h2>Node Inspector</h2>
          <p>Edit the selected node and its runtime contract.</p>
        </div>
        <div className="inspector-body">
          <label>
            Node ID
            <input value={selectedNode.id} readOnly />
          </label>
          <label>
            Label
            <input
              value={selectedNode.label}
              onChange={(event) =>
                onGraphChange(updateNode(graph, selectedNode.id, (node) => ({ ...node, label: event.target.value })))
              }
            />
          </label>
          <label>
            Description
            <textarea
              rows={3}
              value={selectedNode.description ?? ""}
              onChange={(event) =>
                onGraphChange(
                  updateNode(graph, selectedNode.id, (node) => ({ ...node, description: event.target.value })),
                )
              }
            />
          </label>
          <label>
            Position X
            <input
              type="number"
              value={selectedNode.position.x}
              onChange={(event) =>
                onGraphChange(
                  updateNode(graph, selectedNode.id, (node) => ({
                    ...node,
                    position: { ...node.position, x: Number(event.target.value) },
                  })),
                )
              }
            />
          </label>
          <label>
            Position Y
            <input
              type="number"
              value={selectedNode.position.y}
              onChange={(event) =>
                onGraphChange(
                  updateNode(graph, selectedNode.id, (node) => ({
                    ...node,
                    position: { ...node.position, y: Number(event.target.value) },
                  })),
                )
              }
            />
          </label>
          <div className="inspector-meta">
            <span>Category: {selectedNode.category}</span>
            <span>Kind: {selectedNode.kind}</span>
            <span>Provider: {selectedNode.provider_label}</span>
          </div>
          {contract ? (
            <div className="contract-card">
              <strong>Contract</strong>
              <span>Accepts: {contract.accepted_inputs.join(", ")}</span>
              <span>Produces: {contract.produced_outputs.join(", ")}</span>
            </div>
          ) : null}
          {selectedNode.kind === "model" ? (
            <>
              <label>
                Model Provider
                <select
                  value={String(selectedNode.config.provider_name ?? selectedNode.model_provider_name ?? "mock")}
                  onChange={(event) => {
                    const nextProvider = availableModelProviders.find((provider) => {
                      const providerName = provider.provider_id.replace("provider.", "");
                      return providerName === event.target.value;
                    });
                    if (!nextProvider) {
                      return;
                    }
                    const nextProviderName = nextProvider.provider_id.replace("provider.", "");
                    const nextModelName = defaultModelName(nextProviderName);
                    onGraphChange(
                      updateNode(graph, selectedNode.id, (node) => ({
                        ...node,
                        model_provider_name: nextProviderName,
                        config: {
                          ...node.config,
                          provider_name: nextProviderName,
                          model: nextModelName || node.config.model,
                          max_tokens: nextProviderName === "claude" ? Number(node.config.max_tokens ?? 1024) : node.config.max_tokens,
                        },
                      })),
                    );
                  }}
                >
                  {availableModelProviders.map((provider) => {
                    const providerName = provider.provider_id.replace("provider.", "");
                    return (
                      <option key={provider.provider_id} value={providerName}>
                        {provider.display_name}
                      </option>
                    );
                  })}
                </select>
              </label>
              {onOpenProviderDetails ? (
                <button
                  type="button"
                  className="secondary-button"
                  onClick={() => onOpenProviderDetails(selectedNode.id)}
                >
                  Learn More About Provider
                </button>
              ) : null}
              <label>
                Model Provider Name
                <input
                  value={String(selectedNode.config.provider_name ?? selectedNode.model_provider_name ?? "")}
                  onChange={(event) =>
                    onGraphChange(
                      updateNode(graph, selectedNode.id, (node) => ({
                        ...node,
                        model_provider_name: event.target.value,
                        config: { ...node.config, provider_name: event.target.value },
                      })),
                    )
                  }
                />
              </label>
              <label>
                Prompt Name
                <input
                  value={String(selectedNode.config.prompt_name ?? selectedNode.prompt_name ?? "")}
                  onChange={(event) =>
                    onGraphChange(
                      updateNode(graph, selectedNode.id, (node) => ({
                        ...node,
                        prompt_name: event.target.value,
                        config: { ...node.config, prompt_name: event.target.value, mode: event.target.value },
                      })),
                    )
                  }
                />
              </label>
              <label>
                Model Name
                <input
                  value={String(selectedNode.config.model ?? "")}
                  onChange={(event) =>
                    onGraphChange(
                      updateNode(graph, selectedNode.id, (node) => ({
                        ...node,
                        config: { ...node.config, model: event.target.value },
                      })),
                    )
                  }
                />
              </label>
              <label>
                Temperature
                <input
                  type="number"
                  step="0.1"
                  value={String(selectedNode.config.temperature ?? "")}
                  onChange={(event) =>
                    onGraphChange(
                      updateNode(graph, selectedNode.id, (node) => ({
                        ...node,
                        config: {
                          ...node.config,
                          temperature: event.target.value === "" ? "" : Number(event.target.value),
                        },
                      })),
                    )
                  }
                />
              </label>
              <label>
                Max Tokens
                <input
                  type="number"
                  value={String(selectedNode.config.max_tokens ?? "")}
                  onChange={(event) =>
                    onGraphChange(
                      updateNode(graph, selectedNode.id, (node) => ({
                        ...node,
                        config: {
                          ...node.config,
                          max_tokens: event.target.value === "" ? "" : Number(event.target.value),
                        },
                      })),
                    )
                  }
                />
              </label>
              <label>
                API Base
                <input
                  value={String(selectedNode.config.api_base ?? "")}
                  onChange={(event) =>
                    onGraphChange(
                      updateNode(graph, selectedNode.id, (node) => ({
                        ...node,
                        config: { ...node.config, api_base: event.target.value },
                      })),
                    )
                  }
                />
              </label>
              <label>
                API Key Env Var
                <input
                  value={String(selectedNode.config.api_key_env_var ?? "")}
                  onChange={(event) =>
                    onGraphChange(
                      updateNode(graph, selectedNode.id, (node) => ({
                        ...node,
                        config: { ...node.config, api_key_env_var: event.target.value },
                      })),
                    )
                  }
                />
              </label>
              <label>
                System Prompt
                <textarea
                  rows={5}
                  value={String(selectedNode.config.system_prompt ?? "")}
                  onChange={(event) =>
                    onGraphChange(
                      updateNode(graph, selectedNode.id, (node) => ({
                        ...node,
                        config: { ...node.config, system_prompt: event.target.value },
                      })),
                    )
                  }
                />
              </label>
              <label>
                User Message Template
                <textarea
                  rows={5}
                  value={String(selectedNode.config.user_message_template ?? "{input_payload}")}
                  onChange={(event) =>
                    onGraphChange(
                      updateNode(graph, selectedNode.id, (node) => ({
                        ...node,
                        config: { ...node.config, user_message_template: event.target.value },
                      })),
                    )
                  }
                />
              </label>
              <label>
                Response Mode
                <select
                  value={String(selectedNode.config.response_mode ?? "message")}
                  onChange={(event) =>
                    onGraphChange(
                      updateNode(graph, selectedNode.id, (node) => ({
                        ...node,
                        config: { ...node.config, response_mode: event.target.value },
                      })),
                    )
                  }
                >
                  <option value="message">message</option>
                  <option value="tool_call">tool_call</option>
                </select>
              </label>
              <div className="checkbox-grid">
                <strong>Available Tools</strong>
                {(catalog?.tools ?? []).map((tool) => {
                  const isChecked = allowedTools.includes(tool.name);
                  return (
                    <label key={tool.name} className="checkbox-option">
                      <input
                        type="checkbox"
                        checked={isChecked}
                        onChange={(event) => {
                          const nextTools = event.target.checked
                            ? [...allowedTools, tool.name]
                            : allowedTools.filter((name) => name !== tool.name);
                          onGraphChange(
                            updateNode(graph, selectedNode.id, (node) => ({
                              ...node,
                              config: {
                                ...node.config,
                                allowed_tool_names: nextTools,
                                preferred_tool_name:
                                  nextTools.length > 0 ? String(node.config.preferred_tool_name ?? nextTools[0]) : "",
                              },
                            })),
                          );
                        }}
                      />
                      <span>{tool.name}</span>
                    </label>
                  );
                })}
              </div>
              <label>
                Preferred Tool Name
                <input
                  value={String(selectedNode.config.preferred_tool_name ?? "")}
                  onChange={(event) =>
                    onGraphChange(
                      updateNode(graph, selectedNode.id, (node) => ({
                        ...node,
                        config: { ...node.config, preferred_tool_name: event.target.value },
                      })),
                    )
                  }
                />
              </label>
            </>
          ) : null}
          {selectedNode.kind === "tool" ? (
            <label>
              Tool
              <select
                value={String(selectedNode.config.tool_name ?? selectedNode.tool_name ?? "")}
                onChange={(event) =>
                  onGraphChange(
                    updateNode(graph, selectedNode.id, (node) => ({
                      ...node,
                      tool_name: event.target.value,
                      config: { ...node.config, tool_name: event.target.value },
                    })),
                  )
                }
              >
                {(catalog?.tools ?? []).map((tool) => (
                  <option key={tool.name} value={tool.name}>
                    {tool.name}
                  </option>
                ))}
              </select>
            </label>
          ) : null}
          {selectedNode.kind === "data" ? (
            <>
              <label>
                Data Mode
                <select
                  value={String(selectedNode.config.mode ?? "passthrough")}
                  onChange={(event) =>
                    onGraphChange(
                      updateNode(graph, selectedNode.id, (node) => ({
                        ...node,
                        config: { ...node.config, mode: event.target.value },
                      })),
                    )
                  }
                >
                  <option value="passthrough">passthrough</option>
                  <option value="template">template</option>
                </select>
              </label>
              <label>
                Template
                <textarea
                  rows={4}
                  value={String(selectedNode.config.template ?? "{input_payload}")}
                  onChange={(event) =>
                    onGraphChange(
                      updateNode(graph, selectedNode.id, (node) => ({
                        ...node,
                        config: { ...node.config, template: event.target.value },
                      })),
                    )
                  }
                />
              </label>
            </>
          ) : null}
        </div>
      </section>
    );
  }

  if (selectedEdge) {
    return (
      <section className="panel inspector-panel">
        <div className="panel-header">
          <h2>Edge Inspector</h2>
          <p>Configure routing and condition behavior for the selected edge.</p>
        </div>
        <div className="inspector-body">
          <label>
            Edge ID
            <input value={selectedEdge.id} readOnly />
          </label>
          <label>
            Label
            <input
              value={selectedEdge.label}
              onChange={(event) =>
                onGraphChange(updateEdge(graph, selectedEdge.id, (edge) => ({ ...edge, label: event.target.value })))
              }
            />
          </label>
          <label>
            Kind
            <select
              value={selectedEdge.kind}
              onChange={(event) =>
                onGraphChange(
                  updateEdge(graph, selectedEdge.id, (edge) => ({
                    ...edge,
                    kind: event.target.value,
                    condition:
                      event.target.value === "conditional"
                        ? edge.condition ?? {
                            id: `${edge.id}-condition`,
                            label: "Validation error",
                            type: "result_status_equals",
                            value: "validation_error",
                            path: null,
                          }
                        : null,
                  })),
                )
              }
            >
              <option value="standard">standard</option>
              <option value="conditional">conditional</option>
            </select>
          </label>
          <label>
            Priority
            <input
              type="number"
              value={selectedEdge.priority}
              onChange={(event) =>
                onGraphChange(
                  updateEdge(graph, selectedEdge.id, (edge) => ({ ...edge, priority: Number(event.target.value) })),
                )
              }
            />
          </label>
          {selectedEdge.kind === "conditional" && selectedEdge.condition ? (
            <>
              <label>
                Condition Label
                <input
                  value={selectedEdge.condition.label}
                  onChange={(event) =>
                    onGraphChange(
                      updateEdge(graph, selectedEdge.id, (edge) => ({
                        ...edge,
                        condition: { ...edge.condition!, label: event.target.value },
                      })),
                    )
                  }
                />
              </label>
              <label>
                Condition Type
                <select
                  value={selectedEdge.condition.type}
                  onChange={(event) =>
                    onGraphChange(
                      updateEdge(graph, selectedEdge.id, (edge) => ({
                        ...edge,
                        condition: { ...edge.condition!, type: event.target.value },
                      })),
                    )
                  }
                >
                  <option value="result_status_equals">result_status_equals</option>
                  <option value="result_has_error">result_has_error</option>
                  <option value="result_payload_path_equals">result_payload_path_equals</option>
                </select>
              </label>
              <label>
                Condition Value
                <input
                  value={String(selectedEdge.condition.value ?? "")}
                  onChange={(event) =>
                    onGraphChange(
                      updateEdge(graph, selectedEdge.id, (edge) => ({
                        ...edge,
                        condition: { ...edge.condition!, value: event.target.value },
                      })),
                    )
                  }
                />
              </label>
              <label>
                Condition Path
                <input
                  value={String(selectedEdge.condition.path ?? "")}
                  onChange={(event) =>
                    onGraphChange(
                      updateEdge(graph, selectedEdge.id, (edge) => ({
                        ...edge,
                        condition: { ...edge.condition!, path: event.target.value },
                      })),
                    )
                  }
                />
              </label>
            </>
          ) : null}
        </div>
      </section>
    );
  }

  return (
    <section className="panel inspector-panel">
      <div className="panel-header">
        <h2>Graph Inspector</h2>
        <p>Edit the top-level agent metadata and start node.</p>
      </div>
      <div className="inspector-body">
        <label>
          Graph ID
          <input value={graph.graph_id} onChange={(event) => onGraphChange({ ...graph, graph_id: event.target.value })} />
        </label>
        <label>
          Name
          <input value={graph.name} onChange={(event) => onGraphChange({ ...graph, name: event.target.value })} />
        </label>
        <label>
          Description
          <textarea
            rows={4}
            value={graph.description}
            onChange={(event) => onGraphChange({ ...graph, description: event.target.value })}
          />
        </label>
        <label>
          Start Node
          <select
            value={graph.start_node_id}
            onChange={(event) => onGraphChange({ ...graph, start_node_id: event.target.value })}
          >
            <option value="">Select a start node</option>
            {graph.nodes
              .filter((node) => node.category === "start")
              .map((node) => (
                <option key={node.id} value={node.id}>
                  {node.label}
                </option>
              ))}
          </select>
        </label>
        <div className="inspector-meta">
          <span>Nodes: {graph.nodes.length}</span>
          <span>Edges: {graph.edges.length}</span>
        </div>
      </div>
    </section>
  );
}
