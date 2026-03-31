import type { ChangeEvent } from "react";

import {
  defaultModelName,
  findProviderDefinition,
  isWireJunctionNode,
  modelProviderDefinitions,
  providerDefaultConfig,
  providerModelName,
} from "../lib/editor";
import { useRenderDiagnostics } from "../lib/dragDiagnostics";
import type { EditorCatalog, GraphDefinition, GraphEdge, GraphNode, NodeProviderDefinition, ToolDefinition } from "../lib/types";

type GraphInspectorProps = {
  graph: GraphDefinition | null;
  catalog: EditorCatalog | null;
  selectedNodeId: string | null;
  selectedEdgeId: string | null;
  onGraphChange: (graph: GraphDefinition) => void;
  onOpenProviderDetails?: (nodeId: string) => void;
  onSaveNode?: (node: GraphNode) => void;
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

function isToolOnline(tool: ToolDefinition): boolean {
  return tool.available !== false;
}

function isToolEnabled(tool: ToolDefinition): boolean {
  return tool.enabled !== false;
}

function toolStatusLabel(tool: ToolDefinition): string {
  if (!isToolEnabled(tool)) {
    return "disabled";
  }
  if (!isToolOnline(tool)) {
    return "offline";
  }
  return "ready";
}

export function GraphInspector({
  graph,
  catalog,
  selectedNodeId,
  selectedEdgeId,
  onGraphChange,
  onOpenProviderDetails,
  onSaveNode,
}: GraphInspectorProps) {
  useRenderDiagnostics(
    "GraphInspector",
    true,
    {
      hasGraph: Boolean(graph),
      selectedNodeId: selectedNodeId ?? "none",
      selectedEdgeId: selectedEdgeId ?? "none",
      nodeCount: graph?.nodes.length ?? 0,
      edgeCount: graph?.edges.length ?? 0,
    },
    12,
  );

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
    if (isWireJunctionNode(selectedNode)) {
      return (
        <section className="panel inspector-panel">
          <div className="panel-header">
            <h2>Wire Point</h2>
            <p>This floating junction anchors a routed wire segment and can be dragged to reshape the path.</p>
          </div>
          <div className="inspector-body">
            <label>
              Node ID
              <input value={selectedNode.id} readOnly />
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
              <span>Kind: floating junction</span>
              <span>Purpose: wire routing</span>
            </div>
          </div>
        </section>
      );
    }

    const contract = catalog?.contracts[selectedNode.category];
    const allowedTools = Array.isArray(selectedNode.config.allowed_tool_names)
      ? (selectedNode.config.allowed_tool_names as string[])
      : [];
    const availableModelProviders = modelProviderDefinitions(catalog);
    const selectedProviderName = String(selectedNode.config.provider_name ?? selectedNode.model_provider_name ?? "mock");
    const selectedProvider = findProviderDefinition(catalog, selectedProviderName);
    const providerConfigFields = selectedProvider?.config_fields ?? [];
    const providerStatus = catalog?.provider_statuses?.[selectedProviderName];
    const catalogTools = catalog?.tools ?? [];
    const isDiscordStartNode = selectedNode.kind === "input" && selectedNode.provider_id === "start.discord_message";
    const isManualStartNode =
      selectedNode.kind === "input" &&
      (selectedNode.provider_id === "start.manual_run" || selectedNode.provider_id === "core.input");

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
          {selectedNode.kind === "input" ? (
            <>
              <label>
                Start Trigger
                <input value={isDiscordStartNode ? "discord_message" : "manual_run"} readOnly />
              </label>
              {isManualStartNode ? (
                <div className="contract-card">
                  <strong>Manual Run Start</strong>
                  <span>This node is triggered by clicking Run in the editor.</span>
                  <span>Payload source: input payload passed to the run request.</span>
                </div>
              ) : null}
              {isDiscordStartNode ? (
                <>
                  <label>
                    Discord Bot Token Env Var
                    <input
                      value={String(selectedNode.config.discord_bot_token_env_var ?? "{DISCORD_BOT_TOKEN}")}
                      placeholder="{DISCORD_BOT_TOKEN}"
                      onChange={(event) =>
                        onGraphChange(
                          updateNode(graph, selectedNode.id, (node) => ({
                            ...node,
                            config: {
                              ...node.config,
                              trigger_mode: "discord_message",
                              discord_bot_token_env_var: event.target.value,
                            },
                          })),
                        )
                      }
                    />
                  </label>
                  <label>
                    Discord Channel ID
                    <input
                      value={String(selectedNode.config.discord_channel_id ?? "")}
                      placeholder="123456789012345678"
                      onChange={(event) =>
                        onGraphChange(
                          updateNode(graph, selectedNode.id, (node) => ({
                            ...node,
                            config: {
                              ...node.config,
                              trigger_mode: "discord_message",
                              discord_channel_id: event.target.value,
                            },
                          })),
                        )
                      }
                    />
                  </label>
                  <label className="checkbox-option">
                    <input
                      type="checkbox"
                      checked={Boolean(selectedNode.config.ignore_bot_messages ?? true)}
                      onChange={(event) =>
                        onGraphChange(
                          updateNode(graph, selectedNode.id, (node) => ({
                            ...node,
                            config: {
                              ...node.config,
                              trigger_mode: "discord_message",
                              ignore_bot_messages: event.target.checked,
                            },
                          })),
                        )
                      }
                    />
                    <span>Ignore bot-authored messages</span>
                  </label>
                  <label className="checkbox-option">
                    <input
                      type="checkbox"
                      checked={Boolean(selectedNode.config.ignore_self_messages ?? true)}
                      onChange={(event) =>
                        onGraphChange(
                          updateNode(graph, selectedNode.id, (node) => ({
                            ...node,
                            config: {
                              ...node.config,
                              trigger_mode: "discord_message",
                              ignore_self_messages: event.target.checked,
                            },
                          })),
                        )
                      }
                    />
                    <span>Ignore this bot's own messages</span>
                  </label>
                </>
              ) : null}
            </>
          ) : null}
          {selectedNode.kind === "model" ? (
            <>
              <label>
                Model Provider
                <select
                  value={selectedProviderName}
                  onChange={(event) => {
                    const nextProvider = availableModelProviders.find((provider) => {
                      const providerName = provider.provider_id.replace("provider.", "");
                      return providerName === event.target.value;
                    });
                    if (!nextProvider) {
                      return;
                    }
                    const nextProviderName = providerModelName(nextProvider);
                    const nextProviderConfig = providerDefaultConfig(nextProvider);
                    const providerConfigKeys = Array.from(
                      new Set(
                        availableModelProviders.flatMap((provider) => [
                          "provider_name",
                          ...((provider.config_fields ?? []).map((field) => field.key)),
                        ]),
                      ),
                    );
                    onGraphChange(
                      updateNode(graph, selectedNode.id, (node) => {
                        const nextConfig = { ...node.config };
                        providerConfigKeys.forEach((key) => delete nextConfig[key]);
                        return {
                          ...node,
                          model_provider_name: nextProviderName,
                          config: {
                            ...nextConfig,
                            ...nextProviderConfig,
                            provider_name: nextProviderName,
                            model:
                              typeof nextProviderConfig.model === "string"
                                ? nextProviderConfig.model
                                : defaultModelName(nextProviderName, catalog) || node.config.model,
                          },
                        };
                      }),
                    );
                  }}
                >
                  {availableModelProviders.map((provider) => {
                    const providerName = providerModelName(provider);
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
              {providerStatus ? (
                <div className="contract-card">
                  <strong>Provider Health</strong>
                  <span>{providerStatus.message}</span>
                  {(providerStatus.warnings ?? []).map((warning) => (
                    <span key={warning}>{warning}</span>
                  ))}
                </div>
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
              {providerConfigFields.map((field) => {
                const value = selectedNode.config[field.key];
                const isNumberField = field.input_type === "number";
                const isSelectField = field.input_type === "select" && (field.options?.length ?? 0) > 0;
                const isModelSelectField = isSelectField && field.key === "model";
                const currentValue = String(value ?? "");
                const selectOptions =
                  isSelectField && currentValue && !field.options?.some((option) => option.value === currentValue)
                    ? [...(field.options ?? []), { value: currentValue, label: `Custom: ${currentValue}` }]
                    : (field.options ?? []);
                const datalistId = `${selectedNode.id}-${field.key}-options`;
                const inputProps = isNumberField ? { type: "number" } : {};
                return (
                  <label key={field.key}>
                    {field.label}
                    {isModelSelectField ? (
                      <>
                        <input
                          list={datalistId}
                          value={currentValue}
                          placeholder={field.placeholder || "Select or type a model id"}
                          onChange={(event) =>
                            onGraphChange(
                              updateNode(graph, selectedNode.id, (node) => ({
                                ...node,
                                config: {
                                  ...node.config,
                                  [field.key]: event.target.value,
                                },
                              })),
                            )
                          }
                        />
                        <datalist id={datalistId}>
                          {selectOptions.map((option) => (
                            <option key={option.value} value={option.value}>
                              {option.label}
                            </option>
                          ))}
                        </datalist>
                      </>
                    ) : isSelectField ? (
                      <select
                        value={currentValue}
                        onChange={(event) =>
                          onGraphChange(
                            updateNode(graph, selectedNode.id, (node) => ({
                              ...node,
                              config: {
                                ...node.config,
                                [field.key]: event.target.value,
                              },
                            })),
                          )
                        }
                      >
                        {selectOptions.map((option) => (
                          <option key={option.value} value={option.value}>
                            {option.label}
                          </option>
                        ))}
                      </select>
                    ) : (
                      <input
                        {...inputProps}
                        value={currentValue}
                        placeholder={field.placeholder || undefined}
                        onChange={(event) =>
                          onGraphChange(
                            updateNode(graph, selectedNode.id, (node) => ({
                              ...node,
                              config: {
                                ...node.config,
                                [field.key]:
                                  isNumberField && event.target.value !== "" ? Number(event.target.value) : event.target.value,
                              },
                            })),
                          )
                        }
                      />
                    )}
                  </label>
                );
              })}
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
                {catalogTools.map((tool) => {
                  const isChecked = allowedTools.includes(tool.name);
                  const canSelectTool = isToolEnabled(tool) && isToolOnline(tool);
                  return (
                    <label key={tool.name} className="checkbox-option">
                      <input
                        type="checkbox"
                        checked={isChecked}
                        disabled={!isChecked && !canSelectTool}
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
                      <span>
                        {tool.name}
                        <small>{toolStatusLabel(tool)}</small>
                      </span>
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
            <>
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
                  {catalogTools.map((tool) => (
                    <option key={tool.name} value={tool.name}>
                      {tool.name} ({toolStatusLabel(tool)})
                    </option>
                  ))}
                </select>
              </label>
              {(() => {
                const selectedTool = catalogTools.find(
                  (tool) => tool.name === String(selectedNode.config.tool_name ?? selectedNode.tool_name ?? ""),
                );
                if (!selectedTool || selectedTool.source_type !== "mcp") {
                  return null;
                }
                return (
                  <label className="checkbox-option">
                    <input
                      type="checkbox"
                      checked={Boolean(selectedNode.config.include_mcp_tool_context)}
                      onChange={(event) =>
                        onGraphChange(
                          updateNode(graph, selectedNode.id, (node) => ({
                            ...node,
                            config: {
                              ...node.config,
                              include_mcp_tool_context: event.target.checked,
                            },
                          })),
                        )
                      }
                    />
                    <span>
                      Pass MCP Context To Model Provider
                      <small>Expose this tool node's MCP metadata to connected or targeted model nodes as `mcp_tool_context`.</small>
                    </span>
                  </label>
                );
              })()}
            </>
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
              {String(selectedNode.config.mode ?? "passthrough") === "template" ? (
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
              ) : null}
            </>
          ) : null}
          {onSaveNode ? (
            <div className="inspector-save-section">
              <button
                type="button"
                className="secondary-button inspector-save-button"
                onClick={() => onSaveNode(selectedNode)}
              >
                <svg viewBox="0 0 24 24" aria-hidden="true">
                  <path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2Z" />
                  <path d="M17 21v-8H7v8M7 3v5h8" />
                </svg>
                Save to Library
              </button>
              <span className="inspector-save-hint">Save this node's configuration for reuse in the Add menu.</span>
            </div>
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
