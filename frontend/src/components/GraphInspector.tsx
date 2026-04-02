import {
  defaultModelName,
  findProviderDefinition,
  inferModelResponseMode,
  isPromptBlockNode,
  isWireJunctionNode,
  modelProviderDefinitions,
  PROMPT_BLOCK_PROVIDER_ID,
  providerDefaultConfig,
  providerModelName,
} from "../lib/editor";
import {
  getContextBuilderBindings,
  slugifyContextBuilderPlaceholder,
  type ContextBuilderBindingRow,
} from "../lib/contextBuilderBindings";
import { getNodeInstanceLabel } from "../lib/nodeInstanceLabels";
import { insertTokenAtEnd, listPromptBlockAvailableVariables, PROMPT_BLOCK_STARTERS, renderPromptBlockPreview } from "../lib/promptBlockEditor";
import { useRenderDiagnostics } from "../lib/dragDiagnostics";
import type {
  EditorCatalog,
  GraphDefinition,
  GraphEdge,
  GraphNode,
  RunState,
  ToolDefinition,
} from "../lib/types";

type GraphInspectorProps = {
  graph: GraphDefinition | null;
  catalog: EditorCatalog | null;
  runState: RunState | null;
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

function uniqueStrings(values: string[]): string[] {
  return [...new Set(values.filter((value) => value.trim().length > 0))];
}

const CONTEXT_BUILDER_PROVIDER_ID = "core.context_builder";
const CONTEXT_BUILDER_IDENTIFIER_PATTERN = /^[A-Za-z_][A-Za-z0-9_]*$/;
const CONTEXT_BUILDER_TOKEN_PATTERN = /\{([A-Za-z_][A-Za-z0-9_]*)\}/g;
const CONTEXT_BUILDER_BASE_VARIABLES = ["current_node_id", "graph_id", "input_payload", "run_id"];

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function buildContextBuilderTemplate(bindings: ContextBuilderBindingRow[]): string {
  return bindings
    .map((binding) => `# ${binding.sourceLabel}\n{${binding.placeholder}}`)
    .join("\n\n");
}

function extractTemplateTokens(template: string): string[] {
  return uniqueStrings(Array.from(template.matchAll(CONTEXT_BUILDER_TOKEN_PATTERN)).map((match) => match[1] ?? ""));
}

function stringifyPreviewValue(value: unknown): string {
  if (typeof value === "string") {
    return value;
  }
  if (value === null || value === undefined) {
    return "";
  }
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function truncatePreview(value: string, limit = 180): string {
  return value.length > limit ? `${value.slice(0, limit - 1)}...` : value;
}

function getContextBuilderSourcePreview(runState: RunState | null, sourceNodeId: string): string | null {
  const rawOutput = runState?.node_outputs?.[sourceNodeId];
  if (rawOutput === undefined) {
    return null;
  }
  const payload = isRecord(rawOutput) && Object.prototype.hasOwnProperty.call(rawOutput, "payload") ? rawOutput.payload : rawOutput;
  return truncatePreview(stringifyPreviewValue(payload));
}

function getPromptBlockPreview(node: GraphNode): string {
  const role = String(node.config.role ?? "user").trim() || "user";
  const name = String(node.config.name ?? "").trim();
  const content = String(node.config.content ?? "").trim();
  const header = name ? `${role} (${name})` : role;
  return truncatePreview(`${header}: ${content}`.trim());
}

function getContextBuilderSourcePreviewFromGraph(graph: GraphDefinition, runState: RunState | null, sourceNodeId: string): string | null {
  const sourceNode = graph.nodes.find((candidate) => candidate.id === sourceNodeId) ?? null;
  if (sourceNode && isPromptBlockNode(sourceNode)) {
    return getPromptBlockPreview(sourceNode);
  }
  return getContextBuilderSourcePreview(runState, sourceNodeId);
}

function getModelPromptBlockNodes(graph: GraphDefinition, modelNode: GraphNode): GraphNode[] {
  const candidateNodeIds = new Set<string>();
  const configuredNodeIds = Array.isArray(modelNode.config.prompt_block_node_ids)
    ? modelNode.config.prompt_block_node_ids.map((nodeId) => String(nodeId))
    : [];
  configuredNodeIds.forEach((nodeId) => candidateNodeIds.add(nodeId));
  graph.edges
    .filter((edge) => edge.kind === "binding" && edge.target_id === modelNode.id)
    .forEach((edge) => candidateNodeIds.add(edge.source_id));
  return [...candidateNodeIds]
    .map((nodeId) => graph.nodes.find((node) => node.id === nodeId) ?? null)
    .filter((node): node is GraphNode => node !== null && isPromptBlockNode(node));
}

function renderContextBuilderPreview(template: string, variables: Record<string, string>): string {
  return template.replace(CONTEXT_BUILDER_TOKEN_PATTERN, (_, token: string) => variables[token] ?? `{${token}}`);
}

function getModelMcpContextNodes(graph: GraphDefinition, modelNode: GraphNode): GraphNode[] {
  const candidateNodeIds = new Set<string>();
  const configuredTargetIds = Array.isArray(modelNode.config.tool_target_node_ids)
    ? modelNode.config.tool_target_node_ids.map((nodeId) => String(nodeId))
    : [];
  configuredTargetIds.forEach((nodeId) => candidateNodeIds.add(nodeId));
  graph.edges
    .filter((edge) => edge.kind === "binding" && edge.target_id === modelNode.id)
    .forEach((edge) => candidateNodeIds.add(edge.source_id));
  return [...candidateNodeIds]
    .map((nodeId) => graph.nodes.find((node) => node.id === nodeId) ?? null)
    .filter((node): node is GraphNode => node !== null && node.kind === "mcp_context_provider");
}

function describeMcpExecutorBinding(binding: unknown): string {
  if (!binding || typeof binding !== "object") {
    return "implicit latest incoming edge";
  }
  const bindingRecord = binding as Record<string, unknown>;
  const bindingType = String(bindingRecord.type ?? "latest_output");
  if (bindingType === "first_available_envelope") {
    const sources = Array.isArray(bindingRecord.sources)
      ? bindingRecord.sources.map((sourceId) => String(sourceId)).filter((sourceId) => sourceId.trim().length > 0)
      : [];
    return sources.length > 0 ? `${bindingType} from ${sources.join(", ")}` : bindingType;
  }
  const sourceId = String(bindingRecord.source ?? "").trim();
  return sourceId ? `${bindingType} from ${sourceId}` : bindingType;
}

export function GraphInspector({
  graph,
  catalog,
  runState,
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
  const formatNodeLabel = (node: GraphNode) => getNodeInstanceLabel(graph, node);
  const executorFollowUpEnabled =
    selectedNode?.kind === "mcp_tool_executor" && Boolean(selectedNode.config.enable_follow_up_decision);
  const isPromptDrivenNode = selectedNode?.kind === "model" || executorFollowUpEnabled;
  const selectedModelResponseMode =
    selectedNode?.kind === "model"
      ? inferModelResponseMode(graph, selectedNode)
      : executorFollowUpEnabled
        ? String(selectedNode.config.response_mode ?? "auto")
        : null;

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
    const mcpCatalogTools = catalogTools.filter((tool) => tool.source_type === "mcp");
    const standardCatalogTools = catalogTools.filter((tool) => tool.source_type !== "mcp");
    const followUpSelectableTools = selectedNode.kind === "mcp_tool_executor" ? mcpCatalogTools : standardCatalogTools;
    const mcpToolByName = new Map(mcpCatalogTools.map((tool) => [tool.name, tool] as const));
    const selectedMcpToolNames = Array.isArray(selectedNode.config.tool_names)
      ? (selectedNode.config.tool_names as string[])
      : [];
    const mcpContextProvidersForModel = selectedNode.kind === "model" ? getModelMcpContextNodes(graph, selectedNode) : [];
    const modelCallableMcpTools =
      isPromptDrivenNode
        ? mcpContextProvidersForModel.flatMap((node) => {
            const nodeToolNames = Array.isArray(node.config.tool_names)
              ? node.config.tool_names.map((toolName) => String(toolName)).filter((toolName) => toolName.trim().length > 0)
              : [];
            if (node.config.expose_mcp_tools === false) {
              return [];
            }
            return nodeToolNames.map((toolName) => {
              const tool = mcpToolByName.get(toolName);
              const status = tool ? toolStatusLabel(tool) : "unknown";
              return `${toolName} (${status}) via ${formatNodeLabel(node)}`;
            });
          })
        : [];
    const modelPromptContextProviders =
      isPromptDrivenNode
        ? uniqueStrings(
            mcpContextProvidersForModel
              .filter((node) => Boolean(node.config.include_mcp_tool_context))
              .map((node) => formatNodeLabel(node)),
          )
        : [];
    const modelTargetedMcpNodeIds =
      selectedNode.kind === "model" && Array.isArray(selectedNode.config.tool_target_node_ids)
        ? uniqueStrings(selectedNode.config.tool_target_node_ids.map((nodeId) => String(nodeId)))
        : [];
    const modelPromptBlockNodes = selectedNode.kind === "model" ? getModelPromptBlockNodes(graph, selectedNode) : [];
    const mcpToolExposureEnabled = selectedNode.kind === "mcp_context_provider" ? selectedNode.config.expose_mcp_tools !== false : false;
    const executorBindingSummary = selectedNode.kind === "mcp_tool_executor" ? describeMcpExecutorBinding(selectedNode.config.input_binding) : "";
    const executorFollowUpResponseMode =
      selectedNode.kind === "mcp_tool_executor" ? String(selectedNode.config.response_mode ?? "auto") : "auto";
    const isDiscordStartNode = selectedNode.kind === "input" && selectedNode.provider_id === "start.discord_message";
    const isManualStartNode =
      selectedNode.kind === "input" &&
      (selectedNode.provider_id === "start.manual_run" || selectedNode.provider_id === "core.input");
    const isContextBuilderNode = selectedNode.kind === "data" && selectedNode.provider_id === CONTEXT_BUILDER_PROVIDER_ID;
    const isPromptBlockDataNode = selectedNode.kind === "data" && selectedNode.provider_id === PROMPT_BLOCK_PROVIDER_ID;
    const contextBuilderBindings = isContextBuilderNode ? getContextBuilderBindings(selectedNode, graph) : [];
    const generatedContextBuilderTemplate = buildContextBuilderTemplate(contextBuilderBindings);
    const rawContextBuilderTemplate = isContextBuilderNode ? String(selectedNode.config.template ?? "") : "";
    const contextBuilderTemplate = isContextBuilderNode
      ? rawContextBuilderTemplate.trim().length > 0
        ? rawContextBuilderTemplate
        : generatedContextBuilderTemplate
      : "";
    const contextBuilderTemplateTokens = isContextBuilderNode ? extractTemplateTokens(contextBuilderTemplate) : [];
    const contextBuilderDuplicatePlaceholders = isContextBuilderNode
      ? contextBuilderBindings
          .map((binding) => binding.placeholder)
          .filter((placeholder, index, placeholders) => placeholders.indexOf(placeholder) !== index)
      : [];
    const contextBuilderInvalidPlaceholders = isContextBuilderNode
      ? contextBuilderBindings
          .map((binding) => binding.placeholder)
          .filter((placeholder) => !CONTEXT_BUILDER_IDENTIFIER_PATTERN.test(placeholder))
      : [];
    const contextBuilderAvailableVariables = new Set(
      isContextBuilderNode
        ? [
            ...contextBuilderBindings.map((binding) => binding.placeholder),
            ...Object.keys(graph.env_vars ?? {}),
            ...CONTEXT_BUILDER_BASE_VARIABLES,
          ]
        : [],
    );
    const contextBuilderUnmatchedTokens = isContextBuilderNode
      ? contextBuilderTemplateTokens.filter((token) => !contextBuilderAvailableVariables.has(token))
      : [];
    const contextBuilderPreviewVariables = isContextBuilderNode
      ? Object.fromEntries(
          contextBuilderBindings.map((binding) => [
            binding.placeholder,
            getContextBuilderSourcePreviewFromGraph(graph, runState, binding.sourceNodeId) ?? "",
          ]),
        )
      : {};
    const contextBuilderRenderedPreview = isContextBuilderNode
      ? renderContextBuilderPreview(contextBuilderTemplate, {
          ...Object.fromEntries(Object.entries(graph.env_vars ?? {}).map(([key, value]) => [key, String(value)])),
          current_node_id: selectedNode.id,
          graph_id: graph.graph_id,
          input_payload: "",
          run_id: runState?.run_id ?? "",
          ...contextBuilderPreviewVariables,
        })
      : "";
    const contextBuilderHasPreviewData = isContextBuilderNode
      ? contextBuilderBindings.some((binding) => getContextBuilderSourcePreviewFromGraph(graph, runState, binding.sourceNodeId) !== null)
      : false;
    const promptBlockAvailableVariables = isPromptBlockDataNode ? listPromptBlockAvailableVariables(graph) : [];
    const promptBlockRenderedPreview = isPromptBlockDataNode ? renderPromptBlockPreview(selectedNode, graph, runState) : "";
    const updateContextBuilderBindings = (bindings: ContextBuilderBindingRow[]) =>
      onGraphChange(
        updateNode(graph, selectedNode.id, (node) => ({
          ...node,
          config: {
            ...node.config,
            input_bindings: bindings.map((binding) => ({
              source_node_id: binding.sourceNodeId,
              placeholder: binding.placeholder,
              binding: binding.binding,
            })),
          },
        })),
      );
    const updateContextBuilderTemplate = (template: string) =>
      onGraphChange(
        updateNode(graph, selectedNode.id, (node) => ({
          ...node,
          config: {
            ...node.config,
            template,
          },
        })),
      );

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
          {isPromptDrivenNode ? (
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
                {selectedNode.kind === "model" ? (
                  <>
                    <input type="text" value={selectedModelResponseMode ?? "message"} readOnly />
                    <small>
                      Uses the node response mode when configured. Otherwise it falls back to graph wiring: tool-call
                      routes enable structured tool decisions, both outputs allow parsed message and tool branches in
                      parallel, and message-only wiring keeps it in message mode.
                    </small>
                  </>
                ) : (
                  <>
                    <select
                      value={executorFollowUpResponseMode}
                      onChange={(event) =>
                        onGraphChange(
                          updateNode(graph, selectedNode.id, (node) => ({
                            ...node,
                            config: { ...node.config, response_mode: event.target.value },
                          })),
                        )
                      }
                    >
                      <option value="auto">auto</option>
                      <option value="tool_call">tool_call</option>
                      <option value="message">message</option>
                    </select>
                    <small>Controls whether the executor's follow-up model may request another MCP tool, emit a message, or do either.</small>
                  </>
                )}
              </label>
              <div className="checkbox-grid">
                <strong>{selectedNode.kind === "mcp_tool_executor" ? "Allowed MCP Tools" : "Direct Registry Tools"}</strong>
                {followUpSelectableTools.map((tool) => {
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
              {selectedNode.kind === "model" ? (
                <div className="contract-card">
                  <strong>MCP Tools From Context Providers</strong>
                  <span>
                    Callable MCP tools: {modelCallableMcpTools.length > 0 ? modelCallableMcpTools.join(", ") : "None"}
                  </span>
                  <span>
                    Prompt context sources: {modelPromptContextProviders.length > 0 ? modelPromptContextProviders.join(", ") : "None"}
                  </span>
                  {modelTargetedMcpNodeIds.length > 0 ? (
                    <span>Targeted MCP provider IDs: {modelTargetedMcpNodeIds.join(", ")}</span>
                  ) : (
                    <span>MCP tools are supplied through connected or targeted MCP Context Provider nodes.</span>
                  )}
                </div>
              ) : null}
              {selectedNode.kind === "model" ? (
                <div className="contract-card">
                  <strong>Bound Prompt Blocks</strong>
                  <span>
                    Direct prompt messages:{" "}
                    {modelPromptBlockNodes.length > 0
                      ? modelPromptBlockNodes.map((node) => `${formatNodeLabel(node)} (${String(node.config.role ?? "user")})`).join(", ")
                      : "None"}
                  </span>
                  <span>Bind Prompt Block nodes into the model to inject additional system, user, or assistant messages before the standard user template.</span>
                </div>
              ) : null}
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
                  {standardCatalogTools.map((tool) => (
                    <option key={tool.name} value={tool.name}>
                      {tool.name} ({toolStatusLabel(tool)})
                    </option>
                  ))}
                </select>
              </label>
            </>
          ) : null}
          {selectedNode.kind === "mcp_context_provider" ? (
            <>
              <div className="inspector-meta">
                <span>Acts as a source-only context provider. No input connection is required.</span>
              </div>
              <div className="checkbox-grid">
                <strong>Registered MCP Tools</strong>
                {mcpCatalogTools.map((tool) => {
                  const isChecked = selectedMcpToolNames.includes(tool.name);
                  const canSelectTool = isToolEnabled(tool) && isToolOnline(tool);
                  return (
                    <label key={tool.name} className="checkbox-option">
                      <input
                        type="checkbox"
                        checked={isChecked}
                        disabled={!isChecked && !canSelectTool}
                        onChange={(event) => {
                          const nextTools = event.target.checked
                            ? [...selectedMcpToolNames, tool.name]
                            : selectedMcpToolNames.filter((name) => name !== tool.name);
                          onGraphChange(
                            updateNode(graph, selectedNode.id, (node) => ({
                              ...node,
                              config: {
                                ...node.config,
                                tool_names: nextTools,
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
              <label className="checkbox-option">
                <input
                  type="checkbox"
                  checked={mcpToolExposureEnabled}
                  onChange={(event) =>
                    onGraphChange(
                      updateNode(graph, selectedNode.id, (node) => ({
                        ...node,
                        config: {
                          ...node.config,
                          expose_mcp_tools: event.target.checked,
                        },
                      })),
                    )
                  }
                />
                <span>
                  Expose MCP Tools To Connected API Nodes
                  <small>Makes the selected MCP tools callable by connected or targeted API/model nodes when the tools are enabled and online.</small>
                </span>
              </label>
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
                  Inject MCP Prompt Context Into Connected API Nodes
                  <small>Adds descriptive MCP tool metadata to the connected model system prompt. This does not control tool callability.</small>
                </span>
              </label>
            </>
          ) : null}
          {selectedNode.kind === "mcp_tool_executor" ? (
            <>
              <label className="checkbox-option">
                <input
                  type="checkbox"
                  checked={executorFollowUpEnabled}
                  onChange={(event) =>
                    onGraphChange(
                      updateNode(graph, selectedNode.id, (node) => ({
                        ...node,
                        config: {
                          ...node.config,
                          enable_follow_up_decision: event.target.checked,
                        },
                      })),
                    )
                  }
                />
                <span>
                  Enable Follow-Up Decision
                  <small>Let the executor inspect each MCP result with a model and decide whether to stop or call another exposed MCP tool.</small>
                </span>
              </label>
              {executorFollowUpEnabled ? (
                <label className="checkbox-option">
                  <input
                    type="checkbox"
                    checked={Boolean(selectedNode.config.validate_last_tool_success ?? true)}
                    onChange={(event) =>
                      onGraphChange(
                        updateNode(graph, selectedNode.id, (node) => ({
                          ...node,
                          config: {
                            ...node.config,
                            validate_last_tool_success: event.target.checked,
                          },
                        })),
                      )
                    }
                  />
                  <span>
                    Stop On Failed Tool Result
                    <small>When enabled, the executor will not ask the follow-up model for another MCP tool after a failed tool execution.</small>
                  </span>
                </label>
              ) : null}
              <div className="inspector-meta">
                <span>Dispatch mode: single MCP tool call from upstream API output</span>
                <span>Input binding: {executorBindingSummary}</span>
                <span>
                  Follow-up decision: {executorFollowUpEnabled ? "enabled via internal model loop" : "disabled"}
                </span>
                <span>Routes: on finish / on failure / terminal output</span>
              </div>
            </>
          ) : null}
          {selectedNode.kind === "data" ? (
            <>
              {selectedNode.provider_id === "core.data_display" ? (
                <div className="inspector-meta">
                  <span>Display mode: visualizer envelope inspection</span>
                  <span>Behavior: passes the original payload through unchanged</span>
                  <span>Visualizer: shows the full incoming envelope under node output details</span>
                </div>
              ) : isPromptBlockDataNode ? (
                <>
                  <div className="contract-card">
                    <strong>Prompt Block</strong>
                    <span>Creates one binding-only prompt message that can feed a Context Builder or bind directly into a model.</span>
                    <span>Role determines whether the block is treated as a system, user, or assistant message downstream.</span>
                  </div>
                  <div className="context-builder-placeholder-bar">
                    <button
                      type="button"
                      className="secondary-button context-builder-inline-button"
                      onClick={() =>
                        onGraphChange(
                          updateNode(graph, selectedNode.id, (node) => ({
                            ...node,
                            config: {
                              ...node.config,
                              content: String(
                                node.config.content ??
                                  PROMPT_BLOCK_STARTERS[String(node.config.role ?? "user")] ??
                                  PROMPT_BLOCK_STARTERS.user,
                              ).trim().length > 0
                                ? node.config.content
                                : PROMPT_BLOCK_STARTERS[String(node.config.role ?? "user")] ?? PROMPT_BLOCK_STARTERS.user,
                              mode: "prompt_block",
                            },
                          })),
                        )
                      }
                    >
                      Insert Starter
                    </button>
                    {promptBlockAvailableVariables.map((token) => (
                      <button
                        key={token}
                        type="button"
                        className="secondary-button context-builder-token-button"
                        onClick={() =>
                          onGraphChange(
                            updateNode(graph, selectedNode.id, (node) => ({
                              ...node,
                              config: {
                                ...node.config,
                                content: insertTokenAtEnd(String(node.config.content ?? ""), `{${token}}`),
                                mode: "prompt_block",
                              },
                            })),
                          )
                        }
                      >
                        {`{${token}}`}
                      </button>
                    ))}
                  </div>
                  <label>
                    Message Role
                    <select
                      value={String(selectedNode.config.role ?? "user")}
                      onChange={(event) =>
                        onGraphChange(
                          updateNode(graph, selectedNode.id, (node) => ({
                            ...node,
                            config: { ...node.config, role: event.target.value, mode: "prompt_block" },
                          })),
                        )
                      }
                    >
                      <option value="system">system</option>
                      <option value="user">user</option>
                      <option value="assistant">assistant</option>
                    </select>
                  </label>
                  <label>
                    Message Name
                    <input
                      value={String(selectedNode.config.name ?? "")}
                      placeholder="Optional label for the message block"
                      onChange={(event) =>
                        onGraphChange(
                          updateNode(graph, selectedNode.id, (node) => ({
                            ...node,
                            config: { ...node.config, name: event.target.value, mode: "prompt_block" },
                          })),
                        )
                      }
                    />
                  </label>
                  <label>
                    Message Content
                    <textarea
                      rows={6}
                      value={String(selectedNode.config.content ?? "")}
                      placeholder="Enter the message content to inject into downstream prompt assembly."
                      onChange={(event) =>
                        onGraphChange(
                          updateNode(graph, selectedNode.id, (node) => ({
                            ...node,
                            config: { ...node.config, content: event.target.value, mode: "prompt_block" },
                          })),
                        )
                      }
                    />
                  </label>
                  <div className="inspector-meta">
                    <span>Draft: {getPromptBlockPreview(selectedNode) || "Add content to preview this prompt block."}</span>
                    <span>Binding mode: source-only prompt block</span>
                    <span>Available variables: {promptBlockAvailableVariables.length > 0 ? promptBlockAvailableVariables.join(", ") : "None"}</span>
                  </div>
                  <div className="contract-card">
                    <strong>{runState ? "Rendered Preview" : "Template Preview"}</strong>
                    <pre className="context-builder-preview">{promptBlockRenderedPreview || "Prompt block output will appear here."}</pre>
                    <span>
                      {runState
                        ? "Preview uses the latest run input and graph variables."
                        : "Preview shows the prompt template structure until the graph has run."}
                    </span>
                  </div>
                </>
              ) : selectedNode.provider_id === CONTEXT_BUILDER_PROVIDER_ID ? (
                <>
                  <div className="contract-card">
                    <strong>Context Builder</strong>
                    <span>Connect any number of upstream text nodes, rename their placeholders, and compose one prompt block.</span>
                    <span>
                      Connected inputs: {contextBuilderBindings.length > 0 ? String(contextBuilderBindings.length) : "None yet"}
                    </span>
                  </div>
                  <div className="checkbox-grid">
                    <strong>Connected Inputs</strong>
                    {contextBuilderBindings.length > 0 ? (
                      contextBuilderBindings.map((binding, index) => {
                        const sourcePreview = getContextBuilderSourcePreviewFromGraph(graph, runState, binding.sourceNodeId);
                        return (
                          <div key={binding.sourceNodeId} className="context-builder-binding-card">
                            <div className="context-builder-binding-header">
                              <div>
                                <strong>{binding.sourceLabel}</strong>
                                <small>{binding.sourceNodeId}</small>
                              </div>
                              <div className="context-builder-binding-actions">
                                <button
                                  type="button"
                                  className="secondary-button context-builder-inline-button"
                                  disabled={index === 0}
                                  onClick={() => {
                                    if (index === 0) {
                                      return;
                                    }
                                    const nextBindings = [...contextBuilderBindings];
                                    [nextBindings[index - 1], nextBindings[index]] = [nextBindings[index], nextBindings[index - 1]];
                                    updateContextBuilderBindings(nextBindings);
                                  }}
                                >
                                  Up
                                </button>
                                <button
                                  type="button"
                                  className="secondary-button context-builder-inline-button"
                                  disabled={index === contextBuilderBindings.length - 1}
                                  onClick={() => {
                                    if (index === contextBuilderBindings.length - 1) {
                                      return;
                                    }
                                    const nextBindings = [...contextBuilderBindings];
                                    [nextBindings[index], nextBindings[index + 1]] = [nextBindings[index + 1], nextBindings[index]];
                                    updateContextBuilderBindings(nextBindings);
                                  }}
                                >
                                  Down
                                </button>
                              </div>
                            </div>
                            <label>
                              Placeholder
                              <input
                                value={binding.placeholder}
                                onChange={(event) => {
                                  const nextBindings = contextBuilderBindings.map((candidate) =>
                                    candidate.sourceNodeId === binding.sourceNodeId
                                      ? {
                                          ...candidate,
                                          placeholder: slugifyContextBuilderPlaceholder(
                                            event.target.value,
                                            candidate.sourceLabel,
                                          ),
                                          autoGenerated: false,
                                        }
                                      : candidate,
                                  );
                                  updateContextBuilderBindings(nextBindings);
                                }}
                              />
                            </label>
                            <div className="inspector-meta">
                              <span>Token: {`{${binding.placeholder}}`}</span>
                              <span>{binding.autoGenerated ? "Auto-generated from the source label" : "Custom placeholder"}</span>
                              <span>
                                Preview: {sourcePreview !== null && sourcePreview.length > 0 ? sourcePreview : "Run the graph to preview this source."}
                              </span>
                            </div>
                          </div>
                        );
                      })
                    ) : (
                      <p className="inspector-hint">Connect upstream nodes to start building named placeholders.</p>
                    )}
                  </div>
                  <div className="context-builder-placeholder-bar">
                    {contextBuilderBindings.map((binding) => (
                      <button
                        key={binding.sourceNodeId}
                        type="button"
                        className="secondary-button context-builder-token-button"
                        onClick={() =>
                          updateContextBuilderTemplate(
                            contextBuilderTemplate.trim().length > 0
                              ? `${contextBuilderTemplate}${contextBuilderTemplate.endsWith("\n") ? "" : "\n"}{${binding.placeholder}}`
                              : `{${binding.placeholder}}`,
                          )
                        }
                      >
                        {`{${binding.placeholder}}`}
                      </button>
                    ))}
                    {contextBuilderBindings.length > 0 ? (
                      <button
                        type="button"
                        className="secondary-button context-builder-inline-button"
                        onClick={() => updateContextBuilderTemplate(generatedContextBuilderTemplate)}
                      >
                        Regenerate Template
                      </button>
                    ) : null}
                  </div>
                  <label>
                    Prompt Template
                    <textarea
                      rows={8}
                      value={contextBuilderTemplate}
                      placeholder="Connect upstream nodes to insert placeholders here."
                      onChange={(event) => updateContextBuilderTemplate(event.target.value)}
                    />
                  </label>
                  <div className="inspector-meta">
                    <span>
                      Available variables:{" "}
                      {contextBuilderAvailableVariables.size > 0
                        ? [...contextBuilderAvailableVariables].join(", ")
                        : "Connect sources to create placeholders."}
                    </span>
                    {contextBuilderDuplicatePlaceholders.length > 0 ? (
                      <span>Duplicate placeholders: {uniqueStrings(contextBuilderDuplicatePlaceholders).join(", ")}</span>
                    ) : null}
                    {contextBuilderInvalidPlaceholders.length > 0 ? (
                      <span>Invalid placeholders: {uniqueStrings(contextBuilderInvalidPlaceholders).join(", ")}</span>
                    ) : null}
                    {contextBuilderUnmatchedTokens.length > 0 ? (
                      <span>Template tokens without a source: {uniqueStrings(contextBuilderUnmatchedTokens).join(", ")}</span>
                    ) : null}
                  </div>
                  <div className="contract-card">
                    <strong>{contextBuilderHasPreviewData ? "Rendered Preview" : "Template Preview"}</strong>
                    <pre className="context-builder-preview">{contextBuilderRenderedPreview || "Template output will appear here."}</pre>
                    <span>
                      {contextBuilderHasPreviewData
                        ? "Preview uses the latest run outputs from connected nodes."
                        : "Preview shows template structure until the graph has run."}
                    </span>
                  </div>
                </>
              ) : (
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
              )}
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
              <option value="binding">binding</option>
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
                  {formatNodeLabel(node)}
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
