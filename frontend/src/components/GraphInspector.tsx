import { useEffect, useState } from "react";

import { previewSpreadsheetRows } from "../lib/api";
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
import { getGraphEnvVars, resolveGraphEnvReferences } from "../lib/graphEnv";
import {
  getContextBuilderBindings,
  slugifyContextBuilderPlaceholder,
  type ContextBuilderBindingRow,
} from "../lib/contextBuilderBindings";
import { getNodeInstanceLabel } from "../lib/nodeInstanceLabels";
import { insertTokenAtEnd, listPromptBlockAvailableVariables, PROMPT_BLOCK_STARTERS, renderPromptBlockPreview } from "../lib/promptBlockEditor";
import { resolveToolNodeDetails } from "../lib/toolNodeDetails";
import { useRenderDiagnostics } from "../lib/dragDiagnostics";
import type {
  EditorCatalog,
  GraphDefinition,
  GraphEdge,
  GraphNode,
  RunState,
  SpreadsheetPreviewResult,
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

function toolCanonicalName(tool: ToolDefinition): string {
  return tool.canonical_name ?? tool.name;
}

function toolLabel(tool: ToolDefinition): string {
  return tool.display_name ?? tool.name;
}

function toolMatchesReference(tool: ToolDefinition, reference: string): boolean {
  const normalizedReference = reference.trim();
  if (!normalizedReference) {
    return false;
  }
  return [toolCanonicalName(tool), tool.name, ...(tool.aliases ?? [])].includes(normalizedReference);
}

function uniqueStrings(values: string[]): string[] {
  return [...new Set(values.filter((value) => value.trim().length > 0))];
}

const CONTEXT_BUILDER_PROVIDER_ID = "core.context_builder";
const SPREADSHEET_ROW_PROVIDER_ID = "core.spreadsheet_rows";
const CONTEXT_BUILDER_IDENTIFIER_PATTERN = /^[A-Za-z_][A-Za-z0-9_]*$/;
const CONTEXT_BUILDER_TOKEN_PATTERN = /\{([A-Za-z_][A-Za-z0-9_]*)\}/g;
const CONTEXT_BUILDER_BASE_VARIABLES = ["current_node_id", "documents", "graph_id", "input_payload", "run_id"];

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

function buildPromptOnlyMcpToolDecisionContract(hasToolContext: boolean, callableToolNames: string[]): string {
  if (!hasToolContext || callableToolNames.length > 0) {
    return "";
  }
  return [
    "MCP Tool Decision Output",
    "When MCP tool metadata is present in prompt context but no MCP tools are directly callable, you must respond using this exact structure:",
    "",
    "Uses Tool: True|False",
    'Tool Call Schema: {"tool_name":"<tool name>","arguments":{...}} or NA',
    "DELIMITER",
    "<Explain why the tool schema is needed or why no tool is needed, and describe the next step required to finish the user's request.>",
    "",
    "Rules",
    "- Set `Uses Tool` to `True` only when one of the tools described in the MCP Tool Context is required.",
    "- When `Uses Tool` is `True`, `Tool Call Schema` must be a single JSON object containing exactly `tool_name` and `arguments`.",
    "- When `Uses Tool` is `False`, `Tool Call Schema` must be `NA`.",
    "- Do not claim that you already called a tool unless you were given an actual tool result.",
    "- The content after `DELIMITER` must be plain-language guidance for the next processing step.",
  ].join("\n");
}

function buildMcpToolGuidanceBlock(callableToolNames: string[], guidanceText: string): string {
  let guidanceLines: string[] = [];
  if (callableToolNames.length > 0) {
    guidanceLines = [
      "MCP Tool Guidance",
      "Use MCP tools only when a listed live capability is needed to answer the request or complete the task.",
      "Call only MCP tools that are explicitly exposed to you and follow their schemas exactly.",
      "Do not invent MCP tool names or arguments.",
      "If no exposed MCP tool is necessary, continue without calling one.",
    ];
  }
  if (guidanceText.trim().length > 0) {
    guidanceLines = guidanceLines.length > 0
      ? [...guidanceLines, "", "Connected MCP Tool Notes:", guidanceText]
      : ["MCP Tool Guidance", guidanceText];
  }
  return guidanceLines.join("\n").trim();
}

function mcpToolPlaceholderToken(index: number): string {
  return `MCP_TOOL_${index + 1}`;
}

function buildMcpToolPlaceholderTemplate(
  tools: Array<{
    placeholderToken: string;
    displayName: string;
  }>,
): string {
  if (tools.length === 0) {
    return "";
  }
  return [
    "You are a tool calling assistant.",
    "",
    "{mcp_tool_guidance_block}",
    "",
    "You have these tools:",
    "",
    ...tools.map((tool) => `# ${tool.displayName}\n{${tool.placeholderToken}}`),
  ].join("\n\n");
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
  const [spreadsheetPreview, setSpreadsheetPreview] = useState<SpreadsheetPreviewResult | null>(null);
  const [spreadsheetPreviewError, setSpreadsheetPreviewError] = useState<string | null>(null);
  const [isSpreadsheetPreviewLoading, setIsSpreadsheetPreviewLoading] = useState(false);

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
  const spreadsheetPreviewKey =
    selectedNode?.provider_id === SPREADSHEET_ROW_PROVIDER_ID
      ? JSON.stringify({
          id: selectedNode.id,
          file_format: selectedNode.config.file_format ?? "auto",
          file_path: selectedNode.config.file_path ?? "",
          sheet_name: selectedNode.config.sheet_name ?? "",
          header_row_index: selectedNode.config.header_row_index ?? 1,
          start_row_index: selectedNode.config.start_row_index ?? 2,
          empty_row_policy: selectedNode.config.empty_row_policy ?? "skip",
        })
      : "none";
  useEffect(() => {
    setSpreadsheetPreview(null);
    setSpreadsheetPreviewError(null);
    setIsSpreadsheetPreviewLoading(false);
  }, [spreadsheetPreviewKey]);
  const formatNodeLabel = (node: GraphNode) => getNodeInstanceLabel(graph, node);
  const executorFollowUpEnabled =
    selectedNode?.kind === "mcp_tool_executor" && Boolean(selectedNode.config.enable_follow_up_decision);
  const executorRetriesEnabled =
    selectedNode?.kind === "mcp_tool_executor" && Boolean(selectedNode.config.allow_retries ?? true);
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
    const mcpToolByName = new Map<string, ToolDefinition>();
    for (const tool of mcpCatalogTools) {
      for (const identifier of [toolCanonicalName(tool), tool.name, ...(tool.aliases ?? [])]) {
        const normalizedIdentifier = String(identifier).trim();
        if (normalizedIdentifier && !mcpToolByName.has(normalizedIdentifier)) {
          mcpToolByName.set(normalizedIdentifier, tool);
        }
      }
    }
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
              return `${tool ? toolLabel(tool) : toolName} (${status}) via ${formatNodeLabel(node)}`;
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
    const modelPromptGuidanceProviders =
      isPromptDrivenNode
        ? uniqueStrings(
            mcpContextProvidersForModel
              .filter(
                (node) =>
                  Boolean(node.config.include_mcp_tool_context) && String(node.config.usage_hint ?? "").trim().length > 0,
              )
              .map((node) => formatNodeLabel(node)),
          )
        : [];
    const modelTargetedMcpNodeIds =
      selectedNode.kind === "model" && Array.isArray(selectedNode.config.tool_target_node_ids)
        ? uniqueStrings(selectedNode.config.tool_target_node_ids.map((nodeId) => String(nodeId)))
        : [];
    const modelPromptBlockNodes = selectedNode.kind === "model" ? getModelPromptBlockNodes(graph, selectedNode) : [];
    const modelMetadataBindingKeys =
      selectedNode.kind === "model" && isRecord(selectedNode.config.metadata_bindings)
        ? uniqueStrings(Object.keys(selectedNode.config.metadata_bindings).map((key) => String(key)))
        : [];
    const graphEnvVars = getGraphEnvVars(graph);
    const modelSystemPromptTemplate = selectedNode.kind === "model" ? String(selectedNode.config.system_prompt ?? "") : "";
    const modelSystemPromptTokens = selectedNode.kind === "model" ? extractTemplateTokens(modelSystemPromptTemplate) : [];
    const modelDirectRegistryToolSummaries =
      selectedNode.kind === "model"
        ? allowedTools
            .map((toolName) => {
              const tool = standardCatalogTools.find((candidate) => toolMatchesReference(candidate, toolName));
              if (!tool) {
                return {
                  label: toolName,
                  canonicalName: toolName,
                  status: "unknown",
                };
              }
              return {
                label: toolLabel(tool),
                canonicalName: toolCanonicalName(tool),
                status: toolStatusLabel(tool),
              };
            })
            .filter((tool) => tool.canonicalName.trim().length > 0)
        : [];
    const findPromptOverrideNodeForTool = (toolName: string): GraphNode | null => {
      for (const candidate of graph.nodes) {
        if (candidate.kind !== "tool" && candidate.kind !== "mcp_context_provider") {
          continue;
        }
        const configuredToolNames = Array.isArray(candidate.config.tool_names)
          ? candidate.config.tool_names.map((value) => String(value))
          : [];
        const configuredToolName = typeof candidate.config.tool_name === "string" ? [candidate.config.tool_name] : [];
        const candidateToolNames = [...configuredToolNames, ...configuredToolName];
        if (candidateToolNames.some((configuredName) => toolMatchesReference({ name: toolName, description: "", input_schema: {} }, configuredName) || configuredName.trim() === toolName.trim())) {
          return candidate;
        }
        const matchingCatalogTool = catalogTools.find((tool) => toolMatchesReference(tool, toolName));
        if (matchingCatalogTool && candidateToolNames.some((configuredName) => toolMatchesReference(matchingCatalogTool, configuredName))) {
          return candidate;
        }
      }
      return null;
    };
    const modelPromptToolSummaries =
      selectedNode.kind === "model"
        ? mcpContextProvidersForModel.flatMap((node) => {
            const nodeToolNames = Array.isArray(node.config.tool_names)
              ? node.config.tool_names.map((toolName) => String(toolName)).filter((toolName) => toolName.trim().length > 0)
              : [];
            const sourceNodeLabel = formatNodeLabel(node);
            const usageHint = String(node.config.usage_hint ?? "").trim();
            const injectsPromptContext = Boolean(node.config.include_mcp_tool_context);
            const isCallableSource = node.config.expose_mcp_tools !== false;
            return nodeToolNames.map((toolName) => {
              const tool = mcpToolByName.get(toolName) ?? null;
              const canonicalName = tool ? toolCanonicalName(tool) : toolName;
              const overrideNode = findPromptOverrideNodeForTool(canonicalName) ?? node;
              const previewNode: GraphNode = {
                ...overrideNode,
                tool_name: canonicalName,
                config: {
                  ...overrideNode.config,
                  tool_name: canonicalName,
                  tool_names: [canonicalName],
                },
              };
              const resolvedDetails = resolveToolNodeDetails(previewNode, catalog, graph);
              const status = tool ? toolStatusLabel(tool) : "unknown";
              return {
                sourceNodeId: node.id,
                sourceNodeLabel,
                overrideNodeLabel: formatNodeLabel(overrideNode),
                toolName: canonicalName,
                displayName: tool ? toolLabel(tool) : canonicalName,
                status,
                isCallable: isCallableSource && status === "ready",
                injectsPromptContext,
                usageHint,
                renderedPromptText: resolvedDetails.renderedPromptText,
                templateText: resolvedDetails.templateText,
                descriptionText: resolvedDetails.agentDescriptionText,
              };
            });
          })
        : [];
    const modelPromptContextToolSummaries =
      selectedNode.kind === "model"
        ? modelPromptToolSummaries
            .filter((tool) => tool.injectsPromptContext)
            .map((tool, index) => ({
              ...tool,
              placeholderToken: mcpToolPlaceholderToken(index),
            }))
        : [];
    const modelGeneratedMcpPlaceholderTemplate =
      selectedNode.kind === "model" ? buildMcpToolPlaceholderTemplate(modelPromptContextToolSummaries) : "";
    const modelPromptGuidanceBlocks =
      selectedNode.kind === "model"
        ? mcpContextProvidersForModel
            .filter(
              (node) =>
                Boolean(node.config.include_mcp_tool_context) &&
                String(node.config.usage_hint ?? "").trim().length > 0 &&
                Array.isArray(node.config.tool_names) &&
                node.config.tool_names.length > 0,
            )
            .map((node) => {
              const toolNames = (node.config.tool_names as unknown[])
                .map((toolName) => String(toolName))
                .filter((toolName) => toolName.trim().length > 0)
                .map((toolName) => mcpToolByName.get(toolName))
                .filter((tool): tool is ToolDefinition => tool !== undefined)
                .map((tool) => toolLabel(tool));
              const dedupedToolNames = uniqueStrings(toolNames);
              const usageHint = String(node.config.usage_hint ?? "").trim();
              if (!usageHint || dedupedToolNames.length === 0) {
                return "";
              }
              return [`Tools: ${dedupedToolNames.join(", ")}`, "Guidance:", usageHint].join("\n");
            })
            .filter((block) => block.trim().length > 0)
        : [];
    const modelMcpToolContextPrompt =
      selectedNode.kind === "model"
        ? modelPromptContextToolSummaries.map((tool) => tool.renderedPromptText.trim()).filter((text) => text.length > 0).join("\n\n")
        : "";
    const modelCallableMcpToolNames =
      selectedNode.kind === "model"
        ? uniqueStrings(modelPromptToolSummaries.filter((tool) => tool.isCallable).map((tool) => tool.toolName)).sort()
        : [];
    const modelPromptVariables =
      selectedNode.kind === "model"
        ? uniqueStrings([
            ...Object.keys(graphEnvVars),
            "documents",
            "input_payload",
            "run_id",
            "graph_id",
            "current_node_id",
            "available_tools",
            "mcp_available_tool_names",
            "mcp_tool_context",
            "mcp_tool_context_prompt",
            "mcp_tool_context_block",
            "mcp_tool_guidance",
            "mcp_tool_guidance_block",
            "mode",
            "preferred_tool_name",
            "response_mode",
            "prompt_blocks",
            ...modelPromptContextToolSummaries.map((tool) => tool.placeholderToken),
            ...modelMetadataBindingKeys,
          ]).sort()
        : [];
    const modelPreviewVariableValues: Record<string, string> =
      selectedNode.kind === "model"
        ? {
            documents: stringifyPreviewValue(runState?.documents ?? []),
            input_payload: stringifyPreviewValue(runState?.input_payload ?? ""),
            run_id: runState?.run_id ?? "",
            graph_id: graph.graph_id,
            current_node_id: selectedNode.id,
            available_tools: JSON.stringify(
              [
                ...modelDirectRegistryToolSummaries.map((tool) => ({
                  name: tool.canonicalName,
                  description: tool.label,
                  status: tool.status,
                })),
                ...modelPromptToolSummaries
                  .filter((tool) => tool.isCallable)
                  .map((tool) => ({
                    name: tool.toolName,
                    description: tool.descriptionText,
                    status: tool.status,
                  })),
              ],
              null,
              2,
            ),
            mcp_available_tool_names: JSON.stringify(modelCallableMcpToolNames, null, 2),
            mcp_tool_context: JSON.stringify(
              {
                tool_names: modelPromptContextToolSummaries.map((tool) => tool.toolName),
                prompt_blocks: modelPromptContextToolSummaries.map((tool) => tool.renderedPromptText),
                usage_hints_text: modelPromptGuidanceBlocks.join("\n\n"),
              },
              null,
              2,
            ),
            mcp_tool_context_prompt: modelMcpToolContextPrompt,
            mcp_tool_guidance: modelPromptGuidanceBlocks.join("\n\n"),
            mcp_tool_context_block: modelMcpToolContextPrompt.trim().length > 0
              ? `MCP Tool Context\n${modelMcpToolContextPrompt}`
              : "",
            mcp_tool_guidance_block: buildMcpToolGuidanceBlock(
              modelCallableMcpToolNames,
              modelPromptGuidanceBlocks.join("\n\n"),
            ),
            mode: String(selectedNode.config.mode ?? selectedNode.prompt_name ?? ""),
            preferred_tool_name: String(selectedNode.config.preferred_tool_name ?? ""),
            response_mode: selectedModelResponseMode ?? "message",
            prompt_blocks: JSON.stringify(
              modelPromptBlockNodes.map((node) => ({
                role: String(node.config.role ?? "user"),
                name: String(node.config.name ?? ""),
                content: renderPromptBlockPreview(node, graph, runState),
              })),
              null,
              2,
            ),
            ...Object.fromEntries(modelPromptContextToolSummaries.map((tool) => [tool.placeholderToken, tool.renderedPromptText])),
            ...Object.fromEntries(modelMetadataBindingKeys.map((key) => [key, `[bound at runtime: ${key}]`])),
          }
        : {};
    const modelSystemPromptTemplatePreview =
      selectedNode.kind === "model"
        ? resolveGraphEnvReferences(modelSystemPromptTemplate, graph, modelPreviewVariableValues)
        : "";
    const modelPromptOnlyToolContract =
      selectedNode.kind === "model"
        ? buildPromptOnlyMcpToolDecisionContract(modelPromptContextToolSummaries.length > 0, modelCallableMcpToolNames)
        : "";
    const modelMcpToolPlaceholderTokens =
      selectedNode.kind === "model" ? modelPromptContextToolSummaries.map((tool) => tool.placeholderToken) : [];
    const modelHasInlineMcpGuidanceBlock =
      selectedNode.kind === "model" && modelSystemPromptTemplate.includes("{mcp_tool_guidance_block}");
    const modelHasInlineMcpContextCoverage =
      selectedNode.kind === "model" && (
        modelSystemPromptTemplate.includes("{mcp_tool_context_block}")
        || modelSystemPromptTemplate.includes("{mcp_tool_context_prompt}")
        || (
          modelMcpToolPlaceholderTokens.length > 0
          && modelMcpToolPlaceholderTokens.every((token) => modelSystemPromptTemplate.includes(`{${token}}`))
        )
      );
    const modelOptionalPromptVariables =
      selectedNode.kind === "model"
        ? modelPromptVariables.filter(
            (token) => !["mcp_tool_guidance_block", "mcp_tool_context_block", "mcp_tool_context_prompt", ...modelMcpToolPlaceholderTokens].includes(token),
          )
        : [];
    const modelMcpGuidanceBlock =
      selectedNode.kind === "model"
        ? String(modelPreviewVariableValues.mcp_tool_guidance_block ?? "").trim()
        : "";
    const modelMcpContextBlock =
      selectedNode.kind === "model"
        ? String(modelPreviewVariableValues.mcp_tool_context_block ?? "").trim()
        : "";
    const modelPromptAssemblySections =
      selectedNode.kind === "model"
        ? (() => {
            const sections: string[] = [];
            const hasInlineMcpGuidanceBlock = modelSystemPromptTemplate.includes("{mcp_tool_guidance_block}");
            const hasInlineMcpContextCoverage =
              modelSystemPromptTemplate.includes("{mcp_tool_context_block}")
              || modelSystemPromptTemplate.includes("{mcp_tool_context_prompt}")
              || (
                modelPromptContextToolSummaries.length > 0
                && modelPromptContextToolSummaries.every((tool) => modelSystemPromptTemplate.includes(`{${tool.placeholderToken}}`))
              );
            if (modelMcpGuidanceBlock.length > 0 && !hasInlineMcpGuidanceBlock) {
              sections.push(modelMcpGuidanceBlock);
            }
            if (modelMcpContextBlock.length > 0 && !hasInlineMcpContextCoverage) {
              sections.push(modelMcpContextBlock);
            }
            if (modelPromptOnlyToolContract.trim().length > 0) {
              sections.push(modelPromptOnlyToolContract);
            }
            return sections;
          })()
        : [];
    const modelFinalSystemPromptPreview =
      selectedNode.kind === "model"
        ? [modelSystemPromptTemplatePreview.trim(), ...modelPromptAssemblySections.map((section) => section.trim()).filter((section) => section.length > 0)]
            .filter((section) => section.length > 0)
            .join("\n\n")
        : "";
    const mcpToolExposureEnabled = selectedNode.kind === "mcp_context_provider" ? selectedNode.config.expose_mcp_tools !== false : false;
    const executorBindingSummary = selectedNode.kind === "mcp_tool_executor" ? describeMcpExecutorBinding(selectedNode.config.input_binding) : "";
    const executorFollowUpResponseMode =
      selectedNode.kind === "mcp_tool_executor" ? String(selectedNode.config.response_mode ?? "auto") : "auto";
    const isDiscordStartNode = selectedNode.kind === "input" && selectedNode.provider_id === "start.discord_message";
    const isDiscordEndNode = selectedNode.kind === "output" && selectedNode.provider_id === "end.discord_message";
    const isManualStartNode =
      selectedNode.kind === "input" &&
      (selectedNode.provider_id === "start.manual_run" || selectedNode.provider_id === "core.input");
    const isContextBuilderNode = selectedNode.kind === "data" && selectedNode.provider_id === CONTEXT_BUILDER_PROVIDER_ID;
    const isPromptBlockDataNode = selectedNode.kind === "data" && selectedNode.provider_id === PROMPT_BLOCK_PROVIDER_ID;
    const isSpreadsheetRowNode = selectedNode.kind === "data" && selectedNode.provider_id === SPREADSHEET_ROW_PROVIDER_ID;
    const spreadsheetNode = isSpreadsheetRowNode ? selectedNode : null;
    const spreadsheetIteratorState =
      isSpreadsheetRowNode && runState?.iterator_states
        ? (runState.iterator_states[selectedNode.id] as Record<string, unknown> | undefined)
        : undefined;
    const spreadsheetResolvedFilePath =
      spreadsheetNode ? String(resolveGraphEnvReferences(String(spreadsheetNode.config.file_path ?? ""), graph) ?? "") : "";
    async function handleSpreadsheetPreview(): Promise<void> {
      if (!spreadsheetNode) {
        return;
      }
      setIsSpreadsheetPreviewLoading(true);
      setSpreadsheetPreviewError(null);
      try {
        const preview = await previewSpreadsheetRows({
          file_path: spreadsheetResolvedFilePath,
          file_format: String(spreadsheetNode.config.file_format ?? "auto"),
          sheet_name: String(spreadsheetNode.config.sheet_name ?? "") || null,
          header_row_index: Number(spreadsheetNode.config.header_row_index ?? 1) || 1,
          start_row_index: Number(spreadsheetNode.config.start_row_index ?? 2) || 2,
          empty_row_policy: String(spreadsheetNode.config.empty_row_policy ?? "skip"),
        });
        setSpreadsheetPreview(preview);
      } catch (error) {
        setSpreadsheetPreview(null);
        setSpreadsheetPreviewError(error instanceof Error ? error.message : "Failed to preview spreadsheet rows.");
      } finally {
        setIsSpreadsheetPreviewLoading(false);
      }
    }
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
          {selectedNode.kind === "output" ? (
            <>
              <div className="contract-card">
                <strong>{isDiscordEndNode ? "Discord Side-Effect End" : "Canonical Output End"}</strong>
                <span>
                  {isDiscordEndNode
                    ? "Sends the resolved payload to a Discord channel and leaves run final_output unchanged."
                    : "Promotes the resolved payload into the run final_output when this branch completes."}
                </span>
              </div>
              {isDiscordEndNode ? (
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
                              discord_channel_id: event.target.value,
                            },
                          })),
                        )
                      }
                    />
                  </label>
                  <label>
                    Message Template
                    <textarea
                      rows={4}
                      value={String(selectedNode.config.message_template ?? "{message_payload}")}
                      placeholder="{message_payload}"
                      onChange={(event) =>
                        onGraphChange(
                          updateNode(graph, selectedNode.id, (node) => ({
                            ...node,
                            config: {
                              ...node.config,
                              message_template: event.target.value,
                            },
                          })),
                        )
                      }
                    />
                  </label>
                  <div className="contract-card">
                    <strong>Template Variables</strong>
                    <span><code>{"{message_payload}"}</code> renders the resolved payload as text.</span>
                    <span><code>{"{message_json}"}</code> renders JSON for structured payloads.</span>
                  </div>
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
                {selectedNode.kind === "model" && modelPromptContextToolSummaries.length > 0 ? (
                  <div className="context-builder-binding-actions">
                    <button
                      type="button"
                      className="secondary-button context-builder-inline-button"
                      onClick={() =>
                        onGraphChange(
                          updateNode(graph, selectedNode.id, (node) => ({
                            ...node,
                            config: {
                              ...node.config,
                              system_prompt: modelGeneratedMcpPlaceholderTemplate,
                            },
                          })),
                        )
                      }
                    >
                      Build From Connected MCP Tools
                    </button>
                  </div>
                ) : null}
                {selectedNode.kind === "model" && (modelCallableMcpToolNames.length > 0 || modelPromptGuidanceProviders.length > 0) ? (
                  <small>
                    Connected MCP edges already guarantee which tools are in scope. For full inline MCP coverage, include{" "}
                    <code>{"{mcp_tool_guidance_block}"}</code> and either <code>{"{mcp_tool_context_block}"}</code> or
                    every ordered tool placeholder below. Missing MCP sections are appended automatically.
                  </small>
                ) : null}
                {selectedNode.kind === "model" && modelPromptContextToolSummaries.length > 0 ? (
                  <small>
                    Ordered MCP tool placeholders:{" "}
                    {modelPromptContextToolSummaries.map((tool) => (
                      <code key={tool.placeholderToken}>{`{${tool.placeholderToken}}`}</code>
                    ))}{" "}
                    resolve inline at runtime.
                  </small>
                ) : null}
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
                    <select
                      value={String(selectedNode.config.response_mode ?? "auto") || "auto"}
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
                    <small>
                      Choose a fixed mode or leave it on `auto` to follow graph wiring. Current effective mode:{" "}
                      <code>{selectedModelResponseMode ?? "message"}</code>.
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
                  const canonicalName = toolCanonicalName(tool);
                  const isChecked = allowedTools.some((name) => toolMatchesReference(tool, name));
                  const canSelectTool = isToolEnabled(tool) && isToolOnline(tool);
                  return (
                    <label key={canonicalName} className="checkbox-option">
                      <input
                        type="checkbox"
                        checked={isChecked}
                        disabled={!isChecked && !canSelectTool}
                        onChange={(event) => {
                          const nextTools = event.target.checked
                            ? [...allowedTools.filter((name) => !toolMatchesReference(tool, name)), canonicalName]
                            : allowedTools.filter((name) => !toolMatchesReference(tool, name));
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
                        {toolLabel(tool)}
                        {toolLabel(tool) !== canonicalName ? <small><code>{canonicalName}</code></small> : null}
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
              {selectedNode.kind === "model" ? (
                <>
                  <div className="contract-card">
                    <strong>System Prompt Assembly</strong>
                    <span>
                      Template placeholders: {modelSystemPromptTokens.length > 0 ? modelSystemPromptTokens.join(", ") : "None"}
                    </span>
                    <span>
                      Required MCP guidance:{" "}
                      {modelMcpGuidanceBlock.length > 0
                        ? modelHasInlineMcpGuidanceBlock
                          ? "inline"
                          : "auto-appended"
                        : "not needed"}
                    </span>
                    <span>
                      Required MCP context:{" "}
                      {modelPromptContextToolSummaries.length > 0
                        ? modelHasInlineMcpContextCoverage
                          ? "inline"
                          : "auto-appended"
                        : "not needed"}
                    </span>
                    <span>
                      Prompt block messages stay separate from the system prompt:{" "}
                      {modelPromptBlockNodes.length > 0 ? `${modelPromptBlockNodes.length} bound block${modelPromptBlockNodes.length === 1 ? "" : "s"}` : "None"}
                    </span>
                  </div>
                  {modelGeneratedMcpPlaceholderTemplate ? (
                    <section className="tool-details-modal-preview">
                      <div className="tool-details-modal-preview-header">
                        <strong>Generated MCP Placeholder Template</strong>
                        <span>
                          This scaffold is built from connected MCP prompt-context edges and can replace the system prompt
                          with ordered placeholders before runtime.
                        </span>
                      </div>
                      <pre>{modelGeneratedMcpPlaceholderTemplate}</pre>
                    </section>
                  ) : null}
                  <div className="checkbox-grid">
                    <strong>Required MCP Placeholders</strong>
                    <span className="inspector-meta">
                      Necessary for full inline MCP control. If these are omitted, the runtime appends the missing MCP
                      sections automatically.
                    </span>
                    <div className="context-builder-placeholder-bar">
                      {modelMcpGuidanceBlock.length > 0 ? (
                        <button
                          type="button"
                          className="secondary-button context-builder-token-button context-builder-token-button--required"
                          onClick={() =>
                            onGraphChange(
                              updateNode(graph, selectedNode.id, (node) => ({
                                ...node,
                                config: {
                                  ...node.config,
                                  system_prompt: insertTokenAtEnd(String(node.config.system_prompt ?? ""), "{mcp_tool_guidance_block}"),
                                },
                              })),
                            )
                          }
                        >
                          mcp_tool_guidance_block
                        </button>
                      ) : null}
                      {modelPromptContextToolSummaries.length > 0 ? (
                        <button
                          type="button"
                          className="secondary-button context-builder-token-button context-builder-token-button--required"
                          onClick={() =>
                            onGraphChange(
                              updateNode(graph, selectedNode.id, (node) => ({
                                ...node,
                                config: {
                                  ...node.config,
                                  system_prompt: insertTokenAtEnd(String(node.config.system_prompt ?? ""), "{mcp_tool_context_block}"),
                                },
                              })),
                            )
                          }
                        >
                          mcp_tool_context_block
                        </button>
                      ) : null}
                      {modelMcpToolPlaceholderTokens.map((token) => (
                        <button
                          key={token}
                          type="button"
                          className="secondary-button context-builder-token-button context-builder-token-button--required"
                          onClick={() =>
                            onGraphChange(
                              updateNode(graph, selectedNode.id, (node) => ({
                                ...node,
                                config: {
                                  ...node.config,
                                  system_prompt: insertTokenAtEnd(String(node.config.system_prompt ?? ""), `{${token}}`),
                                },
                              })),
                            )
                          }
                        >
                          {token}
                        </button>
                      ))}
                    </div>
                    {modelPromptContextToolSummaries.length > 0 ? (
                      <span className="inspector-meta">
                        Context coverage can come from <code>{"{mcp_tool_context_block}"}</code> or from all ordered tool
                        placeholders.
                      </span>
                    ) : null}
                  </div>
                  <div className="checkbox-grid">
                    <strong>Optional Runtime Variables</strong>
                    <span className="inspector-meta">
                      These are available if you want to reference other runtime values explicitly.
                    </span>
                    <div className="context-builder-placeholder-bar">
                      {modelOptionalPromptVariables.map((token) => (
                        <button
                          key={token}
                          type="button"
                          className="secondary-button context-builder-token-button context-builder-token-button--optional"
                          onClick={() =>
                            onGraphChange(
                              updateNode(graph, selectedNode.id, (node) => ({
                                ...node,
                                config: {
                                  ...node.config,
                                  system_prompt: insertTokenAtEnd(String(node.config.system_prompt ?? ""), `{${token}}`),
                                },
                              })),
                            )
                          }
                        >
                          {token}
                        </button>
                      ))}
                    </div>
                  </div>
                  <div className="prompt-preview-tool-list">
                    <div className="contract-card">
                      <strong>Connected MCP Tool Info</strong>
                      <span>
                        Callable MCP tools: {modelCallableMcpToolNames.length > 0 ? modelCallableMcpToolNames.join(", ") : "None"}
                      </span>
                      <span>
                        Prompt context tools:{" "}
                        {modelPromptContextToolSummaries.length > 0
                          ? modelPromptContextToolSummaries.map((tool) => tool.displayName).join(", ")
                          : "None"}
                      </span>
                    </div>
                    {modelPromptToolSummaries.length > 0 ? (
                      modelPromptToolSummaries.map((tool) => (
                        <div
                          key={`${tool.sourceNodeId}-${tool.toolName}`}
                          className="contract-card prompt-preview-tool-card"
                        >
                          <strong>
                            {tool.displayName} <code>{tool.toolName}</code>
                          </strong>
                          <span>Source: {tool.sourceNodeLabel}</span>
                          <span>Prompt override source: {tool.overrideNodeLabel}</span>
                          <span>Status: {tool.status}</span>
                          <span>Callable by model: {tool.isCallable ? "Yes" : "No"}</span>
                          <span>Injects prompt context: {tool.injectsPromptContext ? "Yes" : "No"}</span>
                          {"placeholderToken" in tool ? <span>Placeholder: <code>{`{${tool.placeholderToken}}`}</code></span> : null}
                          <span>Usage guidance: {tool.usageHint || "None"}</span>
                          <span>Prompt template:</span>
                          <pre className="context-builder-preview">{tool.templateText || "No prompt template."}</pre>
                          <span>Rendered MCP context block:</span>
                          <pre className="context-builder-preview">{tool.renderedPromptText || "No MCP prompt block."}</pre>
                        </div>
                      ))
                    ) : (
                      <div className="contract-card">
                        <strong>Connected MCP Tool Info</strong>
                        <span>No MCP Context Provider tools are connected to this model.</span>
                      </div>
                    )}
                  </div>
                  <section className="tool-details-modal-preview">
                    <div className="tool-details-modal-preview-header">
                      <strong>System Prompt Template Preview</strong>
                      <span>This resolves graph env references and currently visible runtime variables before auto-appended MCP sections are added.</span>
                    </div>
                    <pre>{modelSystemPromptTemplatePreview || "Add a system prompt template to preview it here."}</pre>
                  </section>
                  <section className="tool-details-modal-preview">
                    <div className="tool-details-modal-preview-header">
                      <strong>Final System Prompt Preview</strong>
                      <span>This mirrors the current runtime assembly path for MCP guidance, MCP tool context, and prompt-only MCP decision instructions.</span>
                    </div>
                    <pre>{modelFinalSystemPromptPreview || "The final assembled system prompt will appear here."}</pre>
                  </section>
                </>
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
                  value={
                    standardCatalogTools.find((tool) =>
                      toolMatchesReference(tool, String(selectedNode.config.tool_name ?? selectedNode.tool_name ?? "")),
                    )?.canonical_name ??
                    standardCatalogTools.find((tool) =>
                      toolMatchesReference(tool, String(selectedNode.config.tool_name ?? selectedNode.tool_name ?? "")),
                    )?.name ??
                    String(selectedNode.config.tool_name ?? selectedNode.tool_name ?? "")
                  }
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
                    <option key={toolCanonicalName(tool)} value={toolCanonicalName(tool)}>
                      {toolLabel(tool)} ({toolStatusLabel(tool)})
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
                  const canonicalName = toolCanonicalName(tool);
                  const isChecked = selectedMcpToolNames.some((name) => toolMatchesReference(tool, name));
                  const canSelectTool = isToolEnabled(tool) && isToolOnline(tool);
                  return (
                    <label key={canonicalName} className="checkbox-option">
                      <input
                        type="checkbox"
                        checked={isChecked}
                        disabled={!isChecked && !canSelectTool}
                        onChange={(event) => {
                          const nextTools = event.target.checked
                            ? [...selectedMcpToolNames.filter((name) => !toolMatchesReference(tool, name)), canonicalName]
                            : selectedMcpToolNames.filter((name) => !toolMatchesReference(tool, name));
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
                        {toolLabel(tool)}
                        {toolLabel(tool) !== canonicalName ? <small><code>{canonicalName}</code></small> : null}
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
              <label>
                Usage Guidance
                <textarea
                  rows={4}
                  value={String(selectedNode.config.usage_hint ?? "")}
                  onChange={(event) =>
                    onGraphChange(
                      updateNode(graph, selectedNode.id, (node) => ({
                        ...node,
                        config: {
                          ...node.config,
                          usage_hint: event.target.value,
                        },
                      })),
                    )
                  }
                />
                <small>
                  Natural-language hint folded into the connected model&apos;s required MCP guidance block when MCP prompt
                  context is enabled.
                </small>
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
              {executorFollowUpEnabled ? (
                <label className="checkbox-option">
                  <input
                    type="checkbox"
                    checked={executorRetriesEnabled}
                    onChange={(event) =>
                      onGraphChange(
                        updateNode(graph, selectedNode.id, (node) => ({
                          ...node,
                          config: {
                            ...node.config,
                            allow_retries: event.target.checked,
                          },
                        })),
                      )
                    }
                  />
                  <span>
                    Enable Retries
                    <small>When off, this node makes no follow-up model/API calls. It only executes the incoming MCP tool call and forwards the execution result.</small>
                  </span>
                </label>
              ) : null}
              <div className="inspector-meta">
                <span>Dispatch mode: single MCP tool call from upstream API output</span>
                <span>Input binding: {executorBindingSummary}</span>
                <span>
                  Follow-up decision: {executorFollowUpEnabled ? "enabled via internal model loop" : "disabled"}
                </span>
                <span>Retries: {executorFollowUpEnabled ? (executorRetriesEnabled ? "enabled" : "disabled") : "n/a"}</span>
                <span>Routes: on finish / on failure / terminal output</span>
              </div>
            </>
          ) : null}
          {selectedNode.kind === "data" ? (
            <>
              {isSpreadsheetRowNode ? (
                <>
                  <div className="contract-card">
                    <strong>Spreadsheet Rows</strong>
                    <span>Reads a CSV or XLSX file, maps each row to a header-keyed dictionary, and runs downstream nodes once per row in strict sequence.</span>
                    <span>Each row is emitted as `payload.row_data` with `row_index`, `row_number`, `sheet_name`, and `source_file` metadata.</span>
                  </div>
                  <label>
                    File Format
                    <select
                      value={String(selectedNode.config.file_format ?? "auto")}
                      onChange={(event) =>
                        onGraphChange(
                          updateNode(graph, selectedNode.id, (node) => ({
                            ...node,
                            config: { ...node.config, file_format: event.target.value, mode: "spreadsheet_rows" },
                          })),
                        )
                      }
                    >
                      <option value="auto">Auto Detect</option>
                      <option value="csv">CSV</option>
                      <option value="xlsx">Excel (.xlsx)</option>
                    </select>
                  </label>
                  <label>
                    File Path
                    <input
                      value={String(selectedNode.config.file_path ?? "")}
                      placeholder="Path, env var, or leave empty if exactly one CSV/XLSX is attached to the run"
                      onChange={(event) =>
                        onGraphChange(
                          updateNode(graph, selectedNode.id, (node) => ({
                            ...node,
                            config: { ...node.config, file_path: event.target.value, mode: "spreadsheet_rows" },
                          })),
                        )
                      }
                    />
                  </label>
                  <label>
                    Sheet Name
                    <input
                      value={String(selectedNode.config.sheet_name ?? "")}
                      placeholder="Leave blank to use the first sheet"
                      onChange={(event) =>
                        onGraphChange(
                          updateNode(graph, selectedNode.id, (node) => ({
                            ...node,
                            config: { ...node.config, sheet_name: event.target.value, mode: "spreadsheet_rows" },
                          })),
                        )
                      }
                    />
                  </label>
                  <div className="checkbox-grid">
                    <label>
                      Empty Row Policy
                      <select
                        value={String(selectedNode.config.empty_row_policy ?? "skip")}
                        onChange={(event) =>
                          onGraphChange(
                            updateNode(graph, selectedNode.id, (node) => ({
                              ...node,
                              config: { ...node.config, empty_row_policy: event.target.value, mode: "spreadsheet_rows" },
                            })),
                          )
                        }
                      >
                        <option value="skip">Skip empty rows</option>
                        <option value="include">Include empty rows</option>
                      </select>
                    </label>
                  </div>
                  <p className="node-help-text">
                    Row 1 is always treated as the header row. Each later row is emitted as one iteration using those
                    header titles as the parsed row keys.
                  </p>
                  <div className="context-builder-binding-actions">
                    <button
                      type="button"
                      className="secondary-button context-builder-inline-button"
                      onClick={() => void handleSpreadsheetPreview()}
                      disabled={isSpreadsheetPreviewLoading || spreadsheetResolvedFilePath.trim().length === 0}
                    >
                      {isSpreadsheetPreviewLoading ? "Loading Preview..." : "Preview Rows"}
                    </button>
                  </div>
                  <div className="inspector-meta">
                    <span>Resolved file path: {spreadsheetResolvedFilePath || "Enter a file path or graph env reference."}</span>
                    <span>Execution mode: sequential per-row loop through downstream nodes</span>
                    <span>Recommended shape for downstream prompts/tools: `payload.row_data` key-value pairs</span>
                  </div>
                  {spreadsheetIteratorState ? (
                    <div className="contract-card">
                      <strong>Iterator Progress</strong>
                      <span>Status: {String(spreadsheetIteratorState.status ?? "unknown")}</span>
                      <span>
                        Row progress: {String(spreadsheetIteratorState.current_row_index ?? 0)} / {String(spreadsheetIteratorState.total_rows ?? 0)}
                      </span>
                      <span>Sheet: {String(spreadsheetIteratorState.sheet_name ?? "first sheet")}</span>
                    </div>
                  ) : null}
                  {spreadsheetPreviewError ? <div className="tool-details-modal-help">{spreadsheetPreviewError}</div> : null}
                  <div className="contract-card">
                    <strong>Parsed Preview</strong>
                    <span>
                      {spreadsheetPreview
                        ? `${spreadsheetPreview.row_count} row(s) parsed from ${spreadsheetPreview.file_format.toUpperCase()}`
                        : "Run a preview to inspect headers and sample rows before execution."}
                    </span>
                    <pre className="context-builder-preview">
                      {spreadsheetPreview
                        ? JSON.stringify(
                            {
                              headers: spreadsheetPreview.headers,
                              sheet_name: spreadsheetPreview.sheet_name,
                              row_count: spreadsheetPreview.row_count,
                              sample_rows: spreadsheetPreview.sample_rows,
                            },
                            null,
                            2,
                          )
                        : "Preview output will appear here."}
                    </pre>
                  </div>
                </>
              ) : selectedNode.provider_id === "core.data_display" ? (
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
