import { useMemo, useState } from "react";

import type {
  EditorCatalog,
  McpCapabilityDefinition,
  McpServerDraft,
  McpServerStatus,
  McpServerTemplate,
  McpServerTestResult,
  ToolDefinition,
} from "../lib/types";

type McpServerPanelProps = {
  catalog: EditorCatalog | null;
  onBootMcpServer: (serverId: string) => void;
  onStopMcpServer: (serverId: string) => void;
  onRefreshMcpServer: (serverId: string) => void;
  onToggleMcpTool: (toolName: string, enabled: boolean) => void;
  onCreateMcpServer: (server: McpServerDraft) => Promise<unknown>;
  onUpdateMcpServer: (serverId: string, server: McpServerDraft) => Promise<unknown>;
  onDeleteMcpServer: (serverId: string) => Promise<unknown>;
  onTestMcpServer: (server: McpServerDraft) => Promise<McpServerTestResult | null>;
  mcpPendingKey: string | null;
  title?: string;
  description?: string;
  className?: string;
};

type CapabilityFilter = "all" | "tool" | "resource" | "resource_template" | "prompt";

type McpServerFormState = {
  server_id: string;
  display_name: string;
  description: string;
  transport: "stdio" | "http";
  command_text: string;
  cwd: string;
  env_text: string;
  base_url: string;
  timeout_seconds: string;
  auto_boot: boolean;
  persistent: boolean;
};

function createBlankForm(): McpServerFormState {
  return {
    server_id: "",
    display_name: "",
    description: "",
    transport: "stdio",
    command_text: "",
    cwd: "",
    env_text: "",
    base_url: "",
    timeout_seconds: "15",
    auto_boot: false,
    persistent: true,
  };
}

function formFromServer(server: McpServerStatus): McpServerFormState {
  return {
    server_id: server.server_id,
    display_name: server.display_name,
    description: server.description,
    transport: server.transport ?? "stdio",
    command_text: (server.config?.command ?? []).join("\n"),
    cwd: server.config?.cwd ?? "",
    env_text: Object.entries(server.config?.env ?? {})
      .map(([key, value]) => `${key}=${value}`)
      .join("\n"),
    base_url: server.config?.base_url ?? "",
    timeout_seconds: String(server.config?.timeout_seconds ?? 15),
    auto_boot: server.auto_boot,
    persistent: server.persistent,
  };
}

function formFromDraft(draft: McpServerDraft): McpServerFormState {
  return {
    server_id: draft.server_id,
    display_name: draft.display_name,
    description: draft.description,
    transport: draft.transport,
    command_text: draft.command.join("\n"),
    cwd: draft.cwd ?? "",
    env_text: Object.entries(draft.env ?? {})
      .map(([key, value]) => `${key}=${value}`)
      .join("\n"),
    base_url: draft.base_url ?? "",
    timeout_seconds: String(draft.timeout_seconds ?? 15),
    auto_boot: draft.auto_boot,
    persistent: draft.persistent,
  };
}

function parseCommand(commandText: string): string[] {
  return commandText
    .split("\n")
    .map((part) => part.trim())
    .filter(Boolean);
}

function parseEnv(envText: string): Record<string, string> {
  const env: Record<string, string> = {};
  for (const rawLine of envText.split("\n")) {
    const line = rawLine.trim();
    if (!line) {
      continue;
    }
    const equalsIndex = line.indexOf("=");
    if (equalsIndex === -1) {
      env[line] = "";
      continue;
    }
    const key = line.slice(0, equalsIndex).trim();
    if (!key) {
      continue;
    }
    env[key] = line.slice(equalsIndex + 1).trim();
  }
  return env;
}

function toDraft(form: McpServerFormState): McpServerDraft {
  return {
    server_id: form.server_id.trim(),
    display_name: form.display_name.trim(),
    description: form.description.trim(),
    transport: form.transport,
    command: parseCommand(form.command_text),
    cwd: form.cwd.trim() || null,
    env: parseEnv(form.env_text),
    base_url: form.base_url.trim() || null,
    timeout_seconds: Number.parseInt(form.timeout_seconds.trim() || "15", 10) || 15,
    auto_boot: form.auto_boot,
    persistent: form.persistent,
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

function capabilityLabel(capability: McpCapabilityDefinition): string {
  return capability.title || capability.display_name || capability.name;
}

function capabilityReference(capability: McpCapabilityDefinition): string {
  if (typeof capability.metadata?.uri === "string" && capability.metadata.uri.trim()) {
    return capability.metadata.uri;
  }
  if (typeof capability.metadata?.uri_template === "string" && capability.metadata.uri_template.trim()) {
    return capability.metadata.uri_template;
  }
  return capability.canonical_name;
}

function capabilityStatusLabel(capability: McpCapabilityDefinition): string {
  if (capability.available === false) {
    return "offline";
  }
  if (capability.capability_type === "tool" && capability.enabled === false) {
    return "disabled";
  }
  return "discovered";
}

function declaredCapabilityLabels(server: McpServerStatus): string[] {
  return Object.keys(server.declared_capabilities ?? {}).sort();
}

function templateProvenanceLabel(template: McpServerTemplate): string {
  const registry = typeof template.provenance?.registry === "string" ? template.provenance.registry : null;
  const publisher = typeof template.provenance?.publisher === "string" ? template.provenance.publisher : null;
  return [template.source, registry, publisher].filter(Boolean).join(" • ");
}

export function McpServerPanel({
  catalog,
  onBootMcpServer,
  onStopMcpServer,
  onRefreshMcpServer,
  onToggleMcpTool,
  onCreateMcpServer,
  onUpdateMcpServer,
  onDeleteMcpServer,
  onTestMcpServer,
  mcpPendingKey,
  title = "MCP Servers",
  description = "Manage persistent MCP servers and globally enable the tools they publish.",
  className = "",
}: McpServerPanelProps) {
  const mcpServers = catalog?.mcp_servers ?? [];
  const mcpCapabilities = catalog?.mcp_capabilities ?? [];
  const mcpServerTemplates = catalog?.mcp_server_templates ?? [];
  const [editingServerId, setEditingServerId] = useState<string | null>(null);
  const [formState, setFormState] = useState<McpServerFormState>(createBlankForm);
  const [testSnapshot, setTestSnapshot] = useState<McpServerTestResult | null>(null);
  const [localError, setLocalError] = useState<string | null>(null);
  const [capabilityFilter, setCapabilityFilter] = useState<CapabilityFilter>("all");

  const editingServer = useMemo(
    () => mcpServers.find((server) => server.server_id === editingServerId) ?? null,
    [editingServerId, mcpServers],
  );
  const isCreating = editingServerId === "__new__";
  const isEditing = Boolean(editingServerId);

  function beginCreate(templateDraft?: McpServerDraft) {
    setEditingServerId("__new__");
    setFormState(templateDraft ? formFromDraft(templateDraft) : createBlankForm());
    setTestSnapshot(null);
    setLocalError(null);
  }

  function beginEdit(server: McpServerStatus) {
    setEditingServerId(server.server_id);
    setFormState(formFromServer(server));
    setTestSnapshot(null);
    setLocalError(null);
  }

  function cancelEdit() {
    setEditingServerId(null);
    setFormState(createBlankForm());
    setTestSnapshot(null);
    setLocalError(null);
  }

  async function handleSubmit() {
    try {
      const draft = toDraft(formState);
      if (isCreating) {
        await onCreateMcpServer(draft);
      } else if (editingServer) {
        await onUpdateMcpServer(editingServer.server_id, draft);
      }
      cancelEdit();
    } catch (error) {
      setLocalError(error instanceof Error ? error.message : "Unable to save MCP server.");
    }
  }

  async function handleTest() {
    try {
      const result = await onTestMcpServer(toDraft(formState));
      setTestSnapshot(result);
      setLocalError(null);
    } catch (error) {
      setLocalError(error instanceof Error ? error.message : "Unable to test MCP server.");
      setTestSnapshot(null);
    }
  }

  return (
    <section className={`mcp-server-panel ${className}`.trim()}>
      <div className="contract-card">
        <strong>{title}</strong>
        <span>{description}</span>
        <span>Server state is project-level. Graph nodes still choose which ready MCP tools they expose or describe to models.</span>
        <div className="mcp-server-actions">
          <button type="button" className="secondary-button" onClick={() => beginCreate()} disabled={isCreating}>
            Add Server
          </button>
          {isEditing ? (
            <button type="button" className="secondary-button" onClick={cancelEdit}>
              Cancel
            </button>
          ) : null}
        </div>
      </div>
      {mcpServerTemplates.length > 0 ? (
        <div className="mcp-template-browser">
          <div className="mcp-template-browser-header">
            <strong>Server Templates</strong>
            <span>Start from a curated MCP server draft instead of entering every field by hand.</span>
          </div>
          <div className="mcp-template-list">
            {mcpServerTemplates.map((template) => (
              <div key={template.template_id} className="mcp-template-card">
                <div className="mcp-template-card-header">
                  <div>
                    <strong>{template.display_name}</strong>
                    <p>{template.description}</p>
                  </div>
                  <button type="button" className="secondary-button" onClick={() => beginCreate(template.draft)}>
                    Use Template
                  </button>
                </div>
                <p className="mcp-server-meta">
                  <code>{template.template_id}</code>
                  {template.draft.transport ? <span>{template.draft.transport}</span> : null}
                  {template.capability_hints?.map((hint) => <span key={hint}>{hint}</span>)}
                  {templateProvenanceLabel(template) ? <span>{templateProvenanceLabel(template)}</span> : null}
                </p>
              </div>
            ))}
          </div>
        </div>
      ) : null}
      {isEditing ? (
        <div className="mcp-server-card mcp-server-card--editor">
          <div className="mcp-server-card-header">
            <div>
              <strong>{isCreating ? "New MCP Server" : `Edit ${editingServer?.display_name ?? "MCP Server"}`}</strong>
              <p>{isCreating ? "Create a user-managed MCP server definition." : "Update the saved MCP server configuration."}</p>
            </div>
            <span className="status-pill is-muted">{formState.transport}</span>
          </div>
          <div className="mcp-server-form-grid">
            <label>
              Server ID
              <input
                value={formState.server_id}
                onChange={(event) => setFormState((current) => ({ ...current, server_id: event.target.value }))}
                disabled={!isCreating}
                placeholder="weather_remote"
              />
            </label>
            <label>
              Display Name
              <input
                value={formState.display_name}
                onChange={(event) => setFormState((current) => ({ ...current, display_name: event.target.value }))}
                placeholder="Remote Weather"
              />
            </label>
            <label>
              Transport
              <select
                value={formState.transport}
                onChange={(event) =>
                  setFormState((current) => ({
                    ...current,
                    transport: event.target.value === "http" ? "http" : "stdio",
                  }))
                }
              >
                <option value="stdio">stdio subprocess</option>
                <option value="http">remote HTTP</option>
              </select>
            </label>
            <label>
              Timeout Seconds
              <input
                type="number"
                min={1}
                value={formState.timeout_seconds}
                onChange={(event) => setFormState((current) => ({ ...current, timeout_seconds: event.target.value }))}
              />
            </label>
            <label className="mcp-server-form-grid--full">
              Description
              <textarea
                value={formState.description}
                onChange={(event) => setFormState((current) => ({ ...current, description: event.target.value }))}
                rows={2}
              />
            </label>
            {formState.transport === "stdio" ? (
              <>
                <label className="mcp-server-form-grid--full">
                  Command
                  <textarea
                    value={formState.command_text}
                    onChange={(event) => setFormState((current) => ({ ...current, command_text: event.target.value }))}
                    rows={4}
                    placeholder={"python\n-m\nmy_mcp_server"}
                  />
                  <small>Enter one subprocess argument per line.</small>
                </label>
                <label>
                  Working Directory
                  <input
                    value={formState.cwd}
                    onChange={(event) => setFormState((current) => ({ ...current, cwd: event.target.value }))}
                    placeholder="/path/to/project"
                  />
                </label>
                <label className="mcp-server-form-grid--full">
                  Environment Variables
                  <textarea
                    value={formState.env_text}
                    onChange={(event) => setFormState((current) => ({ ...current, env_text: event.target.value }))}
                    rows={4}
                    placeholder={"PYTHONPATH=/path/to/src\nFOO=bar"}
                  />
                  <small>Use one `KEY=value` pair per line.</small>
                </label>
              </>
            ) : (
              <label className="mcp-server-form-grid--full">
                Base URL
                <input
                  value={formState.base_url}
                  onChange={(event) => setFormState((current) => ({ ...current, base_url: event.target.value }))}
                  placeholder="https://example.com/mcp"
                />
              </label>
            )}
            <label className="checkbox-option">
              <input
                type="checkbox"
                checked={formState.auto_boot}
                onChange={(event) => setFormState((current) => ({ ...current, auto_boot: event.target.checked }))}
              />
              <span>
                Auto boot
                <small>Reconnect this server when the app starts.</small>
              </span>
            </label>
            <label className="checkbox-option">
              <input
                type="checkbox"
                checked={formState.persistent}
                onChange={(event) => setFormState((current) => ({ ...current, persistent: event.target.checked }))}
              />
              <span>
                Persist locally
                <small>Keep this server definition in the user-local `.graph-agent` store.</small>
              </span>
            </label>
          </div>
          <div className="mcp-server-actions">
            <button type="button" className="secondary-button" onClick={() => void handleTest()} disabled={mcpPendingKey === `test:${formState.server_id || "draft"}`}>
              {mcpPendingKey === `test:${formState.server_id || "draft"}` ? "Testing..." : "Test Connection"}
            </button>
            <button
              type="button"
              className="secondary-button"
              onClick={() => void handleSubmit()}
              disabled={mcpPendingKey === `create:${formState.server_id}` || mcpPendingKey === `update:${editingServer?.server_id ?? ""}`}
            >
              {isCreating
                ? mcpPendingKey === `create:${formState.server_id}`
                  ? "Creating..."
                  : "Create Server"
                : mcpPendingKey === `update:${editingServer?.server_id ?? ""}`
                  ? "Saving..."
                  : "Save Changes"}
            </button>
          </div>
          {testSnapshot ? (
            <div className="mcp-test-result">
              <p className="mcp-server-message">{testSnapshot.message}</p>
              <p className="mcp-server-meta">
                <span>{testSnapshot.capability_count ?? testSnapshot.capabilities.length} capabilities</span>
                {(testSnapshot.capability_types ?? []).map((item) => (
                  <span key={item}>{item}</span>
                ))}
                {Object.keys(testSnapshot.declared_capabilities ?? {}).map((item) => (
                  <span key={item}>declares {item}</span>
                ))}
              </p>
              {testSnapshot.capabilities.length > 0 ? (
                <div className="mcp-capability-list">
                  {testSnapshot.capabilities.map((capability) => (
                    <div key={capability.canonical_name} className="mcp-capability-card">
                      <div className="mcp-capability-card-header">
                        <strong>{capabilityLabel(capability)}</strong>
                        <span className="status-pill is-muted">{capability.capability_type}</span>
                      </div>
                      <p className="mcp-server-meta">
                        <code>{capabilityReference(capability)}</code>
                        <span>{capabilityStatusLabel(capability)}</span>
                      </p>
                    </div>
                  ))}
                </div>
              ) : null}
            </div>
          ) : null}
          {localError ? <p className="error-text">{localError}</p> : null}
        </div>
      ) : null}
      <div className="mcp-capability-browser-toolbar">
        <strong>Capability Browser</strong>
        <div className="mcp-capability-filter-row">
          {(["all", "tool", "resource", "resource_template", "prompt"] as CapabilityFilter[]).map((filter) => (
            <button
              key={filter}
              type="button"
              className={`secondary-button ${capabilityFilter === filter ? "is-active" : ""}`.trim()}
              onClick={() => setCapabilityFilter(filter)}
            >
              {filter === "all" ? "All" : filter}
            </button>
          ))}
        </div>
      </div>
      {mcpServers.map((server) => {
        const serverTools = (catalog?.tools ?? []).filter((tool) => tool.server_id === server.server_id);
        const serverCapabilities = mcpCapabilities
          .filter((capability) => capability.server_id === server.server_id)
          .filter((capability) => capabilityFilter === "all" || capability.capability_type === capabilityFilter);
        const bootPending = mcpPendingKey === `boot:${server.server_id}`;
        const stopPending = mcpPendingKey === `stop:${server.server_id}`;
        const refreshPending = mcpPendingKey === `refresh:${server.server_id}`;
        const deletePending = mcpPendingKey === `delete:${server.server_id}`;
        return (
          <div key={server.server_id} className="mcp-server-card">
            <div className="mcp-server-card-header">
              <div>
                <strong>{server.display_name}</strong>
                <p>{server.description}</p>
                <p className="mcp-server-meta">
                  <code>{server.server_id}</code>
                  <span>{server.transport}</span>
                  {server.config_summary ? <span>{server.config_summary}</span> : null}
                  {server.source ? <span>{server.source}</span> : null}
                  {typeof server.capability_count === "number" ? <span>{server.capability_count} capabilities</span> : null}
                  {(server.capability_types ?? []).map((item) => (
                    <span key={item}>{item}</span>
                  ))}
                  {declaredCapabilityLabels(server).map((item) => (
                    <span key={item}>declares {item}</span>
                  ))}
                </p>
              </div>
              <span className={`status-pill ${server.running ? "is-ready" : "is-muted"}`}>{server.running ? "running" : "offline"}</span>
            </div>
            <div className="mcp-server-actions">
              <button type="button" className="secondary-button" onClick={() => onBootMcpServer(server.server_id)} disabled={server.running || bootPending}>
                {bootPending ? "Booting..." : "Boot"}
              </button>
              <button
                type="button"
                className="secondary-button"
                onClick={() => onRefreshMcpServer(server.server_id)}
                disabled={!server.running || refreshPending}
              >
                {refreshPending ? "Refreshing..." : "Refresh"}
              </button>
              <button type="button" className="secondary-button" onClick={() => onStopMcpServer(server.server_id)} disabled={!server.running || stopPending}>
                {stopPending ? "Stopping..." : "Stop"}
              </button>
              {server.editable ? (
                <button type="button" className="secondary-button" onClick={() => beginEdit(server)} disabled={deletePending}>
                  Edit
                </button>
              ) : null}
              {server.editable ? (
                <button type="button" className="secondary-button" onClick={() => void onDeleteMcpServer(server.server_id)} disabled={deletePending}>
                  {deletePending ? "Deleting..." : "Delete"}
                </button>
              ) : null}
            </div>
            {server.error ? <p className="error-text">{server.error}</p> : null}
            {serverCapabilities.length > 0 ? (
              <div className="mcp-capability-list">
                {serverCapabilities.map((capability) => (
                  <div key={capability.canonical_name} className="mcp-capability-card">
                    <div className="mcp-capability-card-header">
                      <div>
                        <strong>{capabilityLabel(capability)}</strong>
                        {capability.description ? <p>{capability.description}</p> : null}
                      </div>
                      <span className="status-pill is-muted">{capability.capability_type}</span>
                    </div>
                    <p className="mcp-server-meta">
                      <code>{capabilityReference(capability)}</code>
                      <span>{capabilityStatusLabel(capability)}</span>
                      {capability.schema_warning ? <span>{capability.schema_warning}</span> : null}
                    </p>
                    {capability.metadata && Object.keys(capability.metadata).length > 0 ? (
                      <details className="mcp-capability-details">
                        <summary>Metadata</summary>
                        <pre>{JSON.stringify(capability.metadata, null, 2)}</pre>
                      </details>
                    ) : null}
                  </div>
                ))}
              </div>
            ) : (
              <p className="inspector-hint">No {capabilityFilter === "all" ? "" : `${capabilityFilter} `}capabilities discovered for this server yet.</p>
            )}
            {serverTools.length > 0 ? (
              <div className="mcp-tool-list">
                {serverTools.map((tool) => {
                  const toolName = toolCanonicalName(tool);
                  const pending = mcpPendingKey === `tool:${toolName}`;
                  return (
                    <label key={toolName} className="checkbox-option mcp-tool-option">
                      <input
                        type="checkbox"
                        checked={isToolEnabled(tool)}
                        disabled={pending}
                        onChange={(event) => onToggleMcpTool(toolName, event.target.checked)}
                      />
                      <span>
                        {toolLabel(tool)}
                        {toolLabel(tool) !== toolName ? <small><code>{toolName}</code></small> : null}
                        <small>{toolStatusLabel(tool)}</small>
                        {tool.schema_warning ? <small>{tool.schema_warning}</small> : null}
                      </span>
                    </label>
                  );
                })}
              </div>
            ) : (
              <p className="inspector-hint">No tools registered for this server yet.</p>
            )}
          </div>
        );
      })}
    </section>
  );
}
