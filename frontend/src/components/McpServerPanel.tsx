import { useMemo, useState } from "react";

import type { EditorCatalog, McpServerDraft, McpServerStatus, ToolDefinition } from "../lib/types";

type McpServerPanelProps = {
  catalog: EditorCatalog | null;
  onBootMcpServer: (serverId: string) => void;
  onStopMcpServer: (serverId: string) => void;
  onRefreshMcpServer: (serverId: string) => void;
  onToggleMcpTool: (toolName: string, enabled: boolean) => void;
  onCreateMcpServer: (server: McpServerDraft) => Promise<unknown>;
  onUpdateMcpServer: (serverId: string, server: McpServerDraft) => Promise<unknown>;
  onDeleteMcpServer: (serverId: string) => Promise<unknown>;
  onTestMcpServer: (server: McpServerDraft) => Promise<string | null>;
  mcpPendingKey: string | null;
  title?: string;
  description?: string;
  className?: string;
};

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
  const [editingServerId, setEditingServerId] = useState<string | null>(null);
  const [formState, setFormState] = useState<McpServerFormState>(createBlankForm);
  const [testMessage, setTestMessage] = useState<string | null>(null);
  const [localError, setLocalError] = useState<string | null>(null);

  const editingServer = useMemo(
    () => mcpServers.find((server) => server.server_id === editingServerId) ?? null,
    [editingServerId, mcpServers],
  );
  const isCreating = editingServerId === "__new__";
  const isEditing = Boolean(editingServerId);

  function beginCreate() {
    setEditingServerId("__new__");
    setFormState(createBlankForm());
    setTestMessage(null);
    setLocalError(null);
  }

  function beginEdit(server: McpServerStatus) {
    setEditingServerId(server.server_id);
    setFormState(formFromServer(server));
    setTestMessage(null);
    setLocalError(null);
  }

  function cancelEdit() {
    setEditingServerId(null);
    setFormState(createBlankForm());
    setTestMessage(null);
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
      const message = await onTestMcpServer(toDraft(formState));
      setTestMessage(message);
      setLocalError(null);
    } catch (error) {
      setLocalError(error instanceof Error ? error.message : "Unable to test MCP server.");
      setTestMessage(null);
    }
  }

  return (
    <section className={`mcp-server-panel ${className}`.trim()}>
      <div className="contract-card">
        <strong>{title}</strong>
        <span>{description}</span>
        <span>Server state is project-level. Graph nodes still choose which ready MCP tools they expose or describe to models.</span>
        <div className="mcp-server-actions">
          <button type="button" className="secondary-button" onClick={beginCreate} disabled={isCreating}>
            Add Server
          </button>
          {isEditing ? (
            <button type="button" className="secondary-button" onClick={cancelEdit}>
              Cancel
            </button>
          ) : null}
        </div>
      </div>
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
          {testMessage ? <p className="mcp-server-message">{testMessage}</p> : null}
          {localError ? <p className="error-text">{localError}</p> : null}
        </div>
      ) : null}
      {mcpServers.map((server) => {
        const serverTools = (catalog?.tools ?? []).filter((tool) => tool.server_id === server.server_id);
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
            {serverTools.length > 0 ? (
              <div className="mcp-tool-list">
                {serverTools.map((tool) => {
                  const pending = mcpPendingKey === `tool:${tool.name}`;
                  return (
                    <label key={tool.name} className="checkbox-option mcp-tool-option">
                      <input
                        type="checkbox"
                        checked={isToolEnabled(tool)}
                        disabled={pending}
                        onChange={(event) => onToggleMcpTool(tool.name, event.target.checked)}
                      />
                      <span>
                        {tool.name}
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
