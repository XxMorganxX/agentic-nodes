import type { EditorCatalog, ToolDefinition } from "../lib/types";

type McpServerPanelProps = {
  catalog: EditorCatalog | null;
  onBootMcpServer: (serverId: string) => void;
  onStopMcpServer: (serverId: string) => void;
  onRefreshMcpServer: (serverId: string) => void;
  onToggleMcpTool: (toolName: string, enabled: boolean) => void;
  mcpPendingKey: string | null;
  title?: string;
  description?: string;
  className?: string;
};

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
  mcpPendingKey,
  title = "MCP Servers",
  description = "Manage persistent MCP servers and globally enable the tools they publish.",
  className = "",
}: McpServerPanelProps) {
  const mcpServers = catalog?.mcp_servers ?? [];
  if (mcpServers.length === 0) {
    return null;
  }

  return (
    <section className={`mcp-server-panel ${className}`.trim()}>
      <div className="contract-card">
        <strong>{title}</strong>
        <span>{description}</span>
      </div>
      {mcpServers.map((server) => {
        const serverTools = (catalog?.tools ?? []).filter((tool) => tool.server_id === server.server_id);
        const bootPending = mcpPendingKey === `boot:${server.server_id}`;
        const stopPending = mcpPendingKey === `stop:${server.server_id}`;
        const refreshPending = mcpPendingKey === `refresh:${server.server_id}`;
        return (
          <div key={server.server_id} className="mcp-server-card">
            <div className="mcp-server-card-header">
              <div>
                <strong>{server.display_name}</strong>
                <p>{server.description}</p>
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
