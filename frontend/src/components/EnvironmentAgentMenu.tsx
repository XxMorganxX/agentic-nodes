import { formatRunStatusLabel, type AgentRunLane } from "../lib/runVisualization";
import { RunErrorHover } from "./RunErrorHover";

type EnvironmentAgentMenuProps = {
  agents: AgentRunLane[];
  selectedAgentId: string | null;
  open: boolean;
  onToggle: () => void;
  onSelectAgent: (agentId: string) => void;
};

export function EnvironmentAgentMenu({
  agents,
  selectedAgentId,
  open,
  onToggle,
  onSelectAgent,
}: EnvironmentAgentMenuProps) {
  if (agents.length === 0) {
    return null;
  }

  return (
    <div className={`environment-agent-menu${open ? " is-open" : ""}`}>
      <button
        type="button"
        className="environment-agent-menu-button"
        aria-expanded={open}
        aria-haspopup="listbox"
        onClick={onToggle}
      >
        <span className="environment-agent-menu-indicator" aria-hidden="true" />
        <span>Agents</span>
      </button>
      <div className="environment-agent-menu-panel" role="listbox" aria-label="Environment agents">
        <div className="environment-agent-menu-scroll">
          {agents.map((agent) => (
            <button
              key={agent.agentId}
              type="button"
              role="option"
              aria-selected={selectedAgentId === agent.agentId}
              className={`environment-agent-chip ${selectedAgentId === agent.agentId ? "is-selected" : ""}`}
              onClick={() => onSelectAgent(agent.agentId)}
            >
              <div className="environment-agent-chip-header">
                <strong>{agent.agentName}</strong>
                <span className={`environment-agent-chip-status environment-agent-chip-status--${agent.status}`}>
                  {formatRunStatusLabel(agent.status)}
                </span>
              </div>
              <div className="environment-agent-chip-meta">
                <span>{agent.completedNodes}/{agent.totalNodes} nodes</span>
                <RunErrorHover count={agent.errorCount} summaries={agent.errorSummaries} emptyLabel="No errors" />
                <span>{agent.elapsedLabel}</span>
              </div>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
