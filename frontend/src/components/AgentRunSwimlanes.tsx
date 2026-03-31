import { formatRunStatusLabel, type AgentRunLane } from "../lib/runVisualization";

type AgentRunSwimlanesProps = {
  lanes: AgentRunLane[];
  selectedAgentId: string | null;
  onSelectAgent: (agentId: string) => void;
  onSelectNode?: (agentId: string, nodeId: string | null) => void;
};

function formatEventTypeLabel(eventType: string): string {
  return eventType
    .split(".")
    .flatMap((segment) => segment.split(/[\s_-]+/))
    .filter(Boolean)
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

function formatStructuredValue(value: unknown): string {
  if (typeof value === "string") {
    return value;
  }
  try {
    return JSON.stringify(value, null, 2) ?? String(value);
  } catch {
    return String(value);
  }
}

export function AgentRunSwimlanes({
  lanes,
  selectedAgentId,
  onSelectAgent,
  onSelectNode,
}: AgentRunSwimlanesProps) {
  return (
    <section className="panel agent-swimlanes-panel">
      <div className="panel-header">
        <h2>Agent Run Swimlanes</h2>
        <p>Track each agent over time and click a milestone to focus its graph.</p>
      </div>
      <div className="agent-swimlanes">
        {lanes.map((lane) => (
          <section
            key={lane.agentId}
            className={`agent-swimlane ${selectedAgentId === lane.agentId ? "is-selected" : ""}`}
            onClick={() => onSelectAgent(lane.agentId)}
          >
            <button
              type="button"
              className="agent-swimlane-meta"
              onClick={(event) => {
                event.stopPropagation();
                onSelectAgent(lane.agentId);
              }}
            >
              <div className="agent-swimlane-heading">
                <strong>{lane.agentName}</strong>
                <span className={`agent-swimlane-status agent-swimlane-status--${lane.status}`}>
                  {formatRunStatusLabel(lane.status)}
                </span>
              </div>
              <div className="agent-swimlane-stats">
                <span>{lane.completedNodes}/{lane.totalNodes} nodes</span>
                <span>{lane.transitionCount} transitions</span>
                <span>{lane.retryCount} retries</span>
                <span>{lane.errorCount} errors</span>
                <span>{lane.elapsedLabel}</span>
              </div>
              <div className="agent-swimlane-current">Current: {lane.currentNodeLabel}</div>
            </button>
            <div className="agent-swimlane-track" role="list" aria-label={`${lane.agentName} milestones`}>
              {lane.milestones.length === 0 ? (
                <div className="agent-swimlane-empty">No runtime milestones yet.</div>
              ) : (
                lane.milestones.map((milestone) => (
                  <button
                    key={milestone.id}
                    type="button"
                    role="listitem"
                    className={`agent-swimlane-milestone agent-swimlane-milestone--${milestone.tone}`}
                    title={[
                      milestone.label,
                      milestone.timestampDetail,
                      milestone.relativeTimestampLabel,
                      milestone.deltaLabel,
                    ]
                      .filter(Boolean)
                      .join(" · ")}
                    onClick={(event) => {
                      event.stopPropagation();
                      onSelectAgent(lane.agentId);
                      onSelectNode?.(lane.agentId, milestone.nodeId);
                    }}
                  >
                    <span className="agent-swimlane-dot" />
                    <div className="agent-swimlane-milestone-content">
                      <div className="agent-swimlane-milestone-header">
                        <span className="agent-swimlane-milestone-copy">
                          <strong>{milestone.label}</strong>
                          <span>{formatEventTypeLabel(milestone.eventType)}</span>
                        </span>
                        <span className="agent-swimlane-milestone-time">
                          <strong>{milestone.timestampLabel}</strong>
                          {milestone.relativeTimestampLabel ? <span>{milestone.relativeTimestampLabel}</span> : null}
                          {milestone.deltaLabel ? <span>{milestone.deltaLabel}</span> : null}
                        </span>
                      </div>
                      {milestone.details.length > 0 ? (
                        <div className="agent-swimlane-milestone-details">
                          {milestone.details.map((detail) => (
                            <span key={`${milestone.id}-${detail.label}`} className="agent-swimlane-milestone-detail">
                              <strong>{detail.label}</strong>
                              <span>{detail.value}</span>
                            </span>
                          ))}
                        </div>
                      ) : null}
                      {milestone.dataSections.length > 0 ? (
                        <div className="agent-swimlane-milestone-data">
                          {milestone.dataSections.map((section) => (
                            <div key={`${milestone.id}-${section.label}`} className="agent-swimlane-milestone-json">
                              <strong>{section.label}</strong>
                              <pre>{formatStructuredValue(section.value)}</pre>
                            </div>
                          ))}
                        </div>
                      ) : null}
                    </div>
                  </button>
                ))
              )}
            </div>
          </section>
        ))}
      </div>
    </section>
  );
}
