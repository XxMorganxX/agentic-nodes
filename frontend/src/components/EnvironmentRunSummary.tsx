import { formatRunStatusLabel, type EnvironmentRunSummary as EnvironmentRunSummaryData } from "../lib/runVisualization";

type EnvironmentRunSummaryProps = {
  summary: EnvironmentRunSummaryData;
};

export function EnvironmentRunSummary({ summary }: EnvironmentRunSummaryProps) {
  return (
    <section className="panel environment-run-summary">
      <div className="panel-header environment-run-summary-header">
        <div>
          <h2>Environment Run</h2>
          <p>
            {summary.runId ? `Run ${summary.runId}` : "No run yet."}
            {summary.focusedAgentName ? ` Focused on ${summary.focusedAgentName}.` : ""}
          </p>
        </div>
        <span className={`environment-run-summary-status environment-run-summary-status--${summary.status}`}>
          {formatRunStatusLabel(summary.status)}
        </span>
      </div>
      <dl className="environment-run-summary-grid">
        <div>
          <dt>Total Agents</dt>
          <dd>{summary.totalAgents}</dd>
        </div>
        <div>
          <dt>Running</dt>
          <dd>{summary.runningAgents}</dd>
        </div>
        <div>
          <dt>Completed</dt>
          <dd>{summary.completedAgents}</dd>
        </div>
        <div>
          <dt>Failed</dt>
          <dd>{summary.failedAgents}</dd>
        </div>
        <div>
          <dt>Cancelled</dt>
          <dd>{summary.cancelledAgents}</dd>
        </div>
        <div>
          <dt>Interrupted</dt>
          <dd>{summary.interruptedAgents}</dd>
        </div>
        <div>
          <dt>Queued</dt>
          <dd>{summary.queuedAgents}</dd>
        </div>
        <div>
          <dt>Elapsed</dt>
          <dd>{summary.elapsedLabel}</dd>
        </div>
        <div>
          <dt>Last Heartbeat</dt>
          <dd>{summary.lastHeartbeatLabel}</dd>
        </div>
      </dl>
      <div className="environment-run-summary-active">
        <strong>Active Agents</strong>
        <span>{summary.activeAgentNames.length > 0 ? summary.activeAgentNames.join(", ") : "None right now."}</span>
      </div>
    </section>
  );
}
