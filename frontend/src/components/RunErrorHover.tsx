import type { AgentRunErrorSummary } from "../lib/runVisualization";

type RunErrorHoverProps = {
  count: number;
  summaries: AgentRunErrorSummary[];
  className?: string;
  emptyLabel?: string;
};

const MAX_VISIBLE_ERRORS = 6;

function formatTooltipTitle(summaries: AgentRunErrorSummary[]): string {
  return summaries
    .map((summary) =>
      [`${summary.nodeLabel}: ${summary.message}`, ...summary.metadata].filter(Boolean).join(" · "),
    )
    .join("\n");
}

export function RunErrorHover({
  count,
  summaries,
  className = "",
  emptyLabel = "0 errors",
}: RunErrorHoverProps) {
  if (count <= 0 || summaries.length === 0) {
    return <span className={`run-error-hover run-error-hover--empty ${className}`.trim()}>{emptyLabel}</span>;
  }

  const visibleSummaries = summaries.slice(0, MAX_VISIBLE_ERRORS);
  const remainingCount = Math.max(0, summaries.length - visibleSummaries.length);

  return (
    <span className={`run-error-hover ${className}`.trim()} title={formatTooltipTitle(summaries)}>
      <span className="run-error-hover-trigger">
        {count} error{count === 1 ? "" : "s"}
      </span>
      <span className="run-error-hover-popup" role="tooltip">
        <strong>Run Errors</strong>
        {visibleSummaries.map((summary) => (
          <span key={summary.id} className="run-error-hover-item">
            <span className="run-error-hover-item-header">
              <span>{summary.nodeLabel}</span>
              {summary.errorTypeLabel ? <span>{summary.errorTypeLabel}</span> : null}
            </span>
            <span className="run-error-hover-item-message">{summary.message}</span>
            {summary.metadata.length > 0 ? <span className="run-error-hover-item-meta">{summary.metadata.join(" · ")}</span> : null}
          </span>
        ))}
        {remainingCount > 0 ? (
          <span className="run-error-hover-more">
            +{remainingCount} more error{remainingCount === 1 ? "" : "s"}
          </span>
        ) : null}
      </span>
    </span>
  );
}
