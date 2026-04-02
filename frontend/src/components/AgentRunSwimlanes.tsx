import { useEffect, useMemo, useState } from "react";
import type { KeyboardEvent } from "react";

import { formatRunStatusLabel, type AgentRunLane } from "../lib/runVisualization";
import { RunErrorHover } from "./RunErrorHover";

type AgentRunSwimlanesProps = {
  lanes: AgentRunLane[];
  selectedAgentId: string | null;
  onSelectAgent: (agentId: string) => void;
  onSelectNode?: (agentId: string, nodeId: string | null) => void;
};

function hasActiveTextSelection(): boolean {
  if (typeof window === "undefined") {
    return false;
  }
  return (window.getSelection?.()?.toString() ?? "").trim().length > 0;
}

function handleKeyboardActivate(
  event: KeyboardEvent<HTMLElement>,
  activate: () => void,
): void {
  if (event.key !== "Enter" && event.key !== " ") {
    return;
  }
  event.preventDefault();
  activate();
}

function formatEventTypeLabel(eventType: string): string {
  return eventType
    .split(".")
    .flatMap((segment) => segment.split(/[\s_-]+/))
    .filter(Boolean)
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

const EVENT_TYPE_DISPLAY_ORDER = [
  "run.started",
  "node.started",
  "node.completed",
  "edge.selected",
  "condition.evaluated",
  "retry.triggered",
  "run.completed",
  "run.failed",
  "run.cancelled",
  "run.interrupted",
] as const;

function compareEventTypes(left: string, right: string): number {
  const leftIndex = EVENT_TYPE_DISPLAY_ORDER.indexOf(left as (typeof EVENT_TYPE_DISPLAY_ORDER)[number]);
  const rightIndex = EVENT_TYPE_DISPLAY_ORDER.indexOf(right as (typeof EVENT_TYPE_DISPLAY_ORDER)[number]);
  if (leftIndex !== -1 || rightIndex !== -1) {
    if (leftIndex === -1) {
      return 1;
    }
    if (rightIndex === -1) {
      return -1;
    }
    return leftIndex - rightIndex;
  }
  return left.localeCompare(right);
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
  const eventTypeSummaries = useMemo(() => {
    const counts = new Map<string, number>();
    lanes.forEach((lane) => {
      lane.milestones.forEach((milestone) => {
        counts.set(milestone.eventType, (counts.get(milestone.eventType) ?? 0) + 1);
      });
    });
    return [...counts.entries()]
      .sort(([left], [right]) => compareEventTypes(left, right))
      .map(([eventType, count]) => ({ eventType, count }));
  }, [lanes]);
  const [visibleEventTypes, setVisibleEventTypes] = useState<Record<string, boolean>>({});

  useEffect(() => {
    setVisibleEventTypes((previous) => {
      const next: Record<string, boolean> = {};
      eventTypeSummaries.forEach(({ eventType }) => {
        next[eventType] = previous[eventType] ?? true;
      });
      return next;
    });
  }, [eventTypeSummaries]);

  const allEventTypesVisible =
    eventTypeSummaries.length === 0 || eventTypeSummaries.every(({ eventType }) => visibleEventTypes[eventType] !== false);
  const visibleMilestoneCount = useMemo(
    () =>
      lanes.reduce(
        (total, lane) =>
          total + lane.milestones.filter((milestone) => visibleEventTypes[milestone.eventType] !== false).length,
        0,
      ),
    [lanes, visibleEventTypes],
  );

  return (
    <section className="panel agent-swimlanes-panel">
      <div className="panel-header">
        <h2>Agent Run Swimlanes</h2>
        <p>Track each agent over time and click a milestone to focus its graph.</p>
      </div>
      {eventTypeSummaries.length > 0 ? (
        <div className="agent-swimlane-filters" aria-label="Toggle milestone card types">
          <div className="agent-swimlane-filter-header">
            <strong>Card types</strong>
            <span>
              Showing {visibleMilestoneCount} of {eventTypeSummaries.reduce((total, summary) => total + summary.count, 0)} cards
            </span>
          </div>
          <div className="agent-swimlane-filter-actions">
            <button
              type="button"
              className={`agent-swimlane-filter-chip agent-swimlane-filter-chip--all ${allEventTypesVisible ? "is-active" : ""}`}
              onClick={() =>
                setVisibleEventTypes(
                  Object.fromEntries(eventTypeSummaries.map(({ eventType }) => [eventType, true])),
                )
              }
            >
              Show all
            </button>
            {eventTypeSummaries.map(({ eventType, count }) => {
              const isVisible = visibleEventTypes[eventType] !== false;
              return (
                <button
                  key={eventType}
                  type="button"
                  aria-pressed={isVisible}
                  className={`agent-swimlane-filter-chip ${isVisible ? "is-active" : ""}`}
                  onClick={() =>
                    setVisibleEventTypes((previous) => ({
                      ...previous,
                      [eventType]: !isVisible,
                    }))
                  }
                >
                  <span>{formatEventTypeLabel(eventType)}</span>
                  <strong>{count}</strong>
                </button>
              );
            })}
          </div>
        </div>
      ) : null}
      <div className="agent-swimlanes">
        {lanes.map((lane) => {
          const visibleMilestones = lane.milestones.filter((milestone) => visibleEventTypes[milestone.eventType] !== false);
          return (
            <section
              key={lane.agentId}
              className={`agent-swimlane ${selectedAgentId === lane.agentId ? "is-selected" : ""}`}
              onClick={() => onSelectAgent(lane.agentId)}
            >
              <div
                role="button"
                tabIndex={0}
                className="agent-swimlane-meta"
                onClick={(event) => {
                  event.stopPropagation();
                  if (hasActiveTextSelection()) {
                    return;
                  }
                  onSelectAgent(lane.agentId);
                }}
                onKeyDown={(event) =>
                  handleKeyboardActivate(event, () => {
                    onSelectAgent(lane.agentId);
                  })
                }
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
                  <RunErrorHover count={lane.errorCount} summaries={lane.errorSummaries} />
                  <span>{lane.elapsedLabel}</span>
                </div>
                <div className="agent-swimlane-current">Current: {lane.currentNodeLabel}</div>
              </div>
              <div className="agent-swimlane-track" role="list" aria-label={`${lane.agentName} milestones`}>
                {lane.milestones.length === 0 ? (
                  <div className="agent-swimlane-empty">No runtime milestones yet.</div>
                ) : visibleMilestones.length === 0 ? (
                  <div className="agent-swimlane-empty">All milestone card types are hidden by the current filters.</div>
                ) : (
                  visibleMilestones.map((milestone) => (
                    <div
                      key={milestone.id}
                      role="listitem"
                      tabIndex={0}
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
                        if (hasActiveTextSelection()) {
                          return;
                        }
                        onSelectAgent(lane.agentId);
                        onSelectNode?.(lane.agentId, milestone.nodeId);
                      }}
                      onKeyDown={(event) =>
                        handleKeyboardActivate(event, () => {
                          onSelectAgent(lane.agentId);
                          onSelectNode?.(lane.agentId, milestone.nodeId);
                        })
                      }
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
                    </div>
                  ))
                )}
              </div>
            </section>
          );
        })}
      </div>
    </section>
  );
}
