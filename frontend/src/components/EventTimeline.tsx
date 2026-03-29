import type { RuntimeEvent } from "../lib/types";

type EventTimelineProps = {
  events: RuntimeEvent[];
  embedded?: boolean;
};

export function EventTimeline({ events, embedded = false }: EventTimelineProps) {
  const content = (
    <div className="timeline">
      {events.length === 0 ? (
        <div className="empty-panel">Start a run to see live events.</div>
      ) : (
        events
          .slice()
          .reverse()
          .map((event) => (
            <article key={`${event.timestamp}-${event.event_type}`} className="timeline-item">
              <div className="timeline-meta">
                <span>{event.event_type}</span>
                <span>{new Date(event.timestamp).toLocaleTimeString()}</span>
              </div>
              <p>{event.summary}</p>
            </article>
          ))
      )}
    </div>
  );

  if (embedded) {
    return (
      <section className="panel timeline-panel timeline-panel--embedded">
        <div className="panel-header">
          <h2>Execution Timeline</h2>
          <p>Live events from the graph runtime.</p>
        </div>
        {content}
      </section>
    );
  }

  return (
    <section className="panel timeline-panel">
      <div className="panel-header">
        <h2>Execution Timeline</h2>
        <p>Live events from the graph runtime.</p>
      </div>
      {content}
    </section>
  );
}
