"use client";

type DecisionFeedProps = {
  events: Array<{
    event_id: string;
    event_type: string;
    timestamp: string;
    warehouse_id: string;
    subject_id: string;
    payload: Record<string, unknown>;
  }>;
};

export function DecisionFeed({ events }: DecisionFeedProps) {
  return (
    <section className="panel">
      <div className="panel-header">
        <h2>Decision Feed</h2>
        <span>Live event fabric</span>
      </div>
      <div className="feed-list">
        {events.slice(0, 8).map((event) => (
          <article className="feed-item" key={event.event_id}>
            <div>
              <strong>{event.event_type.replaceAll("_", " ")}</strong>
              <span>{event.warehouse_id}</span>
            </div>
            <p>{event.subject_id}</p>
            <code>{JSON.stringify(event.payload)}</code>
          </article>
        ))}
      </div>
    </section>
  );
}

