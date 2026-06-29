"use client";

import { FormEvent, useState } from "react";

type CopilotPanelProps = {
  onAsk: (question: string) => Promise<void>;
  answer: string;
  sources: string[];
  busy: boolean;
};

export function CopilotPanel({ onAsk, answer, sources, busy }: CopilotPanelProps) {
  const [question, setQuestion] = useState("Why was a reorder triggered for SKU-0001?");

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    await onAsk(question);
  }

  return (
    <section className="panel copilot-panel">
      <div className="panel-header">
        <h2>AI Copilot</h2>
        <span>Operational reasoning with action authority</span>
      </div>
      <form className="copilot-form" onSubmit={handleSubmit}>
        <textarea
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          rows={4}
        />
        <button disabled={busy} type="submit">
          {busy ? "Thinking..." : "Query Copilot"}
        </button>
      </form>
      <div className="copilot-answer">
        <p>{answer || "Ask about reorder logic, anomalies, routing, or risks."}</p>
        <div className="source-list">
          {sources.map((source) => (
            <span key={source}>{source}</span>
          ))}
        </div>
      </div>
    </section>
  );
}

