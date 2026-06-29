"use client";

import { useEffect, useMemo, useState, useTransition } from "react";
import { CopilotPanel } from "@/components/copilot-panel";
import { DecisionFeed } from "@/components/decision-feed";
import { KPIGrid } from "@/components/kpi-grid";
import { callApi, fetchToken, getWebSocketUrl } from "@/lib/api";

type Summary = {
  sku_count: number;
  warehouse_count: number;
  events: number;
  trained_engines: string[];
};

type AnomalyPayload = {
  anomaly_score: number;
  subsystem: string;
  recommended_action: string;
};

type MaintenancePayload = {
  recommendations: Array<{
    failure_probability_72h: number;
    equipment_id: string;
  }>;
};

type EventPayload = {
  events: Array<{
    event_id: string;
    event_type: string;
    timestamp: string;
    warehouse_id: string;
    subject_id: string;
    payload: Record<string, unknown>;
  }>;
};

type CopilotPayload = {
  answer: string;
  sources: string[];
};

export default function Page() {
  const [token, setToken] = useState<string>("");
  const [summary, setSummary] = useState<Summary | null>(null);
  const [anomaly, setAnomaly] = useState<AnomalyPayload | null>(null);
  const [maintenance, setMaintenance] = useState<MaintenancePayload | null>(null);
  const [events, setEvents] = useState<EventPayload["events"]>([]);
  const [copilot, setCopilot] = useState<CopilotPayload>({ answer: "", sources: [] });
  const [busy, startTransition] = useTransition();
  const topMaintenanceRisk = useMemo(
    () => maintenance?.recommendations?.[0]?.failure_probability_72h ?? null,
    [maintenance],
  );

  useEffect(() => {
    let cancelled = false;

    async function bootstrap() {
      const nextToken = await fetchToken();
      if (cancelled) return;
      setToken(nextToken);
      await callApi<Summary>("/api/v1/simulation/bootstrap", nextToken, {
        method: "POST",
        body: JSON.stringify({ seed: 9, sku_count: 12, warehouse_count: 3, days: 90, train: false }),
      });
      const [nextSummary, nextEvents] = await Promise.all([
        callApi<Summary>("/api/v1/simulation/summary", nextToken),
        callApi<EventPayload>("/api/v1/events", nextToken),
      ]);
      if (cancelled) return;
      setSummary(nextSummary);
      setEvents(nextEvents.events);
      void (async () => {
        const nextAnomaly = await callApi<AnomalyPayload>("/api/v1/anomaly/detect", nextToken, {
          method: "POST",
          body: "{}",
        });
        if (!cancelled) {
          setAnomaly(nextAnomaly);
        }
      })();
      void (async () => {
        const nextMaintenance = await callApi<MaintenancePayload>("/api/v1/maintenance/predict", nextToken, {
          method: "POST",
          body: "{}",
        });
        if (!cancelled) {
          setMaintenance(nextMaintenance);
        }
      })();
    }

    bootstrap().catch((error) => {
      console.error(error);
    });

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const socket = new WebSocket(getWebSocketUrl());
    socket.onmessage = (message) => {
      const payload = JSON.parse(message.data) as EventPayload;
      setEvents(payload.events ?? []);
    };
    return () => socket.close();
  }, []);

  async function handleAsk(question: string) {
    if (!token) return;
    startTransition(() => {
      void (async () => {
        const response = await callApi<CopilotPayload>("/api/v1/copilot/query", token, {
          method: "POST",
          body: JSON.stringify({ question }),
        });
        setCopilot(response);
      })();
    });
  }

  return (
    <main className="page-shell">
      <header className="hero">
        <div>
          <span className="eyebrow">Warehouse Intelligence Platform</span>
          <h1>AI as the warehouse control nervous system.</h1>
        </div>
        <div className="hero-metric">
          <strong>{summary?.trained_engines?.length ?? 0}/8</strong>
          <span>engines orchestrated</span>
        </div>
      </header>

      <KPIGrid
        summary={summary}
        anomalyScore={anomaly?.anomaly_score ?? null}
        maintenanceRisk={topMaintenanceRisk}
      />

      <section className="content-grid">
        <DecisionFeed events={events} />
        <section className="panel insights-panel">
          <div className="panel-header">
            <h2>Operational Watchlist</h2>
            <span>Highest-signal AI outputs</span>
          </div>
          <div className="watchlist">
            <article>
              <strong>Anomaly Subsystem</strong>
              <p>{anomaly?.subsystem ?? "Awaiting model output"}</p>
              <span>{anomaly?.recommended_action ?? "Bootstrap the vigilance engine to generate actions."}</span>
            </article>
            <article>
              <strong>Maintenance Hotspot</strong>
              <p>{maintenance?.recommendations?.[0]?.equipment_id ?? "Awaiting model output"}</p>
              <span>
                {topMaintenanceRisk !== null
                  ? `Failure probability ${(topMaintenanceRisk * 100).toFixed(0)}% in the next 72h`
                  : "No risk envelope available yet."}
              </span>
            </article>
          </div>
          min  (hyper_managor)
                    "Tab  / full in the craster"
        </section>
        <CopilotPanel onAsk={handleAsk} answer={copilot.answer} sources={copilot.sources} busy={busy} />
      </section>
    </main>
  );
}


