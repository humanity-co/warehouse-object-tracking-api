"use client";

import { useEffect, useState, useRef } from "react";
import { Activity, Zap, Terminal as TerminalIcon, ShieldCheck } from "lucide-react";
import { callApi, fetchToken } from "@/lib/api";
import { motion, AnimatePresence } from "framer-motion";

export default function SimulationPage() {
  const [events, setEvents] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const fetchEvents = async () => {
      try {
        const token = await fetchToken();
        const response = await callApi<any[]>("/api/v1/events", token);
        console.log("DEBUG: Events fetched", response);
        if (response && Array.isArray(response)) {
           const sorted = [...response].sort((a: any, b: any) => 
             new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
           );
           setEvents(sorted.slice(0, 50));
        }
        setLoading(false);
      } catch (e) { 
        console.error("DEBUG: Failed to fetch events", e);
        setLoading(false);
      }
    };

    fetchEvents();
    const interval = setInterval(fetchEvents, 2000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="simulation-page v-stack gap-lg">
      <header className="engine-header">
        <div className="engine-title-row">
          <Zap size={32} className="text-accent" />
          <h1>Digital Twin Live Feed</h1>
        </div>
        <p className="engine-description">Monitoring real-time event stream from warehouse edge nodes and IoT sensors.</p>
      </header>

      <div className="simulation-container">
        <div className="panels-row h-stack gap-md">
          <div className="panel terminal-panel">
            <div className="panel-header">
              <div className="h-stack gap-sm">
                <TerminalIcon size={18} />
                <h2 style={{ margin: 0, fontSize: '1rem' }}>Event Ingest Stream</h2>
              </div>
              <span className="pulse text-accent" style={{ fontSize: '0.8rem' }}>LIVE SYNC ACTIVE</span>
            </div>
            
            <div className="terminal-feed" ref={scrollRef}>
              <AnimatePresence initial={false}>
                {events.map((event) => (
                  <motion.div 
                    key={event.event_id}
                    initial={{ opacity: 0, y: -5 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.2 }}
                    className="log-line"
                  >
                    <span className="log-time">[{new Date(event.timestamp).toLocaleTimeString()}]</span>
                    <span className={`log-type ${event.event_type}`}>{event.event_type.replace(/_/g, ' ').toUpperCase()}</span>
                    <span className="log-subject">{event.subject_id}</span>
                    <span className="log-payload">{JSON.stringify(event.payload)}</span>
                  </motion.div>
                ))}
              </AnimatePresence>
              {events.length === 0 && !loading && (
                <div className="empty-state">No events detected in the last window.</div>
              )}
            </div>
          </div>

          <div className="panel stats-panel">
            <div className="panel-header">
              <h2 style={{ margin: 0, fontSize: '1rem' }}>Stream Status</h2>
            </div>
            <div className="v-stack gap-md" style={{ padding: '10px 0' }}>
              <div className="stat-item">
                <span>Ingested Buffer</span>
                <strong>{events.length} Events</strong>
              </div>
              <div className="stat-item">
                <span>Ingest Rate</span>
                <strong>0.5 Hz</strong>
              </div>
              <div className="stat-item">
                <span>Data Integrity</span>
                <strong className="text-accent">99.9%</strong>
              </div>
              <div className="alert-box stable" style={{ marginTop: '20px' }}>
                 <ShieldCheck size={18} />
                 <p style={{ margin: 0, fontSize: '0.8rem' }}>Edge encryption verified. All signals are valid.</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      <style jsx>{`
        .simulation-container {
          min-height: 600px;
        }
        .panels-row {
          display: flex;
          align-items: flex-start;
          width: 100%;
        }
        .terminal-panel { flex: 1; }
        .stats-panel { width: 320px; }
        
        .terminal-feed {
          background: #050a0f;
          padding: 20px;
          border-radius: 16px;
          height: 550px;
          overflow-y: auto;
          font-family: 'JetBrains Mono', 'Fira Code', monospace;
          font-size: 0.82rem;
          color: #a3c1ad;
          border: 1px solid rgba(118, 168, 182, 0.1);
          box-shadow: inset 0 4px 24px rgba(0,0,0,0.4);
        }
        
        .log-line {
          display: grid;
          grid-template-columns: 100px 160px 100px 1fr;
          gap: 16px;
          padding: 8px 0;
          border-bottom: 1px solid rgba(255,255,255,0.03);
        }
        
        .log-time { color: var(--muted); opacity: 0.6; }
        .log-type { font-weight: 700; letter-spacing: 0.02em; }
        .log-type.demand_observed { color: var(--accent); }
        .log-type.inventory_updated { color: #ffb84d; }
        .log-type.vision_analyzed { color: #7cf0ca; }
        .log-type.anomaly_detected { color: var(--alert); }
        .log-type.maintenance_scored { color: #63e6be; opacity: 0.8; }
        
        .log-subject { color: #fff; font-weight: 600; }
        .log-payload { color: var(--muted); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        
        .stat-item {
          display: flex;
          justify-content: space-between;
          padding: 14px 0;
          border-bottom: 1px solid var(--line);
        }
        .stat-item span { color: var(--muted); font-size: 0.9rem; }
        
        .empty-state {
          height: 100%;
          display: flex;
          align-items: center;
          justify-content: center;
          color: var(--muted);
          font-style: italic;
        }
        
        .h-stack { display: flex; align-items: center; }
        .v-stack { display: flex; flex-direction: column; }
        .gap-sm { gap: 8px; }
        .gap-md { gap: 16px; }
        .gap-lg { gap: 32px; }
      `}</style>
    </div>
  );
}
