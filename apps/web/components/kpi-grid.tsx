"use client";

type KPIGridProps = {
  summary: {
    sku_count?: number;
    warehouse_count?: number;
    events?: number;
    trained_engines?: string[];
  } | null;
  anomalyScore: number | null;
  maintenanceRisk: number | null;
};

export function KPIGrid({ summary, anomalyScore, maintenanceRisk }: KPIGridProps) {
  const cards = [
    {
      label: "SKUs Under Intelligence",
      value: summary?.sku_count ?? "—",
      hint: "Forecasting, slotting, and replenishment coverage",
    },
    {
      label: "Active Warehouses",
      value: summary?.warehouse_count ?? "—",
      hint: "Simulation-backed network footprint",
    },
    {
      label: "Event Throughput",
      value: summary?.events ?? "—",
      hint: "Demand, inventory, routing, and telemetry events",
    },
    {
      label: "Anomaly Pressure",
      value: anomalyScore !== null ? anomalyScore.toFixed(2) : "—",
      hint: "Composite vigilance score across telemetry and operations",
      tone: anomalyScore && anomalyScore > 0.65 ? "warn" : "neutral",
    },
    {
      label: "Maintenance Risk",
      value: maintenanceRisk !== null ? `${(maintenanceRisk * 100).toFixed(0)}%` : "—",
      hint: "Top predicted 72h failure probability",
      tone: maintenanceRisk && maintenanceRisk > 0.7 ? "alert" : "neutral",
    },
    {
      label: "Engines Online",
      value: summary?.trained_engines?.length ?? 0,
      hint: summary?.trained_engines?.join(", ") ?? "Awaiting bootstrap",
    },
  ];

  return (
    <section className="kpi-grid">
      {cards.map((card) => (
        <article className={`kpi-card ${card.tone ?? ""}`} key={card.label}>
          <span>{card.label}</span>
          <strong>{card.value}</strong>
          <p>{card.hint}</p>
        </article>
      ))}
    </section>
  );
}

