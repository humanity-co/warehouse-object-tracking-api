"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { 
  LayoutDashboard, 
  TrendingUp, 
  Package, 
  Camera, 
  Grid, 
  Map, 
  ShieldAlert, 
  Wrench, 
  MessageSquare,
  Zap,
  Activity
} from "lucide-react";
import { useState } from "react";
import { callApi, fetchToken } from "@/lib/api";

const engines = [
  { id: "dashboard", label: "Dashboard", icon: LayoutDashboard, path: "/" },
  { id: "demand", label: "Demand Forecast", icon: TrendingUp, path: "/engines/demand" },
  { id: "inventory", label: "Inventory Opt", icon: Package, path: "/engines/inventory" },
  { id: "vision", label: "Computer Vision", icon: Camera, path: "/engines/vision" },
  { id: "slotting", label: "Smart Slotting", icon: Grid, path: "/engines/slotting" },
  { id: "routing", label: "Pick Routing", icon: Map, path: "/engines/routing" },
  { id: "anomaly", label: "Anomaly Detection", icon: ShieldAlert, path: "/engines/anomaly" },
  { id: "maintenance", label: "Predictive Maint", icon: Wrench, path: "/engines/maintenance" },
  { id: "copilot", label: "AI Copilot", icon: MessageSquare, path: "/engines/copilot" },
  { id: "simulation", label: "Digital Twin", icon: Zap, path: "/simulation" },
];

export function Sidebar() {
  const pathname = usePathname();
  const [bootstrapping, setBootstrapping] = useState(false);

  async function handleTriggerFeed() {
    setBootstrapping(true);
    try {
      const token = await fetchToken();
      await callApi("/api/v1/simulation/bootstrap", token, {
        method: "POST",
        body: JSON.stringify({ 
          seed: Math.floor(Math.random() * 1000), 
          sku_count: 25, 
          warehouse_count: 6, 
          days: 120, 
          train: true 
        }),
      });
      alert("Simulation bootstrap triggered successfully!");
    } catch (error) {
      console.error(error);
      alert("Failed to trigger simulation.");
    } finally {
      setBootstrapping(false);
    }
  }

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <Activity className="sidebar-logo-icon" />
        <span className="sidebar-logo-text">WIP Control</span>
      </div>
      
      <nav className="sidebar-nav">
        {engines.map((engine) => {
          const Icon = engine.icon;
          const isActive = pathname === engine.path;
          return (
            <Link 
              key={engine.id} 
              href={engine.path}
              className={`sidebar-link ${isActive ? 'active' : ''}`}
            >
              <Icon size={20} />
              <span>{engine.label}</span>
            </Link>
          );
        })}
      </nav>

      <div className="sidebar-footer">
        <button 
          className={`feed-button ${bootstrapping ? 'loading' : ''}`}
          onClick={handleTriggerFeed}
          disabled={bootstrapping}
        >
          <Zap size={18} />
          <span>{bootstrapping ? 'Feeding...' : 'Trigger Data Feed'}</span>
        </button>
      </div>
    </aside>
  );
}
