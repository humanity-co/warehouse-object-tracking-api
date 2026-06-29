"use client";

import { useEffect, useState, useRef } from "react";
import { useParams } from "next/navigation";
import { callApi, fetchToken } from "@/lib/api";
import { 
  TrendingUp, 
  Package, 
  Camera, 
  Grid, 
  Map, 
  ShieldAlert, 
  Wrench, 
  MessageSquare,
  Activity,
  Send,
  User,
  Bot,
  BrainCircuit,
  AlertTriangle,
  Truck
} from "lucide-react";
import { 
  ResponsiveContainer, 
  AreaChart, 
  Area, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend,
  ComposedChart,
  Bar,
  Line,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  Radar
} from "recharts";
import { motion, AnimatePresence } from "framer-motion";

// Simple hash function for deterministic visual variations
const hash = (str: string) => {
  let h = 0;
  if (!str) return 0;
  for (let i = 0; i < str.length; i++) {
    h = (Math.imul(31, h) + str.charCodeAt(i)) | 0;
  }
  return Math.abs(h);
};

const engineMetadata: Record<string, { label: string, icon: any, description: string }> = {
  demand: { 
    label: "Demand Forecasting", 
    icon: TrendingUp, 
    description: "Temporal Fusion Transformer (TFT) model predicting SKU-level demand with quantile confidence intervals." 
  },
  inventory: { 
    label: "Inventory Optimization", 
    icon: Package, 
    description: "PPO-based Reinforcement Learning agent optimizing reorder points and safety stock levels." 
  },
  vision: { 
    label: "Computer Vision", 
    icon: Camera, 
    description: "YOLOv8 and custom CNNs for damage detection and SKU identification." 
  },
  slotting: { 
    label: "Smart Slotting", 
    icon: Grid, 
    description: "Genetic algorithms optimizing warehouse layout based on velocity and pick frequency." 
  },
  routing: { 
    label: "Pick Path Optimization", 
    icon: Map, 
    description: "Graph Neural Networks (GNNs) solving the Traveling Salesman Problem for picker routes." 
  },
  anomaly: { 
    label: "Anomaly Detection", 
    icon: ShieldAlert, 
    description: "LSTM Autoencoders detecting shifts in operational patterns and sensor data." 
  },
  maintenance: { 
    label: "Predictive Maintenance", 
    icon: Wrench, 
    description: "Temporal Convolutional Networks (TCNs) predicting equipment failure probability." 
  },
  copilot: { 
    label: "AI Copilot", 
    icon: MessageSquare, 
    description: "RAG-driven LLM interface for operational reasoning and natural language querying." 
  },
};

export default function EnginePage() {
  const params = useParams();
  const id = params.id as string;
  const metadata = engineMetadata[id];
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [chatMessages, setChatMessages] = useState<any[]>([]);
  const [inputMessage, setInputMessage] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);
  const [layout, setLayout] = useState<any[]>([]);
  const [events, setEvents] = useState<any[]>([]);
  const [inventoryMapping, setInventoryMapping] = useState<any[]>([]);
  const [pickList, setPickList] = useState<{sku_id: string, quantity: number}[]>([]);
  const [manualRouting, setManualRouting] = useState<any>(null);
  const [bulkInput, setBulkInput] = useState("");
  const [showBulkInput, setShowBulkInput] = useState(false);
  const [safetyOverlay, setSafetyOverlay] = useState(true);
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);
  const [safetyTopology, setSafetyTopology] = useState<any[]>([]);
  const [demandHeatmap, setDemandHeatmap] = useState<any[]>([]);
  const [inventoryHeatmap, setInventoryHeatmap] = useState<any[]>([]);

  useEffect(() => {
    const loadLayout = async () => {
      try {
        const token = await fetchToken();
        const response = await callApi<any[]>("/api/v1/pick-path/layout", token);
        if (Array.isArray(response)) setLayout(response);
        
        const invResponse = await callApi<any[]>("/api/v1/pick-path/inventory", token);
        if (Array.isArray(invResponse)) setInventoryMapping(invResponse);

        if (id === "routing") {
          const safetyResponse = await callApi<any>("/api/v1/routing/safety-topology", token);
          if (safetyResponse && Array.isArray(safetyResponse.nodes)) setSafetyTopology(safetyResponse.nodes);
        } else if (id === "demand") {
          const demandResponse = await callApi<any>("/api/v1/demand/heatmap", token);
          if (demandResponse && Array.isArray(demandResponse.nodes)) setDemandHeatmap(demandResponse.nodes);
        } else if (id === "inventory") {
          const invRiskResponse = await callApi<any>("/api/v1/inventory/heatmap", token);
          if (invRiskResponse && Array.isArray(invRiskResponse.nodes)) setInventoryHeatmap(invRiskResponse.nodes);
        }
      } catch (e) { console.error("Data load failed", e); }
    };
    loadLayout();
  }, [id]);

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [chatMessages]);

  useEffect(() => {
    async function fetchData() {
      if (id === "copilot") {
        setLoading(false);
        return;
      }

      setLoading(true);
      try {
        const token = await fetchToken();
        let endpoint = `/api/v1/${id}`;
        
        if (id === "demand") endpoint = "/api/v1/forecast";
        if (id === "inventory") endpoint = "/api/v1/inventory/SKU-0000/optimize";
        if (id === "vision") endpoint = "/api/v1/vision/scan";
        if (id === "slotting") endpoint = "/api/v1/slotting/optimize";
        if (id === "routing") endpoint = "/api/v1/routing/plan";
        if (id === "anomaly") endpoint = "/api/v1/anomaly/detect";
        if (id === "maintenance") endpoint = "/api/v1/maintenance/predict";

        const response = await callApi(endpoint, token, {
          method: "POST",
          body: JSON.stringify({ sku_id: "SKU-0000", warehouse_id: "WH-01" }),
        });
        setData(response);
      } catch (error) {
        console.error(error);
      }
    }

    const fetchEvents = async () => {
      try {
        const token = await fetchToken();
        const resp = await callApi<any>("/api/v1/events", token);
        if (resp?.data?.events) setEvents(resp.data.events);
      } catch (e) { console.error(e); }
    };

    fetchData();
    fetchEvents();

    const interval = setInterval(() => {
      fetchData();
      fetchEvents();
    }, 2000);
    return () => clearInterval(interval);
  }, [id]);

  const handleSendMessage = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!inputMessage.trim() || isTyping) return;

    const userMsg = { role: "user", text: inputMessage, timestamp: new Date() };
    setChatMessages(prev => [...prev, userMsg]);
    setInputMessage("");
    setIsTyping(true);

    try {
      const token = await fetchToken();
      const response = await callApi("/api/v1/copilot/query", token, {
        method: "POST",
        body: JSON.stringify({ question: inputMessage }),
      }) as any;
      
      const botMsg = { 
        role: "bot", 
        text: response?.answer || "I'm sorry, I couldn't process that request.", 
        sources: response?.sources || [],
        explanation: response?.explanation?.summary || response?.explanation,
        timestamp: new Date() 
      };
      setChatMessages(prev => [...prev, botMsg]);
    } catch (error) {
      console.error("Copilot Error:", error);
      setChatMessages(prev => [...prev, { 
        role: "bot", 
        text: "System error: Copilot temporarily offline.", 
        timestamp: new Date() 
      }]);
    } finally {
      setIsTyping(false);
    }
  };

  if (!metadata) return <div>Engine not found</div>;

  const WarehouseMap = ({ 
    nodes, 
    path = [], 
    highlights = [], 
    onNodeClick,
    mode = id
  }: { 
    nodes: any[], 
    path?: string[], 
    highlights?: string[], 
    onNodeClick?: (nodeId: string) => void,
    mode?: string
  }) => {
    if (!nodes || nodes.length === 0) return null;
    
    const minX = Math.min(...nodes.map(n => n.x)) - 10;
    const maxX = Math.max(...nodes.map(n => n.x)) + 10;
    const minY = Math.min(...nodes.map(n => n.y)) - 10;
    const maxY = Math.max(...nodes.map(n => n.y)) + 10;
    
    const width = maxX - minX;
    const height = maxY - minY;
    
    const racks = nodes.filter(n => n.pick_face);
    
    const pathPoints = (path || []).map(nodeId => {
      const node = nodes.find(n => n.node_id === nodeId);
      return node ? { x: node.x, y: node.y } : null;
    }).filter(p => p !== null) as {x: number, y: number}[];

    return (
      <div className="relative w-full h-full glass-panel overflow-hidden rounded-xl border border-white/10 min-h-[400px]">
        <svg viewBox={`${minX} ${minY} ${width} ${height}`} className="warehouse-svg w-full h-full transition-all duration-500">
          <defs>
            <linearGradient id="pathGradient" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#4F46E5" />
              <stop offset="100%" stopColor="#10B981" />
            </linearGradient>
            <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
              <feGaussianBlur stdDeviation="2" result="blur" />
              <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>
            <radialGradient id="safetyGradient" cx="50%" cy="50%" r="50%">
              <stop offset="0%" stopColor="#EF4444" stopOpacity="0.4" />
              <stop offset="100%" stopColor="#EF4444" stopOpacity="0" />
            </radialGradient>
            <radialGradient id="demandGradient" cx="50%" cy="50%" r="50%">
              <stop offset="0%" stopColor="#8B5CF6" stopOpacity="0.6" />
              <stop offset="100%" stopColor="#8B5CF6" stopOpacity="0" />
            </radialGradient>
            <radialGradient id="riskGradient" cx="50%" cy="50%" r="50%">
              <stop offset="0%" stopColor="#F59E0B" stopOpacity="0.5" />
              <stop offset="100%" stopColor="#F59E0B" stopOpacity="0" />
            </radialGradient>
          </defs>
          
          {/* Aisle Structure */}
          {['A', 'B', 'C', 'D', 'E'].map((zone, i) => (
            <motion.rect
              key={zone}
              x={i * 20 + 5}
              y={0}
              width={10}
              height={height}
              fill="rgba(255,255,255,0.03)"
              rx="4"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
            />
          ))}

          {/* Safety Heatmap Overlay */}
          {safetyOverlay && safetyTopology.map((node, i) => {
            if (node.risk_score < 0.1) return null;
            const layoutNode = nodes.find(n => n.node_id === node.node_id);
            if (!layoutNode) return null;
            return (
              <circle
                key={`risk-${node.node_id}`}
                cx={layoutNode.x}
                cy={layoutNode.y}
                r={8 * node.risk_score}
                fill="url(#safetyGradient)"
                className="animate-pulse"
              />
            );
          })}

          {/* Demand Heatmap Overlay */}
          {mode === "demand" && demandHeatmap.map((node, i) => {
            if (node.intensity < 0.2) return null;
            const layoutNode = nodes.find(n => n.node_id === node.node_id);
            if (!layoutNode) return null;
            return (
              <circle
                key={`demand-${node.node_id}`}
                cx={layoutNode.x}
                cy={layoutNode.y}
                r={10 * node.intensity}
                fill="url(#demandGradient)"
                opacity={0.6}
              />
            );
          })}

          {/* Inventory Risk Heatmap Overlay */}
          {mode === "inventory" && inventoryHeatmap.map((node, i) => {
            if (node.risk_score < 0.2) return null;
            const layoutNode = nodes.find(n => n.node_id === node.node_id);
            if (!layoutNode) return null;
            return (
              <circle
                key={`inv-risk-${node.node_id}`}
                cx={layoutNode.x}
                cy={layoutNode.y}
                r={12 * node.risk_score}
                fill="url(#riskGradient)"
                opacity={0.7}
              />
            );
          })}

          {/* Vision Alert Overlay */}
          {mode === "vision" && [...events].reverse().slice(0, 5).map((e, idx) => {
            if (e.event_type !== "object_detected" || !e.payload?.node_id) return null;
            const node = nodes.find(n => n.node_id === e.payload.node_id);
            if (!node) return null;
            return (
              <motion.g key={`vision-alert-${idx}`} initial={{ scale: 0 }} animate={{ scale: 1 }}>
                <circle cx={node.x} cy={node.y} r={4} fill="#10B981" opacity={0.2}>
                  <animate attributeName="r" values="4;8;4" dur="2s" repeatCount="indefinite" />
                </circle>
                <circle cx={node.x} cy={node.y} r={1.5} fill="#10B981" />
              </motion.g>
            );
          })}

          {/* Racks */}
          {racks.map(node => (
            <motion.rect
              key={node.node_id}
              x={node.x - 1.2}
              y={node.y - 1.5}
              width={2.4}
              height={3}
              rx="0.6"
              className="rack-node cursor-pointer group"
              fill="rgba(118, 168, 182, 0.15)"
              stroke={hoveredNode === node.node_id ? "var(--accent)" : "rgba(118, 168, 182, 0.3)"}
              strokeWidth={hoveredNode === node.node_id ? "0.3" : "0.1"}
              whileHover={{ scale: 1.2, fill: "rgba(118, 168, 182, 0.3)" }}
              onClick={() => onNodeClick?.(node.node_id)}
              onMouseEnter={() => setHoveredNode(node.node_id)}
              onMouseLeave={() => setHoveredNode(null)}
            />
          ))}

          {/* Optimized Path (Routing Only) */}
          {mode === "routing" && pathPoints.length > 1 && (
            <>
              <motion.path
                d={`M ${pathPoints.map(p => `${p.x} ${p.y}`).join(' L ')}`}
                fill="none"
                stroke="url(#pathGradient)"
                strokeWidth="1.5"
                strokeLinecap="round"
                strokeLinejoin="round"
                initial={{ pathLength: 0, opacity: 0 }}
                animate={{ pathLength: 1, opacity: 1 }}
                transition={{ duration: 3, ease: "circOut" }}
                filter="url(#glow)"
              />
              <motion.g
                initial={{ x: pathPoints[0].x, y: pathPoints[0].y }}
                animate={{ 
                  x: pathPoints.map(p => p.x),
                  y: pathPoints.map(p => p.y)
                }}
                transition={{ duration: 10, repeat: Infinity, ease: "linear" }}
              >
                <circle r={2.5} fill="var(--accent)" opacity={0.2} />
                <path d="M -1 -0.5 L 0 1 L 1 -0.5 Z" fill="var(--accent)" filter="url(#glow)" />
              </motion.g>
            </>
          )}

          {/* Depot/Pick Points Highlights */}
          {nodes.map(node => {
            const isHighlighted = highlights.includes(node.node_id);
            const isDepot = node.node_id.includes('DEPOT') || node.node_id.includes('DD');
            
            if (isDepot) return (
              <g key={node.node_id} className="pointer-events-none">
                <circle cx={node.x} cy={node.y} r={3} fill="var(--accent)" opacity={0.1}>
                  <animate attributeName="r" values="3;5;3" dur="2s" repeatCount="indefinite" />
                </circle>
                <circle cx={node.x} cy={node.y} r={1.2} fill="var(--accent)" filter="url(#glow)" />
              </g>
            );

            if (!isHighlighted) return null;

            return (
              <motion.g key={`highlight-${node.node_id}`} initial={{ scale: 0 }} animate={{ scale: 1 }}>
                <circle cx={node.x} cy={node.y} r={2} fill="#FFE066" opacity={0.2} filter="url(#glow)" />
                <circle cx={node.x} cy={node.y} r={0.8} fill="#FFE066" />
                <motion.circle
                  cx={node.x}
                  cy={node.y}
                  r={1.5}
                  fill="none"
                  stroke="#FFE066"
                  strokeWidth="0.2"
                  animate={{ r: [1.5, 3, 1.5], opacity: [1, 0, 1] }}
                  transition={{ repeat: Infinity, duration: 2 }}
                />
              </motion.g>
            );
          })}
        </svg>
        
        {/* Map Controls HUD */}
        <div className="absolute bottom-4 left-4 flex gap-2">
           <div className="glass-panel px-3 py-1 text-[10px] font-bold text-white/60 flex items-center gap-2">
             <div className={`w-2 h-2 rounded-full ${mode === 'demand' ? 'bg-purple-500' : mode === 'inventory' ? 'bg-amber-500' : 'bg-emerald-500'}`} />
             {id.toUpperCase()} ENGINE ACTIVE
           </div>
        </div>
        <div className="absolute bottom-4 right-4 flex gap-2">
           <button 
             onClick={() => setSafetyOverlay(!safetyOverlay)}
             className={`px-3 py-1.5 rounded-lg text-[10px] font-bold tracking-tighter transition-all border ${safetyOverlay ? 'bg-red-500/20 border-red-500/50 text-red-200' : 'bg-white/5 border-white/10 text-white/50'}`}
           >
             SAFETY: {safetyOverlay ? 'ON' : 'OFF'}
           </button>
        </div>
      </div>
    );
  };

  const renderDemandView = (forecast: any) => {
    if (!forecast?.horizons) return <div className="no-data"><AlertTriangle /> No signal received from engine node.</div>;
    const chartData = Object.entries(forecast.horizons).map(([key, val]: [string, any]) => ({
      name: key,
      p10: Math.round(val.p10),
      p50: Math.round(val.p50),
      p90: Math.round(val.p90),
    }));

    return (
      <div className="v-stack gap-lg">
        <div className="map-panel panel bg-strong overflow-hidden h-[300px]">
           <WarehouseMap nodes={layout} mode="demand" />
        </div>
        <div className="chart-container">
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={chartData}>
              <defs>
                <linearGradient id="colorP50" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="var(--accent)" stopOpacity={0.8}/>
                  <stop offset="95%" stopColor="var(--accent)" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--line)" />
              <XAxis dataKey="name" stroke="var(--muted)" />
              <YAxis stroke="var(--muted)" />
              <Tooltip 
                contentStyle={{ background: 'var(--panel-strong)', border: '1px solid var(--line)', borderRadius: '8px' }}
                itemStyle={{ color: 'var(--accent)' }}
              />
              <Legend />
              <Area type="monotone" dataKey="p90" stroke="#7cf0ca" fill="transparent" strokeDasharray="5 5" />
              <Area type="monotone" dataKey="p50" stroke="var(--accent)" fillOpacity={1} fill="url(#colorP50)" />
              <Area type="monotone" dataKey="p10" stroke="#ffb84d" fill="transparent" strokeDasharray="5 5" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
        <div className="metric-row">
          <div className="mini-card highlight">
            <span>Model Name</span>
            <strong>{forecast.model_name}</strong>
          </div>
          <div className="mini-card">
            <span>Confidence Score</span>
            <strong>{(forecast.explanation?.confidence * 100).toFixed(1)}%</strong>
          </div>
          <div className="mini-card">
            <span>Drift Detected</span>
            <strong className={forecast.drift_detected ? 'text-warn' : 'text-accent'}>
              {forecast.drift_detected ? 'Retrain Active' : 'Stable'}
            </strong>
          </div>
        </div>
      </div>
    );
  };

  const renderInventoryView = (inventory: any) => {
    if (!inventory?.sku_id) return <div className="no-data"><AlertTriangle /> No signal received from engine node.</div>;
    
    const projectionData = inventory.stock_projection || [];
    const costData = [
      { subject: 'Holding', A: inventory.cost_breakdown?.holding || 0, fullMark: 100 },
      { subject: 'Shortage', A: inventory.cost_breakdown?.shortage || 0, fullMark: 100 },
      { subject: 'Transport', A: inventory.cost_breakdown?.transport || 0, fullMark: 100 },
      { subject: 'Opportunity', A: inventory.cost_breakdown?.opportunity || 0, fullMark: 100 },
    ];

    return (
      <div className="v-stack gap-lg">
        <div className="map-panel panel bg-strong overflow-hidden h-[300px]">
           <WarehouseMap nodes={layout} mode="inventory" />
        </div>
        <div className="inventory-grid">
          <div className="chart-panel">
            <h3>14-Day Stock Projection</h3>
            <ResponsiveContainer width="100%" height={250}>
              <ComposedChart data={projectionData}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--line)" />
                <XAxis dataKey="day" label={{ value: 'Days Out', position: 'insideBottom', offset: -5 }} stroke="var(--muted)" />
                <YAxis stroke="var(--muted)" />
                <Tooltip contentStyle={{ background: 'var(--panel-strong)', border: '1px solid var(--line)' }} />
                <Legend />
                <Bar dataKey="on_hand" name="Projected On-Hand" fill="var(--accent)" radius={[4, 4, 0, 0]} />
                <Line type="monotone" dataKey="reorder_point" name="Reorder Point" stroke="var(--warn)" strokeDasharray="5 5" />
                <Line type="monotone" dataKey="safety_stock" name="Safety Stock" stroke="var(--muted)" strokeDasharray="3 3" />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
          
          <div className="chart-panel">
            <h3>Cost Efficiency Radar</h3>
            <ResponsiveContainer width="100%" height={250}>
              <RadarChart cx="50%" cy="50%" outerRadius="80%" data={costData}>
                <PolarGrid stroke="var(--line)" />
                <PolarAngleAxis dataKey="subject" tick={{ fill: 'var(--muted)', fontSize: 12 }} />
                <Radar
                  name="Cost Impact"
                  dataKey="A"
                  stroke="var(--accent)"
                  fill="var(--accent)"
                  fillOpacity={0.5}
                />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="metric-row">
          <div className="mini-card highlight">
            <span>Target Reorder</span>
            <strong>{inventory.reorder_quantity} Units</strong>
          </div>
          <div className="mini-card">
            <span>Lead Time</span>
            <strong>{inventory.reorder_eta_days} Days</strong>
          </div>
          <div className="mini-card">
            <span>Supplier Reliability</span>
            <strong>{(inventory.supplier_reliability * 100).toFixed(1)}%</strong>
          </div>
          <div className="mini-card">
            <span>Transfer Route</span>
            <div className="h-stack gap-xs">
              <Truck size={14} className="text-accent" />
              <strong>{inventory.transfer_route}</strong>
            </div>
          </div>
        </div>

        <div className="explanation-box">
          <BrainCircuit size={16} />
          <p>{inventory.explanation?.summary}</p>
        </div>
      </div>
    );
  };

  const renderVisionView = (vision: any) => {
    if (!vision?.detections) return <div className="no-data"><AlertTriangle /> No signal received from engine node.</div>;
    return (
      <div className="v-stack gap-lg">
        <div className="vision-layout-grid grid grid-cols-1 lg:grid-cols-2 gap-md">
          <div className="map-panel panel bg-strong overflow-hidden h-[400px]">
             <div className="panel-header border-b border-white/5 pb-2 mb-2">
               <span className="text-[10px] font-bold text-accent tracking-widest uppercase">Sensor Topology</span>
             </div>
             <WarehouseMap nodes={layout} mode="vision" />
          </div>
          <div className="video-feed-container h-[400px] relative rounded-xl overflow-hidden border border-white/10">
            <video autoPlay loop muted playsInline className="video-feed">
               <source src="https://assets.mixkit.co/videos/preview/mixkit-warehouse-stock-being-sorted-by-robotic-arms-34405-large.mp4" type="video/mp4" />
            </video>
            <div className="detection-overlay">
              {vision.detections.map((det: any, idx: number) => (
                <motion.div 
                  key={idx}
                  className="detection-box"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  style={{
                    left: `${det.bbox[0]}%`,
                    top: `${det.bbox[1]}%`,
                    width: `${det.bbox[2] - det.bbox[0]}%`,
                    height: `${det.bbox[3] - det.bbox[1]}%`,
                    borderColor: det.status === 'Damaged' ? 'var(--warn)' : 'var(--accent)'
                  }}
                >
                  <span className="detection-label">{det.label} ({Math.round(det.confidence * 100)}%)</span>
                </motion.div>
              ))}
            </div>
          </div>
        </div>

        <div className="metric-row">
           <div className="mini-card highlight">
             <span>Active Nodes</span>
             <strong>{vision.active_nodes || 12}</strong>
           </div>
           <div className="mini-card">
             <span>Detections/Sec</span>
             <strong>{vision.throughput || 24.5}</strong>
           </div>
           <div className="mini-card">
             <span>System Load</span>
             <strong>{(vision.load || 42).toFixed(1)}%</strong>
           </div>
        </div>
      </div>
    );
  };

  const renderSlottingView = (slotting: any) => {
    if (!slotting?.assignments) return <div className="no-data"><AlertTriangle /> No signal received from engine node.</div>;
    return (
      <div className="v-stack gap-lg">
        <div className="metric-row">
          <div className="mini-card highlight">
            <span>Objective Score</span>
            <strong>{(slotting.objective_score * 100).toFixed(1)}%</strong>
          </div>
          <div className="mini-card">
            <span>Constraint Violations</span>
            <strong className={slotting.constraint_violations > 0 ? 'text-warn' : 'text-accent'}>
              {slotting.constraint_violations}
            </strong>
          </div>
        </div>
        <div className="assignment-list">
          {Object.entries(slotting.assignments).slice(0, 5).map(([sku, zone]: [string, any]) => (
            <div key={sku} className="assignment-item">
              <span>{sku}</span>
              <Activity size={12} className="muted-icon" />
              <strong>{zone}</strong>
            </div>
          ))}
        </div>
        <div className="explanation-box">
          <BrainCircuit size={16} />
          <p>{slotting.explanation?.summary}</p>
        </div>
      </div>
    );
  };



  const renderRoutingView = (routing: any) => {
    // Use manualRouting if available, otherwise fallback to polling data or events
    let activeRoute = manualRouting || routing;
    if (!activeRoute?.path_length && !activeRoute?.pick_nodes) {
       const latestEvent = [...events].reverse().find(e => e.event_type === "route_planned");
       if (latestEvent) activeRoute = latestEvent.payload;
    }

    const distance = activeRoute?.path_length || activeRoute?.total_distance;
    const stops = activeRoute?.pick_nodes || activeRoute?.stops || [];
    
    const handleRunManualRouting = async () => {
      if (pickList.length === 0) return;
      try {
        const token = await fetchToken();
        const response = await callApi<any>("/api/v1/routing/plan", token, {
          method: "POST",
          body: JSON.stringify({ skus: pickList.map(p => p.sku_id) })
        });
        if (response.plans?.[0]) setManualRouting(response.plans[0]);
      } catch (e) { console.error("Manual routing failed", e); }
    };

    return (
      <div className="v-stack gap-lg h-full">
        <div className="routing-layout-grid-v2">
          {/* Main Map Area */}
          <div className="v-stack gap-md">
            <div className="map-panel panel bg-strong">
              <div className="panel-header">
                <span className="text-accent uppercase text-xs font-bold tracking-widest">REAL-TIME POSITIONING</span>
              </div>
              <div className="map-container relative">
                <WarehouseMap 
                  nodes={layout} 
                  path={stops} 
                  highlights={pickList.map(p => inventoryMapping.find(i => i.sku_id === p.sku_id)?.node_id).filter(Boolean)} 
                  onNodeClick={(nodeId) => {
                    const inv = inventoryMapping.find(i => i.node_id === nodeId);
                    if (inv) {
                      setPickList(prev => [...prev, { sku_id: inv.sku_id, quantity: 1 }]);
                    }
                  }}
                />
              </div>
            </div>

            {/* Manual Pick Bench - Premium Grid Edition */}
            <div className="panel bg-strong manual-bench flex flex-col h-[400px]">
              <div className="panel-header border-b border-white/5 pb-4 mb-4">
                <div className="v-stack gap-xs">
                  <h3 className="text-lg font-bold">Pick List Engine</h3>
                  <p className="text-[10px] text-white/40 uppercase tracking-widest font-medium">Production Batch Builder</p>
                </div>
                <div className="h-stack gap-sm">
                  <button className={`btn-ghost btn-xs rounded-full px-4 ${!showBulkInput ? 'bg-white/10 text-white' : 'text-white/40'}`} onClick={() => setShowBulkInput(false)}>Structured</button>
                  <button className={`btn-ghost btn-xs rounded-full px-4 ${showBulkInput ? 'bg-white/10 text-white' : 'text-white/40'}`} onClick={() => setShowBulkInput(true)}>Bulk CSV</button>
                  <button 
                    className="bg-accent hover:bg-accent/80 text-black font-bold px-6 py-2 rounded-full flex items-center gap-2 transition-all active:scale-95 text-xs"
                    onClick={handleRunManualRouting}
                    disabled={pickList.length === 0}
                  >
                    <Map size={14} /> COMPUTE OPTIMAL PATH
                  </button>
                </div>
              </div>

              <div className="flex-1 overflow-hidden flex flex-col gap-4">
                {!showBulkInput ? (
                  <div className="h-stack gap-2 p-2 bg-white/5 rounded-xl border border-white/5">
                     <select id="sku-select" className="bg-transparent border-none text-xs focus:ring-0 flex-1">
                        <option value="">Select Target SKU...</option>
                        {inventoryMapping.map(item => (
                          <option key={item.sku_id} value={item.sku_id} className="bg-neutral-900">{item.sku_id} at {item.node_id}</option>
                        ))}
                     </select>
                     <div className="w-px h-6 bg-white/10" />
                     <input type="number" id="qty-input" placeholder="QTY" className="bg-transparent border-none text-xs focus:ring-0 w-16" defaultValue={1} />
                     <button className="bg-white/10 hover:bg-white/20 p-2 rounded-lg transition-colors" onClick={() => {
                       const sku = (document.getElementById('sku-select') as HTMLSelectElement).value;
                       const qty = parseInt((document.getElementById('qty-input') as HTMLInputElement).value);
                       if (sku && qty) setPickList([...pickList, { sku_id: sku, quantity: qty }]);
                     }}>
                       <TrendingUp size={16} className="text-accent" />
                     </button>
                  </div>
                ) : (
                  <div className="v-stack gap-2 flex-1">
                    <textarea 
                      className="flex-1 bg-white/5 border border-white/5 rounded-xl p-4 font-mono text-xs focus:ring-1 focus:ring-accent outline-none placeholder:text-white/20" 
                      placeholder="SKU-0001, 10&#10;SKU-0002, 5"
                      value={bulkInput}
                      onChange={(e) => setBulkInput(e.target.value)}
                    />
                    <button className="bg-white/10 hover:bg-white/20 py-2 rounded-lg text-[10px] font-bold tracking-widest uppercase transition-all" onClick={() => {
                      const lines = bulkInput.split('\n');
                      const newPicks = lines.map(line => {
                        const [sku, qty] = line.split(',').map(s => s.trim());
                        return sku && qty ? { sku_id: sku, quantity: parseInt(qty) } : null;
                      }).filter(Boolean) as {sku_id: string, quantity: number}[];
                      setPickList([...pickList, ...newPicks]);
                      setBulkInput("");
                    }}>Ingest Batch</button>
                  </div>
                )}

                <div className="flex-1 overflow-y-auto px-1 custom-scrollbar">
                  <table className="w-full text-left text-[11px]">
                    <thead className="sticky top-0 bg-strong text-white/40 uppercase text-[9px] tracking-widest">
                      <tr>
                        <th className="pb-2 font-medium">SKU ID</th>
                        <th className="pb-2 font-medium">Location</th>
                        <th className="pb-2 font-medium">Quantity</th>
                        <th className="pb-2 text-right">Actions</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-white/5">
                      {pickList.map((pick, i) => {
                        const inv = inventoryMapping.find(item => item.sku_id === pick.sku_id);
                        return (
                          <motion.tr key={i} initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} className="group hover:bg-white/5">
                            <td className="py-2.5 font-bold text-accent">{pick.sku_id}</td>
                            <td className="py-2.5 font-mono opacity-60">{inv?.node_id || 'UNKNOWN'}</td>
                            <td className="py-2.5">{pick.quantity}</td>
                            <td className="py-2.5 text-right">
                              <button onClick={() => setPickList(pickList.filter((_, idx) => idx !== i))} className="text-white/20 hover:text-red-400 opacity-0 group-hover:opacity-100 transition-all">
                                <ShieldAlert size={14} />
                              </button>
                            </td>
                          </motion.tr>
                        );
                      })}
                      {pickList.length === 0 && (
                        <tr>
                          <td colSpan={4} className="py-8 text-center text-white/20">
                            <Package size={32} className="mx-auto mb-2 opacity-10" />
                            <p>Pick list is currently empty</p>
                            <p className="text-[9px] uppercase tracking-tighter mt-1">Select items on the map or use the builder above</p>
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
          
          {/* Inventory & Stats Column */}
          <div className="route-stats-column v-stack gap-md">
            <div className="mini-card highlight">
              <span>Path Distance</span>
              <strong>{distance ? distance.toFixed(1) : '---'}m</strong>
            </div>
            
            <div className="inventory-ref-table">
              <h4 className="text-xs uppercase muted mb-sm">Aisle Content Mapping</h4>
              <div className="inv-table-header">
                 <span>Rack</span>
                 <span>SKU</span>
                 <span>Qty</span>
              </div>
              <div className="inv-table-body">
                {inventoryMapping.map(item => (
                  <div key={item.sku_id} className="inv-table-row">
                    <span className="font-mono text-xs">{item.node_id}</span>
                    <span className="text-accent font-bold">{item.sku_id}</span>
                    <span className={item.quantity < 10 ? 'text-warn' : ''}>{item.quantity}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  };

  const renderAnomalyView = (anomaly: any) => {
    if (anomaly?.anomaly_score === undefined) return <div className="no-data"><AlertTriangle /> No signal received from engine node.</div>;
    return (
      <div className="v-stack gap-lg">
        <div className="status-gauge">
          <div className="gauge-value" style={{ color: anomaly.anomaly_score > 0.7 ? 'var(--warn)' : 'var(--accent)' }}>
            {(anomaly.anomaly_score * 100).toFixed(1)}%
          </div>
          <div className="gauge-label">Anomaly Pressure</div>
        </div>
        <div className={`alert-box ${anomaly.anomaly_score > 0.7 ? 'warn' : 'stable'}`}>
           <AlertTriangle size={18} />
           <div className="v-stack">
             <strong>Recommended Action</strong>
             <p>{anomaly.recommended_action}</p>
           </div>
        </div>
      </div>
    );
  };

  const renderCopilotChat = () => (
    <div className="copilot-chat-container">
      <div className="chat-history">
        {chatMessages.length === 0 && (
          <div className="chat-empty">
            <BrainCircuit size={48} className="muted-icon pulse" />
            <p>Ready to analyze warehouse operations. Ask me about inventory, forecasting, or anomalies.</p>
          </div>
        )}
        {chatMessages.map((msg, idx) => (
          <motion.div 
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            key={idx} 
            className={`chat-bubble ${msg.role}`}
          >
            <div className="bubble-header">
              {msg.role === 'user' ? <User size={14} /> : <Bot size={14} />}
              <span>{msg.role === 'user' ? 'Operator' : 'Warehouse Intelligence'}</span>
            </div>
            <div className="bubble-text">{msg.text}</div>
            {msg.sources && msg.sources.length > 0 && (
              <div className="bubble-sources">
                {msg.sources.slice(0, 3).map((s: string, i: number) => (
                  <span key={i} className="source-tag">{s.split('::')[1] || s}</span>
                ))}
              </div>
            )}
          </motion.div>
        ))}
        {isTyping && (
          <div className="chat-bubble bot typing">
            <div className="typing-dots"><span>.</span><span>.</span><span>.</span></div>
          </div>
        )}
        <div ref={chatEndRef} />
      </div>
      <form onSubmit={handleSendMessage} className="chat-input-row">
        <input 
          type="text" 
          value={inputMessage}
          onChange={(e) => setInputMessage(e.target.value)}
          placeholder="Ask Copilot about SKU stock or demand signals..."
          disabled={isTyping}
        />
        <button type="submit" disabled={!inputMessage.trim() || isTyping}>
          <Send size={18} />
        </button>
      </form>
    </div>
  );


  const renderGenericView = (engineData: any) => {
    if (!engineData) return null;
    return (
      <div className="generic-engine-view">
        <div className="status-grid">
          {Object.entries(engineData).map(([key, value]: [string, any]) => {
            if (typeof value === 'object') return null;
            return (
              <div key={key} className="mini-card">
                <span>{key.replace(/_/g, ' ')}</span>
                <strong>{String(value)}</strong>
              </div>
            );
          })}
        </div>
        {engineData.explanation && (
          <div className="explanation-snippet">
             <BrainCircuit size={18} />
             <p>{engineData.explanation.summary}</p>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="engine-page">
      <header className="engine-header">
        <div className="engine-title-row">
          <Icon size={32} className="engine-icon" />
          <h1>{metadata.label}</h1>
        </div>
        <p className="engine-description">{metadata.description}</p>
      </header>

      <section className="engine-content">
        <div className="panel data-panel">
          <div className="panel-header">
            <h2>Real-time Intelligence Output</h2>
            <span className={loading ? 'pulse' : ''}>{loading ? 'Inference in progress...' : 'Live Engine Output'}</span>
          </div>
          
          <div className="engine-data-view">
            {loading ? (
              <div className="empty-engine-state">
                <Activity size={48} className="spinner" />
                <p>Waiting for engine signal...</p>
              </div>
            ) : id === "demand" ? (
              renderDemandView(data)
            ) : id === "inventory" ? (
              renderInventoryView(data)
            ) : id === "vision" ? (
              renderVisionView(data)
            ) : id === "slotting" ? (
              renderSlottingView(data)
            ) : id === "routing" ? (
              renderRoutingView(data)
            ) : id === "anomaly" ? (
              renderAnomalyView(data)
            ) : id === "copilot" ? (
              renderCopilotChat()
            ) : (
              renderGenericView(data)
            )}
          </div>
        </div>

        <div className="panel explanation-panel">
          <div className="panel-header">
            <h2>Model Decision Rationale</h2>
          </div>
          <div className="explanation-content">
            <p>This engine analyzes multi-modal signals including:</p>
            <ul>
              <li>Historical demand volatility</li>
              <li>Sensor telemetry (vibration, temperature)</li>
              <li>Spatial constraints and bin availability</li>
              <li>Real-time human-in-the-loop corrections</li>
            </ul>
            <div className="rationale-footer">
              <BrainCircuit size={24} />
              <p>Decision weights influenced by <strong>Active Learning</strong> feedback loops from the warehouse floor.</p>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
