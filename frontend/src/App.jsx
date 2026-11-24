import { useCallback, useEffect, useRef, useState } from "react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  PieChart,
  Pie,
  Cell,
  Legend,
} from "recharts";
import "./App.css";

const API_ENDPOINTS = {
  summary: "/api/monitoring/summary",
  risk: "/api/monitoring/risk-distribution",
  hourly: "/api/monitoring/hourly",
  alerts: "/api/monitoring/alerts",
  recent: "/api/monitoring/recent",
  evidently: "/api/monitoring/evidently/latest",
  conceptDrift: "/api/monitoring/concept-drift",
  performanceHistory: "/api/monitoring/concept-drift/history",
};

const RISK_COLORS = {
  HIGH: "#f87171",
  MEDIUM: "#fbbf24",
  LOW: "#4ade80",
};

function formatProbability(value) {
  return `${(Number(value || 0) * 100).toFixed(2)}%`;
}

function formatAmount(value) {
  return Number(value || 0).toLocaleString("en-US", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
}

function formatTimestamp(value) {
  if (!value) return "—";

  const date = new Date(value);

  if (Number.isNaN(date.getTime())) return value;

  return date.toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function RiskBadge({ level }) {
  const riskLevel = (level || "LOW").toUpperCase();

  return (
    <span className={`risk ${riskLevel.toLowerCase()}`}>
      {riskLevel}
    </span>
  );
}

function TransactionTable({ events, onSelectTransaction }) {
  if (!events.length) {
    return (
      <div className="chart-empty">
        No transactions available.
      </div>
    );
  }

  return (
    <div className="table-wrapper">
      <table className="data-table">
        <thead>
          <tr>
            <th>Transaction ID</th>
            <th>Amount</th>
            <th>Fraud Probability</th>
            <th>Prediction</th>
            <th>Risk</th>
            <th>Timestamp</th>
          </tr>
        </thead>

        <tbody>
          {events.map((event) => (
            <tr
              key={event.event_id}
              className="clickable-row"
              onClick={() => onSelectTransaction(event)}
            >
              <td>#{event.transaction_id}</td>
              <td>{formatAmount(event.transaction_amount)}</td>
              <td>{formatProbability(event.fraud_probability)}</td>
              <td>
                {event.prediction === 1 ? "Fraud" : "Legitimate"}
              </td>
              <td>
                <RiskBadge level={event.risk_level} />
              </td>
              <td>{formatTimestamp(event.timestamp)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function App() {
  const [summary, setSummary] = useState(null);
  const [riskData, setRiskData] = useState([]);
  const [hourlyData, setHourlyData] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [recentEvents, setRecentEvents] = useState([]);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [lastUpdated, setLastUpdated] = useState(null);
  const [activePage, setActivePage] = useState("Overview");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [selectedTransaction, setSelectedTransaction] = useState(null);
  const [transactionSearch, setTransactionSearch] = useState("");
  const [riskFilter, setRiskFilter] = useState("ALL");
  const [csvFile, setCsvFile] = useState(null);
  const [batchResults, setBatchResults] = useState(null);
  const [batchLoading, setBatchLoading] = useState(false);
  const [batchError, setBatchError] = useState("");
  const [evidentlyReport, setEvidentlyReport] = useState(null);
  const [conceptDrift, setConceptDrift] = useState(null);
  const [performanceHistory, setPerformanceHistory] = useState([]);

  // LIVE OPERATIONS
  const [liveEvents, setLiveEvents] = useState([]);
  const [liveConnected, setLiveConnected] = useState(false);
  const [liveStats, setLiveStats] = useState({
    total: 0,
    fraud: 0,
    probabilitySum: 0,
  });
  const [liveLastAlert, setLiveLastAlert] = useState(null);
  const [soundEnabled, setSoundEnabled] = useState(false);
  const [liveError, setLiveError] = useState("");
  const audioContextRef = useRef(null);
  const soundEnabledRef = useRef(false);
    // SIMULATION CONTROLS
  const [simulationStatus, setSimulationStatus] = useState(null);
  const [simulationError, setSimulationError] = useState("");
  const [maxTransactions, setMaxTransactions] = useState("100");
  const [simulationRate, setSimulationRate] = useState("5");

  const simulationState =
    simulationStatus?.state ||
    (simulationStatus?.is_paused
      ? "paused"
      : simulationStatus?.is_running
        ? "running"
        : "idle");

  const simulationIsActive =
    simulationState === "running" || simulationState === "paused";

  // Fetch the current simulation status
  const refreshSimulationStatus = async () => {
    try {
      const response = await fetch("/api/simulation/status");
      const data = await response.json().catch(() => ({}));

      if (!response.ok) {
        throw new Error(data.detail || "Unable to retrieve simulation status.");
      }

      setSimulationStatus(data);
      setSimulationError("");
    } catch (error) {
      setSimulationError(error.message || "Unable to retrieve simulation status.");
    }
  };

  // Start a new simulation
  const startSimulation = async () => {
    try {
      setSimulationError("");

      const rate = Number(simulationRate);
      const limit = maxTransactions.trim();

      if (!Number.isFinite(rate) || rate <= 0) {
        throw new Error("Processing rate must be greater than zero.");
      }

      if (limit && (!Number.isInteger(Number(limit)) || Number(limit) < 1)) {
        throw new Error("Maximum transactions must be a positive whole number.");
      }

      const params = new URLSearchParams();
      if (limit) params.set("max_transactions", limit);
      params.set("base_rate", String(rate));

      const response = await fetch(
        `/api/simulation/start?${params.toString()}`,
        { method: "POST" }
      );

      const data = await response.json().catch(() => ({}));

      if (!response.ok) {
        throw new Error(data.detail || "Unable to start simulation.");
      }

      await refreshSimulationStatus();
    } catch (error) {
      setSimulationError(error.message || "Unable to start simulation.");
    }
  };

  // Pause, resume or stop the current simulation
  const controlSimulation = async (action) => {
    try {
      setSimulationError("");

      const response = await fetch(`/api/simulation/${action}`, {
        method: "POST",
      });

      const data = await response.json().catch(() => ({}));

      if (!response.ok) {
        throw new Error(data.detail || `Unable to ${action} simulation.`);
      }

      await refreshSimulationStatus();
    } catch (error) {
      setSimulationError(error.message || `Unable to ${action} simulation.`);
    }
  };

  // Keep the displayed status up to date
  useEffect(() => {
    refreshSimulationStatus();

    const interval = setInterval(refreshSimulationStatus, 3000);

    return () => clearInterval(interval);
  }, []);

  const [transactionAmount, setTransactionAmount] = useState("");
  const [productCategory, setProductCategory] = useState("");
  const [cardNetwork, setCardNetwork] = useState("");
  const [cardType, setCardType] = useState("");
  const [assessmentLoading, setAssessmentLoading] = useState(false);
  const [assessmentError, setAssessmentError] = useState("");
  const [assessmentResult, setAssessmentResult] = useState(null);
  const [transactionDate, setTransactionDate] = useState("");
  const [transactionChannel, setTransactionChannel] = useState("");

const [card1, setCard1] = useState("");
const [card2, setCard2] = useState("");
const [card3, setCard3] = useState("");
const [card5, setCard5] = useState("");

const [addr1, setAddr1] = useState("");
const [addr2, setAddr2] = useState("");
const [dist1, setDist1] = useState("");

const [emailDomain, setEmailDomain] = useState("");
const [recipientEmailDomain, setRecipientEmailDomain] = useState("");

const [deviceType, setDeviceType] = useState("");
const [deviceInfo, setDeviceInfo] = useState("");
const [browserInfo, setBrowserInfo] = useState("");

const [demoReference, setDemoReference] = useState("");
const [demoTransactions, setDemoTransactions] = useState([]);


const handleAssessment = async () => {
  setAssessmentError("");
  setAssessmentResult(null);

  if (!demoReference) {
    setAssessmentError("Please select a demo transaction first.");
    return;
  }

  setAssessmentLoading(true);

  try {
    const response = await fetch(
      `/api/predict-demo?demo_reference=${encodeURIComponent(demoReference)}`,
      {
        method: "POST",
      }
    );

    const data = await response.json();

    if (!response.ok) {
      throw new Error(
        data.detail || "Transaction assessment failed."
      );
    }

    const result = data.prediction;

    setAssessmentResult(
      `Prediction: ${result.is_fraud ? "FRAUD" : "LEGITIMATE"} | Risk: ${result.risk_level} | Fraud probability: ${result.fraud_probability} | Threshold: ${result.threshold}`
    );

  } catch (error) {
    setAssessmentError(error.message);
  } finally {
    setAssessmentLoading(false);
  }
};


const handleBatchAssessment = async () => {
  if (!csvFile) {
    setBatchError("Please select a CSV file first.");
    return;
  }

  setBatchLoading(true);
  setBatchError("");
  setBatchResults(null);

  try {
    const formData = new FormData();
    formData.append("file", csvFile);

    const response = await fetch(
      "/api/predict-csv?report_type=analyst",
      {
        method: "POST",
        body: formData,
      }
    );

    if (!response.ok) {
      let message = "Batch assessment failed.";

      try {
        const errorData = await response.json();
        message = errorData.detail || message;
      } catch {
        // Keep the default error message.
      }

      throw new Error(message);
    }

    const reportBlob = await response.blob();
    const reportUrl = window.URL.createObjectURL(reportBlob);
    const link = document.createElement("a");

    link.href = reportUrl;
    link.download = "fraudintel_analyst_report.csv";
    document.body.appendChild(link);
    link.click();
    link.remove();

    window.URL.revokeObjectURL(reportUrl);

    setBatchResults({
      total_transactions: Number(
        response.headers.get("X-Total-Transactions") || 0
      ),
      fraud_detected: Number(
        response.headers.get("X-Fraud-Detected") || 0
      ),
      legitimate_transactions: Number(
        response.headers.get("X-Legitimate-Transactions") || 0
      ),
      fraud_detection_rate: 0,
      results: [],
    });

    const total = Number(
      response.headers.get("X-Total-Transactions") || 0
    );
    const fraud = Number(
      response.headers.get("X-Fraud-Detected") || 0
    );

    setBatchResults((previous) => ({
      ...previous,
      fraud_detection_rate: total > 0 ? fraud / total : 0,
    }));

  } catch (err) {
    setBatchError(err.message || "Unable to process the CSV file.");
  } finally {
    setBatchLoading(false);
  }
};

 const fetchDashboardData = useCallback(async () => {
  try {
    const responses = await Promise.all(
      Object.values(API_ENDPOINTS).map(async (endpoint) => {
        const response = await fetch(endpoint);

        if (!response.ok) {
          throw new Error(
            `Request failed: ${endpoint} (${response.status})`
          );
        }

        const text = await response.text();

try {
    return JSON.parse(text);
} catch {
    throw new Error(
        `Invalid JSON from ${endpoint}: ${text.slice(0, 200)}`
    );
}
      })
    );

    const [
      summaryResult,
      riskResult,
      hourlyResult,
      alertsResult,
      recentResult,
      evidentlyResult,
      conceptDriftResult,
      historyResult,
    ] = responses;

    setSummary(summaryResult);
    setRiskData(riskResult.risk_distribution || []);

    setHourlyData(
      [...(hourlyResult.hourly_statistics || [])]
        .reverse()
        .map((item) => ({
          ...item,
          hourLabel: formatTimestamp(item.hour),
        }))
    );

    setAlerts(alertsResult.alerts || []);
    setRecentEvents(recentResult.events || []);

    setEvidentlyReport(evidentlyResult);
    setConceptDrift(conceptDriftResult);
    setPerformanceHistory(historyResult.evaluations || []);

    setLastUpdated(new Date());
    setError("");
  } catch (err) {
    console.error("Dashboard data error:", err);
    alert(`Dashboard error: ${err.message}`);
    setError(
      "Unable to refresh monitoring data. Check that FastAPI is running."
    );
  } finally {
    setLoading(false);
  }
}, []);
  useEffect(() => {
    fetchDashboardData();

    const interval = setInterval(fetchDashboardData, 10000);


    return () => clearInterval(interval);
  }, [fetchDashboardData]);

  useEffect(() => {
    soundEnabledRef.current = soundEnabled;
  }, [soundEnabled]);

  // Receive transaction events pushed by FastAPI over WebSocket.
  useEffect(() => {
    let socket = null;
    let reconnectTimer = null;
    let disposed = false;

    const connect = () => {
      if (disposed) return;

      const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      socket = new WebSocket(
        `${protocol}//${window.location.host}/api/ws/live`
      );

      socket.onopen = () => {
        setLiveConnected(true);
        setLiveError("");
      };

      socket.onmessage = (message) => {
        try {
          const payload = JSON.parse(message.data);
          const event = payload.event || payload;

          if (event.transaction_id === undefined || event.prediction === undefined) {
            return;
          }

          const normalizedEvent = {
            ...event,
            prediction: Number(event.prediction),
            fraud_probability: Number(event.fraud_probability || 0),
          };

          setLiveEvents((previous) => [normalizedEvent, ...previous].slice(0, 100));
          setRecentEvents((previous) => [
            normalizedEvent,
            ...previous.filter(
              (item) => item.event_id !== normalizedEvent.event_id
            ),
          ].slice(0, 100));

          setLiveStats((previous) => ({
            total: previous.total + 1,
            fraud: previous.fraud + (normalizedEvent.prediction === 1 ? 1 : 0),
            probabilitySum:
              previous.probabilitySum + normalizedEvent.fraud_probability,
          }));

          if (normalizedEvent.prediction === 1) {
            setLiveLastAlert(normalizedEvent);
            setAlerts((previous) => [
              normalizedEvent,
              ...previous.filter(
                (item) => item.event_id !== normalizedEvent.event_id
              ),
            ].slice(0, 100));
        if (
          soundEnabledRef.current &&
          audioContextRef.current &&
          audioContextRef.current.state === "running"
        ) {
          const context = audioContextRef.current;
          const now = context.currentTime;

          const sirenFrequencies = [1000, 650];
          const toneDuration = 0.3;
          const cycles = 5;

          for (let i = 0; i < cycles * 2; i++) {
            const oscillator = context.createOscillator();
            const gain = context.createGain();

            const startTime = now + i * toneDuration;
            const endTime = startTime + toneDuration;

            oscillator.type = "sawtooth";

            oscillator.frequency.setValueAtTime(
              sirenFrequencies[i % 2],
              startTime
            );

            gain.gain.setValueAtTime(0.001, startTime);

            gain.gain.linearRampToValueAtTime(
              0.25,
              startTime + 0.03
            );

            gain.gain.linearRampToValueAtTime(
              0.001,
              endTime - 0.03
            );

            oscillator.connect(gain);
            gain.connect(context.destination);

            oscillator.start(startTime);
            oscillator.stop(endTime);
          }
        }
      }

      setLastUpdated(new Date());
    } catch (eventError) {
      console.error("Invalid live event received:", eventError);
    }
  };

  socket.onerror = () => {
    setLiveError("Live connection unavailable. Reconnecting...");
  };

  socket.onclose = () => {
    setLiveConnected(false);

    if (!disposed) {
      reconnectTimer = window.setTimeout(connect, 3000);
    }
  };
};

connect();

return () => {
  disposed = true;

  if (reconnectTimer) {
    window.clearTimeout(reconnectTimer);
  }

  if (socket) {
    socket.onclose = null;
    socket.close();
  }

  setLiveConnected(false);
};
}, []);

useEffect(() => {
  const loadDemoTransactions = async () => {
    try {
      const response = await fetch("/api/demo-transactions");
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Unable to load demo transactions.");
      }

      setDemoTransactions(data.transactions || []);
    } catch (error) {
      setAssessmentError(error.message);
    }
  };

  loadDemoTransactions();
}, []);

  const totalTransactions = summary?.total_transactions ?? 0;
  const fraudAlerts = summary?.fraud_alerts ?? 0;
  const legitimateTransactions =
    summary?.legitimate_transactions ?? 0;

  const alertRate =
    totalTransactions > 0
      ? (fraudAlerts / totalTransactions) * 100
      : 0;

  const averageProbability =
    summary?.average_fraud_probability ?? 0;
  const filteredTransactions = recentEvents.filter((event) => {
  const search = transactionSearch.toLowerCase();

  const matchesSearch =
    String(event.transaction_id).toLowerCase().includes(search) ||
    String(event.risk_level || "").toLowerCase().includes(search) ||
    (event.prediction === 1 ? "fraud" : "legitimate").includes(search);

  const matchesRisk =
    riskFilter === "ALL" || event.risk_level === riskFilter;

  return matchesSearch && matchesRisk;
});

  return (
    <div className="app">
      <aside className={`sidebar ${sidebarOpen ? "sidebar-open" : ""}`}>
        <div className="brand">
          <div className="brand-icon">F</div>

          <div>
            <h1>FraudIntel</h1>
            <span>Fraud Intelligence Platform</span>
          </div>
        </div>

        <nav>
          {[
            "Overview",
            "Live Operations",
            "New Assessment",
            "Batch Assessment",
            "Transactions",
            "Alerts",
            "Monitoring",
            "Model",
          ].map((page) => (
            <a
              key={page}
              href="#"
              className={activePage === page ? "active" : ""}
              onClick={(event) => {
                event.preventDefault();
                setActivePage(page);
                setSidebarOpen(false);
              }}
            >
              {page}
            </a>
          ))}
        </nav>

        <div className="sidebar-status">
          <span className="status-dot"></span>
          System operational
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <button
  className="sidebar-toggle"
  onClick={() => setSidebarOpen(!sidebarOpen)}
  aria-label="Toggle navigation"
>
  ☰
</button>
          <div>
            <h2>
              {activePage === "Overview"
                ? "Fraud Detection Overview"
                : activePage}
            </h2>

            <p>
              {activePage === "Overview"
                ? "Real-time transaction risk monitoring"
                : activePage === "Live Operations"
                  ? "Live simulated transaction monitoring and fraud alerts"
                  : `${activePage} management and insights`}
            </p>
          </div>

          <div className="model-status">
            <span className="status-dot"></span>
            LightGBM v1 · Active
          </div>
        </header>

        {error && (
          <div className="dashboard-error">
            {error}
          </div>
        )}

        {/* OVERVIEW PAGE */}
        {activePage === "Overview" && (
          <div>
            <section className="kpi-grid">
              <div className="card kpi">
                <span>Total Transactions</span>
                <strong>
                  {loading && !summary
                    ? "..."
                    : totalTransactions.toLocaleString()}
                </strong>
                <small>Processed transactions</small>
              </div>

              <div className="card kpi">
                <span>Fraud Alerts</span>
                <strong>
                  {loading && !summary
                    ? "..."
                    : fraudAlerts.toLocaleString()}
                </strong>
                <small>Transactions flagged by the model</small>
              </div>

              <div className="card kpi">
                <span>Alert Rate</span>
                <strong>
                  {loading && !summary
                    ? "..."
                    : `${alertRate.toFixed(2)}%`}
                </strong>
                <small>Share of transactions flagged</small>
              </div>

              <div className="card kpi">
                <span>Avg. Fraud Probability</span>
                <strong>
                  {loading && !summary
                    ? "..."
                    : formatProbability(averageProbability)}
                </strong>
                <small>Across monitored transactions</small>
              </div>
            </section>

            <section className="dashboard-grid">
              <div className="card chart-card">
                <div className="card-header">
                  <div>
                    <h3>Transaction Activity</h3>
                    <p>Transactions processed by hour</p>
                  </div>
                </div>

                <div className="chart-container">
                  {hourlyData.length > 0 ? (
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={hourlyData}>
                        <CartesianGrid
                          strokeDasharray="3 3"
                          stroke="#334155"
                        />
                        <XAxis
                          dataKey="hourLabel"
                          stroke="#94a3b8"
                          tick={{ fontSize: 10 }}
                        />
                        <YAxis
                          stroke="#94a3b8"
                          tick={{ fontSize: 10 }}
                        />
                        <Tooltip
                          contentStyle={{
                            background: "#111827",
                            border: "1px solid #334155",
                            borderRadius: "8px",
                            color: "#e5e7eb",
                          }}
                        />
                        <Legend />
                        <Bar
                          dataKey="total_transactions"
                          name="Transactions"
                          fill="#3b82f6"
                          radius={[4, 4, 0, 0]}
                        />
                        <Bar
                          dataKey="fraud_alerts"
                          name="Fraud alerts"
                          fill="#f87171"
                          radius={[4, 4, 0, 0]}
                        />
                      </BarChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="chart-empty">
                      {loading
                        ? "Loading activity..."
                        : "No activity data available"}
                    </div>
                  )}
                </div>
              </div>

              <div className="card">
                <div className="card-header">
                  <div>
                    <h3>Risk Distribution</h3>
                    <p>Monitored transaction risk levels</p>
                  </div>
                </div>

                <div className="risk-chart-container">
                  {riskData.length > 0 ? (
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={riskData}
                          dataKey="transactions"
                          nameKey="risk_level"
                          cx="50%"
                          cy="45%"
                          outerRadius={85}
                          innerRadius={48}
                          paddingAngle={3}
                        >
                          {riskData.map((item) => (
                            <Cell
                              key={item.risk_level}
                              fill={
                                RISK_COLORS[item.risk_level] || "#64748b"
                              }
                            />
                          ))}
                        </Pie>

                        <Tooltip
                          contentStyle={{
                            background: "#111827",
                            border: "1px solid #334155",
                            borderRadius: "8px",
                            color: "#e5e7eb",
                          }}
                        />
                        <Legend />
                      </PieChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="chart-empty">
                      {loading
                        ? "Loading risk data..."
                        : "No risk data available"}
                    </div>
                  )}
                </div>

                <div className="risk-list">
                  {["HIGH", "MEDIUM", "LOW"].map((level) => {
                    const item = riskData.find(
                      (entry) => entry.risk_level === level
                    );

                    return (
                      <div key={level}>
                        <RiskBadge level={level} />
                        <strong>{item?.transactions ?? 0}</strong>
                      </div>
                    );
                  })}
                </div>
              </div>
            </section>

            <section className="card alerts-card">
              <div className="card-header">
                <div>
                  <h3>Recent Fraud Alerts</h3>
                  <p>
                    Transactions flagged by the fraud detection model
                  </p>
                </div>

                <span className="alert-count">
                  {alerts.length} alerts
                </span>
              </div>

              {alerts.length > 0 ? (
                <TransactionTable events={alerts} />
              ) : (
                <div className="empty-state">
                  <div className="alert-icon">!</div>
                  <div>
                    <strong>No fraud alerts returned</strong>
                    <p>New flagged transactions will appear here.</p>
                  </div>
                </div>
              )}
            </section>

            <section className="card recent-card">
              <div className="card-header">
                <div>
                  <h3>Recent Transactions</h3>
                  <p>
                    Latest events received by the monitoring system
                  </p>
                </div>
              </div>

              <TransactionTable events={recentEvents} />
            </section>
          </div>
        )}



{/* LIVE OPERATIONS PAGE */}
{activePage === "Live Operations" && (
  <div className="live-operations live-monitoring-screen">

    {/* TOP TOOLBAR */}
    <section className="card live-toolbar">
      <div>
        <h3>Live Transaction Operations</h3>
        <p>
          Streaming simulated transactions through FraudIntel_LightGBM.
          Events appear as the API processes them.
        </p>
      </div>

      <div className="live-toolbar-actions">
        <span className={`live-connection ${liveConnected ? "connected" : "disconnected"}`}>
          <span className="live-connection-dot"></span>
          {liveConnected ? "LIVE CONNECTED" : "CONNECTING"}
        </span>

        <button
          type="button"
          className={soundEnabled ? "sound-button enabled" : "sound-button"}
          onClick={async () => {
            if (soundEnabled) {
              setSoundEnabled(false);
              return;
            }

            const AudioContextClass =
              window.AudioContext || window.webkitAudioContext;

            if (!AudioContextClass) {
              setLiveError("This browser does not support audio alerts.");
              return;
            }

            try {
              if (!audioContextRef.current) {
                audioContextRef.current = new AudioContextClass();
              }

              await audioContextRef.current.resume();
              setSoundEnabled(true);
              setLiveError("");
            } catch {
              setLiveError("Unable to enable sound. Check browser audio permissions.");
            }
          }}
        >
          {soundEnabled ? "Mute alarm" : "Enable sound alarm"}
        </button>
      </div>
    </section>

    {liveError && (
      <div className="dashboard-error">{liveError}</div>
    )}

    {/* DEMO NOTICE */}
    <div className="live-demo-notice">
      <strong>SIMULATION MODE</strong>
      <span>
        Anonymised IEEE-CIS dataset records. Demonstration stream,
        not live bank traffic.
      </span>
    </div>

    {/* SESSION METRICS */}
    <section className="kpi-grid live-session-kpis">

      <div className="card kpi">
        <span>Events received</span>
        <strong>{liveStats.total.toLocaleString()}</strong>
        <small>WebSocket transaction events</small>
      </div>

      <div className="card kpi">
        <span>Fraud alerts</span>
        <strong className="live-fraud-value">
          {liveStats.fraud.toLocaleString()}
        </strong>
        <small>Transactions predicted as fraud</small>
      </div>

      <div className="card kpi">
        <span>Session alert rate</span>
        <strong>
          {liveStats.total > 0
            ? `${((liveStats.fraud / liveStats.total) * 100).toFixed(2)}%`
            : "0.00%"}
        </strong>
        <small>Fraud predictions / received events</small>
      </div>

      <div className="card kpi">
        <span>Average fraud probability</span>
        <strong>
          {liveStats.total > 0
            ? formatProbability(liveStats.probabilitySum / liveStats.total)
            : "0.00%"}
        </strong>
        <small>Current browser session</small>
      </div>

    </section>

    {/* LATEST FRAUD ALERT */}
    {liveLastAlert && (
      <section className="live-alert-banner" role="alert">
        <div className="live-alert-icon">!</div>

        <div className="live-alert-content">
          <strong>FRAUD ALERT DETECTED</strong>
          <span>
            Transaction #{liveLastAlert.transaction_id} ·{" "}
            {formatProbability(liveLastAlert.fraud_probability)} fraud
            probability · {liveLastAlert.risk_level} risk
          </span>
        </div>

        <button
          type="button"
          onClick={() => setLiveLastAlert(null)}
          aria-label="Dismiss latest alert"
        >
          Dismiss
        </button>
      </section>
    )}

    {/* MAIN LIVE MONITORING AREA */}
    <section className="live-monitoring-grid">

      {/* LIVE CHART */}
      <section className="card chart-card live-monitoring-chart">
        <div className="card-header">
          <div>
            <h3>Live Fraud Probability</h3>
            <p>Latest transactions received in this browser session</p>
          </div>

          <span className="live-chart-caption">
            {liveEvents.length} recent events
          </span>
        </div>

        <div className="chart-container live-chart-container">
          {liveEvents.length > 0 ? (
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={[...liveEvents].slice(0, 20).reverse().map((event) => ({
                  label: `#${event.transaction_id}`,
                  probability: Number(
                    (event.fraud_probability * 100).toFixed(2)
                  ),
                }))}
              >
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="#334155"
                />

                <XAxis
                  dataKey="label"
                  stroke="#94a3b8"
                  tick={{ fontSize: 10 }}
                />

                <YAxis
                  stroke="#94a3b8"
                  tick={{ fontSize: 10 }}
                  tickFormatter={(value) => `${value}%`}
                />

                <Tooltip
                  formatter={(value) => [
                    `${Number(value).toFixed(2)}%`,
                    "Fraud probability",
                  ]}
                  contentStyle={{
                    background: "#111827",
                    border: "1px solid #334155",
                    borderRadius: "8px",
                    color: "#e5e7eb",
                  }}
                />

                <Bar
                  dataKey="probability"
                  name="Fraud probability"
                  radius={[4, 4, 0, 0]}
                >
                  {[...liveEvents].slice(0, 20).reverse().map((event) => (
                    <Cell
                      key={event.event_id}
                      fill={
                        Number(event.prediction) === 1
                          ? "#ef4444"
                          : "#3b82f6"
                      }
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="chart-empty">
              {liveConnected
                ? "Waiting for the next transaction event..."
                : "Waiting for the live WebSocket connection..."}
            </div>
          )}
        </div>
      </section>

      {/* SIMULATION CONTROLS */}
      <section className="card simulation-control-panel live-simulation-panel">

        <div className="card-header">
          <div>
            <h3>Simulation Controls</h3>
            <p>Manage the transaction stream</p>
          </div>

          <span className={`simulation-status ${simulationState}`}>
            {simulationState.toUpperCase()}
          </span>
        </div>

        {simulationError && (
          <div className="dashboard-error">{simulationError}</div>
        )}

        <div className="simulation-settings">
          <label>
            Maximum transactions
            <input
              type="number"
              min="1"
              value={maxTransactions}
              onChange={(e) => setMaxTransactions(e.target.value)}
              disabled={simulationIsActive}
            />
          </label>

          <label>
            Processing rate (transactions/second)
            <input
              type="number"
              min="0.01"
              step="0.1"
              value={simulationRate}
              onChange={(e) => setSimulationRate(e.target.value)}
              disabled={simulationIsActive}
            />
          </label>
        </div>

        <div className="simulation-status-grid">
          <div>
            <span>Processed</span>
            <strong>
              {(simulationStatus?.processed ?? 0).toLocaleString()}
            </strong>
          </div>

          <div>
            <span>Failed</span>
            <strong>
              {(simulationStatus?.failed ?? 0).toLocaleString()}
            </strong>
          </div>

          <div>
            <span>Simulation ID</span>
            <strong>
              {simulationStatus?.simulation_id
                ? simulationStatus.simulation_id.slice(0, 8)
                : "—"}
            </strong>
          </div>
        </div>

        <div className="simulation-actions">
          <button
            type="button"
            className="primary-button"
            onClick={startSimulation}
            disabled={simulationIsActive}
          >
            Start
          </button>

          <button
            type="button"
            onClick={() => controlSimulation("pause")}
            disabled={simulationState !== "running"}
          >
            Pause
          </button>

          <button
            type="button"
            onClick={() => controlSimulation("resume")}
            disabled={simulationState !== "paused"}
          >
            Resume
          </button>

          <button
            type="button"
            className="danger-button"
            onClick={() => controlSimulation("stop")}
            disabled={!simulationIsActive}
          >
            Stop
          </button>
        </div>

      </section>
    </section>

    {/* COLLAPSIBLE TRANSACTION FEED */}
    <details className="live-feed-details">
      <summary>
        <span>
          <strong>Live Transaction Feed</strong>
          <small>Open to inspect individual transactions</small>
        </span>

        <span className="alert-count">
          {liveEvents.length} events
        </span>
      </summary>

      <div className="live-feed-content">
        <TransactionTable
          events={liveEvents}
          onSelectTransaction={setSelectedTransaction}
        />
      </div>
    </details>

  </div>
)}



{/* NEW ASSESSMENT PAGE */}
{activePage === "New Assessment" && (
  <section className="card assessment-page">
    <div className="card-header">
      <div>
        <h3>New Transaction Assessment</h3>
        <p>Enter transaction details for fraud risk analysis</p>
      </div>
    </div>

    <div className="assessment-form assessment-grid">

      {/* DEMO TRANSACTION */}
      <div className="assessment-demo">
        <h4>Demo Transaction</h4>

        <label>Select a sample transaction</label>
        <select
          value={demoReference}
          onChange={(e) => {
            const reference = e.target.value;
            setDemoReference(reference);

            const sample = demoTransactions.find(
              (item) => item.demo_reference === reference
            );

            if (!sample) return;

            setTransactionAmount(String(sample.TransactionAmt ?? ""));
            setProductCategory(sample.ProductCD ?? "");
            setCard1(String(sample.card1 ?? ""));
            setCard2(String(sample.card2 ?? ""));
            setCard3(String(sample.card3 ?? ""));
            setCardNetwork(sample.card4 ?? "");
            setCardType(sample.card6 ?? "");
            setCard5(String(sample.card5 ?? ""));
            setAddr1(String(sample.addr1 ?? ""));
            setAddr2(String(sample.addr2 ?? ""));
            setDist1(String(sample.dist1 ?? ""));
            setEmailDomain(sample.P_emaildomain ?? "");
            setRecipientEmailDomain(sample.R_emaildomain ?? "");
            setDeviceType(sample.DeviceType ?? "");
            setDeviceInfo(sample.DeviceInfo ?? "");
            setBrowserInfo(sample.id_31 ?? "");
          }}
        >
          <option value="">Choose a sample transaction</option>

          {demoTransactions.map((sample) => (
            <option
              key={sample.demo_reference}
              value={sample.demo_reference}
            >
              {`Sample ${sample.demo_reference} — ${sample.ProductCD} — ${sample.TransactionAmt}`}
            </option>
          ))}
        </select>

        <p>
          Select a real anonymised IEEE-CIS transaction. FraudIntel uses
          its complete feature record for prediction.
        </p>
      </div>

      {/* TRANSACTION DETAILS */}
      <h4>Transaction Details</h4>

      <div className="assessment-field">
        <label>Transaction Amount</label>
        <input
          type="number"
          min="0.01"
          step="0.01"
          placeholder="Transaction amount"
          value={transactionAmount}
          onChange={(e) => setTransactionAmount(e.target.value)}
        />
      </div>

      <div className="assessment-field">
        <label>Transaction Date & Time</label>
        <input
          type="datetime-local"
          value={transactionDate}
          onChange={(e) => setTransactionDate(e.target.value)}
        />
      </div>

      <div className="assessment-field">
        <label>Product Category</label>
        <select
          value={productCategory}
          onChange={(e) => setProductCategory(e.target.value)}
        >
          <option value="">Select category</option>
          <option value="W">W</option>
          <option value="H">H</option>
          <option value="C">C</option>
          <option value="S">S</option>
          <option value="R">R</option>
        </select>
      </div>

      <div className="assessment-field">
        <label>Transaction Channel</label>
        <select
          value={transactionChannel}
          onChange={(e) => setTransactionChannel(e.target.value)}
        >
          <option value="">Select channel</option>
          <option value="web">Web</option>
          <option value="mobile">Mobile</option>
          <option value="pos">POS</option>
          <option value="other">Other</option>
        </select>
      </div>

      {/* CARD DETAILS */}
      <h4>Card Details</h4>

      <div className="assessment-field">
        <label>Card Identifier</label>
        <input
          type="text"
          placeholder="Card identifier"
          value={card1}
          onChange={(e) => setCard1(e.target.value)}
        />
      </div>

      <div className="assessment-field">
        <label>Card Information</label>
        <input
          type="text"
          placeholder="Optional card information"
          value={card2}
          onChange={(e) => setCard2(e.target.value)}
        />
      </div>

      <div className="assessment-field">
        <label>Card Network</label>
        <select
          value={cardNetwork}
          onChange={(e) => setCardNetwork(e.target.value)}
        >
          <option value="">Select network</option>
          <option value="visa">Visa</option>
          <option value="mastercard">Mastercard</option>
          <option value="discover">Discover</option>
          <option value="american express">American Express</option>
        </select>
      </div>

      <div className="assessment-field">
        <label>Card Type</label>
        <select
          value={cardType}
          onChange={(e) => setCardType(e.target.value)}
        >
          <option value="">Select type</option>
          <option value="credit">Credit</option>
          <option value="debit">Debit</option>
        </select>
      </div>

      <div className="assessment-field">
        <label>Card Issuing Information</label>
        <input
          type="text"
          placeholder="Optional card information"
          value={card3}
          onChange={(e) => setCard3(e.target.value)}
        />
      </div>

      <div className="assessment-field">
        <label>Additional Card Information</label>
        <input
          type="text"
          placeholder="Optional card information"
          value={card5}
          onChange={(e) => setCard5(e.target.value)}
        />
      </div>

      {/* ADDRESS & EMAIL */}
      <h4>Address & Email</h4>

      <div className="assessment-field">
        <label>Billing Address</label>
        <input
          type="text"
          placeholder="Billing address"
          value={addr1}
          onChange={(e) => setAddr1(e.target.value)}
        />
      </div>

      <div className="assessment-field">
        <label>Address Region</label>
        <input
          type="text"
          placeholder="Address region"
          value={addr2}
          onChange={(e) => setAddr2(e.target.value)}
        />
      </div>

      <div className="assessment-field">
        <label>Purchaser Email Domain</label>
        <input
          type="text"
          placeholder="example.com"
          value={emailDomain}
          onChange={(e) => setEmailDomain(e.target.value)}
        />
      </div>

      <div className="assessment-field">
        <label>Recipient Email Domain</label>
        <input
          type="text"
          placeholder="example.com"
          value={recipientEmailDomain}
          onChange={(e) => setRecipientEmailDomain(e.target.value)}
        />
      </div>

      <div className="assessment-field">
        <label>Distance Indicator</label>
        <input
          type="number"
          step="0.01"
          placeholder="Distance indicator"
          value={dist1}
          onChange={(e) => setDist1(e.target.value)}
        />
      </div>

      {/* DEVICE DETAILS */}
      <h4>Device Information</h4>

      <div className="assessment-field">
        <label>Device Type</label>
        <input
          type="text"
          placeholder="e.g. desktop, mobile"
          value={deviceType}
          onChange={(e) => setDeviceType(e.target.value)}
        />
      </div>

      <div className="assessment-field">
        <label>Device Information</label>
        <input
          type="text"
          placeholder="Device information"
          value={deviceInfo}
          onChange={(e) => setDeviceInfo(e.target.value)}
        />
      </div>

      <div className="assessment-field">
        <label>Browser Information</label>
        <input
          type="text"
          placeholder="Chrome, Firefox, Safari"
          value={browserInfo}
          onChange={(e) => setBrowserInfo(e.target.value)}
        />
      </div>

      {/* ACTION */}
      <button
        className="assessment-submit"
        type="button"
        onClick={handleAssessment}
        disabled={assessmentLoading}
      >
        {assessmentLoading ? "Assessing..." : "Assess Transaction"}
      </button>

      {assessmentError && (
        <p className="error-message assessment-message">
          {assessmentError}
        </p>
      )}

      {assessmentResult && (
        <div className="batch-results assessment-message">
          <h4>Assessment Result</h4>
          <p>{assessmentResult}</p>
        </div>
      )}

    </div>
  </section>
)}

{/* BATCH ASSESSMENT PAGE */}
{activePage === "Batch Assessment" && (
  <section className="card">
    <div className="card-header">
      <div>
        <h3>CSV Batch Assessment</h3>
        <p>Upload transaction data for bulk fraud analysis</p>
      </div>
    </div>

    <div className="assessment-form">
      <label>Upload Transaction CSV</label>

      <input
        type="file"
        accept=".csv"
        onChange={(e) => {
          setCsvFile(e.target.files[0] || null);
          setBatchResults(null);
          setBatchError("");
        }}
      />

      {csvFile && (
        <small>Selected file: {csvFile.name}</small>
      )}

      <p>Upload a CSV containing prepared transaction features.</p>

      <button
        type="button"
        onClick={handleBatchAssessment}
        disabled={!csvFile || batchLoading}
      >
        {batchLoading
          ? "Assessing Transactions..."
          : "Run Batch Assessment"}
      </button>

      {batchError && (
        <p className="error-message">{batchError}</p>
      )}

      {batchResults && (
        <div className="batch-results">
          <h3>Batch Assessment Results</h3>

          <div className="batch-summary">
            <p>Total transactions: {batchResults.total_transactions}</p>
            <p>Fraud detected: {batchResults.fraud_detected}</p>
            <p>
              Legitimate transactions:{" "}
              {batchResults.legitimate_transactions}
            </p>
            <p>
              Fraud detection rate:{" "}
              {(batchResults.fraud_detection_rate * 100).toFixed(2)}%
            </p>
          </div>

          <h4>Transaction-Level Results</h4>

          <div className="table-container">
            <table>
              <thead>
                <tr>
                  <th>Transaction</th>
                  <th>Prediction</th>
                  <th>Fraud Probability</th>
                  <th>Risk Level</th>
                </tr>
              </thead>

              <tbody>
                {batchResults.results.map((result, index) => (
                  <tr key={index}>
                    <td>{index + 1}</td>
                    <td>
                      {result.prediction === 1 ? "Fraud" : "Legitimate"}
                    </td>
                    <td>
                      {(result.fraud_probability * 100).toFixed(2)}%
                    </td>
                    <td>{result.risk_level}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  </section>
)}

{/* TRANSACTIONS PAGE */}
{activePage === "Transactions" && (
  <section className="card recent-card">
    <div className="card-header">
      <div>
        <h3>Transaction History</h3>
        <p>Search and filter the latest recorded transactions</p>
      </div>

      <span className="alert-count">
        {filteredTransactions.length} records
      </span>
    </div>

    <div className="transaction-filters">
      <input
        type="text"
        placeholder="Search transaction ID, risk or prediction..."
        value={transactionSearch}
        onChange={(event) => setTransactionSearch(event.target.value)}
      />

      <select
        value={riskFilter}
        onChange={(event) => setRiskFilter(event.target.value)}
      >
        <option value="ALL">All risk levels</option>
        <option value="HIGH">High risk</option>
        <option value="MEDIUM">Medium risk</option>
        <option value="LOW">Low risk</option>
      </select>
    </div>

    <TransactionTable
      events={filteredTransactions}
      onSelectTransaction={setSelectedTransaction}
    />
  </section>
)}


        {/* ALERTS PAGE */}
        {activePage === "Alerts" && (
          <section className="card alerts-card">
            <div className="card-header">
              <div>
                <h3>Fraud Alerts</h3>
                <p>Transactions identified as potential fraud</p>
              </div>
              <span className="alert-count">
                {alerts.length} alerts
              </span>
            </div>

            <TransactionTable events={alerts} />
          </section>
        )}


{/* MONITORING PAGE */}
{activePage === "Monitoring" && (
  <div>
    {/* EXISTING MONITORING KPIs */}
    <section className="kpi-grid">
      <div className="card kpi">
        <span>Total Transactions</span>
        <strong>{totalTransactions.toLocaleString()}</strong>
      </div>

      <div className="card kpi">
        <span>Fraud Alerts</span>
        <strong>{fraudAlerts.toLocaleString()}</strong>
      </div>

      <div className="card kpi">
        <span>Legitimate Transactions</span>
        <strong>{legitimateTransactions.toLocaleString()}</strong>
      </div>

      <div className="card kpi">
        <span>Average Fraud Probability</span>
        <strong>{formatProbability(averageProbability)}</strong>
      </div>
    </section>

    {/* EXISTING CHARTS */}
    <section className="dashboard-grid">
      <div className="card chart-card">
        <div className="card-header">
          <div>
            <h3>Hourly Activity</h3>
            <p>Monitoring transaction volumes</p>
          </div>
        </div>

        <div className="chart-container">
          {hourlyData.length > 0 ? (
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={hourlyData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="hourLabel" stroke="#94a3b8" />
                <YAxis stroke="#94a3b8" />
                <Tooltip />
                <Legend />
                <Bar dataKey="total_transactions" name="Transactions" fill="#3b82f6" />
                <Bar dataKey="fraud_alerts" name="Fraud alerts" fill="#f87171" />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="chart-empty">No monitoring data available</div>
          )}
        </div>
      </div>

      <div className="card">
        <div className="card-header">
          <div>
            <h3>Risk Distribution</h3>
            <p>Current monitored risk levels</p>
          </div>
        </div>

        <div className="risk-chart-container">
          {riskData.length > 0 ? (
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={riskData}
                  dataKey="transactions"
                  nameKey="risk_level"
                  cx="50%"
                  cy="45%"
                  outerRadius={85}
                  innerRadius={48}
                >
                  {riskData.map((item) => (
                    <Cell
                      key={item.risk_level}
                      fill={RISK_COLORS[item.risk_level] || "#64748b"}
                    />
                  ))}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          ) : (
            <div className="chart-empty">No risk data available</div>
          )}
        </div>
      </div>
    </section>

    {/* DATA DRIFT MONITORING */}
    <section className="card monitoring-section">
      <div className="card-header">
        <div>
          <h3>Data Drift Monitoring</h3>
          <p>Evidently feature distribution monitoring</p>
        </div>
        <span className="status-dot"></span>
      </div>

      {evidentlyReport?.report_url ? (
        <div>
          <p>Latest Evidently monitoring report is available.</p>
          <a
            href={evidentlyReport.report_url}
            target="_blank"
            rel="noreferrer"
            className="primary-button"
          >
            View Evidently Report
          </a>
        </div>
      ) : (
        <div className="chart-empty">
          {evidentlyReport?.status === "not_found"
            ? "No Evidently report available yet."
            : "Evidently report information unavailable."}
        </div>
      )}
    </section>

    {/* CONCEPT DRIFT MONITORING */}
    <section className="card monitoring-section">
      <div className="card-header">
        <div>
          <h3>Model Performance & Concept Drift</h3>
          <p>Chronological evaluation of fraud detection performance</p>
        </div>

        <span
          className={`monitoring-status ${
            conceptDrift?.drift_alert ? "status-danger" : "status-success"
          }`}
        >
          {conceptDrift?.status === "concept_drift_detected"
            ? "Concept drift detected"
            : conceptDrift?.status === "baseline_established"
              ? "Baseline established"
              : conceptDrift?.status === "no_significant_drift"
                ? "No significant drift"
                : "Insufficient data"}
        </span>
      </div>

      {conceptDrift?.baseline && conceptDrift?.latest_window ? (
        <>
          <div className="performance-comparison">
            {[
              ["Precision", "precision"],
              ["Recall", "recall"],
              ["F1 Score", "f1"],
              ["False Negative Rate", "false_negative_rate"],
            ].map(([label, key]) => (
              <div className="card performance-metric" key={key}>
                <span>{label}</span>
                <div>
                  <small>Baseline</small>
                  <strong>
                    {(conceptDrift.baseline[key] * 100).toFixed(2)}%
                  </strong>
                </div>
                <div>
                  <small>Latest</small>
                  <strong>
                    {(conceptDrift.latest_window[key] * 100).toFixed(2)}%
                  </strong>
                </div>
              </div>
            ))}
          </div>

          <div className="monitoring-details">
            <div>
              <span>Evaluation windows</span>
              <strong>{conceptDrift.total_windows}</strong>
            </div>
            <div>
              <span>Latest fraud cases</span>
              <strong>{conceptDrift.latest_window.fraud_count}</strong>
            </div>
            <div>
              <span>False negatives</span>
              <strong>{conceptDrift.latest_window.false_negatives}</strong>
            </div>
            <div>
              <span>Statistical p-value</span>
              <strong>
                {conceptDrift.latest_window.p_value == null
                  ? "N/A"
                  : conceptDrift.latest_window.p_value.toFixed(6)}
              </strong>
            </div>
          </div>
        </>
      ) : (
        <div className="chart-empty">
          {conceptDrift?.message || "Waiting for sufficient labelled transactions."}
        </div>
      )}
    </section>

    {/* HISTORICAL PERFORMANCE */}
    <section className="card monitoring-section">
      <div className="card-header">
        <div>
          <h3>Historical Performance Trends</h3>
          <p>Persisted model evaluation windows from DuckDB</p>
        </div>
      </div>

      {performanceHistory.length > 0 ? (
        <>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={[...performanceHistory].reverse().map((item) => ({
                  ...item,
                  chart_label: `${String(item.evaluated_at).slice(11, 19)} · W${item.window_number}`,
                }))}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis
                  dataKey="chart_label"
                  stroke="#94a3b8"
                  angle={-35}
                  textAnchor="end"
                  interval={0}
                  height={70}
                  label={{
                    value: "Evaluation time and window",
                    position: "insideBottom",
                    offset: -5,
                  }}
                />
                <YAxis
                  stroke="#94a3b8"
                  domain={[0, 1]}
                  tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
                />
                <Tooltip formatter={(value) => `${(value * 100).toFixed(2)}%`} />
                <Legend />
                <Line type="monotone" dataKey="precision" name="Precision" stroke="#3b82f6" />
                <Line type="monotone" dataKey="recall" name="Recall" stroke="#22c55e" />
                <Line type="monotone" dataKey="f1" name="F1 Score" stroke="#a78bfa" />
                <Line type="monotone" dataKey="false_negative_rate" name="False Negative Rate" stroke="#f87171" />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="table-wrapper">
            <table>
              <thead>
                <tr>
                  <th>Evaluation</th>
                  <th>Window</th>
                  <th>Transactions</th>
                  <th>Precision</th>
                  <th>Recall</th>
                  <th>F1</th>
                  <th>FNR</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {performanceHistory.map((item) => (
                  <tr key={`${item.evaluation_id}-${item.window_number}`}>
                    <td>{new Date(item.evaluated_at).toLocaleString()}</td>
                    <td>{item.window_number}</td>
                    <td>{item.transactions.toLocaleString()}</td>
                    <td>{(item.precision * 100).toFixed(2)}%</td>
                    <td>{(item.recall * 100).toFixed(2)}%</td>
                    <td>{(item.f1 * 100).toFixed(2)}%</td>
                    <td>{(item.false_negative_rate * 100).toFixed(2)}%</td>
                    <td>{item.drift_status}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      ) : (
        <div className="chart-empty">No historical performance evaluations available.</div>
      )}
    </section>
  </div>
)}

        {/* MODEL PAGE */}
        {activePage === "Model" && (
          <section className="card">
            <div className="card-header">
              <div>
                <h3>Model Information</h3>
                <p>Active fraud detection model</p>
              </div>
              <span className="status-dot"></span>
            </div>

            <div className="risk-list">
              <div>
                <span>Model</span>
                <strong>LightGBM</strong>
              </div>
              <div>
                <span>Version</span>
                <strong>1</strong>
              </div>
              <div>
                <span>Registry</span>
                <strong>MLflow</strong>
              </div>
              <div>
                <span>Status</span>
                <strong>Active</strong>
              </div>
              <div>
                <span>Decision Threshold</span>
                <strong>0.05</strong>
              </div>
            </div>
          </section>
        )}

        <footer className="dashboard-footer">
          <span>
            {lastUpdated
              ? `Last updated: ${lastUpdated.toLocaleTimeString()}`
              : "Waiting for monitoring data"}
          </span>

          <span>
            {legitimateTransactions.toLocaleString()} legitimate transactions
          </span>

          <button onClick={fetchDashboardData}>
            Refresh data
          </button>
        </footer>
      </main>
    </div>
  );
}

export default App;

