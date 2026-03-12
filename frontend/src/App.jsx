import { useState, useEffect, useCallback } from "react";
import {
  Shield, Zap, AlertTriangle, Loader2, Link2, Type, TrendingUp,
  AlertCircle, ShieldAlert, ShieldCheck, Globe, Scan, Eye, X, Search,
  Lock, Unlock, Hash, AtSign, LayoutDashboard, MessageSquare,
  Link as LinkIcon, History, Settings, Activity, Phone
} from "lucide-react";
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid
} from "recharts";
import axios from "axios";

const API_URL = "http://localhost:5000";

// Sample data
const SAMPLE_MESSAGES = [
  { label: "Safe Message", text: "Hi! How are you? Want to grab lunch tomorrow?", type: "safe" },
  { label: "Bank Scam", text: "URGENT! Your bank account has been suspended. Click here to verify: bit.ly/xyz123", type: "phishing" },
  { label: "Prize Scam", text: "Congratulations! You've won $10,000! Call NOW at 1-800-555-1234 to claim your prize!", type: "phishing" },
];
const SAMPLE_URLS = [
  { label: "Safe URL", url: "https://www.google.com", type: "safe" },
  { label: "IP Phishing", url: "http://192.168.1.1/sbi/login?user=admin", type: "phishing" },
  { label: "Brand Spoof", url: "http://paypal.secure-login.xyz/verify/account", type: "phishing" },
];
const SAMPLE_FULLSCAN = [
  { label: "Safe Notice", text: "Institution wise Diploma spot admission for Govt/Aided Polytechnic Colleges shall be conducted from 16-11-2021 to 20-11-2021 to the vacant seats available in the institution. For more information please visit www.polyadmission.org", type: "safe" },
  { label: "Bank Phish", text: "URGENT! Your SBI account has been suspended due to unusual activity. Verify immediately to avoid permanent block: http://sbi.login-secure.xyz/verify/account?id=8472", type: "phishing" },
  { label: "Prize Scam", text: "Congratulations! You won a $5000 Amazon gift card! Claim NOW before it expires at http://amazon-prize.tk/claim?winner=true&code=WIN2024", type: "phishing" },
];

const NAV_ITEMS = [
  { key: "dashboard", label: "Dashboard", icon: LayoutDashboard },
  { key: "sms", label: "SMS Scan", icon: MessageSquare },
  { key: "url", label: "URL Scan", icon: LinkIcon },
  { key: "fullscan", label: "Full Scan", icon: Scan },
  { key: "history", label: "History", icon: History },
  { key: "settings", label: "Settings", icon: Settings },
];

const SCAN_TABS = [
  { key: "sms", label: "SMS / Text", icon: Type },
  { key: "url", label: "URL Check", icon: Globe },
  { key: "fullscan", label: "Full Scan", icon: Scan },
];

export default function App() {
  const [activeNav, setActiveNav] = useState("dashboard");
  const [activeTab, setActiveTab] = useState("sms");
  const [message, setMessage] = useState("");
  const [url, setUrl] = useState("");
  const [includeVisual, setIncludeVisual] = useState(false);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [apiOnline, setApiOnline] = useState(false);
  const [showHeatmap, setShowHeatmap] = useState(false);
  const [scanHistory, setScanHistory] = useState([]);
  const [threatHistory, setThreatHistory] = useState([]);

  // Counters
  const smsCount = scanHistory.filter(s => s.scanType === "sms").length;
  const urlCount = scanHistory.filter(s => s.scanType === "url").length;
  const fullCount = scanHistory.filter(s => s.scanType === "fullscan").length;
  const totalScans = scanHistory.length;
  const phishingCount = scanHistory.filter(s => s.isPhishing).length;
  const safeCount = totalScans - phishingCount;
  const safePercent = totalScans > 0 ? Math.round((safeCount / totalScans) * 100) : 100;
  const threatPercent = totalScans > 0 ? Math.round((phishingCount / totalScans) * 100) : 0;

  // Health check
  useEffect(() => {
    const check = async () => {
      try {
        const res = await axios.get(`${API_URL}/api/health`, { timeout: 2000 });
        setApiOnline(res.data.status === "online");
      } catch { setApiOnline(false); }
    };
    check();
    const iv = setInterval(check, 3000);
    return () => clearInterval(iv);
  }, []);

  // Load persisted history from database on startup
  useEffect(() => {
    const loadHistory = async () => {
      try {
        const res = await axios.get(`${API_URL}/api/history?limit=50`);
        if (res.data?.success && res.data.history?.length > 0) {
          setScanHistory(res.data.history);
          // Reconstruct threat chart data from loaded history
          const chartData = res.data.history.slice().reverse().map((h, i) => ({
            name: `#${i + 1}`,
            threat: Math.round(h.score),
            safe: 100 - Math.round(h.score),
          })).slice(-15);
          setThreatHistory(chartData);
        }
      } catch { /* API not ready yet, history will be empty */ }
    };
    loadHistory();
  }, []);

  useEffect(() => {
    setResult(null);
    setError(null);
  }, [activeTab]);

  const addToHistory = useCallback((scanType, input, score, isPhishing, riskLevel) => {
    const entry = {
      id: Date.now(),
      scanType,
      input: input.length > 50 ? input.slice(0, 50) + "…" : input,
      score: parseFloat((score > 1 ? score : score * 100).toFixed(2)),
      isPhishing,
      riskLevel: riskLevel || (isPhishing ? "High" : "Safe"),
      time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    };
    setScanHistory(prev => [entry, ...prev].slice(0, 50));
    // Prepend new entry, keep only last 50
    // [newScan, ...oldScans].slice(0, 50)
    setThreatHistory(prev => {
      const next = [...prev, {
        name: `#${prev.length + 1}`,
        threat: entry.score,
        safe: 100 - entry.score,
      }];
      return next.slice(-15);
    });
  }, []);

  const displayScore = (score) => {
    if (score === undefined || score === null) return 0;
    return score > 1 ? Math.round(score) : Math.round(score * 100);
  };

  const clearScanHistory = async () => {
    try {
      await axios.delete(`${API_URL}/api/history`);
    } catch { /* ignore */ }
    setScanHistory([]);
    setThreatHistory([]);
  };

  const analyzeMessage = async () => {
    if (!message.trim()) return;
    setLoading(true); setError(null); setResult(null);
    try {
      const res = await axios.post(`${API_URL}/api/analyze`, { message });
      setResult({ type: "sms", data: res.data });
      addToHistory("sms", message, res.data.threat_score, res.data.is_phishing);
    } catch (err) {
      setError(err.response?.data?.error || "Failed to connect to API.");
    } finally { setLoading(false); }
  };

  const analyzeUrl = async () => {
    if (!url.trim()) return;
    setLoading(true); setError(null); setResult(null);
    try {
      const res = await axios.post(`${API_URL}/api/analyze-url`, { url });
      setResult({ type: "url", data: res.data });
      addToHistory("url", url, res.data.threat_score, res.data.is_phishing, res.data.risk_level);
    } catch (err) {
      setError(err.response?.data?.error || "Failed to analyze URL.");
    } finally { setLoading(false); }
  };

  const fullScan = async () => {
    if (!message.trim()) return;
    setLoading(true); setError(null); setResult(null);
    try {
      const res = await axios.post(`${API_URL}/api/full-scan`, {
        message, include_visual: includeVisual,
      });
      setResult({ type: "fullscan", data: res.data });
      addToHistory("fullscan", message, res.data.combined_threat_score,
        res.data.combined_threat_score >= 0.5, res.data.risk_level);
    } catch (err) {
      setError(err.response?.data?.error || "Full scan failed.");
    } finally { setLoading(false); }
  };

  const handleAnalyze = () => {
    if (activeTab === "sms") analyzeMessage();
    else if (activeTab === "url") analyzeUrl();
    else fullScan();
  };

  const loadSample = (sample) => {
    if (sample.text) setMessage(sample.text);// Call SMS API
    if (sample.url) setUrl(sample.url);// Call URL API
    setResult(null); setError(null);  // Call Full Scan API
  };

  const canAnalyze = () => {
    if (!apiOnline) return false;
    if (activeTab === "url") return !!url.trim();
    return !!message.trim();
  };

  const getThreatColor = (score) => {
    const s = score > 1 ? score : score * 100;
    if (s < 30) return { gradient: "linear-gradient(90deg, #34d399, #10b981)", text: "var(--accent-green)" };
    if (s < 60) return { gradient: "linear-gradient(90deg, #fbbf24, #f59e0b)", text: "var(--accent-amber)" };
    if (s < 85) return { gradient: "linear-gradient(90deg, #fb923c, #f87171)", text: "var(--accent-red)" };
    return { gradient: "linear-gradient(90deg, #f87171, #ef4444)", text: "var(--accent-red)" };
  };

  // Navigate to scan tab
  const handleNavClick = (key) => {
    setActiveNav(key);
    if (key === "sms" || key === "url" || key === "fullscan") {
      setActiveTab(key);
    }
  };

  return (
    <div className="dashboard">
      {/* ============ SIDEBAR ============ */}
      <aside className="sidebar animate-slide-left">
        <div className="sidebar-brand">
          <div className="sidebar-brand-icon">
            <Shield size={22} color="white" />
          </div>
          <h2>PhishGuard</h2>
        </div>

        <nav className="sidebar-nav">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            return (
              <div key={item.key}
                className={`nav-item ${activeNav === item.key ? "active" : ""}`}
                onClick={() => handleNavClick(item.key)}>
                <Icon size={18} />
                <span>{item.label}</span>
              </div>
            );
          })}
        </nav>

        <div className="sidebar-status">
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
            <span style={{
              width: 8, height: 8, borderRadius: "50%",
              background: apiOnline ? "var(--accent-green)" : "var(--accent-red)",
              display: "inline-block",
            }}
              className={apiOnline ? "animate-pulse-online" : ""}
            />
            <span style={{ fontSize: 12, fontWeight: 600, color: apiOnline ? "var(--accent-green)" : "var(--accent-red)" }}>
              {apiOnline ? "AI Engine Online" : "AI Offline"}
            </span>
          </div>
          <span style={{ fontSize: 11, color: "var(--text-muted)" }}>
            ML Models • SMS + URL + Visual
          </span>
        </div>
      </aside>

      {/* ============ MAIN CONTENT ============ */}
      <main className="main-content">
        {/* Header */}
        <div className="main-header animate-fade-in">
          <div>
            <h1>
              {activeNav === "dashboard" && "Threat Dashboard"}
              {activeNav === "sms" && "SMS / Text Scan"}
              {activeNav === "url" && "URL Scan"}
              {activeNav === "fullscan" && "Full Scan"}
              {activeNav === "history" && "Scan History"}
              {activeNav === "settings" && "Settings"}
            </h1>
            <p style={{ fontSize: 13, color: "var(--text-secondary)", marginTop: 2 }}>
              {activeNav === "dashboard" && "Real-time phishing detection & analysis"}
              {activeNav === "sms" && "Analyze SMS & text messages for phishing threats"}
              {activeNav === "url" && "Check URLs for phishing indicators"}
              {activeNav === "fullscan" && "Combined SMS + URL + Visual analysis"}
              {activeNav === "history" && "Browse all previous scan results"}
              {activeNav === "settings" && "System configuration & information"}
            </p>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <span style={{ fontSize: 12, color: "var(--text-muted)" }}>
              {new Date().toLocaleDateString("en-US", { weekday: "long", month: "short", day: "numeric" })}
            </span>
          </div>
        </div>

        {/* ========== DASHBOARD VIEW ========== */}
        {activeNav === "dashboard" && (
          <>
            {/* Stat Cards */}
            <div className="stat-cards">
              <StatCard label="SMS Scans" value={smsCount} badge={`${smsCount} total`} color="var(--accent-cyan)" delay="delay-1" />
              <StatCard label="URL Scans" value={urlCount} badge={`${urlCount} total`} color="var(--accent-purple)" delay="delay-2" />
              <StatCard label="Full Scans" value={fullCount} badge={`${fullCount} total`} color="var(--accent-pink)" delay="delay-3" />
            </div>

            {/* Chart */}
            <div className="chart-section animate-fade-in-up delay-3">
              <div className="chart-header">
                <h3 style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <Activity size={16} color="var(--accent-purple)" /> Threat History
                </h3>
                <div className="chart-tabs">
                  <span className="chart-tab active">Recent</span>
                  <span className="chart-tab">All</span>
                </div>
              </div>
              {threatHistory.length > 0 ? (
                <ResponsiveContainer width="100%" height={180}>
                  <AreaChart data={threatHistory}>
                    <defs>
                      <linearGradient id="colorThreat" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#7c64ff" stopOpacity={0.35} />
                        <stop offset="100%" stopColor="#7c64ff" stopOpacity={0} />
                      </linearGradient>
                      <linearGradient id="colorSafe" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#3ecfff" stopOpacity={0.2} />
                        <stop offset="100%" stopColor="#3ecfff" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                    <XAxis dataKey="name" stroke="rgba(255,255,255,0.15)" tick={{ fontSize: 11, fill: '#555d72' }} />
                    <YAxis domain={[0, 100]} stroke="rgba(255,255,255,0.08)" tick={{ fontSize: 11, fill: '#555d72' }} />
                    <Tooltip
                      contentStyle={{
                        background: "#1a2030", border: "1px solid rgba(255,255,255,0.1)",
                        borderRadius: 10, fontSize: 12, color: "#e8eaed"
                      }}
                    />
                    <Area type="monotone" dataKey="threat" stroke="#7c64ff" strokeWidth={2.5}
                      fill="url(#colorThreat)" name="Threat %" />
                    <Area type="monotone" dataKey="safe" stroke="#3ecfff" strokeWidth={1.5}
                      fill="url(#colorSafe)" name="Safe %" />
                  </AreaChart>
                </ResponsiveContainer>
              ) : (
                <div style={{ height: 180, display: "flex", alignItems: "center", justifyContent: "center", color: "var(--text-muted)", fontSize: 13 }}>
                  <Activity size={16} style={{ marginRight: 8, opacity: 0.5 }} /> Run scans to see threat trends
                </div>
              )}
            </div>

            {/* Quick Scan Panel on Dashboard */}
            <div className="input-panel animate-fade-in-up delay-4">
              <div className="input-tabs">
                {SCAN_TABS.map(tab => {
                  const Icon = tab.icon;
                  return (
                    <button key={tab.key}
                      className={`input-tab ${activeTab === tab.key ? "active" : ""}`}
                      onClick={() => setActiveTab(tab.key)}>
                      <Icon size={15} /> {tab.label}
                    </button>
                  );
                })}
              </div>

              {activeTab === "sms" && (
                <textarea className="input-field" rows={4} value={message}
                  onChange={e => setMessage(e.target.value)}
                  placeholder="Paste a suspicious SMS or email message…" />
              )}
              {activeTab === "url" && (
                <input type="text" className="input-field" value={url}
                  onChange={e => setUrl(e.target.value)}
                  placeholder="Enter a suspicious URL (e.g. http://suspicious-site.com/login)" />
              )}
              {activeTab === "fullscan" && (
                <>
                  <textarea className="input-field" rows={5} value={message}
                    onChange={e => setMessage(e.target.value)}
                    placeholder={"Paste the full message here — URLs auto-detected\n\nExample: URGENT! Your SBI account has been suspended. Verify at http://sbi.login-secure.xyz/verify"} />
                  <label className="checkbox-wrapper">
                    <input type="checkbox" checked={includeVisual} onChange={e => setIncludeVisual(e.target.checked)} />
                    <span style={{ fontSize: 12.5, color: "var(--text-secondary)" }}>
                      <Eye size={14} style={{ display: "inline", marginRight: 4, verticalAlign: "middle" }} />
                      Include Visual Spoofing Analysis (slower)
                    </span>
                  </label>
                </>
              )}

              <button className="btn-scan" onClick={handleAnalyze} disabled={loading || !canAnalyze()}>
                {loading ? (
                  <><Loader2 size={18} className="animate-spin" /> Analyzing…</>
                ) : (
                  <><Zap size={18} /> {activeTab === "fullscan" ? "Launch Full Scan" : "Initiate Threat Scan"}</>
                )}
              </button>

              {/* Samples */}
              <div className="samples-row">
                {(activeTab === "sms" ? SAMPLE_MESSAGES : activeTab === "url" ? SAMPLE_URLS : SAMPLE_FULLSCAN)
                  .map((s, i) => (
                    <button key={i} className={`sample-btn ${s.type === "safe" ? "safe" : "danger"}`}
                      onClick={() => loadSample(s)}>{s.label}</button>
                  ))}
              </div>
            </div>

            {/* Error */}
            {error && (
              <div style={{
                display: "flex", alignItems: "center", gap: 12, padding: "14px 18px",
                background: "var(--glow-red)", border: "1px solid rgba(248,113,113,0.2)",
                borderRadius: "var(--radius-md)", fontSize: 13
              }}>
                <AlertCircle size={18} color="var(--accent-red)" />
                <span style={{ color: "var(--accent-red)" }}>{error}</span>
              </div>
            )}

            {/* Recent Scans on Dashboard - compact */}
            {scanHistory.length > 0 && (
              <div className="scan-history animate-fade-in-up delay-5">
                <div className="scan-history-header">
                  <h3 style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    <History size={16} color="var(--accent-cyan)" /> Recent Scans
                  </h3>
                  <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                    <span style={{ fontSize: 12, color: "var(--text-muted)" }}>
                      {scanHistory.length} scans
                    </span>
                    <button onClick={() => handleNavClick("history")} style={{
                      padding: "4px 10px", borderRadius: 6, fontSize: 11, fontWeight: 600,
                      background: "rgba(124,100,255,0.1)", border: "1px solid rgba(124,100,255,0.2)",
                      color: "var(--accent-purple)", cursor: "pointer"
                    }}>View All</button>
                  </div>
                </div>
                <div className="scan-row scan-row-head">
                  <span>Time</span><span>Input</span><span>Type</span><span>Score</span><span>Status</span>
                </div>
                {scanHistory.slice(0, 6).map(s => (
                  <div key={s.id} className="scan-row">
                    <span style={{ color: "var(--text-muted)", fontSize: 12 }}>{s.time}</span>
                    <span style={{ color: "var(--text-secondary)", fontSize: 12.5, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{s.input}</span>
                    <span>
                      <span className={`badge ${s.scanType === "sms" ? "badge-purple" : s.scanType === "url" ? "badge-warning" : "badge-safe"}`}>
                        {s.scanType.toUpperCase()}
                      </span>
                    </span>
                    <span style={{
                      fontFamily: "'Outfit', sans-serif", fontWeight: 700,
                      color: s.score < 30 ? "var(--accent-green)" : s.score < 60 ? "var(--accent-amber)" : "var(--accent-red)"
                    }}>
                      {s.score}
                    </span>
                    <span>
                      <span className={`badge ${s.isPhishing ? "badge-danger" : "badge-safe"}`}>
                        {s.isPhishing ? "⚠ Phishing" : "✓ Safe"}
                      </span>
                    </span>
                  </div>
                ))}
              </div>
            )}
          </>
        )}

        {/* ========== SMS / URL / FULLSCAN VIEW ========== */}
        {(activeNav === "sms" || activeNav === "url" || activeNav === "fullscan") && (
          <>
            <div className="input-panel animate-fade-in-up">
              <div className="input-tabs">
                {SCAN_TABS.map(tab => {
                  const Icon = tab.icon;
                  return (
                    <button key={tab.key}
                      className={`input-tab ${activeTab === tab.key ? "active" : ""}`}
                      onClick={() => { setActiveTab(tab.key); setActiveNav(tab.key); }}>
                      <Icon size={15} /> {tab.label}
                    </button>
                  );
                })}
              </div>

              {activeTab === "sms" && (
                <textarea className="input-field" rows={6} value={message}
                  onChange={e => setMessage(e.target.value)}
                  placeholder="Paste a suspicious SMS or email message…" />
              )}
              {activeTab === "url" && (
                <input type="text" className="input-field" value={url}
                  onChange={e => setUrl(e.target.value)}
                  placeholder="Enter a suspicious URL (e.g. http://suspicious-site.com/login)" />
              )}
              {activeTab === "fullscan" && (
                <>
                  <textarea className="input-field" rows={6} value={message}
                    onChange={e => setMessage(e.target.value)}
                    placeholder={"Paste the full message here — URLs auto-detected\n\nExample: URGENT! Your SBI account has been suspended. Verify at http://sbi.login-secure.xyz/verify"} />
                  <label className="checkbox-wrapper">
                    <input type="checkbox" checked={includeVisual} onChange={e => setIncludeVisual(e.target.checked)} />
                    <span style={{ fontSize: 12.5, color: "var(--text-secondary)" }}>
                      <Eye size={14} style={{ display: "inline", marginRight: 4, verticalAlign: "middle" }} />
                      Include Visual Spoofing Analysis (slower)
                    </span>
                  </label>
                </>
              )}

              <button className="btn-scan" onClick={handleAnalyze} disabled={loading || !canAnalyze()}>
                {loading ? (
                  <><Loader2 size={18} className="animate-spin" /> Analyzing…</>
                ) : (
                  <><Zap size={18} /> {activeTab === "fullscan" ? "Launch Full Scan" : "Initiate Threat Scan"}</>
                )}
              </button>

              {/* Samples */}
              <div className="samples-row">
                {(activeTab === "sms" ? SAMPLE_MESSAGES : activeTab === "url" ? SAMPLE_URLS : SAMPLE_FULLSCAN)
                  .map((s, i) => (
                    <button key={i} className={`sample-btn ${s.type === "safe" ? "safe" : "danger"}`}
                      onClick={() => loadSample(s)}>{s.label}</button>
                  ))}
              </div>
            </div>

            {/* Error */}
            {error && (
              <div style={{
                display: "flex", alignItems: "center", gap: 12, padding: "14px 18px",
                background: "var(--glow-red)", border: "1px solid rgba(248,113,113,0.2)",
                borderRadius: "var(--radius-md)", fontSize: 13
              }}>
                <AlertCircle size={18} color="var(--accent-red)" />
                <span style={{ color: "var(--accent-red)" }}>{error}</span>
              </div>
            )}
          </>
        )}

        {/* ========== HISTORY VIEW ========== */}
        {activeNav === "history" && (
          <div className="scan-history animate-fade-in-up" style={{ marginTop: 0 }}>
            <div className="scan-history-header">
              <h3 style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <History size={16} color="var(--accent-cyan)" /> All Scan History
              </h3>
              <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                <span style={{ fontSize: 12, color: "var(--text-muted)" }}>
                  {scanHistory.length} scans
                </span>
                <button onClick={clearScanHistory} style={{
                  padding: "4px 10px", borderRadius: 6, fontSize: 11, fontWeight: 600,
                  background: "rgba(248,113,113,0.1)", border: "1px solid rgba(248,113,113,0.2)",
                  color: "var(--accent-red)", cursor: "pointer"
                }}>Clear All</button>
              </div>
            </div>
            {scanHistory.length > 0 ? (
              <>
                <div className="scan-row scan-row-head">
                  <span>Time</span><span>Input</span><span>Type</span><span>Score</span><span>Status</span>
                </div>
                {scanHistory.map(s => (
                  <div key={s.id} className="scan-row">
                    <span style={{ color: "var(--text-muted)", fontSize: 12 }}>{s.time}</span>
                    <span style={{ color: "var(--text-secondary)", fontSize: 12.5, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{s.input}</span>
                    <span>
                      <span className={`badge ${s.scanType === "sms" ? "badge-purple" : s.scanType === "url" ? "badge-warning" : "badge-safe"}`}>
                        {s.scanType.toUpperCase()}
                      </span>
                    </span>
                    <span style={{
                      fontFamily: "'Outfit', sans-serif", fontWeight: 700,
                      color: s.score < 30 ? "var(--accent-green)" : s.score < 60 ? "var(--accent-amber)" : "var(--accent-red)"
                    }}>
                      {s.score}
                    </span>
                    <span>
                      <span className={`badge ${s.isPhishing ? "badge-danger" : "badge-safe"}`}>
                        {s.isPhishing ? "⚠ Phishing" : "✓ Safe"}
                      </span>
                    </span>
                  </div>
                ))}
              </>
            ) : (
              <div style={{ textAlign: "center", padding: "40px 0", color: "var(--text-muted)", fontSize: 13 }}>
                <History size={24} style={{ opacity: 0.3, marginBottom: 8 }} />
                <p>No scan history yet. Run a scan to see results here.</p>
              </div>
            )}
          </div>
        )}

        {/* ========== SETTINGS VIEW ========== */}
        {activeNav === "settings" && (
          <div className="animate-fade-in-up">
            {/* System Info */}
            <div style={{ padding: 20, background: "var(--card)", border: "1px solid var(--border)", borderRadius: "var(--radius-lg)", marginBottom: 16 }}>
              <h3 style={{ fontSize: 14, fontWeight: 700, marginBottom: 16, display: "flex", alignItems: "center", gap: 8 }}>
                <Shield size={16} color="var(--accent-purple)" /> System Information
              </h3>
              <div className="stat-row"><span className="stat-row-label">Application</span><span className="stat-row-value">PhishGuard AI v2.0</span></div>
              <div className="stat-row"><span className="stat-row-label">AI Engine</span><span className="stat-row-value" style={{ color: apiOnline ? "var(--accent-green)" : "var(--accent-red)" }}>{apiOnline ? "Online" : "Offline"}</span></div>
              <div className="stat-row"><span className="stat-row-label">SMS Model</span><span className="stat-row-value">Random Forest + TF-IDF (500 features)</span></div>
              <div className="stat-row"><span className="stat-row-label">URL Model</span><span className="stat-row-value">Random Forest (30 lexical features)</span></div>
              <div className="stat-row"><span className="stat-row-label">Visual Detection</span><span className="stat-row-value">pHash + SSIM (47 trusted sites)</span></div>
              <div className="stat-row"><span className="stat-row-label">Phishing Threshold</span><span className="stat-row-value">≥ 50%</span></div>
            </div>

            {/* Score Weights */}
            <div style={{ padding: 20, background: "var(--card)", border: "1px solid var(--border)", borderRadius: "var(--radius-lg)", marginBottom: 16 }}>
              <h3 style={{ fontSize: 14, fontWeight: 700, marginBottom: 16, display: "flex", alignItems: "center", gap: 8 }}>
                <Scan size={16} color="var(--accent-cyan)" /> Full Scan Weights
              </h3>
              <ScoreBar label="SMS Analysis" weight="40%" score={0.4} performed={true} />
              <ScoreBar label="URL Analysis" weight="45%" score={0.45} performed={true} />
              <ScoreBar label="Visual Analysis" weight="15%" score={0.15} performed={true} />
              <p style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 12 }}>
                Weights are re-normalized when fewer channels are active.
              </p>
            </div>

            {/* Risk Levels */}
            <div style={{ padding: 20, background: "var(--card)", border: "1px solid var(--border)", borderRadius: "var(--radius-lg)" }}>
              <h3 style={{ fontSize: 14, fontWeight: 700, marginBottom: 16, display: "flex", alignItems: "center", gap: 8 }}>
                <AlertTriangle size={16} color="var(--accent-amber)" /> Risk Level Thresholds
              </h3>
              <div className="stat-row"><span className="stat-row-label">LOW</span><span className="stat-row-value" style={{ color: "var(--accent-green)" }}>0 – 29%</span></div>
              <div className="stat-row"><span className="stat-row-label">MEDIUM</span><span className="stat-row-value" style={{ color: "var(--accent-amber)" }}>30 – 59%</span></div>
              <div className="stat-row"><span className="stat-row-label">HIGH</span><span className="stat-row-value" style={{ color: "#fb923c" }}>60 – 84%</span></div>
              <div className="stat-row"><span className="stat-row-label">CRITICAL</span><span className="stat-row-value" style={{ color: "var(--accent-red)" }}>85 – 100%</span></div>
            </div>
          </div>
        )}
      </main>

      {/* ============ RIGHT PANEL ============ */}
      <aside className="right-panel">
        {/* Overall Stats */}
        <div style={{ textAlign: "center", paddingBottom: 14, borderBottom: "1px solid var(--border)" }}
          className="animate-fade-in">
          <p style={{ fontSize: 11, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: 1, marginBottom: 6 }}>
            Total Scans
          </p>
          <p style={{ fontFamily: "'Outfit', sans-serif", fontSize: 42, fontWeight: 800, lineHeight: 1 }}
            className="text-gradient">{totalScans}</p>
        </div>

        {/* Safety Donuts */}
        <div className="right-section animate-fade-in delay-1">
          <h4>Safety Rate</h4>
          <div className="donut-container">
            <div>
              <div className="donut-value" style={{ color: "var(--accent-green)" }}>{safePercent}%</div>
              <div className="donut-label">Safe</div>
            </div>
            <div style={{ width: 1, height: 36, background: "var(--border)" }} />
            <div>
              <div className="donut-value" style={{ color: "var(--accent-red)" }}>{threatPercent}%</div>
              <div className="donut-label">Threats</div>
            </div>
          </div>
        </div>

        {/* Detailed Stats */}
        <div className="right-section animate-fade-in delay-2">
          <h4>Breakdown</h4>
          <div className="stat-row">
            <span className="stat-row-label">SMS Scans</span>
            <span className="stat-row-value" style={{ color: "var(--accent-cyan)" }}>{smsCount}</span>
          </div>
          <div className="stat-row">
            <span className="stat-row-label">URL Scans</span>
            <span className="stat-row-value" style={{ color: "var(--accent-purple)" }}>{urlCount}</span>
          </div>
          <div className="stat-row">
            <span className="stat-row-label">Full Scans</span>
            <span className="stat-row-value" style={{ color: "var(--accent-pink)" }}>{fullCount}</span>
          </div>
          <div style={{ borderTop: "1px solid var(--border)", marginTop: 10, paddingTop: 10 }}>
            <div className="stat-row">
              <span className="stat-row-label">Phishing Found</span>
              <span className="stat-row-value" style={{ color: "var(--accent-red)" }}>{phishingCount}</span>
            </div>
            <div className="stat-row">
              <span className="stat-row-label">Safe Messages</span>
              <span className="stat-row-value" style={{ color: "var(--accent-green)" }}>{safeCount}</span>
            </div>
          </div>
        </div>

        {/* Threats detected */}
        <div className="right-section animate-fade-in delay-3">
          <h4>Detections</h4>
          <div style={{ textAlign: "center", padding: "10px 0" }}>
            <p style={{ fontFamily: "'Outfit', sans-serif", fontSize: 34, fontWeight: 700, color: phishingCount > 0 ? "var(--accent-red)" : "var(--accent-green)" }}>
              {phishingCount}
            </p>
            <p style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 4 }}>
              {phishingCount > 0 ? "Threats Detected" : "No Threats Yet"}
            </p>
          </div>
        </div>

        {/* Footer */}
        <div style={{ marginTop: "auto", textAlign: "center", fontSize: 11, color: "var(--text-muted)", padding: "12px 0" }}>
          PhishGuard AI v2.0<br />ML-Powered Detection
        </div>
      </aside>

      {/* ============ RESULT MODAL ============ */}
      {result && (
        <div className="result-overlay" onClick={() => setResult(null)}>
          <div className="result-modal" onClick={e => e.stopPropagation()} style={{ position: "relative" }}>
            <button className="result-close" onClick={() => setResult(null)}><X size={16} /></button>

            {result.type === "sms" && <SmsResult data={result.data} displayScore={displayScore} getThreatColor={getThreatColor} />}
            {result.type === "url" && result.data?.success && <UrlResult data={result.data} displayScore={displayScore} getThreatColor={getThreatColor} />}
            {result.type === "fullscan" && result.data?.success && (
              <FullScanResult data={result.data} displayScore={displayScore} getThreatColor={getThreatColor}
                setShowHeatmap={setShowHeatmap} apiUrl={API_URL} />
            )}
          </div>
        </div>
      )}

      {/* Heatmap Modal */}
      {showHeatmap && (
        <div className="heatmap-overlay" onClick={() => setShowHeatmap(false)}>
          <div className="heatmap-modal" onClick={e => e.stopPropagation()}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
              <h3 style={{ fontSize: 16, fontWeight: 700 }}>Visual Difference Heatmap</h3>
              <button onClick={() => setShowHeatmap(false)} style={{ background: "none", border: "none", color: "var(--text-secondary)", cursor: "pointer" }}>
                <X size={18} />
              </button>
            </div>
            <img src={`${API_URL}/api/heatmap?t=${Date.now()}`} alt="Difference Heatmap" />
            <p style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 12 }}>
              Red/warm areas indicate visual differences between the suspect and trusted site.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

/* ======================== Stat Card ======================== */
function StatCard({ label, value, badge, color, delay }) {
  return (
    <div className={`stat-card animate-fade-in-up ${delay}`}>
      <span className="stat-card-label">{label}</span>
      <span className="stat-card-value" style={{ color }}>{value}</span>
      <span className="stat-card-sub">{badge}</span>
      <span className="stat-card-badge" style={{ background: `${color}15`, color }}>{badge}</span>
    </div>
  );
}

/* ======================== SMS Result ======================== */
function SmsResult({ data, displayScore, getThreatColor }) {
  const score = displayScore(data.threat_score);
  const colors = getThreatColor(score);
  return (
    <div>
      <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 4 }}>
        {data.is_phishing
          ? <ShieldAlert size={28} color="var(--accent-red)" />
          : <ShieldCheck size={28} color="var(--accent-green)" />}
        <h2 style={{ color: data.is_phishing ? "var(--accent-red)" : "var(--accent-green)" }}>
          {data.is_phishing ? "PHISHING DETECTED" : "MESSAGE SAFE"}
        </h2>
      </div>
      <p style={{ fontSize: 13, color: "var(--text-secondary)", marginBottom: 16 }}>
        Confidence: {(data.confidence * 100).toFixed(1)}%
      </p>

      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", marginBottom: 8 }}>
        <span style={{ fontSize: 12, color: "var(--text-muted)" }}>Threat Score</span>
        <span style={{ fontFamily: "'Outfit'", fontSize: 36, fontWeight: 800, color: colors.text }}>{score}</span>
      </div>
      <div className="result-gauge">
        <div className="result-gauge-fill" style={{ width: `${score}%`, background: colors.gradient }} />
      </div>

      {data.is_phishing && (
        <div style={{
          display: "flex", alignItems: "center", gap: 10, padding: "12px 16px", borderRadius: "var(--radius-sm)",
          background: "var(--glow-red)", border: "1px solid rgba(248,113,113,0.15)", marginTop: 14, fontSize: 13
        }}>
          <AlertTriangle size={18} color="var(--accent-red)" />
          <span style={{ color: "var(--accent-red)", fontWeight: 500 }}>Do NOT click links or share personal information.</span>
        </div>
      )}

      {data.features && (
        <div className="result-features" style={{ marginTop: 18 }}>
          <FeatureCard icon={<AlertTriangle size={15} />} label="Urgency" value={data.features.urgency_keywords} danger={data.features.urgency_keywords > 0} />
          <FeatureCard icon={<TrendingUp size={15} />} label="Financial" value={data.features.financial_keywords} danger={data.features.financial_keywords > 0} />
          <FeatureCard icon={<Zap size={15} />} label="Action" value={data.features.action_keywords} danger={data.features.action_keywords > 0} />
          <FeatureCard icon={<Link2 size={15} />} label="URLs" value={data.features.has_url ? "Yes" : "No"} danger={data.features.has_url} />
          <FeatureCard icon={<Phone size={15} />} label="Phone" value={data.features.has_phone ? "Yes" : "No"} danger={data.features.has_phone} />
          <FeatureCard icon={<Type size={15} />} label="CAPS" value={data.features.excessive_caps ? "Yes" : "No"} danger={data.features.excessive_caps} />
        </div>
      )}
    </div>
  );
}

/* ======================== URL Result ======================== */
function UrlResult({ data, displayScore, getThreatColor }) {
  const score = displayScore(data.threat_score);
  const colors = getThreatColor(score);
  return (
    <div>
      <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 4 }}>
        <Globe size={28} color={data.is_phishing ? "var(--accent-red)" : "var(--accent-green)"} />
        <h2 style={{ color: data.is_phishing ? "var(--accent-red)" : "var(--accent-green)" }}>
          {data.is_phishing ? "PHISHING URL" : "URL SAFE"}
        </h2>
      </div>
      <p style={{ fontSize: 12, color: "var(--text-muted)", marginBottom: 16, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
        {data.url}
      </p>

      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", marginBottom: 8 }}>
        <div>
          <span style={{ fontSize: 12, color: "var(--text-muted)" }}>Threat Score</span>
          <span className={`badge ${data.is_phishing ? "badge-danger" : "badge-safe"}`} style={{ marginLeft: 10 }}>
            {data.risk_level}
          </span>
        </div>
        <span style={{ fontFamily: "'Outfit'", fontSize: 36, fontWeight: 800, color: colors.text }}>{score}</span>
      </div>
      <div className="result-gauge">
        <div className="result-gauge-fill" style={{ width: `${score}%`, background: colors.gradient }} />
      </div>

      {/* Quick Indicators */}
      <div className="quick-indicators">
        <QI label="IP Address" active={data.features?.has_ip_address} icon={<Hash size={13} />} />
        <QI label="Shortened" active={data.features?.is_shortened} icon={<Link2 size={13} />} />
        <QI label="Suspicious" active={data.features?.has_suspicious_words} icon={<AlertTriangle size={13} />} />
        <QI label="HTTPS" active={data.features?.has_https} icon={data.features?.has_https ? <Lock size={13} /> : <Unlock size={13} />} good />
        <QI label="Brand Spoof" active={data.features?.has_brand_in_subdomain} icon={<AtSign size={13} />} />
      </div>

      {/* Top Risk Features */}
      {data.top_risk_features?.length > 0 && (
        <div style={{ marginTop: 14 }}>
          <p style={{ fontSize: 12, fontWeight: 600, color: "var(--text-secondary)", marginBottom: 10, display: "flex", alignItems: "center", gap: 6 }}>
            <TrendingUp size={14} color="var(--accent-cyan)" /> Top Risk Features
          </p>
          {data.top_risk_features.slice(0, 5).map((feat, i) => {
            const val = data.features?.[feat] ?? 0;
            const pct = Math.min(100, Math.max(8, typeof val === "number" ? (val / Math.max(val, 1)) * 100 : 50));
            return (
              <div key={i} className="result-risk-bar">
                <span className="result-risk-label">{feat.replace(/_/g, " ")}</span>
                <div className="result-risk-track">
                  <div className="result-risk-fill" style={{ width: `${pct}%`, background: "linear-gradient(90deg, var(--accent-red), #fb923c)" }} />
                </div>
                <span style={{ fontSize: 11, color: "var(--text-secondary)", width: 40, textAlign: "right" }}>
                  {typeof val === "number" ? (Number.isInteger(val) ? val : val.toFixed(2)) : val}
                </span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

/* ======================== Full Scan Result ======================== */
function FullScanResult({ data, displayScore, getThreatColor, setShowHeatmap, apiUrl }) {
  const score = displayScore(data.combined_threat_score);
  const colors = getThreatColor(score);
  const smsScore = data.sms_analysis?.threat_score ?? null;
  const urlScore = data.url_analysis?.threat_score ?? null;
  const visualScore = data.visual_analysis?.visual_threat_score ?? null;

  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 16 }}>
        <div>
          <h2 style={{ color: colors.text }}>COMBINED THREAT ANALYSIS</h2>
          <p style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 4 }}>
            Channels: {data.analyses_performed?.join(", ") || "none"}
          </p>
        </div>
        <div style={{ textAlign: "right" }}>
          <span style={{ fontFamily: "'Outfit'", fontSize: 44, fontWeight: 800, color: colors.text }}>{score}</span>
          <br />
          <span className={`badge ${score >= 60 ? "badge-danger" : score >= 30 ? "badge-warning" : "badge-safe"}`}>
            {data.risk_level}
          </span>
        </div>
      </div>

      <div className="result-gauge">
        <div className="result-gauge-fill" style={{ width: `${score}%`, background: colors.gradient }} />
      </div>

      {/* Score Breakdown */}
      <div className="score-breakdown" style={{ marginTop: 20 }}>
        <p style={{ fontSize: 12, fontWeight: 600, color: "var(--text-secondary)", marginBottom: 6 }}>Score Breakdown</p>
        <ScoreBar label="SMS Score" weight="40%" score={smsScore} performed={data.analyses_performed?.includes("sms")} />
        <ScoreBar label="URL Score" weight="45%" score={urlScore} performed={data.analyses_performed?.includes("url")} />
        <ScoreBar label="Visual Score" weight="15%" score={visualScore} performed={data.analyses_performed?.includes("visual")} />
      </div>

      {/* Extracted URLs */}
      {data.url_analysis?.urls_checked?.length > 0 && (
        <div style={{ marginTop: 18, padding: 16, background: "rgba(255,255,255,0.02)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)" }}>
          <p style={{ fontSize: 13, fontWeight: 600, marginBottom: 8, display: "flex", alignItems: "center", gap: 6 }}>
            <Link2 size={14} color="var(--accent-cyan)" /> Auto-Extracted URLs ({data.url_analysis.urls_checked.length})
          </p>
          {data.url_analysis.urls_checked.map((u, i) => (
            <p key={i} style={{ fontSize: 11.5, color: "var(--text-muted)", padding: "2px 0", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>• {u}</p>
          ))}
          {data.url_analysis.url && (
            <p style={{ fontSize: 11.5, color: "var(--accent-amber)", marginTop: 6 }}>
              ⚠ Highest risk: <span style={{ color: "white", fontWeight: 600 }}>{data.url_analysis.url}</span>
            </p>
          )}
        </div>
      )}

      {/* Visual Spoofing */}
      {data.analyses_performed?.includes("visual") && data.visual_analysis && (
        <div style={{ marginTop: 18, padding: 16, background: "rgba(255,255,255,0.02)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)" }}>
          <p style={{ fontSize: 13, fontWeight: 600, marginBottom: 10, display: "flex", alignItems: "center", gap: 6 }}>
            <Eye size={14} color="var(--accent-purple)" /> Visual Spoofing Check
          </p>
          {data.visual_analysis.error ? (
            <p style={{ fontSize: 12, color: "var(--accent-amber)" }}>⚠ {data.visual_analysis.error}</p>
          ) : (
            <>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 10 }}>
                <div>
                  {data.visual_analysis.best_match_site && (
                    <p style={{ fontSize: 12.5, color: "var(--text-secondary)" }}>
                      Closest Match: <span style={{ fontWeight: 700, color: "white", textTransform: "uppercase" }}>{data.visual_analysis.best_match_site}</span>
                      <span style={{ color: "var(--text-muted)", marginLeft: 4 }}>({data.visual_analysis.best_match_url})</span>
                    </p>
                  )}
                  <p style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 4 }}>
                    pHash Distance: <span style={{ fontWeight: 700, color: "white" }}>{data.visual_analysis.phash_distance ?? "—"}</span>
                    <span style={{ marginLeft: 12 }}>
                      SSIM: <span style={{ fontWeight: 700, color: "white" }}>
                        {data.visual_analysis.ssim_score > 0
                          ? `${(data.visual_analysis.ssim_score * 100).toFixed(1)}%`
                          : "Skipped (pHash too different)"}
                      </span>
                    </span>
                  </p>
                  <p style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 2 }}>
                    Method: <span style={{ color: "var(--accent-cyan)" }}>{data.visual_analysis.analysis_method}</span>
                    {" · "}Compared against <span style={{ color: "white", fontWeight: 600 }}>47 trusted sites</span>
                  </p>
                </div>
                <div style={{ display: "flex", gap: 8 }}>
                  <span className={`badge ${data.visual_analysis.spoofing_detected ? "badge-danger" : "badge-safe"}`}>
                    {data.visual_analysis.spoofing_detected ? "⚠ SPOOFING" : "✓ NO CLONING"}
                  </span>
                  {data.visual_analysis.heatmap_available && (
                    <button onClick={() => setShowHeatmap(true)}
                      style={{
                        padding: "4px 12px", borderRadius: 8, background: "var(--glow-purple)", border: "1px solid rgba(124,100,255,0.2)",
                        color: "var(--accent-purple)", fontSize: 11, fontWeight: 600, cursor: "pointer"
                      }}>
                      Heatmap
                    </button>
                  )}
                </div>
              </div>
            </>
          )}
        </div>
      )}

      {/* SMS Details */}
      {data.sms_analysis && !data.sms_analysis.error && data.sms_analysis.features && (
        <div style={{ marginTop: 18 }}>
          <p style={{ fontSize: 12, fontWeight: 600, color: "var(--text-secondary)", marginBottom: 10, display: "flex", alignItems: "center", gap: 6 }}>
            <Type size={14} color="var(--accent-cyan)" /> SMS Indicators
          </p>
          <div className="result-features">
            <FeatureCard icon={<AlertTriangle size={15} />} label="Urgency" value={data.sms_analysis.features.urgency_keywords} danger={data.sms_analysis.features.urgency_keywords > 0} />
            <FeatureCard icon={<TrendingUp size={15} />} label="Financial" value={data.sms_analysis.features.financial_keywords} danger={data.sms_analysis.features.financial_keywords > 0} />
            <FeatureCard icon={<Zap size={15} />} label="Action" value={data.sms_analysis.features.action_keywords} danger={data.sms_analysis.features.action_keywords > 0} />
          </div>
        </div>
      )}

      <p style={{ marginTop: 14, fontSize: 11, color: "var(--text-muted)", textAlign: "right" }}>
        Analysis time: {data.total_analysis_time_ms?.toFixed(0)}ms
      </p>
    </div>
  );
}

/* ======================== Shared ======================== */
function FeatureCard({ icon, label, value, danger }) {
  return (
    <div className={`result-feature-card ${danger ? "danger" : ""}`}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 5, marginBottom: 4 }}>
        <span style={{ color: danger ? "var(--accent-red)" : "var(--text-muted)" }}>{icon}</span>
        <span className="result-feature-label">{label}</span>
      </div>
      <div className="result-feature-value" style={{ color: danger ? "var(--accent-red)" : "var(--text-primary)" }}>
        {value}
      </div>
    </div>
  );
}

function QI({ label, active, icon, good }) {
  const isActive = active === 1 || active === true;
  const isDanger = good ? !isActive : isActive;
  return (
    <div className={`q-indicator ${isDanger ? "active" : ""} ${good && isActive ? "good" : ""}`}>
      {icon} {label}
    </div>
  );
}

function ScoreBar({ label, weight, score, performed }) {
  const pct = score !== null ? (score > 1 ? score : score * 100) : 0;
  const color = pct < 30 ? "var(--accent-green)" : pct < 60 ? "var(--accent-amber)" : pct < 85 ? "#fb923c" : "var(--accent-red)";
  return (
    <div className="score-bar-row" style={{ opacity: performed ? 1 : 0.3 }}>
      <span className="score-bar-label">{label}</span>
      <span className="score-bar-weight">{weight}</span>
      <div className="score-bar-track">
        {performed && <div className="score-bar-fill" style={{ width: `${Math.max(pct, 3)}%`, background: color }} />}
      </div>
      <span className="score-bar-value">{performed ? Math.round(pct) : "—"}</span>
    </div>
  );
}
