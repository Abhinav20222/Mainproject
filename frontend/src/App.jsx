import { useState, useEffect } from "react";
import { Shield, Zap, AlertTriangle, CheckCircle, Loader2, Link2, Phone, Type, TrendingUp, AlertCircle, ShieldAlert, ShieldCheck, Globe, Scan, Eye, X, Search, Lock, Unlock, Hash, AtSign } from "lucide-react";
import axios from "axios";

const API_URL = "http://localhost:5000";

// Sample messages for quick testing
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

// Tab modes
const TABS = [
  { key: "sms", label: "SMS / Text", icon: Type },
  { key: "url", label: "URL Check", icon: Globe },
  { key: "fullscan", label: "Full Scan", icon: Scan },
];

export default function App() {
  const [activeTab, setActiveTab] = useState("sms");
  const [message, setMessage] = useState("");
  const [url, setUrl] = useState("");
  const [includeVisual, setIncludeVisual] = useState(false);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [apiOnline, setApiOnline] = useState(false);
  const [showHeatmap, setShowHeatmap] = useState(false);

  // Check API health on mount
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const res = await axios.get(`${API_URL}/api/health`, { timeout: 2000 });
        setApiOnline(res.data.status === "online");
      } catch {
        setApiOnline(false);
      }
    };
    checkHealth();
    const interval = setInterval(checkHealth, 1000);
    return () => clearInterval(interval);
  }, []);

  // Clear results when switching tabs
  useEffect(() => {
    setResult(null);
    setError(null);
  }, [activeTab]);

  const analyzeMessage = async () => {
    if (!message.trim()) return;
    setLoading(true); setError(null); setResult(null);
    try {
      const res = await axios.post(`${API_URL}/api/analyze`, { message });
      setResult({ type: "sms", data: res.data });
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
    } catch (err) {
      setError(err.response?.data?.error || "Failed to analyze URL.");
    } finally { setLoading(false); }
  };

  const fullScan = async () => {
    if (!message.trim()) return;
    setLoading(true); setError(null); setResult(null);
    try {
      const res = await axios.post(`${API_URL}/api/full-scan`, {
        message: message,
        include_visual: includeVisual,
      });
      setResult({ type: "fullscan", data: res.data });
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
    if (sample.text) setMessage(sample.text);
    if (sample.url) setUrl(sample.url);
    setResult(null); setError(null);
  };

  const getThreatColor = (score) => {
    const s = score > 1 ? score : score * 100;
    if (s < 30) return { bg: "from-green-500 to-emerald-400", text: "text-green-400", glow: "shadow-green-500/50", badge: "bg-green-500/20 text-green-400" };
    if (s < 60) return { bg: "from-yellow-500 to-amber-400", text: "text-yellow-400", glow: "shadow-yellow-500/50", badge: "bg-yellow-500/20 text-yellow-400" };
    if (s < 85) return { bg: "from-orange-500 to-red-400", text: "text-orange-400", glow: "shadow-orange-500/50", badge: "bg-orange-500/20 text-orange-400" };
    return { bg: "from-red-600 to-rose-500", text: "text-red-400", glow: "shadow-red-500/50", badge: "bg-red-500/20 text-red-400" };
  };

  const getThreatIcon = (isPhishing) =>
    isPhishing ? <ShieldAlert className="w-8 h-8" /> : <ShieldCheck className="w-8 h-8" />;

  const canAnalyze = () => {
    if (!apiOnline) return false;
    if (activeTab === "sms") return !!message.trim();
    if (activeTab === "url") return !!url.trim();
    return !!message.trim();
  };

  // Normalize score to 0-100 for display
  const displayScore = (score) => {
    if (score === undefined || score === null) return 0;
    return score > 1 ? Math.round(score) : Math.round(score * 100);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 text-white px-4 py-8">
      {/* Animated background grid */}
      <div className="fixed inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAiIGhlaWdodD0iNDAiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PGRlZnM+PHBhdHRlcm4gaWQ9ImdyaWQiIHdpZHRoPSI0MCIgaGVpZ2h0PSI0MCIgcGF0dGVyblVuaXRzPSJ1c2VyU3BhY2VPblVzZSI+PHBhdGggZD0iTSAwIDEwIEwgNDAgMTAgTSAxMCAwIEwgMTAgNDAgTSAwIDIwIEwgNDAgMjAgTSAyMCAwIEwgMjAgNDAgTSAwIDMwIEwgNDAgMzAgTSAzMCAwIEwgMzAgNDAiIGZpbGw9Im5vbmUiIHN0cm9rZT0icmdiYSgxMDAsMjAwLDI1NSwwLjAzKSIgc3Ryb2tlLXdpZHRoPSIxIi8+PC9wYXR0ZXJuPjwvZGVmcz48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSJ1cmwoI2dyaWQpIi8+PC9zdmc+')] opacity-50 pointer-events-none"></div>

      <div className="relative z-10 max-w-4xl mx-auto space-y-8">

        {/* Header */}
        <div className="flex items-center justify-between bg-white/5 border border-white/10 rounded-3xl p-6 backdrop-blur-xl">
          <div className="flex items-center gap-4">
            <div className="w-14 h-14 bg-gradient-to-br from-cyan-400 to-purple-600 rounded-2xl flex items-center justify-center shadow-lg shadow-cyan-500/30">
              <Shield className="w-7 h-7" />
            </div>
            <div>
              <h1 className="text-2xl font-bold bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent">
                PhishGuard AI
              </h1>
              <p className="text-sm text-gray-400">Multi-Channel Threat Detection • SMS • URL • Visual</p>
            </div>
          </div>
          <div className={`flex items-center gap-2 px-4 py-2 rounded-full ${apiOnline ? 'bg-green-500/10 text-green-400' : 'bg-red-500/10 text-red-400'}`}>
            <span className={`w-2 h-2 rounded-full ${apiOnline ? 'bg-green-400 animate-pulse' : 'bg-red-400'}`}></span>
            <span className="text-sm font-medium">{apiOnline ? 'AI Online' : 'AI Offline'}</span>
          </div>
        </div>

        {/* Main Card */}
        <div className="bg-gradient-to-br from-white/10 to-white/5 border border-white/20 rounded-3xl p-8 backdrop-blur-xl shadow-2xl">

          {/* Tab Switcher */}
          <div className="flex gap-2 mb-6 bg-black/30 rounded-2xl p-1.5">
            {TABS.map((tab) => {
              const Icon = tab.icon;
              return (
                <button key={tab.key} onClick={() => setActiveTab(tab.key)}
                  className={`flex-1 flex items-center justify-center gap-2 py-3 rounded-xl text-sm font-semibold transition-all
                    ${activeTab === tab.key
                      ? 'bg-gradient-to-r from-cyan-500/30 to-purple-500/30 text-cyan-300 border border-cyan-500/30 shadow-lg shadow-cyan-500/10'
                      : 'text-gray-400 hover:text-gray-200 hover:bg-white/5'}`}>
                  <Icon className="w-4 h-4" />
                  {tab.label}
                </button>
              );
            })}
          </div>

          {/* Input Section */}
          <div className="space-y-4">
            {activeTab === "sms" && (
              <div>
                <label className="text-sm font-medium text-gray-300 flex items-center gap-2 mb-2">
                  <Type className="w-4 h-4 text-cyan-400" />
                  Message Text
                </label>
                <textarea value={message} onChange={(e) => setMessage(e.target.value)}
                  className="w-full h-32 bg-black/40 border border-white/20 rounded-2xl p-5 text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-cyan-500/50 focus:border-cyan-500/50 resize-none transition-all"
                  placeholder="Paste a suspicious SMS or email message..."
                />
              </div>
            )}

            {activeTab === "url" && (
              <div>
                <label className="text-sm font-medium text-gray-300 flex items-center gap-2 mb-2">
                  <Globe className="w-4 h-4 text-purple-400" />
                  URL to Check
                </label>
                <input type="text" value={url} onChange={(e) => setUrl(e.target.value)}
                  className="w-full bg-black/40 border border-white/20 rounded-2xl p-5 text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-purple-500/50 focus:border-purple-500/50 transition-all"
                  placeholder="Enter a suspicious URL (e.g. http://suspicious-site.com/login)"
                />
              </div>
            )}

            {activeTab === "fullscan" && (
              <>
                <div>
                  <label className="text-sm font-medium text-gray-300 flex items-center gap-2 mb-2">
                    <Scan className="w-4 h-4 text-purple-400" />
                    Paste Full Message (text + URLs will be auto-detected)
                  </label>
                  <textarea value={message} onChange={(e) => setMessage(e.target.value)}
                    className="w-full h-40 bg-black/40 border border-white/20 rounded-2xl p-5 text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-purple-500/50 focus:border-purple-500/50 resize-none transition-all"
                    placeholder="Paste the entire suspicious message here&#10;&#10;Example: URGENT! Your SBI account has been suspended. Verify at http://sbi.login-secure.xyz/verify"
                  />
                  <p className="text-xs text-gray-500 mt-1.5 flex items-center gap-1">
                    <Search className="w-3 h-3" /> URLs will be automatically extracted and analyzed separately
                  </p>
                </div>
                <label className="flex items-center gap-3 cursor-pointer group">
                  <input type="checkbox" checked={includeVisual} onChange={(e) => setIncludeVisual(e.target.checked)}
                    className="w-5 h-5 rounded bg-black/40 border-white/20 text-cyan-500 focus:ring-cyan-500/50" />
                  <span className="text-sm text-gray-300 group-hover:text-white transition-colors">
                    <Eye className="w-4 h-4 inline mr-1" /> Include Visual Spoofing Analysis (slower, requires Chrome)
                  </span>
                </label>
              </>
            )}
          </div>

          {/* Analyze Button */}
          <button onClick={handleAnalyze} disabled={loading || !canAnalyze()}
            className="mt-6 w-full py-5 rounded-2xl bg-gradient-to-r from-cyan-500 via-purple-500 to-pink-500 font-bold flex items-center justify-center gap-3 hover:scale-[1.02] hover:shadow-lg hover:shadow-purple-500/30 transition-all disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100">
            {loading ? (
              <><Loader2 className="w-5 h-5 animate-spin" /> Analyzing...</>
            ) : (
              <><Zap className="w-5 h-5" /> {activeTab === "fullscan" ? "Launch Full Scan" : "Initiate Threat Scan"}</>
            )}
          </button>

          {/* Sample Items */}
          <div className="mt-6">
            <p className="text-sm text-gray-400 mb-3">Quick test samples:</p>
            <div className="grid grid-cols-3 gap-3">
              {(activeTab === "sms" ? SAMPLE_MESSAGES : activeTab === "url" ? SAMPLE_URLS : SAMPLE_FULLSCAN).map((sample, idx) => (
                <button key={idx} onClick={() => loadSample(sample)}
                  className={`py-3 px-4 rounded-xl border text-sm font-medium transition-all hover:scale-[1.02] ${sample.type === 'safe'
                    ? 'bg-green-500/10 border-green-500/30 text-green-400 hover:bg-green-500/20'
                    : 'bg-red-500/10 border-red-500/30 text-red-400 hover:bg-red-500/20'}`}>
                  {sample.label}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Error Display */}
        {error && (
          <div className="bg-red-500/10 border border-red-500/30 rounded-2xl p-6 flex items-center gap-4">
            <AlertCircle className="w-6 h-6 text-red-400 flex-shrink-0" />
            <p className="text-red-300">{error}</p>
          </div>
        )}

        {/* ==================== SMS RESULTS ==================== */}
        {result?.type === "sms" && (
          <SmsResultCard result={result.data} getThreatColor={getThreatColor} getThreatIcon={getThreatIcon} displayScore={displayScore} />
        )}

        {/* ==================== URL RESULTS ==================== */}
        {result?.type === "url" && result.data?.success && (
          <UrlResultCard result={result.data} getThreatColor={getThreatColor} displayScore={displayScore} />
        )}

        {/* ==================== FULL SCAN RESULTS ==================== */}
        {result?.type === "fullscan" && result.data?.success && (
          <FullScanResultCard result={result.data} getThreatColor={getThreatColor} displayScore={displayScore}
            showHeatmap={showHeatmap} setShowHeatmap={setShowHeatmap} apiUrl={API_URL} />
        )}

        {/* Footer */}
        <div className="text-center text-gray-500 text-sm">
          Protected by Machine Learning • SMS + URL + Visual Detection • PhishGuard AI
        </div>
      </div>

      {/* Heatmap Modal */}
      {showHeatmap && (
        <div className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center p-8" onClick={() => setShowHeatmap(false)}>
          <div className="bg-slate-900 border border-white/20 rounded-2xl p-6 max-w-2xl w-full" onClick={e => e.stopPropagation()}>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-bold text-white">Visual Difference Heatmap</h3>
              <button onClick={() => setShowHeatmap(false)} className="text-gray-400 hover:text-white"><X className="w-5 h-5" /></button>
            </div>
            <img src={`${API_URL}/api/heatmap?t=${Date.now()}`} alt="Difference Heatmap" className="w-full rounded-xl border border-white/10" />
            <p className="text-sm text-gray-400 mt-3">Red/warm areas indicate visual differences between the suspect and trusted site.</p>
          </div>
        </div>
      )}
    </div>
  );
}

/* ==================== SMS Result Card ==================== */
function SmsResultCard({ result, getThreatColor, getThreatIcon, displayScore }) {
  const score = displayScore(result.threat_score);
  const colors = getThreatColor(score);
  return (
    <div className={`bg-gradient-to-br from-white/10 to-white/5 border rounded-3xl p-8 backdrop-blur-xl shadow-2xl animate-fade-in ${result.is_phishing ? 'border-red-500/30' : 'border-green-500/30'}`}>
      <div className="flex items-center justify-between mb-8">
        <div className="flex items-center gap-4">
          <div className={`w-16 h-16 rounded-2xl flex items-center justify-center ${result.is_phishing ? 'bg-gradient-to-br from-red-500 to-rose-600 shadow-lg shadow-red-500/40' : 'bg-gradient-to-br from-green-500 to-emerald-600 shadow-lg shadow-green-500/40'}`}>
            {getThreatIcon(result.is_phishing)}
          </div>
          <div>
            <h2 className={`text-2xl font-bold ${result.is_phishing ? 'text-red-400' : 'text-green-400'}`}>
              {result.is_phishing ? 'PHISHING DETECTED' : 'MESSAGE SAFE'}
            </h2>
            <p className="text-gray-400">Confidence: {(result.confidence * 100).toFixed(1)}%</p>
          </div>
        </div>
        <div className={`text-center px-6 py-3 rounded-2xl ${result.is_phishing ? 'bg-red-500/20' : 'bg-green-500/20'}`}>
          <div className={`text-4xl font-bold ${colors.text}`}>{score}</div>
          <div className="text-xs text-gray-400 uppercase tracking-wider">Threat Score</div>
        </div>
      </div>
      <ThreatGauge score={score} colors={colors} />
      {result.is_phishing && <PhishingWarning />}
      {result.features && <SmsFeatureGrid features={result.features} />}
    </div>
  );
}

/* ==================== URL Result Card ==================== */
function UrlResultCard({ result, getThreatColor, displayScore }) {
  const score = displayScore(result.threat_score);
  const colors = getThreatColor(score);
  return (
    <div className={`bg-gradient-to-br from-white/10 to-white/5 border rounded-3xl p-8 backdrop-blur-xl shadow-2xl animate-fade-in ${result.is_phishing ? 'border-red-500/30' : 'border-green-500/30'}`}>
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-4">
          <div className={`w-16 h-16 rounded-2xl flex items-center justify-center ${result.is_phishing ? 'bg-gradient-to-br from-red-500 to-rose-600 shadow-lg shadow-red-500/40' : 'bg-gradient-to-br from-green-500 to-emerald-600 shadow-lg shadow-green-500/40'}`}>
            <Globe className="w-8 h-8" />
          </div>
          <div>
            <h2 className={`text-2xl font-bold ${result.is_phishing ? 'text-red-400' : 'text-green-400'}`}>
              {result.is_phishing ? 'PHISHING URL' : 'URL SAFE'}
            </h2>
            <p className="text-gray-400 text-sm truncate max-w-md">{result.url}</p>
          </div>
        </div>
        <div className="text-center">
          <div className={`text-4xl font-bold ${colors.text}`}>{score}</div>
          <div className="text-xs text-gray-400 uppercase">Threat Score</div>
          <div className={`mt-1 px-3 py-1 rounded-full text-xs font-bold ${colors.badge}`}>{result.risk_level}</div>
        </div>
      </div>

      <ThreatGauge score={score} colors={colors} />

      {/* Quick Indicators */}
      <div className="flex gap-3 my-6">
        <QuickIndicator label="IP Address" active={result.features?.has_ip_address} icon={<Hash className="w-4 h-4" />} />
        <QuickIndicator label="Shortened" active={result.features?.is_shortened} icon={<Link2 className="w-4 h-4" />} />
        <QuickIndicator label="Suspicious Words" active={result.features?.has_suspicious_words} icon={<AlertTriangle className="w-4 h-4" />} />
        <QuickIndicator label="HTTPS" active={result.features?.has_https} icon={result.features?.has_https ? <Lock className="w-4 h-4" /> : <Unlock className="w-4 h-4" />} good />
        <QuickIndicator label="Brand Spoof" active={result.features?.has_brand_in_subdomain} icon={<AtSign className="w-4 h-4" />} />
      </div>

      {/* Top Risk Features */}
      {result.top_risk_features?.length > 0 && (
        <div className="space-y-3">
          <h3 className="text-sm font-semibold text-gray-300 flex items-center gap-2">
            <TrendingUp className="w-4 h-4 text-cyan-400" /> Top Risk Features
          </h3>
          {result.top_risk_features.slice(0, 5).map((feat, i) => {
            const val = result.features?.[feat] ?? 0;
            const maxVal = Math.max(val, 1);
            const pct = Math.min(100, (val / maxVal) * 100);
            return (
              <div key={i} className="flex items-center gap-3">
                <span className="text-xs text-gray-400 w-44 truncate">{feat.replace(/_/g, ' ')}</span>
                <div className="flex-1 h-2 bg-gray-800 rounded-full overflow-hidden">
                  <div className="h-full bg-gradient-to-r from-red-500 to-orange-400 rounded-full" style={{ width: `${Math.max(pct, 8)}%` }} />
                </div>
                <span className="text-xs text-gray-400 w-12 text-right">{typeof val === 'number' ? (Number.isInteger(val) ? val : val.toFixed(2)) : val}</span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

/* ==================== Full Scan Result Card ==================== */
function FullScanResultCard({ result, getThreatColor, displayScore, showHeatmap, setShowHeatmap, apiUrl }) {
  const score = displayScore(result.combined_threat_score);
  const colors = getThreatColor(score);

  const smsScore = result.sms_analysis?.threat_score ?? null;
  const urlScore = result.url_analysis?.threat_score ?? null;
  const visualScore = result.visual_analysis?.visual_threat_score ?? null;

  return (
    <div className={`bg-gradient-to-br from-white/10 to-white/5 border rounded-3xl p-8 backdrop-blur-xl shadow-2xl animate-fade-in ${score >= 60 ? 'border-red-500/30' : score >= 30 ? 'border-yellow-500/30' : 'border-green-500/30'}`}>
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className={`text-2xl font-bold ${colors.text}`}>COMBINED THREAT ANALYSIS</h2>
          <p className="text-gray-400 text-sm">Channels: {result.analyses_performed?.join(', ') || 'none'}</p>
        </div>
        <div className="text-center">
          <div className={`text-5xl font-bold ${colors.text}`}>{score}</div>
          <div className={`mt-1 px-3 py-1 rounded-full text-xs font-bold ${colors.badge}`}>{result.risk_level}</div>
        </div>
      </div>

      <ThreatGauge score={score} colors={colors} />

      {/* Score Breakdown */}
      <div className="mt-8 space-y-4">
        <h3 className="text-sm font-semibold text-gray-300">Score Breakdown</h3>
        <ScoreBar label="SMS Score" weight="40%" score={smsScore} performed={result.analyses_performed?.includes('sms')} />
        <ScoreBar label="URL Score" weight="45%" score={urlScore} performed={result.analyses_performed?.includes('url')} />
        <ScoreBar label="Visual Score" weight="15%" score={visualScore} performed={result.analyses_performed?.includes('visual')} />
      </div>

      {/* Extracted URLs */}
      {result.url_analysis?.urls_checked?.length > 0 && (
        <div className="mt-6 p-4 bg-black/30 border border-white/10 rounded-2xl">
          <h3 className="text-sm font-bold text-gray-200 flex items-center gap-2 mb-2">
            <Link2 className="w-4 h-4 text-cyan-400" /> Auto-Extracted URLs ({result.url_analysis.urls_checked.length})
          </h3>
          {result.url_analysis.urls_checked.map((u, i) => (
            <p key={i} className="text-xs text-gray-400 truncate py-0.5">• {u}</p>
          ))}
          {result.url_analysis.url && (
            <p className="text-xs text-yellow-400 mt-2">⚠ Highest risk: <span className="text-white font-medium">{result.url_analysis.url}</span></p>
          )}
        </div>
      )}

      {/* Visual Spoofing Card */}
      {result.visual_analysis && !result.visual_analysis.error && (
        <div className="mt-6 p-5 bg-black/30 border border-white/10 rounded-2xl">
          <h3 className="text-sm font-bold text-gray-200 flex items-center gap-2 mb-3">
            <Eye className="w-4 h-4 text-purple-400" /> Visual Spoofing Check
          </h3>
          <div className="flex items-center justify-between">
            <div>
              {result.visual_analysis.best_match_site && (
                <p className="text-sm text-gray-300">Best Match: <span className="font-bold text-white uppercase">{result.visual_analysis.best_match_site}</span>
                  <span className="text-gray-500 ml-1">({result.visual_analysis.best_match_url})</span></p>
              )}
              {result.visual_analysis.ssim_score > 0 && (
                <p className="text-sm text-gray-400 mt-1">SSIM Similarity: <span className="font-bold text-white">{(result.visual_analysis.ssim_score * 100).toFixed(1)}%</span></p>
              )}
            </div>
            <div className="flex items-center gap-3">
              <span className={`px-3 py-1 rounded-full text-xs font-bold ${result.visual_analysis.spoofing_detected ? 'bg-red-500/20 text-red-400' : 'bg-green-500/20 text-green-400'}`}>
                {result.visual_analysis.spoofing_detected ? '⚠ SPOOFING DETECTED' : '✓ NO VISUAL MATCH'}
              </span>
              {result.visual_analysis.heatmap_available && (
                <button onClick={() => setShowHeatmap(true)}
                  className="px-3 py-1 rounded-xl bg-purple-500/20 border border-purple-500/30 text-purple-300 text-xs font-medium hover:bg-purple-500/30 transition-all">
                  View Heatmap
                </button>
              )}
            </div>
          </div>
        </div>
      )}

      {/* SMS Details */}
      {result.sms_analysis && !result.sms_analysis.error && result.sms_analysis.features && (
        <div className="mt-6">
          <h3 className="text-sm font-semibold text-gray-300 mb-3 flex items-center gap-2">
            <Type className="w-4 h-4 text-cyan-400" /> SMS Indicators
          </h3>
          <SmsFeatureGrid features={result.sms_analysis.features} compact />
        </div>
      )}

      <div className="mt-4 text-xs text-gray-500 text-right">
        Analysis time: {result.total_analysis_time_ms?.toFixed(0)}ms
      </div>
    </div>
  );
}

/* ==================== Shared Components ==================== */

function ThreatGauge({ score, colors }) {
  return (
    <div className="mb-6">
      <div className="flex justify-between text-xs text-gray-500 mb-1">
        <span>Safe</span><span>Suspicious</span><span>Dangerous</span><span>Critical</span>
      </div>
      <div className="h-3.5 bg-gray-800 rounded-full overflow-hidden">
        <div className={`h-full bg-gradient-to-r ${colors.bg} transition-all duration-1000 ease-out rounded-full ${colors.glow} shadow-lg`}
          style={{ width: `${score}%` }} />
      </div>
    </div>
  );
}

function PhishingWarning() {
  return (
    <div className="mb-6 p-4 bg-red-500/10 border border-red-500/30 rounded-xl flex items-center gap-3">
      <AlertTriangle className="w-6 h-6 text-red-400 flex-shrink-0" />
      <p className="text-red-300 font-medium">⚠️ This shows signs of a phishing attack. Do NOT click any links or share personal information.</p>
    </div>
  );
}

function SmsFeatureGrid({ features, compact }) {
  return (
    <div className={`grid ${compact ? 'grid-cols-3' : 'grid-cols-2 md:grid-cols-3'} gap-3`}>
      <FeatureCard icon={<AlertTriangle className="w-4 h-4" />} label="Urgency" value={features.urgency_keywords} danger={features.urgency_keywords > 0} />
      <FeatureCard icon={<TrendingUp className="w-4 h-4" />} label="Financial" value={features.financial_keywords} danger={features.financial_keywords > 0} />
      <FeatureCard icon={<Zap className="w-4 h-4" />} label="Action" value={features.action_keywords} danger={features.action_keywords > 0} />
      <FeatureCard icon={<Link2 className="w-4 h-4" />} label="URLs" value={features.has_url ? "Yes" : "No"} danger={features.has_url} />
      <FeatureCard icon={<Phone className="w-4 h-4" />} label="Phone" value={features.has_phone ? "Yes" : "No"} danger={features.has_phone} />
      <FeatureCard icon={<Type className="w-4 h-4" />} label="CAPS" value={features.excessive_caps ? "Yes" : "No"} danger={features.excessive_caps} />
    </div>
  );
}

function FeatureCard({ icon, label, value, danger }) {
  return (
    <div className={`p-3 rounded-xl border transition-all ${danger ? 'bg-red-500/10 border-red-500/30' : 'bg-white/5 border-white/10'}`}>
      <div className="flex items-center gap-2 mb-1">
        <span className={danger ? 'text-red-400' : 'text-gray-400'}>{icon}</span>
        <span className="text-xs text-gray-400">{label}</span>
      </div>
      <div className={`text-lg font-bold ${danger ? 'text-red-400' : 'text-gray-200'}`}>{value}</div>
    </div>
  );
}

function QuickIndicator({ label, active, icon, good }) {
  const isActive = active === 1 || active === true;
  const isDanger = good ? !isActive : isActive;
  return (
    <div className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium border transition-all
      ${isDanger ? 'bg-red-500/15 border-red-500/30 text-red-400' : 'bg-white/5 border-white/10 text-gray-500'}`}>
      {icon}
      {label}
    </div>
  );
}

function ScoreBar({ label, weight, score, performed }) {
  const pct = score !== null ? (score > 1 ? score : score * 100) : 0;
  return (
    <div className={`flex items-center gap-3 ${!performed ? 'opacity-30' : ''}`}>
      <span className="text-xs text-gray-400 w-24">{label}</span>
      <span className="text-xs text-gray-500 w-10">{weight}</span>
      <div className="flex-1 h-2.5 bg-gray-800 rounded-full overflow-hidden">
        {performed && (
          <div className={`h-full rounded-full transition-all duration-700 ${pct < 30 ? 'bg-green-500' : pct < 60 ? 'bg-yellow-500' : pct < 85 ? 'bg-orange-500' : 'bg-red-500'}`}
            style={{ width: `${Math.max(pct, 3)}%` }} />
        )}
      </div>
      <span className="text-xs font-bold text-gray-300 w-10 text-right">{performed ? Math.round(pct) : '—'}</span>
    </div>
  );
}
