import { useState, useEffect, useCallback, useRef, useMemo } from "react";

// ============================================================================
// Constants & Mock Data
// ============================================================================

const VERDICT_CONFIG = {
  authentic: { label: "Authentic", color: "#10b981", bg: "#10b98120", icon: "✓" },
  likely_authentic: { label: "Likely Authentic", color: "#34d399", bg: "#34d39920", icon: "◑" },
  uncertain: { label: "Uncertain", color: "#f59e0b", bg: "#f59e0b20", icon: "?" },
  likely_fake: { label: "Likely Fake", color: "#f97316", bg: "#f9731620", icon: "⚠" },
  fake: { label: "Fake", color: "#ef4444", bg: "#ef444420", icon: "✗" },
};

const MANIPULATION_TYPES = [
  "Face Swap", "Face Reenactment", "GAN Generated", "Diffusion Generated",
  "Voice Clone", "Lip Sync", "Inpainting", "Audio Splice",
];

const MODELS = [
  { name: "EfficientNet-B4", version: "3.0.0", type: "Manipulation", accuracy: 96.2, status: "loaded" },
  { name: "Xception", version: "2.1.0", type: "Manipulation", accuracy: 95.1, status: "loaded" },
  { name: "Capsule Network v2", version: "2.0.0", type: "Manipulation", accuracy: 94.8, status: "loaded" },
  { name: "Multi-Attention Net", version: "1.5.0", type: "Manipulation", accuracy: 96.5, status: "loaded" },
  { name: "Frequency Aware Net", version: "1.2.0", type: "Frequency", accuracy: 93.7, status: "loaded" },
  { name: "RetinaFace", version: "2.0.0", type: "Face Detection", accuracy: 99.1, status: "loaded" },
  { name: "RawNet3", version: "2.0.0", type: "Audio", accuracy: 95.8, status: "loaded" },
  { name: "SyncNet v2", version: "1.5.0", type: "Lip Sync", accuracy: 92.3, status: "loaded" },
  { name: "GAN Forensics v3", version: "3.0.0", type: "GAN Detection", accuracy: 94.5, status: "loaded" },
];

const generateAnalyses = (count) => {
  const verdicts = ["authentic", "likely_authentic", "uncertain", "likely_fake", "fake"];
  const types = ["image", "video", "audio"];
  const statuses = ["completed", "completed", "completed", "analyzing", "queued"];
  const filenames = ["photo_001.jpg", "interview.mp4", "podcast.mp3", "selfie.png", "clip_final.mov", "recording.wav", "portrait.jpg", "news_clip.mp4"];
  return Array.from({ length: count }, (_, i) => ({
    id: `analysis-${String(i + 1).padStart(4, "0")}`,
    filename: filenames[i % filenames.length],
    type: types[i % types.length],
    status: statuses[i % statuses.length],
    verdict: verdicts[Math.floor(Math.random() * verdicts.length)],
    confidence: +(0.5 + Math.random() * 0.5).toFixed(3),
    manipulationProbability: +(Math.random() * 0.8).toFixed(3),
    processingTime: Math.floor(800 + Math.random() * 4000),
    createdAt: new Date(Date.now() - Math.random() * 7 * 86400000).toISOString(),
    faces: Math.floor(Math.random() * 4),
    componentScores: {
      face_manipulation: +(Math.random()).toFixed(2),
      frequency_analysis: +(Math.random()).toFixed(2),
      gan_detection: +(Math.random()).toFixed(2),
      noise_analysis: +(Math.random()).toFixed(2),
      compression: +(Math.random() * 0.5).toFixed(2),
      metadata: +(Math.random() * 0.3).toFixed(2),
      temporal: +(Math.random()).toFixed(2),
    },
  }));
};

const MOCK_ANALYSES = generateAnalyses(50);
const TIMELINE_DATA = Array.from({ length: 30 }, (_, i) => ({
  date: new Date(Date.now() - (29 - i) * 86400000).toLocaleDateString("en-US", { month: "short", day: "numeric" }),
  total: Math.floor(5 + Math.random() * 20),
  fake: Math.floor(Math.random() * 8),
  authentic: Math.floor(3 + Math.random() * 12),
}));

// ============================================================================
// Utility Components
// ============================================================================

const cn = (...classes) => classes.filter(Boolean).join(" ");

const Card = ({ children, className = "", onClick, hover = false }) => (
  <div
    onClick={onClick}
    className={cn(
      "bg-slate-800/60 backdrop-blur-sm border border-slate-700/50 rounded-xl",
      hover && "hover:border-blue-500/40 hover:bg-slate-800/80 cursor-pointer transition-all duration-300",
      className
    )}
  >
    {children}
  </div>
);

const Badge = ({ children, variant = "default", className = "" }) => {
  const variants = {
    default: "bg-slate-700 text-slate-300",
    success: "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30",
    warning: "bg-amber-500/20 text-amber-400 border border-amber-500/30",
    danger: "bg-red-500/20 text-red-400 border border-red-500/30",
    info: "bg-blue-500/20 text-blue-400 border border-blue-500/30",
    purple: "bg-purple-500/20 text-purple-400 border border-purple-500/30",
  };
  return (
    <span className={cn("inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium", variants[variant], className)}>
      {children}
    </span>
  );
};

const ProgressBar = ({ value, max = 100, color = "#3b82f6", height = 6, label }) => (
  <div>
    {label && <div className="flex justify-between text-xs text-slate-400 mb-1"><span>{label}</span><span>{Math.round((value / max) * 100)}%</span></div>}
    <div className="w-full rounded-full overflow-hidden" style={{ height, background: "#1e293b" }}>
      <div className="h-full rounded-full transition-all duration-700 ease-out" style={{ width: `${(value / max) * 100}%`, background: color }} />
    </div>
  </div>
);

const VerdictBadge = ({ verdict }) => {
  const config = VERDICT_CONFIG[verdict] || VERDICT_CONFIG.uncertain;
  return (
    <span
      className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold"
      style={{ background: config.bg, color: config.color, border: `1px solid ${config.color}40` }}
    >
      <span>{config.icon}</span>
      {config.label}
    </span>
  );
};

const StatCard = ({ label, value, change, icon, color = "#3b82f6" }) => (
  <Card className="p-5">
    <div className="flex items-center justify-between mb-3">
      <span className="text-sm text-slate-400">{label}</span>
      <div className="w-10 h-10 rounded-lg flex items-center justify-center text-lg" style={{ background: `${color}20` }}>
        {icon}
      </div>
    </div>
    <div className="text-2xl font-bold text-white">{value}</div>
    {change !== undefined && (
      <div className={cn("text-xs mt-1", change >= 0 ? "text-emerald-400" : "text-red-400")}>
        {change >= 0 ? "↑" : "↓"} {Math.abs(change)}% vs last week
      </div>
    )}
  </Card>
);

const MiniChart = ({ data, width = 120, height = 40, color = "#3b82f6" }) => {
  if (!data?.length) return null;
  const max = Math.max(...data);
  const min = Math.min(...data);
  const range = max - min || 1;
  const points = data.map((v, i) => `${(i / (data.length - 1)) * width},${height - ((v - min) / range) * (height - 4) - 2}`).join(" ");
  return (
    <svg width={width} height={height} className="overflow-visible">
      <polyline points={points} fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
};

const BarChart = ({ data, labels, colors, height = 200 }) => {
  const max = Math.max(...data, 1);
  return (
    <div className="flex items-end gap-2 justify-around" style={{ height }}>
      {data.map((v, i) => (
        <div key={i} className="flex flex-col items-center gap-1 flex-1">
          <span className="text-xs text-slate-400">{v}</span>
          <div
            className="w-full rounded-t-md transition-all duration-500 min-w-[24px] max-w-[48px]"
            style={{ height: `${(v / max) * (height - 40)}px`, background: colors?.[i] || "#3b82f6" }}
          />
          <span className="text-[10px] text-slate-500 text-center leading-tight">{labels?.[i]}</span>
        </div>
      ))}
    </div>
  );
};

const DonutChart = ({ segments, size = 160, strokeWidth = 20 }) => {
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const total = segments.reduce((s, seg) => s + seg.value, 0) || 1;
  let offset = 0;

  return (
    <svg width={size} height={size} className="transform -rotate-90">
      <circle cx={size / 2} cy={size / 2} r={radius} fill="none" stroke="#1e293b" strokeWidth={strokeWidth} />
      {segments.map((seg, i) => {
        const pct = seg.value / total;
        const dashArray = `${pct * circumference} ${circumference}`;
        const dashOffset = -offset * circumference;
        offset += pct;
        return (
          <circle
            key={i} cx={size / 2} cy={size / 2} r={radius} fill="none"
            stroke={seg.color} strokeWidth={strokeWidth}
            strokeDasharray={dashArray} strokeDashoffset={dashOffset}
            strokeLinecap="round" className="transition-all duration-700"
          />
        );
      })}
    </svg>
  );
};

// ============================================================================
// Page Components
// ============================================================================

const DashboardPage = () => {
  const verdictCounts = useMemo(() => {
    const counts = { authentic: 0, likely_authentic: 0, uncertain: 0, likely_fake: 0, fake: 0 };
    MOCK_ANALYSES.filter(a => a.status === "completed").forEach(a => { if (counts[a.verdict] !== undefined) counts[a.verdict]++; });
    return counts;
  }, []);

  const fakeRate = useMemo(() => {
    const completed = MOCK_ANALYSES.filter(a => a.status === "completed");
    const fakes = completed.filter(a => a.verdict === "fake" || a.verdict === "likely_fake");
    return completed.length ? ((fakes.length / completed.length) * 100).toFixed(1) : 0;
  }, []);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Dashboard</h1>
          <p className="text-sm text-slate-400 mt-1">Overview of your deepfake detection activity</p>
        </div>
        <Badge variant="info">Last 7 days</Badge>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard label="Total Analyses" value={MOCK_ANALYSES.length} change={12} icon="📊" color="#3b82f6" />
        <StatCard label="Fakes Detected" value={verdictCounts.fake + verdictCounts.likely_fake} change={-5} icon="🔍" color="#ef4444" />
        <StatCard label="Avg Confidence" value="87.2%" change={3} icon="🎯" color="#10b981" />
        <StatCard label="Detection Rate" value={`${fakeRate}%`} change={8} icon="⚡" color="#f59e0b" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Card className="p-6 lg:col-span-2">
          <h3 className="text-sm font-medium text-slate-300 mb-4">Analysis Timeline</h3>
          <div className="flex items-end gap-1 h-[180px]">
            {TIMELINE_DATA.map((d, i) => (
              <div key={i} className="flex-1 flex flex-col items-center gap-0.5">
                <div className="w-full flex flex-col-reverse gap-0.5">
                  <div className="w-full rounded-t-sm bg-emerald-500/70" style={{ height: `${d.authentic * 6}px` }} />
                  <div className="w-full rounded-t-sm bg-red-500/70" style={{ height: `${d.fake * 6}px` }} />
                </div>
                {i % 5 === 0 && <span className="text-[9px] text-slate-500 mt-1">{d.date}</span>}
              </div>
            ))}
          </div>
          <div className="flex gap-4 mt-3 text-xs text-slate-400">
            <span className="flex items-center gap-1"><span className="w-2 h-2 rounded bg-emerald-500" />Authentic</span>
            <span className="flex items-center gap-1"><span className="w-2 h-2 rounded bg-red-500" />Fake</span>
          </div>
        </Card>

        <Card className="p-6">
          <h3 className="text-sm font-medium text-slate-300 mb-4">Verdict Distribution</h3>
          <div className="flex justify-center mb-4">
            <DonutChart
              segments={Object.entries(verdictCounts).map(([k, v]) => ({
                value: v, color: VERDICT_CONFIG[k]?.color || "#666",
              }))}
              size={140}
              strokeWidth={18}
            />
          </div>
          <div className="space-y-2">
            {Object.entries(verdictCounts).map(([key, count]) => (
              <div key={key} className="flex items-center justify-between text-xs">
                <span className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full" style={{ background: VERDICT_CONFIG[key]?.color }} />
                  <span className="text-slate-400">{VERDICT_CONFIG[key]?.label}</span>
                </span>
                <span className="text-slate-300 font-medium">{count}</span>
              </div>
            ))}
          </div>
        </Card>
      </div>

      <Card className="p-6">
        <h3 className="text-sm font-medium text-slate-300 mb-4">Recent Analyses</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-slate-400 text-xs border-b border-slate-700/50">
                <th className="text-left pb-3 font-medium">File</th>
                <th className="text-left pb-3 font-medium">Type</th>
                <th className="text-left pb-3 font-medium">Verdict</th>
                <th className="text-left pb-3 font-medium">Confidence</th>
                <th className="text-left pb-3 font-medium">Time</th>
                <th className="text-left pb-3 font-medium">Date</th>
              </tr>
            </thead>
            <tbody>
              {MOCK_ANALYSES.slice(0, 10).map((a) => (
                <tr key={a.id} className="border-b border-slate-800/50 hover:bg-slate-800/30 transition-colors">
                  <td className="py-3 text-white font-medium">{a.filename}</td>
                  <td className="py-3"><Badge variant={a.type === "video" ? "purple" : a.type === "audio" ? "info" : "default"}>{a.type}</Badge></td>
                  <td className="py-3">{a.status === "completed" ? <VerdictBadge verdict={a.verdict} /> : <Badge variant="warning">{a.status}</Badge>}</td>
                  <td className="py-3 text-slate-300">{a.status === "completed" ? `${(a.confidence * 100).toFixed(1)}%` : "—"}</td>
                  <td className="py-3 text-slate-400">{a.processingTime}ms</td>
                  <td className="py-3 text-slate-500">{new Date(a.createdAt).toLocaleDateString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
};

const UploadPage = ({ onAnalyze }) => {
  const [isDragging, setIsDragging] = useState(false);
  const [files, setFiles] = useState([]);
  const [analyzing, setAnalyzing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState(null);
  const fileRef = useRef(null);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
    const dropped = Array.from(e.dataTransfer?.files || []);
    if (dropped.length) setFiles(prev => [...prev, ...dropped]);
  }, []);

  const handleFileSelect = (e) => {
    const selected = Array.from(e.target?.files || []);
    if (selected.length) setFiles(prev => [...prev, ...selected]);
  };

  const startAnalysis = () => {
    if (!files.length) return;
    setAnalyzing(true);
    setProgress(0);
    setResult(null);

    const steps = ["Preprocessing", "Face Detection", "Manipulation Analysis", "Frequency Analysis", "GAN Detection", "Noise Analysis", "Ensemble Prediction", "Complete"];
    let step = 0;

    const interval = setInterval(() => {
      step++;
      setProgress(Math.min(100, (step / steps.length) * 100));
      if (step >= steps.length) {
        clearInterval(interval);
        setAnalyzing(false);
        setResult(MOCK_ANALYSES[Math.floor(Math.random() * MOCK_ANALYSES.length)]);
      }
    }, 600);
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white">Upload & Analyze</h1>
        <p className="text-sm text-slate-400 mt-1">Upload images, videos, or audio files for deepfake detection</p>
      </div>

      {!result ? (
        <>
          <Card
            className={cn(
              "p-12 border-2 border-dashed transition-all duration-300 text-center",
              isDragging ? "border-blue-400 bg-blue-500/10" : "border-slate-600 hover:border-slate-500"
            )}
            onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
          >
            <div className="text-5xl mb-4">{isDragging ? "📥" : "🔍"}</div>
            <h3 className="text-lg font-medium text-white mb-2">
              {isDragging ? "Drop files here" : "Drop files or click to upload"}
            </h3>
            <p className="text-sm text-slate-400 mb-4">
              Supports JPEG, PNG, MP4, AVI, MOV, WAV, MP3 — up to 500MB
            </p>
            <button
              onClick={() => fileRef.current?.click()}
              className="px-6 py-2.5 bg-blue-600 hover:bg-blue-500 text-white rounded-lg text-sm font-medium transition-colors"
            >
              Browse Files
            </button>
            <input ref={fileRef} type="file" multiple accept="image/*,video/*,audio/*" onChange={handleFileSelect} className="hidden" />
          </Card>

          {files.length > 0 && (
            <Card className="p-6">
              <h3 className="text-sm font-medium text-slate-300 mb-3">Selected Files ({files.length})</h3>
              <div className="space-y-2 mb-4">
                {files.map((f, i) => (
                  <div key={i} className="flex items-center justify-between py-2 px-3 bg-slate-900/50 rounded-lg">
                    <div className="flex items-center gap-3">
                      <span className="text-lg">{f.type?.startsWith("image") ? "🖼️" : f.type?.startsWith("video") ? "🎥" : "🎵"}</span>
                      <div>
                        <div className="text-sm text-white">{f.name}</div>
                        <div className="text-xs text-slate-500">{(f.size / 1024 / 1024).toFixed(2)} MB</div>
                      </div>
                    </div>
                    <button onClick={() => setFiles(prev => prev.filter((_, j) => j !== i))} className="text-slate-500 hover:text-red-400 transition-colors">✕</button>
                  </div>
                ))}
              </div>

              {analyzing ? (
                <div className="space-y-3">
                  <ProgressBar value={progress} color="#3b82f6" height={8} label="Analyzing..." />
                  <p className="text-xs text-slate-400 text-center animate-pulse">Running detection pipeline...</p>
                </div>
              ) : (
                <button
                  onClick={startAnalysis}
                  className="w-full py-3 bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-500 hover:to-cyan-500 text-white rounded-lg font-medium transition-all duration-300 shadow-lg shadow-blue-500/25"
                >
                  🔬 Start Analysis
                </button>
              )}
            </Card>
          )}
        </>
      ) : (
        <AnalysisResultView result={result} onBack={() => { setResult(null); setFiles([]); }} />
      )}
    </div>
  );
};

const AnalysisResultView = ({ result, onBack }) => {
  const config = VERDICT_CONFIG[result.verdict] || VERDICT_CONFIG.uncertain;

  return (
    <div className="space-y-6 animate-fade-in">
      <button onClick={onBack} className="text-sm text-slate-400 hover:text-white transition-colors flex items-center gap-1">
        ← Back to upload
      </button>

      <Card className="p-8 text-center" style={{ borderColor: `${config.color}30` }}>
        <div className="text-6xl mb-3">{config.icon === "✓" ? "✅" : config.icon === "✗" ? "🚨" : "⚠️"}</div>
        <div className="text-3xl font-bold mb-2" style={{ color: config.color }}>{config.label}</div>
        <p className="text-slate-400">Confidence: {(result.confidence * 100).toFixed(1)}% | Processing: {result.processingTime}ms</p>
        <div className="mt-4 inline-flex gap-2">
          <Badge variant="info">ID: {result.id}</Badge>
          <Badge variant="default">{result.type.toUpperCase()}</Badge>
          <Badge variant="default">{result.faces} face{result.faces !== 1 ? "s" : ""} detected</Badge>
        </div>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card className="p-6">
          <h3 className="text-sm font-medium text-slate-300 mb-4">Component Scores</h3>
          <div className="space-y-3">
            {Object.entries(result.componentScores).map(([key, value]) => (
              <ProgressBar
                key={key}
                value={value * 100}
                label={key.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase())}
                color={value > 0.7 ? "#ef4444" : value > 0.4 ? "#f59e0b" : "#10b981"}
                height={6}
              />
            ))}
          </div>
        </Card>

        <Card className="p-6">
          <h3 className="text-sm font-medium text-slate-300 mb-4">Detection Breakdown</h3>
          <BarChart
            data={Object.values(result.componentScores).map(v => Math.round(v * 100))}
            labels={Object.keys(result.componentScores).map(k => k.split("_")[0])}
            colors={Object.values(result.componentScores).map(v => v > 0.7 ? "#ef4444" : v > 0.4 ? "#f59e0b" : "#10b981")}
            height={180}
          />
        </Card>
      </div>

      <Card className="p-6">
        <h3 className="text-sm font-medium text-slate-300 mb-3">Analysis Explanation</h3>
        <p className="text-slate-400 text-sm leading-relaxed">
          The analysis examined the uploaded {result.type} using {MODELS.length} detection models in an ensemble configuration.
          {result.faces > 0 && ` ${result.faces} face(s) were detected and individually analyzed for manipulation artifacts.`}
          {" "}The frequency domain analysis checked for DCT, FFT, and wavelet anomalies.
          GAN fingerprint detection searched for signatures from known generators including StyleGAN, ProGAN, and diffusion models.
          Noise consistency analysis verified sensor noise patterns across the image regions.
          The weighted ensemble of all component scores produced a final verdict of <strong style={{ color: config.color }}>{config.label}</strong> with {(result.confidence * 100).toFixed(1)}% confidence.
        </p>
      </Card>

      <div className="flex gap-3">
        <button className="px-5 py-2.5 bg-blue-600 hover:bg-blue-500 text-white rounded-lg text-sm font-medium transition-colors">📄 Generate Report</button>
        <button className="px-5 py-2.5 bg-slate-700 hover:bg-slate-600 text-white rounded-lg text-sm font-medium transition-colors">📊 Export JSON</button>
        <button className="px-5 py-2.5 bg-slate-700 hover:bg-slate-600 text-white rounded-lg text-sm font-medium transition-colors">🔄 Re-analyze</button>
      </div>
    </div>
  );
};

const HistoryPage = () => {
  const [filter, setFilter] = useState("all");
  const [search, setSearch] = useState("");

  const filtered = useMemo(() => {
    let items = MOCK_ANALYSES;
    if (filter !== "all") items = items.filter(a => a.verdict === filter);
    if (search) items = items.filter(a => a.filename.toLowerCase().includes(search.toLowerCase()) || a.id.includes(search));
    return items;
  }, [filter, search]);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between flex-wrap gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white">Analysis History</h1>
          <p className="text-sm text-slate-400 mt-1">{MOCK_ANALYSES.length} total analyses</p>
        </div>
        <div className="flex gap-2">
          <input
            type="text" placeholder="Search files..." value={search} onChange={e => setSearch(e.target.value)}
            className="px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-sm text-white placeholder-slate-500 focus:border-blue-500 focus:outline-none w-48"
          />
        </div>
      </div>

      <div className="flex gap-2 flex-wrap">
        {["all", "authentic", "likely_authentic", "uncertain", "likely_fake", "fake"].map(v => (
          <button
            key={v}
            onClick={() => setFilter(v)}
            className={cn(
              "px-3 py-1.5 rounded-lg text-xs font-medium transition-colors",
              filter === v ? "bg-blue-600 text-white" : "bg-slate-800 text-slate-400 hover:text-white"
            )}
          >
            {v === "all" ? "All" : VERDICT_CONFIG[v]?.label}
          </button>
        ))}
      </div>

      <div className="grid gap-3">
        {filtered.slice(0, 20).map(a => (
          <Card key={a.id} className="p-4" hover>
            <div className="flex items-center justify-between flex-wrap gap-3">
              <div className="flex items-center gap-4">
                <div className="w-10 h-10 rounded-lg bg-slate-900 flex items-center justify-center text-lg">
                  {a.type === "image" ? "🖼️" : a.type === "video" ? "🎥" : "🎵"}
                </div>
                <div>
                  <div className="text-sm font-medium text-white">{a.filename}</div>
                  <div className="text-xs text-slate-500">{a.id} · {new Date(a.createdAt).toLocaleString()}</div>
                </div>
              </div>
              <div className="flex items-center gap-3">
                {a.status === "completed" ? <VerdictBadge verdict={a.verdict} /> : <Badge variant="warning">{a.status}</Badge>}
                {a.status === "completed" && (
                  <span className="text-xs text-slate-500">{(a.confidence * 100).toFixed(0)}% conf · {a.processingTime}ms</span>
                )}
              </div>
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
};

const ModelsPage = () => (
  <div className="space-y-6">
    <div>
      <h1 className="text-2xl font-bold text-white">ML Models</h1>
      <p className="text-sm text-slate-400 mt-1">{MODELS.length} models loaded in the detection pipeline</p>
    </div>
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      {MODELS.map((m, i) => (
        <Card key={i} className="p-5" hover>
          <div className="flex items-center justify-between mb-3">
            <Badge variant="info">{m.type}</Badge>
            <Badge variant="success">{m.status}</Badge>
          </div>
          <h3 className="text-white font-medium mb-1">{m.name}</h3>
          <p className="text-xs text-slate-500 mb-3">v{m.version}</p>
          <ProgressBar value={m.accuracy} label="Accuracy" color={m.accuracy > 95 ? "#10b981" : "#f59e0b"} height={5} />
          <div className="mt-3 flex gap-2">
            <button className="text-xs text-blue-400 hover:text-blue-300 transition-colors">Metrics →</button>
            <button className="text-xs text-slate-500 hover:text-slate-400 transition-colors">Reload</button>
          </div>
        </Card>
      ))}
    </div>
  </div>
);

const SettingsPage = () => {
  const [darkMode, setDarkMode] = useState(true);
  const [notifications, setNotifications] = useState(true);
  const [autoAnalyze, setAutoAnalyze] = useState(false);

  const Toggle = ({ checked, onChange, label }) => (
    <div className="flex items-center justify-between py-3">
      <span className="text-sm text-slate-300">{label}</span>
      <button
        onClick={() => onChange(!checked)}
        className={cn("w-11 h-6 rounded-full transition-colors relative", checked ? "bg-blue-600" : "bg-slate-700")}
      >
        <div className={cn("w-5 h-5 rounded-full bg-white absolute top-0.5 transition-transform", checked ? "translate-x-5.5 left-[1px]" : "left-[2px]")}
          style={{ transform: checked ? "translateX(22px)" : "translateX(0)" }} />
      </button>
    </div>
  );

  return (
    <div className="space-y-6 max-w-2xl">
      <div>
        <h1 className="text-2xl font-bold text-white">Settings</h1>
        <p className="text-sm text-slate-400 mt-1">Configure your detection preferences</p>
      </div>

      <Card className="p-6">
        <h3 className="text-sm font-medium text-white mb-4">General</h3>
        <div className="divide-y divide-slate-800">
          <Toggle checked={darkMode} onChange={setDarkMode} label="Dark Mode" />
          <Toggle checked={notifications} onChange={setNotifications} label="Push Notifications" />
          <Toggle checked={autoAnalyze} onChange={setAutoAnalyze} label="Auto-analyze on Upload" />
        </div>
      </Card>

      <Card className="p-6">
        <h3 className="text-sm font-medium text-white mb-4">Detection Pipeline</h3>
        <div className="space-y-3">
          {["Face Detection", "Manipulation Detection", "Frequency Analysis", "GAN Detection", "Noise Analysis", "Compression Analysis", "Metadata Analysis", "Audio Analysis", "Lip Sync Detection"].map((name, i) => (
            <Toggle key={i} checked={true} onChange={() => {}} label={name} />
          ))}
        </div>
      </Card>

      <Card className="p-6">
        <h3 className="text-sm font-medium text-white mb-4">Ensemble Weights</h3>
        <div className="space-y-4">
          {[
            ["Face Manipulation", 0.30], ["Frequency Analysis", 0.15], ["GAN Detection", 0.15],
            ["Temporal Consistency", 0.15], ["Noise Analysis", 0.10], ["Compression", 0.05],
            ["Metadata", 0.05], ["Lip Sync", 0.05],
          ].map(([name, weight]) => (
            <div key={name}>
              <div className="flex justify-between text-xs text-slate-400 mb-1">
                <span>{name}</span><span>{(weight * 100).toFixed(0)}%</span>
              </div>
              <div className="w-full h-2 bg-slate-900 rounded-full">
                <div className="h-full bg-blue-500 rounded-full" style={{ width: `${weight * 100}%` }} />
              </div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
};

// ============================================================================
// Main Application
// ============================================================================

const NAV_ITEMS = [
  { id: "dashboard", label: "Dashboard", icon: "📊" },
  { id: "upload", label: "Upload", icon: "🔬" },
  { id: "history", label: "History", icon: "📋" },
  { id: "models", label: "Models", icon: "🧠" },
  { id: "settings", label: "Settings", icon: "⚙️" },
];

export default function DeepFakeDetectorApp() {
  const [page, setPage] = useState("dashboard");
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const renderPage = () => {
    switch (page) {
      case "dashboard": return <DashboardPage />;
      case "upload": return <UploadPage />;
      case "history": return <HistoryPage />;
      case "models": return <ModelsPage />;
      case "settings": return <SettingsPage />;
      default: return <DashboardPage />;
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 flex">
      {/* Sidebar */}
      <aside className={cn(
        "bg-slate-900/80 backdrop-blur-xl border-r border-slate-800/50 flex flex-col transition-all duration-300 shrink-0",
        sidebarOpen ? "w-56" : "w-16"
      )}>
        {/* Logo */}
        <div className="p-4 border-b border-slate-800/50">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center text-white font-bold text-sm shadow-lg shadow-blue-500/30">
              DF
            </div>
            {sidebarOpen && (
              <div>
                <div className="text-sm font-bold text-white leading-tight">DeepFake</div>
                <div className="text-[10px] text-blue-400 font-medium">DETECTOR v2.0</div>
              </div>
            )}
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex-1 p-3 space-y-1">
          {NAV_ITEMS.map(item => (
            <button
              key={item.id}
              onClick={() => setPage(item.id)}
              className={cn(
                "w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-all duration-200",
                page === item.id
                  ? "bg-blue-600/20 text-blue-400 border border-blue-500/30"
                  : "text-slate-400 hover:text-white hover:bg-slate-800/50"
              )}
            >
              <span className="text-lg shrink-0">{item.icon}</span>
              {sidebarOpen && <span className="font-medium">{item.label}</span>}
            </button>
          ))}
        </nav>

        {/* Sidebar toggle */}
        <div className="p-3 border-t border-slate-800/50">
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded-lg text-xs text-slate-500 hover:text-slate-300 hover:bg-slate-800/50 transition-colors"
          >
            {sidebarOpen ? "◀ Collapse" : "▶"}
          </button>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col min-w-0">
        {/* Top Bar */}
        <header className="h-14 border-b border-slate-800/50 bg-slate-900/50 backdrop-blur-sm flex items-center justify-between px-6 shrink-0">
          <div className="flex items-center gap-3">
            <span className="text-sm text-slate-400">
              {NAV_ITEMS.find(n => n.id === page)?.icon} {NAV_ITEMS.find(n => n.id === page)?.label}
            </span>
          </div>
          <div className="flex items-center gap-4">
            <button className="relative p-2 text-slate-400 hover:text-white transition-colors">
              🔔
              <span className="absolute -top-0.5 -right-0.5 w-4 h-4 bg-red-500 rounded-full text-[10px] text-white flex items-center justify-center">3</span>
            </button>
            <div className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-purple-500 flex items-center justify-center text-xs text-white font-bold">
              M
            </div>
          </div>
        </header>

        {/* Page Content */}
        <div className="flex-1 overflow-auto p-6">
          {renderPage()}
        </div>
      </main>
    </div>
  );
}
