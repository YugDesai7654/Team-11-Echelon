
import { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import {
  Upload, FileVideo, FileImage, ShieldCheck, AlertTriangle, XCircle,
  ChevronDown, ChevronUp, Loader2, Send, CheckCircle2, Search, Link as LinkIcon,
  Bot, ShieldAlert, Activity, FileText, Sparkles, Zap, Shield, BarChart3
} from 'lucide-react';
import { cn } from './lib/utils';
import { motion, AnimatePresence } from 'framer-motion';

const API_URL = 'http://localhost:8000';

function App() {
  const [file, setFile] = useState(null);
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [chatMessages, setChatMessages] = useState([]);
  const [chatInput, setChatInput] = useState('');
  const [chatLoading, setChatLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('analyze');
  const [systemOnline, setSystemOnline] = useState(true);
  const resultsRef = useRef(null);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0]);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleAnalyze = async () => {
    if (!file || !text) return;
    setLoading(true);
    setResult(null);
    setChatMessages([]);

    const formData = new FormData();
    formData.append('text', text);
    formData.append('file', file);

    try {
      const response = await axios.post(`${API_URL}/analyze`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setResult(response.data);
      setTimeout(() => {
        resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }, 100);
    } catch (error) {
      console.error("Analysis failed:", error);
      alert("Analysis failed. Please check the backend.");
    } finally {
      setLoading(false);
    }
  };

  const handleChat = async (e) => {
    e.preventDefault();
    if (!chatInput.trim() || !result) return;

    const newMessages = [...chatMessages, { role: 'user', content: chatInput }];
    setChatMessages(newMessages);
    setChatInput('');
    setChatLoading(true);

    try {
      const response = await axios.post(`${API_URL}/chat`, {
        messages: newMessages,
        pipeline_result: result
      });
      setChatMessages([...newMessages, { role: 'assistant', content: response.data.response }]);
    } catch (error) {
      console.error("Chat failed:", error);
      setChatMessages([...newMessages, { role: 'assistant', content: "Sorry, I encountered an error responding to that." }]);
    } finally {
      setChatLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#0a0a0a] text-white font-sans selection:bg-emerald-500 selection:text-black">
      {/* Header / Navigation */}
      <header className="border-b border-zinc-800/50 px-8 py-4 flex items-center justify-between sticky top-0 bg-[#0a0a0a]/95 backdrop-blur-xl z-50">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 bg-zinc-800 rounded-lg flex items-center justify-center border border-zinc-700">
            <svg viewBox="0 0 24 24" className="w-5 h-5 text-emerald-400" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12 2L2 7l10 5 10-5-10-5z"/>
              <path d="M2 17l10 5 10-5"/>
              <path d="M2 12l10 5 10-5"/>
            </svg>
          </div>
          <span className="font-bold tracking-tight text-xl text-white">ECHELON</span>
        </div>
        
        {/* Navigation Tabs */}
        <nav className="hidden md:flex items-center gap-8">
          {['Analyze', 'History', 'About', 'API Docs'].map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab.toLowerCase())}
              className={cn(
                "text-sm font-medium transition-all relative py-2",
                activeTab === tab.toLowerCase() 
                  ? "text-white" 
                  : "text-zinc-500 hover:text-zinc-300"
              )}
            >
              {tab}
              {activeTab === tab.toLowerCase() && (
                <span className="absolute bottom-0 left-0 right-0 h-0.5 bg-white rounded-full" />
              )}
            </button>
          ))}
        </nav>

        {/* System Status */}
        <div className={cn(
          "flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium",
          systemOnline 
            ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20" 
            : "bg-red-500/10 text-red-400 border border-red-500/20"
        )}>
          <span className={cn(
            "w-2 h-2 rounded-full",
            systemOnline ? "bg-emerald-400 animate-pulse" : "bg-red-400"
          )} />
          {systemOnline ? "System Online" : "System Offline"}
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-12">
        {/* Hero Section */}
        <section className="text-center space-y-8 mb-16">
          {/* Powered By Badge */}
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-zinc-900/50 border border-zinc-800">
            <Sparkles className="w-4 h-4 text-amber-400" />
            <span className="text-xs font-medium text-zinc-400 tracking-wider">
              POWERED BY <span className="text-white">CLIP</span> × <span className="text-white">GEMINI</span> × <span className="text-white">VIT</span>
            </span>
          </div>

          {/* Main Title - Serif Font */}
          <h1 className="text-5xl md:text-6xl lg:text-7xl font-serif font-normal text-white leading-tight">
            Multi-Modal Misinformation<br />
            Detection System
          </h1>

          {/* Subtitle */}
          <p className="text-zinc-400 text-lg max-w-2xl mx-auto leading-relaxed">
            Verify claims with AI-powered cross-modal analysis. Our pipeline
            analyzes text-image consistency, detects synthetic media, and provides
            transparent explanations for every verdict.
          </p>

          {/* Feature Pills */}
          <div className="flex flex-wrap justify-center gap-3 mt-8">
            {[
              { icon: LinkIcon, label: 'Cross-Modal Analysis' },
              { icon: Bot, label: 'AI Detection' },
              { icon: ShieldAlert, label: 'Deepfake Scanner' },
              { icon: Shield, label: 'Robustness Check' },
              { icon: BarChart3, label: '7 Deliverables' },
            ].map((feature) => (
              <div
                key={feature.label}
                className="flex items-center gap-2 px-4 py-2 rounded-full bg-zinc-900/50 border border-zinc-800 hover:border-zinc-600 transition-colors cursor-default"
              >
                <feature.icon className="w-4 h-4 text-zinc-400" />
                <span className="text-sm text-zinc-300">{feature.label}</span>
              </div>
            ))}
          </div>
        </section>

        {/* Input Section - Two Cards */}
        <section className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-12">
          {/* Upload Evidence Card */}
          <div className="bg-zinc-900/30 border border-zinc-800 rounded-2xl p-6">
            <div className="flex items-center gap-3 mb-6">
              <div className="w-10 h-10 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-xl flex items-center justify-center">
                <Upload className="w-5 h-5 text-white" />
              </div>
              <h2 className="text-sm font-bold uppercase tracking-widest text-zinc-300">Upload Evidence</h2>
            </div>

            <div
              className="relative group"
              onDrop={handleDrop}
              onDragOver={handleDragOver}
            >
              <input
                type="file"
                onChange={handleFileChange}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
                accept="image/*,video/*"
              />
              <div className={cn(
                "h-56 border-2 border-dashed rounded-xl flex flex-col items-center justify-center transition-all duration-300",
                file 
                  ? "border-emerald-500/30 bg-emerald-500/5" 
                  : "border-zinc-700 group-hover:border-zinc-500 bg-zinc-900/50"
              )}>
                {file ? (
                  <div className="text-center p-4">
                    {file.type.startsWith('video') ? (
                      <FileVideo className="w-12 h-12 text-emerald-400 mx-auto mb-3" />
                    ) : (
                      <FileImage className="w-12 h-12 text-emerald-400 mx-auto mb-3" />
                    )}
                    <span className="block text-white font-medium truncate max-w-[250px]">{file.name}</span>
                    <span className="text-xs text-zinc-500 mt-1">{(file.size / 1024 / 1024).toFixed(2)} MB</span>
                    <div className="mt-3 flex items-center justify-center gap-1 text-emerald-400 text-sm">
                      <CheckCircle2 className="w-4 h-4" />
                      <span>Ready for analysis</span>
                    </div>
                  </div>
                ) : (
                  <div className="text-center p-4 space-y-3">
                    <div className="w-16 h-16 bg-zinc-800 rounded-xl flex items-center justify-center mx-auto">
                      <FileImage className="w-8 h-8 text-zinc-500" />
                    </div>
                    <div>
                      <span className="text-zinc-300 font-medium block">Drag & Drop Image</span>
                      <span className="text-sm text-zinc-500">or click to browse files</span>
                    </div>
                    <div className="flex justify-center gap-2 mt-2">
                      {['JPG', 'PNG', 'WebP'].map((format) => (
                        <span key={format} className="px-2 py-1 text-xs bg-zinc-800 text-zinc-500 rounded">
                          {format}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Claim to Verify Card */}
          <div className="bg-zinc-900/30 border border-zinc-800 rounded-2xl p-6">
            <div className="flex items-center gap-3 mb-6">
              <div className="w-10 h-10 bg-gradient-to-br from-amber-500 to-orange-600 rounded-xl flex items-center justify-center">
                <FileText className="w-5 h-5 text-white" />
              </div>
              <h2 className="text-sm font-bold uppercase tracking-widest text-zinc-300">Claim to Verify</h2>
            </div>

            <div className="space-y-3">
              <label className="text-xs font-medium text-zinc-500 uppercase tracking-wider">
                Enter the text or caption
              </label>
              <textarea
                className="w-full h-48 bg-zinc-900/50 border border-zinc-800 rounded-xl p-4 text-white placeholder:text-zinc-600 focus:outline-none focus:ring-2 focus:ring-zinc-700 focus:border-zinc-600 transition-all resize-none text-sm leading-relaxed"
                placeholder="Example: 'Breaking: Massive floods in Dubai today due to cloud seeding experiment gone wrong...'"
                value={text}
                onChange={(e) => setText(e.target.value)}
              />
            </div>
          </div>
        </section>

        {/* Analyze Button */}
        <div className="flex justify-center mb-16">
          <button
            onClick={handleAnalyze}
            disabled={!file || !text || loading}
            className={cn(
              "px-12 py-4 rounded-xl font-bold text-sm tracking-wider transition-all duration-300 flex items-center gap-3",
              loading
                ? "bg-zinc-800 text-zinc-500 cursor-wait"
                : !file || !text
                  ? "bg-zinc-800 text-zinc-600 cursor-not-allowed"
                  : "bg-zinc-800 hover:bg-zinc-700 text-white border border-zinc-700 hover:border-zinc-600"
            )}
          >
            {loading ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                ANALYZING...
              </>
            ) : (
              <>
                <Search className="w-5 h-5" />
                RUN FULL PIPELINE ANALYSIS
              </>
            )}
          </button>
        </div>

        {/* Loading State */}
        <AnimatePresence>
          {loading && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="max-w-3xl mx-auto mb-12"
            >
              <LoadingView />
            </motion.div>
          )}
        </AnimatePresence>

        {/* Results Section */}
        {result && !loading && (
          <div ref={resultsRef} className="space-y-12 animate-in fade-in slide-in-from-bottom-10 duration-700 ease-out">
            <ResultsView result={result} />
            <DetailedAnalysis result={result} />
            <ChatSection
              messages={chatMessages}
              input={chatInput}
              setInput={setChatInput}
              loading={chatLoading}
              onSubmit={handleChat}
            />
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-zinc-800/50 py-8 mt-16">
        <div className="max-w-7xl mx-auto px-6 flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2 text-zinc-500 text-sm">
            <span>© 2025 Echelon</span>
            <span>•</span>
            <span>Team 11</span>
          </div>
          <div className="text-xs text-zinc-600">
            Multi-Modal Misinformation Detection Pipeline
          </div>
        </div>
      </footer>
    </div>
  );
}

function LoadingView() {
  return (
    <div className="bg-zinc-900/50 border border-zinc-800 p-10 rounded-2xl flex flex-col items-center justify-center space-y-6">
      <div className="w-full max-w-xs space-y-6">
        <div className="flex justify-center">
          <div className="relative">
            <div className="w-16 h-16 border-4 border-zinc-800 border-t-emerald-400 rounded-full animate-spin"></div>
          </div>
        </div>
        <div className="space-y-3">
          <div className="h-1.5 bg-zinc-800 rounded-full overflow-hidden">
            <motion.div
              className="h-full bg-gradient-to-r from-emerald-500 to-cyan-500"
              initial={{ width: "0%" }}
              animate={{ width: "100%" }}
              transition={{ duration: 10, ease: "linear" }}
            />
          </div>
          <p className="text-center text-sm text-zinc-500 font-medium">Running 7 deliverable stages...</p>
        </div>
      </div>
    </div>
  );
}

function ResultsView({ result }) {
  const score = result.truthfulness_score;
  const verdict = result.verdict;

  let verdictColorClass = "text-white";
  let borderColorClass = "border-zinc-800";
  let bgGradient = "from-zinc-900 to-zinc-950";
  let Icon = ShieldCheck;

  if (score >= 80) {
    verdictColorClass = "text-emerald-400";
    borderColorClass = "border-emerald-500/20";
    bgGradient = "from-emerald-950/30 to-zinc-950";
  } else if (score >= 50) {
    verdictColorClass = "text-amber-400";
    borderColorClass = "border-amber-500/20";
    bgGradient = "from-amber-950/30 to-zinc-950";
    Icon = AlertTriangle;
  } else {
    verdictColorClass = "text-red-400";
    borderColorClass = "border-red-500/20";
    bgGradient = "from-red-950/30 to-zinc-950";
    Icon = XCircle;
  }

  return (
    <div className="space-y-8">
      {/* Verdict Header */}
      <div className={cn("relative rounded-3xl p-10 md:p-14 overflow-hidden border bg-gradient-to-br transition-colors", borderColorClass, bgGradient)}>
        <div className="absolute top-0 right-0 p-12 opacity-[0.03] pointer-events-none">
          <Icon className="w-96 h-96" />
        </div>

        <div className="relative z-10 grid grid-cols-1 md:grid-cols-2 gap-12 items-center">
          <div>
            <div className="inline-flex items-center gap-2 mb-6 px-4 py-2 rounded-full bg-black/40 border border-white/5">
              <span className={cn("w-2 h-2 rounded-full animate-pulse", verdictColorClass.replace('text', 'bg'))}></span>
              <span className="text-xs font-bold tracking-widest uppercase text-zinc-300">Final Verdict</span>
            </div>

            <h2 className={cn("text-5xl md:text-6xl font-black tracking-tight mb-6", verdictColorClass)}>
              {verdict.toUpperCase()}
            </h2>

            <div className="bg-black/30 border border-white/5 rounded-xl p-6">
              <p className="text-zinc-300 text-lg leading-relaxed">
                {result.explanation}
              </p>
            </div>
          </div>

          <div className="flex flex-col items-center justify-center">
            <div className="relative w-52 h-52 flex items-center justify-center group">
              <svg className="w-full h-full -rotate-90 transform" viewBox="0 0 100 100">
                <circle cx="50" cy="50" r="45" fill="none" stroke="currentColor" strokeWidth="8" className="text-zinc-800" />
                <circle
                  cx="50" cy="50" r="45"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="8"
                  strokeDasharray={`${score * 2.83} 283`}
                  strokeLinecap="round"
                  className={cn("transition-all duration-1000 ease-out", verdictColorClass)}
                />
              </svg>
              <div className="absolute inset-0 flex flex-col items-center justify-center">
                <span className="text-5xl font-bold text-white">{Math.round(score)}</span>
                <span className="text-xs text-zinc-500 uppercase tracking-widest mt-1">Trust Score</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Metrics Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        <MetricCard
          title="CLIP Consistency"
          value={result.cross_modal_score}
          subtitle="Text-Image Alignment"
          type="score"
          icon={<LinkIcon className="w-4 h-4" />}
        />
        <MetricCard
          title="AI Text Prob"
          value={result.raw_data?.ai_text_result?.raw_score || result.ai_text_probability || 0}
          subtitle="Synthetic Generation"
          type="percentage"
          invert
          icon={<FileText className="w-4 h-4" />}
        />
        <MetricCard
          title="Deepfake Prob"
          value={result.raw_data?.deepfake_probability || result.ai_image_probability || 0}
          subtitle="Visual Manipulation"
          type="percentage"
          invert
          icon={<Activity className="w-4 h-4" />}
        />
      </div>
    </div>
  );
}

function DetailedAnalysis({ result }) {
  const raw = result.raw_data || {};
  const stages = result.stage_results || {};

  const inputData = stages.input_handling?.data || {};
  const contextData = stages.context_detection?.data || {};
  const syntheticData = stages.synthetic_detection?.data || {};
  const robustnessData = stages.robustness_check?.data || {};
  const evalData = stages.evaluation?.data || {};

  return (
    <div className="space-y-6">
      <h3 className="text-2xl font-bold text-white px-2 flex items-center gap-3">
        <BarChart3 className="w-6 h-6 text-zinc-400" />
        Pipeline Analysis Details (D1-D7)
      </h3>

      <div className="grid grid-cols-1 gap-4">
        {/* D1: Input Handling */}
        <DetailCard title="D1: Multi-Modal Input Handling" icon={FileImage}>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            <DetailItem label="Modality" value={inputData.modality || "Unknown"} />
            <DetailItem label="Text Length" value={`${inputData.text_length || 0} chars`} />
            <DetailItem label="Has Image" value={inputData.has_image ? "Yes" : "No"} />
            <DetailItem label="Has Video" value={inputData.has_video ? "Yes" : "No"} />
          </div>
        </DetailCard>

        {/* D2: Cross Modal */}
        <DetailCard title="D2: Cross-Modal Inconsistency Detection" icon={LinkIcon}>
          <div className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <DetailItem
                label="Similarity Score"
                value={result.clip_score?.toFixed(4)}
                status={result.clip_score > 0.25 ? "good" : "bad"}
              />
              <DetailItem label="Analysis Type" value={raw.cross_modal_analysis?.analysis_type || "Standard"} />
              <DetailItem label="Confidence" value={`${((raw.cross_modal_analysis?.confidence || 0) * 100).toFixed(1)}%`} />
            </div>
            {raw.cross_modal_analysis?.explanation && (
              <div className="bg-zinc-950 p-4 rounded-lg border border-zinc-800 text-sm text-zinc-400">
                <span className="text-xs uppercase font-bold text-zinc-600 block mb-1">Interpretation</span>
                {raw.cross_modal_analysis.explanation}
              </div>
            )}
          </div>
        </DetailCard>

        {/* D3: Context */}
        <DetailCard title="D3: Out-of-Context Media Detection" icon={Search}>
          <div className="space-y-4">
            <div className="flex gap-8 items-center border-b border-zinc-800 pb-4">
              <div className="text-center">
                <div className="text-3xl font-mono font-bold text-white">{contextData.context_score?.toFixed(2) || 'N/A'}</div>
                <div className="text-xs uppercase text-zinc-500 tracking-wider">Context Score</div>
              </div>
              <div>
                <div className="text-lg font-bold text-white">{contextData.context_verdict || 'Unknown'}</div>
                <div className="text-sm text-zinc-400">Web search correlation verdict</div>
              </div>
            </div>

            {contextData.search_results?.results?.length > 0 ? (
              <div className="space-y-2">
                <span className="text-xs font-bold text-zinc-500 uppercase">Top Sources</span>
                <ul className="space-y-2">
                  {contextData.search_results.results.slice(0, 3).map((res, i) => (
                    <li key={i} className="bg-zinc-950/50 p-3 rounded border border-zinc-900 flex justify-between items-center group hover:border-zinc-700 transition-colors">
                      <a href={res.url} target="_blank" rel="noreferrer" className="text-sm text-cyan-400 hover:underline truncate max-w-[80%]">
                        {res.title}
                      </a>
                      <Search className="w-3 h-3 text-zinc-600 group-hover:text-white" />
                    </li>
                  ))}
                </ul>
              </div>
            ) : (
              <div className="text-sm text-zinc-500 italic">No direct web matches found.</div>
            )}

            {contextData.context_flags?.length > 0 && (
              <div className="bg-red-500/10 border border-red-500/30 p-3 rounded flex flex-col gap-2">
                {contextData.context_flags.map((flag, i) => (
                  <div key={i} className="flex items-center gap-2 text-red-400 text-sm">
                    <AlertTriangle className="w-4 h-4" /> {flag}
                  </div>
                ))}
              </div>
            )}
          </div>
        </DetailCard>

        {/* D4: Synthetic */}
        <DetailCard title="D4: Synthetic Media Detection" icon={Bot}>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div className="space-y-3">
              <h4 className="text-sm font-semibold text-zinc-300">Text Analysis</h4>
              <div className="flex justify-between items-center bg-zinc-950 p-4 rounded-lg border border-zinc-900">
                <span className="text-sm text-zinc-500">AI Probability</span>
                <span className={cn("font-mono font-bold text-lg", syntheticData.ai_text_probability > 0.5 ? "text-red-400" : "text-emerald-400")}>
                  {((syntheticData.ai_text_probability || 0) * 100).toFixed(1)}%
                </span>
              </div>
              <div className="flex justify-between items-center bg-zinc-950 p-4 rounded-lg border border-zinc-900">
                <span className="text-sm text-zinc-500">Classifier Label</span>
                <span className="text-sm text-white font-medium">{syntheticData.text_label || 'N/A'}</span>
              </div>
            </div>

            <div className="space-y-3">
              <h4 className="text-sm font-semibold text-zinc-300">Visual Analysis</h4>
              {syntheticData.video_detection ? (
                <>
                  <div className="flex justify-between items-center bg-zinc-950 p-4 rounded-lg border border-zinc-900">
                    <span className="text-sm text-zinc-500">Deepfake Prob</span>
                    <span className={cn("font-mono font-bold text-lg", syntheticData.deepfake_probability > 0.5 ? "text-red-400" : "text-emerald-400")}>
                      {((syntheticData.deepfake_probability || 0) * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="flex gap-2 justify-end">
                    {syntheticData.is_deepfake && <span className="text-xs bg-red-500 text-white px-3 py-1 rounded-full font-bold">DEEPFAKE</span>}
                  </div>
                </>
              ) : (
                <div className="flex justify-between items-center bg-zinc-950 p-4 rounded-lg border border-zinc-900">
                  <span className="text-sm text-zinc-500">AI Image Prob</span>
                  <span className={cn("font-mono font-bold text-lg", syntheticData.ai_image_probability > 0.5 ? "text-red-400" : "text-emerald-400")}>
                    {((syntheticData.ai_image_probability || 0) * 100).toFixed(1)}%
                  </span>
                </div>
              )}
            </div>
          </div>
        </DetailCard>

        {/* D5: Explanation - handled in ResultsView */}

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* D6: Robustness */}
          <DetailCard title="D6: Robustness Check" icon={ShieldCheck}>
            <div className="flex items-center justify-between mb-4">
              <span className="text-zinc-400 text-sm">Adversarial Resistance</span>
              <span className={cn("text-2xl font-bold", (robustnessData.robustness_score || 0) > 0.8 ? "text-emerald-400" : "text-amber-400")}>
                {((robustnessData.robustness_score || 0) * 100).toFixed(1)}%
              </span>
            </div>
            <div className="space-y-2">
              {robustnessData.adversarial_flags?.length > 0 ? (
                robustnessData.adversarial_flags.map((flag, i) => (
                  <div key={i} className="text-xs text-red-400 bg-red-950/30 p-3 rounded-lg flex gap-2 items-center">
                    <XCircle className="w-4 h-4" /> {flag}
                  </div>
                ))
              ) : (
                <div className="text-xs text-emerald-400 bg-emerald-950/30 p-3 rounded-lg flex gap-2 items-center">
                  <CheckCircle2 className="w-4 h-4" /> No adversarial patterns detected
                </div>
              )}
            </div>
          </DetailCard>

          {/* D7: Evaluation */}
          <DetailCard title="D7: Quantitative Evaluation" icon={Activity}>
            <div className="space-y-3">
              <MetricRow label="Explanation Quality" value={evalData.metrics?.explanation_quality} />
              <MetricRow label="Robustness Score" value={evalData.metrics?.robustness_score} />
              <MetricRow label="Overall Confidence" value={evalData.metrics?.overall_score} />
            </div>
          </DetailCard>
        </div>
      </div>
    </div>
  );
}

function DetailCard({ title, icon: Icon, children }) {
  const [isOpen, setIsOpen] = useState(true);

  return (
    <div className="bg-zinc-900/40 border border-zinc-800 rounded-xl overflow-hidden transition-all hover:border-zinc-700">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between p-5 bg-zinc-900/30 hover:bg-zinc-800/30 transition-colors"
      >
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-zinc-800 flex items-center justify-center text-zinc-400">
            <Icon className="w-4 h-4" />
          </div>
          <span className="font-semibold text-sm text-zinc-200">{title}</span>
        </div>
        {isOpen ? <ChevronUp className="w-4 h-4 text-zinc-500" /> : <ChevronDown className="w-4 h-4 text-zinc-500" />}
      </button>
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="overflow-hidden"
          >
            <div className="p-5 border-t border-zinc-800/50">
              {children}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

function ChatSection({ messages, input, setInput, loading, onSubmit }) {
  const scrollRef = useRef(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, loading]);

  return (
    <section className="border-t border-zinc-800 pt-12">
      <div className="max-w-4xl mx-auto space-y-6">
        <div className="text-center">
          <h3 className="text-2xl font-bold flex items-center justify-center gap-3">
            <Bot className="w-6 h-6 text-cyan-400" />
            Chat with Evidence
          </h3>
          <p className="text-zinc-500 text-sm mt-2">Ask questions about the analysis results</p>
        </div>

        <div className="bg-zinc-900/50 border border-zinc-800 rounded-2xl overflow-hidden flex flex-col h-[450px]">
          <div className="flex-1 overflow-y-auto p-6 space-y-4 scrollbar-thin scrollbar-thumb-zinc-700 scrollbar-track-transparent" ref={scrollRef}>
            {messages.length === 0 && (
              <div className="h-full flex flex-col items-center justify-center text-zinc-600 space-y-3">
                <ShieldCheck className="w-14 h-14 opacity-20" />
                <p className="text-sm">Ask "Why is this considered fake?" or "What evidence was found?"</p>
              </div>
            )}
            {messages.map((msg, i) => (
              <div key={i} className={cn(
                "max-w-[80%] rounded-2xl p-4 text-sm leading-relaxed",
                msg.role === 'user'
                  ? "bg-white text-black ml-auto rounded-br-sm"
                  : "bg-zinc-800 text-zinc-200 mr-auto rounded-bl-sm"
              )}>
                {msg.content}
              </div>
            ))}
            {loading && (
              <div className="bg-zinc-800 text-zinc-200 mr-auto rounded-2xl rounded-bl-sm p-4 w-20">
                <div className="flex gap-1.5">
                  <div className="w-2.5 h-2.5 bg-zinc-500 rounded-full animate-bounce"></div>
                  <div className="w-2.5 h-2.5 bg-zinc-500 rounded-full animate-bounce [animation-delay:0.1s]"></div>
                  <div className="w-2.5 h-2.5 bg-zinc-500 rounded-full animate-bounce [animation-delay:0.2s]"></div>
                </div>
              </div>
            )}
          </div>
          <form onSubmit={onSubmit} className="p-4 bg-zinc-950 border-t border-zinc-800 flex gap-3">
            <input
              className="flex-1 bg-zinc-900 border border-zinc-800 rounded-xl px-4 py-3 text-sm focus:outline-none focus:border-zinc-600 transition-all text-white placeholder-zinc-600"
              placeholder="Type your question..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
            />
            <button
              type="submit"
              disabled={loading || !input.trim()}
              className="bg-white text-black p-3 rounded-xl disabled:opacity-40 hover:bg-zinc-200 transition-colors"
            >
              <Send className="w-5 h-5" />
            </button>
          </form>
        </div>
      </div>
    </section>
  );
}

const MetricCard = ({ title, value, subtitle, type = "score", invert = false, icon }) => {
  let displayValue = typeof value === 'number' ? value.toFixed(2) : value;
  let statusColor = "text-zinc-200";

  if (typeof value === 'number') {
    if (type === "percentage") {
      displayValue = (value * 100).toFixed(1) + "%";
      const isBad = invert ? value > 0.5 : value < 0.5;
      statusColor = isBad ? "text-red-400" : "text-emerald-400";
    } else {
      const isGood = invert ? value < 0.5 : value > 0.3;
      statusColor = isGood ? "text-emerald-400" : "text-amber-400";
    }
  }

  return (
    <div className="bg-zinc-900/40 border border-zinc-800 p-6 rounded-2xl flex flex-col justify-between transition-all hover:bg-zinc-800/40 hover:border-zinc-700">
      <div className="flex justify-between items-start mb-3">
        <span className="text-zinc-500 text-xs uppercase tracking-wider font-bold">{title}</span>
        {icon && <span className="text-zinc-600">{icon}</span>}
      </div>
      <div>
        <div className={cn("text-3xl font-bold font-mono tracking-tight", statusColor)}>
          {displayValue}
        </div>
        <span className="text-zinc-600 text-xs mt-1 block">{subtitle}</span>
      </div>
    </div>
  );
};

const DetailItem = ({ label, value, status }) => {
  let color = "text-white";
  if (status === 'good') color = "text-emerald-400";
  if (status === 'bad') color = "text-red-400";

  return (
    <div>
      <span className="text-xs text-zinc-500 uppercase tracking-wider block mb-1">{label}</span>
      <span className={cn("font-medium text-sm truncate block", color)}>{value || 'N/A'}</span>
    </div>
  )
}

const MetricRow = ({ label, value }) => (
  <div className="flex items-center justify-between p-3 rounded-lg hover:bg-zinc-800/30 transition-colors">
    <span className="text-sm text-zinc-400">{label}</span>
    <span className="text-sm font-bold font-mono text-white">{((value || 0) * 100).toFixed(1)}%</span>
  </div>
)

export default App;
