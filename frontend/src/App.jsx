import { Shield, Scan, Zap } from "lucide-react";

export default function App() {
  return (
    <div className="min-h-screen bg-black text-white flex items-center justify-center px-6">
      <div className="w-full max-w-4xl space-y-10">

        {/* Header */}
        <div className="flex items-center justify-between bg-white/5 border border-white/10 rounded-3xl p-6 backdrop-blur-xl">
          <div className="flex items-center gap-4">
            <div className="w-14 h-14 bg-gradient-to-br from-cyan-400 to-purple-600 rounded-2xl flex items-center justify-center">
              <Shield />
            </div>
            <div>
              <h1 className="text-2xl font-bold bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent">
                PhishGuard AI
              </h1>
              <p className="text-sm text-gray-400">Neural Threat Detection</p>
            </div>
          </div>
          <span className="text-green-400 font-semibold">● AI Online</span>
        </div>

        {/* Scan Card */}
        <div className="bg-gradient-to-br from-white/10 to-white/5 border border-white/20 rounded-3xl p-8 backdrop-blur-xl shadow-xl">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-xl font-bold flex items-center gap-2">
              <Scan className="text-cyan-400" />
              Neural Scan
            </h2>
            <span className="text-purple-400 text-sm">AI Powered</span>
          </div>

          <textarea
            className="w-full h-40 bg-black/40 border border-white/20 rounded-2xl p-5 text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-cyan-500/40 resize-none"
            placeholder="Paste your message here for AI-powered threat analysis..."
          />

          <button className="mt-6 w-full py-5 rounded-2xl bg-gradient-to-r from-cyan-500 via-purple-500 to-pink-500 font-bold flex items-center justify-center gap-2 hover:scale-[1.02] transition">
            <Zap />
            Initiate Threat Scan
          </button>

          <div className="grid grid-cols-3 gap-4 mt-6">
            {["Sample 1", "Sample 2", "Sample 3"].map(s => (
              <button
                key={s}
                className="py-3 rounded-xl bg-white/5 border border-white/10 hover:bg-white/10"
              >
                {s}
              </button>
            ))}
          </div>
        </div>

      </div>
    </div>
  );
}
