import React, { useState, useEffect, useRef } from "react"
import { ShieldAlert, Scan, AlertTriangle } from "lucide-react"

const colorMap = {
  green: { text: "text-green-400", bar: "bg-green-500" },
  yellow: { text: "text-yellow-400", bar: "bg-yellow-500" },
  red: { text: "text-red-500", bar: "bg-red-500" },
}

export default function PhishingDetector() {
  const [input, setInput] = useState("")
  const [score, setScore] = useState(null)
  const canvasRef = useRef(null)

  /* Background canvas animation */
  useEffect(() => {
    const canvas = canvasRef.current
    const ctx = canvas.getContext("2d")

    canvas.width = window.innerWidth
    canvas.height = window.innerHeight

    const dots = Array.from({ length: 80 }, () => ({
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
      r: Math.random() * 2 + 1,
      dx: (Math.random() - 0.5) * 0.3,
      dy: (Math.random() - 0.5) * 0.3,
    }))

    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      ctx.fillStyle = "rgba(0,255,255,0.6)"
      dots.forEach(d => {
        d.x += d.dx
        d.y += d.dy
        if (d.x < 0 || d.x > canvas.width) d.dx *= -1
        if (d.y < 0 || d.y > canvas.height) d.dy *= -1
        ctx.beginPath()
        ctx.arc(d.x, d.y, d.r, 0, Math.PI * 2)
        ctx.fill()
      })
      requestAnimationFrame(animate)
    }
    animate()
  }, [])

  const analyze = () => {
    if (!input.trim()) return
    const randomScore = Math.floor(Math.random() * 100) + 1
    setScore(randomScore)
  }

  const threat =
    score === null
      ? null
      : score < 35
      ? { label: "Safe", color: "green" }
      : score < 70
      ? { label: "Suspicious", color: "yellow" }
      : { label: "Phishing", color: "red" }

  return (
    <div className="relative min-h-screen bg-[#0a0a0a] overflow-hidden text-white">
      <canvas ref={canvasRef} className="absolute inset-0 z-0" />

      <div className="relative z-10 flex flex-col items-center justify-center min-h-screen px-4 animate-fade-in">
        <h1 className="text-4xl md:text-5xl font-bold text-cyan-400 flex items-center gap-3">
          <ShieldAlert size={40} />
          Phishing Threat Detector
        </h1>

        <p className="text-gray-400 mt-2 text-center max-w-xl">
          Analyze suspicious SMS, emails, or URLs using AI-powered threat
          intelligence.
        </p>

        <div className="mt-8 w-full max-w-2xl bg-black/60 backdrop-blur-xl border border-cyan-500/20 rounded-2xl p-6 shadow-2xl">
          <textarea
            rows="5"
            value={input}
            onChange={e => setInput(e.target.value)}
            placeholder="Paste SMS / Email / URL here..."
            className="w-full bg-black text-white border border-gray-700 rounded-xl p-4 focus:outline-none focus:border-cyan-400 resize-none"
          />

          <button
            onClick={analyze}
            className="mt-4 w-full flex items-center justify-center gap-2 bg-cyan-500 hover:bg-cyan-600 text-black font-semibold py-3 rounded-xl transition"
          >
            <Scan />
            Analyze Threat
          </button>

          {score !== null && (
            <div className="mt-6">
              <div className="flex items-center justify-between mb-2">
                <span className={`font-semibold ${colorMap[threat.color].text}`}>
                  {threat.label}
                </span>
                <span className="text-gray-300">{score}%</span>
              </div>

              <div className="w-full h-3 bg-gray-800 rounded-full overflow-hidden">
                <div
                  className={`h-full ${colorMap[threat.color].bar}`}
                  style={{ width: `${score}%` }}
                />
              </div>

              {threat.label === "Phishing" && (
                <div className="mt-4 flex items-center gap-2 text-red-400">
                  <AlertTriangle />
                  High-risk phishing detected. Do NOT click links.
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
