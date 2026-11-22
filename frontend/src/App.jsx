import React, { useState, useRef, useEffect, useCallback } from 'react'

const API_BASE = 'http://127.0.0.1:8000'

// Native 33-point skeleton edges (Models A, B, C, D)
const SKELETON_33 = [
  [1, 2], [2, 8], [8, 7], [7, 5], [5, 6],
  [2, 3], [3, 4], [4, 5],
  [8, 9], [9, 10], [10, 7],
  [27, 31], [31, 28],
  [11, 12], [12, 33], [33, 14], [14, 16],
  [12, 13], [13, 14], [14, 15], [15, 12],
  [1, 11], [11, 22], [6, 16], [16, 17],
  [22, 21], [21, 24], [24, 26], [26, 18], [18, 17],
  [21, 20], [20, 19], [19, 18],
  [24, 23], [23, 25], [25, 26],
  [30, 32], [32, 29],
]

// Legacy 39-point skeleton edges
const SKELETON_39 = [
  [17, 18], [18, 1], [10, 9], [9, 8], [18, 9],
  [19, 21], [21, 22], [22, 20], [20, 19],
  [16, 31], [31, 35], [35, 11], [15, 36], [36, 37], [37, 12],
  [32, 33], [33, 34], [17, 16], [16, 15], [15, 12], [12, 11], [11, 10],
  [2, 23], [23, 27], [27, 7], [3, 28], [28, 29], [29, 6],
  [24, 25], [25, 26], [1, 2], [2, 3], [3, 6], [6, 7], [7, 8],
]

// Reverse map so copying predictions from Legacy 39 converts cleanly to Native 1..33 GT
const LEGACY39_TO_NATIVE33 = {
  17: 1, 16: 2, 31: 3, 35: 4, 11: 5, 10: 6, 12: 7, 15: 8, 36: 9, 37: 10,
  32: 27, 34: 28, 33: 31,
  18: 11, 19: 12, 21: 13, 22: 14, 20: 15, 9: 16, 39: 33,
  1: 22, 2: 21, 23: 20, 27: 19, 7: 18, 8: 17, 3: 24, 28: 23, 6: 26, 29: 25,
  24: 30, 26: 29, 25: 32,
}

// Human-readable names for all 33 Native FIFA pitch keypoints
const KEYPOINT_LABELS_33 = {
  1: 'P1 — Corner Haut-Gauche',
  2: 'P2 — 18m Haut-Gauche (Ligne but)',
  3: 'P3 — 18m Haut-Gauche (Intérieur)',
  4: 'P4 — 18m Bas-Gauche (Intérieur)',
  5: 'P5 — 18m Bas-Gauche (Ligne but)',
  6: 'P6 — Corner Bas-Gauche',
  7: 'P7 — 6m Bas-Gauche (Ligne but)',
  8: 'P8 — 6m Haut-Gauche (Ligne but)',
  9: 'P9 — 6m Haut-Gauche (Intérieur)',
  10: 'P10 — 6m Bas-Gauche (Intérieur)',
  11: 'P11 — Médiane Haut',
  12: 'P12 — Rond Central Haut',
  13: 'P13 — Rond Central Gauche',
  14: 'P14 — Rond Central Bas',
  15: 'P15 — Rond Central Droite',
  16: 'P16 — Médiane Bas',
  17: 'P17 — Corner Bas-Droite',
  18: 'P18 — 18m Bas-Droite (Ligne but)',
  19: 'P19 — 18m Bas-Droite (Intérieur)',
  20: 'P20 — 18m Haut-Droite (Intérieur)',
  21: 'P21 — 18m Haut-Droite (Ligne but)',
  22: 'P22 — Corner Haut-Droite',
  23: 'P23 — 6m Haut-Droite (Intérieur)',
  24: 'P24 — 6m Haut-Droite (Ligne but)',
  25: 'P25 — 6m Bas-Droite (Intérieur)',
  26: 'P26 — 6m Bas-Droite (Ligne but)',
  27: 'P27 — Arc 18m Gauche (Haut)',
  28: 'P28 — Arc 18m Gauche (Bas)',
  29: 'P29 — Arc 18m Droite (Bas)',
  30: 'P30 — Arc 18m Droite (Haut)',
  31: 'P31 — Sommet Arc Gauche',
  32: 'P32 — Sommet Arc Droite',
  33: 'P33 — Point Central',
}

// ============================================================================
// INLINE 2D TACTICAL PITCH RADAR (FIFA 105m x 68m)
// ============================================================================
function TacticalPitch2D({
  fov = [],
  kpts2d = [],
  players = [],
  ball = null,
  status = 'N/A',
  reprojPx = 0,
}) {
  const PAD_X = 36
  const PAD_Y = 30
  const PW = 528
  const PH = 340
  const sx = (x) => PAD_X + (x / 105) * PW
  const sy = (y) => PAD_Y + (y / 68) * PH
  const sr = (r) => (r / 105) * PW

  const fovPoints = fov.length >= 3 ? fov.map(([x, y]) => `${sx(x)},${sy(y)}`).join(' ') : ''
  const rArc = sr(9.15)
  const dyArc = Math.sqrt(9.15 * 9.15 - 5.5 * 5.5)
  const leftArcPath = `M ${sx(16.5)} ${sy(34 - dyArc)} A ${rArc} ${rArc} 0 0 1 ${sx(16.5)} ${sy(34 + dyArc)}`
  const rightArcPath = `M ${sx(88.5)} ${sy(34 - dyArc)} A ${rArc} ${rArc} 0 0 0 ${sx(88.5)} ${sy(34 + dyArc)}`
  const statusColor = status === 'OK' ? '#22c55e' : status === 'HOLD' ? '#f59e0b' : '#64748b'

  return (
    <div
      style={{
        background: 'linear-gradient(180deg, #111922 0%, #0b1118 100%)',
        border: '1px solid #1e2d3d',
        borderRadius: 14,
        padding: 12,
        boxShadow: '0 12px 32px rgba(0,0,0,0.5)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8, padding: '0 4px' }}>
        <span style={{ fontSize: 12, fontWeight: 700, letterSpacing: 1, color: '#94a3b8', textTransform: 'uppercase' }}>
          2D Tactical Radar • FIFA 105×68m
        </span>
        <span
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: 6,
            fontSize: 11,
            fontWeight: 700,
            padding: '3px 9px',
            borderRadius: 999,
            background: 'rgba(15, 23, 42, 0.9)',
            border: `1px solid ${statusColor}55`,
            color: statusColor,
          }}
        >
          <span style={{ width: 7, height: 7, borderRadius: '50%', background: statusColor }} />
          {status === 'OK' ? `LOCKED • ${Number(reprojPx || 0).toFixed(1)}px` : status}
        </span>
      </div>

      <svg viewBox="0 0 600 400" style={{ width: '100%', display: 'block', borderRadius: 10 }}>
        <defs>
          <radialGradient id="stadiumGlow" cx="50%" cy="50%" r="70%">
            <stop offset="0%" stopColor="#1e6b32" stopOpacity="0.18" />
            <stop offset="100%" stopColor="#07140b" stopOpacity="0.55" />
          </radialGradient>
          <linearGradient id="fovGrad" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#38bdf8" stopOpacity="0.32" />
            <stop offset="100%" stopColor="#0ea5e9" stopOpacity="0.10" />
          </linearGradient>
          <clipPath id="pitchClip">
            <rect x={PAD_X - 4} y={PAD_Y - 4} width={PW + 8} height={PH + 8} rx="4" />
          </clipPath>
        </defs>

        <rect x="0" y="0" width="600" height="400" rx="10" fill="#102b18" />
        {[...Array(12)].map((_, i) => (
          <rect
            key={i}
            x={PAD_X + (i * PW) / 12}
            y={PAD_Y}
            width={PW / 12 + 0.5}
            height={PH}
            fill={i % 2 === 0 ? '#1f6c34' : '#195a2b'}
          />
        ))}
        <rect x={PAD_X} y={PAD_Y} width={PW} height={PH} fill="url(#stadiumGlow)" />

        {fovPoints && (
          <g clipPath="url(#pitchClip)">
            <polygon points={fovPoints} fill="url(#fovGrad)" stroke="#38bdf8" strokeWidth="1.8" strokeLinejoin="round" />
          </g>
        )}

        <g fill="none" stroke="rgba(241, 248, 242, 0.92)" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round">
          <rect x={PAD_X} y={PAD_Y} width={PW} height={PH} />
          <line x1={sx(52.5)} y1={PAD_Y} x2={sx(52.5)} y2={PAD_Y + PH} />
          <circle cx={sx(52.5)} cy={sy(34)} r={sr(9.15)} />
          <rect x={sx(0)} y={sy(13.84)} width={sx(16.5) - sx(0)} height={sy(54.16) - sy(13.84)} />
          <rect x={sx(0)} y={sy(24.84)} width={sx(5.5) - sx(0)} height={sy(43.16) - sy(24.84)} />
          <path d={leftArcPath} />
          <rect x={sx(88.5)} y={sy(13.84)} width={sx(105) - sx(88.5)} height={sy(54.16) - sy(13.84)} />
          <rect x={sx(99.5)} y={sy(24.84)} width={sx(105) - sx(99.5)} height={sy(43.16) - sy(24.84)} />
          <path d={rightArcPath} />
          <path d={`M ${sx(0) + 7} ${sy(0)} A 7 7 0 0 1 ${sx(0)} ${sy(0) + 7}`} />
          <path d={`M ${sx(0)} ${sy(68) - 7} A 7 7 0 0 1 ${sx(0) + 7} ${sy(68)}`} />
          <path d={`M ${sx(105) - 7} ${sy(0)} A 7 7 0 0 0 ${sx(105)} ${sy(0) + 7}`} />
          <path d={`M ${sx(105)} ${sy(68) - 7} A 7 7 0 0 0 ${sx(105) - 7} ${sy(68)}`} />
        </g>

        <g fill="rgba(255,255,255,0.10)" stroke="rgba(226,232,240,0.8)" strokeWidth="1.5">
          <rect x={sx(0) - 11} y={sy(30.34)} width="11" height={sy(37.66) - sy(30.34)} rx="1.5" />
          <rect x={sx(105)} y={sy(30.34)} width="11" height={sy(37.66) - sy(30.34)} rx="1.5" />
        </g>

        <circle cx={sx(52.5)} cy={sy(34)} r="2.8" fill="#f8fafc" />
        <circle cx={sx(11)} cy={sy(34)} r="2.2" fill="#f8fafc" />
        <circle cx={sx(94)} cy={sy(34)} r="2.2" fill="#f8fafc" />

        {kpts2d.map(([x, y, conf], i) => (
          <g key={i}>
            <circle cx={sx(x)} cy={sy(y)} r="5.5" fill="rgba(34, 197, 94, 0.25)" />
            <circle
              cx={sx(x)}
              cy={sy(y)}
              r="3.2"
              fill={conf >= 0.5 ? '#4ade80' : '#facc15'}
              stroke="#052e16"
              strokeWidth="1.2"
            />
          </g>
        ))}

        {players.map((p) => (
          <circle
            key={p.id}
            cx={sx(p.x)}
            cy={sy(p.y)}
            r="7"
            fill={p.team === 'A' ? '#38bdf8' : p.team === 'B' ? '#f43f5e' : '#eab308'}
            stroke="#ffffff"
            strokeWidth="1.8"
          />
        ))}

        {ball && (
          <circle cx={sx(ball[0])} cy={sy(ball[1])} r="4.8" fill="#fde047" stroke="#0f172a" strokeWidth="1.6" />
        )}
      </svg>
    </div>
  )
}

// ============================================================================
// MAIN APPLICATION: STUDIO + USER GROUND TRUTH TEST LAB
// ============================================================================
export default function App() {
  const [activeTab, setActiveTab] = useState('studio') // 'studio' | 'testlab'

  // Models & Video state
  const [models, setModels] = useState([])
  const [selectedModel, setSelectedModel] = useState('model_D')
  const [currentFilename, setCurrentFilename] = useState('')
  const [videoUrl, setVideoUrl] = useState(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [status, setStatus] = useState('Chargement des modèles et de la vidéo par défaut...')

  // Analysis timeline state
  const [timeline, setTimeline] = useState([])
  const [schema, setSchema] = useState('native33')
  const [fps, setFps] = useState(30)
  const [curFrameIdx, setCurFrameIdx] = useState(1)
  const [curFrame, setCurFrame] = useState(null)
  const [summary, setSummary] = useState(null)
  const [lastMetrics, setLastMetrics] = useState(null)

  // Studio overlay toggles
  const [showKeypoints, setShowKeypoints] = useState(true)
  const [showWireframe, setShowWireframe] = useState(true)
  const [showOutliers, setShowOutliers] = useState(false)

  // Test Lab: User Ground Truth state { [frameStr]: { [kptIdStr]: [nx, ny] } }
  const [userGtFrames, setUserGtFrames] = useState({})
  const [gtDirty, setGtDirty] = useState(false)
  const [gtNotice, setGtNotice] = useState('')
  const [selectedKptId, setSelectedKptId] = useState('1')
  const [draggingKptId, setDraggingKptId] = useState(null)
  const [showPredsInLab, setShowPredsInLab] = useState(true)

  // Test Lab: Benchmark Leaderboard state
  const [isBenchmarking, setIsBenchmarking] = useState(false)
  const [forceRecalc, setForceRecalc] = useState(false)
  const [leaderboard, setLeaderboard] = useState([])
  const [leaderboardHasGt, setLeaderboardHasGt] = useState(false)

  const videoRef = useRef(null)
  const studioCanvasRef = useRef(null)
  const labVideoRef = useRef(null)
  const labCanvasRef = useRef(null)

  // Load saved Ground Truth for a video filename
  const fetchUserGt = useCallback(async (fname) => {
    if (!fname) return
    try {
      const res = await fetch(`${API_BASE}/api/gt?filename=${encodeURIComponent(fname)}`)
      if (res.ok) {
        const data = await res.json()
        setUserGtFrames(data.frames || {})
        setGtDirty(false)
      }
    } catch {
      // ignore
    }
  }, [])

  // Load analysis JSON returned by backend
  const loadAnalysisResponse = useCallback(async (data) => {
    setVideoUrl(data.video_url)
    setCurrentFilename(data.filename || '')
    setSummary(data.summary)
    setSchema(data.schema || 'native33')
    setLastMetrics({
      modelLabel: data.model_label,
      elapsed: data.elapsed_sec,
      avgFps: data.avg_fps,
      msPerFrame: data.ms_per_frame,
      jitterPx: data.jitter_px,
    })

    const tlRes = await fetch(data.analysis_url)
    const tlData = await tlRes.json()
    const tl = tlData.timeline || []
    setTimeline(tl)
    setFps(tlData.fps || 30)
    if (tl.length > 0) {
      setCurFrame(tl[0])
      setCurFrameIdx(tl[0].frame || 1)
    }
    setStatus(
      `[${data.model_label}] Terminé en ${data.elapsed_sec}s (${data.avg_fps} FPS • ${data.ms_per_frame}ms) | Radar OK: ${data.summary.radar_ok_pct}% | Jitter: ${data.jitter_px}px`
    )
  }, [])

  // Initial startup: load models + default video ("Video Project 2.mp4") + saved GT
  useEffect(() => {
    let active = true
    fetch(`${API_BASE}/api/models`)
      .then((r) => r.json())
      .then((d) => {
        if (!active) return
        const list = d.models || []
        setModels(list)
        if (list.length > 0) setSelectedModel(list[0].id)

        if (d.default_video?.filename && d.default_video?.video_url) {
          setCurrentFilename(d.default_video.filename)
          setVideoUrl(d.default_video.video_url)
          fetchUserGt(d.default_video.filename)
          setStatus(`Vidéo par défaut prête: "${d.default_video.filename}" — cliquez sur "▶ Analyser" ou allez dans Test Lab.`)
        } else {
          setStatus('Prêt — choisissez un modèle et chargez une vidéo MP4.')
        }
      })
      .catch(() => {
        if (active) setStatus('Erreur: Backend FastAPI non joignable sur http://127.0.0.1:8000')
      })
    return () => { active = false }
  }, [fetchUserGt])

  // Analyze the currently loaded (or default) video with selectedModel WITHOUT re-uploading
  const handleAnalyzeExisting = async (overrideModelId) => {
    const mid = overrideModelId || selectedModel
    if (!currentFilename) {
      setStatus('Aucune vidéo chargée. Chargez un fichier MP4 d’abord.')
      return
    }
    setIsAnalyzing(true)
    setStatus(`Analyse de "${currentFilename}" avec [${mid}] sur OpenVINO...`)
    try {
      const res = await fetch(
        `${API_BASE}/api/analyze-existing?filename=${encodeURIComponent(currentFilename)}&model_id=${encodeURIComponent(mid)}`,
        { method: 'POST' }
      )
      if (!res.ok) throw new Error(await res.text())
      const data = await res.json()
      await loadAnalysisResponse(data)
    } catch (err) {
      setStatus(`Erreur pendant l’analyse: ${err.message || 'vérifiez FastAPI.'}`)
    } finally {
      setIsAnalyzing(false)
    }
  }

  // Upload a new MP4 video and analyze it
  const handleUpload = async (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    setIsAnalyzing(true)
    setStatus(`Upload + Analyse de "${file.name}" avec [${selectedModel}]...`)
    const form = new FormData()
    form.append('file', file)
    try {
      const res = await fetch(`${API_BASE}/api/analyze?model_id=${encodeURIComponent(selectedModel)}`, {
        method: 'POST',
        body: form,
      })
      if (!res.ok) throw new Error(await res.text())
      const data = await res.json()
      await loadAnalysisResponse(data)
      await fetchUserGt(data.filename || file.name)
      setLeaderboard([])
    } catch (err) {
      setStatus(`Erreur upload/analyse: ${err.message || 'vérifiez FastAPI.'}`)
    } finally {
      setIsAnalyzing(false)
      e.target.value = ''
    }
  }

  // ============================================================================
  // STUDIO CANVAS OVERLAY (60fps RAF synced to video)
  // ============================================================================
  const drawStudioOverlay = useCallback((frameData) => {
    const canvas = studioCanvasRef.current
    const video = videoRef.current
    if (!canvas || !video) return
    const rect = video.getBoundingClientRect()
    if (rect.width === 0 || rect.height === 0) return

    const cw = Math.round(rect.width)
    const ch = Math.round(rect.height)
    if (canvas.width !== cw || canvas.height !== ch) {
      canvas.width = cw
      canvas.height = ch
    }
    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, cw, ch)
    if (!frameData) return

    if (showWireframe && frameData.wireframe?.length) {
      ctx.save()
      ctx.strokeStyle = 'rgba(56, 189, 248, 0.78)'
      ctx.lineWidth = 2
      for (const poly of frameData.wireframe) {
        if (poly.length < 2) continue
        ctx.beginPath()
        ctx.moveTo(poly[0][0] * cw, poly[0][1] * ch)
        for (let i = 1; i < poly.length; i++) ctx.lineTo(poly[i][0] * cw, poly[i][1] * ch)
        ctx.stroke()
      }
      ctx.restore()
    }

    if (showKeypoints && frameData.kpts_img?.length) {
      const okMap = {}
      for (const item of frameData.kpts_img) {
        const [id, nx, ny, conf, ok = 1] = item
        if (ok === 1) okMap[id] = [nx * cw, ny * ch, conf]
      }
      if (!showWireframe) {
        const edges = schema === 'native33' ? SKELETON_33 : SKELETON_39
        ctx.strokeStyle = 'rgba(56, 189, 248, 0.85)'
        ctx.lineWidth = 2
        for (const [a, b] of edges) {
          if (okMap[a] && okMap[b]) {
            ctx.beginPath()
            ctx.moveTo(okMap[a][0], okMap[a][1])
            ctx.lineTo(okMap[b][0], okMap[b][1])
            ctx.stroke()
          }
        }
      }

      ctx.font = '600 11px system-ui, monospace'
      for (const item of frameData.kpts_img) {
        const [id, nx, ny, conf, ok = 1] = item
        if (ok === 0 && !showOutliers) continue
        const x = nx * cw
        const y = ny * ch

        ctx.beginPath()
        ctx.arc(x, y, ok === 1 ? 6 : 5, 0, Math.PI * 2)
        ctx.fillStyle = ok === 1 ? 'rgba(34, 197, 94, 0.28)' : 'rgba(239, 68, 68, 0.30)'
        ctx.fill()

        ctx.beginPath()
        ctx.arc(x, y, 3.5, 0, Math.PI * 2)
        ctx.fillStyle = ok === 1 ? (conf >= 0.5 ? '#4ade80' : '#facc15') : '#ef4444'
        ctx.fill()
        ctx.strokeStyle = '#090d12'
        ctx.lineWidth = 1.2
        ctx.stroke()

        const label = `P${id}`
        const tw = ctx.measureText(label).width
        ctx.fillStyle = 'rgba(9, 13, 18, 0.72)'
        ctx.fillRect(x + 6, y - 14, tw + 6, 14)
        ctx.fillStyle = ok === 1 ? '#f8fafc' : '#fca5a5'
        ctx.fillText(label, x + 9, y - 3)
      }
    }
  }, [schema, showKeypoints, showWireframe, showOutliers])

  useEffect(() => {
    if (activeTab !== 'studio') return
    let rafId = null
    const tick = () => {
      if (videoRef.current && timeline.length > 0) {
        const idx = Math.min(timeline.length - 1, Math.max(0, Math.round(videoRef.current.currentTime * fps)))
        const frameData = timeline[idx]
        setCurFrame(frameData)
        setCurFrameIdx(frameData.frame || idx + 1)
        drawStudioOverlay(frameData)
      }
      rafId = requestAnimationFrame(tick)
    }
    rafId = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(rafId)
  }, [activeTab, timeline, fps, drawStudioOverlay])

  // ============================================================================
  // TEST LAB: ANNOTATOR LOGIC (CLICK / DRAG / COPY PREDS / SAVE GT)
  // ============================================================================
  const frameKey = String(curFrameIdx)
  const currentGtPoints = userGtFrames[frameKey] || {}

  const seekLabToFrame = useCallback((targetFrame) => {
    const maxF = timeline.length > 0 ? timeline.length : 500
    const clamped = Math.max(1, Math.min(maxF, targetFrame))
    setCurFrameIdx(clamped)
    if (timeline.length > 0) {
      const fObj = timeline[Math.min(timeline.length - 1, clamped - 1)]
      setCurFrame(fObj)
    }
    if (labVideoRef.current) {
      labVideoRef.current.pause()
      labVideoRef.current.currentTime = Math.max(0, (clamped - 1) / (fps || 30))
    }
  }, [timeline, fps])

  // Draw Test Lab interactive canvas (Model predictions in cyan + User GT in magenta/pink)
  const drawLabCanvas = useCallback(() => {
    const canvas = labCanvasRef.current
    const video = labVideoRef.current
    if (!canvas || !video) return
    const rect = video.getBoundingClientRect()
    if (rect.width === 0 || rect.height === 0) return

    const cw = Math.round(rect.width)
    const ch = Math.round(rect.height)
    if (canvas.width !== cw || canvas.height !== ch) {
      canvas.width = cw
      canvas.height = ch
    }

    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, cw, ch)

    const frameData = timeline.length > 0 ? timeline[Math.min(timeline.length - 1, Math.max(0, curFrameIdx - 1))] : null

    // 1. Draw current model predictions (dashed cyan circles) for comparison
    const predMap33 = {}
    if (showPredsInLab && frameData?.kpts_img?.length) {
      ctx.font = '600 10px system-ui, monospace'
      for (const item of frameData.kpts_img) {
        const [rawId, nx, ny] = item
        const id33 = schema === 'legacy39' ? LEGACY39_TO_NATIVE33[rawId] : rawId
        if (!id33) continue
        const x = nx * cw
        const y = ny * ch
        predMap33[String(id33)] = [x, y]

        ctx.beginPath()
        ctx.arc(x, y, 5, 0, Math.PI * 2)
        ctx.fillStyle = 'rgba(56, 189, 248, 0.35)'
        ctx.fill()
        ctx.strokeStyle = '#38bdf8'
        ctx.lineWidth = 1.4
        ctx.stroke()

        ctx.fillStyle = '#7dd3fc'
        ctx.fillText(`P${id33}`, x + 7, y + 10)
      }
    }

    // 2. Draw User Ground Truth points (bright magenta + error vector to model prediction)
    ctx.font = '700 11px system-ui, monospace'
    for (const [kidStr, coords] of Object.entries(currentGtPoints)) {
      const x = coords[0] * cw
      const y = coords[1] * ch

      // Error line between User GT and Model Prediction
      if (showPredsInLab && predMap33[kidStr]) {
        const [px, py] = predMap33[kidStr]
        ctx.beginPath()
        ctx.moveTo(x, y)
        ctx.lineTo(px, py)
        ctx.strokeStyle = 'rgba(251, 146, 60, 0.9)'
        ctx.lineWidth = 1.8
        ctx.stroke()
      }

      // Crosshair + bright GT dot
      ctx.beginPath()
      ctx.arc(x, y, draggingKptId === kidStr ? 10 : 7, 0, Math.PI * 2)
      ctx.fillStyle = 'rgba(236, 72, 153, 0.32)'
      ctx.fill()

      ctx.beginPath()
      ctx.arc(x, y, 4.2, 0, Math.PI * 2)
      ctx.fillStyle = '#ec4899'
      ctx.fill()
      ctx.strokeStyle = '#ffffff'
      ctx.lineWidth = 1.6
      ctx.stroke()

      const tag = `GT P${kidStr}`
      const tw = ctx.measureText(tag).width
      ctx.fillStyle = 'rgba(15, 23, 42, 0.85)'
      ctx.fillRect(x + 8, y - 16, tw + 8, 15)
      ctx.fillStyle = '#fbcfe8'
      ctx.fillText(tag, x + 12, y - 5)
    }
  }, [timeline, curFrameIdx, schema, showPredsInLab, currentGtPoints, draggingKptId])

  useEffect(() => {
    if (activeTab !== 'testlab') return
    drawLabCanvas()
    const onResize = () => drawLabCanvas()
    window.addEventListener('resize', onResize)
    return () => window.removeEventListener('resize', onResize)
  }, [activeTab, drawLabCanvas])

  // Copy current model predictions on this frame into User GT for fast drag-correction
  const handleCopyPredsToGt = () => {
    const frameData = timeline[Math.min(timeline.length - 1, Math.max(0, curFrameIdx - 1))]
    if (!frameData?.kpts_img?.length) {
      setGtNotice('Aucune prédiction sur cette frame. Cliquez d’abord sur "▶ Analyser".')
      return
    }
    const copied = { ...currentGtPoints }
    for (const item of frameData.kpts_img) {
      const [rawId, nx, ny] = item
      const id33 = schema === 'legacy39' ? LEGACY39_TO_NATIVE33[rawId] : rawId
      if (id33) copied[String(id33)] = [Number(nx.toFixed(4)), Number(ny.toFixed(4))]
    }
    setUserGtFrames((prev) => ({ ...prev, [frameKey]: copied }))
    setGtDirty(true)
    setGtNotice(`Prédictions copiées sur Frame ${curFrameIdx} — glissez (drag) les points mal placés puis sauvegardez!`)
  }

  const handleClearFrameGt = () => {
    setUserGtFrames((prev) => {
      const copy = { ...prev }
      delete copy[frameKey]
      return copy
    })
    setGtDirty(true)
    setGtNotice(`Points GT supprimés pour Frame ${curFrameIdx}.`)
  }

  const handleRemoveSingleGt = (kidStr) => {
    setUserGtFrames((prev) => {
      const cur = { ...(prev[frameKey] || {}) }
      delete cur[kidStr]
      const next = { ...prev }
      if (Object.keys(cur).length === 0) delete next[frameKey]
      else next[frameKey] = cur
      return next
    })
    setGtDirty(true)
  }

  // Canvas pointer events: drag existing GT point OR click to place selectedKptId
  const getNormalizedPointer = (e) => {
    const rect = labCanvasRef.current.getBoundingClientRect()
    const nx = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width))
    const ny = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height))
    return { nx, ny, px: e.clientX - rect.left, py: e.clientY - rect.top, w: rect.width, h: rect.height }
  }

  const handleLabMouseDown = (e) => {
    if (!labCanvasRef.current) return
    const { nx, ny, px, py, w, h } = getNormalizedPointer(e)

    // Check if clicking near an existing GT point (within 14px)
    let hitId = null
    let bestDist = 14
    for (const [kidStr, coords] of Object.entries(currentGtPoints)) {
      const d = Math.hypot(coords[0] * w - px, coords[1] * h - py)
      if (d < bestDist) {
        bestDist = d
        hitId = kidStr
      }
    }

    // Right-click deletes the point under cursor
    if (e.button === 2) {
      if (hitId) handleRemoveSingleGt(hitId)
      return
    }

    if (hitId) {
      setDraggingKptId(hitId)
      setSelectedKptId(hitId)
    } else {
      // Place (or move) the currently selected keypoint ID
      setUserGtFrames((prev) => ({
        ...prev,
        [frameKey]: {
          ...(prev[frameKey] || {}),
          [selectedKptId]: [Number(nx.toFixed(4)), Number(ny.toFixed(4))],
        },
      }))
      setDraggingKptId(selectedKptId)
      setGtDirty(true)
    }
  }

  const handleLabMouseMove = (e) => {
    if (!draggingKptId || !labCanvasRef.current) return
    const { nx, ny } = getNormalizedPointer(e)
    setUserGtFrames((prev) => ({
      ...prev,
      [frameKey]: {
        ...(prev[frameKey] || {}),
        [draggingKptId]: [Number(nx.toFixed(4)), Number(ny.toFixed(4))],
      },
    }))
    setGtDirty(true)
  }

  const handleLabMouseUp = () => {
    if (draggingKptId) setDraggingKptId(null)
  }

  const handleSaveGt = async () => {
    if (!currentFilename) return
    setGtNotice('Sauvegarde du Ground Truth...')
    try {
      const res = await fetch(`${API_BASE}/api/gt?filename=${encodeURIComponent(currentFilename)}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ frames: userGtFrames }),
      })
      if (!res.ok) throw new Error('Erreur sauvegarde GT')
      const data = await res.json()
      setGtDirty(false)
      setGtNotice(`✅ Sauvegardé (${data.frames_count} frames annotées) pour "${currentFilename}"!`)
    } catch {
      setGtNotice('Erreur: impossible de sauvegarder le GT.')
    }
  }

  // Run Benchmark across ALL available OpenVINO models
  const handleRunBenchmarkAll = async () => {
    if (!currentFilename) return
    if (gtDirty) await handleSaveGt()
    setIsBenchmarking(true)
    setGtNotice('⚡ Benchmark en cours sur tous les modèles OpenVINO (patientez quelques secondes)...')
    try {
      const res = await fetch(
        `${API_BASE}/api/benchmark-all?filename=${encodeURIComponent(currentFilename)}&force=${forceRecalc ? 'true' : 'false'}`,
        { method: 'POST' }
      )
      if (!res.ok) throw new Error(await res.text())
      const data = await res.json()
      setLeaderboard(data.leaderboard || [])
      setLeaderboardHasGt(Boolean(data.has_user_gt))
      setGtNotice(
        data.has_user_gt
          ? `🏆 Benchmark terminé! Classement basé sur VOTRE Ground Truth (${data.leaderboard[0]?.gt_pts_total || 0} points).`
          : '✅ Benchmark Vitesse & Stabilité terminé! (Annotez au moins 1 frame GT pour classer par précision réelle.)'
      )
    } catch (err) {
      setGtNotice(`Erreur benchmark: ${err.message || 'vérifiez FastAPI.'}`)
    } finally {
      setIsBenchmarking(false)
    }
  }

  // Inspect a specific model from the benchmark table directly in Studio
  const handleInspectModelFromLeaderboard = async (row) => {
    setSelectedModel(row.model_id)
    setSchema(row.schema || 'native33')
    const tlRes = await fetch(row.analysis_url)
    const tlData = await tlRes.json()
    const tl = tlData.timeline || []
    setTimeline(tl)
    setFps(tlData.fps || 30)
    setSummary({ radar_ok_pct: row.radar_ok_pct, total_frames: tl.length })
    if (tl.length > 0) setCurFrame(tl[0])
    setStatus(`Chargé depuis Benchmark: [${row.model_label}] (${row.avg_fps} FPS | Radar OK: ${row.radar_ok_pct}%)`)
    setActiveTab('studio')
  }

  const annotatedFrameList = Object.keys(userGtFrames)
    .filter((k) => Object.keys(userGtFrames[k] || {}).length > 0)
    .map((k) => Number(k))
    .sort((a, b) => a - b)

  const totalGtPoints = annotatedFrameList.reduce(
    (acc, f) => acc + Object.keys(userGtFrames[String(f)] || {}).length,
    0
  )

  return (
    <div style={{ minHeight: '100vh', background: '#0b1116', color: '#e6edf3', fontFamily: 'system-ui, -apple-system, sans-serif' }}>
      <div style={{ maxWidth: 1380, margin: '0 auto', padding: '18px 22px' }}>
        {/* ================= TOP HEADER BAR ================= */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 12, marginBottom: 14 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 14, flexWrap: 'wrap' }}>
            <h1 style={{ margin: 0, fontSize: 22, fontWeight: 800, color: '#38bdf8', letterSpacing: 0.4 }}>
              PitchVision Studio
            </h1>

            {/* Tab Switcher */}
            <div style={{ display: 'inline-flex', background: '#131c26', border: '1px solid #1e2d3d', borderRadius: 10, padding: 3 }}>
              <button
                onClick={() => setActiveTab('studio')}
                style={{
                  background: activeTab === 'studio' ? '#0284c7' : 'transparent',
                  color: activeTab === 'studio' ? '#fff' : '#94a3b8',
                  border: 'none',
                  borderRadius: 7,
                  padding: '6px 14px',
                  fontSize: 13,
                  fontWeight: 700,
                  cursor: 'pointer',
                }}
              >
                🎬 Studio & Radar 2D
              </button>
              <button
                onClick={() => {
                  setActiveTab('testlab')
                  setTimeout(() => seekLabToFrame(curFrameIdx), 60)
                }}
                style={{
                  background: activeTab === 'testlab' ? '#db2777' : 'transparent',
                  color: activeTab === 'testlab' ? '#fff' : '#94a3b8',
                  border: 'none',
                  borderRadius: 7,
                  padding: '6px 14px',
                  fontSize: 13,
                  fontWeight: 700,
                  cursor: 'pointer',
                }}
              >
                🧪 Test Lab (Mon Benchmark)
              </button>
            </div>
          </div>

          {/* Model Picker + Analyze Button + Upload New Video */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              style={{
                background: '#131c26',
                color: '#e2e8f0',
                border: '1px solid #334155',
                borderRadius: 8,
                padding: '8px 12px',
                fontSize: 13,
                fontWeight: 600,
                cursor: 'pointer',
              }}
            >
              {models.length > 0 ? (
                models.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.label}
                  </option>
                ))
              ) : (
                <option value="model_D">Model D (Default)</option>
              )}
            </select>

            <button
              onClick={() => handleAnalyzeExisting()}
              disabled={isAnalyzing || !currentFilename}
              style={{
                background: isAnalyzing ? '#334155' : '#0284c7',
                color: '#fff',
                border: 'none',
                borderRadius: 8,
                padding: '9px 16px',
                fontSize: 13,
                fontWeight: 700,
                cursor: isAnalyzing || !currentFilename ? 'not-allowed' : 'pointer',
              }}
            >
              {isAnalyzing ? '⏳ Analyse...' : '▶ Analyser'}
            </button>

            <label
              style={{
                background: '#16a34a',
                color: '#fff',
                padding: '9px 14px',
                borderRadius: 8,
                cursor: 'pointer',
                fontWeight: 600,
                fontSize: 13,
              }}
            >
              + Autre Vidéo (MP4)
              <input type="file" accept="video/mp4" onChange={handleUpload} style={{ display: 'none' }} />
            </label>
          </div>
        </div>

        {/* Status Bar */}
        <div
          style={{
            padding: '10px 14px',
            background: '#131c26',
            border: '1px solid #1e2d3d',
            borderRadius: 8,
            marginBottom: 16,
            fontSize: 13,
            color: '#cbd5e1',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            flexWrap: 'wrap',
            gap: 10,
          }}
        >
          <span>{status}</span>
          {currentFilename && (
            <span style={{ color: '#38bdf8', fontWeight: 600, fontSize: 12 }}>
              📹 Vidéo active: {currentFilename}
            </span>
          )}
        </div>

        {/* ================= TAB 1: STUDIO & RADAR 2D ================= */}
        {activeTab === 'studio' && (
          <>
            <div style={{ display: 'flex', gap: 16, marginBottom: 12, flexWrap: 'wrap' }}>
              <label style={{ fontSize: 13, cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 6, color: '#cbd5e1' }}>
                <input type="checkbox" checked={showWireframe} onChange={(e) => setShowWireframe(e.target.checked)} />
                Grille FIFA
              </label>
              <label style={{ fontSize: 13, cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 6, color: '#cbd5e1' }}>
                <input type="checkbox" checked={showKeypoints} onChange={(e) => setShowKeypoints(e.target.checked)} />
                Keypoints
              </label>
              <label style={{ fontSize: 13, cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 6, color: '#fca5a5' }}>
                <input type="checkbox" checked={showOutliers} onChange={(e) => setShowOutliers(e.target.checked)} />
                Outliers
              </label>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1.35fr 1fr', gap: 20, alignItems: 'start' }}>
              <div style={{ position: 'relative', borderRadius: 12, overflow: 'hidden', border: '1px solid #1e2d3d', background: '#000' }}>
                {videoUrl ? (
                  <>
                    <video ref={videoRef} src={videoUrl} controls style={{ width: '100%', display: 'block' }} />
                    <canvas ref={studioCanvasRef} style={{ position: 'absolute', top: 0, left: 0, pointerEvents: 'none' }} />
                  </>
                ) : (
                  <div style={{ height: 380, display: 'grid', placeItems: 'center', color: '#64748b', fontSize: 14 }}>
                    Video Match Preview
                  </div>
                )}
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                <TacticalPitch2D
                  fov={curFrame?.fov || []}
                  kpts2d={curFrame?.kpts_2d || []}
                  players={curFrame?.players || []}
                  ball={curFrame?.ball || null}
                  status={curFrame?.radar_status || 'N/A'}
                  reprojPx={curFrame?.reproj_px || 0}
                />

                {summary && (
                  <div
                    style={{
                      padding: 14,
                      background: '#131c26',
                      border: '1px solid #1e2d3d',
                      borderRadius: 12,
                      fontSize: 13,
                      display: 'grid',
                      gridTemplateColumns: 'repeat(3, 1fr)',
                      gap: 8,
                      color: '#cbd5e1',
                    }}
                  >
                    <div><strong>Radar OK:</strong> {summary.radar_ok_pct}%</div>
                    <div><strong>Inliers:</strong> {curFrame?.inliers ?? 0}</div>
                    <div><strong>Frame:</strong> {curFrame?.frame ?? 0} / {summary.total_frames}</div>
                    {lastMetrics && (
                      <>
                        <div><strong>Vitesse:</strong> {lastMetrics.avgFps} FPS</div>
                        <div><strong>Latence:</strong> {lastMetrics.msPerFrame} ms</div>
                        <div><strong>Jitter:</strong> {lastMetrics.jitterPx} px</div>
                      </>
                    )}
                  </div>
                )}
              </div>
            </div>
          </>
        )}

        {/* ================= TAB 2: TEST LAB (USER GT + BENCHMARK) ================= */}
        {activeTab === 'testlab' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
            {/* Top Benchmark Action Bar */}
            <div
              style={{
                padding: 14,
                background: '#131c26',
                border: '1px solid #1e2d3d',
                borderRadius: 12,
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                flexWrap: 'wrap',
                gap: 12,
              }}
            >
              <div>
                <div style={{ fontWeight: 700, fontSize: 15, color: '#f8fafc' }}>
                  Mon Ground Truth pour "{currentFilename || 'Aucune vidéo'}"
                </div>
                <div style={{ fontSize: 12, color: '#94a3b8', marginTop: 2 }}>
                  {annotatedFrameList.length} frame(s) annotée(s) • {totalGtPoints} point(s) GT sauvegardés (partagés par TOUS les modèles).
                </div>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
                <label style={{ fontSize: 12, color: '#cbd5e1', display: 'flex', alignItems: 'center', gap: 5, cursor: 'pointer' }}>
                  <input type="checkbox" checked={forceRecalc} onChange={(e) => setForceRecalc(e.target.checked)} />
                  Forcer recalcul (sans cache)
                </label>

                <button
                  onClick={handleSaveGt}
                  disabled={!currentFilename}
                  style={{
                    background: gtDirty ? '#db2777' : '#1e293b',
                    color: '#fff',
                    border: '1px solid #475569',
                    borderRadius: 8,
                    padding: '9px 14px',
                    fontSize: 13,
                    fontWeight: 700,
                    cursor: 'pointer',
                  }}
                >
                  💾 Sauvegarder GT {gtDirty ? '*' : ''}
                </button>

                <button
                  onClick={handleRunBenchmarkAll}
                  disabled={isBenchmarking || !currentFilename}
                  style={{
                    background: isBenchmarking ? '#334155' : 'linear-gradient(90deg, #0284c7, #7c3aed)',
                    color: '#fff',
                    border: 'none',
                    borderRadius: 8,
                    padding: '9px 16px',
                    fontSize: 13,
                    fontWeight: 700,
                    cursor: isBenchmarking ? 'not-allowed' : 'pointer',
                  }}
                >
                  {isBenchmarking ? '⏳ Benchmark en cours...' : '⚡ Benchmark Tous les Modèles'}
                </button>
              </div>
            </div>

            {gtNotice && (
              <div style={{ padding: '9px 14px', background: '#172030', border: '1px solid #334155', borderRadius: 8, fontSize: 13, color: '#e2e8f0' }}>
                {gtNotice}
              </div>
            )}

            {/* Interactive Annotator Grid */}
            <div style={{ display: 'grid', gridTemplateColumns: '1.5fr 1fr', gap: 18, alignItems: 'start' }}>
              {/* Left: Paused Video + Interactive Drag/Click Canvas */}
              <div>
                <div style={{ position: 'relative', borderRadius: 12, overflow: 'hidden', border: '1px solid #334155', background: '#000' }}>
                  {videoUrl ? (
                    <>
                      <video
                        ref={labVideoRef}
                        src={videoUrl}
                        muted
                        playsInline
                        onLoadedMetadata={() => seekLabToFrame(curFrameIdx)}
                        onSeeked={drawLabCanvas}
                        style={{ width: '100%', display: 'block' }}
                      />
                      <canvas
                        ref={labCanvasRef}
                        onMouseDown={handleLabMouseDown}
                        onMouseMove={handleLabMouseMove}
                        onMouseUp={handleLabMouseUp}
                        onMouseLeave={handleLabMouseUp}
                        onContextMenu={(e) => e.preventDefault()}
                        style={{
                          position: 'absolute',
                          top: 0,
                          left: 0,
                          cursor: draggingKptId ? 'grabbing' : 'crosshair',
                        }}
                      />
                    </>
                  ) : (
                    <div style={{ height: 360, display: 'grid', placeItems: 'center', color: '#64748b' }}>
                      Chargez une vidéo pour annoter
                    </div>
                  )}
                </div>

                {/* Frame Stepper Controls */}
                <div
                  style={{
                    marginTop: 10,
                    padding: 12,
                    background: '#131c26',
                    border: '1px solid #1e2d3d',
                    borderRadius: 10,
                    display: 'flex',
                    alignItems: 'center',
                    gap: 8,
                    flexWrap: 'wrap',
                  }}
                >
                  <button onClick={() => seekLabToFrame(curFrameIdx - 10)} style={btnMini}>« -10</button>
                  <button onClick={() => seekLabToFrame(curFrameIdx - 1)} style={btnMini}>‹ -1</button>
                  <span style={{ fontSize: 13, fontWeight: 700, minWidth: 115, textAlign: 'center', color: '#38bdf8' }}>
                    Frame {curFrameIdx} / {timeline.length || '?'}
                  </span>
                  <button onClick={() => seekLabToFrame(curFrameIdx + 1)} style={btnMini}>+1 ›</button>
                  <button onClick={() => seekLabToFrame(curFrameIdx + 10)} style={btnMini}>+10 »</button>

                  <input
                    type="range"
                    min={1}
                    max={Math.max(1, timeline.length || 200)}
                    value={curFrameIdx}
                    onChange={(e) => seekLabToFrame(Number(e.target.value))}
                    style={{ flex: 1, minWidth: 140 }}
                  />
                </div>
              </div>

              {/* Right: Annotation Toolbox */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                <div style={{ padding: 14, background: '#131c26', border: '1px solid #1e2d3d', borderRadius: 12 }}>
                  <div style={{ fontWeight: 700, fontSize: 14, marginBottom: 8, color: '#fbcfe8' }}>
                    Outils d’Annotation (Frame {curFrameIdx})
                  </div>
                  <div style={{ fontSize: 12, color: '#94a3b8', marginBottom: 10, lineHeight: 1.5 }}>
                    • <strong>Clic gauche</strong> : placer le point sélectionné.<br />
                    • <strong>Glisser (Drag)</strong> : déplacer un point GT rose.<br />
                    • <strong>Clic droit</strong> sur un point : supprimer.
                  </div>

                  <div style={{ display: 'flex', gap: 8, marginBottom: 10, flexWrap: 'wrap' }}>
                    <button
                      onClick={handleCopyPredsToGt}
                      style={{
                        flex: 1,
                        background: '#0f766e',
                        color: '#fff',
                        border: 'none',
                        borderRadius: 8,
                        padding: '8px 10px',
                        fontSize: 12,
                        fontWeight: 700,
                        cursor: 'pointer',
                      }}
                    >
                      ✨ Copier Prédictions du Modèle
                    </button>
                    <button
                      onClick={handleClearFrameGt}
                      style={{
                        background: '#7f1d1d',
                        color: '#fecaca',
                        border: 'none',
                        borderRadius: 8,
                        padding: '8px 10px',
                        fontSize: 12,
                        fontWeight: 700,
                        cursor: 'pointer',
                      }}
                    >
                      🗑️ Vider Frame
                    </button>
                  </div>

                  <label style={{ display: 'block', fontSize: 12, color: '#cbd5e1', marginBottom: 4 }}>
                    Point à placer au clic (1..33) :
                  </label>
                  <select
                    value={selectedKptId}
                    onChange={(e) => setSelectedKptId(e.target.value)}
                    style={{
                      width: '100%',
                      background: '#0b1116',
                      color: '#f8fafc',
                      border: '1px solid #334155',
                      borderRadius: 8,
                      padding: '8px 10px',
                      fontSize: 12,
                      marginBottom: 10,
                    }}
                  >
                    {Object.entries(KEYPOINT_LABELS_33).map(([id, lbl]) => (
                      <option key={id} value={id}>
                        {lbl}
                      </option>
                    ))}
                  </select>

                  <label style={{ fontSize: 12, color: '#7dd3fc', display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer', marginBottom: 10 }}>
                    <input type="checkbox" checked={showPredsInLab} onChange={(e) => setShowPredsInLab(e.target.checked)} />
                    Afficher prédictions modèle (Cyan) + lignes d’erreur (Orange)
                  </label>

                  {/* Current Frame GT Chips */}
                  <div style={{ fontSize: 12, color: '#94a3b8', marginBottom: 6 }}>
                    Points GT sur Frame {curFrameIdx} ({Object.keys(currentGtPoints).length}) :
                  </div>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, maxHeight: 130, overflowY: 'auto' }}>
                    {Object.keys(currentGtPoints).length === 0 ? (
                      <span style={{ fontSize: 12, color: '#64748b' }}>Aucun point GT sur cette frame.</span>
                    ) : (
                      Object.keys(currentGtPoints)
                        .sort((a, b) => Number(a) - Number(b))
                        .map((kid) => (
                          <span
                            key={kid}
                            onClick={() => setSelectedKptId(kid)}
                            style={{
                              display: 'inline-flex',
                              alignItems: 'center',
                              gap: 5,
                              background: selectedKptId === kid ? '#be185d' : '#1e293b',
                              color: '#f8fafc',
                              padding: '3px 8px',
                              borderRadius: 999,
                              fontSize: 11,
                              fontWeight: 700,
                              cursor: 'pointer',
                            }}
                          >
                            P{kid}
                            <button
                              onClick={(e) => {
                                e.stopPropagation()
                                handleRemoveSingleGt(kid)
                              }}
                              style={{ background: 'none', border: 'none', color: '#fda4af', cursor: 'pointer', padding: 0, fontWeight: 800 }}
                            >
                              ×
                            </button>
                          </span>
                        ))
                    )}
                  </div>
                </div>

                {/* Jump to Annotated Frames */}
                <div style={{ padding: 14, background: '#131c26', border: '1px solid #1e2d3d', borderRadius: 12 }}>
                  <div style={{ fontWeight: 700, fontSize: 13, marginBottom: 8, color: '#cbd5e1' }}>
                    Frames déjà annotées ({annotatedFrameList.length})
                  </div>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                    {annotatedFrameList.length === 0 ? (
                      <span style={{ fontSize: 12, color: '#64748b' }}>
                        Astuce: choisissez 3 à 5 frames variées (gauche, centre, droite), cliquez "✨ Copier Prédictions", corrigez puis sauvegardez.
                      </span>
                    ) : (
                      annotatedFrameList.map((fNum) => (
                        <button
                          key={fNum}
                          onClick={() => seekLabToFrame(fNum)}
                          style={{
                            background: fNum === curFrameIdx ? '#db2777' : '#1e293b',
                            color: '#fff',
                            border: '1px solid #334155',
                            borderRadius: 6,
                            padding: '4px 9px',
                            fontSize: 12,
                            cursor: 'pointer',
                          }}
                        >
                          F{fNum} ({Object.keys(userGtFrames[String(fNum)] || {}).length} pts)
                        </button>
                      ))
                    )}
                  </div>
                </div>
              </div>
            </div>

            {/* ================= LEADERBOARD TABLE ================= */}
            {leaderboard.length > 0 && (
              <div style={{ padding: 16, background: '#131c26', border: '1px solid #1e2d3d', borderRadius: 12, overflowX: 'auto' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
                  <h2 style={{ margin: 0, fontSize: 16, color: '#38bdf8' }}>
                    🏆 Classement Benchmark — {leaderboardHasGt ? 'Jugé sur VOTRE Ground Truth' : 'Vitesse & Stabilité'}
                  </h2>
                  <span style={{ fontSize: 12, color: '#94a3b8' }}>
                    Erreur mesurée en px @ 320px (plus bas = plus précis)
                  </span>
                </div>

                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid #334155', color: '#94a3b8', textAlign: 'left' }}>
                      <th style={thStyle}>#</th>
                      <th style={thStyle}>Modèle</th>
                      <th style={thStyle}>Vitesse (FPS)</th>
                      <th style={thStyle}>Latence</th>
                      <th style={thStyle}>Mon GT Err</th>
                      <th style={thStyle}>PCK@5px</th>
                      <th style={thStyle}>PCK@10px</th>
                      <th style={thStyle}>Recall GT</th>
                      <th style={thStyle}>Jitter (Stab.)</th>
                      <th style={thStyle}>Radar OK</th>
                      <th style={thStyle}>Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {leaderboard.map((row, idx) => (
                      <tr
                        key={row.model_id}
                        style={{
                          borderBottom: '1px solid #1e293b',
                          background: idx === 0 ? 'rgba(34, 197, 94, 0.08)' : 'transparent',
                        }}
                      >
                        <td style={{ ...tdStyle, fontWeight: 800, color: idx === 0 ? '#4ade80' : '#cbd5e1' }}>
                          {idx === 0 ? '🏆 #1' : `#${idx + 1}`}
                        </td>
                        <td style={{ ...tdStyle, fontWeight: 700, color: '#f8fafc' }}>{row.model_label}</td>
                        <td style={{ ...tdStyle, color: '#38bdf8', fontWeight: 700 }}>{row.avg_fps} FPS</td>
                        <td style={tdStyle}>{row.ms_per_frame} ms</td>
                        <td style={{ ...tdStyle, fontWeight: 800, color: row.gt_mean_px != null ? '#f472b6' : '#64748b' }}>
                          {row.gt_mean_px != null ? `${row.gt_mean_px} px` : '—'}
                        </td>
                        <td style={tdStyle}>{row.gt_pck5_pct != null ? `${row.gt_pck5_pct}%` : '—'}</td>
                        <td style={tdStyle}>{row.gt_pck10_pct != null ? `${row.gt_pck10_pct}%` : '—'}</td>
                        <td style={tdStyle}>
                          {row.gt_recall_pct != null ? `${row.gt_recall_pct}% (${row.gt_pts_matched}/${row.gt_pts_total})` : '—'}
                        </td>
                        <td style={tdStyle}>{row.jitter_px} px</td>
                        <td style={{ ...tdStyle, color: row.radar_ok_pct >= 90 ? '#4ade80' : '#facc15', fontWeight: 700 }}>
                          {row.radar_ok_pct}%
                        </td>
                        <td style={tdStyle}>
                          <button
                            onClick={() => handleInspectModelFromLeaderboard(row)}
                            style={{
                              background: '#0284c7',
                              color: '#fff',
                              border: 'none',
                              borderRadius: 6,
                              padding: '5px 10px',
                              fontSize: 12,
                              fontWeight: 700,
                              cursor: 'pointer',
                            }}
                          >
                            Voir
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

const btnMini = {
  background: '#1e293b',
  color: '#e2e8f0',
  border: '1px solid #334155',
  borderRadius: 6,
  padding: '6px 10px',
  fontSize: 12,
  fontWeight: 700,
  cursor: 'pointer',
}

const thStyle = { padding: '10px 8px', fontWeight: 700, fontSize: 12, whiteSpace: 'nowrap' }
const tdStyle = { padding: '10px 8px', whiteSpace: 'nowrap' }