import React from 'react'

export default function TacticalPitch2D({
  fov = [],
  kpts2d = [],
  players = [],
  ball = null,
  status = 'N/A',
  reprojPx = 0,
}) {
  // Broadcast FIFA pitch (105m x 68m) with stadium margin inside 600x400 viewBox
  const PAD_X = 36
  const PAD_Y = 30
  const PW = 528 // 105m
  const PH = 340 // 68m
  const sx = (x) => PAD_X + (x / 105) * PW
  const sy = (y) => PAD_Y + (y / 68) * PH
  const sr = (r) => (r / 105) * PW

  const fovPoints = fov.length >= 3 ? fov.map(([x, y]) => `${sx(x)},${sy(y)}`).join(' ') : ''

  // Exact FIFA penalty arc endpoints (radius 9.15m from penalty spot at 11m / 94m, intersecting 16.5m / 88.5m)
  const rArc = sr(9.15)
  const dyArc = Math.sqrt(9.15 * 9.15 - 5.5 * 5.5) // ~7.31m
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
      {/* Top Tactical Header */}
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
          {status === 'OK' ? `LOCKED • ${reprojPx.toFixed(1)}px` : status}
        </span>
      </div>

      {/* Broadcast SVG Pitch */}
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
          <filter id="softGlow" x="-30%" y="-30%" width="160%" height="160%">
            <feGaussianBlur stdDeviation="2.2" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
          <clipPath id="pitchClip">
            <rect x={PAD_X - 4} y={PAD_Y - 4} width={PW + 8} height={PH + 8} rx="4" />
          </clipPath>
        </defs>

        {/* Outer stadium turf */}
        <rect x="0" y="0" width="600" height="400" rx="10" fill="#102b18" />

        {/* Mowed grass stripes (12 authentic bands) */}
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

        {/* Subtle stadium lighting vignette */}
        <rect x={PAD_X} y={PAD_Y} width={PW} height={PH} fill="url(#stadiumGlow)" />

        {/* Camera FOV Cone (clipped cleanly to pitch area) */}
        {fovPoints && (
          <g clipPath="url(#pitchClip)">
            <polygon points={fovPoints} fill="url(#fovGrad)" stroke="#38bdf8" strokeWidth="1.8" strokeLinejoin="round" />
          </g>
        )}

        {/* Crisp FIFA Markings */}
        <g fill="none" stroke="rgba(241, 248, 242, 0.92)" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round">
          {/* Outer lines + halfway */}
          <rect x={PAD_X} y={PAD_Y} width={PW} height={PH} />
          <line x1={sx(52.5)} y1={PAD_Y} x2={sx(52.5)} y2={PAD_Y + PH} />
          {/* Center circle */}
          <circle cx={sx(52.5)} cy={sy(34)} r={sr(9.15)} />
          {/* Left 18m & 6m boxes + arc */}
          <rect x={sx(0)} y={sy(13.84)} width={sx(16.5) - sx(0)} height={sy(54.16) - sy(13.84)} />
          <rect x={sx(0)} y={sy(24.84)} width={sx(5.5) - sx(0)} height={sy(43.16) - sy(24.84)} />
          <path d={leftArcPath} />
          {/* Right 18m & 6m boxes + arc */}
          <rect x={sx(88.5)} y={sy(13.84)} width={sx(105) - sx(88.5)} height={sy(54.16) - sy(13.84)} />
          <rect x={sx(99.5)} y={sy(24.84)} width={sx(105) - sx(99.5)} height={sy(43.16) - sy(24.84)} />
          <path d={rightArcPath} />
          {/* Corner arcs */}
          <path d={`M ${sx(0) + 7} ${sy(0)} A 7 7 0 0 1 ${sx(0)} ${sy(0) + 7}`} />
          <path d={`M ${sx(0)} ${sy(68) - 7} A 7 7 0 0 1 ${sx(0) + 7} ${sy(68)}`} />
          <path d={`M ${sx(105) - 7} ${sy(0)} A 7 7 0 0 0 ${sx(105)} ${sy(0) + 7}`} />
          <path d={`M ${sx(105)} ${sy(68) - 7} A 7 7 0 0 0 ${sx(105) - 7} ${sy(68)}`} />
        </g>

        {/* Goal nets (depth outside goal lines) */}
        <g fill="rgba(255,255,255,0.10)" stroke="rgba(226,232,240,0.8)" strokeWidth="1.5">
          <rect x={sx(0) - 11} y={sy(30.34)} width="11" height={sy(37.66) - sy(30.34)} rx="1.5" />
          <rect x={sx(105)} y={sy(30.34)} width="11" height={sy(37.66) - sy(30.34)} rx="1.5" />
        </g>

        {/* Center & Penalty Spots */}
        <circle cx={sx(52.5)} cy={sy(34)} r="2.8" fill="#f8fafc" />
        <circle cx={sx(11)} cy={sy(34)} r="2.2" fill="#f8fafc" />
        <circle cx={sx(94)} cy={sy(34)} r="2.2" fill="#f8fafc" />

        {/* Verified 2D Pitch Keypoints */}
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

        {/* Players (Phase 2) */}
        {players.map((p) => (
          <g key={p.id} filter="url(#softGlow)">
            <circle
              cx={sx(p.x)}
              cy={sy(p.y)}
              r="7"
              fill={p.team === 'A' ? '#38bdf8' : p.team === 'B' ? '#f43f5e' : '#eab308'}
              stroke="#ffffff"
              strokeWidth="1.8"
            />
          </g>
        ))}

        {/* Ball (Phase 2) */}
        {ball && (
          <circle
            cx={sx(ball[0])}
            cy={sy(ball[1])}
            r="4.8"
            fill="#fde047"
            stroke="#0f172a"
            strokeWidth="1.6"
            filter="url(#softGlow)"
          />
        )}
      </svg>
    </div>
  )
}