'use client';

import React, { useState, useMemo, useRef, useEffect } from 'react';

// ─── Physics constants ────────────────────────────────────────────────────────
const BETA = 5 / 3;
const HYDROGRAPH = [2, 4, 7, 12, 10, 7, 5, 4, 3, 2, 2, 2]; // m³/s, 12 steps
const Q_BASE = 2;
const N_REACHES = 8;
const DEFAULT_DT = 300;
const DEFAULT_DX = 500;
const DEFAULT_N = 0.04;
const DEFAULT_S0 = 0.001;
const DEFAULT_B = 10;

function computeAlpha(manN: number, S0: number, B: number): number {
  return Math.sqrt(S0) / (manN * Math.pow(B, 2 / 3));
}
function aToQ(A: number, alpha: number): number {
  return alpha * Math.pow(Math.max(A, 0.001), BETA);
}
function qToA(Q: number, alpha: number): number {
  return Math.pow(Math.max(Q, 0.001) / alpha, 1 / BETA);
}
function cel(Q: number, A: number): number {
  return BETA * Q / Math.max(A, 0.001);
}
function courantColor(C: number): string {
  if (C <= 0.8) return '#16a34a';
  if (C <= 1.0) return '#d97706';
  return '#dc2626';
}
function qFill(Q: number, Qmax: number): string {
  const t = Math.max(0, Math.min(1, Q / Qmax));
  return `rgb(${Math.round(219 - t * 190)},${Math.round(234 - t * 175)},${Math.round(255 - t * 65)})`;
}

// ─── Simulation types & runner ────────────────────────────────────────────────
interface TimeStep {
  t: number;
  A: number[];
  Q: number[];
  Q_lim: number[];
  C: number[];
  logLines: string[];
  anyLimiterActive: boolean;
}

function runSimulation(dt: number, useLimiter: boolean, alpha: number): TimeStep[] {
  const dx = DEFAULT_DX;
  const A_base = qToA(Q_BASE, alpha);
  let A = Array(N_REACHES).fill(A_base);
  const steps: TimeStep[] = [];

  for (let t = 0; t < HYDROGRAPH.length; t++) {
    const Q = A.map(a => aToQ(a, alpha));
    const C = Q.map((q, i) => cel(q, A[i]) * dt / dx);
    const Q_upstream = [HYDROGRAPH[t], ...Q.slice(0, -1)];

    const Q_lim = Q.map((q, i) => {
      const V = A[i] * dx;
      return useLimiter ? Math.min(q, V / dt) : q;
    });
    const Q_upstream_lim = [HYDROGRAPH[t], ...Q_lim.slice(0, -1)];

    const logLines = A.map((a, i) => {
      const q_new_A = Math.max(a - (dt / dx) * (Q_lim[i] - Q_upstream_lim[i]), 0.001);
      const q_new = aToQ(q_new_A, alpha);
      return `i=${i}: A ${a.toFixed(2)}→${q_new_A.toFixed(2)} m² | Q ${Q[i].toFixed(2)}→${q_new.toFixed(2)} m³/s | C=${C[i].toFixed(2)}`;
    });

    steps.push({
      t,
      A: [...A],
      Q: [...Q],
      Q_lim: [...Q_lim],
      C: [...C],
      logLines,
      anyLimiterActive: Q_lim.some((ql, i) => ql < Q[i]),
    });

    A = A.map((a, i) => {
      const newA = a - (dt / dx) * (Q_lim[i] - Q_upstream_lim[i]);
      return Math.max(newA, 0.001);
    });
  }
  return steps;
}

// ─── Sub-component: upstream hydrograph SVG ──────────────────────────────────
function HydrographSVG({ stepIdx }: { stepIdx: number }) {
  const W = 420;
  const H = 55;
  const PAD_L = 32;
  const PAD_R = 8;
  const PAD_T = 6;
  const PAD_B = 18;
  const plotW = W - PAD_L - PAD_R;
  const plotH = H - PAD_T - PAD_B;
  const Q_MAX_HYDRO = 14;
  const n = HYDROGRAPH.length;

  const toX = (i: number) => PAD_L + (i / (n - 1)) * plotW;
  const toY = (q: number) => PAD_T + plotH - (q / Q_MAX_HYDRO) * plotH;

  const pts = HYDROGRAPH.map((q, i) => `${toX(i)},${toY(q)}`).join(' ');
  const cx = toX(stepIdx);
  const cy = toY(HYDROGRAPH[stepIdx]);

  return (
    <svg width={W} height={H} className="block">
      {/* Axes */}
      <line x1={PAD_L} y1={PAD_T} x2={PAD_L} y2={PAD_T + plotH} stroke="#94a3b8" strokeWidth={1} />
      <line x1={PAD_L} y1={PAD_T + plotH} x2={PAD_L + plotW} y2={PAD_T + plotH} stroke="#94a3b8" strokeWidth={1} />
      {/* Y axis label */}
      <text x={2} y={PAD_T + plotH / 2} fontSize={8} fill="#64748b" textAnchor="middle"
        transform={`rotate(-90,6,${PAD_T + plotH / 2})`}>Q (m³/s)</text>
      {/* Y tick at max */}
      <text x={PAD_L - 3} y={PAD_T + 4} fontSize={7} fill="#64748b" textAnchor="end">14</text>
      <text x={PAD_L - 3} y={PAD_T + plotH} fontSize={7} fill="#64748b" textAnchor="end">0</text>
      {/* X ticks */}
      {HYDROGRAPH.map((_, i) => (
        <text key={i} x={toX(i)} y={H - 3} fontSize={7} fill="#94a3b8" textAnchor="middle">{i}</text>
      ))}
      {/* Polyline */}
      <polyline points={pts} fill="none" stroke="#3b82f6" strokeWidth={1.5} />
      {/* Vertical orange line */}
      <line x1={cx} y1={PAD_T} x2={cx} y2={PAD_T + plotH} stroke="#f97316" strokeWidth={1.5} strokeDasharray="3,2" />
      {/* Blue dot */}
      <circle cx={cx} cy={cy} r={4} fill="#2563eb" stroke="white" strokeWidth={1.5} />
      {/* Label */}
      <text x={cx + 6} y={cy - 3} fontSize={8} fill="#1d4ed8" fontWeight="bold">
        Q_in = {HYDROGRAPH[stepIdx].toFixed(1)} m³/s
      </text>
      {/* Title */}
      <text x={PAD_L + plotW / 2} y={PAD_T - 1} fontSize={8} fill="#475569" textAnchor="middle">
        Upstream boundary Q_in(t)
      </text>
    </svg>
  );
}

// ─── Sub-component: space-time diagram ───────────────────────────────────────
function SpaceTimeDiagram({
  steps,
  stepIdx,
  showCharacteristic,
}: {
  steps: TimeStep[];
  stepIdx: number;
  showCharacteristic: boolean;
}) {
  const [hovered, setHovered] = useState<{ t: number; i: number } | null>(null);

  const CELL_W = 60;
  const CELL_H = 30;
  const PAD_L = 38;
  const PAD_B = 28;
  const PAD_T = 8;
  const SVG_W = PAD_L + N_REACHES * CELL_W + 4;
  const SVG_H = PAD_T + HYDROGRAPH.length * CELL_H + PAD_B;
  const Q_MAX = 14;

  const QMatrix = steps.map(s => s.Q);

  // Mean Courant of all steps/reaches
  const allC = steps.flatMap(s => s.C);
  const cMean = allC.reduce((a, b) => a + b, 0) / allC.length;
  const cMax = Math.max(...allC);

  // Characteristic: slope = 1/cMean cells per time step (horizontal)
  // x(t) = t / cMean  (in cell units)
  const nT = HYDROGRAPH.length;
  const charPoints: string[] = [];
  for (let t = 0; t <= nT - 1; t++) {
    const col = t / cMean;
    if (col > N_REACHES - 1) break;
    const x = PAD_L + col * CELL_W + CELL_W / 2;
    const y = PAD_T + (nT - 1 - t) * CELL_H + CELL_H / 2;
    charPoints.push(`${x},${y}`);
  }

  return (
    <div className="relative">
      <svg
        width={SVG_W}
        height={SVG_H}
        style={{ display: 'block', overflow: 'visible' }}
      >
        {/* Cells */}
        {QMatrix.map((row, t) =>
          row.map((q, i) => {
            const x = PAD_L + i * CELL_W;
            const y = PAD_T + (nT - 1 - t) * CELL_H;
            const fill = qFill(q, Q_MAX);
            const isCurRow = t === stepIdx;
            const isHov = hovered?.t === t && hovered?.i === i;
            return (
              <g key={`${t}-${i}`}>
                <rect
                  x={x}
                  y={y}
                  width={CELL_W}
                  height={CELL_H}
                  fill={fill}
                  stroke={isCurRow ? '#1d4ed8' : '#cbd5e1'}
                  strokeWidth={isCurRow ? 2 : 0.5}
                  onMouseEnter={() => setHovered({ t, i })}
                  onMouseLeave={() => setHovered(null)}
                  style={{ cursor: 'default' }}
                />
                <text
                  x={x + CELL_W / 2}
                  y={y + CELL_H / 2 + 4}
                  fontSize={9}
                  textAnchor="middle"
                  fill="#1e293b"
                  fontFamily="monospace"
                  style={{ pointerEvents: 'none' }}
                >
                  {q.toFixed(1)}
                </text>
              </g>
            );
          })
        )}

        {/* Column headers */}
        {Array(N_REACHES).fill(0).map((_, i) => (
          <text
            key={i}
            x={PAD_L + i * CELL_W + CELL_W / 2}
            y={PAD_T + nT * CELL_H + 14}
            fontSize={9}
            textAnchor="middle"
            fill="#475569"
          >
            i={i}
          </text>
        ))}

        {/* X axis label */}
        <text
          x={PAD_L + (N_REACHES * CELL_W) / 2}
          y={SVG_H - 2}
          fontSize={9}
          textAnchor="middle"
          fill="#64748b"
        >
          Reach i
        </text>

        {/* Row labels */}
        {Array(nT).fill(0).map((_, t) => (
          <text
            key={t}
            x={PAD_L - 4}
            y={PAD_T + (nT - 1 - t) * CELL_H + CELL_H / 2 + 4}
            fontSize={9}
            textAnchor="end"
            fill={t === stepIdx ? '#1d4ed8' : '#64748b'}
            fontWeight={t === stepIdx ? 'bold' : 'normal'}
          >
            t={t}
          </text>
        ))}

        {/* Y axis label */}
        <text
          x={10}
          y={PAD_T + (nT * CELL_H) / 2}
          fontSize={9}
          textAnchor="middle"
          fill="#64748b"
          transform={`rotate(-90,10,${PAD_T + (nT * CELL_H) / 2})`}
        >
          Time step
        </text>

        {/* Characteristic line */}
        {showCharacteristic && charPoints.length >= 2 && (
          <polyline
            points={charPoints.join(' ')}
            fill="none"
            stroke={cMax > 1 ? '#dc2626' : 'white'}
            strokeWidth={2}
            strokeDasharray={cMax > 1 ? '4,3' : '5,3'}
            opacity={0.85}
          />
        )}
        {showCharacteristic && charPoints.length >= 2 && (
          <text
            x={parseFloat(charPoints[charPoints.length - 1].split(',')[0]) + 4}
            y={parseFloat(charPoints[charPoints.length - 1].split(',')[1]) - 4}
            fontSize={8}
            fill={cMax > 1 ? '#dc2626' : 'white'}
            fontWeight="bold"
          >
            {cMax > 1 ? 'UNSTABLE ZONE ↑' : 'C=1 characteristic'}
          </text>
        )}

        {/* Hover tooltip */}
        {hovered && (() => {
          const tx = PAD_L + hovered.i * CELL_W + CELL_W / 2;
          const ty = PAD_T + (nT - 1 - hovered.t) * CELL_H - 6;
          const q = QMatrix[hovered.t][hovered.i];
          return (
            <g>
              <rect x={tx - 36} y={ty - 16} width={72} height={16} rx={3} fill="#1e293b" opacity={0.9} />
              <text x={tx} y={ty - 5} fontSize={9} textAnchor="middle" fill="white" fontFamily="monospace">
                Q={q.toFixed(2)} m³/s
              </text>
            </g>
          );
        })()}
      </svg>

      {/* Color legend */}
      <div className="mt-2 flex items-center gap-2 text-xs text-slate-500">
        <span>Low Q</span>
        <div className="flex h-3 w-32 rounded overflow-hidden">
          {Array(20).fill(0).map((_, k) => (
            <div key={k} style={{ flex: 1, background: qFill((k / 19) * Q_MAX, Q_MAX) }} />
          ))}
        </div>
        <span>High Q</span>
        <span className="ml-4 text-slate-400">Blue row = current step</span>
      </div>
    </div>
  );
}

// ─── Main widget ──────────────────────────────────────────────────────────────
export default function ExplicitSchemeWidget() {
  const [tab, setTab] = useState<'step' | 'spacetime'>('step');
  const [dt, setDt] = useState(DEFAULT_DT);
  const [useLimiter, setUseLimiter] = useState(false);
  const [showStencil, setShowStencil] = useState(false);
  const [stencilFocus, setStencilFocus] = useState(3);
  const [stepIdx, setStepIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(600);
  const [showCharacteristic, setShowCharacteristic] = useState(true);

  const alpha = useMemo(() => computeAlpha(DEFAULT_N, DEFAULT_S0, DEFAULT_B), []);
  const steps = useMemo(() => runSimulation(dt, useLimiter, alpha), [dt, useLimiter, alpha]);
  const curStep = steps[Math.min(stepIdx, steps.length - 1)];
  const Q_MAX = 14;

  // Reset stepIdx when dt or useLimiter changes (skip-mount pattern)
  const mountedRef = useRef(false);
  useEffect(() => {
    if (!mountedRef.current) { mountedRef.current = true; return; }
    setStepIdx(0);
    setPlaying(false);
  }, [dt, useLimiter]);

  // Animation
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => {
    if (!playing) return;
    if (stepIdx >= steps.length - 1) { setPlaying(false); return; }
    timerRef.current = setTimeout(() => setStepIdx(i => i + 1), speed);
    return () => { if (timerRef.current) clearTimeout(timerRef.current); };
  }, [playing, stepIdx, speed, steps]);

  // Derived
  const cMax = Math.max(...curStep.C);
  const cMean = curStep.C.reduce((a, b) => a + b, 0) / curStep.C.length;

  // Log lines accumulated
  const allLogLines: string[] = [];
  for (let s = 0; s <= stepIdx; s++) {
    steps[s].logLines.forEach(line => {
      allLogLines.push(`t=${steps[s].t * dt}s: ${line}`);
    });
  }
  const visibleLog = allLogLines.slice(-8);

  // Table rows
  const tableRows = Array(N_REACHES).fill(0).map((_, i) => {
    const A = curStep.A[i];
    const Q = curStep.Q[i];
    const u = Q / Math.max(A, 0.001);
    const c = cel(Q, A);
    const C = curStep.C[i];
    const Qlim = curStep.Q_lim[i];
    const active = Qlim < Q;
    return { i, A, Q, u, c, C, Qlim, active };
  });

  // Stencil computed value
  const stencilNewA =
    curStep.A[stencilFocus] -
    (dt / DEFAULT_DX) * (curStep.Q[stencilFocus] - curStep.Q[stencilFocus - 1]);

  return (
    <div className="not-prose my-8 rounded-2xl border border-slate-200 bg-white shadow-xl overflow-hidden font-sans">
      {/* Header */}
      <div className="bg-gradient-to-r from-sky-700 to-blue-900 px-6 py-4">
        <h3 className="text-white font-bold text-lg tracking-tight">
          Explicit Finite Differences — Step by Step
        </h3>
        <p className="text-sky-200 text-sm mt-0.5">
          Watch each reach update, track the Courant number, trigger instability
        </p>
      </div>

      {/* Tab bar */}
      <div className="flex border-b border-slate-200 bg-slate-50">
        {(['step', 'spacetime'] as const).map(t => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-5 py-2.5 text-sm font-medium transition-colors ${
              tab === t
                ? 'border-b-2 border-blue-600 text-blue-700 bg-white'
                : 'text-slate-500 hover:text-slate-700'
            }`}
          >
            {t === 'step' ? '📐 Step Through' : '🗺 Space-Time Diagram'}
          </button>
        ))}
      </div>

      {/* ── TAB 1: Step Through ──────────────────────────────────────────────── */}
      {tab === 'step' && (
        <div className="p-4 space-y-4">
          {/* Upstream hydrograph */}
          <div>
            <HydrographSVG stepIdx={stepIdx} />
          </div>

          {/* Main content: reaches + sidebar */}
          <div className="flex gap-4 flex-wrap">
            {/* Left: reach cells */}
            <div className="flex-1 min-w-0 space-y-2">
              {/* Inflow arrow */}
              <div className="flex items-center gap-1 mb-1">
                <div className="flex items-center gap-1">
                  <div className="w-8 h-0.5 bg-blue-500" />
                  <div className="w-0 h-0 border-t-4 border-b-4 border-l-8 border-transparent border-l-blue-500" />
                </div>
                <span className="text-xs text-blue-700 font-mono font-semibold">
                  Q_in = {HYDROGRAPH[stepIdx].toFixed(1)} m³/s
                </span>
              </div>

              {/* 8-reach visual */}
              <div className="flex gap-1 overflow-x-auto pb-2">
                {Array(N_REACHES).fill(0).map((_, i) => {
                  const q = curStep.Q[i];
                  const c_val = curStep.C[i];
                  const isLimActive = curStep.Q_lim[i] < curStep.Q[i];
                  const fillH = Math.max(8, (q / Q_MAX) * 70);
                  return (
                    <div key={i} className="flex flex-col items-center gap-0.5 min-w-[60px]">
                      <span className="text-[10px] text-slate-500 font-mono">i={i}</span>
                      <div
                        className={`w-14 h-20 rounded border-2 flex flex-col justify-end relative overflow-hidden ${
                          isLimActive ? 'border-orange-400 border-dashed' : 'border-slate-300'
                        }`}
                        style={{ background: '#f1f5f9' }}
                      >
                        {/* Water fill */}
                        <div
                          className="w-full transition-all duration-200 rounded-b"
                          style={{ height: `${fillH}px`, background: qFill(q, Q_MAX) }}
                        />
                        {/* Q label */}
                        <span className="absolute inset-0 flex items-center justify-center text-[11px] font-bold text-slate-800 font-mono">
                          {q.toFixed(2)}
                        </span>
                      </div>
                      {/* Courant badge */}
                      <span
                        className="text-[10px] font-mono px-1 rounded"
                        style={{
                          color: courantColor(c_val),
                          background: courantColor(c_val) + '22',
                        }}
                      >
                        C={c_val.toFixed(2)}
                      </span>
                      {/* Stencil labels */}
                      {showStencil && i === stencilFocus && (
                        <span className="text-[9px] bg-orange-200 text-orange-800 px-1 rounded">cur</span>
                      )}
                      {showStencil && i === stencilFocus - 1 && (
                        <span className="text-[9px] bg-blue-200 text-blue-800 px-1 rounded">up</span>
                      )}
                    </div>
                  );
                })}
              </div>

              {/* Stencil formula */}
              {showStencil && (
                <div className="bg-slate-50 rounded-lg p-3 font-mono text-xs mt-2 border border-slate-200">
                  <span className="bg-orange-200 px-1 rounded">A[{stencilFocus},t+1]</span>
                  {' = '}
                  <span className="bg-orange-200 px-1 rounded">A[{stencilFocus},t]</span>
                  {' − (Δt/Δx)·('}
                  <span className="bg-orange-200 px-1 rounded">Q[{stencilFocus},t]</span>
                  {' − '}
                  <span className="bg-blue-200 px-1 rounded">Q[{stencilFocus - 1},t]</span>
                  {')'}
                  <br />
                  {'= '}
                  {curStep.A[stencilFocus].toFixed(3)}
                  {' − ('}
                  {dt}/{DEFAULT_DX}
                  {')·('}
                  {curStep.Q[stencilFocus].toFixed(3)}
                  {' − '}
                  {curStep.Q[stencilFocus - 1].toFixed(3)}
                  {') = '}
                  {stencilNewA.toFixed(3)}
                  {' m²'}
                </div>
              )}

              {/* Numerical table */}
              <div className="overflow-x-auto mt-2">
                <table className="text-xs font-mono w-full border-collapse">
                  <thead>
                    <tr className="bg-slate-100">
                      <th className="border border-slate-200 px-2 py-1 text-left">i</th>
                      <th className="border border-slate-200 px-2 py-1 text-right">A (m²)</th>
                      <th className="border border-slate-200 px-2 py-1 text-right">Q (m³/s)</th>
                      <th className="border border-slate-200 px-2 py-1 text-right">u (m/s)</th>
                      <th className="border border-slate-200 px-2 py-1 text-right">c (m/s)</th>
                      <th className="border border-slate-200 px-2 py-1 text-right">C</th>
                      <th className="border border-slate-200 px-2 py-1 text-right">Q_lim</th>
                      <th className="border border-slate-200 px-2 py-1 text-center">Active?</th>
                    </tr>
                  </thead>
                  <tbody>
                    {tableRows.map(row => (
                      <tr key={row.i} className={row.active ? 'bg-orange-50' : ''}>
                        <td className="border border-slate-200 px-2 py-0.5">{row.i}</td>
                        <td className="border border-slate-200 px-2 py-0.5 text-right">{row.A.toFixed(3)}</td>
                        <td className="border border-slate-200 px-2 py-0.5 text-right">{row.Q.toFixed(3)}</td>
                        <td className="border border-slate-200 px-2 py-0.5 text-right">{row.u.toFixed(3)}</td>
                        <td className="border border-slate-200 px-2 py-0.5 text-right">{row.c.toFixed(3)}</td>
                        <td
                          className="border border-slate-200 px-2 py-0.5 text-right font-bold"
                          style={{ color: courantColor(row.C) }}
                        >
                          {row.C.toFixed(3)}
                        </td>
                        <td className="border border-slate-200 px-2 py-0.5 text-right">{row.Qlim.toFixed(3)}</td>
                        <td className="border border-slate-200 px-2 py-0.5 text-center">
                          {row.active ? <span className="text-orange-600 font-bold">yes</span> : <span className="text-slate-400">—</span>}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Step log */}
              <div className="bg-slate-900 text-green-300 rounded p-2 text-[10px] font-mono h-24 overflow-y-auto">
                {visibleLog.map((line, k) => (
                  <div key={k}>{line}</div>
                ))}
              </div>

              {/* Controls */}
              <div className="flex items-center gap-2 flex-wrap mt-1">
                <button
                  onClick={() => { setStepIdx(0); setPlaying(false); }}
                  className="px-2 py-1 text-xs bg-slate-100 hover:bg-slate-200 rounded border border-slate-300 font-medium"
                >
                  ⏮ Reset
                </button>
                <button
                  onClick={() => setStepIdx(i => Math.max(0, i - 1))}
                  disabled={stepIdx === 0}
                  className="px-2 py-1 text-xs bg-slate-100 hover:bg-slate-200 rounded border border-slate-300 font-medium disabled:opacity-40"
                >
                  ◀ Back
                </button>
                <button
                  onClick={() => setPlaying(p => !p)}
                  className="px-3 py-1 text-xs bg-blue-600 hover:bg-blue-700 text-white rounded font-medium"
                >
                  {playing ? '⏸ Pause' : '▶ Play'}
                </button>
                <button
                  onClick={() => setStepIdx(i => Math.min(i + 1, steps.length - 1))}
                  disabled={stepIdx >= steps.length - 1}
                  className="px-2 py-1 text-xs bg-slate-100 hover:bg-slate-200 rounded border border-slate-300 font-medium disabled:opacity-40"
                >
                  Step ▶
                </button>
                <span className="text-xs text-slate-500 font-mono">
                  t = {stepIdx} × {dt}s = {stepIdx * dt}s
                </span>
                <div className="flex items-center gap-1 ml-auto">
                  <span className="text-xs text-slate-400">Speed</span>
                  <input
                    type="range"
                    min={100}
                    max={1200}
                    step={100}
                    value={speed}
                    onChange={e => setSpeed(+e.target.value)}
                    className="w-20"
                  />
                  <span className="text-xs text-slate-400">{speed}ms</span>
                </div>
              </div>
            </div>

            {/* Right sidebar: CFL controls */}
            <div className="space-y-3 min-w-[220px] max-w-[240px]">
              {/* Δt slider */}
              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">
                  Δt (s): <span className="font-mono text-blue-700">{dt}</span>
                </label>
                <input
                  type="range"
                  min={100}
                  max={600}
                  step={50}
                  value={dt}
                  onChange={e => setDt(+e.target.value)}
                  className="w-full"
                />
                <div className="flex justify-between text-[10px] text-slate-400 mt-0.5">
                  <span>100s</span>
                  <span>600s</span>
                </div>
              </div>

              {/* C_max badge */}
              <div className="bg-slate-50 rounded p-2 border border-slate-200">
                <div className="text-xs text-slate-600 mb-0.5">Current step C_max</div>
                <div className="flex items-center gap-2">
                  <span
                    className="text-xl font-bold font-mono"
                    style={{ color: courantColor(cMax) }}
                  >
                    {cMax.toFixed(2)}
                  </span>
                  <span
                    className="text-xs font-semibold"
                    style={{ color: courantColor(cMax) }}
                  >
                    {cMax <= 1 ? '✓ Stable' : '✗ Unstable'}
                  </span>
                </div>
              </div>

              {/* CFL formula */}
              <div className="bg-slate-50 rounded p-2 font-mono text-xs border border-slate-200">
                <div className="text-slate-500 mb-1">Courant number:</div>
                C = c · Δt/Δx
                <br />
                = {cMean.toFixed(3)} × {dt} / {DEFAULT_DX}
                <br />
                = {(cMean * dt / DEFAULT_DX).toFixed(3)}
                <div className="text-[10px] text-slate-400 mt-1 not-mono">(mean celerity at this step)</div>
              </div>

              {/* Unstable warning */}
              {cMax > 1 && (
                <div className="bg-red-50 border border-red-300 rounded p-2 text-xs text-red-800">
                  ⚠ C &gt; 1 — explicit scheme UNSTABLE.
                  <br />
                  Oscillations will appear without flux limiter.
                </div>
              )}

              {/* Flux limiter toggle */}
              <label className="flex items-center gap-2 text-sm cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={useLimiter}
                  onChange={e => setUseLimiter(e.target.checked)}
                  className="accent-orange-500"
                />
                <span className="text-slate-700 font-medium">Enable Flux Limiter</span>
              </label>
              {useLimiter && (
                <div className="bg-orange-50 border border-orange-200 rounded p-2 text-xs text-orange-800 font-mono">
                  Q_out = min(Q_Manning, A·Δx/Δt)
                  <br />
                  <span className="not-mono text-orange-700">Limits effective C to 1.</span>
                </div>
              )}
              {curStep.anyLimiterActive && useLimiter && (
                <div className="bg-orange-100 border border-orange-300 rounded p-2 text-xs text-orange-900">
                  Limiter is <strong>active</strong> this step (dashed cells).
                </div>
              )}

              {/* Show stencil toggle */}
              <label className="flex items-center gap-2 text-sm cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={showStencil}
                  onChange={e => setShowStencil(e.target.checked)}
                  className="accent-blue-500"
                />
                <span className="text-slate-700 font-medium">Show Stencil</span>
              </label>
              {showStencil && (
                <div className="space-y-1">
                  <div className="flex items-center gap-2 text-xs text-slate-600">
                    <span>Focus reach:</span>
                    <input
                      type="range"
                      min={1}
                      max={7}
                      value={stencilFocus}
                      onChange={e => setStencilFocus(+e.target.value)}
                      className="w-20"
                    />
                    <span className="font-mono font-bold text-orange-700">{stencilFocus}</span>
                  </div>
                  <div className="text-[10px] text-slate-500 leading-relaxed">
                    <span className="bg-orange-200 px-0.5 rounded">orange = current reach</span>
                    {'  '}
                    <span className="bg-blue-200 px-0.5 rounded">blue = upstream</span>
                  </div>
                </div>
              )}

              {/* Courant legend */}
              <div className="border-t border-slate-200 pt-2 space-y-1">
                <div className="text-[10px] font-semibold text-slate-500 uppercase tracking-wide">Courant legend</div>
                {[
                  { label: 'C ≤ 0.8 — optimal', color: '#16a34a' },
                  { label: '0.8 < C ≤ 1 — marginal', color: '#d97706' },
                  { label: 'C > 1 — unstable', color: '#dc2626' },
                ].map(({ label, color }) => (
                  <div key={label} className="flex items-center gap-1.5 text-[10px]">
                    <div className="w-3 h-3 rounded-sm" style={{ background: color + '33', border: `1.5px solid ${color}` }} />
                    <span style={{ color }}>{label}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── TAB 2: Space-Time Diagram ─────────────────────────────────────────── */}
      {tab === 'spacetime' && (
        <div className="p-4 space-y-4">
          {/* Controls row */}
          <div className="flex items-center gap-6 flex-wrap">
            {/* Δt slider (mirrored) */}
            <div className="flex items-center gap-2">
              <label className="text-xs font-semibold text-slate-700">Δt (s):</label>
              <input
                type="range"
                min={100}
                max={600}
                step={50}
                value={dt}
                onChange={e => setDt(+e.target.value)}
                className="w-28"
              />
              <span className="text-xs font-mono text-blue-700">{dt}s</span>
            </div>

            {/* Flux limiter */}
            <label className="flex items-center gap-2 text-sm cursor-pointer select-none">
              <input
                type="checkbox"
                checked={useLimiter}
                onChange={e => setUseLimiter(e.target.checked)}
                className="accent-orange-500"
              />
              <span className="text-slate-700">Flux Limiter</span>
            </label>

            {/* Characteristic */}
            <label className="flex items-center gap-2 text-sm cursor-pointer select-none">
              <input
                type="checkbox"
                checked={showCharacteristic}
                onChange={e => setShowCharacteristic(e.target.checked)}
                className="accent-blue-500"
              />
              <span className="text-slate-700">Show Characteristic</span>
            </label>

            {/* C_max indicator */}
            <div className="flex items-center gap-1.5 text-xs">
              <span className="text-slate-500">Global C_max:</span>
              <span
                className="font-mono font-bold"
                style={{ color: courantColor(Math.max(...steps.flatMap(s => s.C))) }}
              >
                {Math.max(...steps.flatMap(s => s.C)).toFixed(2)}
              </span>
              {Math.max(...steps.flatMap(s => s.C)) > 1 && (
                <span className="text-red-600 font-semibold">✗ Unstable</span>
              )}
            </div>
          </div>

          {/* Explanation banner */}
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 text-xs text-blue-800 leading-relaxed">
            Each cell shows Q (m³/s) at reach i, time step t. Colour intensity = discharge magnitude.
            The <strong>diagonal characteristic line</strong> shows how information propagates: slope = Δx/(c·Δt) cells per step.
            When C &gt; 1, the characteristic is steeper than the grid — information cannot propagate fast enough and the scheme becomes unstable.
            <strong> Blue row</strong> = current step in the Step Through tab.
          </div>

          {/* Diagram */}
          <div className="overflow-x-auto">
            <SpaceTimeDiagram
              steps={steps}
              stepIdx={stepIdx}
              showCharacteristic={showCharacteristic}
            />
          </div>

          {/* Navigation (sync with step tab) */}
          <div className="flex items-center gap-2 flex-wrap pt-1">
            <button
              onClick={() => { setStepIdx(0); setPlaying(false); }}
              className="px-2 py-1 text-xs bg-slate-100 hover:bg-slate-200 rounded border border-slate-300 font-medium"
            >
              ⏮ Reset
            </button>
            <button
              onClick={() => setStepIdx(i => Math.max(0, i - 1))}
              disabled={stepIdx === 0}
              className="px-2 py-1 text-xs bg-slate-100 hover:bg-slate-200 rounded border border-slate-300 font-medium disabled:opacity-40"
            >
              ◀ Back
            </button>
            <button
              onClick={() => setPlaying(p => !p)}
              className="px-3 py-1 text-xs bg-blue-600 hover:bg-blue-700 text-white rounded font-medium"
            >
              {playing ? '⏸ Pause' : '▶ Play'}
            </button>
            <button
              onClick={() => setStepIdx(i => Math.min(i + 1, steps.length - 1))}
              disabled={stepIdx >= steps.length - 1}
              className="px-2 py-1 text-xs bg-slate-100 hover:bg-slate-200 rounded border border-slate-300 font-medium disabled:opacity-40"
            >
              Step ▶
            </button>
            <span className="text-xs text-slate-500 font-mono">
              Highlighted row: t={stepIdx} (t={stepIdx * dt}s)
            </span>
            <div className="flex items-center gap-1 ml-auto">
              <span className="text-xs text-slate-400">Speed</span>
              <input
                type="range"
                min={100}
                max={1200}
                step={100}
                value={speed}
                onChange={e => setSpeed(+e.target.value)}
                className="w-20"
              />
              <span className="text-xs text-slate-400">{speed}ms</span>
            </div>
          </div>
        </div>
      )}

      {/* Footer */}
      <div className="bg-slate-50 border-t border-slate-200 px-5 py-2.5 flex flex-wrap gap-4 text-[11px] text-slate-500">
        <span>
          <strong className="text-slate-700">Channel:</strong> B={DEFAULT_B}m, S₀={DEFAULT_S0}, n={DEFAULT_N}
        </span>
        <span>
          <strong className="text-slate-700">Grid:</strong> Δx={DEFAULT_DX}m, {N_REACHES} reaches
        </span>
        <span>
          <strong className="text-slate-700">α</strong> = {alpha.toFixed(4)}
        </span>
        <span>
          <strong className="text-slate-700">β</strong> = 5/3 (Manning power)
        </span>
        <span className="ml-auto">
          Kinematic wave: ∂A/∂t + ∂Q/∂x = 0, Q = αA^β
        </span>
      </div>
    </div>
  );
}
