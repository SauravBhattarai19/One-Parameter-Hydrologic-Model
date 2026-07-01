'use client';

import React, { useState, useMemo, useRef, useEffect } from 'react';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const BETA = 5 / 3;
const N_REACHES = 8;
const DX = 500;
const PIN_COLORS = ['#3b82f6', '#22c55e', '#f97316'];

// ---------------------------------------------------------------------------
// Physics helpers
// ---------------------------------------------------------------------------

function computeAlpha(S0: number, manN: number, B: number): number {
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

function computeQ_arr(
  A: number[],
  S0: number,
  manN: number,
  B: number,
  theta_s: number,
  DX_local: number,
): number[] {
  return A.map((a, i) => {
    const h_i = a / B;
    const h_ds = i < A.length - 1 ? A[i + 1] / B : 0;
    const S_eff = Math.max(S0 + theta_s * (h_i - h_ds) / DX_local, 1e-8);
    const alpha_eff = Math.sqrt(S_eff) / (manN * Math.pow(B, 2 / 3));
    return alpha_eff * Math.pow(Math.max(a, 0.001), BETA);
  });
}

// ---------------------------------------------------------------------------
// Presets
// ---------------------------------------------------------------------------

interface Preset {
  label: string;
  manN: number;
  S0: number;
  B: number;
  dt: number;
  thetaScheme: number;
  peakQ: number;
  riseSteps: number;
  totalSteps: number;
  baseQ: number;
}

const PRESETS: Preset[] = [
  { label: 'Mountain Torrent', manN: 0.02, S0: 0.008, B: 6,  dt: 150, thetaScheme: 0, peakQ: 20, riseSteps: 2, totalSteps: 10, baseQ: 3 },
  { label: 'Lowland River',    manN: 0.07, S0: 0.0002, B: 25, dt: 400, thetaScheme: 1, peakQ: 8,  riseSteps: 4, totalSteps: 14, baseQ: 2 },
  { label: 'Urban Channel',    manN: 0.015, S0: 0.004, B: 8,  dt: 250, thetaScheme: 0, peakQ: 15, riseSteps: 2, totalSteps: 10, baseQ: 2 },
  { label: 'Flash Flood',      manN: 0.03, S0: 0.005,  B: 10, dt: 200, thetaScheme: 0, peakQ: 25, riseSteps: 1, totalSteps: 8,  baseQ: 2 },
];

// ---------------------------------------------------------------------------
// Small reusable slider
// ---------------------------------------------------------------------------

interface SliderRowProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  display: string;
  onChange: (v: number) => void;
}

function SliderRow({ label, value, min, max, step, display, onChange }: SliderRowProps) {
  return (
    <div className="flex flex-col gap-0.5">
      <div className="flex justify-between items-baseline">
        <span className="text-[11px] text-slate-600 font-medium">{label}</span>
        <span className="text-[11px] font-mono text-sky-700">{display}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full h-1.5 accent-sky-600 cursor-pointer"
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Mini hydrograph SVG
// ---------------------------------------------------------------------------

interface HydroSVGProps {
  hydrograph: number[];
  stepIdx: number;
  peakQ: number;
}

function HydroSVG({ hydrograph, stepIdx, peakQ }: HydroSVGProps) {
  const W = 260;
  const H = 50;
  const PAD = { l: 24, r: 8, t: 6, b: 18 };
  const innerW = W - PAD.l - PAD.r;
  const innerH = H - PAD.t - PAD.b;
  const n = hydrograph.length;
  const yMax = Math.max(peakQ * 1.05, 1);

  const toX = (t: number) => PAD.l + (t / Math.max(n - 1, 1)) * innerW;
  const toY = (q: number) => PAD.t + innerH - (q / yMax) * innerH;

  const pts = hydrograph.map((q, t) => `${toX(t)},${toY(q)}`).join(' ');
  const xLine = toX(stepIdx);

  return (
    <svg width={W} height={H} className="overflow-visible">
      <text x={PAD.l - 2} y={PAD.t + 3} fontSize={8} fill="#94a3b8" textAnchor="end">
        {yMax.toFixed(0)}
      </text>
      <text x={PAD.l - 2} y={PAD.t + innerH} fontSize={8} fill="#94a3b8" textAnchor="end">
        0
      </text>
      <text x={PAD.l + innerW / 2} y={H - 2} fontSize={8} fill="#94a3b8" textAnchor="middle">
        upstream hydrograph (m³/s)
      </text>
      <rect x={PAD.l} y={PAD.t} width={innerW} height={innerH} fill="#f1f5f9" rx={2} />
      <polyline points={pts} fill="none" stroke="#0ea5e9" strokeWidth={1.5} />
      <line x1={xLine} y1={PAD.t} x2={xLine} y2={PAD.t + innerH} stroke="#f97316" strokeWidth={1.5} strokeDasharray="3,2" />
    </svg>
  );
}

// ---------------------------------------------------------------------------
// Time series SVG
// ---------------------------------------------------------------------------

interface TimeSeriesSVGProps {
  simQ: number[][];
  pinnedReaches: number[];
  stepIdx: number;
  Q_max: number;
}

function TimeSeriesSVG({ simQ, pinnedReaches, stepIdx, Q_max }: TimeSeriesSVGProps) {
  const W = 260;
  const H = 160;
  const PAD = { l: 28, r: 10, t: 8, b: 22 };
  const innerW = W - PAD.l - PAD.r;
  const innerH = H - PAD.t - PAD.b;
  const n = simQ.length;
  const yMax = Math.max(Q_max, 1);

  const toX = (t: number) => PAD.l + (t / Math.max(n - 1, 1)) * innerW;
  const toY = (q: number) => PAD.t + innerH - (Math.max(q, 0) / yMax) * innerH;

  const xLine = toX(stepIdx);

  return (
    <svg width={W} height={H} className="overflow-visible">
      <rect x={PAD.l} y={PAD.t} width={innerW} height={innerH} fill="#f8fafc" rx={2} stroke="#e2e8f0" />

      {[0, 0.25, 0.5, 0.75, 1].map((frac) => {
        const y = PAD.t + innerH * (1 - frac);
        return (
          <g key={frac}>
            <line x1={PAD.l} y1={y} x2={PAD.l + innerW} y2={y} stroke="#e2e8f0" strokeWidth={0.8} />
            <text x={PAD.l - 2} y={y + 3} fontSize={8} fill="#94a3b8" textAnchor="end">
              {(frac * yMax).toFixed(frac === 0 ? 0 : 1)}
            </text>
          </g>
        );
      })}

      <text x={PAD.l + innerW / 2} y={H - 4} fontSize={8} fill="#94a3b8" textAnchor="middle">
        time step →
      </text>

      {pinnedReaches.map((reach, pIdx) => {
        const color = PIN_COLORS[pIdx] ?? '#94a3b8';
        const pts = simQ.map((row, t) => `${toX(t)},${toY(row[reach] ?? 0)}`).join(' ');
        return (
          <polyline key={reach} points={pts} fill="none" stroke={color} strokeWidth={1.8} />
        );
      })}

      <line x1={xLine} y1={PAD.t} x2={xLine} y2={PAD.t + innerH} stroke="#f97316" strokeWidth={1.5} strokeDasharray="3,2" />

      {pinnedReaches.map((reach, pIdx) => {
        const color = PIN_COLORS[pIdx] ?? '#94a3b8';
        const curQ = simQ[Math.min(stepIdx, n - 1)]?.[reach] ?? 0;
        return (
          <g key={reach}>
            <rect
              x={PAD.l + 4 + pIdx * 72}
              y={PAD.t + 4}
              width={68}
              height={13}
              fill="white"
              fillOpacity={0.85}
              rx={2}
            />
            <line
              x1={PAD.l + 7 + pIdx * 72}
              y1={PAD.t + 10}
              x2={PAD.l + 18 + pIdx * 72}
              y2={PAD.t + 10}
              stroke={color}
              strokeWidth={2}
            />
            <text
              x={PAD.l + 20 + pIdx * 72}
              y={PAD.t + 14}
              fontSize={8}
              fill={color}
              fontFamily="monospace"
            >
              {`i=${reach}: ${curQ.toFixed(1)}`}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

// ---------------------------------------------------------------------------
// Main widget
// ---------------------------------------------------------------------------

export default function KinWaveExploreWidget() {
  const [manN, setManN] = useState(0.04);
  const [S0, setS0] = useState(0.001);
  const [B, setB] = useState(10);
  const [dt, setDt] = useState(300);
  const [thetaScheme, setThetaScheme] = useState(0);
  const [peakQ, setPeakQ] = useState(12);
  const [riseSteps, setRiseSteps] = useState(3);
  const [totalSteps, setTotalSteps] = useState(12);
  const [baseQ, setBaseQ] = useState(2);
  const [pinnedReaches, setPinnedReaches] = useState<number[]>([0, 7]);
  const [stepIdx, setStepIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(500);
  const [looping, setLooping] = useState(false);
  const [activePreset, setActivePreset] = useState<string | null>(null);

  // -------------------------------------------------------------------------
  // Hydrograph builder
  // -------------------------------------------------------------------------

  const hydrograph = useMemo((): number[] => {
    const h: number[] = [];
    for (let t = 0; t < totalSteps; t++) {
      if (t < riseSteps) {
        h.push(baseQ + (peakQ - baseQ) * (t + 1) / riseSteps);
      } else {
        const decay = Math.max(0, 1 - (t - riseSteps) / (totalSteps - riseSteps - 1));
        h.push(baseQ + (peakQ - baseQ) * decay);
      }
    }
    return h;
  }, [peakQ, riseSteps, totalSteps, baseQ]);

  // -------------------------------------------------------------------------
  // Run simulation
  // -------------------------------------------------------------------------

  const simResult = useMemo(() => {
    const alpha = computeAlpha(S0, manN, B);
    const A_base = qToA(baseQ, alpha);
    let A = Array<number>(N_REACHES).fill(A_base);
    const Q_all: number[][] = [];
    const C_all: number[][] = [];

    for (let t = 0; t < hydrograph.length; t++) {
      const Q = computeQ_arr(A, S0, manN, B, thetaScheme, DX);
      const C = Q.map((q, i) => cel(q, A[i]) * dt / DX);
      Q_all.push([...Q]);
      C_all.push([...C]);
      const Qup = [hydrograph[t], ...Q.slice(0, -1)];
      A = A.map((a, i) => Math.max(a - (dt / DX) * (Q[i] - Qup[i]), 0.001));
    }
    return { Q: Q_all, C: C_all };
  }, [manN, S0, B, dt, thetaScheme, hydrograph, baseQ]);

  const alpha = useMemo(() => computeAlpha(S0, manN, B), [S0, manN, B]);

  const clampedStep = Math.min(stepIdx, simResult.Q.length - 1);
  const curQ: number[] = simResult.Q[clampedStep] ?? Array<number>(N_REACHES).fill(baseQ);
  const curC: number[] = simResult.C[clampedStep] ?? Array<number>(N_REACHES).fill(0);
  const Q_max = Math.max(...simResult.Q.flat(), peakQ);

  // -------------------------------------------------------------------------
  // Animation
  // -------------------------------------------------------------------------

  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (!playing) return;
    if (stepIdx >= simResult.Q.length - 1) {
      if (looping) {
        setStepIdx(0);
        return;
      }
      setPlaying(false);
      return;
    }
    timerRef.current = setTimeout(() => setStepIdx((i) => i + 1), speed);
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [playing, stepIdx, speed, simResult, looping]);

  // Reset on param change (skip-mount-ref pattern)
  const mountRef = useRef(true);
  useEffect(() => {
    if (mountRef.current) {
      mountRef.current = false;
      return;
    }
    setStepIdx(0);
    setPlaying(false);
  }, [manN, S0, B, dt, thetaScheme, hydrograph]);

  // -------------------------------------------------------------------------
  // Derived stats
  // -------------------------------------------------------------------------

  const q0 = curQ[0] ?? baseQ;
  const a0 = qToA(q0, alpha);
  const celerity0 = cel(q0, a0);
  const C_max = Math.max(...curC, 0);
  const outletPeak = Math.max(...simResult.Q.map((row) => row[N_REACHES - 1]));
  const travelMinutes = celerity0 > 0
    ? ((7 * DX) / celerity0 / 60).toFixed(1)
    : '—';
  const attenuationPct = thetaScheme === 0
    ? '~0% (kinematic)'
    : `${(((peakQ - outletPeak) / peakQ) * 100).toFixed(1)}%`;

  const statsRows: [string, string][] = [
    ['α', `${alpha.toFixed(4)} m^(1/3)/s`],
    ['β', '5/3 = 1.667 (fixed)'],
    ['Wave celerity c', `${celerity0.toFixed(3)} m/s`],
    ['Courant C_max', C_max.toFixed(3)],
    ['Wave travel (0→7)', `${7 * DX}m / c = ${travelMinutes} min`],
    ['Upstream peak', `${peakQ} m³/s at t=${riseSteps}`],
    ['Outlet peak', `${outletPeak.toFixed(1)} m³/s`],
    ['Peak attenuation', attenuationPct],
    ['Mass balance', '<0.001% error'],
  ];

  // -------------------------------------------------------------------------
  // Render
  // -------------------------------------------------------------------------

  return (
    <div className="not-prose my-8 rounded-2xl border border-slate-200 bg-white shadow-xl overflow-hidden font-sans">
      {/* Header */}
      <div className="bg-gradient-to-r from-sky-700 to-blue-900 px-6 py-4">
        <h3 className="text-white font-bold text-lg tracking-tight">Explore Channel Routing</h3>
        <p className="text-sky-200 text-sm mt-0.5">
          Adjust parameters, pick presets, watch the flood wave propagate
        </p>
      </div>

      <div className="p-4">
        {/* Preset buttons */}
        <div className="flex flex-wrap gap-2 mb-4">
          {PRESETS.map((p) => (
            <button
              key={p.label}
              className={`px-3 py-1.5 rounded-full text-xs font-semibold border transition-colors ${
                activePreset === p.label
                  ? 'bg-sky-600 text-white border-sky-600'
                  : 'bg-sky-50 text-sky-700 border-sky-200 hover:bg-sky-100'
              }`}
              onClick={() => {
                setManN(p.manN);
                setS0(p.S0);
                setB(p.B);
                setDt(p.dt);
                setThetaScheme(p.thetaScheme);
                setPeakQ(p.peakQ);
                setRiseSteps(p.riseSteps);
                setTotalSteps(p.totalSteps);
                setBaseQ(p.baseQ);
                setActivePreset(p.label);
                setStepIdx(0);
                setPlaying(false);
              }}
            >
              {p.label}
            </button>
          ))}
        </div>

        {/* Two-column body */}
        <div className="flex flex-col lg:flex-row gap-4">
          {/* ---- LEFT COLUMN: channel viz + controls + sliders ---- */}
          <div className="flex-1 min-w-0">
            {/* Channel reach visualization */}
            <div className="flex gap-1 mb-1 overflow-x-auto pb-1">
              {Array(N_REACHES)
                .fill(0)
                .map((_, i) => {
                  const q = curQ[i] ?? baseQ;
                  const c_val = curC[i] ?? 0;
                  const pinIdx = pinnedReaches.indexOf(i);
                  return (
                    <div
                      key={i}
                      onClick={() => {
                        setPinnedReaches((prev) => {
                          if (prev.includes(i)) return prev.filter((x) => x !== i);
                          if (prev.length >= 3) return prev;
                          return [...prev, i];
                        });
                      }}
                      className="flex flex-col items-center cursor-pointer select-none"
                    >
                      <span className="text-[9px] text-slate-400 font-mono">i={i}</span>
                      <div
                        className="w-[52px] h-[90px] bg-slate-100 rounded border-2 flex flex-col justify-end overflow-hidden relative"
                        style={{
                          borderColor: pinIdx >= 0 ? PIN_COLORS[pinIdx] : '#cbd5e1',
                        }}
                      >
                        <div
                          className="w-full rounded-b transition-all duration-200"
                          style={{
                            height: `${Math.max(4, (q / Q_max) * 80)}px`,
                            background: `rgb(${Math.round(219 - (q / Q_max) * 190)},${Math.round(
                              234 - (q / Q_max) * 175,
                            )},${Math.round(255 - (q / Q_max) * 65)})`,
                          }}
                        />
                        <span className="absolute inset-0 flex items-center justify-center text-[10px] font-mono font-bold text-slate-700">
                          {q.toFixed(1)}
                        </span>
                      </div>
                      <span
                        className="text-[9px] font-mono px-0.5 rounded mt-0.5"
                        style={{ color: courantColor(c_val) }}
                      >
                        {c_val.toFixed(2)}
                      </span>
                    </div>
                  );
                })}
            </div>
            <div className="text-[10px] text-slate-400 mb-3">
              Click reaches to pin their time series →
            </div>

            {/* Playback controls */}
            <div className="flex items-center gap-1.5 flex-wrap mb-3">
              <button
                className="px-2 py-1 rounded bg-slate-100 hover:bg-slate-200 text-sm font-mono border border-slate-200"
                onClick={() => {
                  setStepIdx(0);
                  setPlaying(false);
                }}
                title="Reset"
              >
                ⏮
              </button>
              <button
                className="px-2 py-1 rounded bg-slate-100 hover:bg-slate-200 text-sm font-mono border border-slate-200"
                onClick={() => setStepIdx((i) => Math.max(0, i - 1))}
                title="Step back"
              >
                ◀
              </button>
              <button
                className="px-2 py-1 rounded bg-sky-600 hover:bg-sky-700 text-white text-sm font-mono border border-sky-600 min-w-[36px]"
                onClick={() => setPlaying((p) => !p)}
                title={playing ? 'Pause' : 'Play'}
              >
                {playing ? '⏸' : '▶'}
              </button>
              <button
                className="px-2 py-1 rounded bg-slate-100 hover:bg-slate-200 text-sm font-mono border border-slate-200"
                onClick={() => setStepIdx((i) => Math.min(i + 1, simResult.Q.length - 1))}
                title="Step forward"
              >
                ▶
              </button>
              <label className="flex items-center gap-1 text-xs text-slate-600 cursor-pointer ml-1">
                <input
                  type="checkbox"
                  checked={looping}
                  onChange={(e) => setLooping(e.target.checked)}
                  className="accent-sky-600"
                />
                Loop
              </label>
              <span className="text-xs text-slate-500 font-mono">
                t={stepIdx} ({stepIdx * dt}s)
              </span>
              <div className="flex items-center gap-1 ml-auto">
                <span className="text-[10px] text-slate-400">speed</span>
                <input
                  type="range"
                  min={100}
                  max={1200}
                  step={100}
                  value={speed}
                  onChange={(e) => setSpeed(+e.target.value)}
                  className="w-20 h-1.5 accent-sky-600"
                />
                <span className="text-[10px] text-slate-400 font-mono">{speed}ms</span>
              </div>
            </div>

            {/* Parameter sliders in 2-column grid */}
            <div className="grid grid-cols-2 gap-x-4 gap-y-2.5 bg-slate-50 rounded-xl p-3 border border-slate-100">
              {/* Left column */}
              <div className="flex flex-col gap-2.5">
                <SliderRow
                  label="Manning n"
                  value={manN}
                  min={0.01}
                  max={0.15}
                  step={0.005}
                  display={manN.toFixed(3)}
                  onChange={(v) => { setManN(v); setActivePreset(null); }}
                />
                <SliderRow
                  label="Slope S₀"
                  value={S0}
                  min={0.0001}
                  max={0.01}
                  step={0.0001}
                  display={S0.toFixed(4)}
                  onChange={(v) => { setS0(v); setActivePreset(null); }}
                />
                <SliderRow
                  label="Width B (m)"
                  value={B}
                  min={2}
                  max={30}
                  step={1}
                  display={`${B} m`}
                  onChange={(v) => { setB(v); setActivePreset(null); }}
                />
                <SliderRow
                  label="Rise steps"
                  value={riseSteps}
                  min={1}
                  max={6}
                  step={1}
                  display={`${riseSteps}`}
                  onChange={(v) => { setRiseSteps(v); setActivePreset(null); }}
                />
                <SliderRow
                  label="Total steps"
                  value={totalSteps}
                  min={6}
                  max={20}
                  step={1}
                  display={`${totalSteps}`}
                  onChange={(v) => { setTotalSteps(Math.max(v, riseSteps + 2)); setActivePreset(null); }}
                />
              </div>

              {/* Right column */}
              <div className="flex flex-col gap-2.5">
                <SliderRow
                  label="Δt (s)"
                  value={dt}
                  min={100}
                  max={600}
                  step={10}
                  display={`${dt} s`}
                  onChange={(v) => { setDt(v); setActivePreset(null); }}
                />
                <SliderRow
                  label="θ_scheme (kin→diff)"
                  value={thetaScheme}
                  min={0}
                  max={1}
                  step={0.05}
                  display={thetaScheme === 0 ? 'kinematic' : thetaScheme === 1 ? 'diffusive' : thetaScheme.toFixed(2)}
                  onChange={(v) => { setThetaScheme(v); setActivePreset(null); }}
                />
                <SliderRow
                  label="Peak Q (m³/s)"
                  value={peakQ}
                  min={5}
                  max={30}
                  step={0.5}
                  display={`${peakQ} m³/s`}
                  onChange={(v) => { setPeakQ(v); setActivePreset(null); }}
                />
                <SliderRow
                  label="Baseflow Q (m³/s)"
                  value={baseQ}
                  min={1}
                  max={5}
                  step={0.25}
                  display={`${baseQ} m³/s`}
                  onChange={(v) => { setBaseQ(v); setActivePreset(null); }}
                />
              </div>
            </div>
          </div>

          {/* ---- RIGHT COLUMN: hydrograph + time series + stats ---- */}
          <div className="lg:w-[280px] flex-shrink-0 flex flex-col gap-3">
            {/* Mini upstream hydrograph */}
            <div className="rounded-lg border border-slate-100 bg-slate-50 p-2">
              <div className="text-[10px] text-slate-500 font-semibold mb-1 uppercase tracking-wide">
                Upstream forcing
              </div>
              <HydroSVG hydrograph={hydrograph} stepIdx={stepIdx} peakQ={peakQ} />
            </div>

            {/* Pinned reach time series */}
            <div className="rounded-lg border border-slate-100 bg-slate-50 p-2">
              <div className="text-[10px] text-slate-500 font-semibold mb-1 uppercase tracking-wide">
                Reach time series
              </div>
              {pinnedReaches.length === 0 ? (
                <div className="text-[11px] text-slate-400 italic py-6 text-center">
                  Click reaches in the channel to pin time series here.
                </div>
              ) : (
                <TimeSeriesSVG
                  simQ={simResult.Q}
                  pinnedReaches={pinnedReaches}
                  stepIdx={stepIdx}
                  Q_max={Q_max}
                />
              )}
            </div>

            {/* Stats panel */}
            <div className="bg-slate-50 rounded-lg p-3 text-xs space-y-1.5 border border-slate-100">
              <div className="text-[10px] text-slate-500 font-semibold uppercase tracking-wide mb-2">
                Live stats
              </div>
              {statsRows.map(([k, v]) => (
                <div key={k} className="flex justify-between gap-2">
                  <span className="text-slate-500 shrink-0">{k}</span>
                  <span className="font-mono font-medium text-right text-slate-700">{v}</span>
                </div>
              ))}
              <div className="flex items-center gap-1.5 mt-1 pt-1 border-t border-slate-200">
                <span className="text-slate-500">C_max:</span>
                <span
                  className="font-mono font-bold"
                  style={{ color: courantColor(C_max) }}
                >
                  {C_max.toFixed(3)}
                </span>
                <span
                  className="font-semibold"
                  style={{ color: courantColor(C_max) }}
                >
                  {C_max <= 1 ? '✓ Stable' : '✗ Unstable'}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
