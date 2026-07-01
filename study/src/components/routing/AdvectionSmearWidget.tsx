'use client';

import React, { useState, useMemo } from 'react';

// ─────────────────────────────────────────────────────────────────────────
// AdvectionSmearWidget — the puzzle & the payoff.
//
// Pure advection ∂Q/∂t + c·∂Q/∂x = 0 has the exact solution Q(x,t)=Q₀(x−ct):
// the pulse just SLIDES downstream, shape unchanged.  Yet the first-order
// upwind scheme  Qᵢⁿ⁺¹ = C·Qᵢ₋₁ⁿ + (1−C)·Qᵢⁿ  smears it.  This widget runs
// both and overlays them so the numerical diffusion is impossible to miss —
// and, because C = c·Δt/Δx with c,Δx fixed, the C slider IS the Δt knob:
// slide it DOWN (smaller Δt) and watch the peak melt (the "Δt trap").
// ─────────────────────────────────────────────────────────────────────────

const N = 64;          // grid cells
const START = 8;       // initial pulse centre (cell)
const TRAVEL = 44;     // cells the wave should translate before we compare
const C_FIXED = 1.45;  // wave celerity [m/s]  (matches Ch.4 default)
const DX_FIXED = 500;  // cell size [m]        (matches Ch.4 default)

function gaussian(i: number, centre: number, sigma: number): number {
  const z = (i - centre) / sigma;
  return Math.exp(-0.5 * z * z);
}

interface SimResult {
  numerical: number[];
  truth: number[];
  initial: number[];
  trueCentre: number;
  nSteps: number;
  retention: number;   // peak kept, 0..1
  unstable: boolean;
}

function simulate(C: number, sigma: number): SimResult {
  const initial = Array.from({ length: N }, (_, i) => gaussian(i, START, sigma));
  const nSteps = Math.max(1, Math.round(TRAVEL / C));
  const trueCentre = START + C * nSteps;
  const truth = Array.from({ length: N }, (_, i) => gaussian(i, trueCentre, sigma));

  let Q = [...initial];
  let unstable = false;
  for (let s = 0; s < nSteps; s++) {
    const Qn = new Array<number>(N);
    Qn[0] = 0; // clean upstream baseflow boundary
    for (let i = 1; i < N; i++) {
      Qn[i] = C * Q[i - 1] + (1 - C) * Q[i];
    }
    Q = Qn;
    let mx = 0;
    for (let i = 0; i < N; i++) mx = Math.max(mx, Math.abs(Q[i]));
    if (!isFinite(mx) || mx > 50) { unstable = true; break; }
  }

  // C>1 ⇒ α<0 ⇒ unconditionally unstable (any grid-scale noise grows): flag it
  // categorically even if this smooth, finite run did not overflow in nSteps.
  unstable = unstable || C > 1.0001;
  const peak = unstable ? 0 : Math.max(...Q);
  return { numerical: Q, truth, initial, trueCentre, nSteps, retention: peak, unstable };
}

function retentionColor(r: number): string {
  if (r >= 0.85) return '#16a34a';
  if (r >= 0.6) return '#d97706';
  return '#dc2626';
}

function LabeledSlider({
  label, value, min, max, step, display, onChange,
}: {
  label: string; value: number; min: number; max: number; step: number;
  display?: string; onChange: (v: number) => void;
}) {
  return (
    <div className="flex flex-col gap-0.5">
      <div className="flex items-center justify-between">
        <label className="text-xs font-semibold text-slate-500">{label}</label>
        <span className="text-xs font-mono text-slate-700 tabular-nums">{display ?? value}</span>
      </div>
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full h-1.5 rounded-full accent-sky-600 cursor-pointer"
      />
    </div>
  );
}

// SVG line chart of Q vs cell index
const W = 540, H = 220, PADL = 34, PADR = 12, PADT = 14, PADB = 26;
function px(i: number): number { return PADL + (i / (N - 1)) * (W - PADL - PADR); }
function py(q: number): number { return H - PADB - q * (H - PADT - PADB); }

function pathOf(arr: number[]): string {
  return arr.map((q, i) => `${i === 0 ? 'M' : 'L'}${px(i).toFixed(1)},${py(Math.min(q, 1.15)).toFixed(1)}`).join(' ');
}

export default function AdvectionSmearWidget() {
  const [C, setC] = useState(0.5);
  const [sigma, setSigma] = useState(3.0);
  const [showTruth, setShowTruth] = useState(true);
  const [showInitial, setShowInitial] = useState(true);

  const sim = useMemo(() => simulate(C, sigma), [C, sigma]);
  const dt = (C * DX_FIXED) / C_FIXED;       // derived: C = c·Δt/Δx
  const diffStrength = Math.max(0, 1 - C);   // α ∝ (1−C)
  const rPct = Math.round(sim.retention * 100);

  return (
    <div className="not-prose my-6 rounded-2xl border border-slate-200 bg-white shadow-sm overflow-hidden">
      <div className="bg-gradient-to-r from-sky-600 to-indigo-600 px-5 py-3">
        <h3 className="text-white font-bold text-base">The flood peak the grid throws away</h3>
        <p className="text-sky-100 text-xs mt-0.5">
          The true kinematic wave just slides downstream, shape intact. The upwind grid smears it. The Courant
          number C decides how much.
        </p>
      </div>

      <div className="grid md:grid-cols-[1fr_auto] gap-5 p-5">
        {/* Chart */}
        <div>
          <svg viewBox={`0 0 ${W} ${H}`} className="w-full select-none" style={{ maxHeight: 240 }}>
            {/* axes */}
            <line x1={PADL} y1={H - PADB} x2={W - PADR} y2={H - PADB} stroke="#cbd5e1" strokeWidth={1} />
            <line x1={PADL} y1={PADT} x2={PADL} y2={H - PADB} stroke="#cbd5e1" strokeWidth={1} />
            <text x={(W) / 2} y={H - 6} textAnchor="middle" className="fill-slate-400" fontSize={10}>
              distance downstream  (grid cell)
            </text>
            <text x={10} y={PADT + 6} className="fill-slate-400" fontSize={10}>Q</text>

            {/* initial pulse */}
            {showInitial && (
              <path d={pathOf(sim.initial)} fill="none" stroke="#fbbf24" strokeWidth={1.5}
                strokeDasharray="2 3" opacity={0.7} />
            )}
            {/* true translated pulse */}
            {showTruth && !sim.unstable && (
              <path d={pathOf(sim.truth)} fill="none" stroke="#64748b" strokeWidth={2} strokeDasharray="5 4" />
            )}
            {/* numerical pulse */}
            {!sim.unstable && (
              <path d={pathOf(sim.numerical)} fill="none" stroke="#0284c7" strokeWidth={2.5} />
            )}
            {/* peak markers */}
            {!sim.unstable && (
              <>
                <line x1={px(sim.trueCentre)} y1={py(1)} x2={px(sim.trueCentre)} y2={H - PADB}
                  stroke="#64748b" strokeWidth={1} strokeDasharray="2 2" opacity={0.5} />
              </>
            )}
            {sim.unstable && (
              <g>
                <rect x={PADL + 30} y={H / 2 - 26} width={W - PADL - PADR - 60} height={52} rx={8}
                  fill="#fef2f2" stroke="#dc2626" strokeWidth={1.5} />
                <text x={W / 2} y={H / 2 - 4} textAnchor="middle" className="fill-red-600 font-bold" fontSize={15}>
                  ⚠ Model Unstable
                </text>
                <text x={W / 2} y={H / 2 + 14} textAnchor="middle" className="fill-red-500" fontSize={11}>
                  C = {C.toFixed(2)} &gt; 1 — the solution blew up (NaN / ∞ guard tripped)
                </text>
              </g>
            )}
          </svg>

          {/* legend */}
          <div className="flex flex-wrap gap-x-4 gap-y-1 mt-1 text-[11px]">
            <span className="flex items-center gap-1"><span className="inline-block w-4 h-0.5 bg-amber-400" style={{ borderTop: '1.5px dashed' }} /> initial pulse</span>
            <span className="flex items-center gap-1"><span className="inline-block w-4 h-0.5" style={{ borderTop: '2px dashed #64748b' }} /> true wave (PDE, no smear)</span>
            <span className="flex items-center gap-1"><span className="inline-block w-4 h-0.5 bg-sky-600" /> numerical (upwind)</span>
          </div>
        </div>

        {/* Controls + readout */}
        <div className="w-full md:w-64 flex flex-col gap-3">
          <LabeledSlider label="Courant number  C = cΔt/Δx" value={C} min={0.05} max={1.5} step={0.05}
            display={C.toFixed(2)} onChange={setC} />
          <LabeledSlider label="Initial pulse sharpness" value={sigma} min={1.5} max={5} step={0.5}
            display={`σ = ${sigma.toFixed(1)} cells`} onChange={setSigma} />

          <div className="flex gap-3 text-[11px] text-slate-600">
            <label className="flex items-center gap-1 cursor-pointer">
              <input type="checkbox" checked={showTruth} onChange={(e) => setShowTruth(e.target.checked)} /> true wave
            </label>
            <label className="flex items-center gap-1 cursor-pointer">
              <input type="checkbox" checked={showInitial} onChange={(e) => setShowInitial(e.target.checked)} /> initial
            </label>
          </div>

          <div className="rounded-xl border border-slate-200 bg-slate-50 p-3 text-xs space-y-1.5">
            <div className="flex justify-between"><span className="text-slate-500">peak kept</span>
              <span className="font-bold tabular-nums" style={{ color: sim.unstable ? '#dc2626' : retentionColor(sim.retention) }}>
                {sim.unstable ? '—' : `${rPct}%`}
              </span>
            </div>
            <div className="flex justify-between"><span className="text-slate-500">numerical diffusion ∝ (1−C)</span>
              <span className="font-mono tabular-nums text-slate-700">{diffStrength.toFixed(2)}</span></div>
            <div className="flex justify-between"><span className="text-slate-500">time steps to travel</span>
              <span className="font-mono tabular-nums text-slate-700">{sim.unstable ? '✗' : sim.nSteps}</span></div>
            <hr className="border-slate-200" />
            <div className="flex justify-between"><span className="text-slate-500">Δx (fixed)</span>
              <span className="font-mono tabular-nums text-slate-700">{DX_FIXED} m</span></div>
            <div className="flex justify-between"><span className="text-slate-500">c (fixed)</span>
              <span className="font-mono tabular-nums text-slate-700">{C_FIXED} m/s</span></div>
            <div className="flex justify-between"><span className="text-slate-500">⇒ Δt = C·Δx/c</span>
              <span className="font-mono tabular-nums font-bold text-indigo-700">{dt.toFixed(0)} s</span></div>
          </div>
        </div>
      </div>

      {/* The Δt trap callout */}
      <div className="px-5 pb-5">
        <div className="rounded-xl border border-amber-200 bg-amber-50 p-3 text-[13px] text-amber-900 leading-relaxed">
          <strong>The Δt trap.</strong> With the grid (Δx) and the wave (c) fixed, the <em>C slider is the Δt
          knob</em>. Drag C <em>down</em> — a smaller, "safer" time step — and the peak gets <em>worse</em>: more
          steps, each one averaging neighbours, so more smear. The sweet spot is <strong>C = 1</strong> (exact,
          no smear), not C → 0. This is why OPM runs <code className="font-mono">CFL_TARGET = 0.85</code> instead
          of a tiny step: a sharp, physical peak.
        </div>
      </div>
    </div>
  );
}
