'use client';

import React, { useState, useMemo } from 'react';

// ─────────────────────────────────────────────────────────────────────────
// GridCharacteristicWidget — the information / geometry view (mental model 4).
//
// Grid celerity  c_grid = Δx/Δt  = the fastest information can move on the
// mesh = exactly one cell per step (the upwind stencil reaches one cell up).
// Courant number  C = cΔt/Δx = c / c_grid = how many cells the wave crosses
// per step.  The exact update traces the characteristic back to the departure
// point xᵢ − cΔt = xᵢ − C·Δx, then the scheme LINEARLY INTERPOLATES between
// the two bracketing nodes:
//   C=1  → departure lands on node i−1 → exact copy, no error.
//   0<C<1→ departure lands between nodes → interpolation cuts the corner of a
//          curved profile → shaves the peak → THIS is numerical diffusion.
//   C>1  → departure falls OUTSIDE the stencil → domain of dependence
//          violated → unstable.
// ─────────────────────────────────────────────────────────────────────────

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

// curved "true" profile sampled on the grid (rising limb so corner-cut shows)
function qProfile(x: number): number { const z = (x - 4.3) / 1.1; return Math.exp(-z * z); }

// space-time grid geometry
const GW = 380, GH = 240;
const colX = (c: number) => 28 + c * 60;     // columns 0..5 → 28..328
const TOP = 46, BOT = 188;                    // t = n+1 (top), t = n (bottom)

export default function GridCharacteristicWidget() {
  const [C, setC] = useState(0.6);

  const colDep = 4 - C;                        // departure column on row n
  const inside = C <= 1.0001;                  // within stencil [node i−1, node i]
  const exact = Math.abs(C - 1) < 0.02;
  const verdict = exact ? 'exact' : inside ? 'diffuse' : 'unstable';
  const vColor = verdict === 'exact' ? '#16a34a' : verdict === 'diffuse' ? '#d97706' : '#dc2626';

  // interpolation panel numbers
  const q3 = qProfile(3), q4 = qProfile(4);
  const interpVal = C * q3 + (1 - C) * q4;     // = upwind blend = chord value at xdep
  const trueVal = qProfile(colDep);
  const gap = trueVal - interpVal;             // peak under-shot = diffusion (when inside)

  const bottomNodes = useMemo(() => [2, 3, 4, 5], []);

  // interpolation mini-plot geometry
  const PW = 300, PH = 150, X0 = 2.4, X1 = 4.9;
  const ipx = (x: number) => 26 + ((x - X0) / (X1 - X0)) * (PW - 40);
  const ipy = (q: number) => PH - 22 - q * (PH - 40);
  const curve = Array.from({ length: 50 }, (_, k) => {
    const x = X0 + (k / 49) * (X1 - X0);
    return `${k === 0 ? 'M' : 'L'}${ipx(x).toFixed(1)},${ipy(qProfile(x)).toFixed(1)}`;
  }).join(' ');

  return (
    <div className="not-prose my-6 rounded-2xl border border-slate-200 bg-white shadow-sm overflow-hidden">
      <div className="bg-gradient-to-r from-cyan-600 to-sky-700 px-5 py-3">
        <h3 className="text-white font-bold text-base">Can the wave outrun the grid?</h3>
        <p className="text-cyan-50 text-xs mt-0.5">
          C = how many cells the wave crosses per step. Trace it back to where the water came from — and see
          why missing the node smears the peak.
        </p>
      </div>

      <div className="p-5 grid lg:grid-cols-2 gap-5">
        {/* Panel A — space-time grid */}
        <div>
          <div className="text-xs font-bold text-slate-500 uppercase mb-1">space–time grid · trace the characteristic back</div>
          <svg viewBox={`0 0 ${GW} ${GH}`} className="w-full select-none">
            {/* domain-of-dependence wedge (between C=0 vertical and C=1 line) */}
            <path d={`M${colX(4)},${TOP} L${colX(4)},${BOT} L${colX(3)},${BOT} Z`}
              fill={inside ? '#dcfce7' : '#fee2e2'} opacity={0.6} />
            {/* grid lines */}
            {[TOP, BOT].map((y, k) => (
              <line key={k} x1={colX(0)} y1={y} x2={colX(5)} y2={y} stroke="#e2e8f0" strokeWidth={1} />
            ))}
            {/* time labels */}
            <text x={8} y={TOP + 4} fontSize={11} className="fill-slate-500 font-mono">n+1</text>
            <text x={12} y={BOT + 4} fontSize={11} className="fill-slate-500 font-mono">n</text>
            {/* C=1 reference (one cell / step = stencil edge) */}
            <line x1={colX(4)} y1={TOP} x2={colX(3)} y2={BOT} stroke="#94a3b8" strokeWidth={1.5} strokeDasharray="4 3" />
            <text x={colX(3.4)} y={BOT - 50} fontSize={9} className="fill-slate-400" transform={`rotate(-58 ${colX(3.4)} ${BOT - 50})`}>
              grid speed (C=1, 1 cell/step)
            </text>

            {/* characteristic */}
            <line x1={colX(4)} y1={TOP} x2={colX(colDep)} y2={BOT} stroke={vColor} strokeWidth={2.5} />
            {/* arrow head at departure */}
            <circle cx={colX(colDep)} cy={BOT} r={5} fill={vColor} />

            {/* bottom nodes */}
            {bottomNodes.map((c) => {
              const isStencil = c === 3 || c === 4;
              return (
                <g key={c}>
                  <circle cx={colX(c)} cy={BOT} r={isStencil ? 6 : 4}
                    fill={isStencil ? '#0ea5e9' : '#cbd5e1'} stroke="#fff" strokeWidth={1.5} />
                  <text x={colX(c)} y={BOT + 18} textAnchor="middle" fontSize={9} className="fill-slate-500 font-mono">
                    {c === 2 ? 'i−2' : c === 3 ? 'i−1' : c === 4 ? 'i' : 'i+1'}
                  </text>
                </g>
              );
            })}
            {/* target node */}
            <circle cx={colX(4)} cy={TOP} r={6} fill={vColor} stroke="#fff" strokeWidth={1.5} />
            <text x={colX(4)} y={TOP - 10} textAnchor="middle" fontSize={9} className="fill-slate-600 font-mono">solve here (i, n+1)</text>

            {/* departure label */}
            <text x={colX(colDep)} y={BOT + 32} textAnchor="middle" fontSize={9} style={{ fill: vColor }}>
              came from xᵢ − C·Δx
            </text>
          </svg>

          <div className="rounded-lg px-3 py-2 mt-1 text-[13px] font-semibold text-center"
            style={{ background: vColor + '18', color: vColor }}>
            {verdict === 'exact' && 'C = 1 — lands exactly on node i−1. Exact copy, zero numerical diffusion.'}
            {verdict === 'diffuse' && 'Departure lands between nodes → must interpolate → numerical diffusion.'}
            {verdict === 'unstable' && 'C > 1 — departure is OUTSIDE the stencil. Domain of dependence violated → unstable.'}
          </div>
        </div>

        {/* Panel B — interpolation corner-cut + velocities */}
        <div>
          <div className="text-xs font-bold text-slate-500 uppercase mb-1">why a miss smears it · interpolation vs truth</div>
          <svg viewBox={`0 0 ${PW} ${PH}`} className="w-full rounded-lg bg-slate-50">
            {/* axes */}
            <line x1={26} y1={PH - 22} x2={PW - 8} y2={PH - 22} stroke="#cbd5e1" strokeWidth={1} />
            {/* true curve */}
            <path d={curve} fill="none" stroke="#0ea5e9" strokeWidth={2} />
            {/* chord between node i−1 and node i (the interpolation line) */}
            <line x1={ipx(3)} y1={ipy(q3)} x2={ipx(4)} y2={ipy(q4)} stroke="#d97706" strokeWidth={2} strokeDasharray="4 3" />
            {/* nodes */}
            <circle cx={ipx(3)} cy={ipy(q3)} r={4} fill="#0ea5e9" />
            <circle cx={ipx(4)} cy={ipy(q4)} r={4} fill="#0ea5e9" />
            <text x={ipx(3)} y={PH - 8} textAnchor="middle" fontSize={9} className="fill-slate-500 font-mono">i−1</text>
            <text x={ipx(4)} y={PH - 8} textAnchor="middle" fontSize={9} className="fill-slate-500 font-mono">i</text>
            {/* departure vertical */}
            <line x1={ipx(colDep)} y1={ipy(0)} x2={ipx(colDep)} y2={ipy(Math.max(trueVal, interpVal)) - 4}
              stroke={vColor} strokeWidth={1} strokeDasharray="2 2" />
            {/* interpolated (chord) point + true point */}
            <circle cx={ipx(colDep)} cy={ipy(interpVal)} r={4} fill="#d97706" stroke="#fff" strokeWidth={1.5} />
            <circle cx={ipx(colDep)} cy={ipy(trueVal)} r={4} fill="#16a34a" stroke="#fff" strokeWidth={1.5} />
            {/* gap bracket */}
            {inside && gap > 0.005 && (
              <line x1={ipx(colDep) + 7} y1={ipy(trueVal)} x2={ipx(colDep) + 7} y2={ipy(interpVal)}
                stroke="#dc2626" strokeWidth={1.5} />
            )}
          </svg>
          <div className="flex flex-wrap gap-x-4 gap-y-0.5 text-[10px] mt-1">
            <span className="flex items-center gap-1"><span className="w-3 h-0.5 bg-sky-500 inline-block" /> true profile</span>
            <span className="flex items-center gap-1"><span className="w-3 h-0.5 inline-block" style={{ borderTop: '2px dashed #d97706' }} /> interpolation (the scheme)</span>
            <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-emerald-600 inline-block" /> true value</span>
            <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-amber-600 inline-block" /> computed value</span>
          </div>
          <div className="text-[12px] text-slate-600 mt-1">
            peak shortfall (true − computed) ={' '}
            <span className="font-mono font-bold" style={{ color: gap > 0.005 && inside ? '#dc2626' : '#16a34a' }}>
              {inside ? gap.toFixed(3) : 'n/a (outside stencil)'}
            </span> — that gap, every step, <em>is</em> the smear.
          </div>

          {/* velocity strip: grid vs wave vs water */}
          <div className="mt-3 rounded-xl border border-slate-200 bg-slate-50 p-3">
            <div className="text-[11px] font-bold text-slate-500 uppercase mb-2">three different speeds</div>
            {[
              { lab: 'grid celerity  Δx/Δt', frac: 1 / C, col: '#94a3b8', note: '1 cell / step (info limit)' },
              { lab: 'wave front  c', frac: 1, col: '#0ea5e9', note: 'the flood signal' },
              { lab: 'water  u = 0.6 c', frac: 0.6, col: '#1d4ed8', note: 'the molecules' },
            ].map((r, k) => (
              <div key={k} className="flex items-center gap-2 mb-1.5">
                <span className="w-28 shrink-0 text-[10px] text-slate-600">{r.lab}</span>
                <div className="flex-1 h-3 bg-slate-200 rounded">
                  <div className="h-3 rounded" style={{ width: `${Math.min(100, r.frac * 60)}%`, background: r.col }} />
                </div>
                <span className="w-28 shrink-0 text-[9px] text-slate-400">{r.note}</span>
              </div>
            ))}
            <p className="text-[11px] text-slate-500 mt-1">
              C = c ÷ (Δx/Δt) = wave speed ÷ grid speed. Keep the wave from out-running the grid: C ≤ 1.
            </p>
          </div>
        </div>
      </div>

      <div className="px-5 pb-5">
        <LabeledSlider label="Courant number  C  (drag past 1 to break the grid)" value={C} min={0.2} max={1.6} step={0.05}
          display={C.toFixed(2)} onChange={setC} />
      </div>
    </div>
  );
}
