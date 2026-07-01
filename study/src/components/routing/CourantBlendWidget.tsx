'use client';

import React, { useState, useMemo } from 'react';

// ─────────────────────────────────────────────────────────────────────────
// CourantBlendWidget — the update rule as a weighted average (mental model 1).
//
//   Qᵢⁿ⁺¹ = C·Qᵢ₋₁ⁿ + (1−C)·Qᵢⁿ
//
// For 0≤C≤1 the weights are both ≥0 and sum to 1 → a WEIGHTED AVERAGE: the
// result is trapped between the upstream and local values (bounded ⇒ stable).
// At C>1 the weight (1−C) goes NEGATIVE → no longer an average but an
// EXTRAPOLATION → the result escapes the band and grows every step ⇒ blow-up.
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

const SPIKE = [2, 2, 12, 2, 2, 2, 2, 2, 2]; // baseflow 2, one spike of 12
const BASE = 2;

function march(C: number, steps: number): number[][] {
  const rows: number[][] = [[...SPIKE]];
  let Q = [...SPIKE];
  for (let s = 0; s < steps; s++) {
    const Qn = [...Q];
    Qn[0] = BASE;
    for (let i = 1; i < Q.length; i++) Qn[i] = C * Q[i - 1] + (1 - C) * Q[i];
    Q = Qn;
    rows.push([...Q]);
  }
  return rows;
}

// colour a marched value: in-band grey-blue, overshoot/undershoot red
function cellStyle(v: number): { bg: string; fg: string } {
  if (!isFinite(v) || v > 12.001 || v < BASE - 0.001) return { bg: '#fee2e2', fg: '#b91c1c' };
  const t = Math.max(0, Math.min(1, (v - BASE) / (12 - BASE)));
  return { bg: `rgb(${Math.round(224 - t * 5)},${Math.round(242 - t * 60)},${Math.round(254 - t * 11)})`, fg: '#0f172a' };
}

export default function CourantBlendWidget() {
  const [C, setC] = useState(0.6);
  const [up, setUp] = useState(10);
  const [loc, setLoc] = useState(4);

  const result = C * up + (1 - C) * loc;
  const lo = Math.min(up, loc), hi = Math.max(up, loc);
  const inBand = result >= lo - 1e-9 && result <= hi + 1e-9;
  const wUp = C, wLoc = 1 - C;

  const rows = useMemo(() => march(C, 6), [C]);
  const blewUp = rows[rows.length - 1].some((v) => !isFinite(v) || v > 12.001 || v < BASE - 0.001);

  // bar geometry
  const BAR_MAX = 16, BH = 120;
  const barH = (q: number) => Math.max(2, (Math.min(Math.abs(q), BAR_MAX) / BAR_MAX) * BH);

  return (
    <div className="not-prose my-6 rounded-2xl border border-slate-200 bg-white shadow-sm overflow-hidden">
      <div className="bg-gradient-to-r from-emerald-600 to-sky-600 px-5 py-3">
        <h3 className="text-white font-bold text-base">Average, or extrapolation? The weight (1−C) decides</h3>
        <p className="text-emerald-50 text-xs mt-0.5">
          The new value is a blend of upstream and here. Two non-negative weights = a safe average. A negative
          weight = a runaway.
        </p>
      </div>

      <div className="p-5 grid md:grid-cols-[auto_1fr] gap-6">
        {/* Bars */}
        <div>
          <div className="flex items-end gap-4" style={{ height: BH + 28 }}>
            {[
              { label: 'Qᵢ₋₁ⁿ', sub: 'upstream', v: up, fill: '#4f46e5' },
              { label: 'Qᵢⁿ', sub: 'here', v: loc, fill: '#64748b' },
              { label: 'Qᵢⁿ⁺¹', sub: 'result', v: result, fill: inBand ? '#16a34a' : '#dc2626' },
            ].map((b, k) => (
              <div key={k} className="flex flex-col items-center justify-end" style={{ height: BH + 28 }}>
                <span className="text-[11px] font-mono tabular-nums mb-0.5"
                  style={{ color: b.fill }}>{b.v.toFixed(2)}</span>
                <div className="w-12 rounded-t-md transition-all" style={{ height: barH(b.v), background: b.fill }} />
                <span className="text-[11px] font-semibold text-slate-700 mt-1">{b.label}</span>
                <span className="text-[9px] text-slate-400">{b.sub}</span>
              </div>
            ))}
          </div>
          {/* band annotation */}
          <div className="mt-2 text-[11px] text-center">
            <span className={inBand ? 'text-emerald-700' : 'text-red-600 font-semibold'}>
              {inBand
                ? `inside the [${lo}, ${hi}] band → bounded average`
                : `outside the [${lo}, ${hi}] band → extrapolation`}
            </span>
          </div>
        </div>

        {/* Controls + equation */}
        <div className="space-y-3">
          <div className="rounded-lg bg-slate-900 px-3 py-2 font-mono text-sm text-slate-100 overflow-x-auto">
            Qᵢⁿ⁺¹ ={' '}
            <span className="text-sky-300">{C.toFixed(2)}</span>·<span className="text-indigo-300">{up}</span>
            {' + '}
            <span className={wLoc < 0 ? 'text-red-400 font-bold' : 'text-emerald-300'}>({wLoc.toFixed(2)})</span>·
            <span className="text-slate-300">{loc}</span>
            {' = '}
            <span className={inBand ? 'text-emerald-300 font-bold' : 'text-red-400 font-bold'}>{result.toFixed(2)}</span>
          </div>

          <div className="flex flex-wrap gap-2">
            <span className="inline-flex items-center gap-1 rounded-full bg-sky-100 text-sky-800 px-2.5 py-1 text-xs font-semibold">
              weight on upstream: C = {wUp.toFixed(2)}
            </span>
            <span className={`inline-flex items-center gap-1 rounded-full px-2.5 py-1 text-xs font-semibold ${
              wLoc < 0 ? 'bg-red-100 text-red-700' : 'bg-emerald-100 text-emerald-800'}`}>
              weight on here: (1−C) = {wLoc.toFixed(2)} {wLoc < 0 ? '← NEGATIVE!' : ''}
            </span>
          </div>

          <LabeledSlider label="Courant number  C" value={C} min={0} max={1.5} step={0.05}
            display={C.toFixed(2)} onChange={setC} />
          <div className="grid grid-cols-2 gap-3">
            <LabeledSlider label="upstream Qᵢ₋₁" value={up} min={2} max={14} step={1} onChange={setUp} />
            <LabeledSlider label="here Qᵢ" value={loc} min={2} max={14} step={1} onChange={setLoc} />
          </div>
        </div>
      </div>

      {/* Multi-step march */}
      <div className="px-5 pb-5">
        <div className="text-xs font-semibold text-slate-500 mb-2">
          March a single spike 6 steps with this C — bounded decay, or runaway?
        </div>
        <div className="space-y-1 overflow-x-auto">
          {rows.map((row, s) => (
            <div key={s} className="flex items-center gap-1">
              <span className="w-12 shrink-0 text-[10px] text-slate-400 text-right pr-1">
                {s === 0 ? 'start' : `step ${s}`}
              </span>
              {row.map((v, i) => {
                const st = cellStyle(v);
                return (
                  <div key={i}
                    className="w-9 h-7 shrink-0 rounded flex items-center justify-center text-[10px] font-mono tabular-nums border border-slate-200"
                    style={{ background: st.bg, color: st.fg }}>
                    {isFinite(v) ? v.toFixed(1) : '∞'}
                  </div>
                );
              })}
            </div>
          ))}
        </div>
        <div className={`mt-2 rounded-lg px-3 py-2 text-[13px] ${
          blewUp ? 'bg-red-50 border border-red-200 text-red-800'
                 : 'bg-emerald-50 border border-emerald-200 text-emerald-900'}`}>
          {blewUp
            ? '⚠ C > 1: negative weight → the spike overshoots its band and grows. Red cells are above the start peak or below baseflow — the seed of a blow-up.'
            : 'C ≤ 1: the spike stays trapped between baseflow and its starting height, translating downstream while gently flattening. Bounded = stable.'}
        </div>
      </div>
    </div>
  );
}
