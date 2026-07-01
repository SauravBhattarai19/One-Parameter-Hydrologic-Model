'use client';

import React, { useMemo, useState } from 'react';

// ─────────────────────────────────────────────────────────────────────────
// OPM's unified per-cell effective runoff, ported conceptually from
// runoff_input.py's RunoffEngine._opm_effective_runoff:
//
//   pervious_frac = 1.0          if cell is in the VSA (saturation-excess)
//                  = excess_frac  otherwise (infiltration-excess / Horton)
//
//   r_eff = P · [ Imp + (1 − Imp) · pervious_frac ]
//
// Decomposed into three additive, mass-balance-attributable streams:
//   r_imperv = P · Imp                                  (urban shedding)
//   r_Dunne  = P · (1 − Imp) · 1[in VSA]                 (saturation-excess)
//   r_Horton = P · (1 − Imp) · excess_frac · 1[off VSA]  (infiltration-excess)
//
// By construction r_imperv + r_Dunne + r_Horton === r_eff exactly.
// ─────────────────────────────────────────────────────────────────────────

interface Decomposition {
  perviousFrac: number;
  rEffFrac: number; // r_eff / P
  rImpervFrac: number; // r_imperv / P
  rDunneFrac: number; // r_Dunne / P
  rHortonFrac: number; // r_Horton / P
}

function decompose(imp: number, inVSA: boolean, excessFrac: number): Decomposition {
  const perviousFrac = inVSA ? 1.0 : excessFrac;
  const rImpervFrac = imp;
  const rDunneFrac = (1 - imp) * (inVSA ? 1 : 0);
  const rHortonFrac = (1 - imp) * (inVSA ? 0 : excessFrac);
  const rEffFrac = imp + (1 - imp) * perviousFrac;
  return { perviousFrac, rEffFrac, rImpervFrac, rDunneFrac, rHortonFrac };
}

// ─────────────────────────────────────────────────────────────────────────
// Presets — exact reference table from the spec
// ─────────────────────────────────────────────────────────────────────────
interface Preset {
  label: string;
  imp: number;
  inVSA: boolean;
  excessFrac: number;
  expected: number; // r_eff / P, for the readout / sanity check
}

const PRESETS: Preset[] = [
  { label: 'Rural, saturated', imp: 0, inVSA: true, excessFrac: 0, expected: 1.0 },
  { label: 'Urban, saturated', imp: 0.3, inVSA: true, excessFrac: 0, expected: 1.0 },
  { label: 'Rural, dry, soil wins', imp: 0, inVSA: false, excessFrac: 0, expected: 0.0 },
  { label: 'Urban, dry, soil wins', imp: 0.3, inVSA: false, excessFrac: 0, expected: 0.3 },
  { label: 'Rural, dry, Horton', imp: 0, inVSA: false, excessFrac: 0.4, expected: 0.4 },
  { label: 'Urban, dry, Horton', imp: 0.3, inVSA: false, excessFrac: 0.4, expected: 0.58 },
];

// ─────────────────────────────────────────────────────────────────────────
// Slider sub-component (matches GreenAmptWidget / sibling conventions)
// ─────────────────────────────────────────────────────────────────────────
interface SliderProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  display?: string;
  onChange: (v: number) => void;
  accent?: string;
  disabled?: boolean;
}

function LabeledSlider({ label, value, min, max, step, display, onChange, accent, disabled }: SliderProps) {
  return (
    <div className={`flex flex-col gap-0.5 transition-opacity ${disabled ? 'opacity-40' : ''}`}>
      <div className="flex items-center justify-between">
        <label className="text-xs font-semibold text-slate-500">{label}</label>
        <span className="text-xs font-mono text-slate-700 tabular-nums">{display ?? value}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
        className={`w-full h-1.5 rounded-full cursor-pointer disabled:cursor-not-allowed ${accent ?? 'accent-amber-600'}`}
      />
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────
// Stacked bar — r_imperv / r_Dunne / r_Horton as fractions of P
// ─────────────────────────────────────────────────────────────────────────
const COLOR_IMPERV = '#d97706'; // amber-600
const COLOR_DUNNE = '#2563eb'; // blue-600
const COLOR_HORTON = '#0d9488'; // teal-600

function StackedBar({ d }: { d: Decomposition }) {
  const W = 460;
  const H = 64;
  const segs = [
    { frac: d.rImpervFrac, color: COLOR_IMPERV, label: 'r_imperv' },
    { frac: d.rDunneFrac, color: COLOR_DUNNE, label: 'r_Dunne' },
    { frac: d.rHortonFrac, color: COLOR_HORTON, label: 'r_Horton' },
  ];

  let xCursor = 0;
  const rects = segs.map((s) => {
    const w = Math.max(s.frac, 0) * W;
    const rect = { x: xCursor, w, ...s };
    xCursor += w;
    return rect;
  });
  const totalW = Math.min(xCursor, W);

  return (
    <div className="flex flex-col gap-1.5">
      <svg
        width={W}
        height={H}
        viewBox={`0 0 ${W} ${H}`}
        className="block w-full"
        style={{ maxWidth: '100%' }}
      >
        {/* Track for the "unused" (infiltrated) portion */}
        <rect x={0} y={8} width={W} height={H - 24} rx={6} fill="#f1f5f9" stroke="#e2e8f0" strokeWidth={1} />
        {/* Stacked segments */}
        {rects.map((r, i) =>
          r.w > 0.3 ? (
            <rect key={i} x={r.x} y={8} width={r.w} height={H - 24} fill={r.color} opacity={0.92} />
          ) : null
        )}
        {/* Outline over the full filled portion */}
        <rect x={0} y={8} width={totalW} height={H - 24} rx={totalW >= W - 1 ? 6 : 0} fill="none" stroke="#475569" strokeWidth={1} />
        {/* r_eff/P readout centered on the filled bar */}
        <text x={W / 2} y={8 + (H - 24) / 2 + 4} textAnchor="middle" fontSize={12} fontWeight={700} fill="#1e293b">
          r_eff / P = {d.rEffFrac.toFixed(2)}
        </text>
        {/* Scale ticks 0, 0.5, 1.0 */}
        {[0, 0.5, 1].map((f) => (
          <text key={f} x={f * W} y={H - 2} textAnchor={f === 0 ? 'start' : f === 1 ? 'end' : 'middle'} fontSize={9} fill="#94a3b8" fontFamily="monospace">
            {f.toFixed(1)}
          </text>
        ))}
      </svg>
      {/* Legend */}
      <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-slate-600">
        <span className="flex items-center gap-1.5">
          <span className="inline-block w-3 h-3 rounded-sm" style={{ background: COLOR_IMPERV }} />
          r_imperv/P = <span className="font-mono font-semibold">{d.rImpervFrac.toFixed(2)}</span> (urban shedding)
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block w-3 h-3 rounded-sm" style={{ background: COLOR_DUNNE }} />
          r_Dunne/P = <span className="font-mono font-semibold">{d.rDunneFrac.toFixed(2)}</span> (saturation-excess)
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block w-3 h-3 rounded-sm" style={{ background: COLOR_HORTON }} />
          r_Horton/P = <span className="font-mono font-semibold">{d.rHortonFrac.toFixed(2)}</span> (infiltration-excess)
        </span>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────
// Main component
// ─────────────────────────────────────────────────────────────────────────
export default function RunoffDecompositionWidget() {
  const [imp, setImp] = useState(0.3);
  const [inVSA, setInVSA] = useState(false);
  const [excessFrac, setExcessFrac] = useState(0.4);
  const [activePreset, setActivePreset] = useState<string | null>('Urban, dry, Horton');

  const d = useMemo(() => decompose(imp, inVSA, excessFrac), [imp, inVSA, excessFrac]);

  function applyPreset(p: Preset) {
    setImp(p.imp);
    setInVSA(p.inVSA);
    setExcessFrac(p.excessFrac);
    setActivePreset(p.label);
  }

  function manualChange(fn: () => void) {
    fn();
    setActivePreset(null);
  }

  return (
    <div className="not-prose my-8 rounded-2xl border border-slate-200 bg-white shadow-xl overflow-hidden font-sans">
      {/* Header */}
      <div className="bg-gradient-to-r from-sky-700 to-emerald-800 px-6 py-4">
        <h3 className="text-white font-bold text-lg tracking-tight">Runoff Decomposition</h3>
        <p className="text-sky-200 text-sm mt-0.5">
          r_eff = P·[Imp + (1−Imp)·pervious_frac] — three mechanisms, one formula
        </p>
      </div>

      <div className="p-6 flex flex-col lg:flex-row gap-6 items-start">
        {/* ---------------------------------------------------------------- */}
        {/* LEFT PANEL — controls + presets                                   */}
        {/* ---------------------------------------------------------------- */}
        <div className="flex flex-col gap-5 min-w-0 w-full lg:w-80">
          <div className="flex flex-col gap-4">
            <LabeledSlider
              label="Imp — impervious (urban) fraction"
              value={imp}
              min={0}
              max={1}
              step={0.01}
              display={imp.toFixed(2)}
              onChange={(v) => manualChange(() => setImp(v))}
              accent="accent-amber-600"
            />

            {/* In-VSA toggle */}
            <div className="flex flex-col gap-1">
              <div className="flex items-center justify-between">
                <label className="text-xs font-semibold text-slate-500">In VSA? (saturated)</label>
                <span className={`text-xs font-mono font-semibold ${inVSA ? 'text-blue-700' : 'text-slate-500'}`}>
                  {inVSA ? 'YES — Dunne' : 'NO — Horton path'}
                </span>
              </div>
              <button
                type="button"
                role="switch"
                aria-checked={inVSA}
                onClick={() => manualChange(() => setInVSA(!inVSA))}
                className={`relative w-full h-9 rounded-full transition-colors flex items-center px-1 ${
                  inVSA ? 'bg-blue-600' : 'bg-slate-300'
                }`}
              >
                <span
                  className={`absolute h-7 w-1/2 rounded-full bg-white shadow transition-transform flex items-center justify-center text-xs font-bold ${
                    inVSA ? 'translate-x-[calc(100%-0.25rem)] text-blue-700' : 'translate-x-0 text-slate-600'
                  }`}
                  style={{ transitionDuration: '150ms' }}
                >
                  {inVSA ? 'YES' : 'NO'}
                </span>
              </button>
            </div>

            <LabeledSlider
              label={`excess_frac — Green-Ampt infiltration-excess${inVSA ? '  (forced irrelevant)' : ''}`}
              value={excessFrac}
              min={0}
              max={1}
              step={0.01}
              display={excessFrac.toFixed(2)}
              onChange={(v) => manualChange(() => setExcessFrac(v))}
              accent="accent-teal-600"
              disabled={inVSA}
            />
            {inVSA && (
              <p className="text-xs text-blue-700 -mt-2 italic">
                Disabled: pervious_frac is forced to 1.0 inside the VSA, so excess_frac has zero
                effect on r_eff regardless of its value.
              </p>
            )}
          </div>

          {/* Key physical insight callout */}
          <div className="bg-blue-50 border-2 border-blue-400 rounded-lg px-4 py-3 text-sm text-blue-900 shadow-sm">
            <p className="font-bold text-blue-800 mb-1">Inside the VSA, impervious fraction is irrelevant</p>
            <p>
              When pervious_frac = 1 (cell saturated), r_eff = P[Imp + (1−Imp)] = <strong>P</strong>{' '}
              regardless of Imp — a saturated cell already sheds everything, so paving it changes
              nothing. Imp only matters <strong>outside</strong> the VSA.
            </p>
          </div>

          {/* Preset table */}
          <div>
            <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-1.5">
              Reference presets (click to load)
            </p>
            <div className="overflow-x-auto">
              <table className="w-full text-xs border-collapse">
                <thead>
                  <tr className="bg-slate-100 text-slate-600">
                    <th className="px-2 py-1 text-left border border-slate-200">Preset</th>
                    <th className="px-2 py-1 text-right border border-slate-200">Imp</th>
                    <th className="px-2 py-1 text-center border border-slate-200">VSA?</th>
                    <th className="px-2 py-1 text-right border border-slate-200">excess</th>
                    <th className="px-2 py-1 text-right border border-slate-200">r_eff/P</th>
                  </tr>
                </thead>
                <tbody>
                  {PRESETS.map((p, idx) => {
                    const isActive = activePreset === p.label;
                    return (
                      <tr
                        key={p.label}
                        onClick={() => applyPreset(p)}
                        className={`cursor-pointer transition-colors ${
                          isActive
                            ? 'bg-emerald-100 font-semibold'
                            : idx % 2 === 0
                            ? 'bg-white hover:bg-emerald-50'
                            : 'bg-slate-50 hover:bg-emerald-50'
                        }`}
                        title={`Click to set Imp=${p.imp}, VSA=${p.inVSA}, excess_frac=${p.excessFrac}`}
                      >
                        <td className="px-2 py-1 border border-slate-200 font-mono">{p.label}</td>
                        <td className="px-2 py-1 text-right border border-slate-200 font-mono">{p.imp.toFixed(1)}</td>
                        <td className="px-2 py-1 text-center border border-slate-200 font-mono">{p.inVSA ? 'Y' : 'N'}</td>
                        <td className="px-2 py-1 text-right border border-slate-200 font-mono">
                          {p.inVSA ? '–' : p.excessFrac.toFixed(1)}
                        </td>
                        <td className="px-2 py-1 text-right border border-slate-200 font-mono font-semibold text-emerald-700">
                          {p.expected.toFixed(2)}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* ---------------------------------------------------------------- */}
        {/* RIGHT PANEL — stacked bar + formula readout                       */}
        {/* ---------------------------------------------------------------- */}
        <div className="flex flex-col gap-4 flex-1 min-w-0">
          <div>
            <p className="text-xs font-semibold text-slate-500 mb-2">
              r_eff / P split into its three additive mechanisms
            </p>
            <div className="bg-slate-50 border border-slate-200 rounded-xl p-4">
              <StackedBar d={d} />
            </div>
          </div>

          {/* Numeric readout */}
          <div className="bg-slate-50 border border-slate-200 rounded-xl px-4 py-3 text-xs font-mono text-slate-700 leading-relaxed">
            <div className="font-bold text-slate-800 text-sm mb-1.5 font-sans">At your current settings</div>
            <div>
              pervious_frac = {inVSA ? '1.0  (in VSA)' : `excess_frac = ${excessFrac.toFixed(2)}  (off VSA)`}
            </div>
            <div className="mt-1">
              r_eff/P = Imp + (1−Imp)·pervious_frac = {imp.toFixed(2)} + {(1 - imp).toFixed(2)}×
              {d.perviousFrac.toFixed(2)} = <strong className="text-emerald-700">{d.rEffFrac.toFixed(2)}</strong>
            </div>
            <div className="mt-1.5 pt-1.5 border-t border-slate-200">
              r_imperv/P = {d.rImpervFrac.toFixed(2)} &nbsp;+&nbsp; r_Dunne/P = {d.rDunneFrac.toFixed(2)} &nbsp;+&nbsp;
              r_Horton/P = {d.rHortonFrac.toFixed(2)} &nbsp;=&nbsp;{' '}
              <strong className="text-slate-800">
                {(d.rImpervFrac + d.rDunneFrac + d.rHortonFrac).toFixed(2)}
              </strong>{' '}
              (sums exactly to r_eff/P — every drop is attributed to a mechanism)
            </div>
          </div>

          {/* Key relations box */}
          <div className="bg-slate-50 border border-slate-200 rounded-xl px-4 py-3 text-xs text-slate-700 font-mono leading-relaxed">
            <div className="font-bold text-slate-800 text-sm mb-1 font-sans">Key relations</div>
            <div>pervious_frac = 1.0 if in VSA, else excess_frac</div>
            <div>r_eff = P · [Imp + (1−Imp)·pervious_frac]</div>
            <div className="mt-1 text-amber-700">r_imperv = P·Imp</div>
            <div className="text-blue-700">r_Dunne = P·(1−Imp)·1[in VSA]</div>
            <div className="text-teal-700">r_Horton = P·(1−Imp)·excess_frac·1[off VSA]</div>
            <div className="mt-1 text-slate-500">r_imperv + r_Dunne + r_Horton ≡ r_eff (exact, by construction)</div>
          </div>
        </div>
      </div>
    </div>
  );
}
