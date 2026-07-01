'use client';

import React, { useMemo, useState } from 'react';

// ─────────────────────────────────────────────────────────────────────────
// Green-Ampt infiltration-excess (Horton) runoff, ported conceptually from
// runoff_input.py's RunoffEngine._update_ga_F / get_effective_1d:
//
//   f_p(F) = K_v · (1 + ψ·Δθ0 / max(F, floor))     infiltration capacity [m/s]
//   f      = min(P, f_p)                            actual infiltration rate
//   F^{n+1} = F^n + f·Δt                             forward-Euler cumulative F
//
// Runoff begins once f_p drops to P (the crossing). Solving f_p(F_c) = P:
//   F_c = K_v·ψ·Δθ0 / (P - K_v)         only when P > K_v
// ─────────────────────────────────────────────────────────────────────────

const MM_HR_TO_M_S = 1 / 1000 / 3600;
const M_S_TO_MM_HR = 1000 * 3600;
const F_FLOOR_M = 1e-9;

interface SimPoint {
  step: number;
  F_m: number; // cumulative infiltration [m]
  fp_mmhr: number; // infiltration capacity [mm/hr]
  f_mmhr: number; // actual infiltration rate this step [mm/hr]
}

/** Forward-Euler Green-Ampt simulation, mirroring runoff_input.py's _update_ga_F. */
function simulate(
  Kv_mmhr: number,
  psi_m: number,
  dtheta0: number,
  P_mmhr: number,
  dt_s: number,
  nSteps: number
): SimPoint[] {
  const Kv = Kv_mmhr * MM_HR_TO_M_S;
  const P = P_mmhr * MM_HR_TO_M_S;
  let F = F_FLOOR_M;
  const out: SimPoint[] = [];
  for (let step = 1; step <= nSteps; step++) {
    const fp = Kv * (1 + (psi_m * dtheta0) / Math.max(F, F_FLOOR_M));
    const f = Math.min(P, fp);
    F = F + f * dt_s;
    out.push({
      step,
      F_m: F,
      fp_mmhr: fp * M_S_TO_MM_HR,
      f_mmhr: f * M_S_TO_MM_HR,
    });
  }
  return out;
}

/** Cumulative infiltration F_c [m] at the crossing f_p = P. Null if P <= Kv (no Horton runoff ever). */
function crossingFc(Kv_mmhr: number, psi_m: number, dtheta0: number, P_mmhr: number): number | null {
  const Kv = Kv_mmhr * MM_HR_TO_M_S;
  const P = P_mmhr * MM_HR_TO_M_S;
  if (P <= Kv) return null;
  return (Kv * psi_m * dtheta0) / (P - Kv);
}

// ─────────────────────────────────────────────────────────────────────────
// Rawls (1983) texture presets — ψ [m] and K_v [mm/hr], from runoff_input.py
// ─────────────────────────────────────────────────────────────────────────
interface TexturePreset {
  name: string;
  psi_m: number;
  Kv_mmhr: number;
}

const RAWLS_PRESETS: TexturePreset[] = [
  { name: 'Sand', psi_m: 0.0495, Kv_mmhr: 117 },
  { name: 'Sandy loam', psi_m: 0.11, Kv_mmhr: 23 },
  { name: 'Loam', psi_m: 0.089, Kv_mmhr: 13 },
  { name: 'Silt loam', psi_m: 0.167, Kv_mmhr: 7 },
  { name: 'Clay loam', psi_m: 0.209, Kv_mmhr: 2 },
  { name: 'Clay', psi_m: 0.316, Kv_mmhr: 0.5 },
];

// Defaults reproduce the worked example: Fc ≈ 16.7 mm, crossing ≈ 50 steps,
// post-threshold runoff = 75% of rain.
const DEFAULT_KV_MMHR = 5;
const DEFAULT_PSI_M = 0.2;
const DEFAULT_DTHETA0 = 0.25;
const DEFAULT_P_MMHR = 20;
const DEFAULT_DT_S = 60;
const N_STEPS = 150;

// ─────────────────────────────────────────────────────────────────────────
// Slider sub-component (matches ManningCelerityWidget conventions)
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
}

function LabeledSlider({ label, value, min, max, step, display, onChange, accent }: SliderProps) {
  return (
    <div className="flex flex-col gap-0.5">
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
        onChange={(e) => onChange(Number(e.target.value))}
        className={`w-full h-1.5 rounded-full cursor-pointer ${accent ?? 'accent-amber-600'}`}
      />
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────
// Main component
// ─────────────────────────────────────────────────────────────────────────
export default function GreenAmptWidget() {
  const [Kv, setKv] = useState(DEFAULT_KV_MMHR); // mm/hr
  const [psi, setPsi] = useState(DEFAULT_PSI_M); // m
  const [dtheta0, setDtheta0] = useState(DEFAULT_DTHETA0); // -
  const [P, setP] = useState(DEFAULT_P_MMHR); // mm/hr
  const [dt, setDt] = useState(DEFAULT_DT_S); // s
  const [activePreset, setActivePreset] = useState<string | null>('— (custom defaults)');

  const sim = useMemo(() => simulate(Kv, psi, dtheta0, P, dt, N_STEPS), [Kv, psi, dtheta0, P, dt]);
  const Fc_m = useMemo(() => crossingFc(Kv, psi, dtheta0, P), [Kv, psi, dtheta0, P]);
  const noRunoff = Fc_m === null;

  // Step index (1-based) at which the simulated f_p first drops to/below P.
  const crossingStep = useMemo(() => {
    if (noRunoff) return null;
    const hit = sim.find((pt) => pt.fp_mmhr <= P);
    return hit ? hit.step : null;
  }, [sim, P, noRunoff]);

  const crossingTimeMin = crossingStep !== null ? (crossingStep * dt) / 60 : null;
  const postThresholdRunoffFrac = !noRunoff ? Math.max(P - Kv, 0) / P : 0;

  // ── Chart geometry ───────────────────────────────────────────────────
  const SVG_W = 460;
  const SVG_H = 280;
  const PAD = { left: 52, right: 16, top: 16, bottom: 40 };
  const plotW = SVG_W - PAD.left - PAD.right;
  const plotH = SVG_H - PAD.top - PAD.bottom;

  // X axis = cumulative infiltration F [mm] (more physically meaningful than raw step count
  // since f_p is a function of F, not of t directly). Cap at a sensible view window.
  const F_mm_series = sim.map((pt) => pt.F_m * 1000);
  const Fc_mm = Fc_m !== null ? Fc_m * 1000 : null;
  const F_view_max = Math.max(
    F_mm_series[F_mm_series.length - 1] ?? 1,
    Fc_mm !== null ? Fc_mm * 1.6 : 0,
    1
  );

  const fp_series = sim.map((pt) => pt.fp_mmhr);
  const fp_view_max = Math.max(...fp_series.slice(0, 5), P, Kv) * 1.15;
  // Clamp the y-axis so the (very large, near-F=0) initial capacity spike doesn't
  // dominate the chart — show enough of the early decline to read clearly.
  const Y_MAX = Math.min(fp_view_max, Math.max(P * 3, Kv * 4, 1));

  function xOf(F_mm: number): number {
    return PAD.left + Math.min(F_mm / F_view_max, 1) * plotW;
  }
  function yOf(rate_mmhr: number): number {
    return PAD.top + plotH - Math.min(rate_mmhr / Y_MAX, 1) * plotH;
  }

  const curvePoints = sim
    .map((pt) => `${xOf(pt.F_m * 1000).toFixed(2)},${yOf(pt.fp_mmhr).toFixed(2)}`)
    .join(' ');

  // Shaded "runoff excess" region: where fp curve sits below the P line (P > fp).
  // Build a closed polygon following the curve, then closing along the P line.
  const shadePoints = useMemo(() => {
    if (noRunoff) return '';
    const pts: string[] = [];
    for (const pt of sim) {
      if (pt.fp_mmhr <= P) {
        pts.push(`${xOf(pt.F_m * 1000).toFixed(2)},${yOf(pt.fp_mmhr).toFixed(2)}`);
      }
    }
    if (pts.length === 0) return '';
    const firstX = pts[0].split(',')[0];
    const lastX = pts[pts.length - 1].split(',')[0];
    return `${firstX},${yOf(P).toFixed(2)} ` + pts.join(' ') + ` ${lastX},${yOf(P).toFixed(2)}`;
  }, [sim, P, noRunoff]); // eslint-disable-line react-hooks/exhaustive-deps

  const gridYFracs = [0, 0.25, 0.5, 0.75, 1.0];
  const gridXFracs = [0, 0.25, 0.5, 0.75, 1.0];

  function handlePreset(preset: TexturePreset) {
    setKv(preset.Kv_mmhr);
    setPsi(preset.psi_m);
    setActivePreset(preset.name);
  }

  return (
    <div className="not-prose my-8 rounded-2xl border border-slate-200 bg-white shadow-xl overflow-hidden font-sans">
      {/* Header */}
      <div className="bg-gradient-to-r from-amber-700 to-orange-900 px-6 py-4">
        <h3 className="text-white font-bold text-lg tracking-tight">
          Green-Ampt Infiltration Capacity
        </h3>
        <p className="text-amber-200 text-sm mt-0.5">
          f_p = K_v(1 + ψ·Δθ₀/F) — capacity falls as the wetting front advances
        </p>
      </div>

      <div className="p-6 flex flex-col lg:flex-row gap-6 items-start">
        {/* ---------------------------------------------------------------- */}
        {/* LEFT PANEL — sliders + presets                                    */}
        {/* ---------------------------------------------------------------- */}
        <div className="flex flex-col gap-5 min-w-0 w-full lg:w-80">
          <div className="flex flex-col gap-3">
            <LabeledSlider
              label="K_v — vertical sat. conductivity"
              value={Kv}
              min={0.1}
              max={150}
              step={0.1}
              display={`${Kv.toFixed(1)} mm/hr`}
              onChange={(v) => {
                setKv(v);
                setActivePreset(null);
              }}
            />
            <LabeledSlider
              label="ψ — wetting-front suction"
              value={psi}
              min={0.01}
              max={0.4}
              step={0.001}
              display={`${psi.toFixed(3)} m`}
              onChange={(v) => {
                setPsi(v);
                setActivePreset(null);
              }}
            />
            <LabeledSlider
              label="Δθ₀ — initial moisture deficit"
              value={dtheta0}
              min={0.01}
              max={0.5}
              step={0.01}
              display={dtheta0.toFixed(2)}
              onChange={setDtheta0}
            />
            <LabeledSlider
              label="P — rainfall rate"
              value={P}
              min={1}
              max={80}
              step={0.5}
              display={`${P.toFixed(1)} mm/hr`}
              onChange={setP}
              accent="accent-sky-600"
            />
            <LabeledSlider
              label="Δt — timestep"
              value={dt}
              min={10}
              max={300}
              step={10}
              display={`${dt} s`}
              onChange={setDt}
              accent="accent-slate-500"
            />
          </div>

          {/* Rawls preset table */}
          <div>
            <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-1.5">
              Rawls (1983) soil-texture presets
            </p>
            <div className="overflow-x-auto">
              <table className="w-full text-xs border-collapse">
                <thead>
                  <tr className="bg-slate-100 text-slate-600">
                    <th className="px-2 py-1 text-left border border-slate-200">Texture</th>
                    <th className="px-2 py-1 text-right border border-slate-200">ψ (m)</th>
                    <th className="px-2 py-1 text-right border border-slate-200">K_v (mm/hr)</th>
                  </tr>
                </thead>
                <tbody>
                  {RAWLS_PRESETS.map((preset, idx) => {
                    const isActive = activePreset === preset.name;
                    return (
                      <tr
                        key={preset.name}
                        onClick={() => handlePreset(preset)}
                        className={`cursor-pointer transition-colors ${
                          isActive
                            ? 'bg-amber-100 font-semibold'
                            : idx % 2 === 0
                            ? 'bg-white hover:bg-amber-50'
                            : 'bg-slate-50 hover:bg-amber-50'
                        }`}
                        title={`Click to set ψ=${preset.psi_m} m, K_v=${preset.Kv_mmhr} mm/hr`}
                      >
                        <td className="px-2 py-1 border border-slate-200 font-mono">
                          {preset.name}
                        </td>
                        <td className="px-2 py-1 text-right border border-slate-200 font-mono">
                          {preset.psi_m.toFixed(3)}
                        </td>
                        <td className="px-2 py-1 text-right border border-slate-200 font-mono">
                          {preset.Kv_mmhr}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            <p className="text-xs text-slate-400 mt-1">
              Click a row to load ψ and K_v together — coarser soils have small ψ &amp; large K_v,
              fine soils the reverse.
            </p>
          </div>

          {/* Gotcha callout — vertical vs lateral Ksat */}
          <div className="bg-amber-50 border-2 border-amber-400 rounded-lg px-4 py-3 text-sm text-amber-900 shadow-sm">
            <p className="font-bold text-amber-800 mb-1">⚠ Don&apos;t confuse the two K_sats</p>
            <p>
              This widget&apos;s <strong>K_v</strong> is the <strong>vertical</strong>{' '}
              surface-infiltration rate (typically 1&ndash;50 mm/hr) — the soil&apos;s ability to
              absorb rain straight down. OPM also has a completely separate{' '}
              <strong>lateral</strong> sandbox-drainage transmissivity, <code>OPM_K_SAT</code>,{' '}
              44 m/day ≈ 1830 mm/hr — about <strong>1000× larger</strong>.
            </p>
            <p className="mt-1.5">
              Using the lateral value here would make f_p ≫ P everywhere, so the soil would never
              be overwhelmed and Horton runoff would never appear — physically wrong. The names
              look similar; the roles are opposite.
            </p>
          </div>
        </div>

        {/* ---------------------------------------------------------------- */}
        {/* RIGHT PANEL — chart + readouts                                    */}
        {/* ---------------------------------------------------------------- */}
        <div className="flex flex-col gap-4 flex-1 min-w-0">
          <div>
            <p className="text-xs font-semibold text-slate-500 mb-1">
              Infiltration capacity f_p vs. cumulative infiltration F — shaded region = Horton
              runoff excess
            </p>
            <svg
              width={SVG_W}
              height={SVG_H}
              viewBox={`0 0 ${SVG_W} ${SVG_H}`}
              className="block w-full"
              style={{ background: '#fffbeb', borderRadius: 8, border: '1px solid #fde68a', maxWidth: '100%' }}
            >
              {/* Gridlines */}
              {gridYFracs.map((f) => (
                <line
                  key={`gy-${f}`}
                  x1={PAD.left}
                  y1={PAD.top + f * plotH}
                  x2={SVG_W - PAD.right}
                  y2={PAD.top + f * plotH}
                  stroke="#fde68a"
                  strokeWidth={1}
                />
              ))}
              {gridXFracs.map((f) => (
                <line
                  key={`gx-${f}`}
                  x1={PAD.left + f * plotW}
                  y1={PAD.top}
                  x2={PAD.left + f * plotW}
                  y2={PAD.top + plotH}
                  stroke="#fde68a"
                  strokeWidth={1}
                />
              ))}

              {/* Shaded excess region (P > f_p) */}
              {!noRunoff && shadePoints && (
                <polygon points={shadePoints} fill="#dc2626" opacity={0.18} />
              )}

              {/* f_p capacity curve */}
              <polyline points={curvePoints} fill="none" stroke="#92400e" strokeWidth={2.5} strokeLinejoin="round" />

              {/* Rainfall rate P reference line */}
              <line
                x1={PAD.left}
                y1={yOf(P)}
                x2={SVG_W - PAD.right}
                y2={yOf(P)}
                stroke="#0284c7"
                strokeWidth={2}
                strokeDasharray="6 4"
              />
              <text x={SVG_W - PAD.right - 4} y={yOf(P) - 5} fontSize={10} textAnchor="end" fill="#0369a1" fontFamily="monospace">
                P = {P.toFixed(1)} mm/hr
              </text>

              {/* K_v asymptote reference line */}
              <line
                x1={PAD.left}
                y1={yOf(Kv)}
                x2={SVG_W - PAD.right}
                y2={yOf(Kv)}
                stroke="#78716c"
                strokeWidth={1}
                strokeDasharray="2 3"
                opacity={0.8}
              />
              <text x={PAD.left + 4} y={yOf(Kv) - 4} fontSize={9} fill="#57534e" fontFamily="monospace">
                K_v = {Kv.toFixed(1)} mm/hr
              </text>

              {/* Crossing point marker */}
              {!noRunoff && Fc_mm !== null && Fc_mm <= F_view_max && (
                <>
                  <line
                    x1={xOf(Fc_mm)}
                    y1={PAD.top}
                    x2={xOf(Fc_mm)}
                    y2={PAD.top + plotH}
                    stroke="#dc2626"
                    strokeWidth={1.5}
                    strokeDasharray="3 3"
                  />
                  <circle cx={xOf(Fc_mm)} cy={yOf(P)} r={5} fill="#dc2626" stroke="white" strokeWidth={1.5} />
                  <text
                    x={xOf(Fc_mm) + 6}
                    y={yOf(P) + 14}
                    fontSize={10}
                    fontWeight="bold"
                    fill="#b91c1c"
                    fontFamily="monospace"
                  >
                    F_c = {Fc_mm.toFixed(1)} mm
                  </text>
                </>
              )}

              {/* Axes */}
              <line x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={PAD.top + plotH} stroke="#78716c" strokeWidth={1.5} />
              <line
                x1={PAD.left}
                y1={PAD.top + plotH}
                x2={SVG_W - PAD.right}
                y2={PAD.top + plotH}
                stroke="#78716c"
                strokeWidth={1.5}
              />

              {/* Y axis labels */}
              {gridYFracs.map((f) => (
                <text
                  key={`yl-${f}`}
                  x={PAD.left - 5}
                  y={PAD.top + plotH - f * plotH + 3}
                  fontSize={9}
                  textAnchor="end"
                  fill="#78716c"
                  fontFamily="monospace"
                >
                  {(f * Y_MAX).toFixed(0)}
                </text>
              ))}

              {/* X axis labels */}
              {gridXFracs.map((f) => (
                <text
                  key={`xl-${f}`}
                  x={PAD.left + f * plotW}
                  y={PAD.top + plotH + 14}
                  fontSize={9}
                  textAnchor="middle"
                  fill="#78716c"
                  fontFamily="monospace"
                >
                  {(f * F_view_max).toFixed(0)}
                </text>
              ))}

              {/* Axis titles */}
              <text x={PAD.left + plotW / 2} y={SVG_H - 6} fontSize={10} textAnchor="middle" fill="#451a03" fontFamily="sans-serif">
                cumulative infiltration F (mm)
              </text>
              <text
                x={14}
                y={PAD.top + plotH / 2}
                fontSize={10}
                textAnchor="middle"
                fill="#451a03"
                fontFamily="sans-serif"
                transform={`rotate(-90, 14, ${PAD.top + plotH / 2})`}
              >
                rate (mm/hr)
              </text>
            </svg>
          </div>

          {/* Numeric readout */}
          {noRunoff ? (
            <div className="bg-emerald-50 border border-emerald-300 rounded-lg px-4 py-3 text-sm text-emerald-900">
              <strong>No Horton runoff:</strong> rainfall (P = {P.toFixed(1)} mm/hr) never exceeds
              soil capacity (K_v = {Kv.toFixed(1)} mm/hr). Since P ≤ K_v, f_p never falls to P — the
              soil always wins, all rain infiltrates indefinitely.
            </div>
          ) : (
            <div className="bg-slate-50 border border-slate-200 rounded-xl px-4 py-3 text-xs font-mono text-slate-700 leading-relaxed">
              <div className="font-bold text-slate-800 text-sm mb-1.5 font-sans">
                At your current settings
              </div>
              <div>
                F_c = K_v·ψ·Δθ₀ / (P − K_v) ={' '}
                <strong className="text-red-700">{Fc_mm?.toFixed(2)} mm</strong>
              </div>
              <div>
                Crossing reached at step{' '}
                <strong className="text-red-700">{crossingStep ?? '—'}</strong>
                {crossingTimeMin !== null && (
                  <>
                    {' '}
                    (t ≈ <strong className="text-red-700">{crossingTimeMin.toFixed(0)} min</strong>{' '}
                    at Δt = {dt} s)
                  </>
                )}
              </div>
              <div>
                After crossing, f_p settles toward K_v ={' '}
                <strong className="text-slate-800">{Kv.toFixed(1)} mm/hr</strong>, so runoff = P −
                K_v = <strong className="text-slate-800">{Math.max(P - Kv, 0).toFixed(1)} mm/hr</strong>
              </div>
              <div className="mt-1">
                Post-threshold runoff fraction = (P − K_v)/P ={' '}
                <strong className="text-orange-700">{(postThresholdRunoffFrac * 100).toFixed(0)}%</strong>{' '}
                of the rain
              </div>
            </div>
          )}

          {/* Key relations box */}
          <div className="bg-slate-50 border border-slate-200 rounded-xl px-4 py-3 text-xs text-slate-700 font-mono leading-relaxed">
            <div className="font-bold text-slate-800 text-sm mb-1 font-sans">Key relations</div>
            <div>f_p = K_v · (1 + ψ·Δθ₀ / F) &nbsp; (infiltration capacity)</div>
            <div>f = min(P, f_p), &nbsp; F&#8319;⁺¹ = F&#8319; + f·Δt &nbsp; (forward Euler)</div>
            <div className="text-red-700 font-semibold mt-1">
              F_c = K_v·ψ·Δθ₀ / (P − K_v) &nbsp; (crossing, only if P &gt; K_v)
            </div>
            <div className="mt-1 text-slate-500">
              Implementation note: real code floors F at 1e-9 m so f_p doesn&apos;t divide by
              exactly zero when F = 0.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
