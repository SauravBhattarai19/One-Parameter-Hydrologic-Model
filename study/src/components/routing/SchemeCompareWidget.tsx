'use client';

import React, { useState, useMemo } from 'react';

// ---------------------------------------------------------------------------
// Physics constants and helpers
// ---------------------------------------------------------------------------

const BETA = 5 / 3;
const HYDROGRAPH = [2, 4, 7, 12, 10, 7, 5, 4, 3, 2, 2, 2];
const N = 6;
const DX = 500;
const DT_FORCED = 500;
const DT_STABLE = 300;
const DEFAULT_N_VAL = 0.04;
const DEFAULT_S0 = 0.001;
const DEFAULT_B = 10;
const Q_BASE = 2;

function computeAlpha(manN: number, S0: number, B: number): number {
  return Math.sqrt(S0) / (manN * Math.pow(B, 2 / 3));
}
const ALPHA = computeAlpha(DEFAULT_N_VAL, DEFAULT_S0, DEFAULT_B);

function aToQ(A: number): number {
  return ALPHA * Math.pow(Math.max(A, 0.001), BETA);
}
function qToA(Q: number): number {
  return Math.pow(Math.max(Q, 0.001) / ALPHA, 1 / BETA);
}
function cel(Q: number, A: number): number {
  return BETA * Q / Math.max(A, 0.001);
}

// ---------------------------------------------------------------------------
// Simulation runners
// ---------------------------------------------------------------------------

function runExplicit(dt: number): number[][] {
  const A_base = qToA(Q_BASE);
  let A = Array(N).fill(A_base);
  const out: number[][] = [];
  for (let t = 0; t < HYDROGRAPH.length; t++) {
    const Q = A.map(a => aToQ(a));
    out.push([...Q]);
    const Qup = [HYDROGRAPH[t], ...Q.slice(0, -1)];
    A = A.map((a, i) => Math.max(a - (dt / DX) * (Q[i] - Qup[i]), 0.001));
  }
  return out;
}

interface FluxLimitedResult {
  Q: number[][];
  V: number[][];
  Q_man: number[][];
  Q_lim_arr: number[][];
  limActive: boolean[][];
}

function runFluxLimited(dt: number): FluxLimitedResult {
  const A_base = qToA(Q_BASE);
  let A = Array(N).fill(A_base);
  const Q_out: number[][] = [];
  const V_out: number[][] = [];
  const Q_man: number[][] = [];
  const Q_lim_arr: number[][] = [];
  const limActive: boolean[][] = [];
  for (let t = 0; t < HYDROGRAPH.length; t++) {
    const Q = A.map(a => aToQ(a));
    const V = A.map(a => a * DX);
    const Qlim = Q.map((q, i) => Math.min(q, V[i] / dt));
    const active = Q.map((q, i) => Qlim[i] < q);
    Q_man.push([...Q]);
    Q_lim_arr.push([...Qlim]);
    limActive.push([...active]);
    V_out.push([...V]);
    Q_out.push([...Qlim]);
    const Qup = [HYDROGRAPH[t], ...Qlim.slice(0, -1)];
    A = A.map((a, i) => Math.max(a - (dt / DX) * (Qlim[i] - Qup[i]), 0.001));
  }
  return { Q: Q_out, V: V_out, Q_man, Q_lim_arr, limActive };
}

function runPreissmann(dt: number, theta: number): number[][] {
  const A_base = qToA(Q_BASE);
  let A = Array(N).fill(A_base);
  const out: number[][] = [];
  for (let t = 0; t < HYDROGRAPH.length; t++) {
    const Q = A.map(a => aToQ(a));
    out.push([...Q]);
    const cBar = Q.map((q, i) => cel(q, A[i]));
    const A_bc = qToA(HYDROGRAPH[Math.min(t + 1, HYDROGRAPH.length - 1)]);
    const b = cBar.map(c => 1 / dt + theta * c / DX);
    const l = cBar.map(c => theta * c / DX);
    const A_new: number[] = new Array(N);
    for (let i = 0; i < N; i++) {
      const A_up_old = i === 0 ? qToA(HYDROGRAPH[t]) : A[i - 1];
      const A_up_new = i === 0 ? A_bc : A_new[i - 1];
      const rhs = A[i] / dt - (1 - theta) * cBar[i] / DX * (A[i] - A_up_old);
      A_new[i] = Math.max((rhs + l[i] * A_up_new) / b[i], 0.001);
    }
    A = A_new;
  }
  return out;
}

// ---------------------------------------------------------------------------
// SVG polyline helper
// ---------------------------------------------------------------------------

interface SeriesConfig {
  values: number[];
  color: string;
  dashed?: boolean;
  label: string;
}

function OutletHydrograph({ series }: { series: SeriesConfig[] }) {
  const W = 440;
  const H = 150;
  const PAD = { top: 12, right: 16, bottom: 30, left: 44 };
  const innerW = W - PAD.left - PAD.right;
  const innerH = H - PAD.top - PAD.bottom;

  const allValues = series.flatMap(s => s.values).filter(v => isFinite(v) && v > -900);
  const yMin = Math.min(...allValues) * 0.9;
  const yMax = Math.max(...allValues) * 1.1;
  const steps = HYDROGRAPH.length;

  function toX(t: number) {
    return PAD.left + (t / (steps - 1)) * innerW;
  }
  function toY(v: number) {
    const clamped = Math.max(yMin, Math.min(yMax, v));
    return PAD.top + innerH - ((clamped - yMin) / (yMax - yMin)) * innerH;
  }

  const yTicks = 4;
  const yTickVals = Array.from({ length: yTicks + 1 }, (_, k) =>
    yMin + (k / yTicks) * (yMax - yMin)
  );

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ fontFamily: 'inherit' }}>
      {/* Grid lines */}
      {yTickVals.map((v, k) => (
        <line
          key={k}
          x1={PAD.left} y1={toY(v)}
          x2={PAD.left + innerW} y2={toY(v)}
          stroke="#e2e8f0" strokeWidth={1}
        />
      ))}
      {/* Y axis labels */}
      {yTickVals.map((v, k) => (
        <text
          key={k}
          x={PAD.left - 4} y={toY(v) + 4}
          textAnchor="end" fontSize={9} fill="#64748b"
        >
          {v.toFixed(1)}
        </text>
      ))}
      {/* X axis labels */}
      {Array.from({ length: steps }, (_, t) => (
        t % 3 === 0 ? (
          <text
            key={t}
            x={toX(t)} y={H - PAD.bottom + 14}
            textAnchor="middle" fontSize={9} fill="#64748b"
          >
            {t}
          </text>
        ) : null
      ))}
      {/* Axis labels */}
      <text x={PAD.left - 36} y={PAD.top + innerH / 2} fontSize={9} fill="#94a3b8"
        transform={`rotate(-90, ${PAD.left - 36}, ${PAD.top + innerH / 2})`}
        textAnchor="middle">
        Q (m³/s)
      </text>
      <text x={PAD.left + innerW / 2} y={H - 2} fontSize={9} fill="#94a3b8" textAnchor="middle">
        Time step
      </text>
      {/* Series */}
      {series.map((s, si) => {
        const pts = s.values
          .map((v, t) => isFinite(v) && v > -900 ? `${toX(t)},${toY(v)}` : null)
          .filter(Boolean)
          .join(' ');
        return (
          <polyline
            key={si}
            points={pts}
            fill="none"
            stroke={s.color}
            strokeWidth={s.dashed ? 1.5 : 2}
            strokeDasharray={s.dashed ? '5,4' : undefined}
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        );
      })}
      {/* Legend */}
      {series.map((s, si) => (
        <g key={si} transform={`translate(${PAD.left + 4}, ${PAD.top + 4 + si * 16})`}>
          <line
            x1={0} y1={6} x2={18} y2={6}
            stroke={s.color} strokeWidth={s.dashed ? 1.5 : 2}
            strokeDasharray={s.dashed ? '4,3' : undefined}
          />
          <text x={22} y={10} fontSize={9} fill="#334155">{s.label}</text>
        </g>
      ))}
    </svg>
  );
}

// ---------------------------------------------------------------------------
// Dependency diagram: explicit vs. implicit coupling (schematic, no numbers)
// ---------------------------------------------------------------------------

function DependencyDiagram({ implicit }: { implicit: boolean }) {
  const W = 200;
  const H = 110;
  const xs = [40, 100, 160];
  const yTop = 28;
  const yBot = 86;
  const r = 11;
  const accent = implicit ? '#7c3aed' : '#475569';

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ fontFamily: 'inherit' }}>
      <defs>
        <marker id={`arrow-${implicit ? 'imp' : 'exp'}`} viewBox="0 0 10 10" refX="8" refY="5"
          markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill={accent} />
        </marker>
      </defs>

      {/* diagonal arrows: n-row feeds n+1-row (own cell + neighbor) */}
      {xs.map((x, i) => (
        <g key={`diag-${i}`}>
          <line
            x1={x} y1={yBot - r} x2={x} y2={yTop + r}
            stroke={accent} strokeWidth={1.3}
            markerEnd={`url(#arrow-${implicit ? 'imp' : 'exp'})`}
          />
          {i > 0 && (
            <line
              x1={xs[i - 1]} y1={yBot - r} x2={x} y2={yTop + r}
              stroke={accent} strokeWidth={1.1}
              markerEnd={`url(#arrow-${implicit ? 'imp' : 'exp'})`}
            />
          )}
        </g>
      ))}

      {/* horizontal coupling arrows between n+1 cells: implicit only */}
      {implicit && xs.slice(0, -1).map((x, i) => (
        <line
          key={`horiz-${i}`}
          x1={x + r + 2} y1={yTop} x2={xs[i + 1] - r - 2} y2={yTop}
          stroke="#7c3aed" strokeWidth={1.6}
          markerEnd="url(#arrow-imp)"
        />
      ))}

      {/* n+1 row: unknowns (outline circles) */}
      {xs.map((x, i) => (
        <circle key={`n1-${i}`} cx={x} cy={yTop} r={r} fill="white"
          stroke={implicit ? '#7c3aed' : '#475569'} strokeWidth={1.8} />
      ))}
      {xs.map((x, i) => (
        <text key={`n1t-${i}`} x={x} y={yTop + 3.5} textAnchor="middle" fontSize={8} fill={implicit ? '#6d28d9' : '#334155'}>
          A{i + 1}ⁿ⁺¹
        </text>
      ))}

      {/* n row: knowns (filled circles) */}
      {xs.map((x, i) => (
        <circle key={`n0-${i}`} cx={x} cy={yBot} r={r} fill="#94a3b8" stroke="#64748b" strokeWidth={1} />
      ))}
      {xs.map((x, i) => (
        <text key={`n0t-${i}`} x={x} y={yBot + 3.5} textAnchor="middle" fontSize={8} fill="white">
          A{i + 1}ⁿ
        </text>
      ))}
    </svg>
  );
}

// ---------------------------------------------------------------------------
// Tab 1: Preissmann Implicit
// ---------------------------------------------------------------------------

interface PreissmannTabProps {
  theta: number;
  setTheta: (v: number) => void;
  stepIdx: number;
  setStepIdx: (v: number) => void;
  Q_ref: number[][];
  Q_preissmann: number[][];
  Q_explicit_unstable: number[][];
}

function PreissmannTab({
  theta,
  setTheta,
  stepIdx,
  setStepIdx,
  Q_ref,
  Q_preissmann,
  Q_explicit_unstable,
}: PreissmannTabProps) {
  const thetaLabel =
    theta < 0.1 ? 'Explicit (unstable!)' :
    theta < 0.55 ? 'Crank-Nicolson (2nd order)' :
    theta < 0.65 ? 'Preissmann standard (★)' :
    'Fully implicit (max stability)';

  // Thomas algorithm forward-sweep reveal: -1 = nothing shown, i = rows 0..i revealed
  const [sweepRow, setSweepRow] = useState(-1);

  // Matrix coefficients at stepIdx
  const cBar = Q_ref[stepIdx].map((q, i) => {
    const A = qToA(q);
    return cel(q, A);
  });
  const b_vals = cBar.map(c => 1 / DT_FORCED + theta * c / DX);
  const l_vals = cBar.map(c => theta * c / DX);

  // RHS: compute from reference A values at stepIdx
  const A_ref = Q_ref[stepIdx].map(q => qToA(q));
  const A_ref_prev = stepIdx > 0 ? Q_ref[stepIdx - 1].map(q => qToA(q)) : A_ref.map(() => qToA(Q_BASE));
  const rhs_vals = A_ref.map((a, i) => {
    const A_up_old = i === 0 ? qToA(HYDROGRAPH[stepIdx]) : A_ref_prev[i - 1];
    return a / DT_FORCED - (1 - theta) * cBar[i] / DX * (a - A_up_old);
  });

  // Solution: forward sweep
  const A_solution: number[] = new Array(N);
  const A_bc = qToA(HYDROGRAPH[Math.min(stepIdx + 1, HYDROGRAPH.length - 1)]);
  for (let i = 0; i < N; i++) {
    const A_up_new = i === 0 ? A_bc : A_solution[i - 1];
    A_solution[i] = Math.max((rhs_vals[i] + l_vals[i] * A_up_new) / b_vals[i], 0.001);
  }

  // Outlet hydrograph series
  const refOutlet = Q_ref.map(row => row[N - 1]);
  const preissOutlet = Q_preissmann.map(row => row[N - 1]);
  const unstableOutlet = Q_explicit_unstable.map(row => row[N - 1]);

  const hydrographSeries: SeriesConfig[] = [
    { values: unstableOutlet, color: '#94a3b8', dashed: true, label: 'Explicit Δt=500s (unstable)' },
    { values: preissOutlet, color: '#7c3aed', dashed: false, label: `Preissmann θ=${theta.toFixed(2)} Δt=500s` },
    { values: refOutlet, color: '#16a34a', dashed: false, label: 'Explicit Δt=300s (reference)' },
  ];

  return (
    <div className="p-5 space-y-5">
      {/* θ slider */}
      <div className="bg-slate-50 rounded-xl border border-slate-200 p-4">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-semibold text-slate-700">
            Weighting parameter θ = <span className="text-violet-700">{theta.toFixed(2)}</span>
          </span>
          <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${
            theta < 0.1 ? 'bg-red-100 text-red-700' :
            theta < 0.55 ? 'bg-blue-100 text-blue-700' :
            theta < 0.65 ? 'bg-violet-100 text-violet-700' :
            'bg-slate-200 text-slate-700'
          }`}>
            {thetaLabel}
          </span>
        </div>
        <input
          type="range"
          min={0} max={1} step={0.05}
          value={theta}
          onChange={e => setTheta(Number(e.target.value))}
          className="w-full accent-violet-600"
        />
        <div className="flex justify-between text-xs text-slate-400 mt-1">
          <span>0 (explicit)</span>
          <span>0.5 (C-N)</span>
          <span>0.6 (standard)</span>
          <span>1 (implicit)</span>
        </div>
      </div>

      {/* What does "implicit" actually mean? */}
      <div>
        <div className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2">
          What does &ldquo;implicit&rdquo; actually mean?
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          <div className="rounded-xl border border-slate-200 bg-slate-50 p-3">
            <div className="text-xs font-bold text-slate-600 text-center mb-1">Explicit</div>
            <DependencyDiagram implicit={false} />
            <p className="text-xs text-slate-500 text-center mt-1">
              Each future value depends only on <span className="font-semibold">past</span> values —
              already known. Compute left→right, one at a time.
            </p>
          </div>
          <div className="rounded-xl border border-violet-200 bg-violet-50/40 p-3">
            <div className="text-xs font-bold text-violet-700 text-center mb-1">Implicit</div>
            <DependencyDiagram implicit={true} />
            <p className="text-xs text-slate-500 text-center mt-1">
              Each future value <span className="font-semibold text-violet-700">also</span> depends on its
              neighbor&apos;s future value — none are known until ALL are solved together.
            </p>
          </div>
        </div>

        <div className="mt-3 rounded-lg bg-violet-50 border border-violet-200 p-3 text-sm text-violet-900">
          <span className="font-semibold">This is what &ldquo;implicit&rdquo; means:</span> the unknowns
          are defined in terms of <em>each other</em> (implicitly), not directly from already-known data
          (explicitly). The trade-off: extra work per step — solving a coupled system instead of a single
          formula — buys you <span className="font-semibold">unconditional stability</span>. Δt can be
          arbitrarily large without blowing up.
        </div>

        <p className="text-xs text-slate-400 italic mt-2">
          Analogy: explicit is a relay race — each runner starts only once the previous one has arrived.
          Implicit is a small group writing a joint statement — nobody finalizes their sentence until
          everyone&apos;s sentences are mutually consistent.
        </p>

        <p className="text-sm text-slate-600 mt-2">
          <span className="font-semibold text-slate-700">Why coupling prevents blow-up:</span> an explicit
          step <em>extrapolates</em> forward from the past — if Δt is too large, the extrapolation
          overshoots, and that overshoot compounds every subsequent step (the same runaway oscillation you
          saw with the explicit scheme once C&gt;1). An implicit step does not extrapolate; it solves for
          the one set of future values that satisfies every reach&apos;s equation simultaneously. There is
          no overshoot to compound — a larger Δt only costs accuracy (more numerical diffusion), never
          stability.
        </p>
      </div>

      {/* Time step selector */}
      <div className="flex items-center gap-3">
        <span className="text-sm font-medium text-slate-600">Matrix at time step:</span>
        <div className="flex gap-1">
          {HYDROGRAPH.map((_, t) => (
            <button
              key={t}
              onClick={() => { setStepIdx(t); setSweepRow(-1); }}
              className={`w-7 h-7 rounded text-xs font-mono font-semibold transition-colors ${
                t === stepIdx
                  ? 'bg-violet-600 text-white'
                  : 'bg-slate-100 text-slate-500 hover:bg-violet-100 hover:text-violet-700'
              }`}
            >
              {t}
            </button>
          ))}
        </div>
      </div>

      {/* Matrix display */}
      <div>
        <div className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2">
          Implicit system matrix (lower-bidiagonal) — step {stepIdx}
        </div>
        <div className="overflow-x-auto">
          <table className="text-xs font-mono border-collapse w-full">
            <thead>
              <tr className="bg-slate-100">
                <th className="px-1 py-1 text-slate-500 font-medium border border-slate-200 text-center">row</th>
                {Array.from({ length: N }, (_, j) => (
                  <th key={j} className="px-2 py-1 text-slate-500 font-medium border border-slate-200 text-center">
                    A<sub>{j + 1}</sub>
                  </th>
                ))}
                <th className="px-2 py-1 text-slate-500 font-medium border border-slate-200 text-center bg-amber-50">
                  RHS (d)
                </th>
                <th className="px-2 py-1 text-slate-500 font-medium border border-slate-200 text-center bg-teal-50">
                  Aⁿ⁺¹
                </th>
              </tr>
            </thead>
            <tbody>
              {Array.from({ length: N }, (_, i) => {
                const isActiveSweep = i === sweepRow;
                return (
                <tr
                  key={i}
                  className={
                    isActiveSweep
                      ? 'bg-violet-50 border-l-4 border-l-violet-500'
                      : 'hover:bg-slate-50'
                  }
                >
                  <td className={`px-1 py-1 border border-slate-200 text-center ${
                    isActiveSweep ? 'text-violet-700 font-bold' : 'text-slate-500'
                  }`}>
                    {i + 1}
                  </td>
                  {Array.from({ length: N }, (_, j) => {
                    const isDiag = j === i;
                    const isSubDiag = j === i - 1;
                    return (
                      <td
                        key={j}
                        className={`px-2 py-1 border border-slate-200 text-center ${
                          isDiag ? 'bg-teal-100 font-bold text-teal-800' :
                          isSubDiag ? 'bg-blue-100 text-blue-800' :
                          'text-slate-300'
                        }`}
                      >
                        {isDiag
                          ? b_vals[i].toFixed(3)
                          : isSubDiag
                          ? `−${l_vals[i].toFixed(3)}`
                          : '0'}
                      </td>
                    );
                  })}
                  <td className="px-2 py-1 border border-slate-200 text-center bg-amber-50 text-amber-800">
                    {rhs_vals[i].toFixed(3)}
                  </td>
                  <td className="px-2 py-1 border border-slate-200 text-center bg-teal-50 font-bold text-teal-900">
                    {A_solution[i].toFixed(3)}
                  </td>
                </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <div className="mt-2 flex flex-wrap gap-4 text-xs text-slate-500">
          <span>
            <span className="inline-block w-3 h-3 rounded bg-teal-100 mr-1 align-middle" />
            b<sub>i</sub> = 1/Δt + θ·c̄/Δx
          </span>
          <span>
            <span className="inline-block w-3 h-3 rounded bg-blue-100 mr-1 align-middle" />
            −l<sub>i</sub> = −θ·c̄/Δx
          </span>
          <span>
            <span className="inline-block w-3 h-3 rounded bg-amber-50 border border-amber-200 mr-1 align-middle" />
            RHS d<sub>i</sub>
          </span>
          <span>
            <span className="inline-block w-3 h-3 rounded bg-teal-50 border border-teal-200 mr-1 align-middle" />
            solution Aⁿ⁺¹
          </span>
        </div>

        {/* Thomas algorithm animated forward sweep */}
        <div className="mt-3 rounded-xl border border-violet-200 bg-white p-3">
          <div className="flex items-center justify-between flex-wrap gap-2 mb-2">
            <span className="text-xs font-semibold text-slate-500 uppercase tracking-wide">
              Thomas algorithm: forward sweep ↓ then back-substitute ↑
            </span>
            <div className="flex gap-2">
              <button
                onClick={() => setSweepRow(-1)}
                className="text-xs font-semibold px-3 py-1.5 rounded-full bg-slate-100 text-slate-500 hover:bg-slate-200 transition"
              >
                ⏮ Reset
              </button>
              <button
                onClick={() => setSweepRow(s => Math.min(s + 1, N - 1))}
                disabled={sweepRow >= N - 1}
                className="text-xs font-semibold px-3 py-1.5 rounded-full bg-violet-600 text-white hover:bg-violet-700 disabled:opacity-40 disabled:cursor-not-allowed transition"
              >
                ▶ Next row
              </button>
              <button
                onClick={() => setSweepRow(N - 1)}
                className="text-xs font-semibold px-3 py-1.5 rounded-full bg-slate-200 text-slate-700 hover:bg-slate-300 transition"
              >
                Show all
              </button>
            </div>
          </div>

          {sweepRow === -1 ? (
            <p className="text-xs text-slate-400 italic">
              Nothing solved yet — row 1 only needs the boundary value (already known), so it can go
              first. Press &ldquo;▶ Next row&rdquo; to reveal each step of the sweep.
            </p>
          ) : (
            <div className="space-y-2">
              {Array.from({ length: sweepRow + 1 }, (_, i) => {
                const A_up = i === 0 ? A_bc : A_solution[i - 1];
                return (
                  <pre
                    key={i}
                    className="bg-slate-50 font-mono text-xs border border-slate-200 rounded-lg p-2 whitespace-pre-wrap overflow-x-auto"
                  >
{`Row ${i + 1}: solve for A${i + 1}ⁿ⁺¹

  ${b_vals[i].toFixed(3)}·A${i + 1}ⁿ⁺¹ − ${l_vals[i].toFixed(3)}·A${i}ⁿ⁺¹ = ${rhs_vals[i].toFixed(3)}

  A${i + 1}ⁿ⁺¹ = (d + l·A_up) / b
            = (${rhs_vals[i].toFixed(3)} + ${l_vals[i].toFixed(3)}×${A_up.toFixed(3)}) / ${b_vals[i].toFixed(3)}
            = ${A_solution[i].toFixed(3)} m²`}
                  </pre>
                );
              })}
            </div>
          )}
        </div>
      </div>

      {/* Outlet hydrograph chart */}
      <div>
        <div className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2">
          Outlet hydrograph (reach 6) — Δt=500 s, C≈1.16
        </div>
        <div className="rounded-xl border border-slate-200 bg-slate-50 p-2">
          <OutletHydrograph series={hydrographSeries} />
        </div>
      </div>

      {/* Info callout */}
      <div className="rounded-lg bg-violet-50 border border-violet-200 p-3 text-sm text-violet-900">
        <span className="font-semibold">Seeing it on this hydrograph:</span> at Δt=500 s the explicit line
        (gray) blows up — C≈1.16, the overshoot from the previous step compounding into the next. The
        Preissmann line (violet) tracks the Δt=300 s reference closely at the same Δt=500 s, because the
        coupled solve above never extrapolates — it just costs a touch of artificial diffusion, tunable via θ.
      </div>

      {/* Courant number info */}
      <div className="flex gap-3 flex-wrap">
        <div className="bg-red-50 border border-red-200 rounded-lg p-3 flex-1 min-w-[140px]">
          <div className="text-xs text-red-500 font-semibold uppercase tracking-wide">Explicit</div>
          <div className="text-lg font-bold text-red-700 mt-0.5">C ≈ 1.16</div>
          <div className="text-xs text-red-600">Δt=500s, unstable</div>
        </div>
        <div className="bg-green-50 border border-green-200 rounded-lg p-3 flex-1 min-w-[140px]">
          <div className="text-xs text-green-500 font-semibold uppercase tracking-wide">Preissmann</div>
          <div className="text-lg font-bold text-green-700 mt-0.5">C ≈ 1.16 ✓</div>
          <div className="text-xs text-green-600">Δt=500s, stable (θ≥0.5)</div>
        </div>
        <div className="bg-slate-50 border border-slate-200 rounded-lg p-3 flex-1 min-w-[140px]">
          <div className="text-xs text-slate-500 font-semibold uppercase tracking-wide">Reference</div>
          <div className="text-lg font-bold text-slate-700 mt-0.5">C ≈ 0.70</div>
          <div className="text-xs text-slate-600">Δt=300s, safe</div>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Tab 2: Flux Limiter (OPM)
// ---------------------------------------------------------------------------

interface FluxLimiterTabProps {
  limiterResult: FluxLimitedResult;
  Q_ref: number[][];
  Q_explicit_unstable: number[][];
}

function FluxLimiterTab({ limiterResult, Q_ref, Q_explicit_unstable }: FluxLimiterTabProps) {
  const displaySteps = [1, 2, 3, 4];

  const refOutlet = Q_ref.map(row => row[N - 1]);
  const limOutlet = limiterResult.Q.map(row => row[N - 1]);
  const unstableOutlet = Q_explicit_unstable.map(row => row[N - 1]);

  const peakRef = Math.max(...refOutlet).toFixed(1);
  const peakLim = Math.max(...limOutlet).toFixed(1);
  const peakUnstable = unstableOutlet.filter(v => isFinite(v) && v > -900);
  const peakUnstableStr = peakUnstable.length > 0 ? Math.max(...peakUnstable).toFixed(1) : 'NaN';

  const hydrographSeries: SeriesConfig[] = [
    { values: unstableOutlet, color: '#94a3b8', dashed: true, label: `Explicit Δt=500s (peak: ${peakUnstableStr})` },
    { values: limOutlet, color: '#16a34a', dashed: false, label: `Flux limited Δt=500s (peak: ${peakLim})` },
    { values: refOutlet, color: '#2563eb', dashed: false, label: `Reference Δt=300s (peak: ${peakRef})` },
  ];

  return (
    <div className="p-5 space-y-5">
      {/* Equation box */}
      <div className="bg-orange-50 border border-orange-200 rounded-lg p-3 font-mono text-sm my-3">
        Q_out = min(Q_Manning, V / Δt)<br />
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= min(α·A^β, &nbsp;&nbsp;A·Δx / Δt)<br />
        <span className="text-orange-700 text-xs">
          ↳ Never drain more volume than stored in one step
        </span>
      </div>

      {/* Table: 4 time steps × 6 reaches */}
      <div>
        <div className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2">
          Flux limiter activity — time steps 1–4, all reaches
        </div>
        <div className="overflow-x-auto">
          <table className="text-xs border-collapse w-full">
            <thead>
              <tr className="bg-slate-100">
                <th className="px-2 py-1.5 border border-slate-200 text-center text-slate-600 font-semibold">t</th>
                <th className="px-2 py-1.5 border border-slate-200 text-center text-slate-600 font-semibold">i</th>
                <th className="px-2 py-1.5 border border-slate-200 text-center text-slate-600 font-semibold">
                  V = A·Δx (m³)
                </th>
                <th className="px-2 py-1.5 border border-slate-200 text-center text-slate-600 font-semibold">
                  Q_Manning
                </th>
                <th className="px-2 py-1.5 border border-slate-200 text-center text-slate-600 font-semibold">
                  V/Δt
                </th>
                <th className="px-2 py-1.5 border border-slate-200 text-center text-slate-600 font-semibold">
                  Q_out
                </th>
                <th className="px-2 py-1.5 border border-slate-200 text-center text-slate-600 font-semibold">
                  Active?
                </th>
              </tr>
            </thead>
            <tbody>
              {displaySteps.flatMap(t =>
                Array.from({ length: N }, (_, i) => {
                  const active = limiterResult.limActive[t][i];
                  return (
                    <tr key={`${t}-${i}`} className={active ? 'bg-orange-100' : 'hover:bg-slate-50'}>
                      {i === 0 ? (
                        <td
                          className="px-2 py-1 border border-slate-200 text-center font-semibold text-slate-700"
                          rowSpan={N}
                        >
                          {t}
                        </td>
                      ) : null}
                      <td className="px-2 py-1 border border-slate-200 text-center text-slate-600">
                        {i + 1}
                      </td>
                      <td className="px-2 py-1 border border-slate-200 text-center font-mono">
                        {limiterResult.V[t][i].toFixed(1)}
                      </td>
                      <td className={`px-2 py-1 border border-slate-200 text-center font-mono ${
                        active ? 'text-orange-800 font-semibold' : 'text-slate-700'
                      }`}>
                        {limiterResult.Q_man[t][i].toFixed(1)}
                      </td>
                      <td className="px-2 py-1 border border-slate-200 text-center font-mono text-slate-700">
                        {(limiterResult.V[t][i] / DT_FORCED).toFixed(1)}
                      </td>
                      <td className={`px-2 py-1 border border-slate-200 text-center font-mono font-bold ${
                        active ? 'text-orange-900' : 'text-slate-800'
                      }`}>
                        {limiterResult.Q_lim_arr[t][i].toFixed(1)}
                      </td>
                      <td className="px-2 py-1 border border-slate-200 text-center">
                        {active ? (
                          <span className="text-orange-700 font-bold text-xs">YES</span>
                        ) : (
                          <span className="text-slate-400 text-xs">—</span>
                        )}
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
        <p className="text-xs text-slate-400 mt-1 italic">
          Q_out = min(Q_Manning, V/Δt) — highlighted rows = limiter active
        </p>
      </div>

      {/* Outlet hydrograph */}
      <div>
        <div className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2">
          Outlet hydrograph (reach 6)
        </div>
        <div className="rounded-xl border border-slate-200 bg-slate-50 p-2">
          <OutletHydrograph series={hydrographSeries} />
        </div>
      </div>

      {/* Info callout */}
      <div className="rounded-lg bg-orange-50 border border-orange-200 p-3 text-sm text-orange-900">
        <span className="font-semibold">Key insight:</span> When Q_Manning would drain more volume than
        is stored in a reach within one time step, the flux limiter caps it at V/Δt. This is a local,
        explicit constraint — no matrix solve needed — and it guarantees cells never go negative.
        The trade-off is additional numerical diffusion (the peak is slightly attenuated vs. reference).
      </div>

      {/* Comparison table */}
      <div>
        <div className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2">
          Side-by-side comparison
        </div>
        <div className="overflow-x-auto">
          <table className="text-sm border-collapse w-full">
            <thead>
              <tr className="bg-slate-100">
                <th className="px-3 py-2 border border-slate-200 text-left text-slate-600 font-semibold">
                  Aspect
                </th>
                <th className="px-3 py-2 border border-slate-200 text-left text-violet-700 font-semibold">
                  Preissmann
                </th>
                <th className="px-3 py-2 border border-slate-200 text-left text-orange-700 font-semibold">
                  Flux Limiter
                </th>
              </tr>
            </thead>
            <tbody>
              {[
                ['Stability mechanism', 'θ-weight at t+1', 'V/Δt cap per cell'],
                ['Matrix solve', 'Yes (N×N banded)', 'No'],
                ['Mass conservation', 'Yes', 'Yes'],
                ['Numerical diffusion', 'Artificial (θ>0.5)', 'Numerical (cap)'],
                ['Used by', 'HEC-RAS, ISIS, SWMM', 'OPM'],
                ['Cost per step', 'O(N) banded', 'O(N) scalar'],
              ].map(([aspect, preiss, flux], idx) => (
                <tr key={idx} className={idx % 2 === 0 ? 'bg-white' : 'bg-slate-50'}>
                  <td className="px-3 py-2 border border-slate-200 font-medium text-slate-700">
                    {aspect}
                  </td>
                  <td className="px-3 py-2 border border-slate-200 text-violet-800">
                    {preiss}
                  </td>
                  <td className="px-3 py-2 border border-slate-200 text-orange-800">
                    {flux}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main widget
// ---------------------------------------------------------------------------

export default function SchemeCompareWidget() {
  const [tab, setTab] = useState<'preissmann' | 'limiter'>('preissmann');
  const [theta, setTheta] = useState(0.6);
  const [stepIdx, setStepIdx] = useState(2);

  const Q_ref = useMemo(() => runExplicit(DT_STABLE), []);
  const Q_explicit_unstable = useMemo(() => runExplicit(DT_FORCED), []);
  const Q_preissmann = useMemo(() => runPreissmann(DT_FORCED, theta), [theta]);
  const limiterResult = useMemo(() => runFluxLimited(DT_FORCED), []);

  return (
    <div className="not-prose my-8 rounded-2xl border border-slate-200 bg-white shadow-xl overflow-hidden font-sans">
      {/* Header */}
      <div className="bg-gradient-to-r from-sky-700 to-blue-900 px-6 py-4">
        <h3 className="text-white font-bold text-lg tracking-tight">Beating the CFL Limit</h3>
        <p className="text-sky-200 text-sm mt-0.5">
          Preissmann implicit scheme vs. OPM flux limiter — same C&gt;1 problem, different solutions
        </p>
      </div>

      {/* Sub-tabs */}
      <div className="flex border-b border-slate-200 bg-slate-50">
        <button
          onClick={() => setTab('preissmann')}
          className={`flex items-center gap-1.5 px-5 py-3 text-sm font-medium transition-colors border-b-2 ${
            tab === 'preissmann'
              ? 'border-violet-600 text-violet-700 bg-white'
              : 'border-transparent text-slate-500 hover:text-slate-700 hover:bg-slate-100'
          }`}
        >
          <span>📊</span> Preissmann Implicit
        </button>
        <button
          onClick={() => setTab('limiter')}
          className={`flex items-center gap-1.5 px-5 py-3 text-sm font-medium transition-colors border-b-2 ${
            tab === 'limiter'
              ? 'border-orange-500 text-orange-700 bg-white'
              : 'border-transparent text-slate-500 hover:text-slate-700 hover:bg-slate-100'
          }`}
        >
          <span>🔒</span> Flux Limiter (OPM)
        </button>
      </div>

      {/* Tab content */}
      {tab === 'preissmann' ? (
        <PreissmannTab
          theta={theta}
          setTheta={setTheta}
          stepIdx={stepIdx}
          setStepIdx={setStepIdx}
          Q_ref={Q_ref}
          Q_preissmann={Q_preissmann}
          Q_explicit_unstable={Q_explicit_unstable}
        />
      ) : (
        <FluxLimiterTab
          limiterResult={limiterResult}
          Q_ref={Q_ref}
          Q_explicit_unstable={Q_explicit_unstable}
        />
      )}
    </div>
  );
}
