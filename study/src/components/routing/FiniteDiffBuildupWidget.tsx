'use client';

import React, { useState, useMemo, useEffect } from 'react';

// ─────────────────────────────────────────────────────────────────────────
// Shared constants (verified twice: log/exp method + cube-root/5th-root method)
// ─────────────────────────────────────────────────────────────────────────
const ALPHA = 0.1703;
const BETA = 1.6667;
const DX = 500; // m
const DT = 300; // s
const R = DT / DX; // 0.60
const A0 = 4.38; // m^2, baseflow area at Q0 = 2 m^3/s

const Q_IN = [2, 10, 12, 8, 2]; // upstream hydrograph, t=0..4

// Ground-truth tables, hard-coded and cross-checked (do not recompute on the fly)
const A_TABLE: number[][] = [
  // t=0..4, i=0..2
  [4.38, 4.38, 4.38],
  [11.52, 4.38, 4.38],
  [12.85, 9.18, 4.38],
  [10.07, 12.27, 7.3],
  [4.38, 10.4, 11.16],
];
const Q_TABLE: number[][] = [
  [2.0, 2.0, 2.0],
  [10.0, 2.0, 2.0],
  [12.0, 6.86, 2.0],
  [8.0, 11.11, 4.67],
  [2.0, 8.44, 9.49],
];

function fmt(n: number, d = 2): string {
  return n.toFixed(d);
}

// ─────────────────────────────────────────────────────────────────────────
// Numbered section badge
// ─────────────────────────────────────────────────────────────────────────
function Badge({ n }: { n: number }) {
  const circled = ['①', '②', '③', '④', '⑤', '⑥'][n - 1];
  return (
    <span className="inline-flex items-center justify-center w-8 h-8 rounded-full bg-indigo-600 text-white font-bold text-base mr-2 shrink-0">
      {circled}
    </span>
  );
}

function SectionHeader({ n, title }: { n: number; title: string }) {
  return (
    <div className="flex items-center mb-3">
      <Badge n={n} />
      <h4 className="text-lg font-bold text-slate-800">{title}</h4>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────
// Section 1 — Why Numerical Methods?
// ─────────────────────────────────────────────────────────────────────────
function SectionProblem() {
  return (
    <div>
      <SectionHeader n={1} title="Why Can't We Solve This Analytically?" />
      <div className="flex flex-col sm:flex-row gap-4">
        <div className="flex-1 rounded-xl border border-blue-200 bg-blue-50 p-4">
          <div className="text-xs font-bold text-blue-700 uppercase mb-2">
            The simple case (solvable)
          </div>
          <pre className="text-xs font-mono text-slate-700 whitespace-pre-wrap leading-relaxed">
{`If wave speed c is CONSTANT and q_lat = 0:

  ∂A/∂t + c·∂A/∂x = 0

Exact solution:
  A(x, t) = A₀(x − c·t)

The initial wave shape just SLIDES
downstream at speed c. No numerical
methods needed.`}
          </pre>
        </div>
        <div className="flex-1 rounded-xl border border-amber-200 bg-amber-50 p-4">
          <div className="text-xs font-bold text-amber-700 uppercase mb-2">
            The real river (not solvable)
          </div>
          <pre className="text-xs font-mono text-slate-700 whitespace-pre-wrap leading-relaxed">
{`For real rivers:
  c = β·α·A^(β−1)  — it DEPENDS ON A.

As the flood rises, A grows → c grows
→ wave accelerates. Different parts
of the wave travel at different speeds.

No closed-form solution exists for
this nonlinear PDE. We must compute
it on a discrete grid.`}
          </pre>
        </div>
      </div>
      <p className="text-sm text-slate-600 mt-3">
        The strategy: divide the river into short cells (Δx = 500 m) and time into short steps
        (Δt = 300 s), then march forward one step at a time using a simple algebraic update
        formula.
      </p>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────
// Section 2 — Discretising Space & Time (interactive grid)
// ─────────────────────────────────────────────────────────────────────────
function SectionGrid() {
  const [showOrder, setShowOrder] = useState(false);

  const cols = 6; // i = 0..5
  const rows = 5; // t = 0..4
  const cw = 50;
  const ch = 36;
  const ox = 36;
  const oy = 12;
  const qInRow = [10, 12, 8, 2]; // t=1..4 boundary values for display

  // Marching order across the 6x5 grid (i=1..5 for each t=1..4): 20 cells total
  const order: { i: number; t: number; n: number }[] = [];
  let counter = 1;
  for (let t = 1; t < rows; t++) {
    for (let i = 1; i < cols; i++) {
      order.push({ i, t, n: counter++ });
    }
  }

  return (
    <div>
      <SectionHeader n={2} title="Replace the River with a Grid" />
      <div className="flex flex-col lg:flex-row gap-6">
        <div className="lg:w-[55%] text-sm text-slate-700 space-y-3">
          <p>
            Chop the channel into N = 8 cells of length Δx = 500 m each. Label them
            i = 0, 1, 2, …, 7. Then divide time into steps of Δt = 300 s. At every grid point
            (i, t) we store one number: <span className="font-mono font-semibold">A[i, t]</span>,
            the cross-sectional area at that location and instant.
          </p>
          <div className="space-y-2">
            <div className="flex items-start gap-2 rounded-lg bg-green-50 border border-green-200 p-2">
              <span className="text-lg leading-none">🟩</span>
              <span>
                <strong>Bottom row (t = 0):</strong> initial condition. Every reach starts at
                baseflow A₀ = 4.38 m² (Q₀ = 2 m³/s).
              </span>
            </div>
            <div className="flex items-start gap-2 rounded-lg bg-amber-50 border border-amber-200 p-2">
              <span className="text-lg leading-none">🟨</span>
              <span>
                <strong>Left column (i = 0):</strong> boundary condition. The upstream
                hydrograph drives the simulation — we prescribe Q_in(t) here.
              </span>
            </div>
            <div className="flex items-start gap-2 rounded-lg bg-slate-50 border border-slate-200 p-2">
              <span className="text-lg leading-none">⬜</span>
              <span>
                <strong>Everything else:</strong> UNKNOWN — computed left-to-right,
                bottom-to-top (forward in time).
              </span>
            </div>
          </div>
          <p>
            Goal: fill in the entire table. Given what we know in the green and amber cells,
            can we systematically compute every white cell? Yes — that&apos;s what the explicit
            scheme does.
          </p>
        </div>

        <div className="lg:w-[45%] flex flex-col items-center">
          <svg width={ox + cols * cw + 10} height={oy + rows * ch + 30} className="select-none">
            {/* cells */}
            {Array.from({ length: rows }).map((_, rIdx) => {
              const t = rows - 1 - rIdx; // draw bottom-up: row 0 at bottom is t=0
              return Array.from({ length: cols }).map((__, i) => {
                const x = ox + i * cw;
                const y = oy + rIdx * ch;
                let fill = 'white';
                let stroke = '#e2e8f0';
                let label = '?';
                let labelColor = '#94a3b8';
                if (t === 0) {
                  fill = '#dcfce7';
                  label = '4.38';
                  labelColor = '#166534';
                } else if (i === 0) {
                  fill = '#fef3c7';
                  label = String(qInRow[t - 1]);
                  labelColor = '#92400e';
                }
                const marchEntry = order.find((o) => o.i === i && o.t === t);
                return (
                  <g key={`${i}-${t}`}>
                    <rect x={x} y={y} width={cw - 2} height={ch - 2} fill={fill} stroke={stroke} />
                    <text
                      x={x + (cw - 2) / 2}
                      y={y + (ch - 2) / 2 + 4}
                      textAnchor="middle"
                      fontSize="10"
                      fontFamily="monospace"
                      fill={labelColor}
                    >
                      {label}
                    </text>
                    {showOrder && marchEntry && (
                      <>
                        <circle
                          cx={x + (cw - 2) / 2}
                          cy={y + (ch - 2) / 2}
                          r={9}
                          fill="#4338ca"
                          opacity={0.85}
                        />
                        <text
                          x={x + (cw - 2) / 2}
                          y={y + (ch - 2) / 2 + 3}
                          textAnchor="middle"
                          fontSize="9"
                          fontWeight="bold"
                          fill="white"
                        >
                          {marchEntry.n}
                        </text>
                      </>
                    )}
                  </g>
                );
              });
            })}
            {/* x labels */}
            {Array.from({ length: cols }).map((_, i) => (
              <text
                key={`xl-${i}`}
                x={ox + i * cw + (cw - 2) / 2}
                y={oy + rows * ch + 14}
                textAnchor="middle"
                fontSize="10"
                fill="#475569"
              >
                i={i}
              </text>
            ))}
            <text
              x={ox + (cols * cw) / 2}
              y={oy + rows * ch + 28}
              textAnchor="middle"
              fontSize="10"
              fill="#64748b"
            >
              → distance downstream (x)
            </text>
            {/* y labels */}
            {Array.from({ length: rows }).map((_, rIdx) => {
              const t = rows - 1 - rIdx;
              return (
                <text
                  key={`yl-${t}`}
                  x={ox - 6}
                  y={oy + rIdx * ch + (ch - 2) / 2 + 4}
                  textAnchor="end"
                  fontSize="10"
                  fill="#475569"
                >
                  t={t}
                </text>
              );
            })}
            <text
              x={10}
              y={oy + (rows * ch) / 2}
              textAnchor="middle"
              fontSize="10"
              fill="#64748b"
              transform={`rotate(-90, 10, ${oy + (rows * ch) / 2})`}
            >
              ↑ time
            </text>
          </svg>
          <button
            onClick={() => setShowOrder((s) => !s)}
            className="mt-2 text-xs font-semibold px-3 py-1.5 rounded-full bg-indigo-600 text-white hover:bg-indigo-700 transition"
          >
            {showOrder ? 'Hide computation order' : 'Show computation order →'}
          </button>
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────
// Section 3 — Derivatives as Differences
// ─────────────────────────────────────────────────────────────────────────
function SectionDerivatives() {
  return (
    <div>
      <SectionHeader n={3} title="What Is a Derivative on a Grid?" />
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
        <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
          <div className="font-semibold text-sm text-slate-800 mb-2">
            Time derivative: ∂A/∂t
          </div>
          <div className="bg-white rounded-lg border border-slate-200 p-2 font-mono text-xs text-center mb-3">
            ∂A/∂t&nbsp;&nbsp;≈&nbsp;&nbsp;(A[i, t+1] − A[i, t]) / Δt
          </div>
          <div className="flex justify-center mb-3">
            <svg width={140} height={120}>
              <rect x={25} y={68} width={90} height={36} fill="#dbeafe" stroke="#3b82f6" />
              <text x={70} y={88} textAnchor="middle" fontSize="11" fontFamily="monospace">
                A[i, t]
              </text>
              <text x={70} y={100} textAnchor="middle" fontSize="8" fill="#1e40af">
                known (now)
              </text>
              <rect
                x={25}
                y={16}
                width={90}
                height={36}
                fill="#f0fdf4"
                stroke="#22c55e"
                strokeDasharray="4 3"
              />
              <text x={70} y={36} textAnchor="middle" fontSize="11" fontFamily="monospace">
                A[i, t+1]
              </text>
              <text x={70} y={48} textAnchor="middle" fontSize="8" fill="#15803d">
                UNKNOWN (next)
              </text>
              <line x1={10} y1={64} x2={10} y2={20} stroke="#64748b" markerEnd="url(#arrUp)" />
              <line x1={10} y1={68} x2={10} y2={104} stroke="#64748b" markerEnd="url(#arrDown)" />
              <defs>
                <marker id="arrUp" markerWidth="6" markerHeight="6" refX="3" refY="5" orient="auto">
                  <path d="M0,5 L3,0 L6,5 Z" fill="#64748b" />
                </marker>
                <marker id="arrDown" markerWidth="6" markerHeight="6" refX="3" refY="0" orient="auto">
                  <path d="M0,0 L3,5 L6,0 Z" fill="#64748b" />
                </marker>
              </defs>
              <text x={10} y={66} textAnchor="start" fontSize="8" fill="#475569" dx={4}>
                Δt
              </text>
            </svg>
          </div>
          <p className="text-xs text-slate-600 italic">
            Temperature at 8am: 15°C. At 9am: 18°C. Rate of change = (18−15)/1 hr = 3°C/hr.
            Same idea.
          </p>
        </div>

        <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
          <div className="font-semibold text-sm text-slate-800 mb-2">
            Space derivative: ∂Q/∂x
          </div>
          <div className="bg-white rounded-lg border border-slate-200 p-2 font-mono text-xs text-center mb-3">
            ∂Q/∂x&nbsp;&nbsp;≈&nbsp;&nbsp;(Q[i, t] − Q[i−1, t]) / Δx
          </div>
          <div className="flex justify-center mb-3">
            <svg width={220} height={80}>
              <rect x={10} y={20} width={85} height={36} fill="#dbeafe" stroke="#3b82f6" />
              <text x={52} y={40} textAnchor="middle" fontSize="10" fontFamily="monospace">
                Q[i−1, t]
              </text>
              <text x={52} y={52} textAnchor="middle" fontSize="8" fill="#1e40af">
                upstream
              </text>
              <rect x={125} y={20} width={85} height={36} fill="#fff7ed" stroke="#f97316" />
              <text x={167} y={40} textAnchor="middle" fontSize="10" fontFamily="monospace">
                Q[i, t]
              </text>
              <text x={167} y={52} textAnchor="middle" fontSize="8" fill="#c2410c">
                current cell
              </text>
              <line
                x1={97}
                y1={38}
                x2={123}
                y2={38}
                stroke="#64748b"
                markerEnd="url(#arrRight)"
              />
              <defs>
                <marker id="arrRight" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
                  <path d="M0,0 L6,3 L0,6 Z" fill="#64748b" />
                </marker>
              </defs>
              <text x={110} y={30} textAnchor="middle" fontSize="8" fill="#475569">
                Δx
              </text>
            </svg>
          </div>
          <p className="text-xs text-slate-600">
            We look <strong>UPSTREAM</strong> (i−1), not downstream (i+1).
          </p>
        </div>
      </div>

      <div className="mt-4 rounded-xl border border-amber-300 bg-amber-50 p-4">
        <div className="font-bold text-sm text-amber-800 mb-1">
          Why the upstream cell? (The upwind scheme)
        </div>
        <p className="text-sm text-slate-700">
          Water flows from i−1 → i → i+1. Information travels WITH the flow — downstream. If
          we looked at Q[i+1,t], we&apos;d be reading data from a cell the flood hasn&apos;t
          reached yet. Using Q[i−1,t] means we look &ldquo;into the wind&rdquo; — toward where
          the flow is coming from. This is the <strong>upwind</strong> approximation, and it&apos;s
          both physically correct and numerically stable.
        </p>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────
// Section 4 — Deriving the Update Formula
// ─────────────────────────────────────────────────────────────────────────
function SectionFormula() {
  const [visibleSteps, setVisibleSteps] = useState(1);

  return (
    <div>
      <SectionHeader n={4} title="Derive the Update Formula Step by Step" />

      <div className="bg-slate-50 rounded-lg p-3 font-mono text-sm text-center mb-3">
        Start: the PDE with continuity and Manning power law:
        <div className="mt-2">∂A/∂t&nbsp;&nbsp;+&nbsp;&nbsp;∂Q/∂x&nbsp;&nbsp;=&nbsp;&nbsp;q_lat&nbsp;&nbsp;&nbsp;&nbsp;where Q = α·A^β</div>
      </div>

      <div className="space-y-3">
        {visibleSteps >= 1 && (
          <div>
            <div className="text-xs font-bold text-slate-500 uppercase mb-1">
              Step 1 — Substitute finite-difference approximations:
            </div>
            <div className="bg-white border border-slate-200 rounded-lg p-3 font-mono text-xs sm:text-sm text-center overflow-x-auto">
              (A[i,t+1] − A[i,t]) / Δt&nbsp;&nbsp;+&nbsp;&nbsp;(Q[i,t] − Q[i−1,t]) / Δx&nbsp;&nbsp;=&nbsp;&nbsp;q_lat
            </div>
            <p className="text-xs text-slate-600 mt-1">
              Every ∂ (calculus) is replaced by Δ (arithmetic). No limits, no calculus — just
              subtraction.
            </p>
          </div>
        )}

        {visibleSteps >= 2 && (
          <div>
            <div className="text-xs font-bold text-slate-500 uppercase mb-1">
              Step 2 — Multiply both sides by Δt:
            </div>
            <div className="bg-white border border-slate-200 rounded-lg p-3 font-mono text-xs sm:text-sm text-center overflow-x-auto">
              (A[i,t+1] − A[i,t])&nbsp;&nbsp;+&nbsp;&nbsp;(Δt/Δx)·(Q[i,t] − Q[i−1,t])&nbsp;&nbsp;=&nbsp;&nbsp;q_lat·Δt
            </div>
          </div>
        )}

        {visibleSteps >= 3 && (
          <div className="bg-green-50 border border-green-300 rounded-lg p-3">
            <div className="text-xs font-bold text-green-700 uppercase mb-1">
              Step 3 — Isolate A[i,t+1] (the ONE unknown):
            </div>
            <div className="font-mono text-xs sm:text-sm text-center overflow-x-auto">
              A[i,t+1]&nbsp;&nbsp;=&nbsp;&nbsp;A[i,t]&nbsp;&nbsp;−&nbsp;&nbsp;(Δt/Δx)·(Q[i,t] − Q[i−1,t])&nbsp;&nbsp;+&nbsp;&nbsp;q_lat·Δt
            </div>
            <p className="text-xs text-green-700 mt-1">
              Everything on the right is KNOWN from time step t. This is the explicit update.
            </p>
          </div>
        )}

        {visibleSteps >= 4 && (
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
            <div className="text-xs font-bold text-blue-700 uppercase mb-1">
              Step 4 — Interpret as a water balance:
            </div>
            <pre className="font-mono text-xs sm:text-sm whitespace-pre-wrap leading-relaxed">
{`A[i,t+1]  =  A[i,t]              ← storage carried forward
          −  (Δt/Δx)·Q[i,t]     ← volume that LEFT downstream
          +  (Δt/Δx)·Q[i−1,t]   ← volume that ARRIVED from upstream
          +  q_lat·Δt            ← lateral inflow (rain, runoff)

New volume = Old volume − Outflow + Inflow + Lateral`}
            </pre>
            <p className="text-xs text-blue-700 mt-1">
              Then: Q[i,t+1] = α·A[i,t+1]^β from Manning. Two lines of arithmetic per cell per
              step.
            </p>
          </div>
        )}
      </div>

      <div className="flex gap-2 mt-3">
        <button
          onClick={() => setVisibleSteps((s) => Math.min(s + 1, 4))}
          disabled={visibleSteps >= 4}
          className="text-xs font-semibold px-3 py-1.5 rounded-full bg-indigo-600 text-white hover:bg-indigo-700 disabled:opacity-40 disabled:cursor-not-allowed transition"
        >
          Show next step ▶
        </button>
        <button
          onClick={() => setVisibleSteps(4)}
          className="text-xs font-semibold px-3 py-1.5 rounded-full bg-slate-200 text-slate-700 hover:bg-slate-300 transition"
        >
          Show all
        </button>
        <button
          onClick={() => setVisibleSteps(1)}
          className="text-xs font-semibold px-3 py-1.5 rounded-full bg-slate-100 text-slate-500 hover:bg-slate-200 transition"
        >
          Reset
        </button>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────
// Section 5 — Hand calculation walkthrough
// ─────────────────────────────────────────────────────────────────────────
interface Step {
  t: number;
  i: number;
}
const STEPS: Step[] = [
  { t: 1, i: 1 },
  { t: 1, i: 2 },
  { t: 2, i: 1 },
  { t: 2, i: 2 },
  { t: 3, i: 1 },
  { t: 3, i: 2 },
  { t: 4, i: 1 },
  { t: 4, i: 2 },
];

function cellState(
  t: number,
  i: number,
  stepIdx: number
): 'boundary' | 'unrevealed' | 'current' | 'source' | 'revealed' {
  if (t === 0 || i === 0) return 'boundary';
  const idxOfThis = STEPS.findIndex((s) => s.t === t && s.i === i);
  if (idxOfThis === -1) return 'unrevealed';
  if (idxOfThis > stepIdx) return 'unrevealed';
  if (idxOfThis === stepIdx) return 'current';
  // revealed already; check if it's a source for the CURRENT step
  if (stepIdx >= 0 && stepIdx < STEPS.length) {
    const cur = STEPS[stepIdx];
    const isSource =
      (t === cur.t - 1 && i === cur.i) || // A[i, t-1]
      (t === cur.t && i === cur.i - 1) || // Q[i-1, t]
      (t === cur.t - 1 && i === cur.i - 1 && false); // not used, kept for clarity
    if (isSource) return 'source';
  }
  return 'revealed';
}

function cellClasses(state: string): string {
  switch (state) {
    case 'boundary':
      return 'bg-amber-50 text-amber-800 font-semibold';
    case 'unrevealed':
      return 'bg-slate-100 text-slate-400';
    case 'current':
      return 'bg-green-100 border-2 border-green-500 font-bold text-green-800';
    case 'source':
      return 'bg-blue-50 border border-blue-300 text-blue-800';
    case 'revealed':
      return 'bg-sky-50 text-slate-700';
    default:
      return '';
  }
}

function SectionHandCalc() {
  const [stepIdx, setStepIdx] = useState(-1);

  const formulaText = useMemo(() => {
    if (stepIdx === -1) {
      return 'Rows t=0 and column i=0 are already known.\nPress "▶ Compute next cell" to start.';
    }
    const { t, i } = STEPS[stepIdx];
    const Aprev = A_TABLE[t - 1][i];
    const Qself = Q_TABLE[t - 1][i];
    const Qup = Q_TABLE[t - 1][i - 1];
    const Anew = A_TABLE[t][i];
    const Qnew = Q_TABLE[t][i];
    const diff = Qself - Qup;
    const term = R * diff;
    return `Computing A[${i}, ${t}]:
  A[${i},${t}] = A[${i},${t - 1}] − (Δt/Δx)·(Q[${i},${t - 1}] − Q[${i - 1},${t - 1}])
          = ${fmt(Aprev)} − ${fmt(R)}·(${fmt(Qself)} − ${fmt(Qup)})
          = ${fmt(Aprev)} − ${fmt(R)}·(${diff >= 0 ? '' : ''}${fmt(diff)})
          = ${fmt(Aprev)} ${term >= 0 ? '+' : '−'} ${fmt(Math.abs(term))}
          = ${fmt(Anew)} m²

Then: Q[${i},${t}] = α·A[${i},${t}]^β
            = ${ALPHA} × ${fmt(Anew)}^${fmt(BETA, 3)}
            = ${fmt(Qnew)} m³/s`;
  }, [stepIdx]);

  const showObservation = stepIdx >= 4;

  return (
    <div>
      <SectionHeader n={5} title="Compute It Yourself — 3 Reaches, 5 Time Steps" />

      <div className="flex flex-wrap gap-2 mb-3">
        {[
          ['Δx', '500 m'],
          ['Δt', '300 s'],
          ['Δt/Δx', '0.60'],
          ['α', '0.1703'],
          ['β', '1.667'],
          ['A₀', '4.38 m²'],
          ['Q₀', '2.00 m³/s'],
        ].map(([k, v]) => (
          <span
            key={k}
            className="text-xs font-mono bg-slate-100 border border-slate-200 rounded-full px-3 py-1"
          >
            <strong>{k}</strong> = {v}
          </span>
        ))}
      </div>

      <table className="text-xs font-mono mb-4 border border-slate-200">
        <thead>
          <tr className="bg-slate-50">
            <th className="px-2 py-1 border border-slate-200">t</th>
            {[0, 1, 2, 3, 4].map((t) => (
              <th key={t} className="px-2 py-1 border border-slate-200">
                {t}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="px-2 py-1 border border-slate-200 font-semibold">Q_in (m³/s)</td>
            {Q_IN.map((q, idx) => (
              <td key={idx} className="px-2 py-1 border border-slate-200 text-center">
                {q}
              </td>
            ))}
          </tr>
        </tbody>
      </table>

      <div className="flex flex-col sm:flex-row gap-6 mb-3">
        <div>
          <div className="text-xs font-bold text-slate-600 mb-1">
            Cross-Sectional Area A (m²)
          </div>
          <table className="text-xs font-mono border border-slate-300">
            <thead>
              <tr>
                <th className="px-2 py-1 border border-slate-300 bg-slate-50">t \ i</th>
                {[0, 1, 2].map((i) => (
                  <th key={i} className="px-2 py-1 border border-slate-300 bg-slate-50">
                    {i}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {[0, 1, 2, 3, 4].map((t) => (
                <tr key={t}>
                  <td className="px-2 py-1 border border-slate-300 bg-slate-50 font-semibold">
                    {t}
                  </td>
                  {[0, 1, 2].map((i) => {
                    const st = cellState(t, i, stepIdx);
                    const val = st === 'unrevealed' ? '?' : fmt(A_TABLE[t][i]);
                    return (
                      <td
                        key={i}
                        className={`px-2 py-1 border border-slate-300 text-center transition-colors ${cellClasses(
                          st
                        )}`}
                      >
                        {val}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div>
          <div className="text-xs font-bold text-slate-600 mb-1">Discharge Q (m³/s)</div>
          <table className="text-xs font-mono border border-slate-300">
            <thead>
              <tr>
                <th className="px-2 py-1 border border-slate-300 bg-slate-50">t \ i</th>
                {[0, 1, 2].map((i) => (
                  <th key={i} className="px-2 py-1 border border-slate-300 bg-slate-50">
                    {i}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {[0, 1, 2, 3, 4].map((t) => (
                <tr key={t}>
                  <td className="px-2 py-1 border border-slate-300 bg-slate-50 font-semibold">
                    {t}
                  </td>
                  {[0, 1, 2].map((i) => {
                    const st = cellState(t, i, stepIdx);
                    const val = st === 'unrevealed' ? '?' : fmt(Q_TABLE[t][i]);
                    return (
                      <td
                        key={i}
                        className={`px-2 py-1 border border-slate-300 text-center transition-colors ${cellClasses(
                          st
                        )}`}
                      >
                        {val}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <pre className="bg-slate-50 rounded-lg p-3 font-mono text-xs border border-slate-200 min-h-[100px] whitespace-pre-wrap overflow-x-auto">
        {formulaText}
      </pre>

      <div className="flex gap-2 mt-3">
        <button
          onClick={() => setStepIdx(-1)}
          className="text-xs font-semibold px-3 py-1.5 rounded-full bg-slate-100 text-slate-500 hover:bg-slate-200 transition"
        >
          ⏮ Reset
        </button>
        <button
          onClick={() => setStepIdx((s) => Math.min(s + 1, STEPS.length - 1))}
          disabled={stepIdx >= STEPS.length - 1}
          className="text-xs font-semibold px-3 py-1.5 rounded-full bg-indigo-600 text-white hover:bg-indigo-700 disabled:opacity-40 disabled:cursor-not-allowed transition"
        >
          ▶ Compute next cell
        </button>
        <button
          onClick={() => setStepIdx(STEPS.length - 1)}
          className="text-xs font-semibold px-3 py-1.5 rounded-full bg-slate-200 text-slate-700 hover:bg-slate-300 transition"
        >
          Show all
        </button>
      </div>

      {showObservation && (
        <div className="mt-4 rounded-xl border border-indigo-200 bg-indigo-50 p-4 text-sm text-slate-700">
          <div className="font-bold text-indigo-800 mb-1">
            👀 The flood wave advances exactly one reach per time step:
          </div>
          <ul className="list-disc ml-5 space-y-0.5">
            <li>Peak Q = 12.00 m³/s at i=0 at t=2</li>
            <li>Peak Q ≈ 11.11 m³/s at i=1 at t=3</li>
            <li>Peak Q ≈ 9.49 m³/s at i=2 at t=4</li>
          </ul>
          <p className="mt-2">
            No peak attenuation — kinematic wave translates without changing shape.
          </p>
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────
// Section 6 — Courant condition + live instability demo
// ─────────────────────────────────────────────────────────────────────────
const STABILITY_HYDROGRAPH = [2, 8, 12, 8, 4, 2, 2, 2];
const C_CELERITY = 1.446; // m/s at Q=10, A=11.52 (reference celerity used for the demo)

function runMiniSim(dt: number): number[][] {
  // 3 reaches (i=0,1,2), i=0 is boundary driven by STABILITY_HYDROGRAPH
  const steps = STABILITY_HYDROGRAPH.length;
  const r = dt / DX;
  let A = [A0, A0, A0];
  const Qrows: number[][] = [];
  for (let t = 0; t < steps; t++) {
    const Qin = STABILITY_HYDROGRAPH[t];
    const Q = A.map((a) => ALPHA * Math.pow(Math.abs(a), BETA) * Math.sign(a || 1));
    const Qup = [Qin, Q[0], Q[1]];
    Qrows.push([Qin, Q[1], Q[2]]);
    const Anew = A.map((a, i) => a - r * (Q[i] - Qup[i]));
    A = Anew;
  }
  return Qrows;
}

// Extended version: also returns the full A trace (including the t=0 initial
// condition row) so the animated comparison can flag negative AREA — the root
// cause of instability — not just the resulting negative/blown-up Q.
interface MiniSimTrace {
  Q: number[][]; // Q[t][i], t = 0..steps-1 (matches STABILITY_HYDROGRAPH indices)
  A: number[][]; // A[t][i], t = 0..steps   (row 0 is the initial condition A0)
}

function runMiniSimFull(dt: number): MiniSimTrace {
  const steps = STABILITY_HYDROGRAPH.length;
  const r = dt / DX;
  let A = [A0, A0, A0];
  const Qrows: number[][] = [];
  const Arows: number[][] = [[...A]];
  for (let t = 0; t < steps; t++) {
    const Qin = STABILITY_HYDROGRAPH[t];
    const Q = A.map((a) => ALPHA * Math.pow(Math.abs(a), BETA) * Math.sign(a || 1));
    const Qup = [Qin, Q[0], Q[1]];
    Qrows.push([Qin, Q[1], Q[2]]);
    const Anew = A.map((a, i) => a - r * (Q[i] - Qup[i]));
    A = Anew;
    Arows.push([...A]);
  }
  return { Q: Qrows, A: Arows };
}

const PRESETS = [
  { label: 'Too Small', dt: 60, accent: 'blue' },
  { label: 'Just Right', dt: 300, accent: 'green' },
  { label: 'Too Big', dt: 500, accent: 'red' },
] as const;

const INPUT_PEAK = 12; // m^3/s, the hydrograph's peak inflow

type Accent = (typeof PRESETS)[number]['accent'];

const ACCENT_BADGE: Record<Accent, string> = {
  blue: 'bg-blue-100 text-blue-700 border border-blue-300',
  green: 'bg-green-100 text-green-700 border border-green-300',
  red: 'bg-red-100 text-red-700 border border-red-300',
};

const ACCENT_HEADER: Record<Accent, string> = {
  blue: 'border-blue-200 bg-blue-50',
  green: 'border-green-200 bg-green-50',
  red: 'border-red-200 bg-red-50',
};

const ACCENT_TEXT: Record<Accent, string> = {
  blue: 'text-blue-700',
  green: 'text-green-700',
  red: 'text-red-700',
};

function SectionStability() {
  const [animStep, setAnimStep] = useState(0); // 0..8, shared across all 3 lanes
  const [playing, setPlaying] = useState(false);

  const maxStableDt = DX / C_CELERITY;
  const maxAnimStep = STABILITY_HYDROGRAPH.length; // 8 (rows t=1..8 in the trace)

  // Precompute the full 8-step A/Q trace for each preset once.
  const traces = useMemo(
    () => PRESETS.map((p) => ({ ...p, trace: runMiniSimFull(p.dt) })),
    []
  );

  // Animation loop — mirrors the playing/useEffect interval pattern in
  // ManningCelerityWidget.tsx (read-only reference, not modified).
  useEffect(() => {
    if (!playing) return;
    const interval = setInterval(() => {
      setAnimStep((s) => {
        if (s >= maxAnimStep) {
          setPlaying(false);
          return s;
        }
        return s + 1;
      });
    }, 700);
    return () => clearInterval(interval);
  }, [playing, maxAnimStep]);

  return (
    <div>
      <SectionHeader n={6} title="The Courant Condition — When C > 1, It Breaks" />

      <p className="text-sm text-slate-700 mb-2">
        The explicit scheme advances information exactly one cell-width Δx per time step. But
        the flood wave travels c·Δt meters per step. If c·Δt &gt; Δx — if the wave moves MORE
        than one cell per step — the scheme loses track of it.
      </p>
      <p className="text-sm text-slate-700 mb-3">
        Think of it like throwing a ball and trying to catch it: if it flies past you in less
        time than it takes your arm to reach, you can&apos;t catch it. The scheme
        &ldquo;misses&rdquo; the wave and produces oscillating, unphysical values.
      </p>

      <div className="flex justify-center mb-4">
        <svg width={300} height={90}>
          <text x={10} y={14} fontSize="9" fill="#475569">
            stable
          </text>
          <rect x={10} y={20} width={60} height={20} fill="white" stroke="#cbd5e1" />
          <rect x={70} y={20} width={60} height={20} fill="white" stroke="#cbd5e1" />
          <rect x={130} y={20} width={60} height={20} fill="#dcfce7" stroke="#22c55e" />
          <line x1={130} y1={30} x2={182} y2={30} stroke="#16a34a" strokeWidth={3} markerEnd="url(#arrG)" />
          <defs>
            <marker id="arrG" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
              <path d="M0,0 L6,3 L0,6 Z" fill="#16a34a" />
            </marker>
            <marker id="arrR" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
              <path d="M0,0 L6,3 L0,6 Z" fill="#dc2626" />
            </marker>
          </defs>
          <text x={300 / 2} y={48} textAnchor="middle" fontSize="9" fill="#15803d">
            C=0.87: wave reaches 87% of cell ✓
          </text>

          <text x={10} y={64} fontSize="9" fill="#475569">
            unstable
          </text>
          <rect x={10} y={70} width={60} height={0} />
          <g transform="translate(0, 50)">
            <rect x={130} y={20} width={60} height={20} fill="#fee2e2" stroke="#dc2626" />
            <rect x={190} y={20} width={60} height={20} fill="white" stroke="#cbd5e1" />
            <line x1={130} y1={30} x2={200} y2={30} stroke="#dc2626" strokeWidth={3} markerEnd="url(#arrR)" />
          </g>
          <text x={300 / 2} y={88} textAnchor="middle" fontSize="9" fill="#b91c1c">
            C=1.16: wave jumps past the cell ✗
          </text>
        </svg>
      </div>

      <div className="bg-slate-800 text-white rounded-lg p-4 font-mono text-sm mb-4 overflow-x-auto">
        <pre className="whitespace-pre-wrap">
{`C = c · Δt / Δx  ≤  1

c  = (5/3)·Q/A = wave celerity [m/s]
Δt = time step size [s]
Δx = cell width [m]

At defaults (Q=10, A=11.52):  c = 1.446 m/s
Max stable Δt = Δx / c = 500 / 1.446 = 345 s
Default Δt = 300 s → C = 0.868  ✓`}
        </pre>
      </div>

      <div className="rounded-xl border border-slate-200 p-4">
        <div className="text-sm font-semibold text-slate-700 mb-1">
          Sub-section B — Watch three Δt choices race the same flood, side by side
        </div>
        <p className="text-xs text-slate-600 mb-3">
          Same upstream hydrograph (2, 8, 12, 8, 4, 2, 2, 2 m³/s), same 3 reaches, same
          formula — only Δt differs. Press Play and watch reach i=2 (the bottom cell in each
          strip, furthest downstream) as the flood arrives.
        </p>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          {traces.map(({ label, dt, accent, trace }) => {
            const C = (C_CELERITY * dt) / DX;
            // Q/A at the current animStep: trace.Q/A rows are indexed 0..7 (Q) and
            // 0..8 (A, row 0 = initial condition). At animStep=0 we show the initial
            // condition row (A only; no Q step has run yet).
            const qRowIdx = animStep - 1; // -1 means "no step yet" — show baseflow Q
            const Q_BASEFLOW = ALPHA * Math.pow(A0, BETA);
            const currentQ = qRowIdx >= 0 ? trace.Q[qRowIdx] : [Q_BASEFLOW, Q_BASEFLOW, Q_BASEFLOW];
            const currentA = trace.A[animStep];

            const qSoFar = trace.Q.slice(0, Math.max(animStep, 0)).map((row) => row[2]);
            const peakSoFar = qSoFar.length > 0 ? Math.max(...qSoFar) : 0;

            const aSoFar = trace.A.slice(0, animStep + 1).map((row) => row[2]);
            const everNegative =
              aSoFar.some((a) => a < 0) || qSoFar.some((q) => q < 0);

            const damped = animStep >= maxAnimStep && peakSoFar < 0.3 * INPUT_PEAK;

            let statusLabel = '✓ stable';
            let statusClasses = 'bg-green-100 text-green-700 border border-green-300';
            if (everNegative) {
              statusLabel = '✗ UNSTABLE';
              statusClasses = 'bg-red-100 text-red-700 border border-red-300';
            } else if (damped) {
              statusLabel = '⚠ heavily damped';
              statusClasses = 'bg-amber-100 text-amber-700 border border-amber-300';
            }

            return (
              <div
                key={label}
                className={`rounded-lg border p-3 ${ACCENT_HEADER[accent as Accent]}`}
              >
                <div className="flex items-center justify-between mb-2 flex-wrap gap-1">
                  <span className={`text-xs font-bold uppercase ${ACCENT_TEXT[accent as Accent]}`}>
                    {label}
                  </span>
                  <span
                    className={`text-xs font-bold px-2 py-0.5 rounded-full ${ACCENT_BADGE[accent as Accent]}`}
                  >
                    Δt = {dt} s → C = {fmt(C)}
                  </span>
                </div>

                {/* Vertical 3-cell reach strip: i=0 (top, upstream) .. i=2 (bottom, downstream) */}
                <table className="text-xs font-mono border border-slate-300 w-full bg-white">
                  <thead>
                    <tr>
                      <th className="px-1.5 py-0.5 border border-slate-300 bg-slate-50">i</th>
                      <th className="px-1.5 py-0.5 border border-slate-300 bg-slate-50">A (m²)</th>
                      <th className="px-1.5 py-0.5 border border-slate-300 bg-slate-50">Q (m³/s)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[0, 1, 2].map((i) => {
                      const aVal = currentA[i];
                      const qVal = currentQ[i];
                      const flagged = aVal < 0 || qVal < 0 || qVal > 20;
                      return (
                        <tr key={i}>
                          <td className="px-1.5 py-0.5 border border-slate-300 bg-slate-50 text-center">
                            {i}
                          </td>
                          <td
                            className={`px-1.5 py-0.5 border border-slate-300 text-center transition-colors ${
                              flagged ? 'bg-red-100 text-red-700 font-bold' : 'text-slate-700'
                            }`}
                          >
                            {fmt(aVal)}
                          </td>
                          <td
                            className={`px-1.5 py-0.5 border border-slate-300 text-center transition-colors ${
                              flagged ? 'bg-red-100 text-red-700 font-bold' : 'text-slate-700'
                            }`}
                          >
                            {fmt(qVal)}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>

                <div className="text-xs text-slate-600 mt-2">
                  Peak Q so far (i=2): <strong className="text-slate-800">{fmt(peakSoFar)} m³/s</strong>
                </div>

                <div className="mt-2">
                  <span className={`text-xs font-bold px-2 py-0.5 rounded-full ${statusClasses}`}>
                    {statusLabel}
                  </span>
                </div>
              </div>
            );
          })}
        </div>

        <div className="flex gap-2 mt-4 items-center flex-wrap">
          <button
            onClick={() => {
              setPlaying(false);
              setAnimStep(0);
            }}
            className="text-xs font-semibold px-3 py-1.5 rounded-full bg-slate-100 text-slate-500 hover:bg-slate-200 transition"
          >
            ⏮ Reset
          </button>
          <button
            onClick={() => setPlaying((p) => !p)}
            disabled={animStep >= maxAnimStep && !playing}
            className="text-xs font-semibold px-3 py-1.5 rounded-full bg-indigo-600 text-white hover:bg-indigo-700 disabled:opacity-40 disabled:cursor-not-allowed transition"
          >
            {playing ? '⏸ Pause' : '▶ Play'}
          </button>
          <button
            onClick={() => setAnimStep((s) => Math.min(s + 1, maxAnimStep))}
            disabled={animStep >= maxAnimStep}
            className="text-xs font-semibold px-3 py-1.5 rounded-full bg-slate-200 text-slate-700 hover:bg-slate-300 disabled:opacity-40 disabled:cursor-not-allowed transition"
          >
            Step ▶
          </button>
          <span className="text-xs text-slate-500">
            Step {animStep} of {maxAnimStep} · Stability boundary: Δt ≈ {Math.round(maxStableDt)} s
          </span>
        </div>

        <p className="text-xs text-slate-600 mt-3">
          At C &gt; 1, the scheme subtracts more water from a cell than it contains — producing
          negative (unphysical) areas. The simulation diverges within a few steps. OPM fixes
          this with a flux limiter (see §4.4).
        </p>

        {animStep >= maxAnimStep && (
          <div className="mt-4 rounded-xl border border-indigo-200 bg-indigo-50 p-4 text-sm text-slate-700">
            <div className="font-bold text-indigo-800 mb-1">
              👀 After all 8 steps — three outcomes from the same flood:
            </div>
            <ul className="list-disc ml-5 space-y-0.5">
              <li>
                <strong>Too Small</strong> (Δt=60s, C=0.17) → peak only reached 2.10 m³/s — the
                flood was almost completely erased by numerical diffusion.
              </li>
              <li>
                <strong>Just Right</strong> (Δt=300s, C=0.87) → peak reached 7.97 m³/s, a
                faithful (if smoothed) translation.
              </li>
              <li>
                <strong>Too Big</strong> (Δt=500s, C=1.45) → area went negative at step 6 — a
                physically impossible result.
              </li>
            </ul>
            <p className="mt-2">
              Counter-intuitively, smaller Δt is not &ldquo;more accurate&rdquo; here — it just
              trades instability for excessive smoothing. Numerical diffusion in the upwind
              scheme is largest as C→0, not smallest, because the scheme blends in a fraction
              (1−C) of old, un-shifted data every step. The sweet spot is C close to (but not
              above) 1.
            </p>
          </div>
        )}
      </div>

      <div className="bg-green-50 border border-green-300 rounded-xl p-4 mt-4 text-sm text-slate-700">
        <div className="font-bold text-green-800 mb-1">
          ✓ You now understand the complete explicit scheme from first principles:
        </div>
        <ul className="list-disc ml-5 space-y-0.5">
          <li>Finite differences replace ∂ with Δ</li>
          <li>Upwind scheme uses upstream Q</li>
          <li>Explicit means one unknown per step</li>
          <li>Courant condition C ≤ 1 ensures stability</li>
        </ul>
        <p className="mt-2">
          The interactive simulation below runs this same algorithm across 8 reaches and 12
          time steps — every cell visible, every number computed.
        </p>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────
// Root widget
// ─────────────────────────────────────────────────────────────────────────
export default function FiniteDiffBuildupWidget() {
  return (
    <div className="not-prose my-8 rounded-2xl border border-slate-200 bg-white shadow-xl overflow-hidden font-sans">
      <div className="bg-gradient-to-r from-indigo-700 to-slate-800 px-6 py-4">
        <h3 className="text-white font-bold text-lg tracking-tight">
          From Calculus to Computation
        </h3>
        <p className="text-indigo-200 text-sm mt-0.5">
          Build the explicit scheme from first principles — then walk through it by hand
        </p>
      </div>
      <div className="p-6 space-y-10">
        <SectionProblem />
        <SectionGrid />
        <SectionDerivatives />
        <SectionFormula />
        <SectionHandCalc />
        <SectionStability />
      </div>
    </div>
  );
}
