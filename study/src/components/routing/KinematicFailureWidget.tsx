'use client';

import React, { useState, useMemo } from 'react';

// ─────────────────────────────────────────────────────────────────────────
// Ground-truth physics, ported from routing_utils.py
//
//   mannings_velocity(depth, slope, n) = (1/n) * depth^(2/3) * slope^0.5
//   cell_discharge(depth, velocity, cell_size) = velocity * cell_size * depth
//   => Q_kinematic = (1/n) * depth^(5/3) * slope^0.5 * cell_size
//
//   Diffusive wave (CASC2D/GSSHA-style), theta = 1 (OPM production default):
//     S_w      = S0 + theta * (h_i - h_ds) / dist
//     S_eff    = max(S_w, 0)
//     WSE_i    = z_i + h_i ;  WSE_ds = z_ds + h_ds
//     h_higher = max(WSE_i, WSE_ds) - max(z_i, z_ds)
//     h_flow   = (1 - theta) * h_i + theta * h_higher
//     Q_diff   = (1/n) * h_flow^(5/3) * S_eff^0.5 * cell_size
// ─────────────────────────────────────────────────────────────────────────

function manningsVelocity(depth: number, slope: number, n: number): number {
  if (depth <= 0 || slope <= 0) return 0;
  return (1.0 / n) * Math.pow(depth, 2.0 / 3.0) * Math.sqrt(slope);
}

function cellDischarge(depth: number, velocity: number, cellSize: number): number {
  return velocity * cellSize * depth;
}

// Fixed scenario inputs (do not change — these are the worked example).
const S0 = 0.001;
const N = 0.05;
const DIST = 100; // m, also used as cell_size in the discharge formula
const Z_A = 5.0;
const Z_B = 4.9;
const H_A = 0.008; // m, held fixed across all scenarios (A is generating steady inflow)
const THETA = 1.0; // OPM production default — "full diffusive"

interface Scenario {
  key: string;
  label: string;
  hB: number;
  blurb: string;
}

const SCENARIOS: Scenario[] = [
  {
    key: 'normal',
    label: 'Normal',
    hB: 0.003,
    blurb: 'B is relatively empty — water drains downhill as expected.',
  },
  {
    key: 'backwater',
    label: 'Backwater',
    hB: 0.05,
    blurb: 'B already holds a pool — it should slow drainage from A.',
  },
  {
    key: 'adverse',
    label: 'Adverse',
    hB: 0.11,
    blurb: "B's water surface is now HIGHER than A's — flow should stop.",
  },
];

interface Result {
  hA: number;
  hB: number;
  wseA: number;
  wseB: number;
  S_w: number;
  S_eff: number;
  h_higher: number;
  h_flow: number;
  Q_kin: number;
  Q_diff: number;
  pctDiff: number;
}

function compute(hB: number): Result {
  const hA = H_A;
  const wseA = Z_A + hA;
  const wseB = Z_B + hB;

  // Kinematic wave: only ever looks at A's own depth and the fixed bed slope S0.
  const vKin = manningsVelocity(hA, S0, N);
  const Q_kin = cellDischarge(hA, vKin, DIST);

  // Diffusive wave: slope of the WATER SURFACE between the two cells.
  const S_w = S0 + (THETA * (hA - hB)) / DIST;
  const S_eff = Math.max(S_w, 0);
  const h_higher = Math.max(wseA, wseB) - Math.max(Z_A, Z_B);
  const h_flow = (1 - THETA) * hA + THETA * h_higher;
  const vDiff = manningsVelocity(h_flow, S_eff, N);
  const Q_diff = S_eff > 0 ? cellDischarge(h_flow, vDiff, DIST) : 0;

  const pctDiff = Q_kin !== 0 ? ((Q_diff - Q_kin) / Q_kin) * 100 : 0;

  return { hA, hB, wseA, wseB, S_w, S_eff, h_higher, h_flow, Q_kin, Q_diff, pctDiff };
}

function fmt(n: number, d = 5): string {
  return n.toFixed(d);
}

// ─────────────────────────────────────────────────────────────────────────
// SVG side-view diagram of the two cells
// ─────────────────────────────────────────────────────────────────────────
const SVG_W = 360;
const SVG_H = 200;
const PAD_L = 30;
const PAD_R = 20;
const PAD_T = 20;
const PAD_B = 30;

// Bed elevations span roughly 4.85 .. 5.05; water surfaces can reach ~5.01.
// Map elevation z (m) to a y pixel, with a fixed reference band.
const Z_MIN = 4.84;
const Z_MAX = 5.04;
function yOfZ(z: number): number {
  const t = (z - Z_MIN) / (Z_MAX - Z_MIN);
  return SVG_H - PAD_B - t * (SVG_H - PAD_T - PAD_B);
}

function CellDiagram({ result }: { result: Result }) {
  const cellW = (SVG_W - PAD_L - PAD_R) / 2;
  const xA0 = PAD_L;
  const xA1 = PAD_L + cellW;
  const xB0 = xA1;
  const xB1 = xA1 + cellW;

  const bedA = yOfZ(Z_A);
  const bedB = yOfZ(Z_B);
  const wseA = yOfZ(result.wseA);
  const wseB = yOfZ(result.wseB);

  const wseHigherInB = result.wseB > result.wseA;

  return (
    <svg
      width={SVG_W}
      height={SVG_H}
      className="block overflow-visible"
      style={{ maxWidth: '100%' }}
    >
      {/* baseline grid */}
      <line
        x1={PAD_L}
        y1={SVG_H - PAD_B}
        x2={SVG_W - PAD_R}
        y2={SVG_H - PAD_B}
        stroke="#e2e8f0"
        strokeWidth={1}
      />

      {/* Bed (brown/tan) — descending step from z_A to z_B */}
      <polygon
        points={`${xA0},${SVG_H - PAD_B} ${xA0},${bedA} ${xA1},${bedA} ${xA1},${bedB} ${xB1},${bedB} ${xB1},${SVG_H - PAD_B}`}
        fill="#c9a577"
        stroke="#92703f"
        strokeWidth={1}
      />

      {/* Water depth on cell A (blue) */}
      <rect
        x={xA0}
        y={wseA}
        width={cellW}
        height={Math.max(bedA - wseA, 0)}
        fill="#60a5fa"
        opacity={0.75}
      />
      {/* Water depth on cell B (blue) */}
      <rect
        x={xB0}
        y={wseB}
        width={cellW}
        height={Math.max(bedB - wseB, 0)}
        fill="#60a5fa"
        opacity={0.75}
      />

      {/* Cell divider */}
      <line
        x1={xA1}
        y1={PAD_T}
        x2={xA1}
        y2={SVG_H - PAD_B}
        stroke="#94a3b8"
        strokeWidth={1}
        strokeDasharray="3,3"
      />

      {/* Water surface line, cell A */}
      <line
        x1={xA0}
        y1={wseA}
        x2={xA1}
        y2={wseA}
        stroke="#1d4ed8"
        strokeWidth={2}
      />
      {/* Water surface line, cell B */}
      <line
        x1={xB0}
        y1={wseB}
        x2={xB1}
        y2={wseB}
        stroke={wseHigherInB ? '#dc2626' : '#1d4ed8'}
        strokeWidth={2}
      />
      {/* Connector showing the WSE jump between cells (visual emphasis) */}
      <line
        x1={xA1}
        y1={wseA}
        x2={xB0}
        y2={wseB}
        stroke={wseHigherInB ? '#dc2626' : '#1d4ed8'}
        strokeWidth={1.5}
        strokeDasharray="2,2"
      />

      {/* Bed labels */}
      <text x={xA0 + cellW / 2} y={bedA + 14} textAnchor="middle" fontSize={9} fill="#6b4f2a">
        z_A = {Z_A.toFixed(2)}
      </text>
      <text x={xB0 + cellW / 2} y={bedB + 14} textAnchor="middle" fontSize={9} fill="#6b4f2a">
        z_B = {Z_B.toFixed(2)}
      </text>

      {/* WSE labels */}
      <text
        x={xA0 + cellW / 2}
        y={wseA - 5}
        textAnchor="middle"
        fontSize={9}
        fontWeight="bold"
        fill="#1d4ed8"
      >
        WSE_A = {result.wseA.toFixed(3)}
      </text>
      <text
        x={xB0 + cellW / 2}
        y={wseB - 5}
        textAnchor="middle"
        fontSize={9}
        fontWeight="bold"
        fill={wseHigherInB ? '#dc2626' : '#1d4ed8'}
      >
        WSE_B = {result.wseB.toFixed(3)}
      </text>

      {/* Cell labels */}
      <text x={xA0 + 4} y={PAD_T - 6} fontSize={10} fontWeight="bold" fill="#334155">
        Cell A (upstream)
      </text>
      <text x={xB0 + 4} y={PAD_T - 6} fontSize={10} fontWeight="bold" fill="#334155">
        Cell B (downstream)
      </text>

      {wseHigherInB && (
        <text
          x={SVG_W / 2}
          y={PAD_T + 8}
          textAnchor="middle"
          fontSize={9}
          fontWeight="bold"
          fill="#dc2626"
        >
          ⚠ WSE_B &gt; WSE_A — water surface slopes UPHILL
        </text>
      )}

      {/* Flow arrow (kinematic always "wants" to push A→B) */}
      <defs>
        <marker id="kfw-arrow" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
          <path d="M0,0 L6,3 L0,6 Z" fill="#2563eb" />
        </marker>
      </defs>
      <line
        x1={xA1 - 30}
        y1={PAD_T + 22}
        x2={xA1 + 2}
        y2={PAD_T + 22}
        stroke="#2563eb"
        strokeWidth={2}
        markerEnd="url(#kfw-arrow)"
      />
      <text x={xA1 - 30} y={PAD_T + 16} fontSize={8} fill="#2563eb">
        Q_kin always flows →
      </text>
    </svg>
  );
}

// ─────────────────────────────────────────────────────────────────────────
// Root widget
// ─────────────────────────────────────────────────────────────────────────
export default function KinematicFailureWidget() {
  const [scenarioKey, setScenarioKey] = useState<string>('normal');

  const scenario = useMemo(
    () => SCENARIOS.find((s) => s.key === scenarioKey) ?? SCENARIOS[0],
    [scenarioKey]
  );
  const result = useMemo(() => compute(scenario.hB), [scenario.hB]);

  const isAdverse = scenario.key === 'adverse';

  return (
    <div className="not-prose my-8 rounded-2xl border border-slate-200 bg-white shadow-xl overflow-hidden font-sans">
      <div className="bg-gradient-to-r from-red-700 to-rose-900 px-6 py-4">
        <h3 className="text-white font-bold text-lg tracking-tight">
          Where Kinematic Wave Fails
        </h3>
        <p className="text-rose-200 text-sm mt-0.5">
          A real, numerically concrete case: two cells, one fixed bed slope, three downstream
          depths
        </p>
      </div>

      <div className="p-6 flex flex-col gap-5">
        {/* Intuition callout */}
        <div className="bg-amber-50 border border-amber-200 rounded-lg p-3 text-sm text-amber-900">
          <span className="font-semibold">Why this happens: </span>
          The kinematic wave computes discharge from cell A&apos;s own depth and the
          <em> fixed bed slope</em> S₀ alone — it never looks at cell B. So it has no way to
          &ldquo;see&rdquo; a pool forming downstream, let alone a water surface that locally
          slopes back uphill. The diffusive wave fixes this by using the slope of the{' '}
          <strong>water surface</strong> (bed + depth) between the two cells instead of just the
          bed — so it can sense backwater and shut off flow when it should.
        </div>

        {/* Scenario tabs */}
        <div>
          <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2">
            Select scenario — h_A is held fixed at {H_A.toFixed(3)} m in all three
          </p>
          <div className="flex gap-2 flex-wrap">
            {SCENARIOS.map((s) => (
              <button
                key={s.key}
                onClick={() => setScenarioKey(s.key)}
                className={`px-4 py-2 rounded-lg text-sm font-semibold transition-colors border ${
                  scenarioKey === s.key
                    ? s.key === 'adverse'
                      ? 'bg-red-600 text-white border-red-600'
                      : 'bg-sky-600 text-white border-sky-600'
                    : 'bg-slate-50 text-slate-700 border-slate-200 hover:bg-slate-100'
                }`}
              >
                {s.label}
                <span className="ml-2 font-mono text-xs opacity-80">
                  h_B={s.hB.toFixed(3)}
                </span>
              </button>
            ))}
          </div>
          <p className="text-xs text-slate-500 mt-2">{scenario.blurb}</p>
        </div>

        {/* Diagram + readout side by side */}
        <div className="flex flex-col lg:flex-row gap-6">
          <div className="flex-1 min-w-0">
            <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2">
              Side view (to scale on elevation)
            </p>
            <CellDiagram result={result} />
          </div>

          <div className="flex-1 min-w-0 flex flex-col gap-3">
            <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide">
              Live readout — {scenario.label} scenario
            </p>
            <div className="overflow-x-auto">
              <table className="border-collapse text-xs w-full">
                <thead>
                  <tr className="bg-slate-100">
                    <th className="border border-slate-200 px-2 py-1 text-left font-semibold">
                      Quantity
                    </th>
                    <th className="border border-slate-200 px-2 py-1 text-right font-semibold">
                      Value
                    </th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="hover:bg-slate-50">
                    <td className="border border-slate-200 px-2 py-1 font-mono">S_w</td>
                    <td
                      className={`border border-slate-200 px-2 py-1 text-right font-mono ${
                        result.S_w < 0 ? 'text-red-600 font-bold' : ''
                      }`}
                    >
                      {fmt(result.S_w, 5)}
                    </td>
                  </tr>
                  <tr className="hover:bg-slate-50">
                    <td className="border border-slate-200 px-2 py-1 font-mono">S_eff</td>
                    <td className="border border-slate-200 px-2 py-1 text-right font-mono">
                      {fmt(result.S_eff, 5)}
                    </td>
                  </tr>
                  <tr className="hover:bg-slate-50">
                    <td className="border border-slate-200 px-2 py-1 font-mono">h_higher</td>
                    <td className="border border-slate-200 px-2 py-1 text-right font-mono">
                      {fmt(result.h_higher, 3)} m
                    </td>
                  </tr>
                  <tr className="hover:bg-slate-50">
                    <td className="border border-slate-200 px-2 py-1 font-mono text-blue-700">
                      Q_kin
                    </td>
                    <td className="border border-slate-200 px-2 py-1 text-right font-mono text-blue-700">
                      {fmt(result.Q_kin, 5)} m³/s
                    </td>
                  </tr>
                  <tr className="hover:bg-slate-50">
                    <td className="border border-slate-200 px-2 py-1 font-mono text-red-700">
                      Q_diff
                    </td>
                    <td className="border border-slate-200 px-2 py-1 text-right font-mono text-red-700">
                      {fmt(result.Q_diff, 5)} m³/s
                    </td>
                  </tr>
                  <tr className="hover:bg-slate-50">
                    <td className="border border-slate-200 px-2 py-1 font-mono">% difference</td>
                    <td
                      className={`border border-slate-200 px-2 py-1 text-right font-mono font-semibold ${
                        result.pctDiff < 0 ? 'text-red-600' : 'text-green-700'
                      }`}
                    >
                      {result.pctDiff >= 0 ? '+' : ''}
                      {result.pctDiff.toFixed(1)}%
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>

            <div className="bg-slate-50 font-mono text-xs p-3 rounded border border-slate-200">
              S0 = {S0} &nbsp;n = {N} &nbsp;dist = {DIST} m
              <br />
              z_A = {Z_A.toFixed(2)} &nbsp;z_B = {Z_B.toFixed(2)} &nbsp;h_A = {H_A.toFixed(3)} m
              &nbsp;h_B = {scenario.hB.toFixed(3)} m
              <br />
              WSE_A = {result.wseA.toFixed(3)} m &nbsp;WSE_B = {result.wseB.toFixed(3)} m
            </div>
          </div>
        </div>

        {/* Adverse-case warning callout */}
        {isAdverse && (
          <div className="bg-red-50 border border-red-300 rounded-lg p-3 text-sm text-red-800">
            <span className="font-semibold">⚠ Kinematic wave is physically wrong here: </span>
            Q_kin = {fmt(result.Q_kin, 5)} m³/s — it keeps draining cell A into cell B even
            though B&apos;s water surface ({result.wseB.toFixed(3)} m) is already{' '}
            <em>higher</em> than A&apos;s ({result.wseA.toFixed(3)} m). The diffusive wave
            computes a negative water-surface slope (S_w = {fmt(result.S_w, 5)}), clamps it to
            S_eff = 0, and correctly halts flow: Q_diff = {fmt(result.Q_diff, 5)} m³/s.
            Kinematic has no mechanism to detect this — it only ever sees h_A and the fixed bed
            slope S₀, both of which are unchanged from the Normal scenario.
          </div>
        )}

        {/* Summary table across all three scenarios */}
        <div>
          <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-1">
            All three scenarios at a glance
          </p>
          <div className="overflow-x-auto">
            <table className="border-collapse text-xs w-full">
              <thead>
                <tr className="bg-slate-100">
                  <th className="border border-slate-200 px-2 py-1 text-left font-semibold">
                    Scenario
                  </th>
                  <th className="border border-slate-200 px-2 py-1 text-right font-semibold">
                    h_B (m)
                  </th>
                  <th className="border border-slate-200 px-2 py-1 text-right font-semibold text-blue-700">
                    Q_kin (m³/s)
                  </th>
                  <th className="border border-slate-200 px-2 py-1 text-right font-semibold text-red-700">
                    Q_diff (m³/s)
                  </th>
                  <th className="border border-slate-200 px-2 py-1 text-right font-semibold">
                    % diff
                  </th>
                </tr>
              </thead>
              <tbody>
                {SCENARIOS.map((s) => {
                  const r = compute(s.hB);
                  const active = s.key === scenarioKey;
                  return (
                    <tr
                      key={s.key}
                      className={active ? 'bg-sky-50' : 'hover:bg-slate-50'}
                    >
                      <td className="border border-slate-200 px-2 py-1 font-semibold">
                        {s.label}
                      </td>
                      <td className="border border-slate-200 px-2 py-1 text-right font-mono">
                        {s.hB.toFixed(3)}
                      </td>
                      <td className="border border-slate-200 px-2 py-1 text-right font-mono text-blue-700">
                        {fmt(r.Q_kin, 5)}
                      </td>
                      <td className="border border-slate-200 px-2 py-1 text-right font-mono text-red-700">
                        {fmt(r.Q_diff, 5)}
                      </td>
                      <td
                        className={`border border-slate-200 px-2 py-1 text-right font-mono ${
                          r.pctDiff < 0 ? 'text-red-600' : 'text-green-700'
                        }`}
                      >
                        {r.pctDiff >= 0 ? '+' : ''}
                        {r.pctDiff.toFixed(1)}%
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>

        {/* OPM connection box */}
        <div className="bg-sky-50 border border-sky-200 rounded-lg p-3 text-xs font-mono text-sky-800">
          In OPM routing_utils.py:
          <br />
          &nbsp;&nbsp;Q_kinematic uses only S0 and h_i — blind to h_ds
          <br />
          &nbsp;&nbsp;Q_diffusive uses S_w = S0 + θ·(h_i − h_ds)/dist, clamped to S_eff = max(S_w,
          0)
          <br />
          &nbsp;&nbsp;DIFFUSION_THETA = 1.0 in production (θ = 1 used throughout this example)
        </div>
      </div>
    </div>
  );
}
