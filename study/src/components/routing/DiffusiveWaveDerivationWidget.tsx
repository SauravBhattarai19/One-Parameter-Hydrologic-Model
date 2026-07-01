'use client';

import React, { useState, useMemo, useRef, useEffect } from 'react';

// ─────────────────────────────────────────────────────────────────────────
// Part A — progressive-reveal derivation, ported verbatim from
// docs/chapter3.tex §10 "Going Deeper (optional): Why It Is Called
// 'Diffusive'" (label sec:why_diffusive_math). Uses the SAME ①②③④⑤ term
// labels established in Chapter 4's SaintVenantWidget.tsx.
// ─────────────────────────────────────────────────────────────────────────

const DIFFUSIVITY_TABLE: { setting: string; s0: string; d: string; behavior: string }[] = [
  { setting: 'Flat valley floor', s0: '0.001', d: '≈ 0.15', behavior: 'strong attenuation (need diffusive)' },
  { setting: 'Moderate slope', s0: '0.010', d: '≈ 0.015', behavior: 'mild attenuation' },
  { setting: 'Steep hillslope', s0: '0.050', d: '≈ 0.003', behavior: 'negligible (kinematic is fine)' },
];

// ─────────────────────────────────────────────────────────────────────────
// Part B — live mini-simulation of the boxed advection-diffusion equation
//   ∂h'/∂t + c ∂h'/∂x = D ∂²h'/∂x²
// Upwind for advection, central difference for diffusion.
//
// Stability budget (generic pedagogical grid, not a literal OPM port):
//   N = 90 cells, dx = 1 (unit length), domain = 90 units
//   c = 1.0 (unit speed)
//   D slider range: [0, 0.45]  (unit²/time)
//   dt chosen so BOTH limits hold for the entire slider range:
//     CFL:        c·dt/dx <= 1        -> dt <= 1/1.0 = 1
//     Diffusion:  D·dt/dx^2 <= 0.5    -> dt <= 0.5/0.45 = 1.111
//   dt = 0.5 gives c·dt/dx = 0.5 (<=1, comfortable margin) and at D_max=0.45,
//   D·dt/dx^2 = 0.225 (<=0.5, comfortable margin). Several sub-steps are run
//   per animation frame so the pulse visibly evolves.
// ─────────────────────────────────────────────────────────────────────────

const N_CELLS = 90;
const DX_SIM = 1;
const C_SIM = 1.0;
const DT_SIM = 0.5; // satisfies CFL (0.5) and diffusion limit (<=0.5*dx^2/D for D<=0.45 -> 0.225) for all slider D
const SUBSTEPS_PER_FRAME = 4;
const D_MIN = 0;
const D_MAX = 0.45;
const PULSE_CENTER = 12;
const PULSE_WIDTH = 3;
const PULSE_HEIGHT = 1;

function initialPulse(): number[] {
  const arr = new Array(N_CELLS).fill(0);
  for (let i = 0; i < N_CELLS; i++) {
    const dist = i - PULSE_CENTER;
    arr[i] = PULSE_HEIGHT * Math.exp(-(dist * dist) / (2 * PULSE_WIDTH * PULSE_WIDTH));
  }
  return arr;
}

function stepAdvDiff(h: number[], D: number): number[] {
  const out = new Array(N_CELLS).fill(0);
  for (let i = 0; i < N_CELLS; i++) {
    const hC = h[i];
    const hL = i > 0 ? h[i - 1] : h[0];
    const hR = i < N_CELLS - 1 ? h[i + 1] : h[N_CELLS - 1];

    // Upwind advection (c > 0, flow moves left -> right, so use backward difference)
    const dhdx_adv = (hC - hL) / DX_SIM;

    // Central difference diffusion
    const d2hdx2 = (hR - 2 * hC + hL) / (DX_SIM * DX_SIM);

    const dhdt = -C_SIM * dhdx_adv + D * d2hdx2;
    out[i] = hC + DT_SIM * dhdt;
  }
  return out;
}

function runSubsteps(h: number[], D: number, n: number): number[] {
  let cur = h;
  for (let s = 0; s < n; s++) cur = stepAdvDiff(cur, D);
  return cur;
}

// SVG scaling helpers (makePath pattern, as in DiffusiveWaveWidget.tsx)
const SIM_SVG_W = 560;
const SIM_SVG_H = 200;
const SIM_PAD = { l: 38, r: 14, t: 14, b: 28 };

function simXScale(i: number): number {
  return SIM_PAD.l + (i / (N_CELLS - 1)) * (SIM_SVG_W - SIM_PAD.l - SIM_PAD.r);
}
function simYScale(v: number): number {
  return SIM_SVG_H - SIM_PAD.b - (v / PULSE_HEIGHT) * (SIM_SVG_H - SIM_PAD.t - SIM_PAD.b);
}
function makeSimPath(values: number[]): string {
  return values
    .map((v, i) => `${i === 0 ? 'M' : 'L'}${simXScale(i).toFixed(1)},${simYScale(Math.max(v, 0)).toFixed(2)}`)
    .join(' ');
}

// ─────────────────────────────────────────────────────────────────────────
// Small reusable bits matching the established visual language
// ─────────────────────────────────────────────────────────────────────────

function TermBadge({
  circle,
  symbol,
  label,
  color,
  dropped,
}: {
  circle: string;
  symbol: string;
  label: string;
  color: 'sky' | 'violet' | 'amber' | 'red' | 'green';
  dropped: boolean;
}) {
  const colorMap: Record<typeof color, string> = {
    sky: 'bg-sky-50 border-sky-300 text-sky-800',
    violet: 'bg-violet-50 border-violet-300 text-violet-800',
    amber: 'bg-amber-50 border-amber-300 text-amber-800',
    red: 'bg-red-50 border-red-300 text-red-800',
    green: 'bg-green-50 border-green-300 text-green-800',
  };
  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg border font-mono text-xs ${
        dropped ? 'bg-slate-50 border-slate-200 text-slate-400 line-through' : colorMap[color]
      }`}
    >
      <span className="font-bold">{circle}</span>
      <span>{symbol}</span>
      <span className="not-italic font-sans text-[10px] opacity-80">({label})</span>
    </span>
  );
}

export default function DiffusiveWaveDerivationWidget() {
  // ── Part A: progressive reveal state ──────────────────────────────────
  const [visibleSteps, setVisibleSteps] = useState(1);
  const TOTAL_STEPS = 4;

  // ── Part B: live mini-simulation state ────────────────────────────────
  const [D, setD] = useState(0.15);
  const [playing, setPlaying] = useState(false);
  const [frame, setFrame] = useState(0);
  const [history, setHistory] = useState<number[]>(() => initialPulse());
  const MAX_FRAMES = 220;

  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (!playing) return;
    if (frame >= MAX_FRAMES) {
      setPlaying(false);
      return;
    }
    timerRef.current = setTimeout(() => {
      setHistory((h) => runSubsteps(h, D, SUBSTEPS_PER_FRAME));
      setFrame((f) => f + 1);
    }, 40);
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [playing, frame, D]);

  function handlePlay() {
    if (frame >= MAX_FRAMES) handleReset();
    setPlaying(true);
  }
  function handlePause() {
    setPlaying(false);
  }
  function handleReset() {
    setPlaying(false);
    setFrame(0);
    setHistory(initialPulse());
  }

  const initial = useMemo(() => initialPulse(), []);
  const peakNow = Math.max(...history, 0.001);
  const peakInitial = Math.max(...initial);
  const attenuationPct = ((peakInitial - peakNow) / peakInitial) * 100;

  const gridXs = [0, 15, 30, 45, 60, 75, 89];
  const gridYs = [0, 0.25, 0.5, 0.75, 1.0];

  return (
    <div className="not-prose my-8 rounded-2xl border border-slate-200 bg-white shadow-xl overflow-hidden font-sans">
      {/* Header */}
      <div className="bg-gradient-to-r from-indigo-700 to-purple-900 px-6 py-4">
        <h3 className="text-white font-bold text-lg tracking-tight">
          Deriving the Diffusive-Wave Equation From Scratch
        </h3>
        <p className="text-indigo-200 text-sm mt-0.5">
          Same Saint-Venant starting point as Chapter 4 — one term survives, and that changes everything
        </p>
      </div>

      <div className="p-6 flex flex-col gap-6">
        {/* ── Continuity callout ── */}
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 text-xs text-blue-900">
          You saw in §4.1 that dropping <span className="font-mono font-bold">①②</span> (but keeping{' '}
          <span className="font-mono font-bold">③</span>) gives the diffusive wave — here&apos;s where that
          leads.
        </div>

        {/* ── Step 0: full momentum equation ── */}
        <div>
          <div className="text-xs font-bold text-slate-500 uppercase mb-1">
            Step 0 — The full Saint-Venant momentum equation (Chapter 2 / 4)
          </div>
          <div className="bg-slate-50 font-mono text-xs p-3 rounded flex flex-wrap items-center gap-2 justify-center">
            <TermBadge circle="①" symbol="∂U/∂t" label="local accel." color="sky" dropped={false} />
            <span>+</span>
            <TermBadge circle="②" symbol="U·∂U/∂x" label="convective accel." color="violet" dropped={false} />
            <span>+</span>
            <TermBadge circle="③" symbol="g·∂h/∂x" label="pressure gradient" color="amber" dropped={false} />
            <span>=</span>
            <TermBadge circle="④" symbol="g·S₀" label="gravity / bed slope" color="green" dropped={false} />
            <span>−</span>
            <TermBadge circle="⑤" symbol="g·S_f" label="friction slope" color="red" dropped={false} />
          </div>
          <p className="text-xs text-slate-600 mt-2">
            Recall the Saint-Venant momentum equation from Chapter 2 with its five terms — the same{' '}
            <span className="font-mono">①②③④⑤</span> labels from §4.1.
          </p>
        </div>

        {/* ── Step 1: drop ①②, keep ③ ── */}
        {visibleSteps >= 1 && (
          <div>
            <div className="text-xs font-bold text-slate-500 uppercase mb-1">
              Step 1 — Drop ①② only (NOT ③)
            </div>
            <div className="bg-slate-50 font-mono text-xs p-3 rounded flex flex-wrap items-center gap-2 justify-center">
              <TermBadge circle="①" symbol="∂U/∂t" label="local accel." color="sky" dropped={true} />
              <span className="text-slate-300">+</span>
              <TermBadge circle="②" symbol="U·∂U/∂x" label="convective accel." color="violet" dropped={true} />
              <span>+</span>
              <TermBadge circle="③" symbol="g·∂h/∂x" label="pressure gradient" color="amber" dropped={false} />
              <span>=</span>
              <TermBadge circle="④" symbol="g·S₀" label="gravity / bed slope" color="green" dropped={false} />
              <span>−</span>
              <TermBadge circle="⑤" symbol="g·S_f" label="friction slope" color="red" dropped={false} />
            </div>
            <div className="bg-amber-50 border border-amber-200 rounded-lg p-3 text-xs text-amber-900 mt-2">
              <span className="font-semibold">Kinematic</span> drops <span className="font-mono">①②③</span>;{' '}
              <span className="font-semibold">diffusive</span> drops only{' '}
              <span className="font-mono">①②</span>, keeping the pressure-gradient term{' '}
              <span className="font-mono">③</span> — that&apos;s the ONE difference that lets it see
              backwater.
            </div>
          </div>
        )}

        {/* ── Step 2: Sf = S0 - dh/dx = Sw ── */}
        {visibleSteps >= 2 && (
          <div>
            <div className="text-xs font-bold text-slate-500 uppercase mb-1">
              Step 2 — The momentum balance becomes
            </div>
            <div className="bg-slate-50 font-mono text-xs p-3 rounded text-center">
              g·∂h/∂x = g(S₀ − S_f) &nbsp;⟹&nbsp; S_f = S₀ − ∂h/∂x = S_w
            </div>
            <p className="text-xs text-slate-600 mt-2">
              i.e. the friction slope equals the water-surface slope — the{' '}
              <span className="font-semibold">continuous-form</span> version of the discrete{' '}
              <span className="font-mono">S_w = S₀ + (h_i − h_ds)/dist</span> formula from §5.2. If
              you&apos;ve used that widget, this is the same idea written with a derivative instead of a
              finite difference.
            </p>
          </div>
        )}

        {/* ── Step 3: linearize, boxed result ── */}
        {visibleSteps >= 3 && (
          <div>
            <div className="text-xs font-bold text-slate-500 uppercase mb-1">
              Step 3 — Linearize about a reference state (h₀, V₀)
            </div>
            <p className="text-xs text-slate-600 mb-2">
              Write h = h₀ + h′ with h′ small, substitute into continuity. After the algebra, the small
              disturbance h′ obeys the boxed{' '}
              <span className="font-semibold">advection–diffusion equation</span>:
            </p>
            <div className="bg-white border-2 border-indigo-300 rounded-lg p-4 font-mono text-sm text-center">
              ∂h′/∂t&nbsp;+&nbsp;c_k·∂h′/∂x&nbsp;=&nbsp;D·∂²h′/∂x²
            </div>
            <div className="grid sm:grid-cols-2 gap-3 mt-3">
              <div className="bg-green-50 border border-green-200 rounded-lg p-3 text-xs text-green-900">
                <span className="font-mono font-semibold">c_k = (5/3)·V₀</span>
                <p className="mt-1">
                  The wave speed — the <span className="font-semibold">same celerity formula</span> from
                  Chapter 4 §4.2 (c = β·u with β = 5/3). Nice continuity: the diffusive wave still
                  translates at the kinematic celerity; it just also spreads.
                </p>
              </div>
              <div className="bg-violet-50 border border-violet-200 rounded-lg p-3 text-xs text-violet-900">
                <span className="font-mono font-semibold">D = V₀·h₀ / (2·S₀)</span>
                <p className="mt-1">Hydraulic diffusivity, m²/s — governs how fast the pulse spreads.</p>
              </div>
            </div>
          </div>
        )}

        {/* ── Step 4: why "diffusive" ── */}
        {visibleSteps >= 4 && (
          <div className="bg-indigo-50 border border-indigo-200 rounded-lg p-3 text-xs text-indigo-900">
            <span className="font-semibold">Why it&apos;s called &ldquo;diffusive&rdquo;:</span> this
            equation has the exact same form as the heat/diffusion equation. The advection term{' '}
            <span className="font-mono">c_k·∂h′/∂x</span> <span className="font-semibold">translates</span>{' '}
            the wave; the <span className="font-mono">D·∂²h′/∂x²</span> term{' '}
            <span className="font-semibold">spreads</span> it — exactly like heat spreading from a hot
            spot, or ink dispersing in water. This is the mathematical origin of the attenuation (peak
            flattening) you saw demonstrated in §5.1.
          </div>
        )}

        {/* ── Progressive reveal controls ── */}
        <div className="flex gap-2">
          <button
            onClick={() => setVisibleSteps((s) => Math.min(s + 1, TOTAL_STEPS))}
            disabled={visibleSteps >= TOTAL_STEPS}
            className="text-xs font-semibold px-3 py-1.5 rounded-full bg-indigo-600 text-white hover:bg-indigo-700 disabled:opacity-40 disabled:cursor-not-allowed transition"
          >
            Show next step ▶
          </button>
          <button
            onClick={() => setVisibleSteps(TOTAL_STEPS)}
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

        {/* ── Diffusivity reference table (verbatim from chapter3.tex §10) ── */}
        {visibleSteps >= 3 && (
          <div>
            <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2">
              Reading the diffusivity — D = V₀h₀ / (2S₀)
            </p>
            <div className="overflow-x-auto">
              <table className="w-full text-xs border-collapse">
                <thead>
                  <tr className="bg-slate-100 text-slate-600">
                    <th className="border border-slate-200 px-2 py-1.5 text-left font-semibold">Setting</th>
                    <th className="border border-slate-200 px-2 py-1.5 text-left font-semibold">S₀</th>
                    <th className="border border-slate-200 px-2 py-1.5 text-left font-semibold">D [m²/s]</th>
                    <th className="border border-slate-200 px-2 py-1.5 text-left font-semibold">Behaviour</th>
                  </tr>
                </thead>
                <tbody>
                  {DIFFUSIVITY_TABLE.map((row) => (
                    <tr key={row.setting} className="bg-white">
                      <td className="border border-slate-200 px-2 py-1.5">{row.setting}</td>
                      <td className="border border-slate-200 px-2 py-1.5 font-mono">{row.s0}</td>
                      <td className="border border-slate-200 px-2 py-1.5 font-mono">{row.d}</td>
                      <td className="border border-slate-200 px-2 py-1.5">{row.behavior}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="text-xs text-slate-600 mt-2">
              As S₀ → 0, D → ∞: on perfectly flat ground attenuation is overwhelming and the kinematic
              wave is hopeless — precisely the regime where the diffusive wave is essential.
            </p>
            <div className="bg-green-50 border border-green-200 rounded-lg p-3 text-xs text-green-900 mt-2">
              <span className="font-semibold">Theory explains, the code computes directly.</span> The
              boxed equation is <span className="italic">why</span> the diffusive wave attenuates, but OPM
              never discretises it — it computes the true S_w for each cell pair and feeds it into
              Manning&apos;s equation directly (more accurate, no linearisation error, and it can handle
              strong backwater where S_w goes negative).
            </div>
          </div>
        )}

        {/* ── Part B: live mini-simulation ── */}
        <div className="border-t border-slate-200 pt-6">
          <h4 className="font-bold text-slate-800 text-sm mb-1">
            Watch a Pulse Spread — Solving ∂h′/∂t + c·∂h′/∂x = D·∂²h′/∂x²
          </h4>
          <p className="text-xs text-slate-600 mb-3">
            Upwind scheme for advection, central difference for diffusion. As D increases, the pulse
            spreads and flattens more over the same elapsed time; at D ≈ 0 it translates almost without
            spreading.
          </p>

          <svg
            width={SIM_SVG_W}
            height={SIM_SVG_H}
            className="block overflow-visible mx-auto"
            style={{ maxWidth: '100%' }}
          >
            {/* Grid lines */}
            {gridYs.map((v) => (
              <line
                key={`gy-${v}`}
                x1={SIM_PAD.l}
                y1={simYScale(v)}
                x2={SIM_SVG_W - SIM_PAD.r}
                y2={simYScale(v)}
                stroke="#e2e8f0"
                strokeWidth={1}
              />
            ))}
            {gridXs.map((i) => (
              <line
                key={`gx-${i}`}
                x1={simXScale(i)}
                y1={SIM_PAD.t}
                x2={simXScale(i)}
                y2={SIM_SVG_H - SIM_PAD.b}
                stroke="#e2e8f0"
                strokeWidth={1}
              />
            ))}

            {/* Axes */}
            <line
              x1={SIM_PAD.l}
              y1={SIM_PAD.t}
              x2={SIM_PAD.l}
              y2={SIM_SVG_H - SIM_PAD.b}
              stroke="#94a3b8"
              strokeWidth={1.5}
            />
            <line
              x1={SIM_PAD.l}
              y1={SIM_SVG_H - SIM_PAD.b}
              x2={SIM_SVG_W - SIM_PAD.r}
              y2={SIM_SVG_H - SIM_PAD.b}
              stroke="#94a3b8"
              strokeWidth={1.5}
            />

            {/* Y labels */}
            {[0, 0.5, 1.0].map((v) => (
              <text key={`yl-${v}`} x={SIM_PAD.l - 4} y={simYScale(v) + 3} textAnchor="end" fontSize={9} fill="#64748b">
                {v.toFixed(1)}
              </text>
            ))}

            {/* X label */}
            <text x={(SIM_PAD.l + SIM_SVG_W - SIM_PAD.r) / 2} y={SIM_SVG_H - 4} fontSize={9} fill="#94a3b8" textAnchor="middle">
              position x (cell index)
            </text>
            <text
              x={SIM_PAD.l - 26}
              y={SIM_SVG_H / 2}
              fontSize={9}
              fill="#94a3b8"
              transform={`rotate(-90, ${SIM_PAD.l - 26}, ${SIM_SVG_H / 2})`}
              textAnchor="middle"
            >
              h&apos;
            </text>

            {/* Initial pulse, faint */}
            <path d={makeSimPath(initial)} fill="none" stroke="#94a3b8" strokeWidth={1.5} strokeDasharray="4,3" opacity={0.7} />

            {/* Current pulse */}
            <path d={makeSimPath(history)} fill="none" stroke="#7c3aed" strokeWidth={2.5} />

            {/* Legend */}
            <rect x={SIM_SVG_W - SIM_PAD.r - 110} y={SIM_PAD.t} width={104} height={36} rx={4} fill="white" stroke="#e2e8f0" strokeWidth={1} />
            <line x1={SIM_SVG_W - SIM_PAD.r - 104} y1={SIM_PAD.t + 10} x2={SIM_SVG_W - SIM_PAD.r - 88} y2={SIM_PAD.t + 10} stroke="#94a3b8" strokeWidth={1.5} strokeDasharray="4,3" />
            <text x={SIM_SVG_W - SIM_PAD.r - 84} y={SIM_PAD.t + 13} fontSize={8} fill="#64748b">t = 0 (initial)</text>
            <line x1={SIM_SVG_W - SIM_PAD.r - 104} y1={SIM_PAD.t + 24} x2={SIM_SVG_W - SIM_PAD.r - 88} y2={SIM_PAD.t + 24} stroke="#7c3aed" strokeWidth={2.5} />
            <text x={SIM_SVG_W - SIM_PAD.r - 84} y={SIM_PAD.t + 27} fontSize={8} fill="#7c3aed">current</text>
          </svg>

          <div className="flex flex-col sm:flex-row gap-4 mt-4">
            {/* Controls */}
            <div className="flex items-center gap-2 flex-wrap">
              <button
                onClick={playing ? handlePause : handlePlay}
                className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-indigo-600 text-white hover:bg-indigo-700 transition-colors"
              >
                {playing ? 'Pause' : frame >= MAX_FRAMES ? 'Restart' : 'Play'}
              </button>
              <button
                onClick={handleReset}
                className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-slate-100 text-slate-700 hover:bg-slate-200 transition-colors"
              >
                Reset
              </button>
              <span className="text-xs text-slate-500 font-mono">
                t = {(frame * SUBSTEPS_PER_FRAME * DT_SIM).toFixed(0)}
              </span>
            </div>

            {/* D slider */}
            <div className="flex flex-col gap-1 flex-1 min-w-[180px]">
              <label className="text-xs font-semibold text-slate-600">D = {D.toFixed(3)} (diffusivity)</label>
              <input
                type="range"
                min={D_MIN}
                max={D_MAX}
                step={0.005}
                value={D}
                onChange={(e) => setD(+e.target.value)}
                className="w-full"
              />
              <div className="text-xs text-slate-400 flex justify-between">
                <span>D ≈ 0 (pure translation)</span>
                <span>D large (strong spreading)</span>
              </div>
            </div>
          </div>

          <div className="bg-slate-50 border border-slate-200 rounded-lg p-3 text-xs text-slate-700 mt-3">
            Peak height: <span className="font-mono">{peakNow.toFixed(3)}</span> (started at{' '}
            <span className="font-mono">{peakInitial.toFixed(3)}</span>) — attenuation so far:{' '}
            <span className="font-mono font-semibold">{Math.max(attenuationPct, 0).toFixed(1)}%</span>.
            Larger D flattens the peak faster — this is the same peak-flattening effect from §5.1, now
            traced to a single term in the PDE.
          </div>
        </div>
      </div>
    </div>
  );
}
