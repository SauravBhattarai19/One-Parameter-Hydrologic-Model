'use client';

import React, { useState, useMemo } from 'react';

// ─────────────────────────────────────────────────────────────────────────
// ModifiedEquationWidget — what the computer REALLY solves (mental model 2).
//
// Taylor-expand the upwind scheme and you find it does not solve pure
// advection.  It solves advection + a diffusion term it never asked for:
//
//     ∂Q/∂t + c·∂Q/∂x = α·∂²Q/∂x²,   α = (cΔx/2)(1 − C)
//
// α is the numerical (artificial) diffusion coefficient.  sign(α)=sign(1−C):
//   C<1 → α>0 (real diffusion, smears, stable)
//   C=1 → α=0 (pure advection recovered, exact)
//   C>1 → α<0 (anti-diffusion, ill-posed, explodes)
// ─────────────────────────────────────────────────────────────────────────

function Badge({ n }: { n: number }) {
  const circled = ['①', '②', '③', '④', '⑤', '⑥'][n - 1];
  return (
    <span className="inline-flex items-center justify-center w-7 h-7 rounded-full bg-indigo-600 text-white font-bold text-sm mr-2 shrink-0">
      {circled}
    </span>
  );
}

interface Step { title: string; body: string; }
const STEPS: Step[] = [
  {
    title: 'Start from the upwind scheme',
    body:
`Forward in time, backward (upwind) in space:

  (Qᵢⁿ⁺¹ − Qᵢⁿ)/Δt  +  c·(Qᵢⁿ − Qᵢ₋₁ⁿ)/Δx  =  0`,
  },
  {
    title: 'Taylor-expand the two off-node values',
    body:
`Expand about the point (xᵢ, tₙ):

  Qᵢⁿ⁺¹ = Q + Δt·Qₜ + (Δt²/2)·Qₜₜ + …
  Qᵢ₋₁ⁿ = Q − Δx·Qₓ + (Δx²/2)·Qₓₓ − …`,
  },
  {
    title: 'Substitute and collect',
    body:
`Time slope:  (Qᵢⁿ⁺¹ − Qᵢⁿ)/Δt = Qₜ + (Δt/2)·Qₜₜ + …
Space slope: c·(Qᵢⁿ − Qᵢ₋₁ⁿ)/Δx = c·Qₓ − (cΔx/2)·Qₓₓ + …

Add them (=0):
  Qₜ + cQₓ + (Δt/2)·Qₜₜ − (cΔx/2)·Qₓₓ = 0`,
  },
  {
    title: 'Move the error terms to the right',
    body:
`  Qₜ + c·Qₓ  =  (cΔx/2)·Qₓₓ  −  (Δt/2)·Qₜₜ

The left side is the equation we WANTED.
The right side is the error the grid added.`,
  },
  {
    title: 'Trade the time error for a space error',
    body:
`To leading order the PDE says Qₜ = −c·Qₓ, so
  Qₜₜ = ∂/∂t(−c Qₓ) = −c·∂/∂x(Qₜ) = c²·Qₓₓ

Plug Qₜₜ = c²·Qₓₓ into the right side:
  (cΔx/2)Qₓₓ − (Δt/2)c²Qₓₓ = (cΔx/2)(1 − cΔt/Δx)Qₓₓ`,
  },
  {
    title: 'The modified equation',
    body:
`  ∂Q/∂t + c·∂Q/∂x  =  α·∂²Q/∂x²

        α = (cΔx/2)(1 − C),    C = cΔt/Δx

A diffusion term we never wrote down. α is the
NUMERICAL diffusion coefficient — and its sign is
controlled entirely by the Courant number C.`,
  },
];

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

// analytic Gaussian under diffusion: variance grows by k·(1−C); amplitude area-preserving
const PW = 240, PH = 120, KDIFF = 2.6;
function pulseGaussian(C: number): { d: string; unstable: boolean } {
  const D = 1 - C;                 // ∝ α
  const sig2 = 1 + KDIFF * D;      // evolved variance (σ₀²=1, T folded into KDIFF)
  if (sig2 <= 0.04) return { d: '', unstable: true };
  const sig = Math.sqrt(sig2);
  const amp = 1 / sig;             // area-preserving height
  const pts: string[] = [];
  for (let k = 0; k <= 60; k++) {
    const x = -4 + (8 * k) / 60;
    const q = amp * Math.exp(-(x * x) / (2 * sig2));
    const sx = (PW / 2) + (x / 4) * (PW / 2 - 8);
    const sy = PH - 10 - Math.min(q, 1.8) * (PH - 24) / 1.8;
    pts.push(`${k === 0 ? 'M' : 'L'}${sx.toFixed(1)},${sy.toFixed(1)}`);
  }
  return { d: pts.join(' '), unstable: false };
}

export default function ModifiedEquationWidget() {
  const [shown, setShown] = useState(1);
  const [C, setC] = useState(0.6);
  const [showWhy, setShowWhy] = useState(false);

  const alphaSign = 1 - C;                       // ∝ α
  const regime = C < 0.999 ? 'diffuse' : C > 1.001 ? 'anti' : 'exact';
  const regimeColor = regime === 'diffuse' ? '#2563eb' : regime === 'exact' ? '#16a34a' : '#dc2626';
  const pulse = useMemo(() => pulseGaussian(C), [C]);

  // α number-line: map n=(1−C) ∈ [−0.5, +1] to x ∈ [0,1]
  const NL_W = 300, NL_H = 56;
  const nlx = (n: number) => 8 + ((n + 0.5) / 1.5) * (NL_W - 16);
  const markerN = Math.max(-0.5, Math.min(1, alphaSign));

  return (
    <div className="not-prose my-6 rounded-2xl border border-slate-200 bg-white shadow-sm overflow-hidden">
      <div className="bg-gradient-to-r from-indigo-600 to-violet-600 px-5 py-3">
        <h3 className="text-white font-bold text-base">The equation the computer actually solves</h3>
        <p className="text-indigo-100 text-xs mt-0.5">
          Six lines of Taylor algebra reveal a hidden diffusion term — and the Courant number sets its sign.
        </p>
      </div>

      <div className="grid lg:grid-cols-[1fr_320px] gap-5 p-5">
        {/* Derivation */}
        <div>
          <div className="space-y-2">
            {STEPS.slice(0, shown).map((s, i) => (
              <div key={i} className="rounded-xl border border-slate-200 bg-slate-50 p-3">
                <div className="flex items-center mb-1.5">
                  <Badge n={i + 1} />
                  <h4 className="text-sm font-bold text-slate-800">{s.title}</h4>
                </div>
                <pre className="text-[12.5px] font-mono text-slate-700 whitespace-pre-wrap leading-relaxed pl-9">
{s.body}
                </pre>
              </div>
            ))}
          </div>
          <div className="flex gap-2 mt-3">
            <button
              onClick={() => setShown((v) => Math.min(STEPS.length, v + 1))}
              disabled={shown >= STEPS.length}
              className="rounded-lg bg-indigo-600 text-white text-sm font-semibold px-4 py-1.5 disabled:opacity-40 hover:bg-indigo-700 transition-colors">
              Reveal next ▸
            </button>
            <button
              onClick={() => setShown(STEPS.length)}
              className="rounded-lg border border-slate-300 text-slate-600 text-sm font-semibold px-3 py-1.5 hover:bg-slate-50">
              Show all
            </button>
            <button
              onClick={() => setShown(1)}
              className="rounded-lg border border-slate-300 text-slate-600 text-sm font-semibold px-3 py-1.5 hover:bg-slate-50">
              Reset
            </button>
          </div>

          {/* Why is ∂²/∂x² actually diffusion? */}
          <div className="mt-3">
            <button onClick={() => setShowWhy((v) => !v)}
              className="w-full text-left rounded-lg border border-violet-200 bg-violet-50 px-3 py-2 text-sm font-semibold text-violet-800 hover:bg-violet-100 transition-colors">
              {showWhy ? '▾' : '▸'} How do we KNOW α·∂²Q/∂x² is diffusion — and not just a random term?
            </button>
            {showWhy && (
              <div className="mt-2 rounded-xl border border-slate-200 bg-white p-3 space-y-3 text-[13px] text-slate-700 leading-relaxed">
                <p>
                  We never <em>added</em> it — the Taylor expansion <strong>found</strong> it already hiding in the
                  discrete update. The only question is what a second-derivative term <em>does</em>. Three
                  independent checks all give the same verdict: <strong>diffusion</strong>.
                </p>

                {/* 1 · neighbour average */}
                <div>
                  <div className="font-semibold text-slate-800 mb-1">① It drags every point toward its neighbours' average</div>
                  <pre className="font-mono text-[11.5px] bg-slate-900 text-slate-100 rounded-lg p-2 overflow-x-auto whitespace-pre">
{`∂²Q/∂x²  ≈  (Q₍ᵢ₊₁₎ − 2Qᵢ + Q₍ᵢ₋₁₎) / Δx²
         =  (2/Δx²)·[ ½(Q₍ᵢ₋₁₎+Q₍ᵢ₊₁₎) − Qᵢ ]
                       └─ neighbours' average ─┘`}
                  </pre>
                  <p className="mt-1">
                    So <span className="font-mono">∂Q/∂t = α·∂²Q/∂x²</span> literally says “drift toward the average
                    of your two neighbours, at rate α.” That is smoothing — the very same averaging as the
                    weighted-update widget above.
                  </p>
                </div>

                {/* curvature table + svg */}
                <div className="grid sm:grid-cols-[1fr_auto] gap-3 items-center">
                  <table className="text-[12px] border-collapse">
                    <thead>
                      <tr className="text-slate-500">
                        <th className="text-left font-semibold pr-3 pb-1">where you are</th>
                        <th className="text-left font-semibold pr-3 pb-1">curvature ∂²Q/∂x²</th>
                        <th className="text-left font-semibold pb-1">so ∂Q/∂t …</th>
                      </tr>
                    </thead>
                    <tbody className="align-top">
                      <tr><td className="pr-3 py-0.5">on a <b>peak</b></td><td className="pr-3 py-0.5 text-red-600">− (concave down)</td><td className="py-0.5">&lt; 0 → peak <b>drops</b></td></tr>
                      <tr><td className="pr-3 py-0.5">in a <b>trough</b></td><td className="pr-3 py-0.5 text-blue-600">+ (concave up)</td><td className="py-0.5">&gt; 0 → dip <b>fills</b></td></tr>
                      <tr><td className="pr-3 py-0.5">a straight ramp</td><td className="pr-3 py-0.5 text-slate-500">0</td><td className="py-0.5">0 → unchanged</td></tr>
                    </tbody>
                  </table>
                  <svg viewBox="0 0 130 84" className="w-32 shrink-0">
                    <line x1={8} y1={42} x2={122} y2={42} stroke="#cbd5e1" strokeWidth={1} strokeDasharray="3 3" />
                    <path d="M8,42 Q35,8 65,42 T122,42" fill="none" stroke="#0ea5e9" strokeWidth={2} />
                    {/* down arrow at crest (~x=35) */}
                    <line x1={35} y1={16} x2={35} y2={32} stroke="#dc2626" strokeWidth={2} />
                    <path d="M31,28 L35,34 L39,28 Z" fill="#dc2626" />
                    {/* up arrow at trough (~x=93) */}
                    <line x1={93} y1={68} x2={93} y2={52} stroke="#2563eb" strokeWidth={2} />
                    <path d="M89,56 L93,50 L97,56 Z" fill="#2563eb" />
                    <text x={65} y={80} textAnchor="middle" fontSize={8} className="fill-slate-400">peaks fall, dips rise → flat</text>
                  </svg>
                </div>

                {/* 2 units · 3 energy */}
                <div className="flex flex-wrap gap-2">
                  <span className="inline-block rounded-lg bg-emerald-50 border border-emerald-200 text-emerald-800 px-2.5 py-1 text-[12px]">
                    ② Units: [α] = (m/s)·(m) = <b>m²/s</b> — the units of a diffusivity (same as heat, ink).
                  </span>
                  <span className="inline-block rounded-lg bg-emerald-50 border border-emerald-200 text-emerald-800 px-2.5 py-1 text-[12px]">
                    ③ Energy: ∫Q² can only <b>fall</b> when α&gt;0 — gradients destroyed, irreversibly.
                  </span>
                </div>

                <p className="text-[12px] text-slate-500">
                  Flip α negative (C&gt;1) and all three reverse: it <em>un-averages</em>, the “diffusivity” is
                  negative, and ∫Q² <em>grows</em> → blow-up. Same math, opposite sign.
                </p>
              </div>
            )}
          </div>
        </div>

        {/* α explorer */}
        <div className="rounded-xl border border-slate-200 p-3">
          <div className="text-xs font-bold text-slate-500 uppercase mb-2">α explorer — pick a Courant number</div>
          <LabeledSlider label="Courant number  C" value={C} min={0.2} max={1.5} step={0.05}
            display={C.toFixed(2)} onChange={setC} />

          <div className="mt-3 rounded-lg px-3 py-2 text-center text-sm font-bold"
            style={{ background: regimeColor + '18', color: regimeColor }}>
            α = (cΔx/2)({(1 - C).toFixed(2)}){' '}
            {regime === 'diffuse' ? '> 0  →  diffusion, smears (stable)'
              : regime === 'exact' ? '= 0  →  exact, no smear'
              : '< 0  →  anti-diffusion, explodes'}
          </div>

          {/* number line */}
          <svg viewBox={`0 0 ${NL_W} ${NL_H}`} className="w-full mt-3">
            <defs>
              <linearGradient id="nlgrad" x1="0" x2="1">
                <stop offset="0%" stopColor="#dc2626" />
                <stop offset="33%" stopColor="#16a34a" />
                <stop offset="100%" stopColor="#2563eb" />
              </linearGradient>
            </defs>
            <rect x={8} y={20} width={NL_W - 16} height={6} rx={3} fill="url(#nlgrad)" opacity={0.35} />
            {/* zero (C=1) tick */}
            <line x1={nlx(0)} y1={14} x2={nlx(0)} y2={32} stroke="#16a34a" strokeWidth={2} />
            <text x={nlx(0)} y={46} textAnchor="middle" fontSize={9} className="fill-emerald-700">α=0 (C=1)</text>
            <text x={nlx(0.9)} y={46} textAnchor="middle" fontSize={9} className="fill-blue-700">α&gt;0 (C&lt;1)</text>
            <text x={nlx(-0.35)} y={46} textAnchor="middle" fontSize={9} className="fill-red-700">α&lt;0 (C&gt;1)</text>
            {/* marker */}
            <circle cx={nlx(markerN)} cy={23} r={6} fill={regimeColor} stroke="#fff" strokeWidth={2} />
          </svg>

          {/* pulse strip */}
          <div className="text-xs font-bold text-slate-500 uppercase mt-2 mb-1">a pulse after diffusing</div>
          <svg viewBox={`0 0 ${PW} ${PH}`} className="w-full rounded-lg bg-slate-50">
            {/* initial faint reference */}
            <path d={pulseGaussian(1).d} fill="none" stroke="#cbd5e1" strokeWidth={1.5} strokeDasharray="3 3" />
            {!pulse.unstable
              ? <path d={pulse.d} fill="none" stroke={regimeColor} strokeWidth={2.5} />
              : (
                <g>
                  <text x={PW / 2} y={PH / 2 - 4} textAnchor="middle" className="fill-red-600 font-bold" fontSize={14}>⚠ Model Unstable</text>
                  <text x={PW / 2} y={PH / 2 + 14} textAnchor="middle" className="fill-red-500" fontSize={10}>negative α self-sharpens without bound</text>
                </g>
              )}
          </svg>
          <p className="text-[11px] text-slate-500 mt-1 leading-snug">
            Dashed grey = the original pulse (C=1, untouched). Solid = after the numerical diffusion α acts:
            spreads when C&lt;1, blows up when C&gt;1.
          </p>
        </div>
      </div>
    </div>
  );
}
