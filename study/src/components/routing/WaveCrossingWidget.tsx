'use client';

import React, { useState, useMemo, useEffect } from 'react';

// ─────────────────────────────────────────────────────────────────────────
// WaveCrossingWidget — why C>1 erupts into oscillations.
//
// PHYSICAL: in one tick Δt the wave travels c·Δt = C cells.  The upwind
// stencil can only follow ONE cell per tick.  So:
//   C<1 → the wave stays inside the cell (creeps); the scheme interpolates
//         → smooth, slightly smeared, bounded.
//   C=1 → the wave lands exactly on the next node; exact.
//   C>1 → the wave LEAPS past node i−1 into a cell the stencil cannot see;
//         forced to extrapolate, the value overshoots to the wrong sign and
//         is multiplied up every step → a growing, sign-flipping sawtooth.
//
// NUMERICAL PROOF (no Fourier): feed the update the jaggedest pattern a grid
// can hold — a sawtooth where Q₍ᵢ₋₁₎ = −Qᵢ.  Then
//   Qᵢⁿ⁺¹ = C·Q₍ᵢ₋₁₎ + (1−C)·Qᵢ = C(−Qᵢ) + (1−C)Qᵢ = (1−2C)·Qᵢ
// so every step the whole zigzag is multiplied by g = 1−2C.  g<0 ⇒ sign flip
// each step (oscillation); |g|>1 ⇒ it grows (blow-up).  Both happen at C>1.
// ─────────────────────────────────────────────────────────────────────────

const N = 9;            // grid cells shown
const SPIKE_AT = 3;     // initial spike location
const MAXSTEP = 9;      // steps precomputed
const C_FIXED = 1.45;   // celerity [m/s]  (Ch.4 default)
const DX_FIXED = 500;   // cell size [m]   (Ch.4 default)
const STEP_MS = 850;

function buildRows(C: number): number[][] {
  const rows: number[][] = [];
  let Q = Array<number>(N).fill(0);
  Q[SPIKE_AT] = 1;
  rows.push([...Q]);
  for (let s = 0; s < MAXSTEP; s++) {
    const Qn = [...Q];
    Qn[0] = 0; // baseflow inflow boundary
    for (let i = 1; i < N; i++) Qn[i] = C * Q[i - 1] + (1 - C) * Q[i];
    Q = Qn;
    rows.push([...Q]);
  }
  return rows;
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

const W = 520, M = 32;
const cellX = (c: number) => M + (c / (N - 1)) * (W - 2 * M);
function courantColor(C: number): string {
  if (C < 0.999) return '#2563eb';
  if (C <= 1.001) return '#16a34a';
  return '#dc2626';
}
function barColor(v: number): string {
  if (!isFinite(v)) return '#7f1d1d';
  if (v < -0.001) return '#dc2626';        // negative → unphysical
  if (v > 1.02) return '#b91c1c';          // overshoot above the source spike
  const t = Math.max(0, Math.min(1, v));
  return `rgb(${Math.round(186 - t * 180)},${Math.round(230 - t * 100)},${Math.round(253 - t * 50)})`;
}

export default function WaveCrossingWidget() {
  const [dt, setDt] = useState(300);
  const [step, setStep] = useState(1);
  const [playing, setPlaying] = useState(false);

  const C = (C_FIXED * dt) / DX_FIXED;
  const rows = useMemo(() => buildRows(C), [C]);
  const maxAbs = (r: number[]) => r.reduce((m, v) => Math.max(m, isFinite(v) ? Math.abs(v) : Infinity), 0);
  const curMax = maxAbs(rows[Math.min(step, MAXSTEP)]);
  const prevMax = maxAbs(rows[Math.max(0, Math.min(step, MAXSTEP) - 1)]);
  const unstable = !isFinite(curMax) || curMax > 1e3;
  const g = 1 - 2 * C; // sawtooth-mode multiplier

  // animation
  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => {
      setStep((s) => {
        if (s >= MAXSTEP) { setPlaying(false); return s; }
        return s + 1;
      });
    }, STEP_MS);
    return () => clearInterval(id);
  }, [playing]);

  const reset = () => { setStep(1); setPlaying(false); };
  const setPreset = (d: number) => { setDt(d); reset(); };

  // wave-front geometry (Panel A)
  const frontCell = SPIKE_AT + C * step;
  const prevCell = SPIKE_AT + C * Math.max(0, step - 1);
  const frontClamped = Math.min(frontCell, N - 1);
  const offGrid = frontCell > N - 1 + 1e-6;

  // Panel B geometry
  const PBH = 150, baseY = 92, halfH = 64;
  const barW = 30;
  const valY = (v: number) => baseY - (Math.max(-2, Math.min(2, v)) / 2) * halfH;

  return (
    <div className="not-prose my-6 rounded-2xl border border-slate-200 bg-white shadow-sm overflow-hidden">
      <div className="bg-gradient-to-r from-rose-600 to-orange-600 px-5 py-3">
        <h3 className="text-white font-bold text-base">When the wave outruns the grid</h3>
        <p className="text-rose-50 text-xs mt-0.5">
          One tick, one cell — that's all the grid can follow. Watch what happens when the wave tries to cross
          more.
        </p>
      </div>

      {/* Controls */}
      <div className="px-5 pt-4 grid md:grid-cols-[1fr_auto] gap-4 items-end">
        <div className="space-y-2">
          <div className="flex flex-wrap gap-2">
            {[
              { lab: 'Tiny Δt — wave creeps', d: 90 },
              { lab: 'Sweet spot  C = 1', d: 345 },
              { lab: 'Too big — C > 1', d: 600 },
            ].map((p) => (
              <button key={p.d} onClick={() => setPreset(p.d)}
                className="rounded-full border border-slate-300 text-slate-600 text-xs font-semibold px-3 py-1 hover:bg-slate-50">
                {p.lab}
              </button>
            ))}
          </div>
          <LabeledSlider label="time step  Δt  (c = 1.45 m/s, Δx = 500 m fixed)" value={dt} min={60} max={780} step={10}
            display={`${dt} s`} onChange={(v) => { setDt(v); }} />
        </div>
        <div className="flex items-center gap-3">
          <div className="text-center">
            <div className="text-[10px] uppercase font-bold text-slate-400">Courant</div>
            <div className="text-2xl font-bold tabular-nums" style={{ color: courantColor(C) }}>C = {C.toFixed(2)}</div>
          </div>
          <div className="flex gap-1.5">
            <button onClick={() => setPlaying((p) => !p)}
              className="rounded-lg bg-rose-600 text-white text-sm font-semibold px-3 py-1.5 hover:bg-rose-700">
              {playing ? '❚❚' : '▶'}
            </button>
            <button onClick={() => { setPlaying(false); setStep((s) => Math.min(MAXSTEP, s + 1)); }}
              className="rounded-lg border border-slate-300 text-slate-600 text-sm font-semibold px-3 py-1.5 hover:bg-slate-50">
              Step ▸
            </button>
            <button onClick={reset}
              className="rounded-lg border border-slate-300 text-slate-600 text-sm font-semibold px-3 py-1.5 hover:bg-slate-50">
              ↺
            </button>
          </div>
        </div>
      </div>

      {/* Panel A — stride vs grid reach */}
      <div className="px-5 pt-4">
        <div className="text-xs font-bold text-slate-500 uppercase mb-1">
          ① the real wave · stride = C cells per tick · grid can follow only 1
        </div>
        <svg viewBox={`0 0 ${W} 116`} className="w-full select-none">
          {/* cells */}
          {Array.from({ length: N }, (_, c) => (
            <g key={c}>
              <line x1={cellX(c)} y1={20} x2={cellX(c)} y2={86} stroke="#e2e8f0" strokeWidth={1} />
              <circle cx={cellX(c)} cy={86} r={3} fill="#cbd5e1" />
            </g>
          ))}
          <line x1={cellX(0)} y1={86} x2={cellX(N - 1)} y2={86} stroke="#cbd5e1" strokeWidth={1} />
          <text x={cellX(SPIKE_AT)} y={104} textAnchor="middle" fontSize={9} className="fill-slate-400">start</text>

          {/* footprints of previous ticks */}
          {Array.from({ length: step + 1 }, (_, k) => {
            const fc = Math.min(SPIKE_AT + C * k, N - 1);
            return <line key={k} x1={cellX(fc)} y1={70} x2={cellX(fc)} y2={86} stroke="#fb923c" strokeWidth={1} opacity={0.4} />;
          })}

          {/* last hop decomposition: green first cell (grid reach) + red overrun */}
          {step >= 1 && (
            <>
              <line x1={cellX(prevCell)} y1={58} x2={cellX(Math.min(prevCell + 1, frontClamped))} y2={58}
                stroke="#16a34a" strokeWidth={5} strokeLinecap="round" />
              {C > 1.001 && (
                <line x1={cellX(Math.min(prevCell + 1, N - 1))} y1={58} x2={cellX(frontClamped)} y2={58}
                  stroke="#dc2626" strokeWidth={5} strokeLinecap="round" />
              )}
              <text x={cellX(prevCell + C / 2 < N - 1 ? prevCell + C / 2 : N - 1.2)} y={50} textAnchor="middle"
                fontSize={9} style={{ fill: courantColor(C) }}>
                this tick: {C.toFixed(2)} cell{C === 1 ? '' : 's'}
              </text>
            </>
          )}

          {/* wave front token */}
          <g style={{ transition: 'transform 0.6s ease' }} transform={`translate(${cellX(frontClamped)},0)`}>
            <path d="M0,86 L-7,72 L7,72 Z" fill={courantColor(C)} />
            <circle cx={0} cy={68} r={5} fill={courantColor(C)} />
          </g>
          {offGrid && (
            <text x={W - 40} y={40} textAnchor="end" fontSize={11} className="fill-red-600 font-bold">⚡ ran off the grid →</text>
          )}
        </svg>
        <div className="text-[12px] mt-0.5" style={{ color: courantColor(C) }}>
          {C < 0.999 && '↳ stride < 1 cell: the wave is still inside the cell — the scheme interpolates between the two nodes it can see. Bounded, but a little smeared each tick.'}
          {Math.abs(C - 1) <= 0.02 && '↳ stride = 1 cell exactly: the wave lands on the next node every tick. No interpolation, no smear — exact.'}
          {C > 1.02 && `↳ stride > 1 cell: the wave leapt PAST node i−1 by ${(C - 1).toFixed(2)} cell into a cell the stencil can't see. It must extrapolate → overshoot.`}
        </div>
      </div>

      {/* Panel B — what the numbers do */}
      <div className="px-5 pt-4">
        <div className="text-xs font-bold text-slate-500 uppercase mb-1">
          ② the numerical solution · one spike, marched {step} step{step === 1 ? '' : 's'}
        </div>
        <svg viewBox={`0 0 ${W} ${PBH}`} className="w-full select-none">
          {/* zero baseline */}
          <line x1={M - 6} y1={baseY} x2={W - M + 6} y2={baseY} stroke="#94a3b8" strokeWidth={1} />
          <text x={M - 8} y={baseY + 3} textAnchor="end" fontSize={8} className="fill-slate-400">0</text>
          {!unstable && rows[Math.min(step, MAXSTEP)].map((v, i) => {
            const top = valY(v);
            const h = Math.abs(top - baseY);
            const overflow = Math.abs(v) > 2;
            return (
              <g key={i}>
                <rect x={cellX(i) - barW / 2} y={Math.min(top, baseY)} width={barW} height={Math.max(1, h)}
                  rx={2} fill={barColor(v)} />
                <text x={cellX(i)} y={v >= 0 ? Math.min(top, baseY) - 3 : Math.max(top, baseY) + 11}
                  textAnchor="middle" fontSize={9} className="fill-slate-600 font-mono tabular-nums">
                  {overflow ? (v > 0 ? '▲' : '▼') : ''}{v.toFixed(2)}
                </text>
              </g>
            );
          })}
          {unstable && (
            <g>
              <rect x={M + 40} y={baseY - 30} width={W - 2 * M - 80} height={60} rx={8} fill="#fef2f2" stroke="#dc2626" strokeWidth={1.5} />
              <text x={W / 2} y={baseY - 6} textAnchor="middle" className="fill-red-600 font-bold" fontSize={15}>⚠ Model Unstable</text>
              <text x={W / 2} y={baseY + 14} textAnchor="middle" className="fill-red-500" fontSize={11}>
                the sawtooth has multiplied past 1000× — values have blown up
              </text>
            </g>
          )}
        </svg>
        <div className="flex items-center justify-between text-[12px] mt-0.5">
          <span className="text-slate-500">
            peak magnitude this step:{' '}
            <span className="font-mono font-bold" style={{ color: unstable ? '#dc2626' : C > 1.001 ? '#dc2626' : '#0f172a' }}>
              {unstable ? '∞' : curMax.toFixed(2)}
            </span>
            {step > 0 && isFinite(curMax) && isFinite(prevMax) && prevMax > 0 && (
              <span className="text-slate-400"> ({curMax >= prevMax ? '↑' : '↓'} ×{(curMax / prevMax).toFixed(2)} vs last step)</span>
            )}
          </span>
          <span className="text-slate-400">blue = positive · <span className="text-red-600">red = negative / overshoot</span></span>
        </div>
      </div>

      {/* Math strip */}
      <div className="px-5 py-4">
        <div className="rounded-xl border border-slate-200 bg-slate-50 p-3 space-y-2 text-[13px]">
          <div className="font-mono text-slate-800">
            Qᵢⁿ⁺¹ = <span style={{ color: courantColor(C) }}>{C.toFixed(2)}</span>·Q₍ᵢ₋₁₎ +{' '}
            <span className={1 - C < 0 ? 'text-red-600 font-bold' : 'text-emerald-700'}>({(1 - C).toFixed(2)})</span>·Qᵢ
            {1 - C < 0 && <span className="text-red-600"> ← negative weight = extrapolation</span>}
          </div>
          <div className="border-t border-slate-200 pt-2">
            <span className="text-slate-600">Jaggedest pattern (sawtooth, Q₍ᵢ₋₁₎ = −Qᵢ) is multiplied each step by</span>
            <div className="mt-1 flex flex-wrap items-center gap-2">
              <span className="font-mono font-bold text-slate-800">g = 1 − 2C = {g.toFixed(2)}</span>
              <span className={`rounded-full px-2.5 py-0.5 text-xs font-semibold ${
                Math.abs(g) < 0.999 ? 'bg-blue-100 text-blue-800'
                  : Math.abs(g) <= 1.001 ? 'bg-emerald-100 text-emerald-800'
                  : 'bg-red-100 text-red-700'}`}>
                {Math.abs(g) < 0.999 ? 'shrinks → wiggles die (diffusion)'
                  : Math.abs(g) <= 1.001 ? 'size held, flips sign → neutral knife-edge'
                  : 'flips sign AND grows → oscillates & explodes'}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
