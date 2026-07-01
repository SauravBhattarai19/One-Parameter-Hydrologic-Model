'use client';

import React, { useState, useMemo, useEffect } from 'react';

// ---------------------------------------------------------------------------
// Physics helpers
// ---------------------------------------------------------------------------

function computeAlpha(n: number, S0: number, B: number): number {
  return Math.sqrt(S0) / (n * Math.pow(B, 2 / 3));
}

const BETA = 5 / 3;

function aToQ(A: number, alpha: number): number {
  return alpha * Math.pow(Math.max(A, 0.001), BETA);
}

function qColor(Q: number, Qmax: number): string {
  const t = Math.max(0, Math.min(1, Q / Qmax));
  return `rgb(${Math.round(219 - t * 190)},${Math.round(234 - t * 175)},${Math.round(255 - t * 65)})`;
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface SliderProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  display?: string;
  onChange: (v: number) => void;
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function LabeledSlider({ label, value, min, max, step, display, onChange }: SliderProps) {
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
        className="w-full h-1.5 rounded-full accent-sky-600 cursor-pointer"
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export default function ManningCelerityWidget() {
  const [manN, setManN] = useState(0.04);
  const [S0, setS0] = useState(0.001);
  const [B, setB] = useState(10);
  const [dt, setDt] = useState(300);
  const [dx, setDx] = useState(500);
  const [h, setH] = useState(1.15);
  const [playing, setPlaying] = useState(false);
  const [animX, setAnimX] = useState(0);
  const [visibleSteps, setVisibleSteps] = useState(1);

  // Derived scalars
  const alpha = useMemo(() => computeAlpha(manN, S0, B), [manN, S0, B]);
  const A_sel = B * h;
  const Q_sel = aToQ(A_sel, alpha);
  const u_sel = Q_sel / A_sel;
  const c_sel = BETA * u_sel;
  const C_sel = c_sel * dt / dx;

  // Q_max from h=3.0
  const Q_max = useMemo(() => aToQ(B * 3.0, alpha), [alpha, B]);

  // Animation loop
  useEffect(() => {
    if (!playing) return;
    const interval = setInterval(() => {
      setAnimX((x) => {
        if (x >= 1) {
          setPlaying(false);
          return 0;
        }
        return x + 0.005;
      });
    }, 16);
    return () => clearInterval(interval);
  }, [playing]);

  // Preset depths + selected depth for table
  const presetDepths = [0.5, 0.8, 1.0, 1.5, 2.0, 2.5];
  const allDepths: number[] = [...presetDepths];
  const hRounded = Math.round(h * 1000) / 1000;
  const hAlreadyIn = presetDepths.some((d) => Math.abs(d - hRounded) < 0.001);
  if (!hAlreadyIn) {
    const insertIdx = allDepths.findIndex((d) => d > hRounded);
    if (insertIdx === -1) allDepths.push(hRounded);
    else allDepths.splice(insertIdx, 0, hRounded);
  }

  // Rating curve SVG data
  const SVG_W = 240;
  const SVG_H = 200;
  const PAD = { left: 42, right: 12, top: 12, bottom: 32 };
  const plotW = SVG_W - PAD.left - PAD.right;
  const plotH = SVG_H - PAD.top - PAD.bottom;

  function qToX(q: number): number {
    return PAD.left + (q / Q_max) * plotW;
  }
  function hToY(hv: number): number {
    return PAD.top + plotH - (hv / 3.0) * plotH;
  }

  const curvePoints: string = Array.from({ length: 41 }, (_, i) => {
    const hv = (i / 40) * 3.0;
    const q = aToQ(B * hv, alpha);
    return `${qToX(q)},${hToY(hv)}`;
  }).join(' ');

  // Tangent line at selected point
  // dQ/dh = c_sel * B, but in SVG we need screen coords
  // slope in data space: Δq / Δh = c_sel * B
  // tangent ±0.3 m in h
  const dh = 0.3;
  const dQ_tangent = c_sel * B * dh;
  const tx1 = qToX(Math.max(0, Q_sel - dQ_tangent));
  const ty1 = hToY(h + dh);
  const tx2 = qToX(Math.min(Q_max, Q_sel + dQ_tangent));
  const ty2 = hToY(h - dh);

  // Gridlines
  const hGrids = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
  const qGridFracs = [0.25, 0.5, 0.75, 1.0];

  // Wave animation
  const waveXpx = Math.min(animX * 280 + 10, 290);
  const particleXpx = (k: number) => Math.min((animX / (5 / 3)) * 280 + k * 40 + 10, 290);
  const waveDistM = (animX * 280) / 280 * 3.0;
  const particleDistM = (animX / (5 / 3)) * 3.0;
  const ratio = animX > 0.01 ? (waveDistM / Math.max(particleDistM, 0.001)).toFixed(2) : '1.67';

  return (
    <div className="not-prose my-8 rounded-2xl border border-slate-200 bg-white shadow-xl overflow-hidden font-sans">
      {/* Header */}
      <div className="bg-gradient-to-r from-sky-700 to-blue-900 px-6 py-4">
        <h3 className="text-white font-bold text-lg tracking-tight">
          Manning&apos;s Rating Curve &amp; Wave Celerity
        </h3>
        <p className="text-sky-200 text-sm mt-0.5">
          c = dQ/dA = (5/3)u — the wave always outruns the water
        </p>
      </div>

      <div className="p-6 flex flex-col lg:flex-row gap-6 items-start">
        {/* ---------------------------------------------------------------- */}
        {/* LEFT PANEL                                                        */}
        {/* ---------------------------------------------------------------- */}
        <div className="flex flex-col gap-5 min-w-0 w-full lg:w-80">
          {/* Sliders */}
          <div className="flex flex-col gap-3">
            <LabeledSlider
              label="Manning n"
              value={manN}
              min={0.01}
              max={0.15}
              step={0.005}
              display={manN.toFixed(3)}
              onChange={setManN}
            />
            <LabeledSlider
              label="Slope S₀"
              value={S0}
              min={0.0001}
              max={0.01}
              step={0.0001}
              display={S0.toFixed(4)}
              onChange={setS0}
            />
            <LabeledSlider
              label="Width B (m)"
              value={B}
              min={2}
              max={20}
              step={1}
              display={`${B} m`}
              onChange={setB}
            />
            <LabeledSlider
              label="Δt (s)"
              value={dt}
              min={100}
              max={600}
              step={50}
              display={`${dt} s`}
              onChange={setDt}
            />
            <LabeledSlider
              label="Δx (m)"
              value={dx}
              min={100}
              max={1000}
              step={100}
              display={`${dx} m`}
              onChange={setDx}
            />
            {/* Depth slider */}
            <LabeledSlider
              label="Depth h (m)"
              value={h}
              min={0.1}
              max={3.0}
              step={0.05}
              display={`${h.toFixed(2)} m`}
              onChange={setH}
            />
          </div>

          {/* Derived parameter display boxes */}
          <div className="flex gap-2">
            <div className="bg-sky-50 border border-sky-200 rounded px-3 py-1 text-sm font-mono flex-1 text-center">
              α = {alpha.toFixed(4)}{' '}
              <span className="text-xs text-slate-500">m^(1/3)/s</span>
            </div>
            <div className="bg-sky-50 border border-sky-200 rounded px-3 py-1 text-sm font-mono flex-1 text-center">
              β = 1.667{' '}
              <span className="text-xs text-slate-500">(fixed)</span>
            </div>
          </div>

          {/* Numerical table */}
          <div className="overflow-x-auto">
            <table className="w-full text-xs font-mono border-collapse">
              <thead>
                <tr className="bg-slate-100 text-slate-600">
                  <th className="px-1.5 py-1 text-right border border-slate-200">h</th>
                  <th className="px-1.5 py-1 text-right border border-slate-200">A</th>
                  <th className="px-1.5 py-1 text-right border border-slate-200">Q</th>
                  <th className="px-1.5 py-1 text-right border border-slate-200">u</th>
                  <th className="px-1.5 py-1 text-right border border-slate-200">c</th>
                  <th className="px-1.5 py-1 text-right border border-slate-200">Cr</th>
                  <th className="px-1.5 py-1 text-center border border-slate-200">Status</th>
                </tr>
              </thead>
              <tbody>
                {allDepths.map((hv, idx) => {
                  const Av = B * hv;
                  const Qv = aToQ(Av, alpha);
                  const uv = Qv / Av;
                  const cv = BETA * uv;
                  const Cv = cv * dt / dx;
                  const isSelected = Math.abs(hv - hRounded) < 0.001;
                  const isEven = idx % 2 === 0;
                  const rowBg = isSelected
                    ? 'bg-sky-100'
                    : isEven
                    ? 'bg-white'
                    : 'bg-slate-50';
                  const rowFont = isSelected ? 'font-semibold' : '';
                  return (
                    <tr key={hv} className={`${rowBg} ${rowFont}`}>
                      <td className="px-1.5 py-0.5 text-right border border-slate-200">
                        {hv.toFixed(2)}
                      </td>
                      <td className="px-1.5 py-0.5 text-right border border-slate-200">
                        {Av.toFixed(1)}
                      </td>
                      <td className="px-1.5 py-0.5 text-right border border-slate-200">
                        {Qv.toFixed(2)}
                      </td>
                      <td className="px-1.5 py-0.5 text-right border border-slate-200">
                        {uv.toFixed(3)}
                      </td>
                      <td className="px-1.5 py-0.5 text-right border border-slate-200">
                        {cv.toFixed(3)}
                      </td>
                      <td className="px-1.5 py-0.5 text-right border border-slate-200">
                        {Cv.toFixed(3)}
                      </td>
                      <td className="px-1.5 py-0.5 text-center border border-slate-200">
                        {Cv <= 1 ? (
                          <span className="text-green-600">&#10003; Stable</span>
                        ) : (
                          <span className="text-red-600">&#10007; Unstable</span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            <p className="text-xs text-slate-400 mt-1">
              h (m) | A (m²) | Q (m³/s) | u (m/s) | c=5u/3 | Cr=cΔt/Δx
            </p>
          </div>
        </div>

        {/* ---------------------------------------------------------------- */}
        {/* RIGHT PANEL                                                       */}
        {/* ---------------------------------------------------------------- */}
        <div className="flex flex-col gap-5 flex-1 min-w-0">
          {/* Rating curve */}
          <div>
            <p className="text-xs font-semibold text-slate-500 mb-1">
              Rating Curve Q(h) — tangent slope = c·B = dQ/dh
            </p>
            <svg
              width={SVG_W}
              height={SVG_H}
              className="block"
              style={{ background: '#f8fafc', borderRadius: 8, border: '1px solid #e2e8f0' }}
            >
              {/* Gridlines h */}
              {hGrids.map((hg) => (
                <line
                  key={`hg-${hg}`}
                  x1={PAD.left}
                  y1={hToY(hg)}
                  x2={SVG_W - PAD.right}
                  y2={hToY(hg)}
                  stroke="#e2e8f0"
                  strokeWidth={1}
                />
              ))}
              {/* Gridlines Q */}
              {qGridFracs.map((f) => (
                <line
                  key={`qg-${f}`}
                  x1={qToX(f * Q_max)}
                  y1={PAD.top}
                  x2={qToX(f * Q_max)}
                  y2={PAD.top + plotH}
                  stroke="#e2e8f0"
                  strokeWidth={1}
                />
              ))}

              {/* Rating curve fill */}
              <polyline
                points={`${PAD.left},${PAD.top + plotH} ` + curvePoints + ` ${qToX(Q_max)},${PAD.top + plotH}`}
                fill="url(#curveGrad)"
                opacity={0.25}
                stroke="none"
              />
              <defs>
                <linearGradient id="curveGrad" x1="0" y1="0" x2="1" y2="0">
                  <stop offset="0%" stopColor="#0ea5e9" />
                  <stop offset="100%" stopColor="#1e3a8a" />
                </linearGradient>
              </defs>

              {/* Rating curve line */}
              <polyline
                points={curvePoints}
                fill="none"
                stroke="#1e3a8a"
                strokeWidth={2.5}
                strokeLinejoin="round"
              />

              {/* Secant line from origin (A=0, Q=0) to the selected point — average rate u */}
              <line
                x1={qToX(0)}
                y1={hToY(0)}
                x2={qToX(Q_sel)}
                y2={hToY(h)}
                stroke="#0ea5e9"
                strokeWidth={1.5}
                strokeDasharray="2 2"
                opacity={0.85}
              />
              <text
                x={qToX(Q_sel * 0.45)}
                y={hToY(h * 0.45) + 12}
                fontSize={8}
                fill="#0369a1"
                fontFamily="monospace"
              >
                secant: Q/A = u
              </text>

              {/* Tangent line */}
              <line
                x1={tx1}
                y1={ty1}
                x2={tx2}
                y2={ty2}
                stroke="#f59e0b"
                strokeWidth={2}
                strokeDasharray="4 3"
              />
              <text
                x={qToX(Q_sel) + 8}
                y={hToY(h) - 8}
                fontSize={9}
                fill="#b45309"
                fontFamily="monospace"
              >
                tangent: dQ/dA = c
              </text>

              {/* Selected point */}
              <circle
                cx={qToX(Q_sel)}
                cy={hToY(h)}
                r={6}
                fill="#f97316"
                stroke="white"
                strokeWidth={2}
              />

              {/* Horizontal line from y-axis to point */}
              <line
                x1={PAD.left}
                y1={hToY(h)}
                x2={qToX(Q_sel)}
                y2={hToY(h)}
                stroke="#f97316"
                strokeWidth={1}
                strokeDasharray="3 2"
                opacity={0.6}
              />

              {/* Axes */}
              <line
                x1={PAD.left}
                y1={PAD.top}
                x2={PAD.left}
                y2={PAD.top + plotH}
                stroke="#475569"
                strokeWidth={1.5}
              />
              <line
                x1={PAD.left}
                y1={PAD.top + plotH}
                x2={SVG_W - PAD.right}
                y2={PAD.top + plotH}
                stroke="#475569"
                strokeWidth={1.5}
              />

              {/* Y axis labels */}
              {hGrids.map((hg) => (
                <text
                  key={`hl-${hg}`}
                  x={PAD.left - 4}
                  y={hToY(hg) + 3}
                  fontSize={9}
                  textAnchor="end"
                  fill="#64748b"
                  fontFamily="monospace"
                >
                  {hg.toFixed(1)}
                </text>
              ))}

              {/* X axis labels */}
              {qGridFracs.map((f) => (
                <text
                  key={`ql-${f}`}
                  x={qToX(f * Q_max)}
                  y={PAD.top + plotH + 14}
                  fontSize={9}
                  textAnchor="middle"
                  fill="#64748b"
                  fontFamily="monospace"
                >
                  {(f * Q_max).toFixed(1)}
                </text>
              ))}

              {/* Axis titles */}
              <text
                x={PAD.left + plotW / 2}
                y={SVG_H - 2}
                fontSize={10}
                textAnchor="middle"
                fill="#334155"
                fontFamily="sans-serif"
              >
                Q (m³/s)
              </text>
              <text
                x={9}
                y={PAD.top + plotH / 2}
                fontSize={10}
                textAnchor="middle"
                fill="#334155"
                fontFamily="sans-serif"
                transform={`rotate(-90, 9, ${PAD.top + plotH / 2})`}
              >
                h (m)
              </text>
            </svg>

            {/* Selected point readout */}
            <div className="mt-2 flex gap-3 flex-wrap text-xs font-mono text-slate-600">
              <span>
                h = <strong className="text-sky-700">{h.toFixed(2)} m</strong>
              </span>
              <span>
                Q = <strong className="text-sky-700">{Q_sel.toFixed(3)} m³/s</strong>
              </span>
              <span>
                u = <strong className="text-blue-700">{u_sel.toFixed(3)} m/s</strong>
              </span>
              <span>
                c = <strong className="text-orange-600">{c_sel.toFixed(3)} m/s</strong>
              </span>
              <span className={C_sel <= 1 ? 'text-green-700 font-semibold' : 'text-red-600 font-semibold'}>
                Cr = {C_sel.toFixed(3)}
              </span>
            </div>
            <p className="text-xs text-slate-500 mt-1.5 max-w-xs">
              Because the rating curve bends upward (it&apos;s convex — Q&apos;&apos; &gt; 0 since
              β &gt; 1), the tangent at any point is always steeper than the secant from the
              origin. That&apos;s a purely geometric proof that c &gt; u for any convex rating
              curve, not just this one — the result generalizes to any power law Q = α·A^β with
              β &gt; 1, not only wide rectangular channels.
            </p>
          </div>

          {/* Wave vs particle animation */}
          <div>
            <p className="text-xs font-semibold text-slate-500 mb-1">
              Wave crest (amber) vs water particles (blue) — c = (5/3) u
            </p>

            <svg
              width={300}
              height={80}
              className="block"
              style={{ borderRadius: 8, border: '1px solid #bfdbfe' }}
            >
              {/* Channel background */}
              <rect x={0} y={15} width={300} height={50} fill="#dbeafe" rx={0} />
              {/* Channel border lines */}
              <line x1={0} y1={15} x2={300} y2={15} stroke="#93c5fd" strokeWidth={1.5} />
              <line x1={0} y1={65} x2={300} y2={65} stroke="#93c5fd" strokeWidth={1.5} />

              {/* Water particles */}
              {[0, 1, 2, 3, 4].map((k) => {
                const px = particleXpx(k);
                return (
                  <circle
                    key={k}
                    cx={px}
                    cy={40}
                    r={6}
                    fill="#3b82f6"
                    stroke="#1d4ed8"
                    strokeWidth={1.5}
                    opacity={px < 292 ? 1 : 0}
                  />
                );
              })}

              {/* Wave crest */}
              <rect
                x={waveXpx - 1}
                y={15}
                width={2}
                height={50}
                fill="#f59e0b"
                opacity={0.9}
              />
              <polygon
                points={`${waveXpx - 6},15 ${waveXpx + 6},15 ${waveXpx},8`}
                fill="#f59e0b"
              />

              {/* Labels */}
              <text x={waveXpx} y={6} fontSize={8} textAnchor="middle" fill="#92400e" fontFamily="sans-serif">
                wave
              </text>

              {/* Channel bed */}
              <rect x={0} y={65} width={300} height={15} fill="#94a3b8" />
            </svg>

            {/* Counters */}
            <div className="flex gap-4 mt-2 text-xs font-mono text-slate-600">
              <span>
                Wave:{' '}
                <strong className="text-amber-600">{waveDistM.toFixed(1)} m</strong>
              </span>
              <span>
                Particles:{' '}
                <strong className="text-blue-600">{particleDistM.toFixed(1)} m</strong>
              </span>
              <span>
                Ratio:{' '}
                <strong className="text-slate-800">{ratio}×</strong>
              </span>
            </div>

            {/* Play / Pause / Reset */}
            <div className="flex gap-2 mt-2">
              <button
                onClick={() => setPlaying((p) => !p)}
                className="px-3 py-1 rounded-md text-xs font-semibold bg-sky-600 text-white hover:bg-sky-700 active:bg-sky-800 transition-colors"
              >
                {playing ? 'Pause' : 'Play'}
              </button>
              <button
                onClick={() => {
                  setPlaying(false);
                  setAnimX(0);
                }}
                className="px-3 py-1 rounded-md text-xs font-semibold bg-slate-200 text-slate-700 hover:bg-slate-300 active:bg-slate-400 transition-colors"
              >
                Reset
              </button>
            </div>

            {/* Callout */}
            <div className="bg-amber-50 border border-amber-200 rounded-lg px-4 py-2 text-sm text-amber-900 mt-3 max-w-xs">
              The flood WAVE arrives 5/3 &times; sooner than the water molecules that created it.
            </div>
          </div>

          {/* Physics summary box */}
          <div className="bg-slate-50 border border-slate-200 rounded-xl px-4 py-3 text-xs text-slate-700 font-mono leading-relaxed max-w-xs">
            <div className="font-bold text-slate-800 text-sm mb-1 font-sans">Key relations</div>
            <div>Q = α · A^β, &nbsp; α = √S₀ / (n · B^(2/3))</div>
            <div>β = 5/3 &nbsp; (wide rectangular channel)</div>
            <div>u = Q / A &nbsp; (flow velocity)</div>
            <div className="text-orange-700 font-semibold mt-1">c = dQ/dA = β · u = (5/3) u</div>
            <div>Cr = c · Δt / Δx &nbsp; (Courant number)</div>
            <div className="mt-1 text-slate-500">Stable explicit scheme: Cr ≤ 1</div>
          </div>
        </div>
      </div>

      {/* ------------------------------------------------------------------ */}
      {/* DERIVATION SECTION — why c = (5/3)u, not just asserted              */}
      {/* ------------------------------------------------------------------ */}
      <div className="px-6 pb-6">
        <div className="border-t border-slate-200 pt-6">
          <h4 className="text-lg font-bold text-slate-800 mb-4">
            Why Is the Wave Faster Than the Water? — Deriving c = (5/3)u
          </h4>

          {/* Part A — progressive-reveal algebraic derivation */}
          <div className="mb-6">
            <div className="bg-slate-50 rounded-lg p-3 font-mono text-sm text-center mb-3">
              Start: Manning&apos;s equation, valid for ANY channel shape:
              <div className="mt-2">Q = (1/n) · A · R^(2/3) · S₀^(1/2)</div>
            </div>

            <div className="space-y-3">
              {visibleSteps >= 1 && (
                <div>
                  <div className="text-xs font-bold text-slate-500 uppercase mb-1">
                    Step 1 — Specialize to a wide rectangular channel (B ≫ h):
                  </div>
                  <div className="bg-white border border-slate-200 rounded-lg p-3 font-mono text-xs sm:text-sm text-center overflow-x-auto">
                    A = B · h, &nbsp; wetted perimeter P ≈ B (side walls negligible) &nbsp;⇒&nbsp; R = A / P ≈ h
                  </div>
                  <p className="text-xs text-slate-600 mt-1">
                    When the channel is much wider than it is deep, the two short side walls
                    barely add to the wetted perimeter — so the hydraulic radius collapses to
                    just the depth.
                  </p>
                </div>
              )}

              {visibleSteps >= 2 && (
                <div>
                  <div className="text-xs font-bold text-slate-500 uppercase mb-1">
                    Step 2 — Substitute R ≈ h into Manning&apos;s equation:
                  </div>
                  <div className="bg-white border border-slate-200 rounded-lg p-3 font-mono text-xs sm:text-sm text-center overflow-x-auto">
                    Q = (1/n) · (B·h) · h^(2/3) · S₀^(1/2) &nbsp;=&nbsp; (S₀^(1/2)/n) · B · h^(5/3)
                  </div>
                </div>
              )}

              {visibleSteps >= 3 && (
                <div className="bg-green-50 border border-green-300 rounded-lg p-3">
                  <div className="text-xs font-bold text-green-700 uppercase mb-1">
                    Step 3 — Eliminate h in favor of A, since h = A/B:
                  </div>
                  <div className="font-mono text-xs sm:text-sm text-center overflow-x-auto">
                    Q = (S₀^(1/2)/n) · B · (A/B)^(5/3) &nbsp;=&nbsp; [S₀^(1/2) / (n · B^(2/3))] · A^(5/3) &nbsp;=&nbsp; α · A^β
                  </div>
                  <p className="text-xs text-green-700 mt-1">
                    This is the power-law rating curve — every (h, Q) pair collapses onto one
                    curve in A.
                  </p>
                </div>
              )}

              {visibleSteps >= 4 && (
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
                  <div className="text-xs font-bold text-blue-700 uppercase mb-1">
                    Step 4 — Differentiate to get the celerity:
                  </div>
                  <div className="font-mono text-xs sm:text-sm text-center overflow-x-auto">
                    c = dQ/dA = α·β·A^(β−1) &nbsp;=&nbsp; β·(α·A^β)/A &nbsp;=&nbsp; β·(Q/A) &nbsp;=&nbsp; β·u
                  </div>
                  <p className="text-xs text-blue-700 mt-1">
                    With β = 5/3 for a wide rectangular channel: c = (5/3)·u.
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

          {/* Part B — physical intuition */}
          <div className="bg-amber-50 border border-amber-200 rounded-lg px-4 py-3 text-sm text-amber-900 mb-6">
            <p>
              A rise in depth does two things at once: (1) there&apos;s simply more
              cross-sectional area to carry water — that&apos;s the factor of A; and (2) the
              water that&apos;s already there speeds up, because R^(2/3) grows too — a deeper
              channel has proportionally less wetted perimeter dragging on the flow, so
              friction&apos;s grip loosens and velocity rises. The two effects compound
              multiplicatively: A¹ from the extra area, A^(2/3) from the extra speed, giving
              Q ∝ A^(5/3). Because Q grows faster than A, the marginal rate dQ/dA exceeds the
              average rate Q/A — and a disturbance (the wave) travels at the marginal rate.
            </p>
            <p className="mt-2 italic">
              It&apos;s the same reason a wider highway lane doesn&apos;t just fit more cars — it
              also lets them drive faster. The marginal car entering downstream moves traffic
              more than the average flow rate would suggest.
            </p>
          </div>

          {/* Part C is layered onto the existing rating-curve SVG above (secant line + caption). */}

          {/* Part D — worked numeric readout, reusing live reactive variables */}
          <div className="bg-slate-50 border border-slate-200 rounded-xl px-4 py-3 text-xs font-mono text-slate-700 leading-relaxed max-w-md">
            <div className="font-bold text-slate-800 text-sm mb-1 font-sans">
              At your current settings
            </div>
            <div>
              A = <strong className="text-sky-700">{A_sel.toFixed(2)} m²</strong>, &nbsp; Q ={' '}
              <strong className="text-sky-700">{Q_sel.toFixed(2)} m³/s</strong>
            </div>
            <div>
              u = Q/A = <strong className="text-blue-700">{u_sel.toFixed(3)} m/s</strong>
            </div>
            <div>
              c = (5/3)u = <strong className="text-orange-600">{c_sel.toFixed(3)} m/s</strong>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
