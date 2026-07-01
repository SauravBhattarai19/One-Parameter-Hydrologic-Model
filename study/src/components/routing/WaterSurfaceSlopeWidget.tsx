'use client';

import React, { useMemo, useState } from 'react';

// ---------------------------------------------------------------------------
// Fixed geometry for this widget
// ---------------------------------------------------------------------------
const Z_I = 5.0; // bed elevation, upstream cell (m)
const Z_DS = 4.9; // bed elevation, downstream cell (m)
const DIST = 100; // distance between cell centers (m)
const S0 = (Z_I - Z_DS) / DIST; // bed slope = 0.001
const HALT_THRESHOLD_M = S0 * DIST; // h_ds - h_i above this halts flow = 0.10 m

// ---------------------------------------------------------------------------
// Small reusable slider (matches established widget styling)
// ---------------------------------------------------------------------------
interface SliderProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  display: string;
  onChange: (v: number) => void;
  accent?: string;
}

function LabeledSlider({ label, value, min, max, step, display, onChange, accent }: SliderProps) {
  return (
    <div className="flex flex-col gap-0.5">
      <div className="flex items-center justify-between">
        <label className="text-xs font-semibold text-slate-500">{label}</label>
        <span className="text-xs font-mono text-slate-700 tabular-nums">{display}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className={`w-full h-1.5 rounded-full cursor-pointer ${accent ?? 'accent-sky-600'}`}
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------
export default function WaterSurfaceSlopeWidget() {
  const [hI, setHI] = useState(0.008); // m, upstream depth
  const [hDs, setHDs] = useState(0.003); // m, downstream depth

  // --- Physics ---------------------------------------------------------
  const Sw = useMemo(() => S0 + (hI - hDs) / DIST, [hI, hDs]);
  const Seff = Math.max(Sw, 0);
  const diff_mm = (hDs - hI) * 1000; // current h_ds - h_i, in mm, for the threshold callout
  const halted = Seff <= 0;

  // Classify the regime for labeling / coloring
  let regime: 'normal' | 'equal' | 'backwater';
  if (Math.abs(hI - hDs) < 1e-6) regime = 'equal';
  else if (hI > hDs) regime = 'normal';
  else regime = 'backwater';

  // Flow-direction arrow: shrink + recolor as Seff -> 0
  // Reference scale: Seff at the "equal" case (=S0) renders at full length
  const arrowFrac = Math.min(Seff / S0, 1.4); // can exceed 1 slightly in the "normal" case
  const arrowLenPx = 18 + Math.min(arrowFrac, 1) * 46; // 18..64 px
  function arrowColor(frac: number): string {
    if (frac <= 0) return '#94a3b8'; // slate (halted)
    if (frac < 0.25) return '#dc2626'; // red
    if (frac < 0.6) return '#f59e0b'; // amber
    return '#16a34a'; // green
  }
  const arrowCol = arrowColor(arrowFrac);

  // --- SVG side-view diagram --------------------------------------------
  const SVG_W = 360;
  const SVG_H = 200;
  const PAD_L = 50;
  const PAD_R = 30;
  const cellIx = PAD_L + 50;
  const cellDsx = SVG_W - PAD_R - 50;

  // Vertical mapping: exaggerate depth relative to the tiny bed drop so it's visible.
  // Bed drop (0.10 m) maps to a modest pixel span; depth is then exaggerated further
  // on top of that by DEPTH_EXAGGERATION so 0-150 mm depths are clearly readable.
  const BED_DROP_PX = 30; // pixels for the real 0.10 m bed elevation drop
  const bedPxPerM = BED_DROP_PX / (Z_I - Z_DS);
  const DEPTH_EXAGGERATION = 220; // depth rendered at 220x its true scale vs. the bed
  const BASELINE_Y = 150; // y-pixel for z_ds (downstream bed, the lower bed)

  function bedY(z: number): number {
    // higher z -> smaller y (higher on screen); referenced to Z_DS at BASELINE_Y
    return BASELINE_Y - (z - Z_DS) * bedPxPerM;
  }
  function wseY(z: number, h: number): number {
    return bedY(z) - h * bedPxPerM * DEPTH_EXAGGERATION;
  }

  const zIy = bedY(Z_I);
  const zDsy = bedY(Z_DS);
  const wseIy = wseY(Z_I, hI);
  const wseDsy = wseY(Z_DS, hDs);

  const cellHalfW = 38;

  // WSE line tilt direction for a quick visual label
  const wseSlopePx = wseDsy - wseIy; // positive => downstream surface lower on screen => "downhill" (normal-ish)
  const tiltLabel =
    Math.abs(wseDsy - wseIy) < 1.0
      ? 'level'
      : wseSlopePx > 0
      ? 'tilts downhill →'
      : 'tilts uphill (adverse) →';

  return (
    <div className="not-prose my-8 rounded-2xl border border-slate-200 bg-white shadow-xl overflow-hidden font-sans">
      {/* Header */}
      <div className="bg-gradient-to-r from-teal-700 to-emerald-900 px-6 py-4">
        <h3 className="text-white font-bold text-lg tracking-tight">
          Follow the Water Surface, Not the Bed
        </h3>
        <p className="text-teal-200 text-sm mt-0.5">
          S_w = S₀ + (h_i − h_ds) / Δx, clamped at zero — the diffusive-wave fix
        </p>
      </div>

      <div className="p-6 flex flex-col lg:flex-row gap-8 items-start">
        {/* LEFT: sliders + readouts */}
        <div className="w-full lg:w-80 shrink-0 flex flex-col gap-4">
          <div className="flex flex-col gap-3">
            <LabeledSlider
              label="h_i (upstream depth)"
              value={hI}
              min={0}
              max={0.15}
              step={0.001}
              display={`${(hI * 1000).toFixed(0)} mm`}
              onChange={setHI}
              accent="accent-blue-600"
            />
            <LabeledSlider
              label="h_ds (downstream depth)"
              value={hDs}
              min={0}
              max={0.15}
              step={0.001}
              display={`${(hDs * 1000).toFixed(0)} mm`}
              onChange={setHDs}
              accent="accent-cyan-600"
            />
          </div>

          {/* Formula / readout box */}
          <div className="bg-slate-50 font-mono text-xs p-3 rounded space-y-1">
            <div>S₀ = (z_i − z_ds) / Δx = {S0.toFixed(5)}</div>
            <div>
              S_w = S₀ + (h_i − h_ds)/Δx ={' '}
              <strong className={Sw < 0 ? 'text-red-600' : 'text-slate-800'}>
                {Sw.toFixed(5)}
              </strong>
              {Sw < 0 && <span className="text-red-600"> (negative — would reverse flow)</span>}
            </div>
            <div>
              S_eff = max(S_w, 0) ={' '}
              <strong className={halted ? 'text-slate-500' : 'text-emerald-700'}>
                {Seff.toFixed(5)}
              </strong>
              {halted && <span className="text-slate-500"> — clamp engaged</span>}
            </div>
          </div>

          {/* Threshold callout */}
          <div className="bg-amber-50 border border-amber-200 rounded-lg px-4 py-3 text-xs text-amber-900">
            <p className="font-semibold mb-1">Halt threshold for this geometry</p>
            <p>
              Flow halts once h_ds − h_i exceeds S₀·Δx ={' '}
              <strong>{(HALT_THRESHOLD_M * 1000).toFixed(0)} mm</strong>.
            </p>
            <p className="mt-1">
              Current h_ds − h_i ={' '}
              <strong className={diff_mm > HALT_THRESHOLD_M * 1000 - 1e-6 ? 'text-red-700' : 'text-amber-900'}>
                {diff_mm.toFixed(0)} mm
              </strong>{' '}
              ({((diff_mm / (HALT_THRESHOLD_M * 1000)) * 100).toFixed(0)}% of threshold)
            </p>
          </div>

          {/* Intuition callout */}
          <div className="bg-emerald-50 border border-emerald-200 rounded-lg px-4 py-3 text-sm text-emerald-900">
            Picture two ponds connected by a pipe. Water in the pipe does not care how
            the ground tilts beneath it — it only cares about the slope of its own
            surface, pond to pond. If the downstream pond happens to be fuller than
            gravity on the bed alone would suggest, that surface slope shrinks, flow
            slows, and — if the downstream pond is full enough to out-stand the
            upstream one — flow stops entirely rather than running backward.
          </div>
        </div>

        {/* RIGHT: SVG diagram + flow arrow */}
        <div className="flex-1 flex flex-col gap-4 min-w-0">
          <div>
            <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2">
              Side View — depth exaggerated {DEPTH_EXAGGERATION}× for visibility
            </p>
            <svg
              width={SVG_W}
              height={SVG_H}
              className="block"
              style={{ background: '#f0f9ff', borderRadius: 8, border: '1px solid #bae6fd', maxWidth: '100%' }}
            >
              {/* Bed (brown), descending step from i to ds */}
              <polygon
                points={`${cellIx - cellHalfW},${zIy} ${cellIx + cellHalfW},${zIy} ${cellDsx - cellHalfW},${zDsy} ${cellDsx - cellHalfW},${SVG_H - 10} ${cellIx - cellHalfW},${SVG_H - 10}`}
                fill="#92704a"
              />
              <polygon
                points={`${cellDsx - cellHalfW},${zDsy} ${cellDsx + cellHalfW},${zDsy} ${cellDsx + cellHalfW},${SVG_H - 10} ${cellDsx - cellHalfW},${SVG_H - 10}`}
                fill="#92704a"
              />
              {/* Bed top lines for clarity */}
              <line x1={cellIx - cellHalfW} y1={zIy} x2={cellIx + cellHalfW} y2={zIy} stroke="#5b4326" strokeWidth={2} />
              <line x1={cellIx + cellHalfW} y1={zIy} x2={cellDsx - cellHalfW} y2={zDsy} stroke="#5b4326" strokeWidth={2} strokeDasharray="3,3" />
              <line x1={cellDsx - cellHalfW} y1={zDsy} x2={cellDsx + cellHalfW} y2={zDsy} stroke="#5b4326" strokeWidth={2} />

              {/* Water (blue) on top of each bed, up to its exaggerated WSE */}
              <rect
                x={cellIx - cellHalfW}
                y={wseIy}
                width={cellHalfW * 2}
                height={Math.max(zIy - wseIy, 0)}
                fill="#38bdf8"
                opacity={0.85}
              />
              <rect
                x={cellDsx - cellHalfW}
                y={wseDsy}
                width={cellHalfW * 2}
                height={Math.max(zDsy - wseDsy, 0)}
                fill="#38bdf8"
                opacity={0.85}
              />

              {/* WSE connecting line — the star of the show */}
              <line
                x1={cellIx}
                y1={wseIy}
                x2={cellDsx}
                y2={wseDsy}
                stroke={halted ? '#94a3b8' : Sw >= S0 ? '#16a34a' : '#dc2626'}
                strokeWidth={3}
                strokeDasharray={halted ? '5,4' : undefined}
              />
              <circle cx={cellIx} cy={wseIy} r={3.5} fill="#0369a1" />
              <circle cx={cellDsx} cy={wseDsy} r={3.5} fill="#0369a1" />

              {/* Cell labels */}
              <text x={cellIx} y={SVG_H - 2} fontSize={10} textAnchor="middle" fill="#334155">
                cell i (z={Z_I.toFixed(2)})
              </text>
              <text x={cellDsx} y={SVG_H - 2} fontSize={10} textAnchor="middle" fill="#334155">
                cell ds (z={Z_DS.toFixed(2)})
              </text>

              {/* Flow-direction arrow between the cells, shrinks to 0 / halts */}
              <g>
                {!halted ? (
                  <>
                    <line
                      x1={(cellIx + cellDsx) / 2 - arrowLenPx / 2}
                      y1={Math.min(wseIy, wseDsy) - 18}
                      x2={(cellIx + cellDsx) / 2 + arrowLenPx / 2}
                      y2={Math.min(wseIy, wseDsy) - 18}
                      stroke={arrowCol}
                      strokeWidth={3}
                      markerEnd="url(#arrowhead)"
                    />
                    <text
                      x={(cellIx + cellDsx) / 2}
                      y={Math.min(wseIy, wseDsy) - 24}
                      fontSize={10}
                      textAnchor="middle"
                      fill={arrowCol}
                      fontWeight={600}
                    >
                      flow →
                    </text>
                  </>
                ) : (
                  <text
                    x={(cellIx + cellDsx) / 2}
                    y={Math.min(wseIy, wseDsy) - 18}
                    fontSize={12}
                    textAnchor="middle"
                    fill="#64748b"
                    fontWeight={700}
                    letterSpacing={1}
                  >
                    HALTED
                  </text>
                )}
              </g>
              <defs>
                <marker id="arrowhead" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
                  <path d="M0,0 L8,4 L0,8 Z" fill={arrowCol} />
                </marker>
              </defs>

              {/* WSE tilt label */}
              <text
                x={(cellIx + cellDsx) / 2}
                y={Math.min(wseIy, wseDsy) - 36}
                fontSize={9}
                textAnchor="middle"
                fill="#0f766e"
              >
                WSE {tiltLabel}
              </text>
            </svg>
          </div>

          {/* Regime readout */}
          <div className="bg-slate-50 font-mono text-xs p-3 rounded">
            <div className="flex items-center gap-2">
              <span
                className={`inline-block w-2.5 h-2.5 rounded-full ${
                  regime === 'normal'
                    ? 'bg-green-600'
                    : regime === 'equal'
                    ? 'bg-amber-500'
                    : 'bg-red-600'
                }`}
              />
              <span className="font-sans font-semibold text-slate-700">
                {regime === 'normal' && 'Normal — downstream relatively emptier, S_w slightly exceeds S₀'}
                {regime === 'equal' && 'Equal depths — (h_i − h_ds) term vanishes, S_w = S₀ exactly (matches kinematic)'}
                {regime === 'backwater' && !halted && 'Backwater building — S_w < S₀, flow slowing'}
                {regime === 'backwater' && halted && 'Backwater dominant — S_w ≤ 0, flow halted by the clamp'}
              </span>
            </div>
          </div>

          {/* Quick preset buttons to jump to the three illustrative cases */}
          <div className="flex gap-2 flex-wrap">
            <button
              onClick={() => {
                setHI(0.008);
                setHDs(0.003);
              }}
              className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-green-100 text-green-800 hover:bg-green-200 transition-colors"
            >
              Normal (8 / 3 mm)
            </button>
            <button
              onClick={() => {
                setHI(0.006);
                setHDs(0.006);
              }}
              className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-amber-100 text-amber-800 hover:bg-amber-200 transition-colors"
            >
              Equal (6 / 6 mm)
            </button>
            <button
              onClick={() => {
                setHI(0.0);
                setHDs(0.12);
              }}
              className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-red-100 text-red-800 hover:bg-red-200 transition-colors"
            >
              Backwater, halted (0 / 120 mm)
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
