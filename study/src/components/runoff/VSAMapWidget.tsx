'use client';

import React, { useEffect, useMemo, useState } from 'react';

// ─────────────────────────────────────────────────────────────────────────
// Spatial companion to VSAEquationBuilderWidget.tsx — same OPM VSA physics
// (Pradhan & Ogden, 2010), but applied to a small synthetic 2-D grid so the
// saturated area's GROWTH is visible as a spreading map, not just a 1-D row
// of cells. One scalar threshold A_t(t), computed from a single divide-cell
// sandbox, is compared against every cell's (precomputed) upslope area —
// this is the single-sandbox case (see PerPolygonVSAWidget for the
// multi-sandbox / spatially-variable-rain extension, not built here).
//
//   Eq.10  A_t⁽⁰⁾ = A_outlet / (1 − ln(Q_min/Q_max))      depends on Q only
//   Eq.4   H_a = [A_t0/(A_t0−A1)] · ln(SD_min/SD_max⁽⁰⁾)   depends on SDmax0
//   Eq.12  sandbox water balance (forward Euler, lateral Darcy drainage)
//   Eq.5   A_t(t) = Ha·A1 / (Ha − ln(SD_min/SD_max(t)))    dynamic threshold
//   Eq.9   cell ∈ VSA  <=>  upslope_area(cell) > A_t(t)
//
// Fixed floors: Q_min = 0.001 m^3/s, SD_min = 0.001 m.
// ─────────────────────────────────────────────────────────────────────────

const Q_MIN = 0.001; // m^3/s, fixed floor
const SD_MIN = 0.001; // m, fixed floor

const N_ROWS = 14; // row 0 = top = most upstream divide; last row = outlet
const N_COLS = 10;
const DX = 100; // m
const A1 = DX * DX; // 10,000 m^2, one cell's area
const A_OUTLET = N_ROWS * N_COLS * A1; // same total catchment, every terrain

const DT = 60; // s, forward-Euler step
const N_STEPS = 90; // total animation steps (90 * 60s = 1.5 hr storm)

function fmt(n: number, d = 2): string {
  return n.toFixed(d);
}

// ─────────────────────────────────────────────────────────────────────────
// Physics — mirrors vsa_opm.py / VSAEquationBuilderWidget.tsx exactly
// ─────────────────────────────────────────────────────────────────────────

function computeAtInit(aOutlet: number, qMax: number): number {
  return aOutlet / (1 - Math.log(Q_MIN / qMax));
}

function computeHa(atInit: number, a1: number, sdMax0: number): number {
  const ratio = atInit / (atInit - a1);
  return ratio * Math.log(SD_MIN / sdMax0);
}

function computeAtDynamic(ha: number, a1: number, sdMax: number, aOutlet: number): number {
  const denom = ha - Math.log(SD_MIN / sdMax);
  const atRaw = (ha * a1) / denom;
  return Math.min(Math.max(atRaw, a1), aOutlet);
}

interface SandboxState {
  z: number; // water-table height above impervious base, m
  sdMax: number; // current root-zone deficit, m
}

function sandboxStep(
  prev: SandboxState,
  klat: number,
  sDiv: number,
  dx: number,
  a1: number,
  phi: number,
  pDiv: number,
  dt: number,
  sdMax0: number
): SandboxState {
  const qb = klat * sDiv * prev.z * dx; // Eq.12, lateral Darcy drainage
  const dV = (pDiv * a1 - qb) * dt; // net volume this step
  const zNew = Math.max(0, prev.z + dV / (a1 * phi));
  const sdMaxNew = Math.max(SD_MIN, sdMax0 - zNew);
  return { z: zNew, sdMax: sdMaxNew };
}

// ─────────────────────────────────────────────────────────────────────────
// Synthetic terrain → upslope area. All four share the same Aoutlet
// (nRows*nCols*cellArea); only WHERE area concentrates differs.
// ─────────────────────────────────────────────────────────────────────────

type Terrain = 'hillslope' | 'valley' | 'bowl' | 'branch';

const TERRAINS: { key: Terrain; label: string }[] = [
  { key: 'hillslope', label: 'Hillslope' },
  { key: 'valley', label: 'Valley' },
  { key: 'bowl', label: 'Bowl' },
  { key: 'branch', label: 'Branch' },
];

/**
 * Deterministic synthetic upslope area [m^2] for one cell, by terrain
 * archetype. row 0 = most upstream (top), row = nRows-1 = outlet (bottom).
 */
function computeUpslopeArea(
  terrain: Terrain,
  row: number,
  col: number,
  nRows: number,
  nCols: number,
  cellArea: number
): number {
  if (terrain === 'hillslope') {
    // Pure sheet flow: every column drains straight downhill, independent of
    // its neighbours. Area only grows with row depth -> uniform horizontal
    // bands, never a channel.
    return (row + 1) * cellArea;
  }

  if (terrain === 'valley') {
    // Strong linear convergence onto one central channel column. The center
    // column collects ALL upslope hillslope area from the full width above
    // each row; off-center columns keep only their own small local wedge
    // (a thin strip of width-weighted area tapering away from center).
    const center = (nCols - 1) / 2;
    const distFromCenter = Math.abs(col - center);
    const isCenter = distFromCenter < 0.5; // single center column (nCols even -> two tied center cols)
    if (isCenter) {
      // Collects everything upstream across the full row width.
      return (row + 1) * nCols * cellArea;
    }
    // Off-center: small local hillslope wedge only, decaying with distance
    // from the channel so the channel reads as a sharp, narrow high-area line.
    const localWidth = Math.max(1, nCols / 2 - distFromCenter);
    return (row + 1) * localWidth * cellArea * 0.15;
  }

  if (terrain === 'bowl') {
    // Parabolic/radial convergence: broader, smoother weighting across
    // columns (less peaked than the valley) that increases toward the
    // bottom-center, producing a widening wedge rather than a thin line.
    const center = (nCols - 1) / 2;
    const normDist = Math.abs(col - center) / Math.max(center, 1); // 0..1
    const lateralWeight = 1 - 0.7 * normDist * normDist; // smooth parabola, never near 0
    const downstreamFrac = (row + 1) / nRows; // 0..1, grows toward outlet
    // Area grows with both row depth and a downstream-amplified lateral bowl.
    return (row + 1) * cellArea * lateralWeight * (0.6 + 0.4 * downstreamFrac * nCols);
  }

  // branch: dendritic — two tributaries in the upper half (at ~1/4 and ~3/4
  // width) that each independently accumulate area, merging into a single
  // channel for the lower half.
  const mid = (nRows - 1) / 2;
  const quarter = nCols / 4;
  const threeQuarter = (3 * nCols) / 4;
  const center = (nCols - 1) / 2;

  if (row <= mid) {
    // Upper half: two independent tributary channels.
    const distA = Math.abs(col - quarter);
    const distB = Math.abs(col - threeQuarter);
    const nearestDist = Math.min(distA, distB);
    const isChannel = nearestDist < 0.5;
    if (isChannel) {
      // Each tributary collects area from its own half-width above this row.
      return (row + 1) * (nCols / 2) * cellArea;
    }
    const localWidth = Math.max(1, nCols / 4 - nearestDist);
    return (row + 1) * localWidth * cellArea * 0.15;
  }

  // Lower half: tributaries have merged into one central trunk channel that
  // now carries the combined upstream area of both branches plus everything
  // accumulated since the merge.
  const distFromCenter = Math.abs(col - center);
  const isTrunk = distFromCenter < 0.5;
  const areaAtMerge = (mid + 1) * (nCols / 2) * cellArea * 2; // both tributaries combined
  const rowsSinceMerge = row - mid;
  if (isTrunk) {
    return areaAtMerge + rowsSinceMerge * nCols * cellArea;
  }
  const localWidth = Math.max(1, nCols / 2 - distFromCenter);
  return (row + 1) * localWidth * cellArea * 0.15;
}

function buildUpslopeGrid(terrain: Terrain): number[][] {
  const grid: number[][] = [];
  for (let r = 0; r < N_ROWS; r++) {
    const rowArr: number[] = [];
    for (let c = 0; c < N_COLS; c++) {
      rowArr.push(computeUpslopeArea(terrain, r, c, N_ROWS, N_COLS, A1));
    }
    grid.push(rowArr);
  }
  return grid;
}

// ─────────────────────────────────────────────────────────────────────────
// Small presentational helpers (conventions matched to VSAEquationBuilderWidget)
// ─────────────────────────────────────────────────────────────────────────

interface SliderProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  display: string;
  onChange: (v: number) => void;
}

function LabeledSlider({ label, value, min, max, step, display, onChange }: SliderProps) {
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
        className="w-full h-1.5 rounded-full accent-sky-600 cursor-pointer"
      />
    </div>
  );
}

function terrainBtnClass(active: boolean): string {
  return active
    ? 'px-3 py-1.5 rounded-lg text-xs font-semibold bg-sky-600 text-white transition-colors'
    : 'px-3 py-1.5 rounded-lg text-xs font-semibold bg-slate-100 text-slate-600 border border-slate-200 hover:bg-slate-200 transition-colors';
}

// ─────────────────────────────────────────────────────────────────────────
// Root widget
// ─────────────────────────────────────────────────────────────────────────

export default function VSAMapWidget() {
  const [terrain, setTerrain] = useState<Terrain>('valley');
  const [rainMmHr, setRainMmHr] = useState(20); // storm intensity at divide
  const [sdMax0, setSdMax0] = useState(0.1); // antecedent dryness (bigger = drier)
  const [qMax, setQMax] = useState(0.5); // m^3/s, pre-storm baseflow peak
  const [step, setStep] = useState(0); // animation step, 0..N_STEPS
  const [playing, setPlaying] = useState(false);

  // Fixed sandbox parameters (not the pedagogical focus of this widget;
  // VSAEquationBuilderWidget already lets users explore these directly).
  const phi = 0.35;
  const klat = 44 / 86400; // m/day -> m/s
  const sDiv = 0.05; // m/m

  const dx = DX;
  const pDiv = (rainMmHr / 1000) / 3600; // mm/hr -> m/s

  // Eq.10 — initial threshold (depends only on Aoutlet, Qmin, Qmax)
  const atInit = useMemo(() => computeAtInit(A_OUTLET, qMax), [qMax]);

  // Eq.4 — Ha, computed once from sdMax0 (a drier start => more negative Ha
  // => more room for At(t) to fall as the storm proceeds)
  const ha = useMemo(() => computeHa(atInit, A1, sdMax0), [atInit, sdMax0]);

  // Eq.12 — march the divide-cell sandbox forward N_STEPS times from z=0
  const sandboxTrace = useMemo(() => {
    const trace: SandboxState[] = [{ z: 0, sdMax: sdMax0 }];
    let cur = trace[0];
    for (let n = 0; n < N_STEPS; n++) {
      cur = sandboxStep(cur, klat, sDiv, dx, A1, phi, pDiv, DT, sdMax0);
      trace.push(cur);
    }
    return trace;
  }, [sdMax0, klat, sDiv, dx, phi, pDiv]);

  // Eq.5 — dynamic threshold at every step (timeseries for the strip below)
  const atSeries = useMemo(
    () => sandboxTrace.map((s) => computeAtDynamic(ha, A1, s.sdMax, A_OUTLET)),
    [sandboxTrace, ha]
  );

  // Upslope-area grid for the selected terrain (recomputed only on terrain change)
  const upslopeGrid = useMemo(() => buildUpslopeGrid(terrain), [terrain]);
  const flatUpslope = useMemo(() => upslopeGrid.flat(), [upslopeGrid]);
  const totalCells = N_ROWS * N_COLS;

  // VSA fraction at every step (for the timeseries strip)
  const vsaFractionSeries = useMemo(
    () =>
      atSeries.map((at) => (flatUpslope.filter((up) => up > at).length / totalCells) * 100),
    [atSeries, flatUpslope, totalCells]
  );

  const activeAt = step === 0 ? atInit : atSeries[step];
  const vsaMask = useMemo(
    () => upslopeGrid.map((row) => row.map((up) => up > activeAt)),
    [upslopeGrid, activeAt]
  );
  const nSaturated = vsaMask.flat().filter(Boolean).length;
  const pctSaturated = (nSaturated / totalCells) * 100;

  // ── Animation loop ──────────────────────────────────────────────────────
  useEffect(() => {
    if (!playing) return;
    const interval = setInterval(() => {
      setStep((s) => {
        if (s >= N_STEPS) {
          setPlaying(false);
          return N_STEPS;
        }
        return s + 1;
      });
    }, 180);
    return () => clearInterval(interval);
  }, [playing]);

  const handlePlayPause = () => {
    if (step >= N_STEPS && !playing) setStep(0); // restart if at the end
    setPlaying((p) => !p);
  };
  const handleReset = () => {
    setPlaying(false);
    setStep(0);
  };

  // ── Grid SVG geometry ────────────────────────────────────────────────────
  const CELL = 26;
  const gridW = N_COLS * CELL;
  const gridH = N_ROWS * CELL;
  const maxUpslope = Math.max(...flatUpslope);

  function elevationFill(up: number): string {
    // Neutral slate->teal ramp by (log-ish) upslope area, just for visual
    // terrain context — independent of saturation state.
    const t = Math.sqrt(up / maxUpslope);
    const lo: [number, number, number] = [226, 232, 240]; // slate-200
    const hi: [number, number, number] = [148, 163, 184]; // slate-400
    const r = Math.round(lo[0] + t * (hi[0] - lo[0]));
    const g = Math.round(lo[1] + t * (hi[1] - lo[1]));
    const b = Math.round(lo[2] + t * (hi[2] - lo[2]));
    return `rgb(${r},${g},${b})`;
  }

  // ── Timeseries strip geometry ────────────────────────────────────────────
  const TS_W = 560;
  const TS_H = 140;
  const TS_PAD = { left: 46, right: 46, top: 10, bottom: 22 };
  const plotW = TS_W - TS_PAD.left - TS_PAD.right;
  const plotH = TS_H - TS_PAD.top - TS_PAD.bottom;
  const maxAt = Math.max(...atSeries);
  const minAt = Math.min(...atSeries);

  function stepToX(s: number): number {
    return TS_PAD.left + (s / N_STEPS) * plotW;
  }
  function atToY(at: number): number {
    const t = maxAt === minAt ? 0.5 : (at - minAt) / (maxAt - minAt);
    return TS_PAD.top + plotH - t * plotH; // higher At -> higher on chart
  }
  function pctToY(pct: number): number {
    return TS_PAD.top + plotH - (pct / 100) * plotH;
  }

  const atPath = atSeries.map((at, i) => `${stepToX(i)},${atToY(at)}`).join(' ');
  const pctPath = vsaFractionSeries.map((p, i) => `${stepToX(i)},${pctToY(p)}`).join(' ');

  const isDefaultSettings = rainMmHr === 20 && sdMax0 === 0.1 && qMax === 0.5;

  return (
    <div className="not-prose my-8 rounded-2xl border border-slate-200 bg-white shadow-xl overflow-hidden font-sans">
      {/* Header */}
      <div className="bg-gradient-to-r from-sky-700 to-emerald-800 px-6 py-4">
        <h3 className="text-white font-bold text-lg tracking-tight">
          Watching the VSA Grow Across a Watershed
        </h3>
        <p className="text-sky-200 text-sm mt-0.5">
          One scalar threshold A_t(t), compared against every cell&apos;s upslope area — the
          spatial companion to the VSA equation builder
        </p>
      </div>

      <div className="p-6 space-y-6">
        {/* ── Terrain selector ──────────────────────────────────────────── */}
        <div>
          <div className="text-xs font-bold text-slate-500 uppercase mb-2">
            Terrain type — where upslope area concentrates
          </div>
          <div className="flex flex-wrap gap-1.5">
            {TERRAINS.map((t) => (
              <button
                key={t.key}
                onClick={() => setTerrain(t.key)}
                className={terrainBtnClass(terrain === t.key)}
              >
                {t.label}
              </button>
            ))}
          </div>
        </div>

        {/* ── Storm + antecedent sliders ───────────────────────────────── */}
        <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
          <div className="text-xs font-bold text-slate-500 uppercase mb-3">
            Storm &amp; antecedent-state controls
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <LabeledSlider
              label="Storm intensity P_div"
              value={rainMmHr}
              min={1}
              max={60}
              step={1}
              display={`${rainMmHr} mm/hr`}
              onChange={(v) => {
                setRainMmHr(v);
                handleReset();
              }}
            />
            <LabeledSlider
              label="Antecedent dryness SD_max⁽⁰⁾"
              value={sdMax0}
              min={0.02}
              max={0.3}
              step={0.005}
              display={`${fmt(sdMax0, 3)} m`}
              onChange={(v) => {
                setSdMax0(v);
                handleReset();
              }}
            />
            <LabeledSlider
              label="Q_max (pre-storm baseflow peak)"
              value={qMax}
              min={0.05}
              max={3}
              step={0.05}
              display={`${fmt(qMax, 2)} m³/s`}
              onChange={(v) => {
                setQMax(v);
                handleReset();
              }}
            />
          </div>
          <p className="text-xs text-slate-500 italic mt-3">
            A drier antecedent state has more room to expand — to see the VSA visibly grow you
            need a drier start and a heavier storm.
          </p>
          {!isDefaultSettings && (
            <button
              onClick={() => {
                setRainMmHr(20);
                setSdMax0(0.1);
                setQMax(0.5);
                handleReset();
              }}
              className="mt-3 text-xs font-semibold px-3 py-1.5 rounded-full bg-slate-200 text-slate-700 hover:bg-slate-300 transition"
            >
              ⟲ Reset to defaults
            </button>
          )}
        </div>

        {/* ── Map + readouts ────────────────────────────────────────────── */}
        <div className="flex flex-col lg:flex-row gap-6 items-start">
          {/* Grid map */}
          <div className="flex flex-col gap-2 flex-shrink-0">
            <svg width={gridW} height={gridH} style={{ display: 'block', borderRadius: 8, overflow: 'hidden' }}>
              {upslopeGrid.map((row, r) =>
                row.map((up, c) => {
                  const saturated = vsaMask[r][c];
                  return (
                    <rect
                      key={`${r}-${c}`}
                      x={c * CELL}
                      y={r * CELL}
                      width={CELL}
                      height={CELL}
                      fill={saturated ? '#2563eb' : elevationFill(up)}
                      fillOpacity={saturated ? 0.85 : 1}
                      stroke="white"
                      strokeWidth={1}
                    />
                  );
                })
              )}
              {/* Outlet marker at bottom row, center */}
              <text
                x={gridW / 2}
                y={gridH - 4}
                textAnchor="middle"
                fontSize={9}
                fontWeight={700}
                fill="#1e3a5f"
                opacity={0.6}
              >
                ↓ outlet
              </text>
              <text x={gridW / 2} y={10} textAnchor="middle" fontSize={9} fontWeight={700} fill="#1e3a5f" opacity={0.6}>
                divide ↑
              </text>
            </svg>
            <p className="text-xs text-slate-400 text-center">
              Blue = saturated (in the VSA) &nbsp;·&nbsp; gray-scale = relative upslope area
              (elevation-shaded, not saturation)
            </p>
          </div>

          {/* Readouts + animation controls */}
          <div className="flex flex-col gap-3 flex-1 min-w-[220px]">
            <div className="bg-slate-50 rounded-xl border border-slate-200 p-4 text-sm">
              <div className="flex justify-between py-0.5">
                <span className="text-slate-500">Time</span>
                <span className="font-mono font-semibold text-slate-800">
                  t = {step * DT}s ({fmt((step * DT) / 60, 0)} min)
                </span>
              </div>
              <div className="flex justify-between py-0.5">
                <span className="text-slate-500">A_t(t)</span>
                <span className="font-mono font-semibold text-sky-700">
                  {fmt(activeAt, 0)} m²
                </span>
              </div>
              <div className="flex justify-between py-0.5">
                <span className="text-slate-500">VSA fraction</span>
                <span className="font-mono font-semibold text-blue-700">
                  {fmt(pctSaturated, 1)}% ({nSaturated}/{totalCells} cells)
                </span>
              </div>
              <div className="flex justify-between py-0.5">
                <span className="text-slate-500">H_a (fixed this run)</span>
                <span className="font-mono font-semibold text-slate-700">{fmt(ha, 2)}</span>
              </div>
              <div className="flex justify-between py-0.5">
                <span className="text-slate-500">A_outlet</span>
                <span className="font-mono font-semibold text-slate-700">
                  {A_OUTLET.toLocaleString()} m²
                </span>
              </div>
            </div>

            {/* Play / pause / step / reset */}
            <div className="flex flex-wrap items-center gap-2">
              <button
                onClick={handlePlayPause}
                className="px-3 py-1.5 rounded-md text-xs font-semibold bg-sky-600 text-white hover:bg-sky-700 active:bg-sky-800 transition-colors"
              >
                {playing ? 'Pause' : step >= N_STEPS ? 'Replay' : 'Play'}
              </button>
              <button
                onClick={() => {
                  setPlaying(false);
                  setStep((s) => Math.max(s - 1, 0));
                }}
                disabled={step <= 0}
                className="px-3 py-1.5 rounded-md text-xs font-semibold bg-slate-200 text-slate-700 hover:bg-slate-300 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
              >
                ◀ Step
              </button>
              <button
                onClick={() => {
                  setPlaying(false);
                  setStep((s) => Math.min(s + 1, N_STEPS));
                }}
                disabled={step >= N_STEPS}
                className="px-3 py-1.5 rounded-md text-xs font-semibold bg-slate-200 text-slate-700 hover:bg-slate-300 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
              >
                Step ▶
              </button>
              <button
                onClick={handleReset}
                className="px-3 py-1.5 rounded-md text-xs font-semibold bg-slate-100 text-slate-500 hover:bg-slate-200 transition-colors"
              >
                ⏮ Reset
              </button>
            </div>
            <input
              type="range"
              min={0}
              max={N_STEPS}
              step={1}
              value={step}
              onChange={(e) => {
                setPlaying(false);
                setStep(Number(e.target.value));
              }}
              className="w-full h-1.5 rounded-full accent-sky-600 cursor-pointer"
            />
          </div>
        </div>

        {/* ── Timeseries strip ──────────────────────────────────────────── */}
        <div>
          <div className="text-xs font-bold text-slate-500 uppercase mb-2">
            A_t(t) and VSA fraction over the storm
          </div>
          <svg width="100%" viewBox={`0 0 ${TS_W} ${TS_H}`} preserveAspectRatio="xMidYMid meet">
            {/* axes */}
            <line
              x1={TS_PAD.left}
              y1={TS_PAD.top}
              x2={TS_PAD.left}
              y2={TS_H - TS_PAD.bottom}
              stroke="#cbd5e1"
            />
            <line
              x1={TS_PAD.left}
              y1={TS_H - TS_PAD.bottom}
              x2={TS_W - TS_PAD.right}
              y2={TS_H - TS_PAD.bottom}
              stroke="#cbd5e1"
            />
            {/* A_t(t) curve, left axis */}
            <polyline points={atPath} fill="none" stroke="#0369a1" strokeWidth={2} />
            {/* VSA % curve, right axis */}
            <polyline points={pctPath} fill="none" stroke="#dc2626" strokeWidth={2} strokeDasharray="4,3" />
            {/* current-position marker */}
            <line
              x1={stepToX(step)}
              y1={TS_PAD.top}
              x2={stepToX(step)}
              y2={TS_H - TS_PAD.bottom}
              stroke="#0f172a"
              strokeWidth={1}
              strokeDasharray="2,2"
            />
            <circle cx={stepToX(step)} cy={atToY(activeAt)} r={3.5} fill="#0369a1" />
            <circle cx={stepToX(step)} cy={pctToY(pctSaturated)} r={3.5} fill="#dc2626" />
            {/* axis labels */}
            <text x={TS_PAD.left} y={TS_H - 6} fontSize={9} fill="#64748b">
              t=0
            </text>
            <text x={TS_W - TS_PAD.right} y={TS_H - 6} fontSize={9} fill="#64748b" textAnchor="end">
              t={N_STEPS * DT}s
            </text>
            <text x={6} y={TS_PAD.top + 8} fontSize={9} fill="#0369a1" fontWeight={700}>
              A_t
            </text>
            <text x={TS_W - 6} y={TS_PAD.top + 8} fontSize={9} fill="#dc2626" fontWeight={700} textAnchor="end">
              VSA %
            </text>
          </svg>
          <div className="flex gap-4 text-xs text-slate-500 mt-1 justify-center">
            <span className="flex items-center gap-1">
              <span className="inline-block w-3 h-0.5 bg-sky-700" /> A_t(t) [m²], falling
            </span>
            <span className="flex items-center gap-1">
              <span
                className="inline-block w-3 h-0.5 bg-red-600"
                style={{ borderTop: '2px dashed #dc2626', background: 'none' }}
              />
              VSA fraction [%], rising
            </span>
          </div>
        </div>

        {/* ── Closing caption ───────────────────────────────────────────── */}
        <div className="bg-amber-50 border border-amber-200 rounded-lg p-3 text-sm text-amber-900">
          Soil under valley bottoms and channel confluences saturates first because upslope area
          concentrates there — not because the soil itself is special.
        </div>
      </div>
    </div>
  );
}
