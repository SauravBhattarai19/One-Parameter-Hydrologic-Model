'use client';

import React, { useEffect, useMemo, useRef, useState } from 'react';

// ─────────────────────────────────────────────────────────────────────────
// Per-polygon VSA — OPM_PER_POLYGON = True (the default in config.py).
//
// A single shared VSA "sandbox" (see VSAEquationBuilderWidget.tsx) assumes
// the whole basin wets up together. Real catchments are partitioned into
// rainfall zones (Thiessen/IDW polygons around each gauge), and OPM gives
// EACH zone its own independent sandbox: its own divide cell, its own
// water-table state z, its own SD_max(t), its own threshold A_t(t) — all
// driven by that zone's own local rainfall. This widget runs the exact
// same equations as the single-zone case (Pradhan & Ogden, 2010 — same
// Eq.10 / Eq.4 / Eq.12 / Eq.5 / Eq.9 family) three times in parallel, with
// everything held identical across zones EXCEPT the rain rate — so any
// divergence you see is caused purely by rainfall heterogeneity, nothing
// else.
//
//   Eq.10  A_t^(0) = A_outlet / (1 - ln(Qmin/Qmax))      — initial threshold
//   Eq.4   H_a = [At0/(At0-A1)] * ln(SDmin/SDmax0)        — constant, once
//   Eq.12  qb = Klat*Sdiv*z*dx ; dV = (P_zone*A1-qb)*dt ; z update
//   Eq.5   A_t(t) = Ha*A1 / (Ha - ln(SDmin/SDmax(t)))     — dynamic threshold
//   Eq.9   cell in VSA  <=>  upslope_area > A_t(t)
// ─────────────────────────────────────────────────────────────────────────

const Q_MIN = 0.001; // m^3/s, fixed floor
const SD_MIN = 0.001; // m, fixed floor
const N_CELLS = 10;

const A_OUTLET = 100000; // m^2
const DX = 100; // m
const A1 = DX * DX; // 10,000 m^2
const Q_MAX = 1.0; // m^3/s
const SD_MAX0 = 0.1; // m
const PHI = 0.35; // drainable porosity
const K_LAT = 5.09e-4; // m/s (≈44 m/day)
const S_DIV = 0.05; // m/m
const DT = 60; // s

const MAX_HISTORY = 60; // cap on stored timesteps (60 * 60s = 1 hour of sim time)

function fmt(n: number, d = 2): string {
  if (!Number.isFinite(n)) return '—';
  return n.toFixed(d);
}

function fmtSci(n: number, d = 2): string {
  if (n === 0) return '0';
  if (!Number.isFinite(n)) return '—';
  return n.toExponential(d);
}

// ─────────────────────────────────────────────────────────────────────────
// Physics — identical math to VSAEquationBuilderWidget.tsx, run per zone
// ─────────────────────────────────────────────────────────────────────────

function computeAtInit(aOutlet: number, qMax: number): number {
  return aOutlet / (1 - Math.log(Q_MIN / qMax));
}

function computeHa(atInit: number, a1: number, sdMax0: number): number {
  const ratio = atInit / (atInit - a1);
  return ratio * Math.log(SD_MIN / sdMax0);
}

interface ZoneState {
  z: number; // water-table height above impervious base, m
  sdMax: number; // current root-zone deficit, m
  at: number; // current dynamic threshold area, m^2
  vsaPct: number; // current VSA percentage (0-100)
}

function computeAtDynamic(ha: number, a1: number, sdMax: number, aOutlet: number): number {
  const denom = ha - Math.log(SD_MIN / sdMax);
  const atRaw = (ha * a1) / denom;
  return Math.min(Math.max(atRaw, a1), aOutlet);
}

const UPSLOPE_AREAS = Array.from({ length: N_CELLS }, (_, i) => (i + 1) * A1);

function vsaPercentFor(at: number): number {
  const nWet = UPSLOPE_AREAS.filter((up) => up > at).length;
  return (nWet / N_CELLS) * 100;
}

function zoneStep(prev: ZoneState, pZone: number, ha: number): ZoneState {
  const qb = K_LAT * S_DIV * prev.z * DX; // Eq.12, lateral Darcy drainage
  const dV = (pZone * A1 - qb) * DT; // net volume this step
  const zNew = Math.max(0, prev.z + dV / (A1 * PHI)); // water-table rise
  const sdMaxNew = Math.max(SD_MIN, SD_MAX0 - zNew); // remaining deficit
  const atNew = computeAtDynamic(ha, A1, sdMaxNew, A_OUTLET);
  return { z: zNew, sdMax: sdMaxNew, at: atNew, vsaPct: vsaPercentFor(atNew) };
}

function initialZoneState(atInit: number): ZoneState {
  return { z: 0, sdMax: SD_MAX0, at: atInit, vsaPct: vsaPercentFor(atInit) };
}

// ─────────────────────────────────────────────────────────────────────────
// Zone configuration — only rain rate differs between zones
// ─────────────────────────────────────────────────────────────────────────

interface ZoneConfig {
  key: string;
  label: string;
  sub: string;
  color: string;
  colorBg: string;
  colorBorder: string;
  colorText: string;
  defaultRainMmHr: number;
}

const ZONES: ZoneConfig[] = [
  {
    key: 'heavy',
    label: 'Zone A — Heavy tributary',
    sub: 'Gauge sits under the storm core',
    color: '#dc2626',
    colorBg: 'bg-red-50',
    colorBorder: 'border-red-200',
    colorText: 'text-red-700',
    defaultRainMmHr: 35,
  },
  {
    key: 'moderate',
    label: 'Zone B — Moderate tributary',
    sub: 'Gauge sits on the storm fringe',
    color: '#d97706',
    colorBg: 'bg-amber-50',
    colorBorder: 'border-amber-200',
    colorText: 'text-amber-700',
    defaultRainMmHr: 20,
  },
  {
    key: 'dry',
    label: 'Zone C — Dry tributary',
    sub: 'Gauge sits outside the storm',
    color: '#2563eb',
    colorBg: 'bg-blue-50',
    colorBorder: 'border-blue-200',
    colorText: 'text-blue-700',
    defaultRainMmHr: 5,
  },
];

// ─────────────────────────────────────────────────────────────────────────
// Small presentational helpers
// ─────────────────────────────────────────────────────────────────────────

interface SparklineProps {
  values: number[];
  color: string;
  yMax: number;
  height?: number;
  unit?: string;
}

function Sparkline({ values, color, yMax, height = 56, unit }: SparklineProps) {
  const W = 220;
  const H = height;
  const PAD = 4;
  const n = values.length;

  const pts =
    n > 1
      ? values
          .map((v, i) => {
            const x = PAD + (i / (n - 1)) * (W - 2 * PAD);
            const clamped = Math.max(0, Math.min(yMax, v));
            const y = H - PAD - (clamped / yMax) * (H - 2 * PAD);
            return `${x.toFixed(1)},${y.toFixed(1)}`;
          })
          .join(' ')
      : '';

  const lastVal = values[values.length - 1] ?? 0;
  const lastX = n > 1 ? PAD + ((n - 1) / (n - 1)) * (W - 2 * PAD) : PAD;
  const lastY = H - PAD - (Math.max(0, Math.min(yMax, lastVal)) / yMax) * (H - 2 * PAD);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ height }}>
      <rect x={0} y={0} width={W} height={H} fill="#f8fafc" rx={4} />
      {n > 1 && (
        <polyline
          points={pts}
          fill="none"
          stroke={color}
          strokeWidth={2}
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      )}
      {n >= 1 && <circle cx={lastX} cy={lastY} r={3} fill={color} />}
      {unit && (
        <text x={W - 4} y={11} textAnchor="end" fontSize={8} fill="#94a3b8">
          {unit}
        </text>
      )}
    </svg>
  );
}

interface ZonePanelProps {
  config: ZoneConfig;
  rainMmHr: number;
  onRainChange: (v: number) => void;
  state: ZoneState;
  atInit: number;
  ha: number;
  atHistory: number[];
  vsaHistory: number[];
  timestep: number;
}

function ZonePanel({
  config,
  rainMmHr,
  onRainChange,
  state,
  atInit,
  ha,
  atHistory,
  vsaHistory,
  timestep,
}: ZonePanelProps) {
  return (
    <div className="rounded-xl border border-slate-200 overflow-hidden flex flex-col bg-white">
      <div className="px-3 py-2 text-white" style={{ backgroundColor: config.color }}>
        <div className="text-xs font-bold">{config.label}</div>
        <div className="text-[10px] opacity-90">{config.sub}</div>
      </div>

      <div className="p-3 flex flex-col gap-3">
        {/* Rain slider — independent per zone */}
        <div className="flex flex-col gap-0.5">
          <div className="flex items-center justify-between">
            <label className="text-xs font-semibold text-slate-500">Rain rate P (this zone only)</label>
            <span className="text-xs font-mono tabular-nums" style={{ color: config.color }}>
              {fmt(rainMmHr, 0)} mm/hr
            </span>
          </div>
          <input
            type="range"
            min={0}
            max={60}
            step={1}
            value={rainMmHr}
            onChange={(e) => onRainChange(Number(e.target.value))}
            className="w-full h-1.5 rounded-full cursor-pointer"
            style={{ accentColor: config.color }}
          />
        </div>

        {/* Live readouts */}
        <div className="bg-slate-50 font-mono text-xs p-3 rounded grid grid-cols-2 gap-y-1">
          <span className="text-slate-500">z (water table)</span>
          <span className="text-right">{fmtSci(state.z, 2)} m</span>
          <span className="text-slate-500">SD_max(t)</span>
          <span className="text-right">{fmt(state.sdMax, 4)} m</span>
          <span className="text-slate-500">A_t(t)</span>
          <span className="text-right">{fmt(state.at, 0)} m²</span>
          <span className="text-slate-500">VSA%</span>
          <span className="text-right font-bold" style={{ color: config.color }}>
            {fmt(state.vsaPct, 0)}%
          </span>
        </div>

        {/* t=0 reference */}
        <div className="text-[10px] text-slate-400 font-mono">
          t=0: A_t⁽⁰⁾ = {fmt(atInit, 0)} m², H_a = {fmt(ha, 2)} (identical across all zones)
        </div>

        {/* Sparklines */}
        <div className="flex flex-col gap-2">
          <div>
            <div className="text-[10px] font-semibold text-slate-400 uppercase mb-0.5">
              A_t(t) trace
            </div>
            <Sparkline values={atHistory} color={config.color} yMax={A_OUTLET} unit="m²" />
          </div>
          <div>
            <div className="text-[10px] font-semibold text-slate-400 uppercase mb-0.5">
              VSA% trace
            </div>
            <Sparkline values={vsaHistory} color={config.color} yMax={100} unit="%" />
          </div>
        </div>

        <div className="text-[10px] text-slate-400 font-mono text-right">
          t = {timestep * DT}s (step {timestep})
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────
// Root widget
// ─────────────────────────────────────────────────────────────────────────

export default function PerPolygonVSAWidget() {
  const [rainRates, setRainRates] = useState<Record<string, number>>(() => {
    const init: Record<string, number> = {};
    for (const z of ZONES) init[z.key] = z.defaultRainMmHr;
    return init;
  });

  const [timestep, setTimestep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Shared, zone-independent quantities (Eq.10 / Eq.4) — identical for all
  // zones since A_outlet, Q_max, SD_max0, A1 are shared. Only rain differs.
  const atInit = useMemo(() => computeAtInit(A_OUTLET, Q_MAX), []);
  const ha = useMemo(() => computeHa(atInit, A1, SD_MAX0), [atInit]);

  // Per-zone state history. Recomputed from scratch whenever a rain rate
  // changes the trajectory — each zone marches forward independently from
  // its own pDiv, but all zones share the same clock `timestep`.
  const histories = useMemo(() => {
    const out: Record<string, ZoneState[]> = {};
    for (const z of ZONES) {
      const pZone = (rainRates[z.key] / 1000) / 3600; // mm/hr -> m/s
      const trace: ZoneState[] = [initialZoneState(atInit)];
      let cur = trace[0];
      for (let n = 0; n < MAX_HISTORY; n++) {
        cur = zoneStep(cur, pZone, ha);
        trace.push(cur);
      }
      out[z.key] = trace;
    }
    return out;
  }, [rainRates, atInit, ha]);

  // Clamp timestep if histories shrink (they don't, but stay safe)
  const clampedTimestep = Math.min(timestep, MAX_HISTORY);

  const handleStep = () => {
    setTimestep((t) => Math.min(t + 1, MAX_HISTORY));
  };

  const handleReset = () => {
    setIsPlaying(false);
    setTimestep(0);
  };

  const handleTogglePlay = () => {
    setIsPlaying((p) => !p);
  };

  useEffect(() => {
    if (isPlaying) {
      intervalRef.current = setInterval(() => {
        setTimestep((t) => {
          if (t >= MAX_HISTORY) {
            setIsPlaying(false);
            return t;
          }
          return t + 1;
        });
      }, 400);
    }
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [isPlaying]);

  // Stop playing automatically once we hit the end
  useEffect(() => {
    if (clampedTimestep >= MAX_HISTORY && isPlaying) {
      setIsPlaying(false);
    }
  }, [clampedTimestep, isPlaying]);

  const isAtDefaults = ZONES.every((z) => rainRates[z.key] === z.defaultRainMmHr);

  return (
    <div className="not-prose my-8 rounded-2xl border border-slate-200 bg-white shadow-xl overflow-hidden font-sans">
      {/* Header */}
      <div className="bg-gradient-to-r from-sky-700 to-emerald-800 px-6 py-4">
        <h3 className="text-white font-bold text-lg tracking-tight">
          One Sandbox Per Rainfall Zone
        </h3>
        <p className="text-sky-200 text-sm mt-0.5">
          OPM_PER_POLYGON = True (the default) — each tributary gets its own independent VSA
          sandbox, driven by its own gauge
        </p>
      </div>

      <div className="p-6 flex flex-col gap-6">
        {/* Intro */}
        <p className="text-sm text-slate-700">
          A single shared VSA sandbox (see the equation-builder widget above) assumes the{' '}
          <em>whole</em> basin wets up together — one <span className="font-mono">z</span>, one{' '}
          <span className="font-mono">SD_max(t)</span>, one <span className="font-mono">A_t(t)</span>{' '}
          for the entire catchment. Real catchments are split into rainfall zones (Thiessen/IDW
          polygons around each gauge), and OPM runs the <strong>exact same equations</strong> —
          Eq.10, Eq.4, Eq.12, Eq.5, Eq.9 — independently inside each zone. Below, all three zones
          share identical soil and slope parameters; only their rain rate differs. Adjust each
          slider independently, then press Step or Play to watch all three sandboxes evolve on
          the same shared clock.
        </p>

        {/* Shared parameters strip */}
        <div className="flex flex-wrap gap-2 text-xs font-mono text-slate-500">
          <span className="bg-slate-50 border border-slate-200 rounded-full px-2 py-0.5">
            A_outlet = {A_OUTLET.toLocaleString()} m²
          </span>
          <span className="bg-slate-50 border border-slate-200 rounded-full px-2 py-0.5">
            A₁ = {A1.toLocaleString()} m² (Δx = {DX} m)
          </span>
          <span className="bg-slate-50 border border-slate-200 rounded-full px-2 py-0.5">
            Q_max = {Q_MAX.toFixed(2)} m³/s
          </span>
          <span className="bg-slate-50 border border-slate-200 rounded-full px-2 py-0.5">
            SD_max⁽⁰⁾ = {SD_MAX0.toFixed(2)} m
          </span>
          <span className="bg-slate-50 border border-slate-200 rounded-full px-2 py-0.5">
            φ = {PHI.toFixed(2)}
          </span>
          <span className="bg-slate-50 border border-slate-200 rounded-full px-2 py-0.5">
            K_lat = {K_LAT.toExponential(2)} m/s (44 m/day)
          </span>
          <span className="bg-slate-50 border border-slate-200 rounded-full px-2 py-0.5">
            S_div = {S_DIV.toFixed(2)}
          </span>
          <span className="bg-slate-50 border border-slate-200 rounded-full px-2 py-0.5">
            Δt = {DT} s
          </span>
          <span className="bg-emerald-50 border border-emerald-200 text-emerald-700 rounded-full px-2 py-0.5 font-semibold">
            only P varies per zone
          </span>
        </div>

        {/* Shared animation controls */}
        <div className="bg-slate-50 border border-slate-200 rounded-lg p-3 flex flex-wrap items-center gap-2">
          <span className="text-xs font-semibold text-slate-600 mr-1">
            Shared clock (advances all 3 zones together):
          </span>
          <button
            onClick={handleReset}
            className="text-xs font-semibold px-3 py-1.5 rounded-full bg-slate-200 text-slate-700 hover:bg-slate-300 transition"
          >
            ⏮ Reset
          </button>
          <button
            onClick={handleStep}
            disabled={clampedTimestep >= MAX_HISTORY || isPlaying}
            className="text-xs font-semibold px-3 py-1.5 rounded-full bg-sky-600 text-white hover:bg-sky-700 disabled:opacity-40 disabled:cursor-not-allowed transition"
          >
            Step → (advance one Δt)
          </button>
          <button
            onClick={handleTogglePlay}
            disabled={clampedTimestep >= MAX_HISTORY && !isPlaying}
            className="text-xs font-semibold px-3 py-1.5 rounded-full bg-emerald-600 text-white hover:bg-emerald-700 disabled:opacity-40 disabled:cursor-not-allowed transition"
          >
            {isPlaying ? '⏸ Pause' : '▶ Play'}
          </button>
          <span className="text-xs text-slate-500 font-mono ml-auto">
            t = {clampedTimestep * DT}s (step {clampedTimestep} / {MAX_HISTORY})
          </span>
          {!isAtDefaults && (
            <button
              onClick={() => {
                const init: Record<string, number> = {};
                for (const z of ZONES) init[z.key] = z.defaultRainMmHr;
                setRainRates(init);
                handleReset();
              }}
              className="text-xs font-semibold px-3 py-1.5 rounded-full bg-slate-100 text-slate-500 hover:bg-slate-200 transition"
            >
              ⟲ Reset rain rates
            </button>
          )}
        </div>

        {/* Zone panels */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {ZONES.map((z) => {
            const trace = histories[z.key];
            const state = trace[clampedTimestep];
            const atHistory = trace.slice(0, clampedTimestep + 1).map((s) => s.at);
            const vsaHistory = trace.slice(0, clampedTimestep + 1).map((s) => s.vsaPct);
            return (
              <ZonePanel
                key={z.key}
                config={z}
                rainMmHr={rainRates[z.key]}
                onRainChange={(v) => setRainRates((prev) => ({ ...prev, [z.key]: v }))}
                state={state}
                atInit={atInit}
                ha={ha}
                atHistory={atHistory}
                vsaHistory={vsaHistory}
                timestep={clampedTimestep}
              />
            );
          })}
        </div>

        {/* t=0 identical-initialization callout */}
        <div className="bg-sky-50 border border-sky-200 rounded-lg p-3 text-sm text-sky-900">
          <span className="font-semibold">Notice at t=0: </span>
          all three zones start from the <strong>identical</strong> A_t⁽⁰⁾ ={' '}
          <span className="font-mono">{fmt(atInit, 0)} m²</span> and H_a ={' '}
          <span className="font-mono">{fmt(ha, 2)}</span> — because{' '}
          <span className="font-mono">A_outlet</span>, <span className="font-mono">Q_max</span>,{' '}
          and <span className="font-mono">SD_max⁽⁰⁾</span> are shared. Nothing distinguishes the
          zones yet. Press Step or Play and watch the rain rate alone pull their trajectories
          apart.
        </div>

        {/* Key insight callout */}
        <div className="bg-amber-50 border border-amber-200 rounded-lg p-4 text-sm text-amber-900">
          <p className="font-semibold mb-1">Key insight</p>
          <p>
            One tributary can be saturating fast while another stays nearly dry — a single shared
            sandbox would average these away and get <strong>both</strong> wrong. Averaging Zone
            A&apos;s 35 mm/hr downpour with Zone C&apos;s 5 mm/hr drizzle into one basin-wide rain
            rate would under-predict how saturated the heavy tributary really gets, and
            over-predict how saturated the dry one gets — exactly backwards from what either
            sub-catchment is actually doing.
          </p>
        </div>

        {/* One-line mention of divide-cell selection */}
        <p className="text-xs text-slate-400 italic">
          One more thing this widget doesn&apos;t visualize: the real model also picks each
          zone&apos;s divide cell independently — the cell with the minimum flow accumulation in
          that zone (the most headwater point), tie-broken by highest elevation.
        </p>
      </div>
    </div>
  );
}
