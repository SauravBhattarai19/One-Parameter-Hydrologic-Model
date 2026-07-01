'use client';

import React, { useMemo, useState } from 'react';

// ── OPM production defaults (verified against the real source) ───────────────
// config.py: CELL_SIZE = None  → auto-detected from dem_100m.tif  →  ≈100 m
// config.py: TIME_STEP_SECONDS = 0.7
const DX = 100; // m
const DEFAULT_DT = 0.7; // s

// Chapter 4's already-failing "Too Small Δt" demo ran at this Courant number.
const CH4_TOO_SMALL_C = 0.17;

type Zone = {
  key: string;
  label: string;
  n: number;
  h: number;
  S0: number;
  color: string; // tailwind-ish hex for badges/SVG
};

// n = 0.03 reproduces the course's verified worked example exactly
// (V=18.26, c=30.43, C=0.213, D_num=1197, D_phys=30.4 @ dt=0.7s).
const ZONES: Zone[] = [
  { key: 'steep', label: 'Steep headwater (sets the global Δt)', n: 0.03, h: 1.0, S0: 0.3, color: '#dc2626' },
  { key: 'moderate', label: 'Moderate midslope', n: 0.05, h: 0.5, S0: 0.01, color: '#f59e0b' },
  { key: 'flat', label: 'Flat Terai valley floor', n: 0.09, h: 0.3, S0: 0.0005, color: '#2563eb' },
];

function computeZone(z: Zone, dt: number) {
  const V = (1 / z.n) * Math.pow(z.h, 2 / 3) * Math.sqrt(z.S0);
  const c = (5 / 3) * V;
  const C = (c * dt) / DX;
  const D_num = ((c * DX) / 2) * (1 - C);
  const D_phys = (V * z.h) / (2 * z.S0);
  const ratioPct = D_phys !== 0 ? (D_num / D_phys) * 100 : 0;
  return { V, c, C, D_num, D_phys, ratioPct };
}

function courantColor(C: number): { text: string; bg: string; border: string; label: string } {
  if (C > 0.5) return { text: 'text-green-700', bg: 'bg-green-50', border: 'border-green-200', label: 'near-ideal' };
  if (C >= 0.05) return { text: 'text-amber-700', bg: 'bg-amber-50', border: 'border-amber-200', label: 'sub-optimal' };
  return { text: 'text-red-700', bg: 'bg-red-50', border: 'border-red-200', label: 'danger zone' };
}

function fmt(x: number, digits = 3): string {
  if (!Number.isFinite(x)) return '—';
  if (Math.abs(x) >= 100) return x.toFixed(1);
  if (Math.abs(x) >= 1) return x.toFixed(2);
  return x.toFixed(digits);
}

export default function GlobalDtTyrannyWidget() {
  const [dt, setDt] = useState(DEFAULT_DT);

  const results = useMemo(() => {
    const r: Record<string, ReturnType<typeof computeZone>> = {};
    for (const z of ZONES) r[z.key] = computeZone(z, dt);
    return r;
  }, [dt]);

  const steep = results.steep;
  const moderate = results.moderate;
  const flat = results.flat;

  const cSteepOverCFlat = flat.C !== 0 ? steep.C / flat.C : Infinity;
  const ch4OverCFlat = flat.C !== 0 ? CH4_TOO_SMALL_C / flat.C : Infinity;

  return (
    <div className="not-prose my-8 rounded-2xl border border-slate-200 bg-white shadow-xl overflow-hidden font-sans">
      {/* Header */}
      <div className="bg-gradient-to-r from-sky-700 to-blue-900 px-6 py-4">
        <h3 className="text-white font-bold text-lg tracking-tight">
          One Global Δt, Many Slopes: The Hidden Cost
        </h3>
        <p className="text-sky-200 text-sm mt-0.5">
          OPM sizes Δt once, off the steepest cell — every gentler cell pays for it
        </p>
      </div>

      <div className="p-6 flex flex-col gap-6">
        {/* Watershed schematic */}
        <div>
          <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2">
            One Watershed, One Δt — Three Very Different Slopes
          </p>
          <svg viewBox="0 0 600 160" className="w-full h-auto" style={{ maxHeight: 180 }}>
            {/* sky */}
            <rect x={0} y={0} width={600} height={160} fill="#f8fafc" />
            {/* river profile: steep (left) -> moderate (middle) -> flat (right) */}
            <polygon
              points="0,150 0,40 140,95 320,128 600,138 600,150"
              fill="#dbeafe"
              opacity={0.6}
            />
            <polyline
              points="0,40 140,95 320,128 600,138"
              fill="none"
              stroke="#1e3a8a"
              strokeWidth={2.5}
            />
            {/* zone bands */}
            <rect x={0} y={0} width={200} height={160} fill="#dc2626" opacity={0.06} />
            <rect x={200} y={0} width={200} height={160} fill="#f59e0b" opacity={0.06} />
            <rect x={400} y={0} width={200} height={160} fill="#2563eb" opacity={0.06} />
            {/* labels */}
            <text x={100} y={20} textAnchor="middle" fontSize={11} fontWeight={700} fill="#dc2626">
              Steep headwater
            </text>
            <text x={100} y={33} textAnchor="middle" fontSize={9} fill="#991b1b">
              S₀ = 0.30
            </text>
            <text x={300} y={20} textAnchor="middle" fontSize={11} fontWeight={700} fill="#b45309">
              Moderate midslope
            </text>
            <text x={300} y={33} textAnchor="middle" fontSize={9} fill="#92400e">
              S₀ = 0.01
            </text>
            <text x={500} y={20} textAnchor="middle" fontSize={11} fontWeight={700} fill="#1d4ed8">
              Flat valley floor
            </text>
            <text x={500} y={33} textAnchor="middle" fontSize={9} fill="#1e40af">
              S₀ = 0.0005
            </text>
            {/* dividing lines */}
            <line x1={200} y1={0} x2={200} y2={160} stroke="#cbd5e1" strokeDasharray="3,3" />
            <line x1={400} y1={0} x2={400} y2={160} stroke="#cbd5e1" strokeDasharray="3,3" />
            {/* one shared dt note */}
            <text x={300} y={155} textAnchor="middle" fontSize={10} fill="#475569" fontStyle="italic">
              ← one global Δt, computed from the steepest cell, applied everywhere →
            </text>
          </svg>
        </div>

        {/* Shared dt slider */}
        <div className="bg-slate-50 border border-slate-200 rounded-lg p-4">
          <label className="flex items-center justify-between text-sm font-semibold text-slate-700 mb-1">
            <span>Global Δt (shared by all three zones, just like OPM)</span>
            <span className="font-mono text-sky-700">{dt.toFixed(2)} s</span>
          </label>
          <input
            type="range"
            min={0.1}
            max={3.0}
            step={0.01}
            value={dt}
            onChange={(e) => setDt(+e.target.value)}
            className="w-full"
          />
          <div className="text-xs text-slate-400 flex justify-between mt-1">
            <span>0.1 s</span>
            <span>OPM default: 0.7 s</span>
            <span>3.0 s</span>
          </div>
        </div>

        {/* Zone cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {ZONES.map((z) => {
            const r = results[z.key];
            const cc = courantColor(r.C);
            return (
              <div
                key={z.key}
                className="rounded-xl border border-slate-200 overflow-hidden flex flex-col"
              >
                <div
                  className="px-3 py-2 text-white text-xs font-bold"
                  style={{ backgroundColor: z.color }}
                >
                  {z.label}
                </div>
                <div className="p-3 flex flex-col gap-2">
                  <div className="bg-slate-50 font-mono text-xs p-3 rounded grid grid-cols-2 gap-y-1">
                    <span className="text-slate-500">n</span>
                    <span className="text-right">{z.n}</span>
                    <span className="text-slate-500">h</span>
                    <span className="text-right">{z.h} m</span>
                    <span className="text-slate-500">S₀</span>
                    <span className="text-right">{z.S0}</span>
                    <span className="text-slate-500">V</span>
                    <span className="text-right">{fmt(r.V)} m/s</span>
                    <span className="text-slate-500">c</span>
                    <span className="text-right">{fmt(r.c)} m/s</span>
                    <span className="text-slate-500">D_num</span>
                    <span className="text-right">{fmt(r.D_num)} m²/s</span>
                    <span className="text-slate-500">D_phys</span>
                    <span className="text-right">{fmt(r.D_phys)} m²/s</span>
                  </div>
                  <div className={`rounded-lg border p-2 ${cc.bg} ${cc.border}`}>
                    <div className={`text-xs font-semibold ${cc.text}`}>
                      C = {r.C.toFixed(4)} ({cc.label})
                    </div>
                    <div className="text-[11px] text-slate-600 mt-0.5">
                      D_num / D_phys ={' '}
                      <span className={`font-mono font-semibold ${cc.text}`}>
                        {r.ratioPct.toFixed(1)}%
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        {/* Live callout 1: C ratio steep vs flat */}
        <div className="bg-amber-50 border border-amber-200 rounded-lg p-3 text-sm text-amber-900">
          At Δt = {dt.toFixed(2)} s, the Courant number at the flat valley-floor zone (C ={' '}
          {flat.C.toFixed(4)}) is{' '}
          <span className="font-bold">
            {Number.isFinite(cSteepOverCFlat) ? cSteepOverCFlat.toFixed(0) : '—'}× smaller
          </span>{' '}
          than at the steep headwater zone (C = {steep.C.toFixed(4)}) — the very zone that
          determined this Δt in the first place.
        </div>

        {/* Live callout 2: comparison to Chapter 4 demo */}
        <div className="bg-red-50 border border-red-200 rounded-lg p-3 text-sm text-red-900">
          Chapter 4&apos;s animated Δt-sensitivity demo showed a <strong>&ldquo;Too Small Δt&rdquo;</strong>{' '}
          lane at C = {CH4_TOO_SMALL_C.toFixed(2)} already producing severe numerical damping of
          a flood peak. Here, the real flat-valley cell&apos;s Courant number (C = {flat.C.toFixed(4)}) is{' '}
          <span className="font-bold">
            {Number.isFinite(ch4OverCFlat) ? ch4OverCFlat.toFixed(0) : '—'}× smaller
          </span>{' '}
          than that already-failing case — OPM&apos;s real valley-floor cells run far deeper into
          the numerical-diffusion danger zone than even that demo&apos;s worst case.
        </div>

        {/* Closing callout: contamination */}
        <div className="bg-green-50 border border-green-200 rounded-lg p-4 text-sm text-green-900">
          <p className="font-semibold mb-1">
            What this means: roughly{' '}
            <span className="font-mono">{flat.ratioPct.toFixed(1)}%</span> of the flat-zone
            diffusivity is numerical, not physical.
          </p>
          <p>
            D_num and D_phys both feed the exact same diffusive-wave equation, so the model
            cannot tell them apart at run time. At the flat valley floor, the upwind scheme&apos;s
            own numerical smoothing (D_num ≈ {fmt(flat.D_num)} m²/s) is already{' '}
            {flat.ratioPct.toFixed(0)}% as large as the physical backwater diffusivity the
            diffusive-wave scheme was built to capture (D_phys ≈ {fmt(flat.D_phys)} m²/s). That
            means a meaningful fraction of the attenuation and smoothing visible in a simulated
            valley-floor hydrograph could be a grid/timestep artifact rather than real backwater
            physics — and OPM currently has no way to report which fraction is which, because
            both effects enter the same equation and only their sum is ever computed.
          </p>
        </div>

        {/* Real code evidence */}
        <div>
          <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2">
            The Real CFL Self-Check — kinematic_wave_router.py
          </p>
          <pre className="bg-slate-50 font-mono text-xs p-3 rounded overflow-x-auto whitespace-pre">
{`max_slope         = float(slope_1d.max().item())
n_min             = float(n.min().item()) if hasattr(n, 'min') else float(n)
V_at_1m_max_slope = (1.0 / n_min) * (1.0 ** (2.0 / 3.0)) * (max_slope ** 0.5)
C_indicator       = V_at_1m_max_slope * dt / dx
safe_dt           = dx / V_at_1m_max_slope`}
          </pre>
          <p className="text-xs text-slate-500 mt-2">
            This check runs once, on the <strong>static bed-slope grid</strong>, using the single
            steepest cell (<code>slope_1d.max()</code>) and the smallest roughness
            (<code>n.min()</code>) anywhere in the watershed — and it runs identically regardless
            of <code>ROUTING_SCHEME</code>. Diffusive routing inherits the kinematic CFL&apos;s
            global Δt; it does not get its own.
          </p>
        </div>
      </div>
    </div>
  );
}
