'use client';

import React, { useState, useMemo } from 'react';

// ─────────────────────────────────────────────────────────────────────────
// AdaptiveDtWidget — global vs local time stepping, and the physics it costs.
//
// Adaptive Δt sizes the step from the flow: Δt = C_target·Δx / c_max.
//   GLOBAL: one Δt for the whole grid, from the single fastest cell.  Every
//           slower cell then runs at C_i = C_target·(c_i/c_max) ≪ C_target,
//           so by §4.4 it suffers numerical diffusion α=(cΔx/2)(1−C_i) with
//           the largest possible (1−C) factor → flat reaches over-damp the
//           peak, unphysically.  GPU-ideal (SIMT lockstep) — why OPM uses it.
//   LOCAL : each cell steps at its own Δt = C_target·Δx/c_i → C_i = C_target
//           everywhere → uniform, minimal numerical diffusion.  CPU-natural,
//           GPU-hard (warp divergence, load imbalance).
//
// Smear a peak picks up crossing a reach of length Δx:  σ² = Δx²·(1−C),
// independent of c — so it is the (1−C) factor, set by the global Δt, that
// decides how badly each reach over-damps.
// ─────────────────────────────────────────────────────────────────────────

const C_TARGET = 0.85;
const N_MAN = 0.04, H = 1.0, DX = 500; // m

const REACHES: { S0: number; tag: string }[] = [
  { S0: 0.05, tag: 'steep headwater' },
  { S0: 0.02, tag: '' },
  { S0: 0.008, tag: '' },
  { S0: 0.003, tag: '' },
  { S0: 0.001, tag: '' },
  { S0: 0.0004, tag: 'flat lowland' },
];

function celerity(S0: number): number {
  const V = (1 / N_MAN) * Math.pow(H, 2 / 3) * Math.sqrt(S0);
  return (5 / 3) * V;
}

function slopeColor(S0: number): string {
  const t = Math.max(0, Math.min(1, (Math.log10(S0) + 3.5) / 2.5)); // ~0 flat → 1 steep
  return `rgb(${Math.round(37 + t * 200)},${Math.round(99 + t * 20)},${Math.round(235 - t * 200)})`;
}
function diffColor(d: number): string {
  if (d <= 0.2) return '#16a34a';
  if (d <= 0.5) return '#d97706';
  return '#dc2626';
}

export default function AdaptiveDtWidget() {
  const [mode, setMode] = useState<'global' | 'local'>('global');

  const data = useMemo(() => {
    const c = REACHES.map((r) => celerity(r.S0));
    const cMax = Math.max(...c);
    const cMin = Math.min(...c);
    const sumC = c.reduce((a, b) => a + b, 0);

    const rows = REACHES.map((r, i) => {
      const Cg = C_TARGET * (c[i] / cMax);
      const Cval = mode === 'global' ? Cg : C_TARGET;
      const dtLocal = (C_TARGET * DX) / c[i];
      return {
        ...r, c: c[i], Cglobal: Cg, C: Cval,
        diff: 1 - Cval, dtLocal,
        oversample: cMax / c[i], // how many ×'s this reach is over-stepped under global
      };
    });

    const dtGlobal = (C_TARGET * DX) / cMax;
    // accumulated numerical smearing variance, in units of Δx² (σ₀ = 2Δx pulse)
    const smear = rows.reduce((a, r) => a + r.diff, 0);
    const retentionOf = (s: number) => 2 / Math.sqrt(4 + s);
    const smearGlobal = rows.reduce((a, r) => a + (1 - r.Cglobal), 0);
    const smearLocal = REACHES.length * (1 - C_TARGET);

    return {
      rows, cMax, cMin, dtGlobal,
      retention: retentionOf(smear),
      retGlobal: retentionOf(smearGlobal),
      retLocal: retentionOf(smearLocal),
      // total cell-updates: global = K·(T/dtGlobal); local = Σ T/dt_i ∝ Σ c_i
      workRatio: (REACHES.length * cMax) / sumC, // global ÷ local
    };
  }, [mode]);

  // hydrograph: area-preserving Gaussians (amp = retention, σ = 2/retention)
  const HW = 300, HH = 130;
  const gauss = (amp: number, sig: number) => {
    const pts: string[] = [];
    for (let k = 0; k <= 60; k++) {
      const x = -6 + (12 * k) / 60;
      const q = amp * Math.exp(-(x * x) / (2 * sig * sig));
      const sx = 8 + ((x + 6) / 12) * (HW - 16);
      const sy = HH - 14 - q * (HH - 28);
      pts.push(`${k === 0 ? 'M' : 'L'}${sx.toFixed(1)},${sy.toFixed(1)}`);
    }
    return pts.join(' ');
  };
  const curRet = mode === 'global' ? data.retGlobal : data.retLocal;

  // channel SVG
  const CW = 540, CH = 150, M = 16;
  const cellW = (CW - 2 * M) / REACHES.length;

  return (
    <div className="not-prose my-6 rounded-2xl border border-slate-200 bg-white shadow-sm overflow-hidden">
      <div className="bg-gradient-to-r from-amber-600 to-rose-600 px-5 py-3">
        <h3 className="text-white font-bold text-base">One clock for the whole river — or one each?</h3>
        <p className="text-amber-50 text-xs mt-0.5">
          A global Δt is set by the single fastest cell. Watch what that does to the slow, flat reaches.
        </p>
      </div>

      {/* mode toggle */}
      <div className="px-5 pt-4 flex items-center gap-3">
        <div className="inline-flex rounded-lg border border-slate-300 overflow-hidden text-sm font-semibold">
          <button onClick={() => setMode('global')}
            className={`px-4 py-1.5 ${mode === 'global' ? 'bg-amber-600 text-white' : 'bg-white text-slate-600 hover:bg-slate-50'}`}>
            Global Δt
          </button>
          <button onClick={() => setMode('local')}
            className={`px-4 py-1.5 ${mode === 'local' ? 'bg-emerald-600 text-white' : 'bg-white text-slate-600 hover:bg-slate-50'}`}>
            Local Δt (per cell)
          </button>
        </div>
        <span className="text-xs text-slate-500">
          {mode === 'global'
            ? `one Δt = ${data.dtGlobal.toFixed(1)} s for all 6 reaches (from the steep cell)`
            : `each reach gets its own Δt = C·Δx/cᵢ`}
        </span>
      </div>

      {/* channel: reaches + per-cell C + numerical-diffusion bars */}
      <div className="px-5 pt-3">
        <svg viewBox={`0 0 ${CW} ${CH}`} className="w-full select-none">
          {data.rows.map((r, i) => {
            const x = M + i * cellW;
            return (
              <g key={i}>
                {/* reach block, colour = slope */}
                <rect x={x + 2} y={16} width={cellW - 4} height={30} rx={4} fill={slopeColor(r.S0)} opacity={0.85} />
                <text x={x + cellW / 2} y={35} textAnchor="middle" fontSize={9} className="fill-white font-semibold">
                  S₀={r.S0}
                </text>
                {/* Courant readout */}
                <text x={x + cellW / 2} y={62} textAnchor="middle" fontSize={11} className="font-mono font-bold"
                  style={{ fill: diffColor(r.diff) }}>
                  C={r.C.toFixed(2)}
                </text>
                {/* numerical diffusion bar (length ∝ 1−C) */}
                <rect x={x + 6} y={72} width={cellW - 12} height={12} rx={2} fill="#f1f5f9" />
                <rect x={x + 6} y={72} width={(cellW - 12) * r.diff} height={12} rx={2} fill={diffColor(r.diff)} />
                <text x={x + cellW / 2} y={100} textAnchor="middle" fontSize={8} className="fill-slate-400">
                  (1−C)={r.diff.toFixed(2)}
                </text>
                {r.tag && (
                  <text x={x + cellW / 2} y={118} textAnchor="middle" fontSize={8} className="fill-slate-500">{r.tag}</text>
                )}
                {/* oversample factor under global */}
                {mode === 'global' && r.oversample > 1.5 && (
                  <text x={x + cellW / 2} y={132} textAnchor="middle" fontSize={8} className="fill-rose-500">
                    {r.oversample.toFixed(0)}× over-stepped
                  </text>
                )}
              </g>
            );
          })}
          <text x={M} y={146} fontSize={8} className="fill-slate-400">flow →</text>
        </svg>
        <div className="text-[11px] text-slate-500 -mt-1">
          bar = numerical diffusion <span className="font-mono">α=(cΔx/2)(1−C)</span> per reach ·{' '}
          <span className="text-emerald-600">green low</span> / <span className="text-red-600">red high</span>
        </div>
      </div>

      {/* hydrograph + stats */}
      <div className="px-5 pt-3 grid md:grid-cols-[auto_1fr] gap-5">
        <div>
          <div className="text-xs font-bold text-slate-500 uppercase mb-1">peak after the 6 reaches</div>
          <svg viewBox={`0 0 ${HW} ${HH}`} className="w-full rounded-lg bg-slate-50" style={{ maxWidth: 320 }}>
            <line x1={8} y1={HH - 14} x2={HW - 8} y2={HH - 14} stroke="#cbd5e1" strokeWidth={1} />
            {/* true (no numerical diffusion) */}
            <path d={gauss(1, 2)} fill="none" stroke="#94a3b8" strokeWidth={1.5} strokeDasharray="4 3" />
            {/* the other mode, faint */}
            <path d={gauss(mode === 'global' ? data.retLocal : data.retGlobal, 2 / (mode === 'global' ? data.retLocal : data.retGlobal))}
              fill="none" stroke={mode === 'global' ? '#10b981' : '#f59e0b'} strokeWidth={1.5} opacity={0.4} />
            {/* current mode */}
            <path d={gauss(curRet, 2 / curRet)} fill="none" stroke={mode === 'global' ? '#d97706' : '#16a34a'} strokeWidth={2.5} />
          </svg>
          <div className="flex flex-wrap gap-x-3 gap-y-0.5 text-[10px] mt-1">
            <span className="flex items-center gap-1"><span className="w-3 h-0.5 inline-block" style={{ borderTop: '1.5px dashed #94a3b8' }} /> true (kinematic, 100%)</span>
            <span className="flex items-center gap-1"><span className="w-3 h-0.5 bg-amber-600 inline-block" /> global {Math.round(data.retGlobal * 100)}%</span>
            <span className="flex items-center gap-1"><span className="w-3 h-0.5 bg-emerald-600 inline-block" /> local {Math.round(data.retLocal * 100)}%</span>
          </div>
        </div>

        <div className="space-y-2 text-[13px]">
          <div className="rounded-xl border p-3"
            style={{ borderColor: mode === 'global' ? '#fcd34d' : '#6ee7b7', background: mode === 'global' ? '#fffbeb' : '#ecfdf5' }}>
            <div className="font-bold mb-1" style={{ color: mode === 'global' ? '#b45309' : '#047857' }}>
              {mode === 'global' ? 'Global Δt — the hidden tax' : 'Local Δt — uniform & physical'}
            </div>
            {mode === 'global' ? (
              <ul className="space-y-0.5 text-slate-700 list-disc pl-4">
                <li>flat reach runs at C={data.rows[5].C.toFixed(2)} — <b>{(data.rows[5].diff / data.rows[0].diff).toFixed(1)}×</b> the numerical diffusion of the steep cell that set Δt</li>
                <li>peak kept: <b>{Math.round(data.retGlobal * 100)}%</b> — yet the true kinematic peak is 100%, so <b>{Math.round((1 - data.retGlobal) * 100)}%</b> of the loss is a grid artifact, not physics</li>
                <li>flat reach is integrated <b>{data.rows[5].oversample.toFixed(0)}×</b> more often than its accuracy needs</li>
              </ul>
            ) : (
              <ul className="space-y-0.5 text-slate-700 list-disc pl-4">
                <li>every reach at C={C_TARGET} → the smallest (1−C)={(1 - C_TARGET).toFixed(2)} the scheme allows, everywhere</li>
                <li>peak kept: <b>{Math.round(data.retLocal * 100)}%</b> — the residual {Math.round((1 - data.retLocal) * 100)}% is the irreducible C={C_TARGET} floor (push C→1 to erase it)</li>
                <li>global does <b>{data.workRatio.toFixed(1)}×</b> the cell-updates local does — local is faster <em>and</em> sharper</li>
              </ul>
            )}
          </div>

          <div className="rounded-xl border border-slate-200 bg-slate-50 p-3">
            <div className="text-[11px] font-bold text-slate-500 uppercase mb-1">on hardware</div>
            {mode === 'global' ? (
              <p className="text-slate-700"><span className="text-emerald-700 font-semibold">✓ GPU-ideal.</span> Every cell does the identical update at the identical time — perfect SIMT lockstep, one kernel. This is exactly why OPM uses it.</p>
            ) : (
              <p className="text-slate-700"><span className="text-amber-700 font-semibold">⚠ GPU-hard.</span> Per-cell Δt means warp divergence + load imbalance. CPU handles it naturally; on GPU you fall back to <b>block/class</b> LTS (bin cells into power-of-2 Δt groups, sub-cycle the fast ones).</p>
            )}
          </div>
        </div>
      </div>

      {/* cures */}
      <div className="px-5 py-4">
        <div className="rounded-xl border border-indigo-200 bg-indigo-50 p-3 text-[12.5px] text-indigo-900">
          <span className="font-bold">Keeping it physical without a rewrite:</span>{' '}
          <b>(1)</b> local/block time stepping → uniform C; <b>(2)</b> a higher-order / TVD flux (Lax–Wendroff, MUSCL)
          kills the O(Δx)(1−C) term and <em>keeps the GPU-friendly global Δt</em>; <b>(3)</b> Muskingum–Cunge —
          tune the numerical diffusion to <em>equal</em> the physical D=Q/(2BS₀) so it stops being an artifact.
        </div>
      </div>
    </div>
  );
}
