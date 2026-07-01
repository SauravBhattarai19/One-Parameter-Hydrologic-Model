'use client';

import React, { useMemo, useState } from 'react';

// ─────────────────────────────────────────────────────────────────────────
// Ground-truth physics, ported verbatim from docs/chapter4.tex ("SERVES:
// soil moisture from greenness") and serves_gee.py:
//
//   ET fraction = clamp(1.33 * NDVI - 0.049, 0, 1)
//   theta       = ET_fraction * (FC - WP) + WP,   clamped to [WP, FC]
//   deficit     = (porosity - theta) * Z_r         <-  this IS OPM's SD_max
//
// Illustrative defaults (clean round numbers chosen for teaching, NOT a real
// basin measurement): NDVI=0.60, FC=0.35, WP=0.15, porosity=0.45, Z_r=1.0 m
//   -> ET_fraction = 1.33*0.60 - 0.049 = 0.749
//   -> theta        = 0.749*(0.35-0.15) + 0.15 = 0.300
//   -> deficit      = (0.45 - 0.300) * 1.0 = 0.150 m
// ─────────────────────────────────────────────────────────────────────────

function clamp(x: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, x));
}

interface Chain {
  ndvi: number;
  etFraction: number;
  theta: number;
  deficit: number;
}

function computeChain(
  ndvi: number,
  fc: number,
  wp: number,
  porosity: number,
  zr: number
): Chain {
  const etFraction = clamp(1.33 * ndvi - 0.049, 0, 1);
  const thetaRaw = etFraction * (fc - wp) + wp;
  const theta = clamp(thetaRaw, wp, fc);
  const deficit = (porosity - theta) * zr;
  return { ndvi, etFraction, theta, deficit };
}

// ─────────────────────────────────────────────────────────────────────────
// Small presentational pieces
// ─────────────────────────────────────────────────────────────────────────

interface ChainBoxProps {
  title: string;
  subtitle: string;
  value: string;
  unit?: string;
  accent: string; // tailwind color stem, e.g. "emerald"
}

function chainBoxClasses(accent: string): { box: string; title: string; value: string } {
  const map: Record<string, { box: string; title: string; value: string }> = {
    emerald: {
      box: 'bg-emerald-50 border-emerald-300',
      title: 'text-emerald-700',
      value: 'text-emerald-800',
    },
    teal: {
      box: 'bg-teal-50 border-teal-300',
      title: 'text-teal-700',
      value: 'text-teal-800',
    },
    sky: {
      box: 'bg-sky-50 border-sky-300',
      title: 'text-sky-700',
      value: 'text-sky-800',
    },
    indigo: {
      box: 'bg-indigo-50 border-indigo-300',
      title: 'text-indigo-700',
      value: 'text-indigo-800',
    },
  };
  return map[accent] ?? map.sky;
}

function ChainBox({ title, subtitle, value, unit, accent }: ChainBoxProps) {
  const cls = chainBoxClasses(accent);
  return (
    <div
      className={`flex-1 min-w-[140px] rounded-xl border-2 ${cls.box} px-3 py-3 flex flex-col items-center text-center gap-1 shadow-sm`}
    >
      <div className={`text-[11px] font-bold uppercase tracking-wide ${cls.title}`}>{title}</div>
      <div className={`text-2xl font-mono font-extrabold tabular-nums ${cls.value}`}>
        {value}
        {unit && <span className="text-sm font-semibold ml-1">{unit}</span>}
      </div>
      <div className="text-xs text-slate-500 leading-snug">{subtitle}</div>
    </div>
  );
}

function Arrow() {
  return (
    <div className="flex items-center justify-center px-1 text-slate-400 select-none shrink-0">
      <svg width="28" height="20" viewBox="0 0 28 20" className="rotate-90 sm:rotate-0">
        <line x1="1" y1="10" x2="23" y2="10" stroke="currentColor" strokeWidth={2} />
        <polygon points="22,5 28,10 22,15" fill="currentColor" />
      </svg>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────
// Root widget
// ─────────────────────────────────────────────────────────────────────────
export default function SatelliteChainWidget() {
  const [ndvi, setNdvi] = useState(0.6);

  // Illustrative soil parameters — held fixed at the defaults used throughout
  // the worked example in the chapter text.
  const FC = 0.35;
  const WP = 0.15;
  const POROSITY = 0.45;
  const Z_R = 1.0;

  const chain = useMemo(
    () => computeChain(ndvi, FC, WP, POROSITY, Z_R),
    [ndvi]
  );

  return (
    <div className="not-prose my-8 rounded-2xl border border-slate-200 bg-white shadow-xl overflow-hidden font-sans">
      {/* Header */}
      <div className="bg-gradient-to-r from-emerald-700 to-teal-900 px-6 py-4">
        <h3 className="text-white font-bold text-lg tracking-tight">
          From Satellite Greenness to SD_max: The SERVES Chain
        </h3>
        <p className="text-emerald-200 text-sm mt-0.5">
          serves_gee.py — turning a satellite NDVI pixel into OPM&apos;s antecedent soil-moisture
          deficit, with no calibration knob
        </p>
      </div>

      <div className="p-6 flex flex-col gap-6">
        <div className="bg-amber-50 border border-amber-200 rounded-lg p-3 text-sm text-amber-900">
          <span className="font-semibold">Why satellites instead of a calibrated parameter: </span>
          Most rainfall-runoff models ask the user to guess or calibrate an antecedent-wetness
          parameter — a knob tuned to past storms that may not transfer to the next storm or the
          next basin. OPM instead reads the <em>actual</em> vegetation greenness, soil type, and
          land cover for the specific date and place from satellite imagery (Landsat by default),
          the same way you would check a weather map before a hike.
        </div>

        {/* Slider */}
        <div className="flex flex-col gap-1 max-w-md">
          <div className="flex items-center justify-between">
            <label className="text-sm font-semibold text-slate-700">
              NDVI (vegetation greenness)
            </label>
            <span className="text-sm font-mono font-bold text-emerald-700 tabular-nums">
              {chain.ndvi.toFixed(2)}
            </span>
          </div>
          <input
            type="range"
            min={0}
            max={0.9}
            step={0.01}
            value={ndvi}
            onChange={(e) => setNdvi(Number(e.target.value))}
            className="w-full h-1.5 rounded-full accent-emerald-600 cursor-pointer"
          />
          <div className="flex justify-between text-[10px] text-slate-400 font-mono">
            <span>0.0 (bare soil)</span>
            <span>~0.9 (dense vegetation)</span>
          </div>
        </div>

        {/* The live chain */}
        <div>
          <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2">
            Live-computed chain
          </p>
          <div className="flex flex-col sm:flex-row items-stretch gap-1 sm:gap-0">
            <ChainBox
              title="NDVI"
              subtitle="satellite greenness index"
              value={chain.ndvi.toFixed(2)}
              accent="emerald"
            />
            <Arrow />
            <ChainBox
              title="ET fraction"
              subtitle="clamp(1.33·NDVI − 0.049, 0, 1)"
              value={chain.etFraction.toFixed(3)}
              accent="teal"
            />
            <Arrow />
            <ChainBox
              title="θ (soil moisture)"
              subtitle="ET_fraction·(FC−WP) + WP"
              value={chain.theta.toFixed(3)}
              accent="sky"
            />
            <Arrow />
            <ChainBox
              title="Deficit = SD_max"
              subtitle="(porosity − θ)·Z_r"
              value={chain.deficit.toFixed(3)}
              unit="m"
              accent="indigo"
            />
          </div>
        </div>

        {/* Illustrative inputs readout */}
        <div className="bg-slate-50 border border-slate-200 rounded-lg p-3 text-xs font-mono text-slate-600">
          <span className="font-sans font-semibold text-slate-500 not-italic">
            Illustrative values
          </span>{' '}
          (clean round numbers chosen for teaching, not a real basin measurement) — held fixed
          while you move the slider:
          <br />
          FC (field capacity) = {FC.toFixed(2)} &nbsp;·&nbsp; WP (wilting point) = {WP.toFixed(2)}{' '}
          &nbsp;·&nbsp; porosity = {POROSITY.toFixed(2)} &nbsp;·&nbsp; Z_r (root-zone depth) ={' '}
          {Z_R.toFixed(1)} m
          <br />
          FC and WP come from SoilGrids; porosity from HiHydroSoil v2.0; Z_r from a land-cover
          lookup table.
        </div>

        {/* Worked-example check at default */}
        <div className="bg-white border border-slate-200 rounded-lg p-3 text-xs text-slate-600">
          <span className="font-semibold text-slate-700">Check at the default NDVI = 0.60: </span>
          ET fraction = 1.33×0.60 − 0.049 = <strong>0.749</strong> &nbsp;→&nbsp; θ =
          0.749×(0.35−0.15) + 0.15 = <strong>0.300</strong> &nbsp;→&nbsp; deficit = (0.45−0.300)×1.0
          = <strong>0.150 m</strong>. Move the slider — every box recomputes instantly.
        </div>

        {/* Closing takeaway callout */}
        <div className="bg-gradient-to-r from-emerald-50 to-indigo-50 border-2 border-emerald-300 rounded-xl p-4 text-sm text-slate-800">
          <span className="font-bold text-emerald-800">The chain in one breath: </span>
          Greener vegetation &rArr; higher NDVI &rArr; wetter soil (&theta; near FC) &rArr; smaller
          deficit &rArr; SD_max shrinks &rArr; a wetter antecedent state &rArr; (in the next
          section&apos;s VSA model) a larger initial saturated area and more runoff from the very
          first drops of a storm. Every link in this chain is physically measured, not guessed.
        </div>
      </div>
    </div>
  );
}
