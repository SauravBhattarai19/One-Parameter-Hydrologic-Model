'use client';

import React, { useState, useMemo } from 'react';

// ─────────────────────────────────────────────────────────────────────────
// Two mechanisms, ported conceptually from the OPM VSA + Green-Ampt model:
//
//   Dunne (saturation-excess):
//     a cell is either IN the variable source area or not — binary switch.
//     saturation < 100%  → runoff = 0% (all infiltrates, soil has room)
//     saturation = 100%  → runoff = 100% of rain, REGARDLESS of rain rate
//
//   Horton (infiltration-excess):
//     runoff = max(rain_rate - infiltration_capacity, 0) / rain_rate
//     rain_rate <= capacity → runoff = 0% (surface keeps up)
//     rain_rate >  capacity → excess sheds, growing with intensity
// ─────────────────────────────────────────────────────────────────────────

const SVG_W = 220;
const SVG_H = 200;

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

// ─────────────────────────────────────────────────────────────────────────
// Left panel: sponge-in-a-tray diagram for Dunne saturation-excess
// ─────────────────────────────────────────────────────────────────────────
function SpongeDiagram({
  saturation,
  rainRate,
  isFull,
}: {
  saturation: number;
  rainRate: number;
  isFull: boolean;
}) {
  const padB = 20;
  const padT = 14;
  const trayTop = SVG_H - padB - 20;
  const trayBottom = SVG_H - padB;
  const spongeLeft = 40;
  const spongeRight = SVG_W - 40;
  const spongeTop = padT + 20;
  const spongeBottom = trayTop;
  const spongeHeight = spongeBottom - spongeTop;

  // Water level inside the sponge rises with saturation (0 = bottom, 1 = rim)
  const fillY = spongeBottom - (saturation / 100) * spongeHeight;

  // Number of rain drops scales loosely with rain rate, purely decorative
  const nDrops = 3 + Math.round((rainRate / 100) * 4);

  return (
    <svg
      width={SVG_W}
      height={SVG_H}
      className="block overflow-visible mx-auto"
      style={{ maxWidth: '100%' }}
    >
      {/* Rain drops falling onto the sponge */}
      {Array.from({ length: nDrops }).map((_, i) => {
        const x = spongeLeft + 10 + (i * (spongeRight - spongeLeft - 20)) / Math.max(nDrops - 1, 1);
        return (
          <line
            key={i}
            x1={x}
            y1={padT - 8}
            x2={x}
            y2={padT + 10}
            stroke="#0ea5e9"
            strokeWidth={2}
            strokeLinecap="round"
            opacity={0.8}
          />
        );
      })}
      <text x={SVG_W / 2} y={padT - 12} textAnchor="middle" fontSize={9} fill="#0369a1">
        rain
      </text>

      {/* Tray (the water table reservoir below) */}
      <rect
        x={spongeLeft - 14}
        y={trayTop}
        width={spongeRight - spongeLeft + 28}
        height={trayBottom - trayTop}
        fill="#bae6fd"
        stroke="#0284c7"
        strokeWidth={1.5}
        rx={3}
      />
      <text
        x={SVG_W / 2}
        y={trayBottom + 14}
        textAnchor="middle"
        fontSize={8.5}
        fill="#0369a1"
      >
        water table
      </text>

      {/* Sponge outline */}
      <rect
        x={spongeLeft}
        y={spongeTop}
        width={spongeRight - spongeLeft}
        height={spongeHeight}
        fill="none"
        stroke="#78716c"
        strokeWidth={2}
        rx={4}
      />

      {/* Saturation fill inside the sponge, rising from the bottom */}
      <rect
        x={spongeLeft}
        y={clamp(fillY, spongeTop, spongeBottom)}
        width={spongeRight - spongeLeft}
        height={clamp(spongeBottom - fillY, 0, spongeHeight)}
        fill={isFull ? '#0369a1' : '#38bdf8'}
        opacity={0.75}
        rx={3}
        style={{ transition: 'y 0.25s ease, height 0.25s ease, fill 0.25s ease' }}
      />

      {/* Rim line — the threshold */}
      <line
        x1={spongeLeft - 4}
        y1={spongeTop}
        x2={spongeRight + 4}
        y2={spongeTop}
        stroke={isFull ? '#dc2626' : '#94a3b8'}
        strokeWidth={isFull ? 2.5 : 1}
        strokeDasharray={isFull ? undefined : '3,3'}
      />
      <text
        x={spongeRight + 6}
        y={spongeTop + 3}
        fontSize={8}
        fill={isFull ? '#dc2626' : '#94a3b8'}
        fontWeight={isFull ? 'bold' : 'normal'}
      >
        rim
      </text>

      {/* Overflow animation: water sheets off the top once full */}
      {isFull && (
        <>
          <path
            d={`M ${spongeLeft + 6},${spongeTop} q -10,10 -4,22 q 6,10 -2,20`}
            fill="none"
            stroke="#0369a1"
            strokeWidth={3}
            strokeLinecap="round"
            opacity={0.85}
          >
            <animate
              attributeName="opacity"
              values="0.4;0.95;0.4"
              dur="1s"
              repeatCount="indefinite"
            />
          </path>
          <path
            d={`M ${spongeRight - 6},${spongeTop} q 10,10 4,22 q -6,10 2,20`}
            fill="none"
            stroke="#0369a1"
            strokeWidth={3}
            strokeLinecap="round"
            opacity={0.85}
          >
            <animate
              attributeName="opacity"
              values="0.95;0.4;0.95"
              dur="1s"
              repeatCount="indefinite"
            />
          </path>
          <text
            x={SVG_W / 2}
            y={spongeTop - 4}
            textAnchor="middle"
            fontSize={9}
            fontWeight="bold"
            fill="#dc2626"
          >
            overflow → runoff
          </text>
        </>
      )}

      {/* Sponge body texture (just a couple of pore dots) */}
      {!isFull &&
        [0.3, 0.55, 0.8].map((f, i) => (
          <circle
            key={i}
            cx={spongeLeft + (spongeRight - spongeLeft) * f}
            cy={spongeTop + spongeHeight * 0.35}
            r={2}
            fill="#a8a29e"
            opacity={0.6}
          />
        ))}
    </svg>
  );
}

// ─────────────────────────────────────────────────────────────────────────
// Right panel: funnel / absorbing-surface diagram for Horton infiltration-excess
// ─────────────────────────────────────────────────────────────────────────
function FunnelDiagram({
  rainRate,
  capacity,
  excessFrac,
}: {
  rainRate: number;
  capacity: number;
  excessFrac: number;
}) {
  const padT = 14;
  const surfaceY = SVG_H - 60;
  const groundBottom = SVG_H - 20;
  const left = 30;
  const right = SVG_W - 30;

  const isExceeding = excessFrac > 0;
  const nDrops = 3 + Math.round((rainRate / 100) * 5);

  return (
    <svg
      width={SVG_W}
      height={SVG_H}
      className="block overflow-visible mx-auto"
      style={{ maxWidth: '100%' }}
    >
      {/* Rain drops, density scales with rain rate */}
      {Array.from({ length: nDrops }).map((_, i) => {
        const x = left + 10 + (i * (right - left - 20)) / Math.max(nDrops - 1, 1);
        return (
          <line
            key={i}
            x1={x}
            y1={padT - 8}
            x2={x}
            y2={surfaceY - 6}
            stroke="#0ea5e9"
            strokeWidth={isExceeding ? 2.5 : 2}
            strokeLinecap="round"
            opacity={0.85}
          />
        );
      })}
      <text x={SVG_W / 2} y={padT - 12} textAnchor="middle" fontSize={9} fill="#0369a1">
        rain rate
      </text>

      {/* Ground / soil block */}
      <rect
        x={left - 10}
        y={surfaceY}
        width={right - left + 20}
        height={groundBottom - surfaceY}
        fill="#d6b88a"
        stroke="#92703f"
        strokeWidth={1.5}
        rx={3}
      />
      <text
        x={SVG_W / 2}
        y={groundBottom + 14}
        textAnchor="middle"
        fontSize={8.5}
        fill="#78502b"
      >
        soil surface
      </text>

      {/* Infiltration capacity arrows going down into the soil (fixed rate) */}
      {[0.25, 0.5, 0.75].map((f, i) => (
        <line
          key={i}
          x1={left + (right - left) * f}
          y1={surfaceY + 6}
          x2={left + (right - left) * f}
          y2={surfaceY + 6 + (capacity / 100) * 26}
          stroke="#92703f"
          strokeWidth={2.5}
          strokeLinecap="round"
          markerEnd="url(#dh-down-arrow)"
        />
      ))}
      <defs>
        <marker id="dh-down-arrow" markerWidth="6" markerHeight="6" refX="3" refY="5" orient="auto">
          <path d="M0,0 L3,6 L6,0 Z" fill="#92703f" />
        </marker>
      </defs>

      {/* Capacity line — hard physical limit, drawn just above the surface */}
      <line
        x1={left - 10}
        y1={surfaceY}
        x2={right + 10}
        y2={surfaceY}
        stroke="#92703f"
        strokeWidth={2}
      />
      <text x={right + 12} y={surfaceY + 3} fontSize={8} fill="#92703f">
        capacity
      </text>

      {/* Excess water pooling / sheeting off above the surface when rain > capacity */}
      {isExceeding && (
        <>
          <rect
            x={left - 10}
            y={surfaceY - 18 * clamp(excessFrac, 0, 1)}
            width={right - left + 20}
            height={18 * clamp(excessFrac, 0, 1)}
            fill="#0369a1"
            opacity={0.5}
            style={{ transition: 'height 0.25s ease, y 0.25s ease' }}
          />
          <path
            d={`M ${left - 6},${surfaceY - 4} q -12,8 -6,18 q 6,10 -2,18`}
            fill="none"
            stroke="#0369a1"
            strokeWidth={3}
            strokeLinecap="round"
            opacity={0.85}
          >
            <animate
              attributeName="opacity"
              values="0.4;0.95;0.4"
              dur="0.9s"
              repeatCount="indefinite"
            />
          </path>
          <path
            d={`M ${right + 6},${surfaceY - 4} q 12,8 6,18 q -6,10 2,18`}
            fill="none"
            stroke="#0369a1"
            strokeWidth={3}
            strokeLinecap="round"
            opacity={0.85}
          >
            <animate
              attributeName="opacity"
              values="0.95;0.4;0.95"
              dur="0.9s"
              repeatCount="indefinite"
            />
          </path>
          <text
            x={SVG_W / 2}
            y={surfaceY - 20 * clamp(excessFrac, 0, 1) - 6}
            textAnchor="middle"
            fontSize={9}
            fontWeight="bold"
            fill="#dc2626"
          >
            excess → runoff
          </text>
        </>
      )}
    </svg>
  );
}

// ─────────────────────────────────────────────────────────────────────────
// Root widget
// ─────────────────────────────────────────────────────────────────────────
export default function DunneHortonWidget() {
  // Dunne (left) state
  const [saturation, setSaturation] = useState(60);
  const [dunneRainRate, setDunneRainRate] = useState(50);

  // Horton (right) state
  const [hortonRainRate, setHortonRainRate] = useState(30);
  const [capacity, setCapacity] = useState(50);

  const isFull = saturation >= 100;
  const dunneRunoffPct = isFull ? 100 : 0;

  const hortonExcess = Math.max(hortonRainRate - capacity, 0);
  const hortonRunoffFrac = hortonRainRate > 0 ? hortonExcess / hortonRainRate : 0;
  const hortonRunoffPct = hortonRunoffFrac * 100;

  return (
    <div className="not-prose my-8 rounded-2xl border border-slate-200 bg-white shadow-xl overflow-hidden font-sans">
      <div className="bg-gradient-to-r from-emerald-700 to-teal-900 px-6 py-4">
        <h3 className="text-white font-bold text-lg tracking-tight">
          Two Ways Rain Becomes Runoff
        </h3>
        <p className="text-emerald-200 text-sm mt-0.5">
          Saturation-excess (Dunne) vs. infiltration-excess (Horton) — capacity vs. intensity
        </p>
      </div>

      <div className="p-6 flex flex-col gap-6">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* ─────────────── LEFT: Dunne saturation-excess ─────────────── */}
          <div className="rounded-xl border border-sky-200 bg-sky-50/40 p-4 flex flex-col gap-4">
            <div>
              <h4 className="text-sm font-bold text-sky-900 uppercase tracking-wide">
                Dunne — Saturation-Excess
              </h4>
              <p className="text-xs text-slate-500 mt-0.5">
                &ldquo;The soil is full from below.&rdquo; Sponge sitting in a shallow tray of
                water.
              </p>
            </div>

            <SpongeDiagram saturation={saturation} rainRate={dunneRainRate} isFull={isFull} />

            {/* Saturation slider */}
            <div className="flex flex-col gap-1">
              <label className="text-xs font-semibold text-slate-600">
                Saturation level: {saturation}%
              </label>
              <input
                type="range"
                min={0}
                max={100}
                step={1}
                value={saturation}
                onChange={(e) => setSaturation(+e.target.value)}
                className="w-full"
              />
              <div className="text-xs text-slate-400 flex justify-between">
                <span>0% bone dry</span>
                <span>100% saturated</span>
              </div>
            </div>

            {/* Rain-rate slider (independent) */}
            <div className="flex flex-col gap-1">
              <label className="text-xs font-semibold text-slate-600">
                Rain rate: {dunneRainRate} mm/hr
              </label>
              <input
                type="range"
                min={0}
                max={100}
                step={1}
                value={dunneRainRate}
                onChange={(e) => setDunneRainRate(+e.target.value)}
                className="w-full"
              />
              <div className="text-xs text-slate-400 flex justify-between">
                <span>drizzle</span>
                <span>downpour</span>
              </div>
            </div>

            {/* Live readout */}
            <div
              className={`rounded-lg border p-3 text-sm font-semibold ${
                isFull
                  ? 'bg-red-50 border-red-300 text-red-800'
                  : 'bg-emerald-50 border-emerald-200 text-emerald-800'
              }`}
            >
              Runoff = {dunneRunoffPct}% of rain
              <div className="text-xs font-normal mt-1 text-slate-600">
                {isFull
                  ? 'Sponge is full — rate is irrelevant. Try moving the rain-rate slider: the runoff stays pinned at 100% no matter what.'
                  : 'Soil still has room — everything infiltrates. Runoff stays at 0% until saturation reaches 100%.'}
              </div>
            </div>

            <p className="text-xs text-slate-500 italic border-t border-sky-200 pt-2">
              Saturation-excess: the soil&apos;s <strong>capacity</strong> decides, not the
              storm&apos;s <strong>intensity</strong>.
            </p>
          </div>

          {/* ─────────────── RIGHT: Horton infiltration-excess ─────────────── */}
          <div className="rounded-xl border border-amber-200 bg-amber-50/40 p-4 flex flex-col gap-4">
            <div>
              <h4 className="text-sm font-bold text-amber-900 uppercase tracking-wide">
                Horton — Infiltration-Excess
              </h4>
              <p className="text-xs text-slate-500 mt-0.5">
                &ldquo;The rain is too fast for the surface.&rdquo; Dry sponge blasted with a
                fire hose.
              </p>
            </div>

            <FunnelDiagram
              rainRate={hortonRainRate}
              capacity={capacity}
              excessFrac={hortonRunoffFrac}
            />

            {/* Rain-rate slider */}
            <div className="flex flex-col gap-1">
              <label className="text-xs font-semibold text-slate-600">
                Rain rate: {hortonRainRate} mm/hr
              </label>
              <input
                type="range"
                min={0}
                max={100}
                step={1}
                value={hortonRainRate}
                onChange={(e) => setHortonRainRate(+e.target.value)}
                className="w-full"
              />
              <div className="text-xs text-slate-400 flex justify-between">
                <span>drizzle</span>
                <span>downpour</span>
              </div>
            </div>

            {/* Infiltration capacity slider */}
            <div className="flex flex-col gap-1">
              <label className="text-xs font-semibold text-slate-600">
                Infiltration capacity: {capacity} mm/hr
              </label>
              <input
                type="range"
                min={0}
                max={100}
                step={1}
                value={capacity}
                onChange={(e) => setCapacity(+e.target.value)}
                className="w-full"
              />
              <div className="text-xs text-slate-400 flex justify-between">
                <span>tight/clayey</span>
                <span>open/sandy</span>
              </div>
            </div>

            {/* Live readout */}
            <div
              className={`rounded-lg border p-3 text-sm font-semibold ${
                hortonRunoffPct > 0
                  ? 'bg-red-50 border-red-300 text-red-800'
                  : 'bg-emerald-50 border-emerald-200 text-emerald-800'
              }`}
            >
              Runoff = {hortonRunoffPct.toFixed(0)}% of rain
              <div className="text-xs font-normal mt-1 text-slate-600">
                {hortonRunoffPct > 0
                  ? `Rain rate (${hortonRainRate} mm/hr) exceeds capacity (${capacity} mm/hr) by ${hortonExcess} mm/hr — that excess sheds as runoff.`
                  : `Rain rate (${hortonRainRate} mm/hr) is at or below capacity (${capacity} mm/hr) — the soil keeps up, everything infiltrates.`}
              </div>
            </div>

            <p className="text-xs text-slate-500 italic border-t border-amber-200 pt-2">
              Infiltration-excess: the storm&apos;s <strong>intensity</strong> decides, not the
              soil&apos;s <strong>capacity</strong>.
            </p>
          </div>
        </div>

        {/* Synthesis */}
        <div className="bg-slate-50 border border-slate-200 rounded-lg p-3 text-xs text-slate-700">
          <span className="font-semibold text-slate-900">OPM runs both at once: </span>
          OPM runs both mechanisms simultaneously and adds their contributions — Chapter 3 §3.3
          builds the Dunne side into the real Variable Source Area equations, and §3.4 builds the
          Horton side into Green-Ampt infiltration.
        </div>
      </div>
    </div>
  );
}
