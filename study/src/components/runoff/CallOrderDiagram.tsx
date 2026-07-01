'use client';

import React from 'react';

type Box = { x: number; y: number; w: number; h: number; lines: string[] };

const BOX_H = 54;
const SVG_W = 760;
const GAP = 14;
const BOX_W = (SVG_W - GAP * 3) / 4;
const ARROW_Y_OFFSET = BOX_H / 2;

const GOOD = '#047857'; // emerald-700
const GOOD_FILL = '#ecfdf5'; // emerald-50
const BAD = '#b91c1c'; // red-700
const BAD_FILL = '#fef2f2'; // red-50
const NEUTRAL = '#475569'; // slate-600
const NEUTRAL_FILL = '#f8fafc'; // slate-50

function StepBox({
  box,
  fill,
  stroke,
  textColor = '#1e293b',
}: {
  box: Box;
  fill: string;
  stroke: string;
  textColor?: string;
}) {
  return (
    <g>
      <rect
        x={box.x}
        y={box.y}
        width={box.w}
        height={box.h}
        rx={10}
        fill={fill}
        stroke={stroke}
        strokeWidth={2}
      />
      {box.lines.map((line, i) => {
        const n = box.lines.length;
        const lineHeight = 14;
        const startY = box.y + box.h / 2 - ((n - 1) * lineHeight) / 2 + 4;
        return (
          <text
            key={i}
            x={box.x + box.w / 2}
            y={startY + i * lineHeight}
            textAnchor="middle"
            fontSize={11}
            fontWeight={i === 0 ? 700 : 500}
            fill={textColor}
            fontFamily="ui-monospace, SFMono-Regular, Menlo, monospace"
          >
            {line}
          </text>
        );
      })}
    </g>
  );
}

function HArrow({
  x1,
  x2,
  y,
  color,
  dashed = false,
}: {
  x1: number;
  x2: number;
  y: number;
  color: string;
  dashed?: boolean;
}) {
  const headLen = 8;
  const headWidth = 5;
  return (
    <g>
      <line
        x1={x1}
        y1={y}
        x2={x2 - headLen}
        y2={y}
        stroke={color}
        strokeWidth={2}
        strokeDasharray={dashed ? '4 3' : undefined}
      />
      <polygon
        points={`${x2},${y} ${x2 - headLen},${y - headWidth} ${x2 - headLen},${y + headWidth}`}
        fill={color}
      />
    </g>
  );
}

// Each row is laid out in its own local coordinate space (y starts at 0) since
// every track gets its own <svg>.
function rowBoxes(lines0: string[], lines1: string[], lines2: string[], lines3: string[]): Box[] {
  const y = 0;
  return [
    { x: 0, y, w: BOX_W, h: BOX_H, lines: lines0 },
    { x: BOX_W + GAP, y, w: BOX_W, h: BOX_H, lines: lines1 },
    { x: (BOX_W + GAP) * 2, y, w: BOX_W, h: BOX_H, lines: lines2 },
    { x: (BOX_W + GAP) * 3, y, w: BOX_W, h: BOX_H, lines: lines3 },
  ];
}

const ROW1: Box[] = rowBoxes(
  ['state^n', 'z, SD_max, F, VSA mask'],
  ['get_effective_1d()', 'reads state^n → runoff'],
  ['update_state()', 'writes state^{n+1}'],
  ['state^{n+1}', 'ready for next step'],
);

const ROW2: Box[] = rowBoxes(
  ['state^n', 'z, SD_max, F, VSA mask'],
  ['update_state()', 'writes state^{n+1} (too early)'],
  ['get_effective_1d()', 'reads state^{n+1} → runoff (wrong)'],
  ['✗ tomorrow’s soil', 'sheds today’s rain'],
);

const SVG_H = BOX_H + 8;

function Track({
  boxes,
  highlightColor,
  highlightFill,
  dashedArrows = false,
}: {
  boxes: Box[];
  highlightColor: string;
  highlightFill: string;
  dashedArrows?: boolean;
}) {
  return (
    <svg
      width={SVG_W}
      height={SVG_H}
      viewBox={`0 0 ${SVG_W} ${SVG_H}`}
      className="block overflow-visible"
      style={{ maxWidth: '100%' }}
    >
      {boxes.slice(0, -1).map((b, i) => (
        <HArrow
          key={i}
          x1={b.x + b.w}
          x2={boxes[i + 1].x}
          y={b.y + ARROW_Y_OFFSET}
          color={highlightColor}
          dashed={dashedArrows}
        />
      ))}
      {boxes.map((b, i) => (
        <StepBox
          key={i}
          box={b}
          fill={i === 1 || i === 2 ? highlightFill : NEUTRAL_FILL}
          stroke={i === 1 || i === 2 ? highlightColor : NEUTRAL}
        />
      ))}
    </svg>
  );
}

export default function CallOrderDiagram() {
  return (
    <div className="not-prose my-8 rounded-2xl border border-slate-200 bg-white shadow-xl overflow-hidden font-sans">
      <div className="bg-gradient-to-r from-emerald-700 to-slate-900 px-6 py-4">
        <h3 className="text-white font-bold text-lg tracking-tight">
          Call Order Matters: Read State, Then Advance It
        </h3>
        <p className="text-emerald-200 text-sm mt-0.5">
          Every timestep the router calls <code>get_effective_1d</code> before{' '}
          <code>update_state</code> — never the other way around
        </p>
      </div>

      <div className="p-6 flex flex-col gap-6">
        {/* Track 1: correct order */}
        <div>
          <p className="text-xs font-semibold text-emerald-700 uppercase tracking-wide mb-2">
            ✓ Correct: read, then advance
          </p>
          <Track boxes={ROW1} highlightColor={GOOD} highlightFill={GOOD_FILL} />
          <div className="bg-emerald-50 border border-emerald-200 rounded-lg p-3 text-xs text-emerald-900 mt-2">
            <code>get_effective_1d</code> reads the VSA mask, <code>F</code>, and{' '}
            <code>z</code> exactly as they stand at step <em>n</em> to decide how much of{' '}
            <em>this step&apos;s</em> rain becomes runoff. Only after that runoff number is
            locked in does <code>update_state</code> write <code>z^&#123;n+1&#125;</code>,{' '}
            <code>SD_max^&#123;n+1&#125;</code>, <code>F^&#123;n+1&#125;</code>, and rebuild
            the VSA mask for the next step.
          </div>
        </div>

        {/* Track 2: reversed order */}
        <div>
          <p className="text-xs font-semibold text-red-700 uppercase tracking-wide mb-2">
            ✗ Reversed: advance, then read (wrong)
          </p>
          <Track boxes={ROW2} highlightColor={BAD} highlightFill={BAD_FILL} dashedArrows />
          <div className="bg-red-50 border border-red-200 rounded-lg p-3 text-xs text-red-900 mt-2">
            ✗ Uses tomorrow&apos;s soil to shed today&apos;s rain. A cell that becomes
            saturated <em>during</em> this step&apos;s rain would be treated as
            already-saturated for the <em>entire</em> step, overstating runoff using
            information that shouldn&apos;t exist yet.
          </div>
        </div>
      </div>

      {/* Caption */}
      <div className="px-6 pb-6">
        <div className="bg-slate-50 border border-slate-200 rounded-lg p-4 text-sm text-slate-700">
          <strong>The rule:</strong> use the current state to produce this step&apos;s
          runoff, then advance the state for the next step. In code (
          <code>kinematic_wave_router.py</code>):
          <pre className="bg-white border border-slate-200 rounded-md p-3 mt-2 text-xs overflow-x-auto">
{`source_1d = runoff_engine.get_effective_1d(t_seconds, rain_1d)  # uses state^n
runoff_engine.update_state(rain_1d, dt)                         # state^n -> state^{n+1}`}
          </pre>
        </div>
      </div>
    </div>
  );
}
