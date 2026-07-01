'use client';

import React, { useState, useMemo, useRef, useEffect } from 'react';

// ─── Default terrain ────────────────────────────────────────────────────────
const DEFAULT_GRID = [
  [7, 6, 5, 4],
  [6, 5, 4, 3],
  [5, 4, 3, 2],
  [4, 3, 2, 1],
];

const CELL = 72; // px per grid cell

// ─── Elevation colour ramp ───────────────────────────────────────────────────
const STOPS: [number, [number, number, number]][] = [
  [0.00, [67, 117, 180]],
  [0.18, [116, 196, 163]],
  [0.40, [161, 218, 115]],
  [0.62, [255, 230, 130]],
  [0.82, [200, 130, 70]],
  [1.00, [245, 245, 244]],
];

function elevColor(z: number, lo: number, hi: number): string {
  const t = hi === lo ? 0.5 : (z - lo) / (hi - lo);
  let i = 0;
  while (i < STOPS.length - 2 && t > STOPS[i + 1][0]) i++;
  const [t0, c0] = STOPS[i];
  const [t1, c1] = STOPS[i + 1];
  const f = (t - t0) / (t1 - t0);
  return `rgb(${Math.round(c0[0] + f * (c1[0] - c0[0]))},${Math.round(c0[1] + f * (c1[1] - c0[1]))},${Math.round(c0[2] + f * (c1[2] - c0[2]))})`;
}

// ─── D8 flow directions ──────────────────────────────────────────────────────
interface D8Entry { dr: number; dc: number; code: number; label: string; diag: boolean }

const D8: D8Entry[] = [
  { dr: 0,  dc: 1,  code: 1,   label: 'E',  diag: false },
  { dr: 1,  dc: 1,  code: 2,   label: 'SE', diag: true  },
  { dr: 1,  dc: 0,  code: 4,   label: 'S',  diag: false },
  { dr: 1,  dc: -1, code: 8,   label: 'SW', diag: true  },
  { dr: 0,  dc: -1, code: 16,  label: 'W',  diag: false },
  { dr: -1, dc: -1, code: 32,  label: 'NW', diag: true  },
  { dr: -1, dc: 0,  code: 64,  label: 'N',  diag: false },
  { dr: -1, dc: 1,  code: 128, label: 'NE', diag: true  },
];

function computeFlow(grid: number[][]): (D8Entry | null)[][] {
  const R = grid.length, C = grid[0].length;
  return grid.map((row, r) => row.map((z, c) => {
    let best: D8Entry | null = null, bestSlope = 0;
    for (const d of D8) {
      const nr = r + d.dr, nc = c + d.dc;
      if (nr < 0 || nr >= R || nc < 0 || nc >= C) continue;
      const dist = d.diag ? Math.SQRT2 : 1;
      const slope = (z - grid[nr][nc]) / dist;
      if (slope > bestSlope) { bestSlope = slope; best = d; }
    }
    return best;
  }));
}

// ─── BFS step type ───────────────────────────────────────────────────────────
interface BFSStep {
  current: { r: number; c: number };
  newCells: { r: number; c: number }[];
  queue: { r: number; c: number }[];
  watershed: Set<string>;
  logLine: string;
}

function computeBFSSteps(
  flow: (D8Entry | null)[][],
  pour: { r: number; c: number },
): BFSStep[] {
  const R = flow.length, C = flow[0].length;
  const steps: BFSStep[] = [];
  const watershed = new Set([`${pour.r},${pour.c}`]);
  const queue: { r: number; c: number }[] = [{ ...pour }];

  while (queue.length) {
    const current = queue.shift()!;
    const newCells: { r: number; c: number }[] = [];
    const logLines: string[] = [];

    for (let dr = -1; dr <= 1; dr++) {
      for (let dc = -1; dc <= 1; dc++) {
        if (!dr && !dc) continue;
        const nr = current.r + dr, nc = current.c + dc;
        if (nr < 0 || nr >= R || nc < 0 || nc >= C) continue;
        if (watershed.has(`${nr},${nc}`)) continue;
        const d = flow[nr][nc];
        if (d && nr + d.dr === current.r && nc + d.dc === current.c) {
          watershed.add(`${nr},${nc}`);
          newCells.push({ r: nr, c: nc });
          queue.push({ r: nr, c: nc });
          logLines.push(
            `Examine (${current.r},${current.c}) → (${nr},${nc}) flows here → added`,
          );
        }
      }
    }

    steps.push({
      current,
      newCells,
      queue: [...queue],
      watershed: new Set(watershed),
      logLine:
        logLines.length > 0
          ? `Examine (${current.r},${current.c}): added ${newCells.length} neighbor(s)`
          : `Examine (${current.r},${current.c}): no upstream neighbors`,
    });
  }
  return steps;
}

// ─── Main component ──────────────────────────────────────────────────────────
export default function BFSDelineateWidget() {
  const [grid, setGrid] = useState<number[][]>(() => DEFAULT_GRID.map(r => [...r]));
  const [pour, setPour] = useState<{ r: number; c: number } | null>(null);
  const [stepIdx, setStepIdx] = useState(-1);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(600);
  const [editMode, setEditMode] = useState(false);

  const flow = useMemo(() => computeFlow(grid), [grid]);
  const lo = useMemo(() => Math.min(...grid.flat()), [grid]);
  const hi = useMemo(() => Math.max(...grid.flat()), [grid]);
  const bfsSteps = useMemo(
    () => (pour ? computeBFSSteps(flow, pour) : []),
    [flow, pour],
  );

  // Skip-mount ref pattern: reset when pour changes (but not on mount)
  const mountedPour = useRef(false);
  useEffect(() => {
    if (!mountedPour.current) { mountedPour.current = true; return; }
    setStepIdx(-1);
    setPlaying(false);
  }, [pour]);

  // Reset everything when grid changes
  const mountedGrid = useRef(false);
  useEffect(() => {
    if (!mountedGrid.current) { mountedGrid.current = true; return; }
    setPour(null);
    setStepIdx(-1);
    setPlaying(false);
  }, [grid]);

  // Animation timer
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => {
    if (!playing) return;
    if (stepIdx >= bfsSteps.length - 1) { setPlaying(false); return; }
    timerRef.current = setTimeout(() => { setStepIdx(i => i + 1); }, speed);
    return () => { if (timerRef.current) clearTimeout(timerRef.current); };
  }, [playing, stepIdx, speed, bfsSteps]);

  // ── Derived display state ────────────────────────────────────────────────
  const currentStep = stepIdx >= 0 && stepIdx < bfsSteps.length
    ? bfsSteps[stepIdx]
    : null;

  function cellFill(r: number, c: number): string {
    const isPour = pour && pour.r === r && pour.c === c;
    if (isPour) return '#1d4ed8';

    if (!currentStep) return elevColor(grid[r][c], lo, hi);

    const key = `${r},${c}`;
    if (currentStep.current.r === r && currentStep.current.c === c) return '#fbbf24';
    if (currentStep.newCells.some(n => n.r === r && n.c === c)) return '#bae6fd';
    if (currentStep.watershed.has(key)) return '#60a5fa';
    return elevColor(grid[r][c], lo, hi);
  }

  function cellTextColor(r: number, c: number): string {
    const isPour = pour && pour.r === r && pour.c === c;
    if (isPour) return '#fff';
    if (!currentStep) return '#1e293b';
    const key = `${r},${c}`;
    if (currentStep.current.r === r && currentStep.current.c === c) return '#1e293b';
    if (currentStep.newCells.some(n => n.r === r && n.c === c)) return '#0c4a6e';
    if (currentStep.watershed.has(key)) return '#fff';
    return '#1e293b';
  }

  // ── Grid interaction ─────────────────────────────────────────────────────
  function handleCellClick(r: number, c: number, e: React.MouseEvent) {
    e.preventDefault();
    if (playing) return;
    if (editMode) {
      setGrid(prev => prev.map((row, ri) => row.map((v, ci) =>
        ri === r && ci === c ? Math.min(v + 1, 20) : v,
      )));
    } else {
      setPour({ r, c });
    }
  }

  function handleCellRightClick(r: number, c: number, e: React.MouseEvent) {
    e.preventDefault();
    if (playing) return;
    setGrid(prev => prev.map((row, ri) => row.map((v, ci) =>
      ri === r && ci === c ? Math.max(v - 1, 0) : v,
    )));
  }

  // ── Controls ─────────────────────────────────────────────────────────────
  function handleReset() {
    setStepIdx(-1);
    setPlaying(false);
  }

  function handleBack() {
    setPlaying(false);
    setStepIdx(i => Math.max(i - 1, -1));
  }

  function handleStep() {
    setPlaying(false);
    setStepIdx(i => Math.min(i + 1, bfsSteps.length - 1));
  }

  function handlePlayPause() {
    if (stepIdx >= bfsSteps.length - 1) {
      setStepIdx(-1);
      setPlaying(true);
    } else {
      setPlaying(p => !p);
    }
  }

  // ── Queue panel data ──────────────────────────────────────────────────────
  const queueDisplay: { r: number; c: number }[] = currentStep
    ? currentStep.queue
    : (pour && stepIdx === -1 ? [pour] : []);

  // ── Step log (last 6 entries) ─────────────────────────────────────────────
  const logLines = bfsSteps
    .slice(0, stepIdx + 1)
    .map(s => s.logLine)
    .slice(-6);

  // ── SVG dimensions ────────────────────────────────────────────────────────
  const R = grid.length, C = grid[0].length;
  const SVG_W = C * CELL;
  const SVG_H = R * CELL;
  const PAD = 6; // arrow shortening

  // ── Arrow marker id ───────────────────────────────────────────────────────
  const ARROW_COLOR = 'rgba(30,58,138,0.7)';

  return (
    <div className="not-prose my-8 rounded-2xl border border-slate-200 bg-white shadow-xl overflow-hidden font-sans">
      {/* Header */}
      <div className="bg-gradient-to-r from-indigo-600 to-blue-500 px-6 py-4">
        <h3 className="text-white font-bold text-lg tracking-tight">
          BFS Upstream Delineation
        </h3>
        <p className="text-indigo-100 text-sm mt-0.5">
          Click a cell to set the pour point, then animate the upstream search
        </p>
      </div>

      <div className="p-6 flex flex-col lg:flex-row gap-8 items-start">
        {/* ── LEFT: SVG grid ──────────────────────────────────────────────── */}
        <div className="flex flex-col gap-2 items-start">
          <svg
            width={SVG_W}
            height={SVG_H}
            style={{ display: 'block', cursor: playing ? 'default' : 'pointer', userSelect: 'none' }}
          >
            <defs>
              <marker
                id="bfs-arrowhead"
                markerWidth="6"
                markerHeight="6"
                refX="3"
                refY="3"
                orient="auto"
              >
                <path d="M0,0 L0,6 L6,3 z" fill={ARROW_COLOR} />
              </marker>
            </defs>

            {/* Cells */}
            {grid.map((row, r) =>
              row.map((z, c) => {
                const x = c * CELL;
                const y = r * CELL;
                const fill = cellFill(r, c);
                const textCol = cellTextColor(r, c);
                const isPour = pour && pour.r === r && pour.c === c;
                return (
                  <g
                    key={`${r}-${c}`}
                    onClick={e => handleCellClick(r, c, e)}
                    onContextMenu={e => handleCellRightClick(r, c, e)}
                  >
                    <rect
                      x={x}
                      y={y}
                      width={CELL}
                      height={CELL}
                      fill={fill}
                      stroke="#cbd5e1"
                      strokeWidth={1.5}
                    />
                    {/* Elevation label */}
                    <text
                      x={x + CELL / 2}
                      y={y + CELL / 2 - (isPour ? 6 : 0)}
                      textAnchor="middle"
                      dominantBaseline="middle"
                      fontSize={15}
                      fontWeight="600"
                      fill={textCol}
                      style={{ pointerEvents: 'none' }}
                    >
                      {z}
                    </text>
                    {/* POUR label */}
                    {isPour && (
                      <text
                        x={x + CELL / 2}
                        y={y + CELL / 2 + 11}
                        textAnchor="middle"
                        dominantBaseline="middle"
                        fontSize={9}
                        fontWeight="700"
                        fill="#bfdbfe"
                        letterSpacing="0.05em"
                        style={{ pointerEvents: 'none' }}
                      >
                        POUR
                      </text>
                    )}
                    {/* Row,Col coordinate hint */}
                    <text
                      x={x + 4}
                      y={y + 11}
                      fontSize={8}
                      fill={textCol}
                      opacity={0.55}
                      style={{ pointerEvents: 'none' }}
                    >
                      {r},{c}
                    </text>
                  </g>
                );
              }),
            )}

            {/* Flow arrows */}
            {grid.map((row, r) =>
              row.map((_z, c) => {
                const d = flow[r][c];
                if (!d) return null;
                const x1 = c * CELL + CELL / 2;
                const y1 = r * CELL + CELL / 2;
                const x2 = (c + d.dc) * CELL + CELL / 2;
                const y2 = (r + d.dr) * CELL + CELL / 2;
                const len = Math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2);
                const ux = (x2 - x1) / len;
                const uy = (y2 - y1) / len;
                return (
                  <line
                    key={`arrow-${r}-${c}`}
                    x1={x1 + ux * PAD}
                    y1={y1 + uy * PAD}
                    x2={x2 - ux * (PAD + 6)}
                    y2={y2 - uy * (PAD + 6)}
                    stroke={ARROW_COLOR}
                    strokeWidth={1.8}
                    markerEnd="url(#bfs-arrowhead)"
                    style={{ pointerEvents: 'none' }}
                  />
                );
              }),
            )}
          </svg>

          {/* Edit toggle */}
          <label className="flex items-center gap-2 text-xs text-slate-600 cursor-pointer select-none mt-1">
            <input
              type="checkbox"
              checked={editMode}
              onChange={e => setEditMode(e.target.checked)}
              className="accent-indigo-600"
            />
            ✏️ Edit Terrain
            <span className="text-slate-400">(L-click ↑, R-click ↓)</span>
          </label>
        </div>

        {/* ── RIGHT: Queue panel + log + controls ─────────────────────────── */}
        <div className="flex flex-col gap-4" style={{ minWidth: 220 }}>

          {/* Queue panel */}
          <div className="rounded-xl border border-slate-200 bg-slate-50 p-3">
            <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2">
              BFS Queue
            </p>
            {!pour && stepIdx === -1 ? (
              <p className="text-xs text-slate-400 italic">Click a cell to set the pour point</p>
            ) : stepIdx === -1 ? (
              <p className="text-xs text-indigo-500 font-medium">Press ▶ to start BFS</p>
            ) : queueDisplay.length === 0 ? (
              <span className="text-xs text-slate-400 italic">empty — BFS complete</span>
            ) : (
              <div className="flex flex-wrap gap-1.5">
                {queueDisplay.map((cell, idx) => (
                  <span
                    key={`q-${idx}`}
                    className="inline-flex items-center rounded-full px-2 py-0.5 text-xs font-mono font-semibold"
                    style={{
                      background: idx === 0 ? '#fbbf24' : '#bfdbfe',
                      color: idx === 0 ? '#1e293b' : '#1e3a5f',
                    }}
                  >
                    ({cell.r},{cell.c})
                  </span>
                ))}
              </div>
            )}
          </div>

          {/* Step log */}
          <div className="rounded-xl border border-slate-200 bg-slate-50 p-3">
            <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2">
              Step Log
            </p>
            <div
              className="font-mono text-xs text-slate-600 space-y-0.5"
              style={{ minHeight: 80 }}
            >
              {logLines.length === 0 ? (
                <span className="text-slate-400 italic">No steps yet</span>
              ) : (
                logLines.map((line, i) => (
                  <div
                    key={i}
                    className="rounded px-1 py-0.5"
                    style={{
                      background: i === logLines.length - 1 ? '#fef9c3' : 'transparent',
                    }}
                  >
                    {line}
                  </div>
                ))
              )}
            </div>
          </div>

          {/* Controls */}
          <div className="rounded-xl border border-slate-200 bg-slate-50 p-3 flex flex-col gap-3">
            <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide">
              Controls
            </p>

            {/* Button row */}
            <div className="flex flex-wrap gap-1.5">
              <button
                onClick={handleReset}
                className="rounded-lg border border-slate-300 bg-white px-2.5 py-1 text-xs font-medium text-slate-700 hover:bg-slate-100 active:bg-slate-200 transition-colors"
                title="Reset to start"
              >
                ⏮ Reset
              </button>
              <button
                onClick={handleBack}
                disabled={stepIdx <= -1}
                className="rounded-lg border border-slate-300 bg-white px-2.5 py-1 text-xs font-medium text-slate-700 hover:bg-slate-100 active:bg-slate-200 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                title="Step back"
              >
                ◀ Back
              </button>
              <button
                onClick={handlePlayPause}
                disabled={!pour}
                className="rounded-lg px-2.5 py-1 text-xs font-semibold transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                style={{
                  background: pour ? (playing ? '#f59e0b' : '#4f46e5') : '#cbd5e1',
                  color: '#fff',
                }}
                title={playing ? 'Pause' : 'Play'}
              >
                {playing ? '⏸ Pause' : '▶ Play'}
              </button>
              <button
                onClick={handleStep}
                disabled={!pour || stepIdx >= bfsSteps.length - 1}
                className="rounded-lg border border-slate-300 bg-white px-2.5 py-1 text-xs font-medium text-slate-700 hover:bg-slate-100 active:bg-slate-200 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                title="Advance one step"
              >
                Step ▶
              </button>
            </div>

            {/* Speed slider */}
            <div className="flex flex-col gap-1">
              <div className="flex justify-between items-center">
                <label className="text-xs text-slate-500 font-medium">Speed</label>
                <span className="text-xs text-slate-400">
                  {speed <= 300 ? 'Fast' : speed <= 600 ? 'Medium' : 'Slow'}
                </span>
              </div>
              <input
                type="range"
                min={200}
                max={1000}
                step={50}
                value={speed}
                onChange={e => setSpeed(Number(e.target.value))}
                className="w-full accent-indigo-600"
                // Invert: left = fast (low ms) done via CSS direction trick below
                style={{ direction: 'rtl' }}
              />
              <div className="flex justify-between text-[10px] text-slate-400">
                <span>Slow</span>
                <span>Fast</span>
              </div>
            </div>

            {/* Progress */}
            <p className="text-xs text-slate-500 font-mono text-center">
              {!pour ? (
                'Select a pour point'
              ) : stepIdx === -1 ? (
                `Step 0 / ${bfsSteps.length}`
              ) : (
                `Step ${stepIdx + 1} / ${bfsSteps.length}`
              )}
            </p>
          </div>

          {/* Callout */}
          <div className="bg-amber-50 border border-amber-200 rounded-lg px-3 py-2 text-xs text-amber-900 mt-4">
            We never follow arrows forward — we look for neighbors whose arrow points
            <strong> BACK</strong> to us.
          </div>
        </div>
      </div>
    </div>
  );
}
