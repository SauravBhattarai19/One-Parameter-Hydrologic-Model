'use client';

import React, { useState, useMemo, useRef, useEffect } from 'react';

// ─── Presets ─────────────────────────────────────────────────────────────────
const PRESETS: Record<string, number[][]> = {
  valley: [[9,8,7,6,6,7,8,9],[8,7,6,5,5,6,7,8],[7,6,5,4,4,5,6,7],[6,5,4,3,3,4,5,6],[5,4,3,2,2,3,4,5],[4,3,2,1,1,2,3,4],[3,2,1,1,1,1,2,3],[2,1,1,1,1,1,1,2]],
  mountain: [[2,2,3,3,3,3,2,2],[2,3,4,5,5,4,3,2],[3,4,6,7,7,6,4,3],[3,5,7,9,9,7,5,3],[3,5,7,9,9,7,5,3],[3,4,6,7,7,6,4,3],[2,3,4,5,5,4,3,2],[2,2,3,3,3,3,2,2]],
  ridge: [[4,5,6,9,9,6,5,4],[3,4,5,8,8,5,4,3],[2,3,4,7,7,4,3,2],[1,2,3,6,6,3,2,1],[1,2,3,6,6,3,2,1],[2,3,4,7,7,4,3,2],[3,4,5,8,8,5,4,3],[4,5,6,9,9,6,5,4]],
  slope: [[9,8,7,6,5,4,3,2],[9,8,7,6,5,4,3,2],[9,8,7,6,5,4,3,2],[9,8,7,6,5,4,3,2],[9,8,7,6,5,4,3,2],[9,8,7,6,5,4,3,2],[9,8,7,6,5,4,3,2],[9,8,7,6,5,4,3,2]],
  basin: [[9,9,9,9,9,9,9,9],[9,7,7,7,7,7,7,9],[9,7,5,5,5,5,7,9],[9,7,5,3,3,5,7,9],[9,7,5,3,3,5,7,9],[9,7,5,5,5,5,7,9],[9,7,7,7,7,7,7,9],[9,9,9,9,9,9,9,9]],
};
const DEFAULT_PRESET = 'mountain';

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
  return grid.map((row, r) =>
    row.map((z, c) => {
      let best: D8Entry | null = null, bestSlope = 0;
      for (const d of D8) {
        const nr = r + d.dr, nc = c + d.dc;
        if (nr < 0 || nr >= R || nc < 0 || nc >= C) continue;
        const dist = d.diag ? Math.SQRT2 : 1;
        const slope = (z - grid[nr][nc]) / dist;
        if (slope > bestSlope) { bestSlope = slope; best = d; }
      }
      return best;
    })
  );
}

function computeFA(flow: (D8Entry | null)[][], grid: number[][]): number[][] {
  const R = grid.length, C = grid[0].length;
  const fa = Array.from({ length: R }, () => Array(C).fill(1));
  const cells: { r: number; c: number; z: number }[] = [];
  for (let r = 0; r < R; r++) for (let c = 0; c < C; c++) cells.push({ r, c, z: grid[r][c] });
  cells.sort((a, b) => b.z - a.z);
  for (const { r, c } of cells) {
    const d = flow[r][c]; if (!d) continue;
    const nr = r + d.dr, nc = c + d.dc;
    if (nr >= 0 && nr < R && nc >= 0 && nc < C) fa[nr][nc] += fa[r][c];
  }
  return fa;
}

function delineate(flow: (D8Entry | null)[][], pour: { r: number; c: number }): Set<string> {
  const R = flow.length, C = flow[0].length;
  const ws = new Set([`${pour.r},${pour.c}`]);
  const q = [{ ...pour }];
  while (q.length) {
    const { r, c } = q.shift()!;
    for (let dr = -1; dr <= 1; dr++) for (let dc = -1; dc <= 1; dc++) {
      if (!dr && !dc) continue;
      const nr = r + dr, nc = c + dc;
      if (nr < 0 || nr >= R || nc < 0 || nc >= C || ws.has(`${nr},${nc}`)) continue;
      const d = flow[nr][nc];
      if (d && nr + d.dr === r && nc + d.dc === c) { ws.add(`${nr},${nc}`); q.push({ r: nr, c: nc }); }
    }
  }
  return ws;
}

function computeStrahler(
  flow: (D8Entry | null)[][],
  fa: number[][],
  grid: number[][],
  threshold: number
): number[][] {
  const R = grid.length, C = grid[0].length;
  const order = Array.from({ length: R }, () => Array(C).fill(0));
  const cells: { r: number; c: number; z: number }[] = [];
  for (let r = 0; r < R; r++) for (let c = 0; c < C; c++) cells.push({ r, c, z: grid[r][c] });
  cells.sort((a, b) => b.z - a.z);
  for (const { r, c } of cells) {
    if (fa[r][c] < threshold) continue;
    const ups: number[] = [];
    for (let dr = -1; dr <= 1; dr++) for (let dc = -1; dc <= 1; dc++) {
      if (!dr && !dc) continue;
      const nr = r + dr, nc = c + dc;
      if (nr < 0 || nr >= R || nc < 0 || nc >= C || fa[nr][nc] < threshold) continue;
      const d = flow[nr][nc];
      if (d && nr + d.dr === r && nc + d.dc === c) ups.push(order[nr][nc]);
    }
    if (ups.length === 0) { order[r][c] = 1; }
    else {
      const mx = Math.max(...ups);
      order[r][c] = ups.filter(o => o === mx).length >= 2 ? mx + 1 : mx;
    }
  }
  return order;
}

// ─── Main component ──────────────────────────────────────────────────────────
type Mode = 'edit' | 'pour' | 'stream';

export default function WatershedExploreWidget() {
  const [preset, setPreset] = useState(DEFAULT_PRESET);
  const [grid, setGrid] = useState<number[][]>(() => PRESETS[DEFAULT_PRESET].map(r => [...r]));
  const [mode, setMode] = useState<Mode>('edit');
  const [pour, setPour] = useState<{ r: number; c: number } | null>(null);
  const [threshold, setThreshold] = useState(4);
  const [showArrows, setShowArrows] = useState(true);
  const [showWatershed, setShowWatershed] = useState(true);
  const [showStream, setShowStream] = useState(true);
  const [showStrahler, setShowStrahler] = useState(false);
  const [hovered, setHovered] = useState<{ r: number; c: number } | null>(null);

  const R = 8, C = 8, CELL = 52;

  const flow      = useMemo(() => computeFlow(grid), [grid]);
  const fa        = useMemo(() => computeFA(flow, grid), [flow, grid]);
  const lo        = useMemo(() => Math.min(...grid.flat()), [grid]);
  const hi        = useMemo(() => Math.max(...grid.flat()), [grid]);
  const watershed = useMemo(() => pour ? delineate(flow, pour) : new Set<string>(), [flow, pour]);
  const strahler  = useMemo(() => computeStrahler(flow, fa, grid, threshold), [flow, fa, grid, threshold]);

  // Reset pour when grid changes (but not on mount)
  const mountRef = useRef(true);
  useEffect(() => {
    if (mountRef.current) { mountRef.current = false; return; }
    setPour(null);
  }, [grid]);

  // When preset changes: update grid, clear pour
  const handlePreset = (name: string) => {
    setPreset(name);
    setGrid(PRESETS[name].map(r => [...r]));
    setPour(null);
  };

  const handleClick = (r: number, c: number) => {
    if (mode === 'edit') {
      setGrid(prev => {
        const next = prev.map(row => [...row]);
        next[r][c] = Math.min(9, next[r][c] + 1);
        return next;
      });
    } else if (mode === 'pour') {
      setPour({ r, c });
    }
    // stream mode: no action
  };

  const handleRightClick = (r: number, c: number) => {
    if (mode === 'edit') {
      setGrid(prev => {
        const next = prev.map(row => [...row]);
        next[r][c] = Math.max(0, next[r][c] - 1);
        return next;
      });
    }
  };

  // ── Stats ────────────────────────────────────────────────────────────────
  const streamCells = useMemo(() => {
    let count = 0;
    for (let r = 0; r < R; r++) for (let c = 0; c < C; c++) if (fa[r][c] >= threshold) count++;
    return count;
  }, [fa, threshold, R, C]);

  const maxFA = useMemo(() => Math.max(...fa.flat()), [fa]);

  const strahlerCounts = useMemo(() => {
    const counts: Record<number, number> = { 1: 0, 2: 0, 3: 0 };
    for (let r = 0; r < R; r++) for (let c = 0; c < C; c++) {
      const o = strahler[r][c];
      if (o === 1) counts[1]++;
      else if (o === 2) counts[2]++;
      else if (o >= 3) counts[3]++;
    }
    return counts;
  }, [strahler, R, C]);

  // ── Strahler colour helper ───────────────────────────────────────────────
  const strahlerColor = (order: number): string => {
    if (order === 1) return '#bae6fd';
    if (order === 2) return '#38bdf8';
    return '#0369a1';
  };

  // ── Mode button style helper ─────────────────────────────────────────────
  const modeBtn = (m: Mode) =>
    mode === m
      ? 'px-3 py-1.5 rounded-lg text-xs font-semibold bg-teal-600 text-white'
      : 'px-3 py-1.5 rounded-lg text-xs font-semibold bg-teal-50 text-teal-700 border border-teal-200';

  const presetBtn = (name: string) =>
    preset === name
      ? 'px-3 py-1 rounded-lg text-xs font-semibold bg-emerald-600 text-white'
      : 'px-3 py-1 rounded-lg text-xs font-semibold bg-slate-100 text-slate-600 border border-slate-200 hover:bg-slate-200';

  // ── Cell fill logic ──────────────────────────────────────────────────────
  const getCellFill = (r: number, c: number): string => {
    const key = `${r},${c}`;
    const isPour = pour && pour.r === r && pour.c === c;
    const z = grid[r][c];

    // Pour point always dark blue
    if (isPour) return '#1d4ed8';

    let fill = elevColor(z, lo, hi);

    if (showWatershed && watershed.has(key)) {
      fill = '#bfdbfe';
    }

    if (showStream && fa[r][c] >= threshold) {
      if (showStrahler) {
        fill = strahlerColor(strahler[r][c]);
      } else {
        fill = '#60a5fa';
      }
    }

    return fill;
  };

  const getCursor = (): string => {
    if (mode === 'edit') return 'cell';
    if (mode === 'pour') return 'crosshair';
    return 'default';
  };

  const svgWidth  = C * CELL;
  const svgHeight = R * CELL;

  return (
    <div className="not-prose my-8 rounded-2xl border border-slate-200 bg-white shadow-xl overflow-hidden font-sans">
      {/* Header */}
      <div className="bg-gradient-to-r from-teal-600 to-emerald-500 px-6 py-4">
        <h3 className="text-white font-bold text-lg tracking-tight">Explore Watersheds</h3>
        <p className="text-teal-100 text-sm mt-0.5">Edit terrain · set pour points · discover drainage basins</p>
      </div>

      <div className="p-4 flex flex-col xl:flex-row gap-6 items-start">

        {/* ── LEFT: SVG grid ──────────────────────────────────────────────── */}
        <div className="flex flex-col gap-3 flex-shrink-0">
          <svg
            width={svgWidth}
            height={svgHeight}
            onContextMenu={e => e.preventDefault()}
            style={{ display: 'block', borderRadius: '8px', overflow: 'hidden' }}
          >
            <defs>
              <marker id="arr" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
                <path d="M0,0 L6,3 L0,6 Z" fill="rgba(30,58,138,0.7)" />
              </marker>
            </defs>

            {/* Cells */}
            {grid.map((row, r) =>
              row.map((z, c) => {
                const key = `${r},${c}`;
                const isPour = pour && pour.r === r && pour.c === c;
                const isHovered = hovered && hovered.r === r && hovered.c === c;
                const cellFill = getCellFill(r, c);
                const textColor = isPour ? 'white' : '#1e3a5f';

                return (
                  <g
                    key={key}
                    onClick={() => handleClick(r, c)}
                    onContextMenu={e => { e.preventDefault(); handleRightClick(r, c); }}
                    onMouseEnter={() => setHovered({ r, c })}
                    onMouseLeave={() => setHovered(null)}
                    style={{ cursor: getCursor() }}
                  >
                    <rect
                      x={c * CELL}
                      y={r * CELL}
                      width={CELL}
                      height={CELL}
                      fill={cellFill}
                      stroke="white"
                      strokeWidth={isHovered ? 2 : 1}
                    />
                    {/* Pour point marker */}
                    {isPour && (
                      <circle
                        cx={c * CELL + CELL / 2}
                        cy={r * CELL + CELL / 2 - 6}
                        r={5}
                        fill="white"
                        opacity={0.5}
                        pointerEvents="none"
                      />
                    )}
                    {/* Hovered overlay */}
                    {isHovered && (
                      <rect
                        x={c * CELL}
                        y={r * CELL}
                        width={CELL}
                        height={CELL}
                        fill="white"
                        opacity={0.15}
                        pointerEvents="none"
                      />
                    )}
                    <text
                      x={c * CELL + CELL / 2}
                      y={r * CELL + CELL / 2 + 4}
                      textAnchor="middle"
                      fontSize={11}
                      fontWeight="bold"
                      fill={textColor}
                      pointerEvents="none"
                    >
                      {z}
                    </text>
                    {/* Hovered cell border highlight */}
                    {isHovered && (
                      <rect
                        x={c * CELL + 1}
                        y={r * CELL + 1}
                        width={CELL - 2}
                        height={CELL - 2}
                        fill="none"
                        stroke="#0f172a"
                        strokeWidth={2}
                        pointerEvents="none"
                      />
                    )}
                  </g>
                );
              })
            )}

            {/* Flow arrows */}
            {showArrows &&
              flow.map((row, r) =>
                row.map((d, c) => {
                  if (!d) return null;
                  const cx = (c + 0.5) * CELL, cy = (r + 0.5) * CELL;
                  const nr = r + d.dr, nc = c + d.dc;
                  const tx = (nc + 0.5) * CELL, ty = (nr + 0.5) * CELL;
                  const len = Math.hypot(tx - cx, ty - cy);
                  const ux = (tx - cx) / len, uy = (ty - cy) / len;
                  const x1 = cx + ux * 8, y1 = cy + uy * 8;
                  const x2 = tx - ux * 8, y2 = ty - uy * 8;
                  return (
                    <line
                      key={`arr-${r}-${c}`}
                      x1={x1} y1={y1} x2={x2} y2={y2}
                      stroke="rgba(30,58,138,0.6)"
                      strokeWidth={1}
                      markerEnd="url(#arr)"
                      pointerEvents="none"
                    />
                  );
                })
              )
            }
          </svg>

          {/* Edit hint below grid */}
          <p className="text-xs text-slate-400 text-center">
            {mode === 'edit'
              ? 'Left-click to raise · Right-click to lower'
              : mode === 'pour'
              ? 'Click a cell to set the pour point'
              : 'Viewing stream network overlay'}
          </p>
        </div>

        {/* ── RIGHT: controls + stats ──────────────────────────────────────── */}
        <div className="flex flex-col gap-4 min-w-[220px] flex-1">

          {/* Preset buttons */}
          <div>
            <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1.5">Preset Terrain</p>
            <div className="flex flex-wrap gap-1.5">
              {Object.keys(PRESETS).map(name => (
                <button key={name} className={presetBtn(name)} onClick={() => handlePreset(name)}>
                  {name}
                </button>
              ))}
            </div>
          </div>

          {/* Mode buttons */}
          <div>
            <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1.5">Interaction Mode</p>
            <div className="flex flex-wrap gap-1.5">
              <button className={modeBtn('edit')} onClick={() => setMode('edit')}>
                ✏️ Edit
              </button>
              <button className={modeBtn('pour')} onClick={() => setMode('pour')}>
                📍 Pour Point
              </button>
              <button className={modeBtn('stream')} onClick={() => setMode('stream')}>
                🌊 Stream Network
              </button>
            </div>
          </div>

          {/* Overlays */}
          <div>
            <p className="text-xs font-semibold text-teal-600 uppercase tracking-wider mb-1.5">Overlays</p>
            <div className="flex flex-col gap-1">
              {(
                [
                  { label: 'Flow Arrows',      value: showArrows,    set: setShowArrows    },
                  { label: 'Watershed Shading', value: showWatershed, set: setShowWatershed },
                  { label: 'Stream Network',    value: showStream,    set: setShowStream    },
                  { label: 'Strahler Colors',   value: showStrahler,  set: setShowStrahler  },
                ] as { label: string; value: boolean; set: (v: boolean) => void }[]
              ).map(({ label, value, set }) => (
                <label key={label} className="flex items-center gap-2 cursor-pointer select-none">
                  <input
                    type="checkbox"
                    checked={value}
                    onChange={e => set(e.target.checked)}
                    className="accent-teal-600 w-3.5 h-3.5"
                  />
                  <span className="text-xs text-slate-600">{label}</span>
                </label>
              ))}
            </div>
          </div>

          {/* Stream threshold slider */}
          <div>
            <p className="text-xs font-semibold text-teal-600 uppercase tracking-wider mb-1.5">Stream Threshold</p>
            <div className="flex items-center gap-2">
              <span className="text-xs text-slate-500 whitespace-nowrap">Min FA (τ):</span>
              <input
                type="range"
                min={1}
                max={32}
                value={threshold}
                onChange={e => setThreshold(Number(e.target.value))}
                className="flex-1 accent-teal-600"
              />
              <span className="text-xs font-mono font-semibold text-teal-700 whitespace-nowrap">τ = {threshold}</span>
            </div>
          </div>

          {/* Stats panel */}
          <div className="bg-slate-50 rounded-xl border border-slate-200 p-3 text-xs">
            {pour ? (
              <>
                <p className="font-semibold text-teal-700 mb-1">Watershed</p>
                <div className="text-slate-600 space-y-0.5 mb-3">
                  <div>
                    Area:{' '}
                    <span className="font-mono font-semibold text-slate-800">{watershed.size}</span>
                    {' '}cells ({Math.round((watershed.size / (R * C)) * 100)}%)
                  </div>
                  <div>
                    Pour point:{' '}
                    <span className="font-mono font-semibold text-slate-800">
                      ({pour.r},{pour.c})
                    </span>{' '}
                    z=<span className="font-mono font-semibold text-slate-800">{grid[pour.r][pour.c]}</span>
                  </div>
                </div>

                <p className="font-semibold text-teal-700 mb-1">Stream Network (τ={threshold})</p>
                <div className="text-slate-600 space-y-0.5">
                  <div>
                    Stream cells:{' '}
                    <span className="font-mono font-semibold text-slate-800">{streamCells}</span>
                  </div>
                  <div>
                    Max FA:{' '}
                    <span className="font-mono font-semibold text-slate-800">{maxFA}</span>
                  </div>
                </div>
              </>
            ) : (
              <p className="text-slate-400 italic text-center py-2">
                — Set a pour point (📍 mode) —
              </p>
            )}

            {/* Hover tooltip */}
            <div className="mt-3 pt-2 border-t border-slate-200">
              {hovered ? (
                <span className="text-slate-500">
                  Hover:{' '}
                  <span className="font-mono font-semibold text-slate-700">
                    ({hovered.r},{hovered.c})
                  </span>{' '}
                  z=<span className="font-mono font-semibold text-slate-700">{grid[hovered.r][hovered.c]}</span>{' '}
                  FA=<span className="font-mono font-semibold text-slate-700">{fa[hovered.r][hovered.c]}</span>
                </span>
              ) : (
                <span className="text-slate-400">Hover: —</span>
              )}
            </div>
          </div>

          {/* Strahler legend */}
          {showStrahler && (
            <div className="bg-sky-50 rounded-xl border border-sky-200 p-3 text-xs">
              <p className="font-semibold text-sky-700 mb-2 uppercase tracking-wider">Strahler Order</p>
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  <span
                    className="w-4 h-4 rounded-sm flex-shrink-0"
                    style={{ backgroundColor: '#bae6fd' }}
                  />
                  <span className="text-slate-600">
                    Order 1 —{' '}
                    <span className="font-mono font-semibold text-slate-800">{strahlerCounts[1]}</span>{' '}
                    cells
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <span
                    className="w-4 h-4 rounded-sm flex-shrink-0"
                    style={{ backgroundColor: '#38bdf8' }}
                  />
                  <span className="text-slate-600">
                    Order 2 —{' '}
                    <span className="font-mono font-semibold text-slate-800">{strahlerCounts[2]}</span>{' '}
                    cells
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <span
                    className="w-4 h-4 rounded-sm flex-shrink-0"
                    style={{ backgroundColor: '#0369a1' }}
                  />
                  <span className="text-slate-600">
                    Order 3+ —{' '}
                    <span className="font-mono font-semibold text-slate-800">{strahlerCounts[3]}</span>{' '}
                    cells
                  </span>
                </div>
              </div>
            </div>
          )}

          {/* Legend: colour ramp */}
          <div>
            <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1">Elevation Ramp</p>
            <div className="flex rounded overflow-hidden h-4" style={{ width: '100%' }}>
              {Array.from({ length: 40 }).map((_, i) => {
                const t = i / 39;
                let si = 0;
                while (si < STOPS.length - 2 && t > STOPS[si + 1][0]) si++;
                const [t0, c0] = STOPS[si];
                const [t1, c1] = STOPS[si + 1];
                const f = (t - t0) / (t1 - t0);
                const r = Math.round(c0[0] + f * (c1[0] - c0[0]));
                const g = Math.round(c0[1] + f * (c1[1] - c0[1]));
                const b = Math.round(c0[2] + f * (c1[2] - c0[2]));
                return (
                  <div key={i} style={{ flex: 1, backgroundColor: `rgb(${r},${g},${b})` }} />
                );
              })}
            </div>
            <div className="flex justify-between text-xs text-slate-400 mt-0.5">
              <span>Low</span>
              <span>High</span>
            </div>
          </div>

        </div>
      </div>
    </div>
  );
}
