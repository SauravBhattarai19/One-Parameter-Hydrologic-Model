'use client';

import React, { useState, useMemo, useEffect, useRef } from 'react';

// ---------------------------------------------------------------------------
// Default ridge terrain (4×4): drains left to (3,0), right to (3,3)
// ---------------------------------------------------------------------------
const DEFAULT_GRID = [
  [5, 9, 9, 5],
  [4, 8, 8, 4],
  [3, 7, 7, 3],
  [2, 6, 6, 2],
];

const CELL_SIZE = 72;

// ---------------------------------------------------------------------------
// Elevation colour ramp
// ---------------------------------------------------------------------------
const STOPS: [number, [number, number, number]][] = [
  [0.00, [67,  117, 180]],
  [0.18, [116, 196, 163]],
  [0.40, [161, 218, 115]],
  [0.62, [255, 230, 130]],
  [0.82, [200, 130,  70]],
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

// ---------------------------------------------------------------------------
// D8 flow direction types and table
// ---------------------------------------------------------------------------
interface D8Entry {
  dr: number;
  dc: number;
  code: number;
  label: string;
  diag: boolean;
}

const D8: D8Entry[] = [
  { dr:  0, dc:  1, code:   1, label: 'E',  diag: false },
  { dr:  1, dc:  1, code:   2, label: 'SE', diag: true  },
  { dr:  1, dc:  0, code:   4, label: 'S',  diag: false },
  { dr:  1, dc: -1, code:   8, label: 'SW', diag: true  },
  { dr:  0, dc: -1, code:  16, label: 'W',  diag: false },
  { dr: -1, dc: -1, code:  32, label: 'NW', diag: true  },
  { dr: -1, dc:  0, code:  64, label: 'N',  diag: false },
  { dr: -1, dc:  1, code: 128, label: 'NE', diag: true  },
];

// ---------------------------------------------------------------------------
// D8 flow computation
// ---------------------------------------------------------------------------
function computeFlow(grid: number[][]): (D8Entry | null)[][] {
  const R = grid.length, C = grid[0].length;
  return grid.map((row, r) =>
    row.map((z, c) => {
      let best: D8Entry | null = null;
      let bestSlope = 0;
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

// ---------------------------------------------------------------------------
// Watershed delineation by upstream tracing
// ---------------------------------------------------------------------------
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
      if (d && nr + d.dr === r && nc + d.dc === c) {
        ws.add(`${nr},${nc}`);
        q.push({ r: nr, c: nc });
      }
    }
  }
  return ws;
}

// ---------------------------------------------------------------------------
// Flow accumulation
// ---------------------------------------------------------------------------
function computeFA(flow: (D8Entry | null)[][], grid: number[][]): number[][] {
  const R = grid.length, C = grid[0].length;
  const fa = Array.from({ length: R }, () => Array(C).fill(1));
  const cells: { r: number; c: number; z: number }[] = [];
  for (let r = 0; r < R; r++) for (let c = 0; c < C; c++) cells.push({ r, c, z: grid[r][c] });
  cells.sort((a, b) => b.z - a.z);
  for (const { r, c } of cells) {
    const d = flow[r][c];
    if (!d) continue;
    const nr = r + d.dr, nc = c + d.dc;
    if (nr >= 0 && nr < R && nc >= 0 && nc < C) fa[nr][nc] += fa[r][c];
  }
  return fa;
}

// ---------------------------------------------------------------------------
// Strahler stream order
// ---------------------------------------------------------------------------
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
    const upstreamOrders: number[] = [];
    for (let dr = -1; dr <= 1; dr++) for (let dc = -1; dc <= 1; dc++) {
      if (!dr && !dc) continue;
      const nr = r + dr, nc = c + dc;
      if (nr < 0 || nr >= R || nc < 0 || nc >= C) continue;
      if (fa[nr][nc] < threshold) continue;
      const d = flow[nr][nc];
      if (d && nr + d.dr === r && nc + d.dc === c) upstreamOrders.push(order[nr][nc]);
    }
    if (upstreamOrders.length === 0) {
      order[r][c] = 1;
    } else {
      const maxO = Math.max(...upstreamOrders);
      const countMax = upstreamOrders.filter(o => o === maxO).length;
      order[r][c] = countMax >= 2 ? maxO + 1 : maxO;
    }
  }
  return order;
}

// ---------------------------------------------------------------------------
// Catchment map: which outlet does each cell drain to?
// ---------------------------------------------------------------------------
function computeCatchmentMap(
  flow: (D8Entry | null)[][],
  outlets: { r: number; c: number; idx: number }[]
): number[][] {
  const R = flow.length, C = flow[0].length;
  const result = Array.from({ length: R }, () => Array(C).fill(-1));
  for (const outlet of outlets) {
    const ws = delineate(flow, { r: outlet.r, c: outlet.c });
    for (const key of ws) {
      const [r, c] = key.split(',').map(Number);
      result[r][c] = outlet.idx;
    }
  }
  return result;
}

// ---------------------------------------------------------------------------
// Trace a single cell's downstream path to an outlet or sink
// ---------------------------------------------------------------------------
function tracePath(
  flow: (D8Entry | null)[][],
  start: { r: number; c: number }
): { r: number; c: number }[] {
  const R = flow.length, C = flow[0].length;
  const path: { r: number; c: number }[] = [start];
  const visited = new Set([`${start.r},${start.c}`]);
  let cur = start;
  for (let i = 0; i < R * C; i++) {
    const d = flow[cur.r][cur.c];
    if (!d) break;
    const nr = cur.r + d.dr, nc = cur.c + d.dc;
    if (nr < 0 || nr >= R || nc < 0 || nc >= C) break;
    const key = `${nr},${nc}`;
    if (visited.has(key)) break;
    visited.add(key);
    path.push({ r: nr, c: nc });
    cur = { r: nr, c: nc };
  }
  return path;
}

// ---------------------------------------------------------------------------
// Flow arrow SVG overlay
// ---------------------------------------------------------------------------
function ArrowSVG({
  flow,
  CELL,
  filterFn,
  color = 'rgba(30,58,138,0.7)',
}: {
  flow: (D8Entry | null)[][];
  CELL: number;
  filterFn?: (r: number, c: number) => boolean;
  color?: string;
}) {
  const markerId = useRef(`arrowhead-${Math.random().toString(36).slice(2)}`).current;

  return (
    <>
      <defs>
        <marker
          id={markerId}
          markerWidth="6"
          markerHeight="6"
          refX="3"
          refY="3"
          orient="auto"
        >
          <path d="M0,0 L0,6 L6,3 z" fill={color} />
        </marker>
      </defs>
      {flow.map((row, r) =>
        row.map((d, c) => {
          if (!d) return null;
          if (filterFn && !filterFn(r, c)) return null;
          const cx = c * CELL + CELL / 2;
          const cy = r * CELL + CELL / 2;
          const ex = cx + d.dc * (CELL / 2 - 12);
          const ey = cy + d.dr * (CELL / 2 - 12);
          const sx = cx - d.dc * (CELL / 2 - 24);
          const sy = cy - d.dr * (CELL / 2 - 24);
          return (
            <line
              key={`${r}-${c}`}
              x1={sx} y1={sy}
              x2={ex} y2={ey}
              stroke={color}
              strokeWidth={1.5}
              markerEnd={`url(#${markerId})`}
            />
          );
        })
      )}
    </>
  );
}

// ---------------------------------------------------------------------------
// Main widget component
// ---------------------------------------------------------------------------
export default function CatchmentMapWidget() {
  const [grid, setGrid] = useState<number[][]>(() => DEFAULT_GRID.map(r => [...r]));
  const [tab, setTab] = useState<'catchment' | 'divide' | 'strahler'>('catchment');
  const [threshold, setThreshold] = useState(2);

  // Animation state (Tab 2)
  const [animating, setAnimating] = useState(false);
  const [trailA, setTrailA] = useState<{ r: number; c: number }[]>([]);
  const [trailB, setTrailB] = useState<{ r: number; c: number }[]>([]);
  const [animStep, setAnimStep] = useState(0);
  const animRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // ── Derived data ──────────────────────────────────────────────────────────
  const flow = useMemo(() => computeFlow(grid), [grid]);
  const lo = useMemo(() => Math.min(...grid.flat()), [grid]);
  const hi = useMemo(() => Math.max(...grid.flat()), [grid]);
  const fa = useMemo(() => computeFA(flow, grid), [flow, grid]);
  const strahlerOrder = useMemo(
    () => computeStrahler(flow, fa, grid, threshold),
    [flow, fa, grid, threshold]
  );

  const outlets = useMemo(() => {
    const R = grid.length, C = grid[0].length;
    const borders: { r: number; c: number; z: number }[] = [];
    for (let r = 0; r < R; r++) for (let c = 0; c < C; c++) {
      if (r === 0 || r === R - 1 || c === 0 || c === C - 1)
        borders.push({ r, c, z: grid[r][c] });
    }
    borders.sort((a, b) => a.z - b.z);
    const leftBorders  = borders.filter(b => b.c <  C / 2);
    const rightBorders = borders.filter(b => b.c >= C / 2);
    const outA = leftBorders[0]  ?? borders[0];
    const outB = rightBorders[0] ?? borders[borders.length - 1];
    return [
      { r: outA.r, c: outA.c, idx: 0 },
      { r: outB.r, c: outB.c, idx: 1 },
    ];
  }, [grid]);

  const catchmentMap = useMemo(() => computeCatchmentMap(flow, outlets), [flow, outlets]);

  const divideCells = useMemo(() => {
    const R = grid.length, C = grid[0].length;
    const divide = new Set<string>();
    for (let r = 0; r < R; r++) for (let c = 0; c < C; c++) {
      const basin = catchmentMap[r][c];
      if (basin === -1) continue;
      for (let dr = -1; dr <= 1; dr++) for (let dc = -1; dc <= 1; dc++) {
        if (!dr && !dc) continue;
        const nr = r + dr, nc = c + dc;
        if (nr < 0 || nr >= R || nc < 0 || nc >= C) continue;
        if (catchmentMap[nr][nc] !== -1 && catchmentMap[nr][nc] !== basin) {
          divide.add(`${r},${c}`);
          divide.add(`${nr},${nc}`);
        }
      }
    }
    return divide;
  }, [catchmentMap, grid]);

  // Pick raindrop seeds for animation: divide-adjacent cell in each basin
  const rainSeeds = useMemo(() => {
    const R = grid.length, C = grid[0].length;
    let seedA: { r: number; c: number } | null = null;
    let seedB: { r: number; c: number } | null = null;
    for (const key of divideCells) {
      const [r, c] = key.split(',').map(Number);
      // Look for a non-divide neighbor in basin A or B
      for (let dr = -1; dr <= 1; dr++) for (let dc = -1; dc <= 1; dc++) {
        if (!dr && !dc) continue;
        const nr = r + dr, nc = c + dc;
        if (nr < 0 || nr >= R || nc < 0 || nc >= C) continue;
        if (divideCells.has(`${nr},${nc}`)) continue;
        const b = catchmentMap[nr][nc];
        if (b === 0 && !seedA) seedA = { r: nr, c: nc };
        if (b === 1 && !seedB) seedB = { r: nr, c: nc };
      }
      if (seedA && seedB) break;
    }
    return { seedA, seedB };
  }, [divideCells, catchmentMap, grid]);

  // Build full paths from seeds
  const fullPathA = useMemo(
    () => (rainSeeds.seedA ? tracePath(flow, rainSeeds.seedA) : []),
    [flow, rainSeeds.seedA]
  );
  const fullPathB = useMemo(
    () => (rainSeeds.seedB ? tracePath(flow, rainSeeds.seedB) : []),
    [flow, rainSeeds.seedB]
  );

  // Animation driver
  useEffect(() => {
    if (!animating) return;
    const maxSteps = Math.max(fullPathA.length, fullPathB.length);
    if (animStep >= maxSteps) {
      setAnimating(false);
      return;
    }
    animRef.current = setTimeout(() => {
      setTrailA(fullPathA.slice(0, animStep + 1));
      setTrailB(fullPathB.slice(0, animStep + 1));
      setAnimStep(s => s + 1);
    }, 300);
    return () => { if (animRef.current) clearTimeout(animRef.current); };
  }, [animating, animStep, fullPathA, fullPathB]);

  function startAnimation() {
    if (animRef.current) clearTimeout(animRef.current);
    setTrailA([]);
    setTrailB([]);
    setAnimStep(0);
    setAnimating(true);
  }

  function stopAnimation() {
    if (animRef.current) clearTimeout(animRef.current);
    setAnimating(false);
  }

  // ── Cell editing ──────────────────────────────────────────────────────────
  function handleCellClick(r: number, c: number, e: React.MouseEvent) {
    e.preventDefault();
    setGrid(prev => {
      const next = prev.map(row => [...row]);
      if (e.button === 2 || e.type === 'contextmenu') {
        next[r][c] = Math.max(1, next[r][c] - 1);
      } else {
        next[r][c] = Math.min(20, next[r][c] + 1);
      }
      return next;
    });
    // Reset animation if terrain changes
    stopAnimation();
    setTrailA([]);
    setTrailB([]);
    setAnimStep(0);
  }

  // ── Statistics ────────────────────────────────────────────────────────────
  const R = grid.length, C = grid[0].length;
  const countA = catchmentMap.flat().filter(v => v === 0).length;
  const countB = catchmentMap.flat().filter(v => v === 1).length;
  const divideCount = divideCells.size;

  const strahlerCounts: Record<number, number> = {};
  for (let r = 0; r < R; r++) for (let c = 0; c < C; c++) {
    const o = strahlerOrder[r][c];
    if (o > 0) strahlerCounts[o] = (strahlerCounts[o] ?? 0) + 1;
  }

  const W = C * CELL_SIZE;
  const H = R * CELL_SIZE;

  // ── Tab button helper ─────────────────────────────────────────────────────
  function TabBtn({
    id,
    label,
  }: {
    id: 'catchment' | 'divide' | 'strahler';
    label: string;
  }) {
    const active = tab === id;
    return (
      <button
        onClick={() => setTab(id)}
        className={
          active
            ? 'px-4 py-1.5 rounded-full text-sm font-semibold bg-orange-500 text-white shadow'
            : 'px-4 py-1.5 rounded-full text-sm font-semibold bg-orange-50 text-orange-700 border border-orange-200 hover:bg-orange-100'
        }
      >
        {label}
      </button>
    );
  }

  // ── SVG cell helpers ──────────────────────────────────────────────────────
  function CellRect({
    r, c, fill, strokeColor = '#94a3b8', strokeWidth = 0.5, opacity = 1,
    extraLabel,
  }: {
    r: number; c: number; fill: string; strokeColor?: string;
    strokeWidth?: number; opacity?: number; extraLabel?: string;
  }) {
    const x = c * CELL_SIZE, y = r * CELL_SIZE;
    const z = grid[r][c];
    return (
      <g opacity={opacity}>
        <rect
          x={x} y={y}
          width={CELL_SIZE} height={CELL_SIZE}
          fill={fill}
          stroke={strokeColor}
          strokeWidth={strokeWidth}
          style={{ cursor: 'pointer' }}
          onClick={e => handleCellClick(r, c, e)}
          onContextMenu={e => handleCellClick(r, c, e)}
        />
        <text
          x={x + CELL_SIZE / 2}
          y={y + CELL_SIZE / 2 - (extraLabel ? 6 : 0)}
          textAnchor="middle"
          dominantBaseline="middle"
          fontSize={13}
          fontWeight="600"
          fill="#1e293b"
          style={{ pointerEvents: 'none', userSelect: 'none' }}
        >
          {z}
        </text>
        {extraLabel && (
          <text
            x={x + CELL_SIZE / 2}
            y={y + CELL_SIZE / 2 + 10}
            textAnchor="middle"
            dominantBaseline="middle"
            fontSize={11}
            fill="#7c2d12"
            style={{ pointerEvents: 'none', userSelect: 'none' }}
          >
            {extraLabel}
          </text>
        )}
      </g>
    );
  }

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div className="not-prose my-8 rounded-2xl border border-slate-200 bg-white shadow-xl overflow-hidden font-sans">
      {/* Header */}
      <div className="bg-gradient-to-r from-orange-500 to-amber-400 px-6 py-4">
        <h3 className="text-white font-bold text-lg tracking-tight">Catchment Maps &amp; Divides</h3>
        <p className="text-orange-100 text-sm mt-0.5">Ridge terrain — two basins separated by a divide</p>
      </div>

      <div className="p-5">
        {/* Tab buttons */}
        <div className="flex gap-2 flex-wrap mb-5">
          <TabBtn id="catchment" label="🗺 Catchment Map" />
          <TabBtn id="divide"    label="⛰ The Divide" />
          <TabBtn id="strahler"  label="📊 Strahler Order" />
        </div>

        {/* ────────────── TAB 1: Catchment Map ────────────── */}
        {tab === 'catchment' && (
          <div>
            <p className="text-xs text-slate-500 mb-3">
              Left-click a cell to raise elevation · Right-click to lower · Watch basins reflow live.
            </p>
            <div className="flex justify-center">
              <svg
                width={W}
                height={H}
                style={{ display: 'block', borderRadius: 8, border: '1px solid #e2e8f0' }}
              >
                {/* Base cells */}
                {grid.map((row, r) =>
                  row.map((_, c) => {
                    const basin = catchmentMap[r][c];
                    const isOutlet = outlets.some(o => o.r === r && o.c === c);
                    const isDivide = divideCells.has(`${r},${c}`);

                    const baseFill =
                      basin === 0 ? '#bfdbfe' :
                      basin === 1 ? '#bbf7d0' :
                      elevColor(grid[r][c], lo, hi);

                    const strokeColor =
                      isDivide   ? '#fb923c' :
                      isOutlet   ? (basin === 0 ? '#1d4ed8' : '#15803d') :
                      '#cbd5e1';

                    const strokeW =
                      isDivide ? 3 :
                      isOutlet ? 4 :
                      0.5;

                    const strokeDash = isDivide ? '6,3' : undefined;

                    const x = c * CELL_SIZE, y = r * CELL_SIZE;
                    const z = grid[r][c];
                    return (
                      <g key={`${r}-${c}`}>
                        <rect
                          x={x} y={y}
                          width={CELL_SIZE} height={CELL_SIZE}
                          fill={baseFill}
                          stroke={strokeColor}
                          strokeWidth={strokeW}
                          strokeDasharray={strokeDash}
                          style={{ cursor: 'pointer' }}
                          onClick={e => handleCellClick(r, c, e)}
                          onContextMenu={e => handleCellClick(r, c, e)}
                        />
                        <text
                          x={x + CELL_SIZE / 2}
                          y={y + CELL_SIZE / 2}
                          textAnchor="middle"
                          dominantBaseline="middle"
                          fontSize={13}
                          fontWeight="600"
                          fill="#1e293b"
                          style={{ pointerEvents: 'none', userSelect: 'none' }}
                        >
                          {z}
                        </text>
                        {/* Outlet marker */}
                        {isOutlet && (
                          <text
                            x={x + CELL_SIZE - 6}
                            y={y + 14}
                            textAnchor="middle"
                            fontSize={14}
                            style={{ pointerEvents: 'none' }}
                          >
                            ▼
                          </text>
                        )}
                      </g>
                    );
                  })
                )}

                {/* D8 flow arrows */}
                <ArrowSVG flow={flow} CELL={CELL_SIZE} />
              </svg>
            </div>

            {/* Basin stats */}
            <div className="mt-3 flex flex-wrap gap-3 justify-center text-xs text-slate-600">
              <span className="flex items-center gap-1">
                <span className="inline-block w-3 h-3 rounded-sm bg-blue-200 border border-blue-400" />
                Basin A: <strong>{countA} cells</strong>
              </span>
              <span className="text-slate-300">|</span>
              <span className="flex items-center gap-1">
                <span className="inline-block w-3 h-3 rounded-sm bg-green-200 border border-green-400" />
                Basin B: <strong>{countB} cells</strong>
              </span>
              <span className="text-slate-300">|</span>
              <span className="flex items-center gap-1">
                <span className="inline-block w-3 h-3 rounded-sm border-2 border-dashed border-orange-400 bg-transparent" />
                Divide: <strong>{divideCount} cells share border</strong>
              </span>
            </div>

            {/* Legend */}
            <div className="mt-3 bg-slate-50 rounded-lg px-3 py-2 text-xs text-slate-600 border border-slate-200">
              <strong>▼</strong> = outlet cell &nbsp;·&nbsp;
              <span className="text-orange-500 font-medium">dashed orange border</span> = divide cell &nbsp;·&nbsp;
              arrows = D8 flow direction
            </div>
          </div>
        )}

        {/* ────────────── TAB 2: The Divide ────────────── */}
        {tab === 'divide' && (
          <div>
            <div className="flex justify-center">
              <svg
                width={W}
                height={H}
                style={{ display: 'block', borderRadius: 8, border: '1px solid #e2e8f0' }}
              >
                {/* Base cells */}
                {grid.map((row, r) =>
                  row.map((_, c) => {
                    const isDivide = divideCells.has(`${r},${c}`);
                    const fill = isDivide
                      ? '#fed7aa'
                      : elevColor(grid[r][c], lo, hi);
                    const strokeColor = isDivide ? '#fb923c' : '#cbd5e1';
                    const strokeW = isDivide ? 4 : 0.5;
                    const x = c * CELL_SIZE, y = r * CELL_SIZE;
                    const z = grid[r][c];
                    return (
                      <g key={`${r}-${c}`} opacity={isDivide ? 1 : 0.5}>
                        <rect
                          x={x} y={y}
                          width={CELL_SIZE} height={CELL_SIZE}
                          fill={fill}
                          stroke={strokeColor}
                          strokeWidth={strokeW}
                          style={{ cursor: 'pointer' }}
                          onClick={e => handleCellClick(r, c, e)}
                          onContextMenu={e => handleCellClick(r, c, e)}
                        />
                        <text
                          x={x + CELL_SIZE / 2}
                          y={y + CELL_SIZE / 2 - (isDivide ? 6 : 0)}
                          textAnchor="middle"
                          dominantBaseline="middle"
                          fontSize={13}
                          fontWeight="600"
                          fill="#1e293b"
                          style={{ pointerEvents: 'none', userSelect: 'none' }}
                        >
                          {z}
                        </text>
                        {isDivide && (
                          <text
                            x={x + CELL_SIZE / 2}
                            y={y + CELL_SIZE / 2 + 10}
                            textAnchor="middle"
                            dominantBaseline="middle"
                            fontSize={11}
                            fill="#7c2d12"
                            style={{ pointerEvents: 'none', userSelect: 'none' }}
                          >
                            ÷
                          </text>
                        )}
                      </g>
                    );
                  })
                )}

                {/* Flow arrows (divide cells only) */}
                <ArrowSVG
                  flow={flow}
                  CELL={CELL_SIZE}
                  filterFn={(r, c) => divideCells.has(`${r},${c}`)}
                  color="rgba(154,52,18,0.8)"
                />

                {/* Raindrop trail A */}
                {trailA.map((pt, i) => (
                  <circle
                    key={`a-${i}`}
                    cx={pt.c * CELL_SIZE + CELL_SIZE / 2}
                    cy={pt.r * CELL_SIZE + CELL_SIZE / 2}
                    r={i === trailA.length - 1 ? 10 : 5}
                    fill="#3b82f6"
                    opacity={i === trailA.length - 1 ? 0.9 : 0.35}
                  />
                ))}

                {/* Raindrop trail B */}
                {trailB.map((pt, i) => (
                  <circle
                    key={`b-${i}`}
                    cx={pt.c * CELL_SIZE + CELL_SIZE / 2}
                    cy={pt.r * CELL_SIZE + CELL_SIZE / 2}
                    r={i === trailB.length - 1 ? 10 : 5}
                    fill="#22c55e"
                    opacity={i === trailB.length - 1 ? 0.9 : 0.35}
                  />
                ))}
              </svg>
            </div>

            {/* Animate button */}
            <div className="mt-3 flex items-center gap-3">
              <button
                onClick={animating ? stopAnimation : startAnimation}
                className={
                  animating
                    ? 'px-4 py-2 rounded-lg text-sm font-semibold bg-slate-200 text-slate-700 hover:bg-slate-300'
                    : 'px-4 py-2 rounded-lg text-sm font-semibold bg-orange-500 text-white hover:bg-orange-600 shadow'
                }
              >
                {animating ? '⏹ Stop' : '▶ Animate Raindrops'}
              </button>
              <span className="text-xs text-slate-500">
                {rainSeeds.seedA != null
                  ? `Blue starts at (${(rainSeeds.seedA as {r:number;c:number}).r},${(rainSeeds.seedA as {r:number;c:number}).c}) → Basin A`
                  : 'No Basin A seed found'}
                &nbsp;·&nbsp;
                {rainSeeds.seedB != null
                  ? `Green starts at (${(rainSeeds.seedB as {r:number;c:number}).r},${(rainSeeds.seedB as {r:number;c:number}).c}) → Basin B`
                  : 'No Basin B seed found'}
              </span>
            </div>

            {/* Raindrop legend */}
            <div className="mt-2 flex gap-4 text-xs text-slate-600">
              <span className="flex items-center gap-1">
                <span className="inline-block w-3 h-3 rounded-full bg-blue-500" />
                Basin A raindrop
              </span>
              <span className="flex items-center gap-1">
                <span className="inline-block w-3 h-3 rounded-full bg-green-500" />
                Basin B raindrop
              </span>
            </div>

            {/* Callout */}
            <div className="bg-orange-50 border border-orange-200 rounded-lg px-3 py-2 text-xs text-orange-900 mt-3">
              A divide cell's D8 arrow points away from the ridge. One raindrop's decision: left river or right river.
            </div>
          </div>
        )}

        {/* ────────────── TAB 3: Strahler Order ────────────── */}
        {tab === 'strahler' && (
          <div>
            {/* Threshold slider */}
            <div className="flex items-center gap-3 mb-4">
              <label className="text-sm font-medium text-slate-700 whitespace-nowrap">
                Min FA (τ): <strong>{threshold}</strong>
              </label>
              <input
                type="range"
                min={1}
                max={8}
                value={threshold}
                onChange={e => setThreshold(Number(e.target.value))}
                className="w-36 accent-sky-500"
              />
              <span className="text-xs text-slate-400">Higher → fewer stream cells</span>
            </div>

            <div className="flex justify-center">
              <svg
                width={W}
                height={H}
                style={{ display: 'block', borderRadius: 8, border: '1px solid #e2e8f0' }}
              >
                {grid.map((row, r) =>
                  row.map((_, c) => {
                    const isStream = fa[r][c] >= threshold;
                    const ord = strahlerOrder[r][c];
                    const fill = !isStream
                      ? elevColor(grid[r][c], lo, hi)
                      : ord >= 3
                      ? '#0369a1'
                      : ord === 2
                      ? '#38bdf8'
                      : '#bae6fd';
                    const textFill = isStream && ord >= 3 ? '#ffffff' : '#1e293b';
                    const x = c * CELL_SIZE, y = r * CELL_SIZE;
                    const z = grid[r][c];
                    return (
                      <g key={`${r}-${c}`}>
                        <rect
                          x={x} y={y}
                          width={CELL_SIZE} height={CELL_SIZE}
                          fill={fill}
                          stroke="#cbd5e1"
                          strokeWidth={0.5}
                          style={{ cursor: 'pointer' }}
                          onClick={e => handleCellClick(r, c, e)}
                          onContextMenu={e => handleCellClick(r, c, e)}
                        />
                        <text
                          x={x + CELL_SIZE / 2}
                          y={y + CELL_SIZE / 2 - (isStream ? 6 : 0)}
                          textAnchor="middle"
                          dominantBaseline="middle"
                          fontSize={13}
                          fontWeight="600"
                          fill={textFill}
                          style={{ pointerEvents: 'none', userSelect: 'none' }}
                        >
                          {z}
                        </text>
                        {isStream && (
                          <text
                            x={x + CELL_SIZE / 2}
                            y={y + CELL_SIZE / 2 + 10}
                            textAnchor="middle"
                            dominantBaseline="middle"
                            fontSize={11}
                            fontWeight="700"
                            fill={textFill}
                            style={{ pointerEvents: 'none', userSelect: 'none' }}
                          >
                            {ord > 0 ? `S${ord}` : ''}
                          </text>
                        )}
                      </g>
                    );
                  })
                )}

                {/* Flow arrows on stream cells only */}
                <ArrowSVG
                  flow={flow}
                  CELL={CELL_SIZE}
                  filterFn={(r, c) => fa[r][c] >= threshold}
                  color="rgba(3,105,161,0.8)"
                />
              </svg>
            </div>

            {/* Legend */}
            <div className="mt-3 flex flex-wrap gap-3 text-xs text-slate-700">
              <span className="flex items-center gap-1">
                <span className="inline-block w-4 h-4 rounded-sm bg-[#bae6fd] border border-slate-300" />
                Order 1: <strong>{strahlerCounts[1] ?? 0} cells</strong>
              </span>
              <span className="flex items-center gap-1">
                <span className="inline-block w-4 h-4 rounded-sm bg-[#38bdf8] border border-slate-300" />
                Order 2: <strong>{strahlerCounts[2] ?? 0} cells</strong>
              </span>
              <span className="flex items-center gap-1">
                <span className="inline-block w-4 h-4 rounded-sm bg-[#0369a1] border border-slate-300" />
                Order 3+: <strong>{Object.entries(strahlerCounts).filter(([k]) => Number(k) >= 3).reduce((s, [,v]) => s + v, 0)} cells</strong>
              </span>
              <span className="flex items-center gap-1 text-slate-400">
                · non-stream: elevation colour
              </span>
            </div>

            {/* Callout */}
            <div className="bg-sky-50 border border-sky-200 rounded-lg px-3 py-2 text-xs text-sky-900 mt-3">
              Two order-1 streams meeting → order 2. Two order-2 streams meeting → order 3.
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
