'use client';

import React, { useState, useEffect, useRef, useCallback } from 'react';

// ─── Constants ────────────────────────────────────────────────────────────────
const CELL_SIZE = 72;

const STOPS: [number, [number, number, number]][] = [
  [0.00, [67,  117, 180]],
  [0.18, [116, 196, 163]],
  [0.40, [161, 218, 115]],
  [0.62, [255, 230, 130]],
  [0.82, [200, 130,  70]],
  [1.00, [245, 245, 244]],
];

const D8 = [
  { dr: 0,  dc:  1, code:   1, label: 'E',  diag: false },
  { dr: 1,  dc:  1, code:   2, label: 'SE', diag: true  },
  { dr: 1,  dc:  0, code:   4, label: 'S',  diag: false },
  { dr: 1,  dc: -1, code:   8, label: 'SW', diag: true  },
  { dr: 0,  dc: -1, code:  16, label: 'W',  diag: false },
  { dr: -1, dc: -1, code:  32, label: 'NW', diag: true  },
  { dr: -1, dc:  0, code:  64, label: 'N',  diag: false },
  { dr: -1, dc:  1, code: 128, label: 'NE', diag: true  },
] as const;

type D8Entry = typeof D8[number];

const DEFAULT_GRID: number[][] = [
  [7, 6, 5, 4],
  [6, 5, 4, 3],
  [5, 4, 3, 2],
  [4, 3, 2, 1],
];

type Mode = 'static' | 'animate' | 'trace';

// ─── Color helpers ────────────────────────────────────────────────────────────
function elevColor(z: number, lo: number, hi: number): string {
  const t = hi === lo ? 0.5 : (z - lo) / (hi - lo);
  let a = STOPS[0], b = STOPS[STOPS.length - 1];
  for (let i = 0; i < STOPS.length - 1; i++) {
    if (t >= STOPS[i][0] && t <= STOPS[i + 1][0]) { a = STOPS[i]; b = STOPS[i + 1]; break; }
  }
  const u = (b[0] - a[0]) === 0 ? 0 : (t - a[0]) / (b[0] - a[0]);
  const lp = (x: number, y: number) => Math.round(x + u * (y - x));
  return `rgb(${lp(a[1][0], b[1][0])},${lp(a[1][1], b[1][1])},${lp(a[1][2], b[1][2])})`;
}

// ─── Flow computation ─────────────────────────────────────────────────────────
function computeFlow(grid: number[][]): (D8Entry | null)[][] {
  const R = grid.length, C = grid[0].length;
  return grid.map((row, r) => row.map((_, c) => {
    let best: D8Entry | null = null, mx = 0;
    for (const d of D8) {
      const nr = r + d.dr, nc = c + d.dc;
      if (nr < 0 || nr >= R || nc < 0 || nc >= C) continue;
      const slope = (grid[r][c] - grid[nr][nc]) / (d.diag ? Math.SQRT2 : 1);
      if (slope > mx) { mx = slope; best = d; }
    }
    return best;
  }));
}

function tracePath(flow: (D8Entry | null)[][], r0: number, c0: number) {
  const path = [{ r: r0, c: c0 }];
  const R = flow.length, C = flow[0].length;
  const visited = new Set([`${r0},${c0}`]);
  let r = r0, c = c0;
  while (true) {
    const d = flow[r][c];
    if (!d) break;
    const nr = r + d.dr, nc = c + d.dc;
    if (nr < 0 || nr >= R || nc < 0 || nc >= C) { path.push({ r: nr, c: nc }); break; }
    if (visited.has(`${nr},${nc}`)) break;
    visited.add(`${nr},${nc}`);
    path.push({ r: nr, c: nc });
    r = nr; c = nc;
  }
  return path;
}

function topoOrder(grid: number[][]): { r: number; c: number }[] {
  const R = grid.length, C = grid[0].length;
  const cells: { r: number; c: number; z: number }[] = [];
  for (let r = 0; r < R; r++) for (let c = 0; c < C; c++) cells.push({ r, c, z: grid[r][c] });
  return cells.sort((a, b) => b.z - a.z);
}

// ─── Arrow SVG element ────────────────────────────────────────────────────────
function D8Arrow({ d, cx, cy, color = '#ffffff' }: { d: D8Entry; cx: number; cy: number; color?: string }) {
  const reach = CELL_SIZE * 0.30;
  // tip: 30% toward neighbor center
  const tx = cx + d.dc * reach;
  const ty = cy + d.dr * reach;
  // tail: opposite side
  const bx = cx - d.dc * reach * 0.6;
  const by = cy - d.dr * reach * 0.6;

  // arrowhead: perpendicular offsets
  const len = Math.sqrt(d.dc * d.dc + d.dr * d.dr);
  const ux = len === 0 ? 0 : d.dc / len;
  const uy = len === 0 ? 0 : d.dr / len;
  const hs = 7; // half-width of arrowhead base
  const hd = 9; // depth of arrowhead

  // arrowhead base point
  const hbx = tx - ux * hd;
  const hby = ty - uy * hd;
  // perpendicular
  const px = -uy * hs;
  const py = ux * hs;

  const pts = `${tx},${ty} ${hbx + px},${hby + py} ${hbx - px},${hby - py}`;

  return (
    <g>
      <line x1={bx} y1={by} x2={tx} y2={ty} stroke={color} strokeWidth={2.5} strokeLinecap="round" />
      <polygon points={pts} fill={color} />
    </g>
  );
}

// ─── Info callout ─────────────────────────────────────────────────────────────
type CBType = 'tip' | 'warn' | 'info';
function CB({ type, children }: { type: CBType; children: React.ReactNode }) {
  const styles: Record<CBType, string> = {
    tip:  'bg-sky-50 border-sky-200 text-sky-900',
    warn: 'bg-amber-50 border-amber-200 text-amber-900',
    info: 'bg-slate-50 border-slate-200 text-slate-700',
  };
  return (
    <div className={`rounded-xl border p-3.5 text-sm leading-relaxed ${styles[type]}`}>
      {children}
    </div>
  );
}

// ─── Main widget ──────────────────────────────────────────────────────────────
export default function FlowDirTeachWidget() {
  const [grid, setGrid]           = useState<number[][]>(() => DEFAULT_GRID.map(r => [...r]));
  const [mode, setMode]           = useState<Mode>('static');
  const [animating, setAnimating] = useState(false);
  const [animStep, setAnimStep]   = useState(0);
  const [revealed, setRevealed]   = useState<Set<string>>(new Set());
  const [speed, setSpeed]         = useState(400);
  const [awaitingClick, setAwaitingClick] = useState(false);
  const [dropPath, setDropPath]   = useState<{ r: number; c: number }[]>([]);
  const [dropStep, setDropStep]   = useState(-1);
  const [dropAnimating, setDropAnimating] = useState(false);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const R = grid.length, C = grid[0].length;
  const flat = grid.flat();
  const zMin = Math.min(...flat), zMax = Math.max(...flat);
  const flow = computeFlow(grid);

  // count pits: cells where flow===null
  const pitCount = flow.flat().filter(f => f === null).length;

  // ─── Animation: assign flow direction cell-by-cell ────────────────────────
  useEffect(() => {
    if (!animating) return;
    const sorted = topoOrder(grid);
    if (animStep >= sorted.length) {
      setAnimating(false);
      return;
    }
    const timer = setTimeout(() => {
      setRevealed(prev => new Set([...prev, `${sorted[animStep].r},${sorted[animStep].c}`]));
      setAnimStep(s => s + 1);
    }, speed);
    return () => clearTimeout(timer);
  }, [animating, animStep, speed, grid]);

  // ─── Raindrop animation ───────────────────────────────────────────────────
  useEffect(() => {
    if (!dropAnimating) return;
    if (dropStep >= dropPath.length - 1) {
      setDropAnimating(false);
      return;
    }
    const timer = setTimeout(() => {
      setDropStep(s => s + 1);
    }, speed);
    return () => clearTimeout(timer);
  }, [dropAnimating, dropStep, dropPath, speed]);

  // ─── Handlers ─────────────────────────────────────────────────────────────
  const handleCellClick = useCallback((r: number, c: number, isRight: boolean) => {
    if (awaitingClick) {
      // Trace raindrop from this cell
      setAwaitingClick(false);
      const path = tracePath(flow, r, c);
      setDropPath(path);
      setDropStep(0);
      setDropAnimating(true);
      return;
    }
    // Raise / lower elevation
    setGrid(prev => {
      const g = prev.map(row => [...row]);
      g[r][c] = isRight ? Math.max(1, g[r][c] - 1) : Math.min(15, g[r][c] + 1);
      return g;
    });
    // reset animation state when grid changes
    setRevealed(new Set());
    setAnimStep(0);
    setAnimating(false);
    setDropPath([]);
    setDropStep(-1);
    setDropAnimating(false);
  }, [awaitingClick, flow]);

  const startAnimate = () => {
    setMode('animate');
    setRevealed(new Set());
    setAnimStep(0);
    setAnimating(true);
    setDropPath([]);
    setDropStep(-1);
    setDropAnimating(false);
    setAwaitingClick(false);
  };

  const startTrace = () => {
    setMode('trace');
    setAwaitingClick(true);
    setDropPath([]);
    setDropStep(-1);
    setDropAnimating(false);
    if (timerRef.current) clearTimeout(timerRef.current);
  };

  const reset = () => {
    setMode('static');
    setAnimating(false);
    setAnimStep(0);
    setRevealed(new Set());
    setAwaitingClick(false);
    setDropPath([]);
    setDropStep(-1);
    setDropAnimating(false);
    if (timerRef.current) clearTimeout(timerRef.current);
  };

  // ─── Info panel message ───────────────────────────────────────────────────
  const sorted = topoOrder(grid);
  let infoType: CBType = 'tip';
  let infoMsg: React.ReactNode = 'D8 assigns flow to the steepest downhill neighbor. Pits (⚠) have no downhill neighbor.';

  if (animating && animStep < sorted.length) {
    const cell = sorted[animStep];
    const d = flow[cell.r][cell.c];
    let slope = 0;
    if (d) {
      const nr = cell.r + d.dr, nc = cell.c + d.dc;
      slope = (grid[cell.r][cell.c] - grid[nr][nc]) / (d.diag ? Math.SQRT2 : 1);
    }
    const dirLabel = d ? d.label : 'none (pit)';
    infoMsg = <>Processing cell ({cell.r},{cell.c}) z={grid[cell.r][cell.c]} &rarr; flows <strong>{dirLabel}</strong> (slope = {slope.toFixed(3)})</>;
    infoType = 'info';
  } else if (!animating && mode === 'animate' && animStep >= sorted.length) {
    infoMsg = `All ${R * C} cells assigned. ${pitCount} ${pitCount === 1 ? 'pit' : 'pits'} detected.`;
    infoType = pitCount > 0 ? 'warn' : 'tip';
  } else if (awaitingClick) {
    infoMsg = 'Click any cell to release a raindrop.';
    infoType = 'info';
  } else if (dropPath.length > 0 && !dropAnimating) {
    const pathStr = dropPath
      .filter(p => {
        const R2 = flow.length, C2 = flow[0].length;
        return p.r >= 0 && p.r < R2 && p.c >= 0 && p.c < C2;
      })
      .map(p => `(${p.r},${p.c})`)
      .join('→');
    const last = dropPath[dropPath.length - 1];
    const outOfBounds = last.r < 0 || last.r >= R || last.c < 0 || last.c >= C;
    const atPit = !outOfBounds && flow[last.r][last.c] === null;
    if (outOfBounds) {
      infoMsg = <>Path: {pathStr} — reached boundary (outlet)</>;
      infoType = 'tip';
    } else if (atPit) {
      infoMsg = <>&#9888; Raindrop stuck in pit at ({last.r},{last.c})!</>;
      infoType = 'warn';
    } else {
      infoMsg = <>Path: {pathStr}</>;
      infoType = 'info';
    }
  }

  // ─── Render helpers ───────────────────────────────────────────────────────
  const W = C * CELL_SIZE, H = R * CELL_SIZE;

  const isRevealed = (r: number, c: number) =>
    mode === 'static' || revealed.has(`${r},${c}`) || (!animating && mode === 'animate' && animStep >= sorted.length);

  const isCurrentAnim = (r: number, c: number) =>
    animating && animStep < sorted.length &&
    sorted[animStep].r === r && sorted[animStep].c === c;

  const dropVisited = new Set(
    dropPath.slice(0, dropStep + 1)
      .filter(p => p.r >= 0 && p.r < R && p.c >= 0 && p.c < C)
      .map(p => `${p.r},${p.c}`)
  );
  const dropHead = dropPath[dropStep] ?? null;
  const isDropHead = (r: number, c: number) =>
    dropHead !== null && dropHead.r === r && dropHead.c === c;

  return (
    <div className="not-prose my-8 rounded-2xl border border-slate-200 bg-white shadow-xl overflow-hidden font-sans">

      {/* Header */}
      <div className="bg-gradient-to-r from-blue-700 to-sky-500 px-6 py-4">
        <h3 className="text-white font-bold text-lg tracking-tight">D8 Flow Direction Explorer</h3>
        <p className="text-sky-100 text-sm mt-0.5">
          Left-click = raise elevation &middot; Right-click = lower &middot; Three interactive modes
        </p>
      </div>

      <div className="p-6 flex flex-col items-center gap-6">

        {/* SVG Grid */}
        <div className="flex flex-col items-center gap-2">
          <svg
            width={W} height={H}
            className="rounded-xl border border-slate-200 shadow-sm select-none"
            style={{ cursor: awaitingClick ? 'crosshair' : 'pointer' }}
            onContextMenu={e => e.preventDefault()}
          >
            {grid.map((row, r) => row.map((z, c) => {
              const cx = c * CELL_SIZE + CELL_SIZE / 2;
              const cy = r * CELL_SIZE + CELL_SIZE / 2;
              const dir = flow[r][c];
              const isPit = dir === null;
              const revealed_ = isRevealed(r, c);
              const isCurrent = isCurrentAnim(r, c);
              const inDropPath = dropVisited.has(`${r},${c}`);
              const isHead = isDropHead(r, c);

              let fill = elevColor(z, zMin, zMax);
              if (isCurrent) fill = '#fbbf24';
              else if (isHead) fill = '#3b82f6';
              else if (inDropPath) fill = elevColor(z, zMin, zMax); // keep terrain, add overlay below

              const textFill = z > (zMin + zMax) / 2 ? '#1e293b' : '#f0f9ff';
              const strokeColor = isPit && revealed_ ? '#ef4444' : isCurrent ? '#d97706' : '#e2e8f0';
              const strokeWidth = isPit && revealed_ ? 2.5 : isCurrent ? 2 : 0.8;

              return (
                <g key={`${r}-${c}`}
                  onClick={e => { e.preventDefault(); handleCellClick(r, c, false); }}
                  onContextMenu={e => { e.preventDefault(); handleCellClick(r, c, true); }}
                >
                  {/* Cell background */}
                  <rect
                    x={c * CELL_SIZE} y={r * CELL_SIZE}
                    width={CELL_SIZE} height={CELL_SIZE}
                    fill={fill}
                    stroke={strokeColor}
                    strokeWidth={strokeWidth}
                  />

                  {/* Light-blue tint overlay for raindrop path */}
                  {inDropPath && !isHead && (
                    <rect
                      x={c * CELL_SIZE} y={r * CELL_SIZE}
                      width={CELL_SIZE} height={CELL_SIZE}
                      fill="rgba(147,197,253,0.45)"
                      pointerEvents="none"
                    />
                  )}

                  {/* Red ring for pit */}
                  {isPit && revealed_ && (
                    <rect
                      x={c * CELL_SIZE + 3} y={r * CELL_SIZE + 3}
                      width={CELL_SIZE - 6} height={CELL_SIZE - 6}
                      fill="none"
                      stroke="#ef4444"
                      strokeWidth={2.5}
                      rx={6}
                      pointerEvents="none"
                    />
                  )}

                  {/* Elevation value */}
                  <text
                    x={cx} y={cy + 5}
                    textAnchor="middle"
                    fontSize={15}
                    fontWeight="700"
                    fill={isCurrent ? '#1e293b' : isHead ? '#ffffff' : textFill}
                    style={{ pointerEvents: 'none', fontFamily: 'monospace' }}
                  >
                    {z}
                  </text>

                  {/* ESRI code in top-left corner */}
                  {revealed_ && dir && (
                    <text
                      x={c * CELL_SIZE + 4} y={r * CELL_SIZE + 11}
                      fontSize={9}
                      fill={inDropPath ? '#1e40af' : '#94a3b8'}
                      style={{ pointerEvents: 'none', fontFamily: 'monospace' }}
                    >
                      {dir.code}
                    </text>
                  )}

                  {/* D8 arrow */}
                  {revealed_ && dir && !isHead && (
                    <D8Arrow
                      d={dir}
                      cx={cx} cy={cy}
                      color={inDropPath ? '#1d4ed8' : '#ffffff'}
                    />
                  )}

                  {/* Pit warning symbol */}
                  {isPit && revealed_ && (
                    <text
                      x={cx} y={cy + 24}
                      textAnchor="middle"
                      fontSize={13}
                      style={{ pointerEvents: 'none' }}
                    >
                      ⚠
                    </text>
                  )}

                  {/* Raindrop head */}
                  {isHead && (
                    <circle cx={cx} cy={cy} r={10} fill="#2563eb" pointerEvents="none" />
                  )}
                </g>
              );
            }))}
          </svg>

          {/* Color legend */}
          <div className="flex items-center gap-2 w-full text-xs text-slate-400 px-1">
            <span>Low</span>
            <div
              className="h-2.5 flex-1 rounded-full"
              style={{ background: 'linear-gradient(to right, rgb(67,117,180),rgb(116,196,163),rgb(161,218,115),rgb(255,230,130),rgb(200,130,70),rgb(245,245,244))' }}
            />
            <span>High</span>
          </div>
          <p className="text-xs text-slate-400 text-center">
            Left-click ▲ raise &nbsp;&middot;&nbsp; Right-click ▼ lower &nbsp;&middot;&nbsp; ESRI code shown top-left of each cell
          </p>
        </div>

        {/* Controls */}
        <div className="w-full max-w-lg flex flex-col gap-3">
          <div className="flex flex-wrap gap-2 justify-center">
            <button
              onClick={startAnimate}
              disabled={animating}
              className="px-4 py-2 rounded-xl bg-sky-600 hover:bg-sky-700 disabled:opacity-40 disabled:cursor-not-allowed text-white text-sm font-semibold transition-colors shadow-sm"
            >
              ▶ Animate Assignment
            </button>
            <button
              onClick={startTrace}
              disabled={animating || dropAnimating}
              className="px-4 py-2 rounded-xl bg-blue-500 hover:bg-blue-600 disabled:opacity-40 disabled:cursor-not-allowed text-white text-sm font-semibold transition-colors shadow-sm"
            >
              💧 Trace Raindrop
            </button>
            <button
              onClick={reset}
              className="px-4 py-2 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-700 text-sm font-semibold transition-colors shadow-sm"
            >
              ↺ Reset
            </button>
          </div>

          {/* Speed slider */}
          <div className="flex items-center gap-3 px-1">
            <span className="text-xs text-slate-500 w-10 text-right">Fast</span>
            <input
              type="range"
              min={100} max={1200} step={50}
              value={speed}
              onChange={e => setSpeed(Number(e.target.value))}
              className="flex-1 accent-sky-600"
            />
            <span className="text-xs text-slate-500 w-10">Slow</span>
            <span className="text-xs font-mono font-bold text-sky-700 w-14 text-right">{speed}ms</span>
          </div>
        </div>

        {/* Info panel */}
        <div className="w-full max-w-lg">
          <CB type={infoType}>{infoMsg}</CB>
        </div>

      </div>
    </div>
  );
}
