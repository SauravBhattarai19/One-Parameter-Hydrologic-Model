'use client';

import React, { useState, useMemo, useRef, useEffect } from 'react';

const DEFAULT_GRID = [
  [7, 6, 5, 4],
  [6, 5, 4, 3],
  [5, 4, 3, 2],
  [4, 3, 2, 1],
];

const CELL = 72;

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

interface D8Entry { dr: number; dc: number; code: number; label: string; diag: boolean }

const D8: D8Entry[] = [
  { dr: 0, dc: 1, code: 1, label: 'E', diag: false },
  { dr: 1, dc: 1, code: 2, label: 'SE', diag: true },
  { dr: 1, dc: 0, code: 4, label: 'S', diag: false },
  { dr: 1, dc: -1, code: 8, label: 'SW', diag: true },
  { dr: 0, dc: -1, code: 16, label: 'W', diag: false },
  { dr: -1, dc: -1, code: 32, label: 'NW', diag: true },
  { dr: -1, dc: 0, code: 64, label: 'N', diag: false },
  { dr: -1, dc: 1, code: 128, label: 'NE', diag: true },
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

function tracePath(flow: (D8Entry | null)[][], r0: number, c0: number): { r: number; c: number }[] {
  const R = flow.length, C = flow[0].length;
  const path = [{ r: r0, c: c0 }];
  const visited = new Set([`${r0},${c0}`]);
  let r = r0, c = c0;
  for (;;) {
    const d = flow[r][c]; if (!d) break;
    const nr = r + d.dr, nc = c + d.dc;
    if (nr < 0 || nr >= R || nc < 0 || nc >= C || visited.has(`${nr},${nc}`)) break;
    visited.add(`${nr},${nc}`); path.push({ r: nr, c: nc }); r = nr; c = nc;
  }
  return path;
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

export default function WatershedConceptWidget() {
  const [grid, setGrid] = useState(() => DEFAULT_GRID.map(r => [...r]));
  const [mode, setMode] = useState<'trace' | 'watershed'>('trace');
  const [selected, setSelected] = useState<{ r: number; c: number } | null>(null);
  const [editMode, setEditMode] = useState(false);

  const flow = useMemo(() => computeFlow(grid), [grid]);
  const lo = useMemo(() => Math.min(...grid.flat()), [grid]);
  const hi = useMemo(() => Math.max(...grid.flat()), [grid]);

  const path = useMemo(
    () => selected && mode === 'trace' && !editMode ? tracePath(flow, selected.r, selected.c) : [],
    [flow, selected, mode, editMode]
  );
  const watershed = useMemo(
    () => selected && mode === 'watershed' && !editMode ? delineate(flow, selected) : new Set<string>(),
    [flow, selected, mode, editMode]
  );

  const mountRef = useRef(true);
  useEffect(() => {
    if (mountRef.current) { mountRef.current = false; return; }
    setSelected(null);
  }, [grid]);

  const pathSet = useMemo(() => new Set(path.map(p => `${p.r},${p.c}`)), [path]);

  function cellFill(r: number, c: number): string {
    const key = `${r},${c}`;
    if (mode === 'trace' && !editMode) {
      if (pathSet.has(key)) return '#bae6fd';
    } else if (mode === 'watershed' && !editMode) {
      if (selected && r === selected.r && c === selected.c) return '#93c5fd';
      if (watershed.has(key)) return '#bfdbfe';
    }
    return elevColor(grid[r][c], lo, hi);
  }

  function handleClick(r: number, c: number) {
    if (editMode) {
      setGrid(prev => {
        const next = prev.map(row => [...row]);
        next[r][c] = Math.min(9, next[r][c] + 1);
        return next;
      });
      return;
    }
    setSelected({ r, c });
  }

  function handleRightClick(r: number, c: number) {
    if (editMode) {
      setGrid(prev => {
        const next = prev.map(row => [...row]);
        next[r][c] = Math.max(0, next[r][c] - 1);
        return next;
      });
      return;
    }
    setSelected({ r, c });
  }

  const SVG_SIZE = CELL * 4;

  const arrows: React.ReactNode[] = [];
  for (let r = 0; r < grid.length; r++) {
    for (let c = 0; c < grid[0].length; c++) {
      const d = flow[r][c];
      if (!d) continue;
      const x1 = c * CELL + CELL / 2;
      const y1 = r * CELL + CELL / 2;
      const x2 = (c + d.dc) * CELL + CELL / 2;
      const y2 = (r + d.dr) * CELL + CELL / 2;
      const len = Math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2);
      const ux = (x2 - x1) / len;
      const uy = (y2 - y1) / len;
      const ex = x2 - ux * 10;
      const ey = y2 - uy * 10;
      arrows.push(
        <line
          key={`arrow-${r}-${c}`}
          x1={x1} y1={y1} x2={ex} y2={ey}
          stroke="rgba(30,58,138,0.7)"
          strokeWidth={1.5}
          markerEnd="url(#arrowhead)"
          pointerEvents="none"
        />
      );
    }
  }

  let infoText: React.ReactNode = null;
  if (!editMode && selected) {
    if (mode === 'trace' && path.length > 0) {
      const steps = path.length - 1;
      const pathStr = path.map(p => `(${p.r},${p.c})`).join(' → ');
      infoText = (
        <p className="text-sm text-slate-700 text-center">
          Path: {pathStr} &nbsp;({steps} step{steps !== 1 ? 's' : ''})
        </p>
      );
    } else if (mode === 'watershed') {
      infoText = (
        <p className="text-sm text-slate-700 text-center">
          {watershed.size} cell{watershed.size !== 1 ? 's' : ''} drain through ({selected.r},{selected.c}).{' '}
          Click the outlet (3,3) to see the whole basin.
        </p>
      );
    }
  } else if (!editMode && !selected) {
    infoText = (
      <p className="text-sm text-slate-500 text-center">
        {mode === 'trace'
          ? 'Click any cell to trace its flow path to the outlet.'
          : 'Click any cell to delineate its upstream watershed.'}
      </p>
    );
  } else if (editMode) {
    infoText = (
      <p className="text-sm text-slate-500 text-center">
        Left-click to raise elevation (+1), right-click to lower (−1). Max 9, min 0.
      </p>
    );
  }

  return (
    <div className="not-prose my-8 rounded-2xl border border-slate-200 bg-white shadow-xl overflow-hidden font-sans">
      <div className="bg-gradient-to-r from-blue-600 to-cyan-500 px-6 py-4">
        <h3 className="text-white font-bold text-lg tracking-tight">Watershed Concept</h3>
        <p className="text-blue-100 text-sm mt-0.5">Click any cell — trace its path or see its full contributing area</p>
      </div>
      <div className="p-6 flex flex-col items-center gap-6">
        <div className="flex items-center gap-3 flex-wrap justify-center">
          <button
            onClick={() => { setMode('trace'); setSelected(null); }}
            className={`px-4 py-1.5 rounded-full text-sm font-medium transition-colors ${
              mode === 'trace'
                ? 'bg-blue-600 text-white shadow'
                : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
            }`}
          >
            💧 Trace Down
          </button>
          <button
            onClick={() => { setMode('watershed'); setSelected(null); }}
            className={`px-4 py-1.5 rounded-full text-sm font-medium transition-colors ${
              mode === 'watershed'
                ? 'bg-blue-600 text-white shadow'
                : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
            }`}
          >
            🔵 Show Watershed
          </button>
          <label className="flex items-center gap-1.5 text-sm text-slate-600 cursor-pointer select-none">
            <input
              type="checkbox"
              checked={editMode}
              onChange={e => { setEditMode(e.target.checked); setSelected(null); }}
              className="rounded"
            />
            ✏️ Edit Terrain
          </label>
        </div>

        <svg
          width={SVG_SIZE}
          height={SVG_SIZE}
          style={{ display: 'block', borderRadius: 8, overflow: 'hidden' }}
        >
          <defs>
            <marker
              id="arrowhead"
              markerWidth="6"
              markerHeight="6"
              refX="3"
              refY="3"
              orient="auto"
            >
              <path d="M0,0 L6,3 L0,6 Z" fill="rgba(30,58,138,0.7)" />
            </marker>
          </defs>

          {grid.map((row, r) =>
            row.map((z, c) => {
              const isSelected = selected?.r === r && selected?.c === c;
              const isPour = mode === 'watershed' && !editMode && isSelected;
              return (
                <g
                  key={`${r}-${c}`}
                  style={{ cursor: 'pointer' }}
                  onClick={() => handleClick(r, c)}
                  onContextMenu={(e) => { e.preventDefault(); handleRightClick(r, c); }}
                >
                  <rect
                    x={c * CELL}
                    y={r * CELL}
                    width={CELL}
                    height={CELL}
                    fill={cellFill(r, c)}
                    stroke={isPour ? '#1d4ed8' : 'white'}
                    strokeWidth={isPour ? 3 : 2}
                  />
                  <text
                    x={c * CELL + CELL / 2}
                    y={r * CELL + CELL / 2 + 5}
                    textAnchor="middle"
                    fontSize={16}
                    fontWeight="bold"
                    fill="#1e3a5f"
                    pointerEvents="none"
                  >
                    {z}
                  </text>
                  {isPour && (
                    <text
                      x={c * CELL + CELL / 2}
                      y={r * CELL + CELL / 2 + 22}
                      textAnchor="middle"
                      fontSize={9}
                      fontWeight="bold"
                      fill="#1d4ed8"
                      pointerEvents="none"
                    >
                      POUR
                    </text>
                  )}
                  {mode === 'trace' && !editMode && isSelected && (
                    <circle
                      cx={c * CELL + CELL / 2}
                      cy={r * CELL + CELL / 2}
                      r={8}
                      fill="#0284c7"
                      pointerEvents="none"
                    />
                  )}
                </g>
              );
            })
          )}

          {arrows}
        </svg>

        <div className="min-h-[2rem] flex flex-col items-center gap-1">
          {infoText}
        </div>

        <div className="bg-blue-50 border border-blue-200 rounded-lg px-4 py-2 text-sm text-blue-800 max-w-sm text-center">
          Tip: In watershed mode, click cell (3,3) to see all 16 cells light up — they all drain to the corner.
        </div>
      </div>
    </div>
  );
}
