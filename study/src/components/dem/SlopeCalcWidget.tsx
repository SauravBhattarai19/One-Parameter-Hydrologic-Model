'use client';

import React, { useState } from 'react';

// ─── Types ────────────────────────────────────────────────────────────────────
type Grid = number[][];

// ─── D8 directions (ESRI scan order: E SE S SW W NW N NE) ────────────────────
const D8 = [
  { dr:  0, dc:  1, code:   1, label: 'E',  diag: false },
  { dr:  1, dc:  1, code:   2, label: 'SE', diag: true  },
  { dr:  1, dc:  0, code:   4, label: 'S',  diag: false },
  { dr:  1, dc: -1, code:   8, label: 'SW', diag: true  },
  { dr:  0, dc: -1, code:  16, label: 'W',  diag: false },
  { dr: -1, dc: -1, code:  32, label: 'NW', diag: true  },
  { dr: -1, dc:  0, code:  64, label: 'N',  diag: false },
  { dr: -1, dc:  1, code: 128, label: 'NE', diag: true  },
];

// ─── Color mapping ────────────────────────────────────────────────────────────
const STOPS: [number, [number, number, number]][] = [
  [0.00, [ 67, 117, 180]],
  [0.18, [116, 196, 163]],
  [0.40, [161, 218, 115]],
  [0.62, [255, 230, 130]],
  [0.82, [200, 130,  70]],
  [1.00, [245, 245, 244]],
];
function elevColor(z: number, lo: number, hi: number): string {
  const t = hi === lo ? 0.5 : (z - lo) / (hi - lo);
  let a = STOPS[0], b = STOPS[STOPS.length - 1];
  for (let i = 0; i < STOPS.length - 1; i++)
    if (t >= STOPS[i][0] && t <= STOPS[i + 1][0]) { a = STOPS[i]; b = STOPS[i + 1]; break; }
  const u = (b[0] - a[0]) === 0 ? 0 : (t - a[0]) / (b[0] - a[0]);
  const lp = (x: number, y: number) => Math.round(x + u * (y - x));
  return `rgb(${lp(a[1][0], b[1][0])},${lp(a[1][1], b[1][1])},${lp(a[1][2], b[1][2])})`;
}

// ─── Slope row type ───────────────────────────────────────────────────────────
interface SlopeRow {
  label: string;
  code: number;
  diag: boolean;
  isEdge: boolean;
  zNbr: number | null;
  dist: number;
  dz: number;
  slope: number | null;
}

function computeSlopes(grid: Grid, r: number, c: number): SlopeRow[] {
  const R = grid.length, C = grid[0].length, z0 = grid[r][c];
  return D8.map(d => {
    const nr = r + d.dr, nc = c + d.dc;
    if (nr < 0 || nr >= R || nc < 0 || nc >= C)
      return { label: d.label, code: d.code, diag: d.diag, isEdge: true, zNbr: null, dist: 0, dz: 0, slope: null };
    const zNbr = grid[nr][nc];
    const dist = d.diag ? Math.SQRT2 : 1;
    const dz = z0 - zNbr;
    return { label: d.label, code: d.code, diag: d.diag, isEdge: false, zNbr, dist, dz, slope: dz / dist };
  });
}

// ─── Default grid ─────────────────────────────────────────────────────────────
const DEFAULT_GRID: Grid = [
  [7, 6, 5, 4],
  [6, 5, 4, 3],
  [5, 4, 3, 2],
  [4, 3, 2, 1],
];

const CELL_SIZE = 72;

// ─── ESRI code diagram layout ─────────────────────────────────────────────────
const ESRI_DIAGRAM = [32, 64, 128, 16, '·', 1, 8, 4, 2];

// ─── Main widget ──────────────────────────────────────────────────────────────
export default function SlopeCalcWidget() {
  const [grid, setGrid]       = useState<Grid>(() => DEFAULT_GRID.map(r => [...r]));
  const [selCell, setSelCell] = useState<[number, number]>([0, 0]);

  const R = grid.length, C = grid[0].length;
  const zMin = Math.min(...grid.flat()), zMax = Math.max(...grid.flat());

  // ── Interact: left-click raises, right-click lowers, both select ──────────
  const interact = (r: number, c: number, isRight: boolean, e: React.MouseEvent) => {
    e.preventDefault();
    setGrid(prev => {
      const g = prev.map(row => [...row]);
      g[r][c] = isRight ? Math.max(1, g[r][c] - 1) : Math.min(15, g[r][c] + 1);
      return g;
    });
    setSelCell([r, c]);
  };

  // ── Slope data for selected cell ──────────────────────────────────────────
  const [sr, sc] = selCell;
  const rows = computeSlopes(grid, sr, sc);
  const z0   = grid[sr][sc];

  const validRows  = rows.filter(row => !row.isEdge && row.slope !== null);
  const maxSlope   = validRows.length > 0 ? Math.max(...validRows.map(r => r.slope!)) : -Infinity;
  const hasDownhill = maxSlope > 0;

  const winnerRows = hasDownhill
    ? validRows.filter(row => Math.abs(row.slope! - maxSlope) < 1e-9)
    : [];
  const isTied     = winnerRows.length > 1;

  // D8 winner = first winner in scan order (already in scan order from D8 array)
  const d8Winner   = winnerRows[0] ?? null;

  // ─── Render ─────────────────────────────────────────────────────────────────
  return (
    <div className="not-prose my-8 rounded-2xl border border-slate-200 bg-white shadow-xl overflow-hidden font-sans">

      {/* Header */}
      <div className="bg-gradient-to-r from-sky-600 to-blue-800 px-6 py-4">
        <h3 className="text-white font-bold text-lg tracking-tight">DEM Slope Calculator</h3>
        <p className="text-sky-100 text-sm mt-0.5">Left-click: raise · Right-click: lower · Click any cell to inspect slopes</p>
      </div>

      <div className="p-6 flex flex-col lg:flex-row gap-8 items-start">

        {/* ── SVG Grid ────────────────────────────────────────────────────── */}
        <div className="flex-shrink-0 flex flex-col items-center gap-3">
          <svg
            width={C * CELL_SIZE}
            height={R * CELL_SIZE}
            className="rounded-xl border border-slate-200 cursor-pointer select-none shadow-sm"
            onContextMenu={e => e.preventDefault()}
          >
            {grid.map((row, r) => row.map((z, c) => {
              const bg      = elevColor(z, zMin, zMax);
              const isSel   = selCell[0] === r && selCell[1] === c;
              const cx      = c * CELL_SIZE + CELL_SIZE / 2;
              const cy      = r * CELL_SIZE + CELL_SIZE / 2;
              const textFill = z > (zMin + zMax) / 2 ? '#1e293b' : '#f0f9ff';
              return (
                <g
                  key={`${r}-${c}`}
                  onClick={e => interact(r, c, false, e)}
                  onContextMenu={e => interact(r, c, true, e)}
                >
                  <rect
                    x={c * CELL_SIZE} y={r * CELL_SIZE}
                    width={CELL_SIZE} height={CELL_SIZE}
                    fill={bg}
                    stroke={isSel ? '#eab308' : '#e2e8f0'}
                    strokeWidth={isSel ? 3 : 0.8}
                  />
                  <text
                    x={cx} y={cy + 6}
                    textAnchor="middle"
                    fontSize={18}
                    fontWeight="700"
                    fill={isSel ? '#1e293b' : textFill}
                    style={{ fontFamily: 'monospace', pointerEvents: 'none' }}
                  >{z}</text>
                </g>
              );
            }))}
          </svg>

          {/* Color bar */}
          <div className="flex items-center gap-2 w-full text-xs text-slate-400 px-1">
            <span>Low</span>
            <div
              className="h-2.5 flex-1 rounded-full"
              style={{ background: 'linear-gradient(to right, rgb(67,117,180),rgb(116,196,163),rgb(161,218,115),rgb(255,230,130),rgb(200,130,70),rgb(245,245,244))' }}
            />
            <span>High</span>
          </div>
        </div>

        {/* ── Info Panel ──────────────────────────────────────────────────── */}
        <div className="flex-1 min-w-0 space-y-4">

          {/* Cell header */}
          <div className="flex items-center gap-2">
            <span className="inline-block w-3 h-3 rounded-sm bg-yellow-300 border-2 border-yellow-500" />
            <span className="text-sm font-semibold text-slate-700">
              Cell ({sr}, {sc}) — elevation <strong>z = {z0} m</strong>
            </span>
          </div>

          {/* Formula callout */}
          <div className="rounded-xl border border-sky-200 bg-sky-50 px-4 py-3 text-sm text-sky-900">
            <span className="font-semibold">Slope formula: </span>
            <code className="font-mono">S = (z_center − z_neighbor) / d</code>
            <span className="text-sky-700">, where </span>
            <code className="font-mono">d = 1</code>
            <span className="text-sky-700"> (cardinal) or </span>
            <code className="font-mono">d = √2 ≈ 1.41</code>
            <span className="text-sky-700"> (diagonal)</span>
          </div>

          {/* Slope table */}
          <div className="overflow-x-auto rounded-xl border border-slate-200 text-xs">
            <table className="w-full text-center border-collapse">
              <thead>
                <tr className="bg-slate-100 text-slate-600">
                  <th className="px-2 py-2 font-semibold">Direction</th>
                  <th className="px-2 py-2 font-semibold">ESRI Code</th>
                  <th className="px-2 py-2 font-semibold">Neighbor</th>
                  <th className="px-2 py-2 font-semibold">Dist</th>
                  <th className="px-2 py-2 font-semibold">z_nbr</th>
                  <th className="px-2 py-2 font-semibold">Δz</th>
                  <th className="px-2 py-2 font-semibold">Slope</th>
                  <th className="px-2 py-2 font-semibold">Status</th>
                </tr>
              </thead>
              <tbody>
                {rows.map(row => {
                  const isWinner = !isTied && winnerRows.some(w => w.code === row.code);
                  const isTiedRow = isTied && winnerRows.some(w => w.code === row.code);

                  // Row background
                  let rowCls = 'border-t border-slate-100 ';
                  if (row.isEdge)                                  rowCls += 'bg-slate-50';
                  else if (isWinner)                               rowCls += 'bg-emerald-50';
                  else if (isTiedRow)                              rowCls += 'bg-amber-50';
                  else if (row.dz < 0)                             rowCls += 'bg-red-50/50';

                  // Neighbor cell label: (nr, nc)
                  const nr = sr + D8.find(d => d.code === row.code)!.dr;
                  const nc = sc + D8.find(d => d.code === row.code)!.dc;
                  const nbrLabel = row.isEdge ? '—' : `(${nr},${nc})`;

                  // Distance string
                  const distStr = row.isEdge ? '—' : (row.diag ? '1.41' : '1.00');

                  // Slope display
                  const slopeStr = row.isEdge || row.slope === null ? '—' : row.slope.toFixed(3);

                  // Status badge
                  let badge: React.ReactNode;
                  if (row.isEdge) {
                    badge = <span className="rounded px-1.5 py-0.5 bg-slate-200 text-slate-500 font-medium">— Edge</span>;
                  } else if (row.dz < 0) {
                    badge = <span className="rounded px-1.5 py-0.5 bg-red-100 text-red-700 font-medium">↑ Uphill</span>;
                  } else if (row.dz === 0) {
                    badge = <span className="rounded px-1.5 py-0.5 bg-slate-200 text-slate-500 font-medium">→ Flat</span>;
                  } else if (isWinner) {
                    badge = <span className="rounded px-1.5 py-0.5 bg-emerald-100 text-emerald-700 font-semibold">✓ Winner</span>;
                  } else if (isTiedRow) {
                    badge = <span className="rounded px-1.5 py-0.5 bg-amber-100 text-amber-700 font-semibold">~ Tied</span>;
                  } else {
                    badge = <span className="rounded px-1.5 py-0.5 bg-sky-100 text-sky-700 font-medium">↓ Downhill</span>;
                  }

                  return (
                    <tr key={row.code} className={rowCls}>
                      <td className="px-2 py-1.5 font-mono font-bold">{row.label}</td>
                      <td className="px-2 py-1.5 font-mono">{row.code}</td>
                      <td className="px-2 py-1.5 text-slate-500">{nbrLabel}</td>
                      <td className="px-2 py-1.5 font-mono text-slate-500">{distStr}</td>
                      <td className="px-2 py-1.5 font-mono">
                        {row.isEdge ? '—' : row.zNbr}
                      </td>
                      <td className={`px-2 py-1.5 font-mono font-semibold ${row.dz > 0 ? 'text-emerald-700' : row.dz < 0 ? 'text-red-600' : 'text-slate-400'}`}>
                        {row.isEdge ? '—' : row.dz.toFixed(0)}
                      </td>
                      <td className={`px-2 py-1.5 font-mono font-semibold ${isWinner ? 'text-emerald-700' : isTiedRow ? 'text-amber-700' : row.slope !== null && row.slope < 0 ? 'text-red-500' : 'text-slate-600'}`}>
                        {slopeStr}
                      </td>
                      <td className="px-2 py-1.5">{badge}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {/* Flow result */}
          {hasDownhill ? (
            <div className="rounded-xl border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm text-emerald-900">
              {isTied ? (
                <>
                  <strong>Tie:</strong> {winnerRows.map(w => w.label).join(' and ')} share max slope{' '}
                  <code className="font-mono">{maxSlope.toFixed(3)}</code>. D8 breaks ties by scan order —
                  water flows <strong>→ {d8Winner?.label}</strong>{' '}
                  <span className="text-emerald-700">(D8 code: {d8Winner?.code})</span>
                </>
              ) : (
                <>
                  Water flows <strong>→ {d8Winner?.label}</strong>{' '}
                  <span className="text-emerald-700">(D8 code: {d8Winner?.code})</span>
                  {' '}with slope <code className="font-mono">{maxSlope.toFixed(3)}</code>
                </>
              )}
            </div>
          ) : (
            <div className="rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900">
              <strong>⚠ This cell is a PIT — no downhill neighbor exists.</strong>
              {' '}D8 assigns no flow direction here. Pit-filling algorithms resolve this.
            </div>
          )}

          {/* ESRI code diagram */}
          <div>
            <p className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider mb-1.5">ESRI D8 Code Diagram</p>
            <table className="border-collapse text-xs text-center font-mono">
              <tbody>
                {[0, 1, 2].map(row => (
                  <tr key={row}>
                    {[0, 1, 2].map(col => {
                      const val = ESRI_DIAGRAM[row * 3 + col];
                      const isCenter = val === '·';
                      return (
                        <td
                          key={col}
                          className={`w-10 h-8 border border-slate-300 font-semibold ${
                            isCenter
                              ? 'bg-yellow-100 text-yellow-800'
                              : 'bg-slate-100 text-slate-600'
                          }`}
                        >
                          {val}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="text-[10px] text-slate-400 mt-1">Powers of 2 assigned clockwise from E. Unique, compact, bitwise-safe.</p>
          </div>

        </div>
      </div>
    </div>
  );
}
