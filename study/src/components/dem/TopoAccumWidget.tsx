'use client';

import React, { useState, useRef, useCallback, useEffect } from 'react';

type GridProps = {
  grid: number[][];
  setGrid: React.Dispatch<React.SetStateAction<number[][]>>;
};

// ─── Constants ────────────────────────────────────────────────────────────────
const CELL = 72;
const DEFAULT_GRID = [
  [7, 6, 5, 4],
  [6, 5, 4, 3],
  [5, 4, 3, 2],
  [4, 3, 2, 1],
];

const D8 = [
  { dr: 0,  dc: 1,  code: 1,   label: 'E',  diag: false },
  { dr: 1,  dc: 1,  code: 2,   label: 'SE', diag: true  },
  { dr: 1,  dc: 0,  code: 4,   label: 'S',  diag: false },
  { dr: 1,  dc: -1, code: 8,   label: 'SW', diag: true  },
  { dr: 0,  dc: -1, code: 16,  label: 'W',  diag: false },
  { dr: -1, dc: -1, code: 32,  label: 'NW', diag: true  },
  { dr: -1, dc: 0,  code: 64,  label: 'N',  diag: false },
  { dr: -1, dc: 1,  code: 128, label: 'NE', diag: true  },
];

// ─── Color helpers ────────────────────────────────────────────────────────────
const ELEV_STOPS: [number, [number, number, number]][] = [
  [0.00, [67,  117, 180]],
  [0.18, [116, 196, 163]],
  [0.40, [161, 218, 115]],
  [0.62, [255, 230, 130]],
  [0.82, [200, 130, 70 ]],
  [1.00, [245, 245, 244]],
];

function elevColor(z: number, lo: number, hi: number): string {
  const t = hi === lo ? 0.5 : (z - lo) / (hi - lo);
  let a = ELEV_STOPS[0], b = ELEV_STOPS[ELEV_STOPS.length - 1];
  for (let i = 0; i < ELEV_STOPS.length - 1; i++) {
    if (t >= ELEV_STOPS[i][0] && t <= ELEV_STOPS[i + 1][0]) {
      a = ELEV_STOPS[i]; b = ELEV_STOPS[i + 1]; break;
    }
  }
  const u = (b[0] - a[0]) === 0 ? 0 : (t - a[0]) / (b[0] - a[0]);
  const lp = (x: number, y: number) => Math.round(x + u * (y - x));
  return `rgb(${lp(a[1][0], b[1][0])},${lp(a[1][1], b[1][1])},${lp(a[1][2], b[1][2])})`;
}

// Topo-order color: step 1 → red, step 16 → blue
function topoStepColor(step: number, total: number): string {
  const t = total <= 1 ? 0 : (step - 1) / (total - 1);
  const r = Math.round(239 + t * (37  - 239));
  const g = Math.round(68  + t * (99  - 68 ));
  const bv = Math.round(68  + t * (235 - 68 ));
  return `rgb(${r},${g},${bv})`;
}

// FA log-scale color: white → sky → navy
function faColor(fa: number, maxFA: number): string {
  if (maxFA <= 1) return 'rgb(241,245,249)';
  const t = Math.log(fa) / Math.log(maxFA);
  const stops: [number, [number, number, number]][] = [
    [0,   [255, 255, 255]],
    [0.4, [147, 197, 253]],
    [0.7, [56,  189, 248]],
    [1,   [2,   132, 199]],
  ];
  let a = stops[0], b = stops[stops.length - 1];
  for (let i = 0; i < stops.length - 1; i++) {
    if (t >= stops[i][0] && t <= stops[i + 1][0]) { a = stops[i]; b = stops[i + 1]; break; }
  }
  const u = (b[0] - a[0]) === 0 ? 0 : (t - a[0]) / (b[0] - a[0]);
  const lp = (x: number, y: number) => Math.round(x + u * (y - x));
  return `rgb(${lp(a[1][0], b[1][0])},${lp(a[1][1], b[1][1])},${lp(a[1][2], b[1][2])})`;
}

// ─── Algorithm helpers ────────────────────────────────────────────────────────
function topoSort(grid: number[][]): { r: number; c: number }[] {
  const R = grid.length, C = grid[0].length;
  const cells: { r: number; c: number }[] = [];
  for (let r = 0; r < R; r++) for (let c = 0; c < C; c++) cells.push({ r, c });
  return cells.sort(
    (a, b) => grid[b.r][b.c] - grid[a.r][a.c] || (a.r * C + a.c) - (b.r * C + b.c)
  );
}

function computeFlow(grid: number[][]): (typeof D8[0] | null)[][] {
  const R = grid.length, C = grid[0].length;
  return grid.map((row, r) =>
    row.map((_, c) => {
      let best: typeof D8[0] | null = null, mx = 0;
      for (const d of D8) {
        const nr = r + d.dr, nc = c + d.dc;
        if (nr < 0 || nr >= R || nc < 0 || nc >= C) continue;
        const s = (grid[r][c] - grid[nr][nc]) / (d.diag ? Math.SQRT2 : 1);
        if (s > mx) { mx = s; best = d; }
      }
      return best;
    })
  );
}

// ─── SVG Arrow (line + arrowhead from cell center toward neighbor) ────────────
function Arrow({ x1, y1, x2, y2, color = 'rgba(255,255,255,0.7)' }: {
  x1: number; y1: number; x2: number; y2: number; color?: string;
}) {
  const dx = x2 - x1, dy = y2 - y1;
  const len = Math.sqrt(dx * dx + dy * dy) || 1;
  const ux = dx / len, uy = dy / len;
  const hs = 8;
  const hx = x2 - hs * ux, hy = y2 - hs * uy;
  return (
    <g>
      <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={color} strokeWidth={2} strokeLinecap="round" />
      <polygon
        points={`${x2},${y2} ${hx + hs * 0.4 * uy},${hy - hs * 0.4 * ux} ${hx - hs * 0.4 * uy},${hy + hs * 0.4 * ux}`}
        fill={color}
      />
    </g>
  );
}

// ─── Utility atoms ────────────────────────────────────────────────────────────
function InfoCard({ label, value, accent }: { label: string; value: string; accent: string }) {
  const s: Record<string, string> = {
    blue:   'bg-sky-50 border-sky-100 text-sky-900',
    teal:   'bg-teal-50 border-teal-100 text-teal-900',
    amber:  'bg-amber-50 border-amber-100 text-amber-900',
    violet: 'bg-violet-50 border-violet-100 text-violet-900',
    green:  'bg-emerald-50 border-emerald-100 text-emerald-900',
    red:    'bg-red-50 border-red-100 text-red-900',
  };
  return (
    <div className={`rounded-xl border px-3 py-2.5 flex flex-col gap-0.5 ${s[accent] ?? s.blue}`}>
      <span className="text-[11px] font-medium opacity-60 uppercase tracking-wide">{label}</span>
      <span className="font-bold text-base leading-tight">{value}</span>
    </div>
  );
}

function CalloutBox({ type, children }: { type: 'tip' | 'info' | 'warn'; children: React.ReactNode }) {
  const s = {
    tip:  'bg-emerald-50 border-emerald-200 text-emerald-900',
    info: 'bg-slate-50 border-slate-200 text-slate-700',
    warn: 'bg-amber-50 border-amber-200 text-amber-900',
  };
  return (
    <div className={`rounded-xl border p-3.5 text-sm leading-relaxed ${s[type]}`}>
      {children}
    </div>
  );
}

// ─── Tab 1: Topological Order ─────────────────────────────────────────────────
function TopoTab({ grid, setGrid }: GridProps) {
  const R = grid.length, C = grid[0].length;
  const zMin = Math.min(...grid.flat()), zMax = Math.max(...grid.flat());
  const sorted = topoSort(grid);
  const total = sorted.length;

  // orderMap[r][c] = 1-based processing order index
  const orderMap: number[][] = Array.from({ length: R }, () => Array(C).fill(0));
  sorted.forEach(({ r, c }, i) => { orderMap[r][c] = i + 1; });

  const [revealed, setRevealed]   = useState(0);
  const [flashing, setFlashing]   = useState<number | null>(null);
  const [animating, setAnimating] = useState(false);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const mountRef = useRef(true);

  // Reset animation when terrain changes
  useEffect(() => {
    if (mountRef.current) { mountRef.current = false; return; }
    if (timerRef.current) clearTimeout(timerRef.current);
    setRevealed(0); setFlashing(null); setAnimating(false);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [grid]);

  const clearTimers = () => { if (timerRef.current) clearTimeout(timerRef.current); };

  const runAnim = useCallback((step: number) => {
    if (step > total) { setAnimating(false); setFlashing(null); return; }
    setFlashing(step);
    timerRef.current = setTimeout(() => {
      setRevealed(step);
      setFlashing(null);
      timerRef.current = setTimeout(() => runAnim(step + 1), 80);
    }, 280);
  }, [total]);

  const handleShowOrder = () => {
    if (animating) return;
    clearTimers();
    setRevealed(0);
    setFlashing(null);
    setAnimating(true);
    timerRef.current = setTimeout(() => runAnim(1), 120);
  };

  const handleReset = () => {
    clearTimers();
    setRevealed(0);
    setFlashing(null);
    setAnimating(false);
  };

  const done = revealed === total && !animating && !flashing;

  return (
    <div className="p-6 flex flex-col lg:flex-row gap-8 items-start">
      {/* Left: grid */}
      <div className="flex-shrink-0 flex flex-col items-center gap-3">
        <svg width={C * CELL} height={R * CELL}
          className="rounded-xl border border-slate-200 shadow-sm select-none cursor-pointer"
          onContextMenu={e => e.preventDefault()}
        >
          {grid.map((row, r) =>
            row.map((z, c) => {
              const order = orderMap[r][c];
              const isFlashing = flashing !== null && order === flashing;
              const isStamped  = order <= revealed;

              let fill = elevColor(z, zMin, zMax);
              if (isFlashing) fill = '#fef08a';
              else if (isStamped) fill = topoStepColor(order, total);

              const cx = c * CELL + CELL / 2, cy = r * CELL + CELL / 2;
              const darkBg = isStamped && !isFlashing;
              const textCol = darkBg
                ? '#fff'
                : (z > (zMin + zMax) / 2 ? '#1e293b' : '#f0f9ff');

              return (
                <g key={`${r}-${c}`}
                  onClick={() => setGrid(prev => { const g=prev.map(row=>[...row]); g[r][c]=Math.min(15,g[r][c]+1); return g; })}
                  onContextMenu={e => { e.preventDefault(); setGrid(prev => { const g=prev.map(row=>[...row]); g[r][c]=Math.max(1,g[r][c]-1); return g; }); }}
                >
                  <rect
                    x={c * CELL} y={r * CELL} width={CELL} height={CELL}
                    fill={fill}
                    stroke={isFlashing ? '#ca8a04' : '#e2e8f0'}
                    strokeWidth={isFlashing ? 2.5 : 0.8}
                  />
                  {/* Elevation value, centered */}
                  <text
                    x={cx} y={cy + 5} textAnchor="middle"
                    fontSize={14} fontWeight="700" fill={textCol}
                    style={{ pointerEvents: 'none' }}
                  >
                    {z}
                  </text>
                  {/* Stamped order number in top-left corner */}
                  {isStamped && (
                    <text
                      x={c * CELL + 5} y={r * CELL + 14}
                      textAnchor="start" fontSize={11} fontWeight="800"
                      fill="rgba(255,255,255,0.92)"
                      style={{ pointerEvents: 'none' }}
                    >
                      {order}
                    </text>
                  )}
                  {/* Flash indicator */}
                  {isFlashing && (
                    <text
                      x={cx} y={cy - 18} textAnchor="middle"
                      fontSize={10} fontWeight="bold" fill="#92400e"
                      style={{ pointerEvents: 'none' }}
                    >
                      ▼
                    </text>
                  )}
                </g>
              );
            })
          )}
        </svg>

        <p className="text-xs text-slate-400">Left-click: raise · Right-click: lower</p>

        {/* Color legend for topo order */}
        {revealed > 0 && (
          <div className="flex items-center gap-2 w-full text-xs text-slate-500 px-1">
            <span className="text-red-600 font-semibold">① upstream</span>
            <div
              className="h-2.5 flex-1 rounded-full"
              style={{ background: 'linear-gradient(to right, rgb(239,68,68), rgb(168,85,247), rgb(37,99,235))' }}
            />
            <span className="text-blue-700 font-semibold">⑯ outlet</span>
          </div>
        )}

        {/* Buttons */}
        <div className="flex gap-2 flex-wrap">
          <button
            onClick={handleShowOrder}
            disabled={animating}
            className="px-4 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 text-white text-sm font-semibold transition-colors"
          >
            {animating ? '⏳ Animating…' : '▶ Show Processing Order'}
          </button>
          <button
            onClick={handleReset}
            className="px-4 py-2 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-700 text-sm font-semibold"
          >
            ↺ Reset
          </button>
        </div>
      </div>

      {/* Right panel */}
      <div className="flex-1 min-w-0 flex flex-col gap-4">
        {/* Explanation box — always visible */}
        <CalloutBox type="tip">
          <strong>Why topological order?</strong> On a pit-free DEM,{' '}
          <strong>sorting cells by elevation descending</strong> gives a valid topological order.
          An upstream cell (high&nbsp;z) always appears before its downstream neighbor (low&nbsp;z).
          This ensures each cell's FA is fully accumulated before it donates downstream.
        </CalloutBox>

        {/* Ordered list */}
        <div>
          <p className="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-2">
            Processing sequence — highest elevation first
          </p>
          <div className="rounded-xl border border-slate-100 bg-slate-50 overflow-hidden max-h-72 overflow-y-auto">
            {sorted.map(({ r, c }, i) => {
              const step = i + 1;
              const isRevd = step <= revealed;
              const isFlsh = flashing === step;
              const isFirst = step === 1, isLast = step === total;
              return (
                <div
                  key={`${r}-${c}`}
                  className={`flex items-center gap-2 px-3 py-1.5 border-b border-slate-100 last:border-0 text-sm transition-colors
                    ${isFlsh ? 'bg-yellow-100' : isRevd ? 'bg-white' : 'opacity-30'}`}
                >
                  <span
                    className="w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold flex-shrink-0 text-white"
                    style={{ background: isRevd || isFlsh ? topoStepColor(step, total) : '#cbd5e1' }}
                  >
                    {step}
                  </span>
                  <span className="font-mono text-slate-600">
                    ({r},{c}) z={grid[r][c]}
                  </span>
                  {isFirst && isRevd && (
                    <span className="ml-auto text-[10px] font-semibold text-red-500 bg-red-50 rounded px-1.5 py-0.5">
                      most upstream
                    </span>
                  )}
                  {isLast && isRevd && (
                    <span className="ml-auto text-[10px] font-semibold text-blue-600 bg-blue-50 rounded px-1.5 py-0.5">
                      outlet
                    </span>
                  )}
                </div>
              );
            })}
          </div>
        </div>

        {/* Highlights after animation */}
        {done && (
          <div className="grid grid-cols-2 gap-2">
            <InfoCard label="① (0,0) z=7" value="Most upstream" accent="red" />
            <InfoCard label="⑯ (3,3) z=1" value="Outlet cell" accent="blue" />
          </div>
        )}

        {!revealed && !animating && (
          <p className="text-sm text-slate-400 italic">
            Click "Show Processing Order" to animate the topological sort.
          </p>
        )}
      </div>
    </div>
  );
}

// ─── Tab 2: Flow Accumulation ─────────────────────────────────────────────────
type FAPhase = 'idle' | 'running' | 'paused' | 'done';

function AccumTab({ grid, setGrid }: GridProps) {
  const R = grid.length, C = grid[0].length;
  const flowDir = computeFlow(grid);
  const sorted  = topoSort(grid);
  const total   = sorted.length;

  const [fa, setFa]               = useState<number[][]>(() => Array.from({ length: R }, () => Array(C).fill(1)));
  const [step, setStep]           = useState(-1);
  const [phase, setPhase]         = useState<FAPhase>('idle');
  const [curCell, setCurCell]     = useState<{ r: number; c: number } | null>(null);
  const [downCell, setDownCell]   = useState<{ r: number; c: number } | null>(null);
  const [donating, setDonating]   = useState(false); // purple flash active
  const [threshold, setThreshold] = useState(2);
  const [speed, setSpeed]         = useState(320);

  const timerRef  = useRef<ReturnType<typeof setTimeout> | null>(null);
  const faRef     = useRef<number[][]>(Array.from({ length: R }, () => Array(C).fill(1)));
  const phaseRef  = useRef<FAPhase>('idle');
  const stepRef   = useRef(-1);

  const clearTimers = () => { if (timerRef.current) clearTimeout(timerRef.current); };

  const accumMountRef = useRef(true);
  useEffect(() => {
    if (accumMountRef.current) { accumMountRef.current = false; return; }
    clearTimers();
    const fresh = Array.from({ length: R }, () => Array(C).fill(1));
    faRef.current = fresh; phaseRef.current = 'idle'; stepRef.current = -1;
    setFa(fresh.map(r => [...r])); setStep(-1); setPhase('idle');
    setCurCell(null); setDownCell(null); setDonating(false); setThreshold(2);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [grid]);

  const maxFA = Math.max(...fa.flat());

  const reset = () => {
    clearTimers();
    const fresh = Array.from({ length: R }, () => Array(C).fill(1));
    faRef.current = fresh;
    phaseRef.current = 'idle';
    stepRef.current = -1;
    setFa(fresh.map(r => [...r]));
    setStep(-1);
    setPhase('idle');
    setCurCell(null);
    setDownCell(null);
    setDonating(false);
  };

  const runStep = useCallback((idx: number, faArr: number[][]) => {
    if (phaseRef.current === 'paused') return;
    if (idx >= total) {
      phaseRef.current = 'done';
      setPhase('done');
      setCurCell(null);
      setDownCell(null);
      setDonating(false);
      return;
    }

    const { r, c } = sorted[idx];
    setCurCell({ r, c });
    setDownCell(null);
    setDonating(false);
    setStep(idx);

    const dir = flowDir[r][c];
    if (dir) {
      const nr = r + dir.dr, nc = c + dir.dc;
      // After a short pause on the yellow cell, do the donation
      timerRef.current = setTimeout(() => {
        if (phaseRef.current === 'paused') return;
        const next = faArr.map(row => [...row]);
        next[nr][nc] += next[r][c];
        faRef.current = next;
        setFa(next.map(row => [...row]));
        setDownCell({ r: nr, c: nc });
        setDonating(true);
        timerRef.current = setTimeout(() => {
          if (phaseRef.current === 'paused') return;
          setDonating(false);
          stepRef.current = idx + 1;
          runStep(idx + 1, next);
        }, Math.max(speed * 0.4, 100));
      }, speed * 0.6);
    } else {
      // No downstream (edge cell / pit) — just advance
      timerRef.current = setTimeout(() => {
        if (phaseRef.current === 'paused') return;
        stepRef.current = idx + 1;
        runStep(idx + 1, faArr);
      }, speed * 0.5);
    }
  }, [total, speed, flowDir, sorted]);

  const start = () => {
    clearTimers();
    const fresh = Array.from({ length: R }, () => Array(C).fill(1));
    faRef.current = fresh;
    phaseRef.current = 'running';
    stepRef.current = 0;
    setFa(fresh.map(r => [...r]));
    setPhase('running');
    setStep(0);
    setCurCell(null);
    setDownCell(null);
    runStep(0, fresh);
  };

  const pause = () => {
    clearTimers();
    phaseRef.current = 'paused';
    setPhase('paused');
  };

  const resume = () => {
    phaseRef.current = 'running';
    setPhase('running');
    runStep(stepRef.current, faRef.current);
  };

  const stepOnce = () => {
    clearTimers();
    if (phase === 'idle') {
      const fresh = Array.from({ length: R }, () => Array(C).fill(1));
      faRef.current = fresh;
      setFa(fresh.map(r => [...r]));
      stepRef.current = 0;
    }
    phaseRef.current = 'paused';
    setPhase('paused');
    const idx = stepRef.current < 0 ? 0 : stepRef.current;
    if (idx >= total) { phaseRef.current = 'done'; setPhase('done'); return; }

    const { r, c } = sorted[idx];
    setCurCell({ r, c });
    const dir = flowDir[r][c];
    const next = faRef.current.map(row => [...row]);
    if (dir) {
      const nr = r + dir.dr, nc = c + dir.dc;
      next[nr][nc] += next[r][c];
      faRef.current = next;
      setFa(next.map(row => [...row]));
      setDownCell({ r: nr, c: nc });
      setDonating(true);
      timerRef.current = setTimeout(() => setDonating(false), 300);
    }
    stepRef.current = idx + 1;
    setStep(idx);
  };

  const skipToEnd = () => {
    clearTimers();
    const faArr = Array.from({ length: R }, () => Array(C).fill(1));
    for (const { r, c } of sorted) {
      const dir = flowDir[r][c];
      if (dir) faArr[r + dir.dr][c + dir.dc] += faArr[r][c];
    }
    faRef.current = faArr;
    phaseRef.current = 'done';
    stepRef.current = total;
    setFa(faArr.map(row => [...row]));
    setPhase('done');
    setStep(total - 1);
    setCurCell(null);
    setDownCell(null);
    setDonating(false);
  };

  const isDone = phase === 'done';
  const isIdle = phase === 'idle';

  // Build info text
  let infoText: React.ReactNode;
  if (isIdle) {
    infoText = (
      <>
        Each cell starts with <strong>FA = 1</strong> (itself). Processing highest-elevation cells first ensures
        upstream donors are counted before downstream cells receive.
      </>
    );
  } else if (isDone) {
    infoText = (
      <>
        <strong>Max FA = {maxFA}</strong> at (3,3) — all {maxFA} cells drain to the bottom-right corner on
        this uniform SE slope. Set τ ≤ 4 to highlight the diagonal drainage spine.
      </>
    );
  } else if (curCell) {
    const dir = flowDir[curCell.r][curCell.c];
    const faVal = fa[curCell.r]?.[curCell.c] ?? 1;
    infoText = (
      <>
        <strong>Step {step + 1}/{total}:</strong> Cell ({curCell.r},{curCell.c}) FA={faVal}
        {dir
          ? <> → donates to ({curCell.r + dir.dr},{curCell.c + dir.dc})</>
          : <> → no downstream (edge/outlet)</>
        }
      </>
    );
  }

  return (
    <div className="p-6 flex flex-col lg:flex-row gap-8 items-start">
      {/* Left: FA grid */}
      <div className="flex-shrink-0 flex flex-col items-center gap-3">
        <svg width={C * CELL} height={R * CELL}
          className="rounded-xl border border-slate-200 shadow-sm select-none cursor-pointer"
          onContextMenu={e => e.preventDefault()}
        >
          {grid.map((row, r) =>
            row.map((_, c) => {
              const faVal = fa[r][c];
              const isCur  = curCell  && curCell.r  === r && curCell.c  === c;
              const isDown = downCell && downCell.r === r && downCell.c === c && donating;
              const isStream = isDone && faVal >= threshold;

              let fill = isIdle ? 'rgb(241,245,249)' : faColor(faVal, maxFA);
              if (isCur)  fill = '#fef08a';  // yellow — processing
              if (isDown) fill = '#a78bfa';  // purple — receiving donation

              const cx = c * CELL + CELL / 2, cy = r * CELL + CELL / 2;
              // Text color: dark on light cells, white on dark
              const brightness = isCur || isDown ? 0.3 : (faVal / (maxFA || 1));
              const txtCol = brightness > 0.55 ? '#1e293b' : '#f0f9ff';

              // Arrow: from center toward downstream neighbor
              const dir = flowDir[r][c];
              let arrowEl: React.ReactNode = null;
              if (!isIdle && dir) {
                const tx = cx + dir.dc * CELL * 0.36;
                const ty = cy + dir.dr * CELL * 0.36;
                arrowEl = <Arrow x1={cx} y1={cy} x2={tx} y2={ty} color={isCur ? 'rgba(180,83,9,0.9)' : 'rgba(255,255,255,0.65)'} />;
              }

              return (
                <g key={`${r}-${c}`}
                  onClick={() => setGrid(prev => { const g=prev.map(row=>[...row]); g[r][c]=Math.min(15,g[r][c]+1); return g; })}
                  onContextMenu={e => { e.preventDefault(); setGrid(prev => { const g=prev.map(row=>[...row]); g[r][c]=Math.max(1,g[r][c]-1); return g; }); }}
                >
                  <rect
                    x={c * CELL} y={r * CELL} width={CELL} height={CELL}
                    fill={fill}
                    stroke={
                      isCur ? '#d97706' :
                      isDown ? '#7c3aed' :
                      isStream ? '#0369a1' :
                      '#e2e8f0'
                    }
                    strokeWidth={isCur || isDown ? 2.5 : isStream ? 2 : 0.8}
                  />

                  {/* FA value */}
                  {!isIdle && (
                    <text
                      x={cx} y={cy + 5} textAnchor="middle"
                      fontSize={faVal >= 10 ? 12 : 15} fontWeight="800"
                      fill={isCur ? '#92400e' : isDown ? '#4c1d95' : txtCol}
                      style={{ pointerEvents: 'none' }}
                    >
                      {faVal}
                    </text>
                  )}

                  {/* Idle: elevation value */}
                  {isIdle && (
                    <text x={cx} y={cy + 5} textAnchor="middle"
                      fontSize={14} fontWeight="700" fill="#64748b"
                      style={{ pointerEvents: 'none' }}>
                      FA=1
                    </text>
                  )}

                  {arrowEl}

                  {/* Label current cell */}
                  {isCur && (
                    <text x={cx} y={r * CELL + 14} textAnchor="middle"
                      fontSize={9} fontWeight="bold" fill="#92400e"
                      style={{ pointerEvents: 'none' }}>
                      proc
                    </text>
                  )}
                  {/* Label downstream */}
                  {isDown && (
                    <text x={cx} y={r * CELL + 14} textAnchor="middle"
                      fontSize={9} fontWeight="bold" fill="#4c1d95"
                      style={{ pointerEvents: 'none' }}>
                      recv
                    </text>
                  )}

                  {/* Stream overlay badge */}
                  {isStream && (
                    <rect
                      x={c * CELL + 2} y={r * CELL + 2}
                      width={8} height={8} rx={2}
                      fill="#0369a1" opacity={0.7}
                    />
                  )}
                </g>
              );
            })
          )}
        </svg>

        {/* FA legend */}
        {!isIdle && (
          <div className="flex items-center gap-2 w-full text-xs text-slate-500 px-1">
            <span>FA=1</span>
            <div className="h-2.5 flex-1 rounded-full"
              style={{ background: 'linear-gradient(to right, rgb(255,255,255), rgb(147,197,253), rgb(56,189,248), rgb(2,132,199))' }} />
            <span>FA={maxFA}</span>
          </div>
        )}

        {/* Color key */}
        <div className="flex gap-3 text-xs text-slate-500 flex-wrap">
          <span className="flex items-center gap-1">
            <span className="w-3 h-3 rounded bg-yellow-300 inline-block" /> Processing
          </span>
          <span className="flex items-center gap-1">
            <span className="w-3 h-3 rounded bg-violet-400 inline-block" /> Receiving
          </span>
          {isDone && (
            <span className="flex items-center gap-1">
              <span className="w-3 h-3 rounded bg-sky-700 inline-block" /> Stream (FA≥τ)
            </span>
          )}
        </div>
      </div>

      {/* Right panel */}
      <div className="flex-1 min-w-0 flex flex-col gap-4">
        {/* Controls */}
        <div className="flex flex-wrap gap-2">
          {isIdle && (
            <button onClick={start}
              className="px-4 py-2 rounded-xl bg-sky-600 hover:bg-sky-700 text-white text-sm font-semibold transition-colors">
              ▶ Play
            </button>
          )}
          {phase === 'running' && (
            <button onClick={pause}
              className="px-4 py-2 rounded-xl bg-amber-500 hover:bg-amber-600 text-white text-sm font-semibold">
              ⏸ Pause
            </button>
          )}
          {phase === 'paused' && (
            <button onClick={resume}
              className="px-4 py-2 rounded-xl bg-sky-600 hover:bg-sky-700 text-white text-sm font-semibold">
              ▶ Resume
            </button>
          )}
          {(isIdle || phase === 'paused') && (
            <button onClick={stepOnce}
              className="px-4 py-2 rounded-xl bg-indigo-500 hover:bg-indigo-600 text-white text-sm font-semibold">
              ⏭ Step
            </button>
          )}
          {!isDone && (
            <button onClick={skipToEnd}
              className="px-4 py-2 rounded-xl bg-slate-500 hover:bg-slate-600 text-white text-sm font-semibold">
              ⏭⏭ Skip to End
            </button>
          )}
          <button onClick={reset}
            className="px-4 py-2 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-700 text-sm font-semibold">
            ↺ Reset
          </button>
        </div>

        {/* Speed slider */}
        {!isDone && (
          <div>
            <label className="text-xs font-semibold text-slate-400 uppercase tracking-wide">
              Animation speed
            </label>
            <div className="flex items-center gap-3 mt-1">
              <span className="text-xs text-slate-400">Fast</span>
              <input type="range" min={80} max={800} step={40} value={speed}
                onChange={e => setSpeed(Number(e.target.value))}
                className="flex-1 accent-sky-600" />
              <span className="text-xs text-slate-400">Slow</span>
            </div>
          </div>
        )}

        {/* Stream threshold slider (after done) */}
        {isDone && (
          <div>
            <label className="text-xs font-semibold text-slate-400 uppercase tracking-wide">
              Stream threshold τ = {threshold}
            </label>
            <input type="range" min={1} max={maxFA} step={1} value={threshold}
              onChange={e => setThreshold(Number(e.target.value))}
              className="w-full mt-1 accent-sky-700" />
            <p className="text-xs text-slate-400 mt-1">
              τ = {threshold} — <strong>{fa.flat().filter(v => v >= threshold).length}</strong> cells form the stream network
            </p>
          </div>
        )}

        {/* Step counter */}
        {!isIdle && (
          <div className="grid grid-cols-2 gap-2">
            <InfoCard
              label={isDone ? 'Steps completed' : `Step`}
              value={isDone ? `${total}/${total}` : `${step + 1}/${total}`}
              accent="teal"
            />
            <InfoCard label="Max FA" value={String(maxFA)} accent="blue" />
            {curCell && !isDone && (
              <InfoCard label="Current cell" value={`(${curCell.r},${curCell.c})`} accent="amber" />
            )}
            {curCell && !isDone && flowDir[curCell.r][curCell.c] && (
              <InfoCard
                label="Donates to"
                value={`(${curCell.r + flowDir[curCell.r][curCell.c]!.dr},${curCell.c + flowDir[curCell.r][curCell.c]!.dc})`}
                accent="violet"
              />
            )}
          </div>
        )}

        {/* Info callout */}
        <CalloutBox type={isDone ? 'tip' : 'info'}>
          {infoText}
        </CalloutBox>

        {/* Algorithm note */}
        {isIdle && (
          <CalloutBox type="tip">
            <strong>Algorithm:</strong> Initialize FA[r][c] = 1 for all cells. Then process in topological order
            (highest elevation first). For each cell, add its FA to its single D8 downstream neighbor.
            The total at any cell = number of upstream cells that drain through it.
          </CalloutBox>
        )}
      </div>
    </div>
  );
}

// ─── Main widget ──────────────────────────────────────────────────────────────
type TabId = 'topo' | 'accum';

export default function TopoAccumWidget() {
  const [tab, setTab] = useState<TabId>('topo');
  const [grid, setGrid] = useState<number[][]>(() => DEFAULT_GRID.map(r => [...r]));

  return (
    <div className="not-prose my-8 rounded-2xl border border-slate-200 bg-white shadow-xl overflow-hidden font-sans">

      {/* Header */}
      <div className="bg-gradient-to-r from-indigo-600 to-sky-700 px-6 py-4">
        <h3 className="text-white font-bold text-lg tracking-tight">
          Topological Order &amp; Flow Accumulation
        </h3>
        <p className="text-indigo-100 text-sm mt-0.5">
          Why upstream-first processing is required · Watch FA propagate cell by cell
        </p>
      </div>

      {/* Tab bar */}
      <div className="flex border-b border-slate-200 bg-slate-50">
        {([
          { id: 'topo',  label: '📋 Topological Order' },
          { id: 'accum', label: '💧 Flow Accumulation' },
        ] as { id: TabId; label: string }[]).map(({ id, label }) => (
          <button
            key={id}
            onClick={() => setTab(id)}
            className={`px-5 py-3 text-sm font-semibold transition-colors border-b-2 -mb-px
              ${tab === id
                ? 'border-indigo-600 text-indigo-700 bg-white'
                : 'border-transparent text-slate-500 hover:text-slate-700 hover:bg-slate-100'
              }`}
          >
            {label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      {tab === 'topo'  && <TopoTab  grid={grid} setGrid={setGrid} />}
      {tab === 'accum' && <AccumTab grid={grid} setGrid={setGrid} />}
    </div>
  );
}
