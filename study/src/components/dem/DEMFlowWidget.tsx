'use client';

import React, { useState, useRef, useCallback } from 'react';

// ─── Types ───────────────────────────────────────────────────────────────────
type Grid = number[][];
interface D8Dir { dr: number; dc: number; code: number; angle: number; label: string; emoji: string; }
type FlowCell = D8Dir | null;
type FlowGrid  = FlowCell[][];
type TabMode   = 'terrain' | 'flowdir' | 'calculate' | 'raindrop';

// ─── D8 lookup — ESRI encoding, scan order: E SE S SW W NW N NE ─────────────
const D8: D8Dir[] = [
  { dr:  0, dc:  1, code:   1, angle:  90, label: 'E',  emoji: '→' },
  { dr:  1, dc:  1, code:   2, angle: 135, label: 'SE', emoji: '↘' },
  { dr:  1, dc:  0, code:   4, angle: 180, label: 'S',  emoji: '↓' },
  { dr:  1, dc: -1, code:   8, angle: 225, label: 'SW', emoji: '↙' },
  { dr:  0, dc: -1, code:  16, angle: 270, label: 'W',  emoji: '←' },
  { dr: -1, dc: -1, code:  32, angle: 315, label: 'NW', emoji: '↖' },
  { dr: -1, dc:  0, code:  64, angle:   0, label: 'N',  emoji: '↑' },
  { dr: -1, dc:  1, code: 128, angle:  45, label: 'NE', emoji: '↗' },
];

// ─── Terrain presets ─────────────────────────────────────────────────────────
const PRESETS: Record<string, Grid> = {
  slope: [
    [9,8,7,6,5,4,3,2],
    [9,8,7,6,5,4,3,2],
    [9,8,7,6,5,4,3,2],
    [9,8,7,6,5,4,3,2],
    [9,8,7,6,5,4,3,2],
    [9,8,7,6,5,4,3,2],
    [9,8,7,6,5,4,3,2],
    [9,8,7,6,5,4,3,2],
  ],
  valley: [
    [9,8,7,6,6,7,8,9],
    [8,7,6,5,5,6,7,8],
    [7,6,5,4,4,5,6,7],
    [6,5,4,3,3,4,5,6],
    [5,4,3,2,2,3,4,5],
    [4,3,2,1,1,2,3,4],
    [4,3,2,1,1,2,3,4],
    [3,2,1,1,1,1,2,3],
  ],
  ridge: [
    [4,5,6,9,9,6,5,4],
    [3,4,5,8,8,5,4,3],
    [2,3,4,7,7,4,3,2],
    [1,2,3,6,6,3,2,1],
    [1,2,3,6,6,3,2,1],
    [2,3,4,7,7,4,3,2],
    [3,4,5,8,8,5,4,3],
    [4,5,6,9,9,6,5,4],
  ],
  basin: [
    [9,9,9,9,9,9,9,9],
    [9,7,7,7,7,7,7,9],
    [9,7,5,5,5,5,7,9],
    [9,7,5,3,3,5,7,9],
    [9,7,5,3,3,5,7,9],
    [9,7,5,5,5,5,7,9],
    [9,7,7,7,7,7,7,9],
    [9,9,9,9,9,9,9,9],
  ],
  mountain: [
    [2,2,3,3,3,3,2,2],
    [2,3,4,5,5,4,3,2],
    [3,4,6,7,7,6,4,3],
    [3,5,7,9,9,7,5,3],
    [3,5,7,9,9,7,5,3],
    [3,4,6,7,7,6,4,3],
    [2,3,4,5,5,4,3,2],
    [2,2,3,3,3,3,2,2],
  ],
};

// ─── Color mapping ────────────────────────────────────────────────────────────
const STOPS: [number,[number,number,number]][] = [
  [0.00,[ 67,117,180]],[0.18,[116,196,163]],[0.40,[161,218,115]],
  [0.62,[255,230,130]],[0.82,[200,130, 70]],[1.00,[245,245,244]],
];
function elevColor(z: number, lo: number, hi: number): string {
  const t = hi===lo ? 0.5 : (z-lo)/(hi-lo);
  let a=STOPS[0], b=STOPS[STOPS.length-1];
  for (let i=0;i<STOPS.length-1;i++) if(t>=STOPS[i][0]&&t<=STOPS[i+1][0]){a=STOPS[i];b=STOPS[i+1];break;}
  const u=(b[0]-a[0])===0?0:(t-a[0])/(b[0]-a[0]);
  const lp=(x:number,y:number)=>Math.round(x+u*(y-x));
  return `rgb(${lp(a[1][0],b[1][0])},${lp(a[1][1],b[1][1])},${lp(a[1][2],b[1][2])})`;
}

// ─── D8 algorithm ─────────────────────────────────────────────────────────────
function computeFlowGrid(grid: Grid): FlowGrid {
  const R=grid.length, C=grid[0].length;
  return grid.map((row,r)=>row.map((_,c)=>{
    let best: FlowCell=null, maxS=0;
    for(const d of D8){
      const nr=r+d.dr, nc=c+d.dc;
      if(nr<0||nr>=R||nc<0||nc>=C) continue;
      const dist=d.dr!==0&&d.dc!==0?Math.SQRT2:1;
      const s=(grid[r][c]-grid[nr][nc])/dist;
      if(s>maxS){maxS=s;best=d;}
    }
    return best;
  }));
}

// Per-cell slope breakdown for every D8 direction
interface SlopeRow {
  d: D8Dir; zNeigh: number|null; isEdge: boolean;
  dist: number; dz: number; slope: number|null;
}
function slopeBreakdown(grid: Grid, r: number, c: number, cellM: number): SlopeRow[] {
  const R=grid.length, C=grid[0].length, z0=grid[r][c];
  return D8.map(d=>{
    const nr=r+d.dr, nc=c+d.dc;
    if(nr<0||nr>=R||nc<0||nc>=C) return{d,zNeigh:null,isEdge:true,dist:0,dz:0,slope:null};
    const zNeigh=grid[nr][nc];
    const dist=d.dr!==0&&d.dc!==0?Math.SQRT2:1;
    const dz=z0-zNeigh;
    return{d,zNeigh,isEdge:false,dist,dz,slope:dz/(dist*cellM)};
  });
}

// Flow path tracing
function tracePath(fg: FlowGrid, r0:number, c0:number): [number,number][] {
  const path:[number,number][]=[[r0,c0]]; const seen=new Set([`${r0},${c0}`]);
  let r=r0,c=c0;
  for(let i=0;i<200;i++){const d=fg[r][c];if(!d)break;r+=d.dr;c+=d.dc;const k=`${r},${c}`;if(seen.has(k))break;seen.add(k);path.push([r,c]);}
  return path;
}

// Interior pit count
function countPits(fg: FlowGrid): number {
  let n=0;
  fg.forEach((row,r)=>row.forEach((_,c)=>{if(!fg[r][c]&&r>0&&r<fg.length-1&&c>0&&c<fg[0].length-1)n++;}));
  return n;
}

// ─── SVG Arrow ────────────────────────────────────────────────────────────────
function Arrow({angle,cx,cy,r,color='#1e293b'}:{angle:number;cx:number;cy:number;r:number;color?:string}){
  const RAD=(angle-90)*Math.PI/180;
  const tx=cx+r*Math.cos(RAD), ty=cy+r*Math.sin(RAD);
  const bx=cx-r*.55*Math.cos(RAD), by=cy-r*.55*Math.sin(RAD);
  const hs=r*.38;
  const l=(angle-90-25)*Math.PI/180, rr=(angle-90+25)*Math.PI/180;
  return(<g><line x1={bx}y1={by}x2={tx}y2={ty}stroke={color}strokeWidth={2.2}strokeLinecap="round"/>
    <polygon points={`${tx},${ty} ${tx-hs*Math.cos(l)},${ty-hs*Math.sin(l)} ${tx-hs*Math.cos(rr)},${ty-hs*Math.sin(rr)}`}fill={color}/></g>);
}

// ─── Main widget ──────────────────────────────────────────────────────────────
const CELL=54;

export default function DEMFlowWidget() {
  const [grid, setGrid]           = useState<Grid>(()=>PRESETS.slope.map(r=>[...r]));
  const [mode, setMode]           = useState<TabMode>('terrain');
  const [selCell, setSelCell]     = useState<[number,number]|null>(null);
  const [showCodes, setShowCodes] = useState(false);
  const [cellM, setCellM]         = useState(30);        // cell size in meters
  const [flowPath, setFlowPath]   = useState<[number,number][]>([]);
  const [animStep, setAnimStep]   = useState(-1);
  const timerRef = useRef<ReturnType<typeof setTimeout>|null>(null);

  const R=grid.length, C=grid[0].length;
  const zMin=Math.min(...grid.flat()), zMax=Math.max(...grid.flat());
  const fg=computeFlowGrid(grid);

  // ── event handling ────────────────────────────────────────────────────────
  const interact = useCallback((r:number,c:number,isRight:boolean)=>{
    if(mode==='terrain'){
      setGrid(prev=>{const g=prev.map(x=>[...x]);g[r][c]=isRight?Math.max(1,g[r][c]-1):Math.min(15,g[r][c]+1);return g;});
      setFlowPath([]);setAnimStep(-1);
    } else if(mode==='calculate'){
      setSelCell([r,c]);
    } else if(mode==='raindrop'){
      if(timerRef.current)clearTimeout(timerRef.current);
      const path=tracePath(fg,r,c);
      setFlowPath(path);setAnimStep(0);
      let s=0;
      const tick=()=>{s++;setAnimStep(s);if(s<path.length-1)timerRef.current=setTimeout(tick,260);};
      timerRef.current=setTimeout(tick,260);
    }
  },[mode,fg]);

  const switchMode=(m:TabMode)=>{setMode(m);setSelCell(null);setFlowPath([]);setAnimStep(-1);};
  const loadPreset=(k:string)=>{setGrid(PRESETS[k].map(r=>[...r]));setSelCell(null);setFlowPath([]);setAnimStep(-1);};

  const pathIdx=(r:number,c:number)=>flowPath.findIndex(([pr,pc])=>pr===r&&pc===c);

  // ── slope data for selected cell ─────────────────────────────────────────
  const slopeRows = selCell ? slopeBreakdown(grid, selCell[0], selCell[1], cellM) : null;
  const maxSlope  = slopeRows ? Math.max(0, ...slopeRows.filter(x=>!x.isEdge&&x.slope!==null).map(x=>x.slope!)) : 0;
  const winners   = slopeRows ? slopeRows.filter(x=>!x.isEdge&&x.slope!==null&&Math.abs(x.slope!-maxSlope)<1e-9&&maxSlope>0) : [];

  // ── render ────────────────────────────────────────────────────────────────
  const TABS: {id:TabMode;label:string}[]=[
    {id:'terrain',   label:'🗺 Terrain'},
    {id:'flowdir',   label:'➡ Flow Direction'},
    {id:'calculate', label:'📐 Slope Calculator'},
    {id:'raindrop',  label:'💧 Raindrop'},
  ];

  return(
    <div className="not-prose my-8 rounded-2xl border border-slate-200 bg-white shadow-xl overflow-hidden font-sans">

      {/* Header */}
      <div className="bg-gradient-to-r from-sky-600 to-blue-800 px-6 py-4">
        <h3 className="text-white font-bold text-lg tracking-tight">DEM & Flow Direction Explorer</h3>
        <p className="text-sky-100 text-sm mt-0.5">Left-click = raise elevation · Right-click = lower · Four interactive modes</p>
      </div>

      {/* Tabs */}
      <div className="flex flex-wrap border-b border-slate-200 bg-slate-50 text-sm">
        {TABS.map(t=>(
          <button key={t.id} onClick={()=>switchMode(t.id)}
            className={`px-4 py-3 font-medium border-b-2 transition-colors whitespace-nowrap ${mode===t.id?'border-sky-600 text-sky-700 bg-white':'border-transparent text-slate-500 hover:text-slate-700'}`}
          >{t.label}</button>
        ))}
      </div>

      <div className="p-6 flex flex-col lg:flex-row gap-8 items-start">

        {/* Grid */}
        <div className="flex-shrink-0 flex flex-col items-center gap-2">
          <svg width={C*CELL} height={R*CELL}
            className="rounded-xl border border-slate-200 cursor-pointer select-none shadow-sm"
            onContextMenu={e=>e.preventDefault()}
          >
            {grid.map((row,r)=>row.map((z,c)=>{
              const bg=elevColor(z,zMin,zMax);
              const pIdx=pathIdx(r,c);
              const onPath=pIdx>=0&&pIdx<=animStep;
              const isHead=pIdx===animStep&&animStep>=0;
              const isSelCell=selCell&&selCell[0]===r&&selCell[1]===c;
              const isOutlet=(mode==='flowdir'||mode==='calculate'||mode==='raindrop')&&!fg[r][c];
              const isEdgeCell=r===0||r===R-1||c===0||c===C-1;
              const cx=c*CELL+CELL/2, cy=r*CELL+CELL/2;

              let fill=bg;
              if(onPath) fill=isHead?'#0284c7':'#93c5fd';
              if(isSelCell) fill='#fef08a';

              return(
                <g key={`${r}-${c}`}
                  onClick={e=>{e.preventDefault();interact(r,c,false);}}
                  onContextMenu={e=>{e.preventDefault();interact(r,c,true);}}
                >
                  <rect x={c*CELL}y={r*CELL}width={CELL}height={CELL}
                    fill={fill}
                    stroke={isSelCell?'#ca8a04':onPath?'#2563eb':'#e2e8f0'}
                    strokeWidth={isSelCell||onPath?2:0.8}/>

                  {/* Terrain mode: z value */}
                  {mode==='terrain'&&(
                    <text x={cx}y={cy+5}textAnchor="middle"fontSize={14}fontWeight="700"
                      fill={z>(zMin+zMax)/2?'#1e293b':'#f0f9ff'}
                      style={{fontFamily:'monospace',pointerEvents:'none'}}
                    >{z}</text>
                  )}

                  {/* Calculate mode: z value faintly */}
                  {mode==='calculate'&&(
                    <text x={cx}y={cy+5}textAnchor="middle"fontSize={13}fontWeight="600"
                      fill={isSelCell?'#1e293b':z>(zMin+zMax)/2?'#334155':'#e2e8f0'}
                      style={{pointerEvents:'none'}}
                    >{z}</text>
                  )}

                  {/* Flow direction arrow */}
                  {(mode==='flowdir'||mode==='raindrop')&&fg[r][c]&&(
                    <Arrow angle={fg[r][c]!.angle}cx={cx}cy={cy}r={CELL*.3}
                      color={onPath?'#0c4a6e':'#334155'}/>
                  )}

                  {/* Calculate mode arrow on selected + neighbors */}
                  {mode==='calculate'&&fg[r][c]&&isSelCell&&(
                    <Arrow angle={fg[r][c]!.angle}cx={cx}cy={cy}r={CELL*.35}color='#0369a1'/>
                  )}

                  {/* D8 code */}
                  {mode==='flowdir'&&showCodes&&fg[r][c]&&(
                    <text x={c*CELL+4}y={r*CELL+12}fontSize={9}fill="#64748b"style={{pointerEvents:'none'}}>{fg[r][c]!.code}</text>
                  )}

                  {/* Pit marker */}
                  {isOutlet&&(
                    <text x={cx}y={cy+6}textAnchor="middle"fontSize={16}style={{pointerEvents:'none'}}>
                      {isEdgeCell?'🌊':'⚠️'}
                    </text>
                  )}

                  {/* Raindrop head */}
                  {isHead&&(
                    <text x={cx}y={cy+6}textAnchor="middle"fontSize={15}style={{pointerEvents:'none'}}>💧</text>
                  )}
                </g>
              );
            }))}
          </svg>

          {/* Color bar */}
          <div className="flex items-center gap-2 w-full text-xs text-slate-400 px-1">
            <span>Low</span>
            <div className="h-2.5 flex-1 rounded-full" style={{background:'linear-gradient(to right, rgb(67,117,180),rgb(116,196,163),rgb(161,218,115),rgb(255,230,130),rgb(200,130,70),rgb(245,245,244))'}}/>
            <span>High</span>
          </div>
          <p className="text-xs text-slate-400 text-center mt-0.5">
            {mode==='terrain'?'Left-click ▲  Right-click ▼':mode==='raindrop'?'Click any cell to release a raindrop':mode==='calculate'?'Click any cell to compute slopes':'D8 arrows = steepest-descent direction'}
          </p>

          {/* Presets */}
          <div className="flex flex-wrap gap-1.5 mt-1">
            {Object.keys(PRESETS).map(k=>(
              <button key={k}onClick={()=>loadPreset(k)}
                className="px-2.5 py-1 rounded-lg bg-slate-100 hover:bg-sky-100 hover:text-sky-800 text-slate-600 text-xs font-medium capitalize transition-colors">
                {k}
              </button>
            ))}
          </div>
        </div>

        {/* Right panel */}
        <div className="flex-1 min-w-0">
          {mode==='terrain'   && <TerrainPanel zMin={zMin} zMax={zMax} R={R} C={C}/>}
          {mode==='flowdir'   && <FlowDirPanel fg={fg} showCodes={showCodes} setShowCodes={setShowCodes}/>}
          {mode==='calculate' && <CalcPanel rows={slopeRows} winners={winners} maxSlope={maxSlope}
                                            selCell={selCell} cellM={cellM} setCellM={setCellM} grid={grid}/>}
          {mode==='raindrop'  && <RaindropPanel flowPath={flowPath} animStep={animStep} fg={fg} R={R} C={C}/>}
        </div>
      </div>
    </div>
  );
}

// ─── Terrain panel ────────────────────────────────────────────────────────────
function TerrainPanel({zMin,zMax,R,C}:{zMin:number;zMax:number;R:number;C:number}){
  return(
    <div className="space-y-3">
      <div className="grid grid-cols-2 gap-3">
        <IC label="Min elevation" value={`${zMin}`} a="blue"/>
        <IC label="Max elevation" value={`${zMax}`} a="amber"/>
        <IC label="Grid size" value={`${R} × ${C}`} a="violet"/>
        <IC label="Relief" value={`${zMax-zMin}`} a="green"/>
      </div>
      <CB type="tip"><strong>DEM = grid of numbers.</strong> Each cell stores one elevation value. Real DEMs come from LiDAR, radar (SRTM 30 m), or stereo imagery. Left-click a cell to raise it, right-click to lower it.</CB>
    </div>
  );
}

// ─── Flow direction panel ─────────────────────────────────────────────────────
function FlowDirPanel({fg,showCodes,setShowCodes}:{fg:FlowGrid;showCodes:boolean;setShowCodes:(v:boolean)=>void}){
  const pits=countPits(fg);
  return(
    <div className="space-y-3">
      <div className="grid grid-cols-2 gap-3">
        <IC label="Interior pits" value={String(pits)} a={pits>0?'red':'green'}/>
        <IC label="Algorithm" value="D8" a="violet"/>
      </div>
      <label className="flex items-center gap-2 text-sm text-slate-600 cursor-pointer select-none">
        <input type="checkbox" checked={showCodes} onChange={e=>setShowCodes(e.target.checked)} className="rounded accent-sky-600"/>
        Show ESRI power-of-2 codes
      </label>
      <div>
        <p className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider mb-1.5">ESRI encoding (3×3 kernel)</p>
        <div className="grid grid-cols-3 gap-1 w-32 text-center text-xs font-mono">
          {[32,64,128,16,'·',1,8,4,2].map((v,i)=>(
            <div key={i}className={`rounded py-1 ${v==='·'?'bg-sky-100 text-sky-800 font-bold':'bg-slate-100 text-slate-600'}`}>{v}</div>
          ))}
        </div>
      </div>
      <CB type="info"><strong>Rule:</strong> slope = Δz ÷ distance. Cardinal distance = Δx; diagonal = √2·Δx. Flow direction = neighbor with largest positive slope. Switch to <em>Slope Calculator</em> to see every number.</CB>
      {pits>0&&<CB type="warn">⚠ {pits} interior pit{pits>1?'s':''} — marked ⚠️ on grid. No downhill neighbor exists. Try the <em>Pit Filling</em> widget below.</CB>}
    </div>
  );
}

// ─── Slope calculator panel ───────────────────────────────────────────────────
function CalcPanel({rows,winners,maxSlope,selCell,cellM,setCellM,grid}:{
  rows:SlopeRow[]|null; winners:SlopeRow[]; maxSlope:number;
  selCell:[number,number]|null; cellM:number; setCellM:(v:number)=>void; grid:Grid;
}){
  const hasTie=winners.length>1;

  return(
    <div className="space-y-4">
      {/* Cell size slider */}
      <div>
        <label className="text-xs font-semibold text-slate-500 uppercase tracking-wide">Cell size Δx</label>
        <div className="flex items-center gap-3 mt-1">
          <input type="range" min={10} max={100} step={5} value={cellM}
            onChange={e=>setCellM(Number(e.target.value))}
            className="flex-1 accent-sky-600"/>
          <span className="text-sm font-mono font-bold text-sky-700 w-14 text-right">{cellM} m</span>
        </div>
        <p className="text-xs text-slate-400 mt-0.5">Real DEM: SRTM ≈ 30 m, LiDAR ≈ 1 m, coarse ≈ 90 m</p>
      </div>

      {!selCell && (
        <CB type="tip">Click any cell on the grid to compute D8 slopes for all 8 neighbours and find which direction water flows.</CB>
      )}

      {rows && selCell && (
        <>
          <div className="flex items-center gap-2">
            <span className="inline-block w-3 h-3 rounded-sm bg-yellow-300 border border-yellow-500"/>
            <span className="text-sm font-semibold text-slate-700">
              Cell ({selCell[0]},{selCell[1]}) — elevation <strong>{grid[selCell[0]][selCell[1]]}</strong> m
            </span>
          </div>

          {/* 3×3 neighborhood mini-grid */}
          <NeighborGrid grid={grid} r={selCell[0]} c={selCell[1]} winners={winners}/>

          {/* Slope table */}
          <div className="overflow-x-auto rounded-xl border border-slate-200 text-xs">
            <table className="w-full text-center border-collapse">
              <thead>
                <tr className="bg-slate-100 text-slate-600">
                  <th className="px-2 py-2 font-semibold">Dir</th>
                  <th className="px-2 py-2 font-semibold">Code</th>
                  <th className="px-2 py-2 font-semibold">Nbr z (m)</th>
                  <th className="px-2 py-2 font-semibold">Dist</th>
                  <th className="px-2 py-2 font-semibold">Δz</th>
                  <th className="px-2 py-2 font-semibold">Slope (m/m)</th>
                  <th className="px-2 py-2 font-semibold">Status</th>
                </tr>
              </thead>
              <tbody>
                {rows.map(row=>{
                  const isWin=!hasTie&&winners.some(w=>w.d.code===row.d.code);
                  const isTied=hasTie&&winners.some(w=>w.d.code===row.d.code);
                  let rowCls='border-t border-slate-100 ';
                  if(isWin) rowCls+='bg-emerald-50';
                  else if(isTied) rowCls+='bg-amber-50';
                  else if(row.slope!==null&&row.slope<0) rowCls+='bg-red-50/50';
                  else if(row.isEdge) rowCls+='bg-slate-50';

                  let status='';
                  if(row.isEdge) status='✗ edge';
                  else if(row.slope===null) status='—';
                  else if(isWin) status='★ FLOW';
                  else if(isTied) status='⚠ tie';
                  else if(row.slope<0) status='↑ uphill';
                  else if(row.slope===0) status='= flat';
                  else status='—';

                  const distStr=row.isEdge?'—':(row.dist===1?`1.00 × ${cellM}m`:`1.41 × ${cellM}m`);
                  const slopeStr=row.isEdge||row.slope===null?'—':row.slope.toFixed(4);
                  const dzStr=row.isEdge?'—':row.dz.toFixed(1);

                  return(
                    <tr key={row.d.code} className={rowCls}>
                      <td className="px-2 py-1.5 font-mono font-bold">{row.d.emoji} {row.d.label}</td>
                      <td className="px-2 py-1.5 font-mono">{row.d.code}</td>
                      <td className="px-2 py-1.5">{row.isEdge?'—':row.zNeigh}</td>
                      <td className="px-2 py-1.5 text-slate-500">{distStr}</td>
                      <td className={`px-2 py-1.5 font-mono ${row.dz>0?'text-emerald-700':row.dz<0?'text-red-600':'text-slate-500'}`}>{dzStr}</td>
                      <td className={`px-2 py-1.5 font-mono font-semibold ${isWin?'text-emerald-700':isTied?'text-amber-700':row.slope!==null&&row.slope<0?'text-red-500':'text-slate-600'}`}>{slopeStr}</td>
                      <td className={`px-2 py-1.5 font-semibold ${isWin?'text-emerald-700':isTied?'text-amber-600':''}`}>{status}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {/* Result callout */}
          {maxSlope>0&&(
            hasTie?(
              <CB type="warn">
                <strong>Tie detected!</strong> {winners.map(w=>w.d.label).join(' and ')} both have slope {maxSlope.toFixed(4)} m/m.
                D8 breaks ties by scan order (E→SE→S→SW→W→NW→N→NE), so <strong>{winners[0].d.label}</strong> wins.
                This is an acknowledged limitation of D8 — consider D∞ for steep, symmetric terrain.
              </CB>
            ):(
              <CB type="tip">
                <strong>Flow direction: {winners[0]?.d.emoji} {winners[0]?.d.label}</strong> (code {winners[0]?.d.code}).
                Slope = {maxSlope.toFixed(4)} m/m ({(maxSlope*100).toFixed(2)} %).
                In real routing, every upstream raindrop eventually follows this direction.
              </CB>
            )
          )}
          {maxSlope===0&&(
            <CB type="warn">
              <strong>No downhill neighbor.</strong> This is a <em>pit</em> (closed depression).
              D8 assigns no flow direction (code = 0). The pit-fill algorithm resolves this.
            </CB>
          )}
        </>
      )}
    </div>
  );
}

// 3×3 neighbourhood mini-grid
function NeighborGrid({grid,r,c,winners}:{grid:Grid;r:number;c:number;winners:SlopeRow[]}){
  const R=grid.length, C=grid[0].length;
  const S=44; // cell size in px
  const z0=grid[r][c];
  const zAll:number[]=[];
  for(let dr=-1;dr<=1;dr++)for(let dc=-1;dc<=1;dc++){const nr=r+dr,nc=c+dc;if(nr>=0&&nr<R&&nc>=0&&nc<C)zAll.push(grid[nr][nc]);}
  const lo=Math.min(...zAll), hi=Math.max(...zAll);
  return(
    <div>
      <p className="text-[11px] text-slate-400 font-semibold uppercase tracking-wide mb-1.5">3×3 Neighbourhood</p>
      <svg width={3*S} height={3*S} className="rounded-lg border border-slate-200">
        {[-1,0,1].map(dr=>[-1,0,1].map(dc=>{
          const nr=r+dr, nc=c+dc;
          const isCenter=dr===0&&dc===0;
          const isEdge=nr<0||nr>=R||nc<0||nc>=C;
          const zv=isEdge?null:grid[nr][nc];
          const cx=(dc+1)*S+S/2, cy=(dr+1)*S+S/2;
          const isWinDir=winners.some(w=>w.d.dr===dr&&w.d.dc===dc);
          let fill=isCenter?'#fef08a':isEdge?'#f1f5f9':elevColor(zv??0,lo,hi);
          if(isWinDir) fill='#bbf7d0';
          return(
            <g key={`${dr},${dc}`}>
              <rect x={(dc+1)*S}y={(dr+1)*S}width={S}height={S}
                fill={fill}stroke={isCenter?'#ca8a04':isWinDir?'#16a34a':'#e2e8f0'}
                strokeWidth={isCenter||isWinDir?2:0.8}/>
              <text x={cx}y={cy+5}textAnchor="middle"fontSize={13}fontWeight={isCenter?'800':'600'}
                fill={isCenter?'#78350f':isEdge?'#94a3b8':'#1e293b'}
                style={{pointerEvents:'none'}}>
                {isEdge?'—':zv}
              </text>
              {isWinDir&&(
                <text x={cx}y={cy+S*0.45}textAnchor="middle"fontSize={9}fill="#16a34a">★</text>
              )}
            </g>
          );
        }))}
      </svg>
      <p className="text-[10px] text-slate-400 mt-1">Yellow = center · Green = flow target</p>
    </div>
  );
}

// ─── Raindrop panel ───────────────────────────────────────────────────────────
function RaindropPanel({flowPath,animStep,fg,R,C}:{flowPath:[number,number][];animStep:number;fg:FlowGrid;R:number;C:number}){
  const done=animStep>=flowPath.length-1&&flowPath.length>0;
  const last=flowPath[flowPath.length-1];
  const reachedEdge=last&&(last[0]===0||last[0]===R-1||last[1]===0||last[1]===C-1);
  return(
    <div className="space-y-3">
      {flowPath.length===0?(
        <CB type="tip">Click any cell to release a virtual raindrop. It follows D8 flow direction cell by cell until it reaches the outlet or falls into a pit.</CB>
      ):(
        <>
          <div className="grid grid-cols-2 gap-3">
            <IC label="Steps taken" value={`${Math.min(animStep+1,flowPath.length)} / ${flowPath.length}`} a="blue"/>
            <IC label="Status" value={!done?'flowing…':reachedEdge?'outlet ✓':'pit ⚠️'} a={!done?'violet':reachedEdge?'green':'red'}/>
          </div>
          {done&&<CB type={reachedEdge?'tip':'warn'}>{reachedEdge?'🌊 Reached the watershed outlet. In a real DEM this becomes a stream or river mouth.':'⚠️ Raindrop fell into a pit — no downhill path. Pit-filling fixes this.'}</CB>}
        </>
      )}
    </div>
  );
}

// ─── Utility atoms ────────────────────────────────────────────────────────────
function IC({label,value,a}:{label:string;value:string;a:string}){
  const s:Record<string,string>={blue:'bg-sky-50 border-sky-100 text-sky-900',amber:'bg-amber-50 border-amber-100 text-amber-900',violet:'bg-violet-50 border-violet-100 text-violet-900',green:'bg-emerald-50 border-emerald-100 text-emerald-900',red:'bg-red-50 border-red-100 text-red-900'};
  return(<div className={`rounded-xl border px-3 py-2.5 flex flex-col gap-0.5 ${s[a]??s.blue}`}>
    <span className="text-[11px] font-medium opacity-60 uppercase tracking-wide">{label}</span>
    <span className="font-bold text-base leading-tight">{value}</span>
  </div>);
}
type CBT='tip'|'info'|'warn';
function CB({type,children}:{type:CBT;children:React.ReactNode}){
  const s:Record<CBT,string>={tip:'bg-sky-50 border-sky-200 text-sky-900',info:'bg-slate-50 border-slate-200 text-slate-700',warn:'bg-amber-50 border-amber-200 text-amber-900'};
  return(<div className={`rounded-xl border p-3.5 text-sm leading-relaxed ${s[type]}`}>{children}</div>);
}
