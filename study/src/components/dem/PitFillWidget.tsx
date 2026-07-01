'use client';

import React, { useState, useRef } from 'react';

// ─── Types ────────────────────────────────────────────────────────────────────
type Grid = number[][];

interface Step {
  type: 'pop' | 'fill';
  r: number; c: number;
  spill: number;
  filledSnapshot: Grid;
  visitedSnapshot: boolean[][];
  heapSnapshot: HeapCell[];
  note: string;
}

interface HeapCell { r:number; c:number; spill:number; }

// ─── Preset terrains for pit-fill demo ────────────────────────────────────────
// A 6×6 DEM with a clear pit (elevation 2) and one low border cell at (4,0)=3
// After Wang & Liu: pits rise to 5 (minimum path from border through ring at z=5)
const PRESET_BASIN: Grid = [
  [7,7,7,7,7,7],
  [7,5,5,5,5,7],
  [7,5,2,2,5,7],
  [7,5,2,2,5,7],
  [3,5,5,5,5,7],  // ← (4,0) = 3: LOW outlet
  [7,7,7,7,7,7],
];

// A more complex 8×8 with two separate pits
const PRESET_TWOPITS: Grid = [
  [8,8,8,8,8,8,8,8],
  [8,6,6,2,6,6,6,8],  // pit at (1,3)=2
  [8,6,6,6,6,6,6,8],
  [8,2,6,6,6,6,6,8],  // pit at (3,1)=2
  [8,6,6,6,6,6,6,8],
  [8,6,6,6,6,6,6,8],
  [3,6,6,6,6,6,6,8],  // ← (6,0)=3: low outlet
  [8,8,8,8,8,8,8,8],
];

// Simple slope with a dip
const PRESET_DIP: Grid = [
  [9,8,7,6,5,4,3,2],
  [9,8,7,6,5,4,3,2],
  [9,8,7,6,5,4,3,2],
  [9,8,7,1,5,4,3,2],  // pit at (3,3)=1
  [9,8,7,6,5,4,3,2],
  [9,8,7,6,5,4,3,2],
  [9,8,7,6,5,4,3,2],
  [9,8,7,6,5,4,3,2],
];

const PRESETS: Record<string,{grid:Grid;desc:string}> = {
  basin: { grid:PRESET_BASIN, desc:'Single circular depression — clear outlet at bottom-left corner (z=3)' },
  twopits: { grid:PRESET_TWOPITS, desc:'Two isolated pits — both fill to the local rim, then drain to the low outlet' },
  dip: { grid:PRESET_DIP, desc:'Slope with a single dip — after filling, the dip blends into the surrounding slope' },
};

// ─── Wang & Liu (2006) pit-fill ───────────────────────────────────────────────
// Returns an ordered list of algorithm steps for animation.
function wangLiu(dem: Grid): Step[] {
  const R=dem.length, C=dem[0].length;
  const filled=dem.map(r=>[...r]);
  const visited=Array.from({length:R},()=>Array(C).fill(false));
  const steps: Step[]=[];

  // Min-heap represented as sorted array (small grids, OK for demo)
  let heap:HeapCell[]=[];
  const heapPush=(h:HeapCell[],cell:HeapCell)=>{
    let lo=0,hi=h.length;
    while(lo<hi){const mid=(lo+hi)>>1;if(h[mid].spill<=cell.spill)lo=mid+1;else hi=mid;}
    h.splice(lo,0,cell);
  };

  // 1. Seed with all border cells
  for(let r=0;r<R;r++) for(let c=0;c<C;c++){
    if(r===0||r===R-1||c===0||c===C-1){
      visited[r][c]=true;
      heap.push({r,c,spill:dem[r][c]});
    }
  }
  heap.sort((a,b)=>a.spill-b.spill);

  // 8-connected neighbours
  const NBRS=[[-1,0],[1,0],[0,-1],[0,1],[-1,-1],[-1,1],[1,-1],[1,1]];

  // 2. Main loop
  while(heap.length>0){
    const {r,c,spill}=heap.shift()!;
    steps.push({
      type:'pop',r,c,spill,
      filledSnapshot:filled.map(x=>[...x]),
      visitedSnapshot:visited.map(x=>[...x]),
      heapSnapshot:[...heap],
      note:`Pop (${r},${c}) — spill height = ${spill}`,
    });

    for(const[dr,dc]of NBRS){
      const nr=r+dr,nc=c+dc;
      if(nr<0||nr>=R||nc<0||nc>=C) continue;
      if(visited[nr][nc]) continue;
      visited[nr][nc]=true;
      const newZ=Math.max(dem[nr][nc],spill);
      const changed=newZ>dem[nr][nc];
      filled[nr][nc]=newZ;
      const cell={r:nr,c:nc,spill:newZ};
      heapPush(heap,cell);
      steps.push({
        type:'fill',r:nr,c:nc,spill:newZ,
        filledSnapshot:filled.map(x=>[...x]),
        visitedSnapshot:visited.map(x=>[...x]),
        heapSnapshot:[...heap],
        note:changed
          ?`Fill (${nr},${nc}): ${dem[nr][nc]} → ${newZ}  (raised by ${newZ-dem[nr][nc]})`
          :`Visit (${nr},${nc}): z=${dem[nr][nc]} ≥ spill — no change`,
      });
    }
  }
  return steps;
}

// ─── Colour helpers ───────────────────────────────────────────────────────────
function elevColor(z:number,lo:number,hi:number):string{
  const t=hi===lo?0.5:(z-lo)/(hi-lo);
  const stops:[number,[number,number,number]][]=[[0,[67,117,180]],[0.18,[116,196,163]],[0.4,[161,218,115]],[0.62,[255,230,130]],[0.82,[200,130,70]],[1,[245,245,244]]];
  let a=stops[0],b=stops[stops.length-1];
  for(let i=0;i<stops.length-1;i++)if(t>=stops[i][0]&&t<=stops[i+1][0]){a=stops[i];b=stops[i+1];break;}
  const u=(b[0]-a[0])===0?0:(t-a[0])/(b[0]-a[0]);
  const lp=(x:number,y:number)=>Math.round(x+u*(y-x));
  return`rgb(${lp(a[1][0],b[1][0])},${lp(a[1][1],b[1][1])},${lp(a[1][2],b[1][2])})`;
}

// ─── Main widget ──────────────────────────────────────────────────────────────
const CELL=52;

export default function PitFillWidget() {
  const [preset,setPreset]   = useState<string>('basin');
  const [rawGrid]            = useState<Grid>(PRESET_BASIN.map(r=>[...r]));
  const [steps,setSteps]     = useState<Step[]>([]);
  const [stepIdx,setStepIdx] = useState(-1);
  const [speed,setSpeed]     = useState(200);
  const timerRef=useRef<ReturnType<typeof setTimeout>|null>(null);

  const currentPreset=PRESETS[preset];
  const dem=currentPreset.grid;
  const R=dem.length, C=dem[0].length;
  const zMin=Math.min(...dem.flat()), zMax=Math.max(...dem.flat());

  // Final filled DEM (pre-computed)
  const [finalFilled]=useState<Grid>(()=>{ const s=wangLiu(PRESET_BASIN);return s[s.length-1]?.filledSnapshot??PRESET_BASIN; });

  // Current display state
  const curStep=steps[stepIdx]??null;
  const displayGrid=curStep?curStep.filledSnapshot:dem;
  const visitedGrid=curStep?curStep.visitedSnapshot:null;

  const phase=stepIdx<0?'idle':stepIdx>=steps.length-1?'done':'running';

  const loadPreset=(key:string)=>{
    if(timerRef.current)clearTimeout(timerRef.current);
    setPreset(key);
    setSteps([]);setStepIdx(-1);
  };

  const startAnimation=()=>{
    const s=wangLiu(dem);
    setSteps(s);setStepIdx(0);
    let idx=0;
    const tick=()=>{
      idx++;
      if(idx<s.length){setStepIdx(idx);timerRef.current=setTimeout(tick,speed);}
      else{setStepIdx(s.length-1);}
    };
    timerRef.current=setTimeout(tick,speed);
  };

  const pause=()=>{if(timerRef.current)clearTimeout(timerRef.current);};
  const reset=()=>{if(timerRef.current)clearTimeout(timerRef.current);setSteps([]);setStepIdx(-1);};
  const skipEnd=()=>{if(timerRef.current)clearTimeout(timerRef.current);const s=wangLiu(dem);setSteps(s);setStepIdx(s.length-1);};
  const stepFwd=()=>setStepIdx(i=>Math.min(i+1,steps.length-1));
  const stepBck=()=>setStepIdx(i=>Math.max(i-1,0));

  // Which cells were raised?
  const raisedCells:Set<string>=new Set();
  for(let r=0;r<R;r++) for(let c=0;c<C;c++){
    if(displayGrid[r][c]>dem[r][c]) raisedCells.add(`${r},${c}`);
  }

  return(
    <div className="not-prose my-8 rounded-2xl border border-slate-200 bg-white shadow-xl overflow-hidden font-sans">

      {/* Header */}
      <div className="bg-gradient-to-r from-orange-500 to-rose-600 px-6 py-4">
        <h3 className="text-white font-bold text-lg tracking-tight">Pit Filling — Wang & Liu (2006)</h3>
        <p className="text-orange-100 text-sm mt-0.5">See exactly how depressions are raised to their lowest spill point before routing</p>
      </div>

      {/* Preset tabs */}
      <div className="flex border-b border-slate-200 bg-slate-50 text-sm">
        {Object.entries(PRESETS).map(([k,v])=>(
          <button key={k}onClick={()=>loadPreset(k)}
            className={`px-4 py-2.5 font-medium border-b-2 capitalize transition-colors ${preset===k?'border-orange-500 text-orange-700 bg-white':'border-transparent text-slate-500 hover:text-slate-700'}`}
          >{k}</button>
        ))}
      </div>

      <div className="p-6 flex flex-col lg:flex-row gap-8 items-start">

        {/* Side-by-side grids */}
        <div className="flex-shrink-0 flex flex-col gap-3">
          <div className="flex gap-6 items-start">
            {/* BEFORE */}
            <div className="flex flex-col items-center gap-1">
              <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide">Before (raw DEM)</p>
              <DemGrid dem={dem} displayGrid={dem} raisedCells={new Set()} visitedGrid={null} curStep={null} zMin={zMin} zMax={zMax} cellPx={CELL}/>
            </div>
            {/* AFTER / CURRENT */}
            <div className="flex flex-col items-center gap-1">
              <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide">
                {phase==='idle'?'After (Wang & Liu filled)':phase==='done'?'After ✓':
                  `Step ${stepIdx+1} / ${steps.length}`}
              </p>
              <DemGrid
                dem={dem}
                displayGrid={phase==='idle'
                  ? (() => { const s=wangLiu(dem); return s[s.length-1]?.filledSnapshot??dem; })()
                  : displayGrid}
                raisedCells={raisedCells}
                visitedGrid={visitedGrid}
                curStep={curStep}
                zMin={zMin} zMax={zMax} cellPx={CELL}/>
            </div>
          </div>

          {/* Legend */}
          <div className="flex flex-wrap gap-3 text-xs text-slate-600 mt-1 px-1">
            <LegendItem color="#fbbf24" label="Currently processing (popped from heap)"/>
            <LegendItem color="#fdba74" label="Raised (filled) cell"/>
            <LegendItem color="#e2e8f0" label="Visited — not raised"/>
            <LegendItem color="#3b82f6" label="Low outlet cell"/>
          </div>
        </div>

        {/* Right controls + info */}
        <div className="flex-1 min-w-0 flex flex-col gap-4">

          {/* Description */}
          <div className="rounded-xl bg-orange-50 border border-orange-200 p-3.5 text-sm text-orange-900 leading-relaxed">
            {currentPreset.desc}
          </div>

          {/* Playback controls */}
          <div className="flex flex-wrap gap-2">
            {phase==='idle'&&<button onClick={startAnimation}className="px-4 py-2 rounded-xl bg-orange-500 hover:bg-orange-600 text-white text-sm font-semibold">▶ Animate</button>}
            {phase==='running'&&<button onClick={pause}className="px-4 py-2 rounded-xl bg-slate-400 hover:bg-slate-500 text-white text-sm font-semibold">⏸ Pause</button>}
            {steps.length>0&&phase!=='idle'&&(
              <>
                <button onClick={stepBck}disabled={stepIdx<=0}className="px-3 py-2 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-700 text-sm font-semibold disabled:opacity-40">◀ Back</button>
                <button onClick={stepFwd}disabled={stepIdx>=steps.length-1}className="px-3 py-2 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-700 text-sm font-semibold disabled:opacity-40">Fwd ▶</button>
              </>
            )}
            {phase!=='done'&&<button onClick={skipEnd}className="px-4 py-2 rounded-xl bg-sky-600 hover:bg-sky-700 text-white text-sm font-semibold">⏭ Skip</button>}
            <button onClick={reset}className="px-4 py-2 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-700 text-sm font-semibold">↺ Reset</button>
          </div>

          {/* Speed */}
          <div>
            <label className="text-xs font-semibold text-slate-500 uppercase tracking-wide">Speed</label>
            <div className="flex items-center gap-3 mt-1">
              <span className="text-xs text-slate-400">Slow</span>
              <input type="range"min={50}max={800}step={50}value={speed}onChange={e=>setSpeed(Number(e.target.value))}className="flex-1 accent-orange-500"/>
              <span className="text-xs text-slate-400">Fast</span>
            </div>
          </div>

          {/* Current step note */}
          {curStep&&(
            <div className={`rounded-xl border p-3.5 text-sm font-mono leading-relaxed ${curStep.type==='fill'&&curStep.spill>dem[curStep.r]?.[curStep.c]?'bg-orange-50 border-orange-200 text-orange-900':'bg-slate-50 border-slate-200 text-slate-700'}`}>
              <span className="font-semibold">{curStep.type==='pop'?'HEAP POP →':'FILL →'}</span> {curStep.note}
            </div>
          )}

          {/* Stats */}
          <div className="grid grid-cols-2 gap-3">
            <IC label="Total steps" value={phase==='idle'?'—':String(steps.length)} a="orange"/>
            <IC label="Cells raised" value={phase==='idle'?'—':String(raisedCells.size)} a="rose"/>
          </div>

          {/* Algorithm explainer */}
          <div className="rounded-xl bg-slate-50 border border-slate-200 p-4 text-sm text-slate-700 space-y-2">
            <p className="font-semibold text-slate-800">Wang & Liu (2006) — Priority Queue Fill</p>
            <ol className="list-decimal list-inside space-y-1 text-xs">
              <li>Seed the min-heap with all <strong>border cells</strong> at their actual elevation.</li>
              <li>Pop the cell with the <strong>lowest spill height</strong>.</li>
              <li>For each unvisited neighbour: <code className="bg-slate-100 px-1 rounded">filled = max(z_neighbour, spill)</code></li>
              <li>Add neighbour to heap with its new fill height. Mark visited.</li>
              <li>Repeat until heap is empty.</li>
            </ol>
            <p className="text-xs text-slate-500">Complexity: O(MN log MN). The low-outlet cell (z=3) reaches the pits first with spill=5 (through the z=5 ring), <em>before</em> the z=7 border cells arrive with spill=7.</p>
          </div>

          {/* Flat area warning after completion */}
          {phase==='done'&&(
            <div className="rounded-xl bg-amber-50 border border-amber-200 p-3.5 text-sm text-amber-900">
              <strong>⚠ Flat areas created.</strong> Raised cells often end up at the same elevation as their neighbours. A secondary <em>flat-area resolution</em> step (e.g., SAGA's "Fill Sinks (Wang & Liu)" then "Flow Direction (D8 – parallel)") assigns tiny artificial gradients so D8 can proceed.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ─── DEM grid sub-component ────────────────────────────────────────────────────
function DemGrid({dem,displayGrid,raisedCells,visitedGrid,curStep,zMin,zMax,cellPx}:{
  dem:Grid; displayGrid:Grid; raisedCells:Set<string>;
  visitedGrid:boolean[][]|null; curStep:Step|null;
  zMin:number; zMax:number; cellPx:number;
}){
  const R=dem.length, C=dem[0].length;
  return(
    <svg width={C*cellPx}height={R*cellPx}className="rounded-xl border border-slate-200 shadow-sm select-none">
      {displayGrid.map((row,r)=>row.map((z,c)=>{
        const isRaised=raisedCells.has(`${r},${c}`);
        const isCur=curStep&&curStep.r===r&&curStep.c===c;
        const isVisited=visitedGrid?.[r]?.[c];
        const isLowOutlet=dem[r][c]===Math.min(...dem.flat())&&(r===0||r===dem.length-1||c===0||c===dem[0].length-1);
        const cx=c*cellPx+cellPx/2, cy=r*cellPx+cellPx/2;

        let fill=elevColor(z,zMin,zMax);
        let stroke='#e2e8f0';
        let sw=0.8;
        if(isLowOutlet){fill='#3b82f6';stroke='#1d4ed8';sw=2;}
        else if(isCur){fill='#fbbf24';stroke='#d97706';sw=2.5;}
        else if(isRaised){fill='#fdba74';stroke='#ea580c';sw=1.5;}
        else if(isVisited){fill='#f1f5f9';}

        const textColor=z>(zMin+zMax)/2?'#1e293b':'#f0f9ff';
        const changedColor=isRaised?'#9a3412':'#1e293b';

        return(
          <g key={`${r}-${c}`}>
            <rect x={c*cellPx}y={r*cellPx}width={cellPx}height={cellPx}fill={fill}stroke={stroke}strokeWidth={sw}/>
            {/* Show original z and new z if raised */}
            {isRaised?(
              <>
                <text x={cx}y={cy-2}textAnchor="middle"fontSize={10}fontWeight="700"fill="#9a3412"
                  style={{pointerEvents:'none'}}>{z}</text>
                <text x={cx}y={cy+10}textAnchor="middle"fontSize={8.5}fill="#c2410c"
                  style={{pointerEvents:'none',textDecoration:'line-through'}}>
                  {dem[r][c]}
                </text>
              </>
            ):(
              <text x={cx}y={cy+5}textAnchor="middle"fontSize={13}fontWeight="700"
                fill={isLowOutlet?'#fff':changedColor}style={{pointerEvents:'none'}}>{z}</text>
            )}
            {isLowOutlet&&(
              <text x={cx}y={cy+cy*0.01+cellPx/2-4}textAnchor="middle"fontSize={9}fill="#bfdbfe"style={{pointerEvents:'none'}}>outlet</text>
            )}
          </g>
        );
      }))}
    </svg>
  );
}

// ─── Tiny atoms ───────────────────────────────────────────────────────────────
function LegendItem({color,label}:{color:string;label:string}){
  return(<div className="flex items-center gap-1.5"><div className="w-3 h-3 rounded-sm flex-shrink-0 border border-slate-300"style={{background:color}}/><span>{label}</span></div>);
}
function IC({label,value,a}:{label:string;value:string;a:string}){
  const s:Record<string,string>={orange:'bg-orange-50 border-orange-100 text-orange-900',rose:'bg-rose-50 border-rose-100 text-rose-900'};
  return(<div className={`rounded-xl border px-3 py-2.5 flex flex-col gap-0.5 ${s[a]??s.orange}`}>
    <span className="text-[11px] font-medium opacity-60 uppercase tracking-wide">{label}</span>
    <span className="font-bold text-base leading-tight">{value}</span>
  </div>);
}
