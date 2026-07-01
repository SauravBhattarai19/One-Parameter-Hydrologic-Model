'use client';

import React, { useState, useRef, useCallback } from 'react';

// ─── Types ────────────────────────────────────────────────────────────────────
type Grid = number[][];
interface D8Dir { dr:number; dc:number; code:number; angle:number; label:string; }
type FlowCell = D8Dir | null;

const D8: D8Dir[] = [
  { dr:0,  dc:1,  code:1,   angle:90,  label:'E'  },
  { dr:1,  dc:1,  code:2,   angle:135, label:'SE' },
  { dr:1,  dc:0,  code:4,   angle:180, label:'S'  },
  { dr:1,  dc:-1, code:8,   angle:225, label:'SW' },
  { dr:0,  dc:-1, code:16,  angle:270, label:'W'  },
  { dr:-1, dc:-1, code:32,  angle:315, label:'NW' },
  { dr:-1, dc:0,  code:64,  angle:0,   label:'N'  },
  { dr:-1, dc:1,  code:128, angle:45,  label:'NE' },
];

// ─── Presets (pit-free for clean accumulation) ────────────────────────────────
const PRESETS: Record<string,Grid> = {
  valley: [
    [9,8,7,6,6,7,8,9],
    [8,7,6,5,5,6,7,8],
    [7,6,5,4,4,5,6,7],
    [6,5,4,3,3,4,5,6],
    [5,4,3,2,2,3,4,5],
    [4,3,2,1,1,2,3,4],
    [3,2,1,1,1,1,2,3],
    [2,1,1,1,1,1,1,2],
  ],
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

// ─── Algorithm helpers ────────────────────────────────────────────────────────
function computeFlowGrid(grid:Grid): FlowCell[][] {
  const R=grid.length, C=grid[0].length;
  return grid.map((row,r)=>row.map((_,c)=>{
    let best:FlowCell=null, maxS=0;
    for(const d of D8){const nr=r+d.dr,nc=c+d.dc;if(nr<0||nr>=R||nc<0||nc>=C)continue;const s=(grid[r][c]-grid[nr][nc])/(d.dr&&d.dc?Math.SQRT2:1);if(s>maxS){maxS=s;best=d;}}
    return best;
  }));
}

// Returns cells sorted by elevation descending (valid D8 topological order for pit-free DEMs)
function topoSort(grid:Grid): [number,number][] {
  const R=grid.length, C=grid[0].length;
  const cells:[number,number,number][]=[];
  for(let r=0;r<R;r++) for(let c=0;c<C;c++) cells.push([grid[r][c],r,c]);
  cells.sort((a,b)=>b[0]-a[0]);
  return cells.map(([,r,c])=>[r,c]);
}

interface FAState { fa: number[][]; step: number; currentCell:[number,number]|null; }

// ─── Color helpers ────────────────────────────────────────────────────────────
function logColor(fa:number, maxFA:number): string {
  if(maxFA<=1) return 'rgb(226,232,240)';
  const t=Math.log(fa)/Math.log(maxFA);
  // white → light blue → sky → dark navy
  const stops:[number,[number,number,number]][]=[[0,[241,245,249]],[0.3,[186,230,253]],[0.6,[56,189,248]],[1,[15,23,42]]];
  let a=stops[0],b=stops[stops.length-1];
  for(let i=0;i<stops.length-1;i++) if(t>=stops[i][0]&&t<=stops[i+1][0]){a=stops[i];b=stops[i+1];break;}
  const u=(b[0]-a[0])===0?0:(t-a[0])/(b[0]-a[0]);
  const lp=(x:number,y:number)=>Math.round(x+u*(y-x));
  return `rgb(${lp(a[1][0],b[1][0])},${lp(a[1][1],b[1][1])},${lp(a[1][2],b[1][2])})`;
}

function elevColor(z:number,lo:number,hi:number):string{
  const t=hi===lo?0.5:(z-lo)/(hi-lo);
  const stops:[number,[number,number,number]][]=[[0,[67,117,180]],[0.4,[161,218,115]],[0.8,[200,130,70]],[1,[245,245,244]]];
  let a=stops[0],b=stops[stops.length-1];
  for(let i=0;i<stops.length-1;i++)if(t>=stops[i][0]&&t<=stops[i+1][0]){a=stops[i];b=stops[i+1];break;}
  const u=(b[0]-a[0])===0?0:(t-a[0])/(b[0]-a[0]);
  const lp=(x:number,y:number)=>Math.round(x+u*(y-x));
  return`rgb(${lp(a[1][0],b[1][0])},${lp(a[1][1],b[1][1])},${lp(a[1][2],b[1][2])})`;
}

// ─── SVG Arrow ────────────────────────────────────────────────────────────────
function Arrow({angle,cx,cy,r:rad}:{angle:number;cx:number;cy:number;r:number}){
  const RAD=(angle-90)*Math.PI/180;
  const tx=cx+rad*Math.cos(RAD),ty=cy+rad*Math.sin(RAD);
  const bx=cx-rad*.5*Math.cos(RAD),by=cy-rad*.5*Math.sin(RAD);
  const hs=rad*.35,l=(angle-90-25)*Math.PI/180,rr=(angle-90+25)*Math.PI/180;
  return<g><line x1={bx}y1={by}x2={tx}y2={ty}stroke="rgba(255,255,255,0.85)"strokeWidth={1.8}strokeLinecap="round"/>
    <polygon points={`${tx},${ty} ${tx-hs*Math.cos(l)},${ty-hs*Math.sin(l)} ${tx-hs*Math.cos(rr)},${ty-hs*Math.sin(rr)}`}fill="rgba(255,255,255,0.85)"/></g>;
}

// ─── Main widget ──────────────────────────────────────────────────────────────
const CELL=52;
type Phase='idle'|'running'|'done';

export default function FlowAccumWidget() {
  const [grid,setGrid]       = useState<Grid>(()=>PRESETS.valley.map(r=>[...r]));
  const [phase,setPhase]     = useState<Phase>('idle');
  const [fa,setFa]           = useState<number[][]>([]);
  const [step,setStep]       = useState(-1);
  const [curCell,setCurCell] = useState<[number,number]|null>(null);
  const [threshold,setThreshold] = useState(5);  // stream threshold
  const [showFA,setShowFA]   = useState(true);
  const [speed,setSpeed]     = useState(150);    // ms per step
  const timerRef=useRef<ReturnType<typeof setTimeout>|null>(null);
  const stepsRef=useRef<[number,number][]>([]);
  const faRef=useRef<number[][]>([]);

  const R=grid.length, C=grid[0].length;
  const zMin=Math.min(...grid.flat()), zMax=Math.max(...grid.flat());
  const fg=computeFlowGrid(grid);
  const maxFA=fa.length?Math.max(...fa.flat()):1;

  const reset=()=>{
    if(timerRef.current)clearTimeout(timerRef.current);
    setPhase('idle');setStep(-1);setCurCell(null);
    setFa(Array.from({length:R},()=>Array(C).fill(1)));
    stepsRef.current=[];faRef.current=Array.from({length:R},()=>Array(C).fill(1));
  };

  const start=()=>{
    const order=topoSort(grid);
    stepsRef.current=order;
    faRef.current=Array.from({length:R},()=>Array(C).fill(1));
    setFa(faRef.current.map(r=>[...r]));
    setStep(0);setCurCell(order[0]??null);setPhase('running');
    runStep(0,order,faRef.current);
  };

  const runStep=(idx:number,order:[number,number][],faArr:number[][])=>{
    if(idx>=order.length){setPhase('done');setCurCell(null);return;}
    const [r,c]=order[idx];
    setCurCell([r,c]);setStep(idx);
    const d=fg[r][c];
    if(d){
      const nr=r+d.dr,nc=c+d.dc;
      faArr=faArr.map(x=>[...x]);
      faArr[nr][nc]+=faArr[r][c];
      faRef.current=faArr;
      setFa(faArr.map(x=>[...x]));
    }
    timerRef.current=setTimeout(()=>runStep(idx+1,order,faArr),speed);
  };

  const skipToEnd=()=>{
    if(timerRef.current)clearTimeout(timerRef.current);
    const order=topoSort(grid);
    const faArr=Array.from({length:R},()=>Array(C).fill(1));
    for(const[r,c]of order){const d=fg[r][c];if(d){faArr[r+d.dr][c+d.dc]+=faArr[r][c];}}
    faRef.current=faArr;setFa(faArr.map(x=>[...x]));setStep(order.length);setCurCell(null);setPhase('done');
  };

  const loadPreset=(k:string)=>{setGrid(PRESETS[k].map(r=>[...r]));reset();};

  const displayGrid=phase==='idle'?null:fa;

  return(
    <div className="not-prose my-8 rounded-2xl border border-slate-200 bg-white shadow-xl overflow-hidden font-sans">

      {/* Header */}
      <div className="bg-gradient-to-r from-emerald-600 to-teal-700 px-6 py-4">
        <h3 className="text-white font-bold text-lg tracking-tight">Flow Accumulation Explorer</h3>
        <p className="text-emerald-100 text-sm mt-0.5">Watch FA counts propagate from hilltops down to channels · Reveal the stream network</p>
      </div>

      <div className="p-6 flex flex-col lg:flex-row gap-8 items-start">

        {/* Grid */}
        <div className="flex-shrink-0 flex flex-col items-center gap-2">
          <svg width={C*CELL} height={R*CELL}
            className="rounded-xl border border-slate-200 shadow-sm select-none">
            {grid.map((row,r)=>row.map((_,c)=>{
              const faVal=displayGrid?displayGrid[r][c]:1;
              const isCur=curCell&&curCell[0]===r&&curCell[1]===c;
              const isStream=displayGrid&&displayGrid[r][c]>=threshold;
              const isDownOf=curCell&&fg[curCell[0]]?.[curCell[1]]&&fg[curCell[0]][curCell[1]]!.dr+curCell[0]===r&&fg[curCell[0]][curCell[1]]!.dc+curCell[1]===c;

              let fill=phase==='idle'?elevColor(grid[r][c],zMin,zMax):logColor(faVal,maxFA);
              if(isCur) fill='#fbbf24';
              if(isDownOf&&!isCur) fill='#a78bfa';

              const cx=c*CELL+CELL/2, cy=r*CELL+CELL/2;
              const textColor=faVal>maxFA*0.4&&phase!=='idle'?'#f0f9ff':'#1e293b';

              return(
                <g key={`${r}-${c}`}>
                  <rect x={c*CELL}y={r*CELL}width={CELL}height={CELL}
                    fill={fill}
                    stroke={isCur?'#d97706':isStream&&phase!=='idle'?'#0284c7':'#e2e8f0'}
                    strokeWidth={isCur?2.5:isStream&&phase!=='idle'?1.5:0.8}/>

                  {/* Phase idle: elevation */}
                  {phase==='idle'&&(
                    <text x={cx}y={cy+5}textAnchor="middle"fontSize={13}fontWeight="700"
                      fill={grid[r][c]>(zMin+zMax)/2?'#1e293b':'#f0f9ff'}
                      style={{pointerEvents:'none'}}>{grid[r][c]}</text>
                  )}

                  {/* Running/done: FA value */}
                  {phase!=='idle'&&showFA&&(
                    <text x={cx}y={cy+5}textAnchor="middle"fontSize={faVal>=100?10:faVal>=10?12:13}
                      fontWeight="700"fill={textColor}style={{pointerEvents:'none'}}>{faVal}</text>
                  )}

                  {/* Flow arrow (always shown in running/done) */}
                  {phase!=='idle'&&fg[r][c]&&(
                    <Arrow angle={fg[r][c]!.angle}cx={cx}cy={cy}r={CELL*.25}/>
                  )}

                  {/* Current cell label */}
                  {isCur&&(
                    <text x={cx}y={cy-CELL/2+14}textAnchor="middle"fontSize={10}fontWeight="bold"fill="#92400e">
                      proc
                    </text>
                  )}
                </g>
              );
            }))}
          </svg>

          {/* Legend */}
          {phase!=='idle'&&(
            <div className="flex items-center gap-2 w-full text-xs text-slate-400 px-1 mt-1">
              <span>FA=1</span>
              <div className="h-2.5 flex-1 rounded-full"style={{background:'linear-gradient(to right, rgb(241,245,249),rgb(186,230,253),rgb(56,189,248),rgb(15,23,42))'}}/>
              <span>FA={maxFA}</span>
            </div>
          )}

          {/* Presets */}
          <div className="flex flex-wrap gap-1.5 mt-1">
            {Object.keys(PRESETS).map(k=>(
              <button key={k}onClick={()=>loadPreset(k)}
                className="px-2.5 py-1 rounded-lg bg-slate-100 hover:bg-emerald-100 hover:text-emerald-800 text-slate-600 text-xs font-medium capitalize transition-colors">
                {k}
              </button>
            ))}
          </div>
        </div>

        {/* Right panel */}
        <div className="flex-1 min-w-0 flex flex-col gap-4">

          {/* Controls */}
          <div className="flex flex-wrap gap-2">
            {phase==='idle'&&<button onClick={start}className="px-4 py-2 rounded-xl bg-emerald-600 hover:bg-emerald-700 text-white text-sm font-semibold transition-colors">▶ Start Animation</button>}
            {phase==='running'&&<button onClick={()=>{if(timerRef.current)clearTimeout(timerRef.current);setPhase('done');}}className="px-4 py-2 rounded-xl bg-slate-400 hover:bg-slate-500 text-white text-sm font-semibold">⏸ Pause</button>}
            {phase!=='idle'&&<button onClick={skipToEnd}className="px-4 py-2 rounded-xl bg-sky-600 hover:bg-sky-700 text-white text-sm font-semibold">⏭ Skip to End</button>}
            <button onClick={reset}className="px-4 py-2 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-700 text-sm font-semibold">↺ Reset</button>
          </div>

          {/* Speed */}
          <div>
            <label className="text-xs font-semibold text-slate-500 uppercase tracking-wide">Animation speed</label>
            <div className="flex items-center gap-3 mt-1">
              <span className="text-xs text-slate-400">Slow</span>
              <input type="range"min={50}max={600}step={50}value={speed}onChange={e=>setSpeed(Number(e.target.value))}className="flex-1 accent-emerald-600"/>
              <span className="text-xs text-slate-400">Fast</span>
            </div>
          </div>

          {/* Stream threshold */}
          {phase!=='idle'&&(
            <div>
              <label className="text-xs font-semibold text-slate-500 uppercase tracking-wide">
                Stream threshold (FA ≥ {threshold})
              </label>
              <div className="flex items-center gap-3 mt-1">
                <input type="range"min={2}max={Math.max(maxFA,2)}value={threshold}onChange={e=>setThreshold(Number(e.target.value))}className="flex-1 accent-blue-600"/>
                <span className="text-sm font-mono font-bold text-blue-700 w-12 text-right">{threshold}</span>
              </div>
              <p className="text-xs text-slate-400 mt-0.5">Cells above threshold = stream channel (outlined in blue)</p>
            </div>
          )}

          {/* FA value toggle */}
          {phase!=='idle'&&(
            <label className="flex items-center gap-2 text-sm text-slate-600 cursor-pointer select-none">
              <input type="checkbox"checked={showFA}onChange={e=>setShowFA(e.target.checked)}className="rounded accent-emerald-600"/>
              Show FA numbers in cells
            </label>
          )}

          {/* Status / info */}
          <div className="space-y-3">
            {phase==='idle'&&(
              <div className="space-y-2">
                <IC label="Cells" value={`${R}×${C} = ${R*C}`} a="teal"/>
                <IC label="Algorithm" value="Topological sort" a="green"/>
                <CB type="tip">
                  <strong>How it works:</strong> cells are sorted highest→lowest elevation (valid D8 topological order).
                  Each cell <em>donates</em> its FA count to its single D8 downstream neighbour.
                  Channels emerge naturally wherever many cells converge.
                </CB>
              </div>
            )}
            {phase==='running'&&curCell&&(
              <div className="space-y-2">
                <IC label={`Step ${step+1} / ${R*C}`} value={`Processing (${curCell[0]},${curCell[1]})`} a="amber"/>
                <IC label="Current FA" value={String(fa[curCell[0]]?.[curCell[1]]??1)} a="teal"/>
                {fg[curCell[0]]?.[curCell[1]]&&(
                  <IC label="Donating to →" value={`(${curCell[0]+fg[curCell[0]][curCell[1]]!.dr},${curCell[1]+fg[curCell[0]][curCell[1]]!.dc})`} a="violet"/>
                )}
                <CB type="info">
                  <strong>Yellow cell</strong> is being processed. Its FA count flows downstream (purple cell) and accumulates there.
                </CB>
              </div>
            )}
            {phase==='done'&&(
              <div className="space-y-2">
                <IC label="Max FA" value={String(maxFA)} a="blue"/>
                <IC label="Stream cells (FA ≥ {threshold})" value={String(fa.flat().filter(v=>v>=threshold).length)} a="teal"/>
                <CB type="tip">
                  <strong>Animation complete.</strong> Dark-blue cells have high flow accumulation = likely stream channels.
                  Drag the stream threshold slider to extract the channel network at different drainage areas.
                  Try different terrain presets to see how shape changes channel density.
                </CB>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── Utility atoms ────────────────────────────────────────────────────────────
function IC({label,value,a}:{label:string;value:string;a:string}){
  const s:Record<string,string>={blue:'bg-sky-50 border-sky-100 text-sky-900',teal:'bg-teal-50 border-teal-100 text-teal-900',amber:'bg-amber-50 border-amber-100 text-amber-900',violet:'bg-violet-50 border-violet-100 text-violet-900',green:'bg-emerald-50 border-emerald-100 text-emerald-900'};
  return(<div className={`rounded-xl border px-3 py-2.5 flex flex-col gap-0.5 ${s[a]??s.blue}`}>
    <span className="text-[11px] font-medium opacity-60 uppercase tracking-wide">{label}</span>
    <span className="font-bold text-base leading-tight">{value}</span>
  </div>);
}
type CBT='tip'|'info'|'warn';
function CB({type,children}:{type:CBT;children:React.ReactNode}){
  const s:Record<CBT,string>={tip:'bg-emerald-50 border-emerald-200 text-emerald-900',info:'bg-slate-50 border-slate-200 text-slate-700',warn:'bg-amber-50 border-amber-200 text-amber-900'};
  return(<div className={`rounded-xl border p-3.5 text-sm leading-relaxed ${s[type]}`}>{children}</div>);
}
