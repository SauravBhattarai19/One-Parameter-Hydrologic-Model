'use client';

import React, { useState, useEffect, useRef, useMemo } from 'react';

// ─── Terrain ──────────────────────────────────────────────────────────────────
const TERRAIN: number[][] = [[5,5,5,5],[3,1,5,5],[5,5,5,5],[5,5,5,5]];

// ─── Color helpers ────────────────────────────────────────────────────────────
const STOPS: [number,[number,number,number]][] = [
  [0.00,[67,117,180]],[0.18,[116,196,163]],[0.40,[161,218,115]],
  [0.62,[255,230,130]],[0.82,[200,130,70]], [1.00,[245,245,244]],
];
function elevColor(z:number,lo:number,hi:number):string {
  const t=hi===lo?0.5:(z-lo)/(hi-lo);
  let a=STOPS[0],b=STOPS[STOPS.length-1];
  for(let i=0;i<STOPS.length-1;i++) if(t>=STOPS[i][0]&&t<=STOPS[i+1][0]){a=STOPS[i];b=STOPS[i+1];break;}
  const u=(b[0]-a[0])===0?0:(t-a[0])/(b[0]-a[0]);
  const lp=(x:number,y:number)=>Math.round(x+u*(y-x));
  return `rgb(${lp(a[1][0],b[1][0])},${lp(a[1][1],b[1][1])},${lp(a[1][2],b[1][2])})`;
}

// ─── D8 ───────────────────────────────────────────────────────────────────────
const D8=[
  {dr:0,dc:1,code:1,label:'E',diag:false},{dr:1,dc:1,code:2,label:'SE',diag:true},
  {dr:1,dc:0,code:4,label:'S',diag:false},{dr:1,dc:-1,code:8,label:'SW',diag:true},
  {dr:0,dc:-1,code:16,label:'W',diag:false},{dr:-1,dc:-1,code:32,label:'NW',diag:true},
  {dr:-1,dc:0,code:64,label:'N',diag:false},{dr:-1,dc:1,code:128,label:'NE',diag:true},
];
function computeFlow(grid:number[][]){
  const R=grid.length,C=grid[0].length;
  return grid.map((row,r)=>row.map((_,c)=>{
    let best:typeof D8[0]|null=null,mx=0;
    for(const d of D8){const nr=r+d.dr,nc=c+d.dc;if(nr<0||nr>=R||nc<0||nc>=C)continue;const s=(grid[r][c]-grid[nr][nc])/(d.diag?Math.SQRT2:1);if(s>mx){mx=s;best=d;}}
    return best;
  }));
}

// ─── Arrow SVG helper ─────────────────────────────────────────────────────────
function Arrow({x1,y1,x2,y2,color='white'}:{x1:number,y1:number,x2:number,y2:number,color?:string}){
  const dx=x2-x1,dy=y2-y1,len=Math.sqrt(dx*dx+dy*dy)||1;
  const ux=dx/len,uy=dy/len,hs=8;
  const hx=x2-hs*ux,hy=y2-hs*uy;
  const pts=`${x2},${y2} ${hx+hs*0.4*uy},${hy-hs*0.4*ux} ${hx-hs*0.4*uy},${hy+hs*0.4*ux}`;
  return (<g><line x1={x1} y1={y1} x2={x2} y2={y2} stroke={color} strokeWidth={2}/><polygon points={pts} fill={color}/></g>);
}

// ─── Wang & Liu step types ────────────────────────────────────────────────────
type CellState = 'unvisited'|'inHeap'|'visited'|'raised';
type Step = {
  popped:{r:number,c:number,spill:number}|null;
  visited:{r:number,c:number,oldZ:number,newZ:number,raised:boolean}|null;
  heapSnapshot:{r:number,c:number,spill:number}[];
  filled:number[][];
  cellStates:CellState[][];
  log:string;
};

// ─── Wang & Liu algorithm ─────────────────────────────────────────────────────
function wangLiuSteps(dem:number[][]): Step[] {
  const R=dem.length,C=dem[0].length;
  const filled=dem.map(r=>[...r]);
  const vis=Array.from({length:R},()=>Array(C).fill(false));
  const raised=Array.from({length:R},()=>Array(C).fill(false));
  type HI={spill:number,r:number,c:number};
  const heap:HI[]=[];
  const steps:Step[]=[];

  const hpush=(item:HI)=>{heap.push(item);heap.sort((a,b)=>a.spill-b.spill);};
  const hpop=()=>heap.shift()!;
  const snap=():CellState[][]=>Array.from({length:R},(_,r)=>Array.from({length:C},(_,c)=>{
    if(raised[r][c]) return 'raised';
    if(vis[r][c]) return 'visited';
    if(heap.some(h=>h.r===r&&h.c===c)) return 'inHeap';
    return 'unvisited';
  }));

  for(let r=0;r<R;r++) for(let c=0;c<C;c++){
    if(r===0||r===R-1||c===0||c===C-1){
      hpush({spill:dem[r][c],r,c});
      vis[r][c]=true;
    }
  }
  steps.push({
    popped:null,visited:null,
    heapSnapshot:[...heap].slice(0,6),
    filled:filled.map(r=>[...r]),
    cellStates:snap(),
    log:`Initialize: push ${R*2+(C-2)*2} border cells to heap`,
  });

  const NBRS=[[-1,0],[1,0],[0,-1],[0,1],[-1,-1],[-1,1],[1,-1],[1,1]];
  while(heap.length>0){
    const {spill,r,c}=hpop();
    for(const [dr,dc] of NBRS){
      const nr=r+dr,nc=c+dc;
      if(nr<0||nr>=R||nc<0||nc>=C||vis[nr][nc]) continue;
      vis[nr][nc]=true;
      const newZ=Math.max(dem[nr][nc],spill);
      const isRaised=newZ>dem[nr][nc];
      const oldZ=dem[nr][nc];
      filled[nr][nc]=newZ;
      if(isRaised) raised[nr][nc]=true;
      hpush({spill:newZ,r:nr,c:nc});
      steps.push({
        popped:{r,c,spill},
        visited:{r:nr,c:nc,oldZ,newZ,raised:isRaised},
        heapSnapshot:[...heap].slice(0,6),
        filled:filled.map(row=>[...row]),
        cellStates:snap(),
        log:isRaised
          ?`Pop (spill=${spill}, (${r},${c})) → (${nr},${nc}): fill=max(${oldZ},${spill})=${newZ} ↑ RAISED`
          :`Pop (spill=${spill}, (${r},${c})) → (${nr},${nc}): fill=max(${oldZ},${spill})=${newZ}`,
      });
    }
  }
  return steps;
}

// ─── Tab 1: The Pit ───────────────────────────────────────────────────────────
const CELL_SIZE = 72;

type EditProps = { grid: number[][], setGrid: React.Dispatch<React.SetStateAction<number[][]>> };

function PitTab({ grid, setGrid }: EditProps){
  const [dropPhase,setDropPhase]=useState<'idle'|'at00'|'moving'|'trapped'>('idle');
  const timer=useRef<ReturnType<typeof setTimeout>|null>(null);

  const startDrop=()=>{
    if(dropPhase!=='idle') {setDropPhase('idle');return;}
    setDropPhase('at00');
    timer.current=setTimeout(()=>setDropPhase('moving'),600);
    timer.current=setTimeout(()=>setDropPhase('trapped'),1200);
  };
  useEffect(()=>()=>{if(timer.current)clearTimeout(timer.current);},[]);

  // Drop pixel positions (center of cell)
  const pos:{x:number,y:number}={
    at00:   {x:0*CELL_SIZE+CELL_SIZE/2,  y:0*CELL_SIZE+CELL_SIZE/2},
    moving: {x:1*CELL_SIZE+CELL_SIZE/2,  y:1*CELL_SIZE+CELL_SIZE/2},
    trapped:{x:1*CELL_SIZE+CELL_SIZE/2,  y:1*CELL_SIZE+CELL_SIZE/2},
  }[dropPhase==='idle'?'at00':dropPhase] ?? {x:CELL_SIZE/2,y:CELL_SIZE/2};

  const showDot = dropPhase!=='idle';
  const shaking = dropPhase==='trapped';

  return(
    <div className="flex flex-col gap-4">
      {/* Callout */}
      <div className="rounded-xl bg-red-50 border border-red-200 p-3.5 text-sm text-red-900 leading-relaxed">
        <span className="font-bold">A PIT</span> is a cell lower than ALL its neighbors. D8 cannot assign a flow direction. Water accumulates forever — which is physically wrong for DEM pre-processing.
      </div>

      <div className="flex flex-col sm:flex-row gap-6 items-start">
        {/* Grid */}
        <div className="flex-shrink-0">
          <p className="text-xs text-slate-500 mt-1">Left-click: raise · Right-click: lower</p>
          <svg
            width={4*CELL_SIZE} height={4*CELL_SIZE}
            className="rounded-xl border border-slate-200 shadow-sm select-none"
          >
            {grid.map((row, ri)=>row.map((z, ci)=>{
              const isPit    = ri===1&&ci===1;
              const isOutlet = ri===1&&ci===0;
              const cx=ci*CELL_SIZE+CELL_SIZE/2, cy=ri*CELL_SIZE+CELL_SIZE/2;
              const zLo=Math.min(...grid.flat()), zHi=Math.max(...grid.flat());

              let fill=elevColor(z,zLo,zHi);
              let stroke='#94a3b8'; let sw=1;
              if(isPit)   {stroke='#ef4444';sw=3;}
              if(isOutlet){stroke='#3b82f6';sw=3;}

              return(
                <g key={`${ri}-${ci}`}
                  style={{ cursor: 'pointer' }}
                  onClick={() => setGrid(prev => { const g=prev.map(r=>[...r]); g[ri][ci]=Math.min(15,g[ri][ci]+1); return g; })}
                  onContextMenu={(e) => { e.preventDefault(); setGrid(prev => { const g=prev.map(r=>[...r]); g[ri][ci]=Math.max(1,g[ri][ci]-1); return g; }); }}
                >
                  <rect x={ci*CELL_SIZE} y={ri*CELL_SIZE} width={CELL_SIZE} height={CELL_SIZE}
                    fill={fill} stroke={stroke} strokeWidth={sw}/>
                  <text x={cx} y={cy-6} textAnchor="middle" fontSize={18} fontWeight="700"
                    fill={isPit?'#991b1b':isOutlet?'#1e3a8a':'#1e293b'}
                    style={{pointerEvents:'none'}}>{z}</text>
                  {isPit&&(
                    <text x={cx} y={cy+14} textAnchor="middle" fontSize={10} fontWeight="700"
                      fill="#dc2626" style={{pointerEvents:'none'}}>⚠ PIT</text>
                  )}
                  {isOutlet&&(
                    <text x={cx} y={cy+14} textAnchor="middle" fontSize={10} fontWeight="700"
                      fill="#1d4ed8" style={{pointerEvents:'none'}}>OUTLET</text>
                  )}
                  {/* row,col label */}
                  <text x={ci*CELL_SIZE+5} y={ri*CELL_SIZE+12} fontSize={8} fill="#64748b"
                    style={{pointerEvents:'none'}}>{ri},{ci}</text>
                </g>
              );
            }))}

            {/* Raindrop */}
            {showDot&&(
              <circle
                cx={pos.x} cy={pos.y} r={10}
                fill="#60a5fa" stroke="#1d4ed8" strokeWidth={2}
                style={{
                  transition: dropPhase==='moving'?'cx 0.5s ease,cy 0.5s ease':'none',
                  animation: shaking?'shake 0.3s ease infinite':'none',
                }}
              >
                {shaking&&(
                  <animate attributeName="cx" values={`${pos.x-4};${pos.x+4};${pos.x-4}`}
                    dur="0.2s" repeatCount="indefinite"/>
                )}
              </circle>
            )}
          </svg>
        </div>

        {/* Right panel */}
        <div className="flex flex-col gap-4 flex-1">
          <button onClick={startDrop}
            className="px-4 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-700 text-white text-sm font-semibold w-fit">
            {dropPhase==='idle'?'▶ Animate Raindrop':'↺ Reset'}
          </button>

          {dropPhase==='trapped'&&(
            <div className="rounded-xl bg-blue-50 border border-blue-200 p-3.5 text-sm text-blue-900 leading-relaxed">
              Raindrop falls from (0,0)=5 straight into pit (1,1)=1. <strong>Trapped!</strong> No downhill neighbor. Steepest slope from (0,0): toward (1,1) is (5-1)/√2≈2.83, toward (1,0) is (5-3)/1=2.0.
            </div>
          )}

          {/* Neighbor table */}
          <div className="rounded-xl border border-slate-200 bg-slate-50 p-3 text-xs font-mono">
            <p className="font-semibold text-slate-700 mb-2 text-sm font-sans">Pit cell (1,1)={grid[1][1]} — all 8 neighbors:</p>
            <table className="w-full text-left">
              <thead><tr className="text-slate-400">
                <th className="pr-3">Neighbor</th><th className="pr-3">z</th><th>≥ pit?</th>
              </tr></thead>
              <tbody>
                {[[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]].map(([dr,dc])=>{
                  const nr=1+dr, nc=1+dc;
                  const lbl=`(${nr},${nc})`;
                  const z=grid[nr]?.[nc]??'—';
                  const pitZ=grid[1][1];
                  return (
                    <tr key={lbl} className="border-t border-slate-200">
                      <td className="pr-3 text-slate-600">{lbl}</td>
                      <td className="pr-3">{z}</td>
                      <td className={Number(z)>=pitZ?'text-green-600 font-bold':'text-red-600 font-bold'}>
                        {Number(z)>=pitZ?'✓':'✗'}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── Tab 2: Wang & Liu ────────────────────────────────────────────────────────
const CELL_SIZE_WL = 72;

const STATE_COLORS:Record<CellState,string>={
  unvisited:'#334155',
  inHeap:'#fbbf24',
  visited:'#34d399',
  raised:'#fb923c',
};

function WangLiuTab({ steps }: { steps: Step[] }){
  const [stepIdx,setStepIdx]=useState(0);
  const [playing,setPlaying]=useState(false);
  const [speed,setSpeed]=useState(400);
  const timerRef=useRef<ReturnType<typeof setTimeout>|null>(null);
  const logRef=useRef<HTMLDivElement>(null);

  const prevStepsLen = useRef(steps.length);
  useEffect(() => {
    if (prevStepsLen.current !== steps.length) {
      prevStepsLen.current = steps.length;
      setStepIdx(0);
      setPlaying(false);
    }
  }, [steps.length]);

  const step=steps[stepIdx];
  const total=steps.length;

  useEffect(()=>{
    if(!playing) return;
    if(stepIdx>=total-1){setPlaying(false);return;}
    timerRef.current=setTimeout(()=>setStepIdx(i=>i+1),speed);
    return()=>{if(timerRef.current)clearTimeout(timerRef.current);};
  },[playing,stepIdx,speed,total]);

  useEffect(()=>{
    if(logRef.current) logRef.current.scrollTop=logRef.current.scrollHeight;
  },[stepIdx]);

  const play=()=>{if(stepIdx>=total-1)setStepIdx(0);setPlaying(true);};
  const pause=()=>{setPlaying(false);if(timerRef.current)clearTimeout(timerRef.current);};
  const reset=()=>{pause();setStepIdx(0);};

  // Highlight: the cell being popped this step
  const poppedKey=step.popped?`${step.popped.r},${step.popped.c}`:null;
  // The cell just visited this step
  const visitedKey=step.visited?`${step.visited.r},${step.visited.c}`:null;

  return(
    <div className="flex flex-col gap-4">
      {/* Key insight */}
      <div className="rounded-xl bg-amber-50 border border-amber-200 p-3.5 text-sm text-amber-900 leading-relaxed">
        Border cell (1,0) has spill=3 — the lowest in the heap. It reaches pit (1,1) BEFORE the z=5 walls do. So the pit fills to <strong>3</strong>, not 5!
      </div>

      <div className="flex flex-col lg:flex-row gap-6 items-start">
        {/* Grid */}
        <div className="flex-shrink-0">
          <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-1.5">
            Step {stepIdx+1} / {total}
          </p>
          <svg
            width={4*CELL_SIZE_WL} height={4*CELL_SIZE_WL}
            className="rounded-xl border border-slate-600 shadow select-none"
            style={{background:'#1e293b'}}
          >
            {step.filled.map((row,r)=>row.map((z,c)=>{
              const key=`${r},${c}`;
              const state=step.cellStates[r][c];
              const isPopped=key===poppedKey;
              const isJustVisited=key===visitedKey;
              const cx=c*CELL_SIZE_WL+CELL_SIZE_WL/2, cy=r*CELL_SIZE_WL+CELL_SIZE_WL/2;
              const origZ=steps[0]?.filled[r][c]??z;
              const wasRaised=step.visited?.raised&&isJustVisited;

              let fill=STATE_COLORS[state];
              let stroke='#475569'; let sw=1;
              if(isPopped){stroke='#facc15';sw=3.5;}
              else if(isJustVisited&&wasRaised){stroke='#ea580c';sw=2.5;}
              else if(isJustVisited){stroke='#6ee7b7';sw=2;}

              const textCol=state==='unvisited'?'#94a3b8':'#1e293b';

              return(
                <g key={key}>
                  <rect x={c*CELL_SIZE_WL} y={r*CELL_SIZE_WL} width={CELL_SIZE_WL} height={CELL_SIZE_WL}
                    fill={fill} stroke={stroke} strokeWidth={sw} rx={2}/>
                  {/* elevation */}
                  {state==='raised'?(
                    <>
                      <text x={cx} y={cy-4} textAnchor="middle" fontSize={16} fontWeight="700"
                        fill="#1e293b" style={{pointerEvents:'none'}}>{z}</text>
                      <text x={cx} y={cy+13} textAnchor="middle" fontSize={9}
                        fill="#7c2d12" style={{pointerEvents:'none',textDecoration:'line-through'}}>{origZ}</text>
                    </>
                  ):(
                    <text x={cx} y={cy+6} textAnchor="middle" fontSize={17} fontWeight="700"
                      fill={textCol} style={{pointerEvents:'none'}}>{z}</text>
                  )}
                  {/* row,col */}
                  <text x={c*CELL_SIZE_WL+4} y={r*CELL_SIZE_WL+12} fontSize={8} fill="#64748b"
                    style={{pointerEvents:'none'}}>{r},{c}</text>
                  {/* popped ring */}
                  {isPopped&&(
                    <rect x={c*CELL_SIZE_WL+3} y={r*CELL_SIZE_WL+3}
                      width={CELL_SIZE_WL-6} height={CELL_SIZE_WL-6}
                      fill="none" stroke="#facc15" strokeWidth={2} rx={2}
                      strokeDasharray="4 2"/>
                  )}
                </g>
              );
            }))}
          </svg>

          {/* Legend */}
          <div className="flex flex-wrap gap-2 mt-2 text-[11px]">
            {([['#334155','Unvisited'],['#fbbf24','In heap'],['#34d399','Visited'],['#fb923c','Raised']] as [string,string][]).map(([c,l])=>(
              <div key={l} className="flex items-center gap-1">
                <div className="w-3 h-3 rounded-sm" style={{background:c}}/>
                <span className="text-slate-500">{l}</span>
              </div>
            ))}
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 rounded-sm border-2 border-yellow-400 border-dashed"/>
              <span className="text-slate-500">Being popped</span>
            </div>
          </div>
        </div>

        {/* Right: heap + log */}
        <div className="flex-1 min-w-0 flex flex-col gap-3">
          {/* Heap */}
          <div className="rounded-xl border border-slate-200 bg-slate-50 overflow-hidden">
            <div className="px-3 py-2 bg-slate-100 border-b border-slate-200">
              <span className="text-xs font-bold text-slate-600 uppercase tracking-wide">Priority Queue (Min-Heap)</span>
            </div>
            <div className="px-3 py-2 space-y-1">
              {step.heapSnapshot.length===0?(
                <p className="text-xs text-slate-400 italic">Heap empty</p>
              ):step.heapSnapshot.map((h,i)=>(
                <div key={i}
                  className={`flex items-center gap-2 rounded-lg px-2.5 py-1 text-xs font-mono ${i===0?'bg-yellow-100 border border-yellow-300 text-yellow-900 font-bold':'bg-white border border-slate-200 text-slate-700'}`}>
                  {i===0&&<span className="text-yellow-600">▶</span>}
                  <span>spill={h.spill}</span>
                  <span className="text-slate-400">·</span>
                  <span>({h.r},{h.c})</span>
                  {i===0&&<span className="ml-auto text-[10px] bg-yellow-200 text-yellow-800 px-1 rounded">next</span>}
                </div>
              ))}
            </div>
          </div>

          {/* Step log */}
          <div className="rounded-xl border border-slate-200 bg-slate-50 overflow-hidden">
            <div className="px-3 py-2 bg-slate-100 border-b border-slate-200">
              <span className="text-xs font-bold text-slate-600 uppercase tracking-wide">Step Log</span>
            </div>
            <div ref={logRef} className="px-3 py-2 space-y-0.5 max-h-48 overflow-y-auto">
              {steps.slice(Math.max(0,stepIdx-7),stepIdx+1).map((s,i,arr)=>{
                const isLast=i===arr.length-1;
                return(
                  <div key={i}
                    className={`text-[11px] font-mono rounded px-1.5 py-0.5 leading-snug ${isLast?'bg-blue-50 border border-blue-200 text-blue-900 font-semibold':'text-slate-500'}`}>
                    {s.log}
                  </div>
                );
              })}
            </div>
          </div>

          {/* Controls */}
          <div className="flex flex-wrap gap-2">
            <button onClick={reset}
              className="px-3 py-2 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-700 text-sm font-semibold">
              ↺ Back
            </button>
            {playing?(
              <button onClick={pause}
                className="px-4 py-2 rounded-xl bg-amber-500 hover:bg-amber-600 text-white text-sm font-semibold">
                ⏸ Pause
              </button>
            ):(
              <button onClick={play} disabled={stepIdx>=total-1}
                className="px-4 py-2 rounded-xl bg-emerald-600 hover:bg-emerald-700 text-white text-sm font-semibold disabled:opacity-40">
                ▶ Play
              </button>
            )}
            <button onClick={()=>setStepIdx(i=>Math.min(i+1,total-1))} disabled={stepIdx>=total-1}
              className="px-3 py-2 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-700 text-sm font-semibold disabled:opacity-40">
              Step ▶
            </button>
          </div>

          {/* Speed */}
          <div>
            <label className="text-xs font-semibold text-slate-500 uppercase tracking-wide">Speed</label>
            <div className="flex items-center gap-3 mt-1">
              <span className="text-xs text-slate-400">Slow</span>
              <input type="range" min={100} max={800} step={50} value={speed}
                onChange={e=>setSpeed(Number(e.target.value))}
                className="flex-1 accent-emerald-500"/>
              <span className="text-xs text-slate-400">Fast</span>
            </div>
          </div>

          {/* Current step detail */}
          {step.visited&&(
            <div className={`rounded-xl border p-3 text-xs font-mono leading-relaxed ${step.visited.raised?'bg-orange-50 border-orange-200 text-orange-900':'bg-emerald-50 border-emerald-200 text-emerald-900'}`}>
              {step.log}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ─── Tab 3: After Filling ─────────────────────────────────────────────────────
const CELL_AFTER = 64;

function AfterTab({ grid, filled, flowBefore, flowAfter }: {
  grid: number[][], filled: number[][],
  flowBefore: (typeof D8[0]|null)[][], flowAfter: (typeof D8[0]|null)[][]
}){
  const pitZ = grid[1]?.[1];
  const filledZ = filled[1]?.[1];
  const afterDir = flowAfter[1]?.[1];
  return(
    <div className="flex flex-col gap-4">
      {/* Callout */}
      <div className="rounded-xl bg-emerald-50 border border-emerald-200 p-3.5 text-sm text-emerald-900 leading-relaxed">
        <strong>Pit cell (1,1):</strong> z={pitZ} → z={filledZ}. Every cell now has a valid D8 flow direction.
      </div>

      <div className="flex flex-col sm:flex-row gap-6 items-start">
        {/* Before */}
        <div className="flex flex-col items-center gap-2">
          <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide">Before (raw DEM)</p>
          <FlowGrid grid={grid} flowDir={flowBefore} label="before"/>
        </div>
        {/* Arrow */}
        <div className="hidden sm:flex items-center self-center">
          <div className="text-2xl text-slate-400">→</div>
        </div>
        {/* After */}
        <div className="flex flex-col items-center gap-2">
          <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide">After (filled)</p>
          <FlowGrid grid={filled} flowDir={flowAfter} label="after"/>
        </div>
      </div>

      {/* D8 table */}
      <div className="rounded-xl border border-slate-200 bg-slate-50 p-3.5 text-xs font-mono">
        <p className="font-semibold text-slate-700 mb-2 font-sans text-sm">Flow direction change at pit cell (1,1):</p>
        <div className="grid grid-cols-3 gap-2 text-center">
          <div className="bg-white border border-red-200 rounded-lg p-2">
            <div className="text-[10px] text-slate-400 uppercase mb-1">Before</div>
            <div className="text-red-600 font-bold text-sm">No direction</div>
            <div className="text-red-500">z={pitZ}, all neighbors higher</div>
          </div>
          <div className="bg-white border border-slate-200 rounded-lg p-2 flex items-center justify-center text-slate-400 text-lg">→</div>
          <div className="bg-white border border-emerald-200 rounded-lg p-2">
            <div className="text-[10px] text-slate-400 uppercase mb-1">After fill</div>
            <div className="text-emerald-600 font-bold text-sm">{afterDir ? afterDir.label : 'No direction'}</div>
            <div className="text-emerald-700">z={filledZ}{afterDir ? `, flows ${afterDir.label}` : ''}</div>
          </div>
        </div>
      </div>
    </div>
  );
}

function FlowGrid({grid,flowDir,label}:{grid:number[][],flowDir:(typeof D8[0]|null)[][],label:string}){
  const R=grid.length,C=grid[0].length;
  const zLo=Math.min(...grid.flat()), zHi=Math.max(...grid.flat());

  return(
    <svg width={C*CELL_AFTER} height={R*CELL_AFTER}
      className="rounded-xl border border-slate-200 shadow-sm select-none">
      {grid.map((row,r)=>row.map((z,c)=>{
        const isPit   =label==='before'&&r===1&&c===1;
        const isFilled=label==='after'  &&r===1&&c===1;
        const cx=c*CELL_AFTER+CELL_AFTER/2, cy=r*CELL_AFTER+CELL_AFTER/2;
        let fill=elevColor(z,zLo,zHi);
        let stroke='#94a3b8'; let sw=1;
        if(isPit)   {stroke='#ef4444';sw=3;}
        if(isFilled){fill='#fb923c';stroke='#ea580c';sw=2.5;}

        const dir=flowDir[r]?.[c];
        let arrowEl:React.ReactNode=null;
        if(dir){
          const ex=cx+dir.dc*(CELL_AFTER/2-10);
          const ey=cy+dir.dr*(CELL_AFTER/2-10);
          const sx=cx-dir.dc*6, sy=cy-dir.dr*6;
          const arrowColor=isFilled?'#166534':isPit?'#dc2626':'white';
          arrowEl=<Arrow x1={sx} y1={sy} x2={ex} y2={ey} color={arrowColor}/>;
        }

        return(
          <g key={`${r}-${c}`}>
            <rect x={c*CELL_AFTER} y={r*CELL_AFTER} width={CELL_AFTER} height={CELL_AFTER}
              fill={fill} stroke={stroke} strokeWidth={sw}/>
            {isPit?(
              <>
                <text x={cx} y={cy-4} textAnchor="middle" fontSize={14} fontWeight="700"
                  fill="#991b1b" style={{pointerEvents:'none'}}>{z}</text>
                <text x={cx} y={cy+12} textAnchor="middle" fontSize={9} fill="#dc2626"
                  style={{pointerEvents:'none'}}>⚠</text>
              </>
            ):isFilled?(
              <>
                <text x={cx} y={cy-4} textAnchor="middle" fontSize={13} fontWeight="700"
                  fill="#1e293b" style={{pointerEvents:'none'}}>{z}</text>
                <text x={cx} y={cy+12} textAnchor="middle" fontSize={9} fill="#9a3412"
                  style={{pointerEvents:'none'}}>1→3</text>
              </>
            ):(
              <text x={cx} y={cy+5} textAnchor="middle" fontSize={14} fontWeight="700"
                fill="#1e293b" style={{pointerEvents:'none'}}>{z}</text>
            )}
            {arrowEl}
          </g>
        );
      }))}
    </svg>
  );
}

// ─── Main export ──────────────────────────────────────────────────────────────
type Tab='pit'|'wangliu'|'after';

export default function PitTeachWidget(){
  const [tab,setTab]=useState<Tab>('pit');
  const [grid, setGrid] = useState<number[][]>(() => TERRAIN.map(r=>[...r]));

  const steps = useMemo(() => wangLiuSteps(grid), [grid]);
  const filled = useMemo(() => steps[steps.length-1]?.filled ?? grid.map(r=>[...r]), [steps, grid]);
  const flowBefore = useMemo(() => computeFlow(grid), [grid]);
  const flowAfter  = useMemo(() => computeFlow(filled), [filled]);

  const TABS:{id:Tab;label:string}[]=[
    {id:'pit',      label:'🕳 The Pit'},
    {id:'wangliu',  label:'⛏ Wang & Liu Algorithm'},
    {id:'after',    label:'✅ After Filling → Flow Direction'},
  ];

  return(
    <div className="not-prose my-8 rounded-2xl border border-slate-200 bg-white shadow-xl overflow-hidden font-sans">
      {/* Header */}
      <div className="bg-gradient-to-r from-slate-700 to-slate-900 px-6 py-4">
        <h3 className="text-white font-bold text-lg tracking-tight">DEM Pit Filling — Interactive Tutorial</h3>
        <p className="text-slate-300 text-sm mt-0.5">What is a pit, why it blocks flow, and how Wang & Liu (2006) fixes it</p>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-slate-200 bg-slate-50 text-sm overflow-x-auto">
        {TABS.map(t=>(
          <button key={t.id} onClick={()=>setTab(t.id)}
            className={`px-4 py-3 font-medium border-b-2 whitespace-nowrap transition-colors ${tab===t.id?'border-slate-700 text-slate-900 bg-white':'border-transparent text-slate-500 hover:text-slate-700'}`}>
            {t.label}
          </button>
        ))}
      </div>

      {/* Content */}
      <div className="p-6">
        {tab==='pit'    &&<PitTab grid={grid} setGrid={setGrid}/>}
        {tab==='wangliu'&&<WangLiuTab steps={steps}/>}
        {tab==='after'  &&<AfterTab grid={grid} filled={filled} flowBefore={flowBefore} flowAfter={flowAfter}/>}
      </div>
    </div>
  );
}
