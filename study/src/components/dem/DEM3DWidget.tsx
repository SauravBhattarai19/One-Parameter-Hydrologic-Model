'use client';

import React, { useRef, useEffect, useState, useCallback } from 'react';

// ─── Types & constants ────────────────────────────────────────────────────────
type Grid = number[][];

const PRESETS: Record<string, Grid> = {
  valley: [
    [9,8,7,6,6,7,8,9],[8,7,6,5,5,6,7,8],[7,6,5,4,4,5,6,7],[6,5,4,3,3,4,5,6],
    [5,4,3,2,2,3,4,5],[4,3,2,1,1,2,3,4],[3,2,1,1,1,1,2,3],[2,1,1,1,1,1,1,2],
  ],
  mountain: [
    [2,2,3,3,3,3,2,2],[2,3,4,5,5,4,3,2],[3,4,6,7,7,6,4,3],[3,5,7,9,9,7,5,3],
    [3,5,7,9,9,7,5,3],[3,4,6,7,7,6,4,3],[2,3,4,5,5,4,3,2],[2,2,3,3,3,3,2,2],
  ],
  ridge: [
    [4,5,6,9,9,6,5,4],[3,4,5,8,8,5,4,3],[2,3,4,7,7,4,3,2],[1,2,3,6,6,3,2,1],
    [1,2,3,6,6,3,2,1],[2,3,4,7,7,4,3,2],[3,4,5,8,8,5,4,3],[4,5,6,9,9,6,5,4],
  ],
  slope: [
    [9,8,7,6,5,4,3,2],[9,8,7,6,5,4,3,2],[9,8,7,6,5,4,3,2],[9,8,7,6,5,4,3,2],
    [9,8,7,6,5,4,3,2],[9,8,7,6,5,4,3,2],[9,8,7,6,5,4,3,2],[9,8,7,6,5,4,3,2],
  ],
  basin: [
    [9,9,9,9,9,9,9,9],[9,7,7,7,7,7,7,9],[9,7,5,5,5,5,7,9],[9,7,5,3,3,5,7,9],
    [9,7,5,3,3,5,7,9],[9,7,5,5,5,5,7,9],[9,7,7,7,7,7,7,9],[9,9,9,9,9,9,9,9],
  ],
};

// ─── Color (returns [r,g,b] for shading) ─────────────────────────────────────
const STOPS: [number, [number,number,number]][] = [
  [0.00,[ 67,117,180]], [0.18,[116,196,163]], [0.40,[161,218,115]],
  [0.62,[255,230,130]], [0.82,[200,130, 70]], [1.00,[245,245,244]],
];
function elevRGB(z: number, lo: number, hi: number): [number,number,number] {
  const t = hi===lo ? 0.5 : (z-lo)/(hi-lo);
  let a=STOPS[0], b=STOPS[STOPS.length-1];
  for(let i=0;i<STOPS.length-1;i++) if(t>=STOPS[i][0]&&t<=STOPS[i+1][0]){a=STOPS[i];b=STOPS[i+1];break;}
  const u=(b[0]-a[0])===0?0:(t-a[0])/(b[0]-a[0]);
  const lp=(x:number,y:number)=>x+u*(y-x);
  return [lp(a[1][0],b[1][0]), lp(a[1][1],b[1][1]), lp(a[1][2],b[1][2])];
}
function rgb2str([r,g,b]: [number,number,number], shade=1): string {
  return `rgb(${Math.round(r*shade)},${Math.round(g*shade)},${Math.round(b*shade)})`;
}

// ─── D8 flow direction ────────────────────────────────────────────────────────
const D8=[
  {dr:0,dc:1,code:1,angle:90},{dr:1,dc:1,code:2,angle:135},{dr:1,dc:0,code:4,angle:180},
  {dr:1,dc:-1,code:8,angle:225},{dr:0,dc:-1,code:16,angle:270},{dr:-1,dc:-1,code:32,angle:315},
  {dr:-1,dc:0,code:64,angle:0},{dr:-1,dc:1,code:128,angle:45},
];
function computeFlow(grid: Grid) {
  const R=grid.length,C=grid[0].length;
  return grid.map((row,r)=>row.map((_,c)=>{
    let best=null as typeof D8[0]|null, mx=0;
    for(const d of D8){const nr=r+d.dr,nc=c+d.dc;if(nr<0||nr>=R||nc<0||nc>=C)continue;const s=(grid[r][c]-grid[nr][nc])/(d.dr&&d.dc?Math.SQRT2:1);if(s>mx){mx=s;best=d;}}
    return best;
  }));
}

// ─── Hillshading ──────────────────────────────────────────────────────────────
// Light from NW at 45° altitude: direction = normalize([-1,+1,1])
const LDIR = [-1/Math.sqrt(3), 1/Math.sqrt(3), 1/Math.sqrt(3)] as const;
function hillshade(grid: Grid, r: number, c: number, hScale: number): number {
  const R=grid.length, C=grid[0].length;
  const dzdx = c>0&&c<C-1 ? (grid[r][c+1]-grid[r][c-1])/2 : c===0 ? grid[r][c+1]-grid[r][c] : grid[r][c]-grid[r][c-1];
  const dzdy = r>0&&r<R-1 ? (grid[r+1][c]-grid[r-1][c])/2 : r===0 ? grid[r+1][c]-grid[r][c] : grid[r][c]-grid[r-1][c];
  const nx=-dzdx*hScale, ny=dzdy*hScale, nz=1.0;
  const len=Math.sqrt(nx*nx+ny*ny+nz*nz);
  const dot=nx*LDIR[0]/len + ny*LDIR[1]/len + nz/len;
  return 0.25 + 0.75*Math.max(0, dot);
}

// ─── 3-D projection ───────────────────────────────────────────────────────────
// World: wx=col(east), wy=elev×hScale(up), wz=row(south)
// Orthographic camera at azimuth phi (from south, CCW), elevation theta.
//   right vector  r = ( cosφ,        0,  −sinφ )
//   screen-up     u = (−sinφ·sinθ,  cosθ, −cosφ·sinθ )
//   depth axis    = ( sinφ,          0,   cosφ  )  — ascending = farther from camera
//
// offX, offY: canvas offset (pass 0,0 for unit/raw projection; canvas centre for display)
function makeProjector(azDeg:number, tiltDeg:number, scale:number, hScale:number, offX:number, offY:number, R:number, C:number) {
  const phi   = azDeg  * Math.PI / 180;
  const theta = tiltDeg * Math.PI / 180;
  const cosP=Math.cos(phi), sinP=Math.sin(phi);
  const cosT=Math.cos(theta), sinT=Math.sin(theta);
  return function project(col:number, row:number, elev:number) {
    const wx = col - (C-1)/2;
    const wz = row - (R-1)/2;
    const wy = elev * hScale;
    const sx = ( wx*cosP - wz*sinP) * scale + offX;
    const sy = ( wx*sinP*sinT - wy*cosT + wz*cosP*sinT) * scale + offY;
    const depth = wx*sinP + wz*cosP;   // ascending = farther from camera
    return { sx, sy, depth };
  };
}

// ─── Main widget ──────────────────────────────────────────────────────────────
const CW = 540, CH = 380;

export default function DEM3DWidget() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [grid, setGrid]       = useState<Grid>(()=>PRESETS.mountain.map(r=>[...r]));
  const [azimuth, setAzimuth] = useState(210);
  const [tilt, setTilt]       = useState(38);
  const [hScale, setHScale]   = useState(1.5);
  const [showFlow, setShowFlow] = useState(false);
  const [shade, setShade]     = useState(true);
  const [hovCell, setHovCell] = useState<{r:number,c:number,z:number}|null>(null);

  // Drag state
  const drag = useRef<{x:number;y:number;az:number;tilt:number}|null>(null);

  // Hit-test cells (for hover): store projected centers
  const cellCenters = useRef<{r:number;c:number;cx:number;cy:number}[]>([]);

  const render = useCallback(() => {
    const canvas = canvasRef.current; if(!canvas) return;
    const ctx = canvas.getContext('2d'); if(!ctx) return;
    const R=grid.length, C=grid[0].length;

    // ── Auto-scale: project the 8 extreme corners at unit scale to get bounding box,
    //    derive SCALE to fill 84 % of the canvas, then centre the BB on the canvas.
    const zMin=Math.min(...grid.flat()), zMax=Math.max(...grid.flat());
    const rawP = makeProjector(azimuth, tilt, 1, hScale, 0, 0, R, C);
    const bPts = [0,C-1].flatMap(c=>[0,R-1].flatMap(r=>[zMin,zMax].map(z=>rawP(c,r,z))));
    const rxs=bPts.map(p=>p.sx), rys=bPts.map(p=>p.sy);
    const [rxMin,rxMax]=[Math.min(...rxs),Math.max(...rxs)];
    const [ryMin,ryMax]=[Math.min(...rys),Math.max(...rys)];
    const xSpan=(rxMax-rxMin)||1, ySpan=(ryMax-ryMin)||1;
    const SCALE = Math.min(CW*0.84/xSpan, CH*0.84/ySpan);
    // Centre the bounding box on the canvas
    const offX = CW/2 - (rxMin+rxMax)/2 * SCALE;
    const offY = CH/2 - (ryMin+ryMax)/2 * SCALE;

    const proj = makeProjector(azimuth, tilt, SCALE, hScale, offX, offY, R, C);
    const fg = showFlow ? computeFlow(grid) : null;

    // Build sorted list of cells (painter's algorithm — ascending depth = far-first)
    const cells = [];
    for(let r=0;r<R;r++) for(let c=0;c<C;c++) {
      const ctr = proj(c+0.5, r+0.5, grid[r][c]);
      cells.push({ r, c, depth: ctr.depth });
    }
    cells.sort((a,b)=>a.depth-b.depth);

    ctx.clearRect(0,0,CW,CH);

    // Draw background gradient
    const bg = ctx.createLinearGradient(0,0,0,CH);
    bg.addColorStop(0,'#1e293b'); bg.addColorStop(1,'#0f172a');
    ctx.fillStyle=bg; ctx.fillRect(0,0,CW,CH);

    // Draw cells back-to-front
    const centers: typeof cellCenters.current = [];
    for(const {r,c} of cells) {
      const z = grid[r][c];
      const p00 = proj(c,   r,   z);
      const p10 = proj(c+1, r,   z);
      const p11 = proj(c+1, r+1, z);
      const p01 = proj(c,   r+1, z);
      const ctr = proj(c+0.5, r+0.5, z);
      centers.push({r,c,cx:ctr.sx,cy:ctr.sy});

      let baseRGB = elevRGB(z, zMin, zMax);
      const sh = shade ? hillshade(grid, r, c, hScale*0.3) : 0.85;
      const isHov = hovCell?.r===r && hovCell?.c===c;

      // Draw south face (between this row and row+1) if visible
      if(r < R-1) {
        const z2 = grid[r+1][c];
        if(z > z2) {
          const pb0 = proj(c,   r+1, z2); const pb1 = proj(c+1, r+1, z2);
          ctx.beginPath();
          ctx.moveTo(p01.sx,p01.sy); ctx.lineTo(p11.sx,p11.sy);
          ctx.lineTo(pb1.sx,pb1.sy); ctx.lineTo(pb0.sx,pb0.sy);
          ctx.closePath();
          ctx.fillStyle = rgb2str(baseRGB, sh*0.55);
          ctx.fill();
        }
      }
      // Draw east face
      if(c < C-1) {
        const z2 = grid[r][c+1];
        if(z > z2) {
          const pe0 = proj(c+1, r,   z2); const pe1 = proj(c+1, r+1, z2);
          ctx.beginPath();
          ctx.moveTo(p10.sx,p10.sy); ctx.lineTo(p11.sx,p11.sy);
          ctx.lineTo(pe1.sx,pe1.sy); ctx.lineTo(pe0.sx,pe0.sy);
          ctx.closePath();
          ctx.fillStyle = rgb2str(baseRGB, sh*0.7);
          ctx.fill();
        }
      }

      // Top face
      ctx.beginPath();
      ctx.moveTo(p00.sx,p00.sy); ctx.lineTo(p10.sx,p10.sy);
      ctx.lineTo(p11.sx,p11.sy); ctx.lineTo(p01.sx,p01.sy);
      ctx.closePath();
      if(isHov) {
        ctx.fillStyle = 'rgba(250,204,21,0.85)';
      } else {
        ctx.fillStyle = rgb2str(baseRGB, sh);
      }
      ctx.fill();
      // Edge
      ctx.strokeStyle = isHov ? '#fbbf24' : 'rgba(0,0,0,0.18)';
      ctx.lineWidth = isHov ? 1.5 : 0.5;
      ctx.stroke();

      // Elevation label on top face (only for hovered cell)
      if(isHov) {
        ctx.fillStyle='#1e293b'; ctx.font='bold 12px monospace'; ctx.textAlign='center';
        ctx.fillText(`z=${z}`, ctr.sx, ctr.sy+4);
      }
    }
    cellCenters.current = centers;

    // Flow arrows
    if(fg && showFlow) {
      for(let r=0;r<R;r++) for(let c=0;c<C;c++) {
        const d=fg[r][c]; if(!d) continue;
        const z=grid[r][c];
        const from = proj(c+0.5, r+0.5, z+0.3);
        const to   = proj(c+0.5+d.dc*0.4, r+0.5+d.dr*0.4, grid[r+d.dr]?.[c+d.dc]!==undefined?grid[r+d.dr][c+d.dc]+0.3:z+0.3);
        ctx.beginPath(); ctx.moveTo(from.sx,from.sy); ctx.lineTo(to.sx,to.sy);
        ctx.strokeStyle='rgba(255,255,255,0.75)'; ctx.lineWidth=1.5; ctx.stroke();
        // arrowhead
        const dx=to.sx-from.sx, dy=to.sy-from.sy, len=Math.sqrt(dx*dx+dy*dy)||1;
        const ux=dx/len, uy=dy/len, hs=5;
        ctx.beginPath();
        ctx.moveTo(to.sx, to.sy);
        ctx.lineTo(to.sx-hs*ux+hs*0.4*uy, to.sy-hs*uy-hs*0.4*ux);
        ctx.lineTo(to.sx-hs*ux-hs*0.4*uy, to.sy-hs*uy+hs*0.4*ux);
        ctx.closePath(); ctx.fillStyle='rgba(255,255,255,0.75)'; ctx.fill();
      }
    }

    // HUD text
    ctx.fillStyle='rgba(255,255,255,0.5)'; ctx.font='11px monospace'; ctx.textAlign='left';
    ctx.fillText(`Az: ${azimuth}°  Tilt: ${tilt}°  H×: ${hScale.toFixed(1)}`, 10, 18);
    ctx.fillText('Drag to rotate', 10, CH-10);
  }, [grid, azimuth, tilt, hScale, showFlow, shade, hovCell]);

  useEffect(()=>{ render(); }, [render]);

  // Mouse drag for rotation
  const onMouseDown = (e: React.MouseEvent) => {
    drag.current = { x: e.clientX, y: e.clientY, az: azimuth, tilt };
  };
  const onMouseMove = useCallback((e: React.MouseEvent) => {
    if(drag.current) {
      const dAz   = (e.clientX - drag.current.x) * 0.5;
      const dTilt = (e.clientY - drag.current.y) * 0.25;
      setAzimuth(((drag.current.az + dAz) % 360 + 360) % 360);
      setTilt(Math.max(10, Math.min(80, drag.current.tilt + dTilt)));
    } else {
      // Hover hit-test
      const rect = canvasRef.current?.getBoundingClientRect();
      if(!rect) return;
      const mx=e.clientX-rect.left, my=e.clientY-rect.top;
      const THRESH=20;
      const found = cellCenters.current.find(c=>Math.hypot(c.cx-mx,c.cy-my)<THRESH);
      setHovCell(found ? {r:found.r,c:found.c,z:grid[found.r][found.c]} : null);
    }
  }, [grid]);
  const onMouseUp = () => { drag.current=null; };

  // Touch support for rotation
  const lastTouch = useRef<{x:number,y:number,az:number,tilt:number}|null>(null);
  const onTouchStart = (e: React.TouchEvent) => {
    const t=e.touches[0];
    lastTouch.current={x:t.clientX,y:t.clientY,az:azimuth,tilt};
  };
  const onTouchMove = (e: React.TouchEvent) => {
    if(!lastTouch.current) return;
    const t=e.touches[0];
    const dAz=(t.clientX-lastTouch.current.x)*0.5;
    const dTilt=(t.clientY-lastTouch.current.y)*0.25;
    setAzimuth(((lastTouch.current.az+dAz)%360+360)%360);
    setTilt(Math.max(10,Math.min(80,lastTouch.current.tilt+dTilt)));
  };

  return (
    <div className="not-prose my-8 rounded-2xl border border-slate-200 bg-slate-900 shadow-xl overflow-hidden font-sans">
      <div className="bg-gradient-to-r from-slate-800 to-slate-900 px-6 py-3 border-b border-slate-700 flex items-center justify-between">
        <div>
          <h3 className="text-white font-bold tracking-tight">3D DEM Viewer</h3>
          <p className="text-slate-400 text-xs mt-0.5">Drag to rotate · Hover a cell for its elevation · Flow arrows optional</p>
        </div>
        {hovCell && (
          <div className="bg-yellow-400/20 border border-yellow-400/40 rounded-lg px-3 py-1.5 text-yellow-300 text-sm font-mono">
            ({hovCell.r},{hovCell.c}) z = {hovCell.z} m
          </div>
        )}
      </div>

      <div className="flex flex-col lg:flex-row">
        {/* Canvas */}
        <canvas ref={canvasRef} width={CW} height={CH}
          className="cursor-grab active:cursor-grabbing flex-shrink-0 block"
          style={{width:'100%',maxWidth:CW,height:'auto'}}
          onMouseDown={onMouseDown} onMouseMove={onMouseMove}
          onMouseUp={onMouseUp} onMouseLeave={onMouseUp}
          onTouchStart={onTouchStart} onTouchMove={onTouchMove} onTouchEnd={()=>{lastTouch.current=null;}}
        />

        {/* Controls */}
        <div className="flex flex-col gap-4 p-5 lg:w-56 bg-slate-800/60">
          {/* Presets */}
          <div>
            <p className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider mb-2">Terrain</p>
            <div className="flex flex-col gap-1.5">
              {Object.keys(PRESETS).map(k=>(
                <button key={k} onClick={()=>setGrid(PRESETS[k].map(r=>[...r]))}
                  className={`text-left px-3 py-1.5 rounded-lg text-sm font-medium capitalize transition-colors ${
                    JSON.stringify(grid)===JSON.stringify(PRESETS[k])
                    ?'bg-sky-600 text-white'
                    :'bg-slate-700 text-slate-300 hover:bg-slate-600'}`}>
                  {k}
                </button>
              ))}
            </div>
          </div>

          {/* Height exaggeration */}
          <div>
            <p className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider mb-1.5">Height exag ×{hScale.toFixed(1)}</p>
            <input type="range" min={0.3} max={4} step={0.1} value={hScale}
              onChange={e=>setHScale(Number(e.target.value))}
              className="w-full accent-sky-500"/>
            <div className="flex justify-between text-[10px] text-slate-500 mt-0.5">
              <span>0.3× flat</span><span>4× steep</span>
            </div>
          </div>

          {/* Azimuth */}
          <div>
            <p className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider mb-1.5">Azimuth {Math.round(azimuth)}°</p>
            <input type="range" min={0} max={359} value={azimuth}
              onChange={e=>setAzimuth(Number(e.target.value))}
              className="w-full accent-sky-500"/>
          </div>

          {/* Tilt */}
          <div>
            <p className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider mb-1.5">Tilt {Math.round(tilt)}°</p>
            <input type="range" min={10} max={80} value={tilt}
              onChange={e=>setTilt(Number(e.target.value))}
              className="w-full accent-sky-500"/>
            <div className="flex justify-between text-[10px] text-slate-500 mt-0.5">
              <span>Flat</span><span>Top-down</span>
            </div>
          </div>

          {/* Toggles */}
          <div className="space-y-2">
            <label className="flex items-center gap-2 text-sm text-slate-300 cursor-pointer">
              <input type="checkbox" checked={showFlow} onChange={e=>setShowFlow(e.target.checked)} className="rounded accent-sky-500"/>
              Flow arrows
            </label>
            <label className="flex items-center gap-2 text-sm text-slate-300 cursor-pointer">
              <input type="checkbox" checked={shade} onChange={e=>setShade(e.target.checked)} className="rounded accent-sky-500"/>
              Hillshading
            </label>
          </div>

          {/* Mini legend */}
          <div className="mt-auto">
            <p className="text-[10px] text-slate-500 mb-1">Elevation</p>
            <div className="h-2 rounded-full w-full" style={{background:'linear-gradient(to right,rgb(67,117,180),rgb(161,218,115),rgb(200,130,70),rgb(245,245,244))'}}/>
            <div className="flex justify-between text-[10px] text-slate-500 mt-0.5"><span>Low</span><span>High</span></div>
          </div>
        </div>
      </div>
    </div>
  );
}
