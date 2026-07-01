'use client';

import React, { useState } from 'react';

// ─── Term metadata ────────────────────────────────────────────────────────────

interface TermMeta {
  id: number;
  symbol: string;
  label: string;
  circle: string;
  color: 'sky' | 'violet' | 'amber' | 'red' | 'green';
  canDrop: boolean;
  magnitude: number;
  magnitudeLabel: string;
}

const TERMS: TermMeta[] = [
  { id: 0, symbol: '∂Q/∂t',       label: 'Local acceleration',   circle: '①', color: 'sky',    canDrop: true,  magnitude: 0.007, magnitudeLabel: '~0.007×' },
  { id: 1, symbol: '∂(Q²/A)/∂x', label: 'Convective accel.',    circle: '②', color: 'violet', canDrop: true,  magnitude: 0.09,  magnitudeLabel: '~F²×'   },
  { id: 2, symbol: 'gA·∂h/∂x',   label: 'Pressure gradient',    circle: '③', color: 'amber',  canDrop: true,  magnitude: 0.15,  magnitudeLabel: '~0.15×'  },
  { id: 3, symbol: 'gA·S_f',      label: 'Friction slope',       circle: '④', color: 'red',    canDrop: false, magnitude: 1.0,   magnitudeLabel: '~1.00×'  },
  { id: 4, symbol: 'gA·S₀',       label: 'Gravity (bed slope)',  circle: '⑤', color: 'green',  canDrop: false, magnitude: 1.0,   magnitudeLabel: '~1.00×'  },
];

// ─── Color maps ───────────────────────────────────────────────────────────────

const COLOR_NORMAL: Record<TermMeta['color'], string> = {
  sky:    'bg-sky-50 border-sky-300 text-sky-800 hover:bg-sky-100',
  violet: 'bg-violet-50 border-violet-300 text-violet-800 hover:bg-violet-100',
  amber:  'bg-amber-50 border-amber-300 text-amber-800 hover:bg-amber-100',
  red:    'bg-red-50 border-red-300 text-red-800 hover:bg-red-100',
  green:  'bg-green-50 border-green-300 text-green-800 hover:bg-green-100',
};

const COLOR_SELECTED: Record<TermMeta['color'], string> = {
  sky:    'bg-sky-100 border-sky-500 text-sky-900 ring-2 ring-sky-300',
  violet: 'bg-violet-100 border-violet-500 text-violet-900 ring-2 ring-violet-300',
  amber:  'bg-amber-100 border-amber-500 text-amber-900 ring-2 ring-amber-300',
  red:    'bg-red-100 border-red-500 text-red-900 ring-2 ring-red-300',
  green:  'bg-green-100 border-green-500 text-green-900 ring-2 ring-green-300',
};

const PANEL_ACCENT: Record<TermMeta['color'], string> = {
  sky:    'text-sky-700',
  violet: 'text-violet-700',
  amber:  'text-amber-700',
  red:    'text-red-700',
  green:  'text-green-700',
};

const PANEL_BADGE: Record<TermMeta['color'], string> = {
  sky:    'bg-sky-100 text-sky-900 border border-sky-300',
  violet: 'bg-violet-100 text-violet-900 border border-violet-300',
  amber:  'bg-amber-100 text-amber-900 border border-amber-300',
  red:    'bg-red-100 text-red-900 border border-red-300',
  green:  'bg-green-100 text-green-900 border border-green-300',
};

// ─── Defaults for α/β display ─────────────────────────────────────────────────

const DEFAULT_N  = 0.04;
const DEFAULT_S0 = 0.001;
const DEFAULT_B  = 10;

function computeAlpha(n: number, s0: number, b: number): number {
  return Math.pow(s0, 0.5) / (n * Math.pow(b, 2 / 3));
}

const ALPHA = computeAlpha(DEFAULT_N, DEFAULT_S0, DEFAULT_B);
const BETA  = 5 / 3;

// ─── MagnitudeBar ─────────────────────────────────────────────────────────────

function MagnitudeBar({ magnitude, active }: { magnitude: number; active: boolean }) {
  const logMin  = Math.log10(0.005);
  const logMax  = Math.log10(2.0);
  const logVal  = Math.log10(magnitude);
  const fraction = Math.max(0, Math.min(1, (logVal - logMin) / (logMax - logMin)));
  const widthPx = Math.round(fraction * 60);

  return (
    <div className="h-1.5 rounded-full bg-slate-200 w-[60px] mt-1">
      <div
        className={`h-1.5 rounded-full transition-all duration-300 ${active ? 'bg-blue-500' : 'bg-slate-300'}`}
        style={{ width: `${widthPx}px` }}
      />
    </div>
  );
}

// ─── Continuity SVG ───────────────────────────────────────────────────────────

function ControlVolumeSVG() {
  return (
    <svg
      viewBox="0 0 560 130"
      className="w-full"
      style={{ height: 130 }}
      aria-label="Control volume schematic for continuity equation derivation"
    >
      {/* Water body trapezoid */}
      <polygon
        points="30,30 530,30 530,100 30,100"
        fill="#dbeafe"
        stroke="#3b82f6"
        strokeWidth="2"
      />

      {/* Left face — Q(x) inlet arrow */}
      <defs>
        <marker id="arrowL" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
          <path d="M0,0 L6,3 L0,6 Z" fill="#1d4ed8" />
        </marker>
        <marker id="arrowR" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
          <path d="M0,0 L6,3 L0,6 Z" fill="#1d4ed8" />
        </marker>
        <marker id="arrowDown" markerWidth="6" markerHeight="6" refX="3" refY="5" orient="auto">
          <path d="M0,0 L3,6 L6,0 Z" fill="#2563eb" />
        </marker>
        <marker id="arrowUp" markerWidth="6" markerHeight="6" refX="3" refY="1" orient="auto">
          <path d="M0,6 L3,0 L6,6 Z" fill="#1d4ed8" />
        </marker>
      </defs>

      {/* Q(x) left arrow */}
      <line x1="6" y1="65" x2="28" y2="65" stroke="#1d4ed8" strokeWidth="2" markerEnd="url(#arrowL)" />
      <text x="4" y="58" fontSize="10" fill="#1d4ed8" fontFamily="monospace">Q(x)</text>

      {/* Q(x+Δx) right arrow — thicker to hint larger Q */}
      <line x1="532" y1="65" x2="554" y2="65" stroke="#1d4ed8" strokeWidth="3" markerEnd="url(#arrowR)" />
      <text x="534" y="58" fontSize="10" fill="#1d4ed8" fontFamily="monospace">Q(x+Δx)</text>

      {/* Rain drops */}
      {[150, 280, 410].map((x, idx) => (
        <g key={idx}>
          <line x1={x} y1="4" x2={x} y2="26" stroke="#2563eb" strokeWidth="1.5" markerEnd="url(#arrowDown)" />
        </g>
      ))}
      <text x="195" y="10" fontSize="9" fill="#1e40af" textAnchor="middle" fontFamily="sans-serif">
        q_lat (rain / lateral)
      </text>

      {/* A(x,t) label inside reach */}
      <text x="280" y="70" fontSize="13" fill="#1d4ed8" textAnchor="middle" fontFamily="monospace" fontWeight="bold">
        A(x,t)
      </text>

      {/* ∂A/∂t rising indicator */}
      <line x1="470" y1="55" x2="470" y2="35" stroke="#1d4ed8" strokeWidth="1.5" markerEnd="url(#arrowUp)" />
      <text x="475" y="53" fontSize="9" fill="#1d4ed8" fontFamily="monospace">∂A/∂t</text>

      {/* Δx label at bottom */}
      <line x1="30"  y1="112" x2="100" y2="112" stroke="#94a3b8" strokeWidth="1" />
      <line x1="530" y1="112" x2="460" y2="112" stroke="#94a3b8" strokeWidth="1" />
      <line x1="30"  y1="108" x2="30"  y2="116" stroke="#94a3b8" strokeWidth="1" />
      <line x1="530" y1="108" x2="530" y2="116" stroke="#94a3b8" strokeWidth="1" />
      <text x="280" y="120" fontSize="10" fill="#64748b" textAnchor="middle" fontFamily="monospace">
        ← Δx →
      </text>
    </svg>
  );
}

// ─── Continuity section ───────────────────────────────────────────────────────

interface ContPill {
  key: 'dAdt' | 'dQdx' | 'qlat';
  label: string;
  color: string;
}

const CONT_PILLS: ContPill[] = [
  { key: 'dAdt',  label: '∂A/∂t',  color: 'bg-sky-100 border-sky-400 text-sky-800 hover:bg-sky-200' },
  { key: 'dQdx',  label: '∂Q/∂x',  color: 'bg-violet-100 border-violet-400 text-violet-800 hover:bg-violet-200' },
  { key: 'qlat',  label: '= q_lat', color: 'bg-green-100 border-green-400 text-green-800 hover:bg-green-200' },
];

const CONT_EXPLANATIONS: Record<ContPill['key'], { title: string; body: string }> = {
  dAdt: {
    title: '∂A/∂t — Rate of storage change',
    body: 'Rate of change of cross-sectional area with time. For a rectangular channel with width B: A = B·h, so ∂A/∂t = B·∂h/∂t — it\'s literally how fast the water level is rising or falling. When a flood arrives, the river swells: ∂A/∂t > 0. When it recedes: ∂A/∂t < 0. This term is how the equation "knows" water is accumulating in the reach.',
  },
  dQdx: {
    title: '∂Q/∂x — Discharge gradient',
    body: 'How discharge changes as you move downstream. If Q(x+Δx) > Q(x), more water is leaving than arriving — the reach is draining. If Q grows downstream, the stored volume must shrink. The negative sign in the derivation encodes this: net outflow (∂Q/∂x > 0) causes storage to decrease (∂A/∂t < 0).',
  },
  qlat: {
    title: 'q_lat — Lateral inflow',
    body: 'Lateral inflow per unit channel length [m²/s]. This is rain falling directly on the river surface, a tributary merging, or subsurface seepage. In OPM, q_lat is the effective runoff [m/s] × cell_area / reach_length — the output of the VSA/Green-Ampt module entering the kinematic wave router.',
  },
};

function ContinuitySection() {
  const [selectedCont, setSelectedCont] = useState<ContPill['key'] | null>(null);

  const handlePill = (key: ContPill['key']) => {
    setSelectedCont(prev => (prev === key ? null : key));
  };

  return (
    <section>
      <h4 className="font-bold text-slate-800 text-base mb-2">
        ① Continuity Equation — Water Conservation
      </h4>

      {/* Control volume SVG */}
      <div className="rounded-xl border border-slate-200 bg-slate-50 overflow-hidden mb-3">
        <ControlVolumeSVG />
      </div>

      {/* Derivation text */}
      <div className="bg-slate-50 rounded-lg p-3 font-mono text-xs text-slate-700 mb-4 leading-relaxed whitespace-pre-line border border-slate-200">
        {`Mass balance for the reach of length Δx:
  d/dt(A · Δx) = Q(x) − Q(x+Δx) + q_lat · Δx
Divide both sides by Δx, take limit Δx → 0:
  ∂A/∂t = −∂Q/∂x + q_lat
  ∴  ∂A/∂t + ∂Q/∂x = q_lat  ✓`}
      </div>

      {/* Clickable pills */}
      <p className="text-xs text-slate-500 mb-2 font-medium">
        Click any term to understand it (continuity always holds — nothing to drop here):
      </p>
      <div className="flex flex-wrap gap-2 mb-3">
        {CONT_PILLS.map(pill => (
          <button
            key={pill.key}
            onClick={() => handlePill(pill.key)}
            className={`border rounded-lg px-3 py-1.5 font-mono text-sm font-medium transition-all duration-150 cursor-pointer select-none ${pill.color} ${selectedCont === pill.key ? 'ring-2 ring-offset-1 ring-blue-400' : ''}`}
          >
            {pill.label}
          </button>
        ))}
      </div>

      {/* Expansion panel */}
      {selectedCont !== null && (
        <div className="bg-blue-50 border border-blue-200 rounded-xl p-4 text-sm text-slate-700 leading-relaxed">
          <p className="font-semibold text-blue-800 mb-1">{CONT_EXPLANATIONS[selectedCont].title}</p>
          <p className="text-slate-600">{CONT_EXPLANATIONS[selectedCont].body}</p>
        </div>
      )}
    </section>
  );
}

// ─── Momentum term explanations ───────────────────────────────────────────────

interface TermExplanation {
  meaning: string;
  origin: string;
  canDrop: false | { why: string; loss: string };
}

const TERM_EXPLANATIONS: TermExplanation[] = [
  // ① ∂Q/∂t
  {
    meaning:
      'The rate at which the momentum stored IN this reach is changing with time. A rising hydrograph means discharge is growing — you are actively accelerating the water mass. This is inertia: the water "resists" being sped up or slowed down.\n\nThink of it like a hose: when you first open a tap, you need extra force to accelerate the stationary water. Once it\'s flowing steadily, you don\'t need that extra force anymore. ∂Q/∂t captures that transient push.',
    origin:
      'From Newton: F = ma. The mass of water in a reach is ρ·A·Δx. Its acceleration is (1/A)·∂Q/∂t [m/s²] (since Q = A·u → ∂Q/∂t = A·∂u/∂t for constant A). So the force per unit length is ρ·∂Q/∂t.',
    canDrop: {
      why:
        'Flood waves evolve over hours to days. A typical river flood doubles its discharge over 3–6 hours:\n  ∂Q/∂t ≈ 100 m³/s / (4 hrs) ≈ 0.007 m²/s²\nCompare to the gravity term gAS₀ = 9.8 × 10 × 0.001 = 0.098 m²/s²\nRatio: 0.007/0.098 ≈ 7% → safely ignored for slowly-varying floods.',
      loss:
        'For dam-break floods, storm surges, and tidal waves — where the flow changes in seconds to minutes — this term matters enormously. Kinematic wave completely fails for these scenarios.',
    },
  },
  // ② ∂(Q²/A)/∂x
  {
    meaning:
      'The momentum that the flow CARRIES WITH IT as it moves through space. As water moves from a deep slow section to a shallow fast section, its velocity changes — and so does its momentum. This is the "kinetic energy gradient" of the flow.\n\nThink of it as: the water downstream is moving faster than upstream — so the water moving from upstream to downstream is "arriving faster than it left". That velocity change requires a force.\n\nThis term is also written as ∂(u²A)/∂x or (for constant A) u·∂u/∂x — the classical convective acceleration from fluid mechanics.',
    origin:
      'The momentum flux through a cross-section is ρ·Q·u = ρ·Q²/A [N/m²]. The divergence of this flux (how much more momentum is leaving downstream vs. entering upstream) gives the net force per unit length: ∂(Q²/A)/∂x.',
    canDrop: {
      why:
        'In subcritical rivers (Froude number F = u/√(gh) < 1), this term scales as F² times the pressure gradient:\n  convective / pressure ≈ F²\nFor a mountain river with F = 0.3: ratio = 0.09. Only 9%.\nMost natural rivers have F between 0.1 and 0.5 → F² = 1% to 25%. Conservative to ignore.',
      loss:
        'Accuracy near hydraulic structures (weirs, bridges, contractions), hydraulic jumps, and any transition between subcritical and supercritical flow. Required for correct modeling of rapidly varied flow profiles.',
    },
  },
  // ③ gA·∂h/∂x
  {
    meaning:
      'The hydrostatic pressure difference between the upstream and downstream faces of the water element. When the water SURFACE slopes downward (∂h/∂x < 0 going downstream), the taller water column on the upstream face pushes harder than the downstream face → flow accelerates.\n\nThis is the BACKWATER TERM — the most important term we drop in kinematic wave.\nCrucially: it lets downstream conditions affect upstream flow. A dam raising the water level downstream tilts the water surface, which changes ∂h/∂x, which propagates upstream — kilometers, in slow-moving water.',
    origin:
      'For hydrostatic pressure, force on a vertical face of area A is ρgh·A/2 (at depth h/2). The net pressure force between faces at x and x+Δx: ρg·∂(Ah)/∂x·Δx ≈ ρgA·∂h/∂x·Δx.',
    canDrop: {
      why:
        'On steep slopes (S₀ > ~0.001), the bed slope term gAS₀ dominates the water-surface slope term gA·∂h/∂x.\nTypical flood-wave slope: ∂h/∂x ≈ 0.00005 (5 cm per km).\nCompare: S₀ = 0.001 (1 m per km) → water surface slope is only 5% of bed slope.\n→ Dropping it introduces ~5% error.',
      loss:
        'BACKWATER EFFECTS. Without this term:\n• A reservoir downstream has NO influence on upstream routing.\n• Tidal rivers cannot be modeled.\n• Flat-land flooding (deltaic channels, floodplains) is wrong — kinematic wave requires a bed slope to "know" which way water flows.\nThe diffusive wave keeps this term and fixes all of the above.',
    },
  },
  // ④ gA·S_f
  {
    meaning:
      'The frictional resistance that opposes flow — the "drag" of the riverbed and banks. S_f is the friction (or energy gradient) slope:\n  S_f = Q|Q|·n² / (A²·R^(4/3))   [Manning-Strickler form]\nwhere n is Manning\'s roughness and R is hydraulic radius.\n\nAlways acts OPPOSITE to flow direction. If water flows downstream, friction pulls it back. Without this term, gravity would accelerate the river to infinite velocity.',
    origin:
      'This and gravity ⑤ are the two "permanent" forces. Together they define the equilibrium: at steady uniform flow, gAS_f = gAS₀ exactly, and ∂Q/∂t = ∂Q/∂x = 0. All simplifications keep both terms — they are the physics we can NEVER ignore.\n\nSetting S_f = S₀ (the kinematic assumption) and using Manning\'s equation for S_f immediately gives Q = αA^β. Friction IS the kinematic wave — it\'s the term that connects depth to discharge.',
    canDrop: false,
  },
  // ⑤ gA·S₀
  {
    meaning:
      'The gravitational force pulling water downhill. S₀ = (z_up − z_down)/Δx is the bed slope [m/m]. The component of gravity along the channel is g·sin(θ) ≈ g·S₀ for gentle slopes.\n\nThis is the ENGINE of all river flow. It is never dropped.',
    origin:
      'In kinematic wave routing, S₀ is known exactly from the DEM (bed elevation raster). It never changes with time. Every cell in OPM has its own S₀ computed from its D8 downstream neighbor: S₀ = (z_i − z_ds) / distance. The spatial variability of S₀ is what makes 2-D kinematic wave routing interesting.',
    canDrop: false,
  },
];

// ─── Explanation Panel ────────────────────────────────────────────────────────

function ExplanationPanel({ selectedTerm }: { selectedTerm: number | null }) {
  if (selectedTerm === null) {
    return (
      <div className="bg-slate-50 rounded-xl border border-slate-200 p-4 sticky top-4 min-h-[200px] flex items-center justify-center">
        <p className="text-slate-400 text-sm text-center italic">
          ← Click any term to understand its physical meaning.
        </p>
      </div>
    );
  }

  const term = TERMS[selectedTerm];
  const expl = TERM_EXPLANATIONS[selectedTerm];
  const accentColor = PANEL_ACCENT[term.color];
  const badgeClass  = PANEL_BADGE[term.color];

  return (
    <div className="bg-slate-50 rounded-xl border border-slate-200 p-4 space-y-4 sticky top-4 overflow-y-auto max-h-[80vh]">
      {/* Term header */}
      <div>
        <span className={`inline-block font-mono text-xl font-bold px-2 py-1 rounded-lg ${badgeClass}`}>
          {term.circle} {term.symbol}
        </span>
        <p className={`mt-1 text-sm font-semibold ${accentColor}`}>{term.label}</p>
      </div>

      {/* Physical meaning */}
      <div>
        <p className="text-[11px] font-bold uppercase tracking-widest text-slate-400 mb-1">Physical Meaning</p>
        <p className="text-sm text-slate-700 leading-relaxed whitespace-pre-line">{expl.meaning}</p>
      </div>

      {/* Mathematical origin */}
      <div>
        <p className="text-[11px] font-bold uppercase tracking-widest text-slate-400 mb-1">Mathematical Origin</p>
        <p className="text-sm text-slate-700 leading-relaxed whitespace-pre-line">{expl.origin}</p>
      </div>

      {/* Droppable extras */}
      {expl.canDrop !== false && (
        <>
          <div>
            <p className="text-[11px] font-bold uppercase tracking-widest text-slate-400 mb-1">Why We Can Drop It</p>
            <div className="bg-white rounded-lg border border-slate-200 p-3">
              <p className="text-sm text-slate-700 leading-relaxed font-mono text-xs whitespace-pre-line">
                {expl.canDrop.why}
              </p>
            </div>
          </div>
          <div>
            <p className="text-[11px] font-bold uppercase tracking-widest text-slate-400 mb-1">What You Lose</p>
            <div className="bg-amber-50 rounded-lg border border-amber-200 p-3">
              <p className="text-sm text-amber-800 leading-relaxed whitespace-pre-line">{expl.canDrop.loss}</p>
            </div>
          </div>
        </>
      )}

      {/* Cannot-drop notice */}
      {expl.canDrop === false && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-3">
          <p className="text-xs font-semibold text-red-700">🔒 Cannot be dropped — this force is always present.</p>
        </div>
      )}
    </div>
  );
}

// ─── Term Box ─────────────────────────────────────────────────────────────────

function TermBox({
  term,
  isSelected,
  isDropped,
  onSelect,
  onToggleDrop,
}: {
  term: TermMeta;
  isSelected: boolean;
  isDropped: boolean;
  onSelect: () => void;
  onToggleDrop: () => void;
}) {
  const baseClass = isDropped
    ? 'bg-slate-100 border-slate-200 text-slate-400 opacity-40 cursor-pointer'
    : isSelected
    ? COLOR_SELECTED[term.color] + ' cursor-pointer'
    : COLOR_NORMAL[term.color] + ' cursor-pointer';

  return (
    <div className="flex items-center gap-2">
      <div className="flex flex-col items-start gap-0.5">
        <button
          onClick={onSelect}
          className={`border rounded-lg px-3 py-2 text-sm font-mono font-medium select-none transition-all duration-200 ${baseClass} ${isDropped ? 'line-through' : ''}`}
          title={isSelected ? `Deselect ${term.label}` : `Learn about ${term.label}`}
        >
          <span className="font-sans text-xs not-italic mr-1">{term.circle}</span>
          {term.symbol}
        </button>
        <span className={`text-xs leading-tight ml-0.5 ${isDropped ? 'text-slate-300' : 'text-slate-500'}`}>
          {term.label}
        </span>
        <MagnitudeBar magnitude={term.magnitude} active={!isDropped} />
        <span className={`text-[10px] font-mono ml-0.5 ${isDropped ? 'text-slate-300' : 'text-slate-400'}`}>
          {term.magnitudeLabel}
        </span>
      </div>

      {/* Drop/Restore or lock */}
      <div className="self-start mt-1">
        {term.canDrop ? (
          <button
            onClick={onToggleDrop}
            className={`text-[11px] px-2 py-0.5 rounded border font-medium transition-all duration-150 ${
              isDropped
                ? 'border-slate-300 bg-slate-50 text-slate-500 hover:bg-slate-100'
                : 'border-red-300 bg-red-50 text-red-600 hover:bg-red-100'
            }`}
            title={isDropped ? `Restore ${term.label}` : `Drop ${term.label} from equation`}
          >
            {isDropped ? '↩ Restore' : '× Drop'}
          </button>
        ) : (
          <span className="text-base" title="Cannot be dropped — always required">🔒</span>
        )}
      </div>
    </div>
  );
}

// ─── Simplification banner ────────────────────────────────────────────────────

function SimplificationBanner({ dropped }: { dropped: Set<number> }) {
  let label: string;
  let colorClass: string;

  if (dropped.has(0) && dropped.has(1) && dropped.has(2)) {
    label      = 'Kinematic Wave ← used in OPM';
    colorClass = 'bg-green-100 border-green-400 text-green-800';
  } else if (dropped.has(0) && dropped.has(1)) {
    label      = 'Diffusive Wave';
    colorClass = 'bg-amber-100 border-amber-400 text-amber-800';
  } else if (dropped.size === 0) {
    label      = 'Full Dynamic Wave (Saint-Venant)';
    colorClass = 'bg-slate-100 border-slate-300 text-slate-700';
  } else {
    label      = 'Simplified Saint-Venant';
    colorClass = 'bg-blue-50 border-blue-300 text-blue-700';
  }

  return (
    <div className={`mt-4 border rounded-lg px-4 py-2 text-sm font-semibold transition-colors duration-300 ${colorClass}`}>
      {label}
    </div>
  );
}

// ─── Kinematic result + derivation ───────────────────────────────────────────

function KinematicDerivation({ isKinematic }: { isKinematic: boolean }) {
  return (
    <div className={`transition-opacity duration-300 ${isKinematic ? 'opacity-100' : 'opacity-40 pointer-events-none'}`}>
      {!isKinematic && (
        <p className="text-sm text-slate-500 italic mb-3">
          Drop terms ①②③ to see the kinematic wave emerge.
        </p>
      )}

      {/* Kinematic balance result */}
      <div className="bg-green-50 border border-green-300 rounded-lg px-4 py-3 mb-3">
        <p className="text-xs font-semibold text-green-700 uppercase tracking-wider mb-1">
          After dropping ①②③ — remaining balance:
        </p>
        <p className="font-mono text-sm text-green-900">
          gA·S_f = gA·S₀ &nbsp;&nbsp;→&nbsp;&nbsp; S_f = S₀
        </p>
        <p className="text-xs text-green-700 mt-1">
          The flow is ALWAYS in local equilibrium: the channel instantaneously adjusts to uniform flow.
          This is the quasi-steady approximation.
        </p>
      </div>

      <div className="space-y-2">
        {/* Step 1 */}
        <div className="bg-slate-50 rounded-lg px-4 py-2.5 border border-slate-200">
          <p className="text-[11px] text-slate-400 font-semibold uppercase tracking-wider mb-0.5">Step 1</p>
          <p className="font-mono text-sm text-slate-800">S_f = S₀</p>
          <p className="text-xs text-slate-500 mt-0.5">
            The flow is ALWAYS in local equilibrium: the channel instantaneously adjusts to uniform flow.
            This is the quasi-steady approximation.
          </p>
        </div>

        {/* Step 2 */}
        <div className="bg-slate-50 rounded-lg px-4 py-2.5 border border-slate-200">
          <p className="text-[11px] text-slate-400 font-semibold uppercase tracking-wider mb-0.5">Step 2</p>
          <p className="font-mono text-sm text-slate-800">Q = (1/n)·A·R^(2/3)·S₀^(1/2)</p>
          <p className="text-xs text-slate-500 mt-0.5">
            Manning&apos;s equation applies at every cell, at every instant.
          </p>
        </div>

        {/* Step 3 */}
        <div className="bg-slate-50 rounded-lg px-4 py-2.5 border border-slate-200">
          <p className="text-[11px] text-slate-400 font-semibold uppercase tracking-wider mb-0.5">Step 3</p>
          <p className="font-mono text-sm text-slate-800">R ≈ h = A/B</p>
          <p className="text-xs text-slate-500 mt-0.5">
            Wide rectangular channel: width &gt;&gt; depth, so hydraulic radius ≈ depth.
          </p>
        </div>

        {/* Step 4 */}
        <div className="bg-green-50 rounded-lg px-4 py-2.5 border border-green-300">
          <p className="text-[11px] text-green-700 font-semibold uppercase tracking-wider mb-0.5">Step 4 — Result</p>
          <p className="font-mono text-base font-bold text-green-900">Q = α · A^β</p>
          <p className="text-xs text-green-700 mt-1">
            β = 5/3 &gt; 1 always → wave celerity c = dQ/dA = (5/3)·u &gt; u → wave outruns water.
          </p>
        </div>

        {/* α / β display */}
        <div className="grid grid-cols-2 gap-2">
          <div className="rounded-lg border border-blue-300 bg-blue-50 px-3 py-2 text-center">
            <p className="text-[10px] text-blue-600 font-medium">α (n={DEFAULT_N}, S₀={DEFAULT_S0}, B={DEFAULT_B}m)</p>
            <p className="font-mono text-lg font-bold text-blue-900">{ALPHA.toFixed(4)}</p>
          </div>
          <div className="rounded-lg border border-purple-300 bg-purple-50 px-3 py-2 text-center">
            <p className="text-[10px] text-purple-600 font-medium">β (universal)</p>
            <p className="font-mono text-lg font-bold text-purple-900">{BETA.toFixed(3)}</p>
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── Simplification table ─────────────────────────────────────────────────────

function SimplificationTable() {
  return (
    <div>
      <h5 className="font-semibold text-slate-700 text-sm mb-2 tracking-tight">Simplification Hierarchy</h5>
      <div className="overflow-x-auto">
        <table className="w-full text-xs border-collapse">
          <thead>
            <tr className="bg-slate-100 text-slate-600">
              <th className="border border-slate-200 px-2 py-1.5 text-left font-semibold">Drop</th>
              <th className="border border-slate-200 px-2 py-1.5 text-left font-semibold">Keep</th>
              <th className="border border-slate-200 px-2 py-1.5 text-left font-semibold">Name</th>
              <th className="border border-slate-200 px-2 py-1.5 text-left font-semibold">Notes</th>
            </tr>
          </thead>
          <tbody>
            <tr className="bg-white">
              <td className="border border-slate-200 px-2 py-1.5 font-mono text-slate-500">—</td>
              <td className="border border-slate-200 px-2 py-1.5 font-mono">①②③④⑤</td>
              <td className="border border-slate-200 px-2 py-1.5 font-medium text-slate-700">Dynamic wave</td>
              <td className="border border-slate-200 px-2 py-1.5 text-slate-500">Full SVE; tides, dam-break</td>
            </tr>
            <tr className="bg-slate-50">
              <td className="border border-slate-200 px-2 py-1.5 font-mono text-slate-500">①②</td>
              <td className="border border-slate-200 px-2 py-1.5 font-mono">③④⑤</td>
              <td className="border border-slate-200 px-2 py-1.5 font-medium text-slate-700">Diffusive wave</td>
              <td className="border border-slate-200 px-2 py-1.5 text-slate-500">Backwater; flat rivers</td>
            </tr>
            <tr className="bg-green-50">
              <td className="border border-slate-200 px-2 py-1.5 font-mono text-slate-500">①②③</td>
              <td className="border border-slate-200 px-2 py-1.5 font-mono">④⑤</td>
              <td className="border border-slate-200 px-2 py-1.5 font-bold text-green-800">Kinematic wave</td>
              <td className="border border-slate-200 px-2 py-1.5 text-green-700">S_f = S₀; used in OPM</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ─── Momentum section ─────────────────────────────────────────────────────────

function MomentumSection() {
  const [selectedTerm, setSelectedTerm] = useState<number | null>(null);
  const [dropped, setDropped]           = useState<Set<number>>(new Set());

  const isKinematic = dropped.has(0) && dropped.has(1) && dropped.has(2);

  const handleSelect = (i: number) => {
    setSelectedTerm(prev => (prev === i ? null : i));
  };

  const toggle = (i: number) => {
    setDropped(prev => {
      const s = new Set(prev);
      if (s.has(i)) s.delete(i); else s.add(i);
      return s;
    });
  };

  const jumpToKinematic = () => {
    setDropped(prev => {
      const s = new Set(prev);
      s.add(0); s.add(1); s.add(2);
      return s;
    });
  };

  const reset = () => {
    setDropped(new Set());
    setSelectedTerm(null);
  };

  return (
    <section>
      <h4 className="font-bold text-slate-800 text-base mb-1">
        ② Momentum Equation — Newton&apos;s 2nd Law for Water
      </h4>

      {/* Derivation intro */}
      <p className="text-sm text-slate-600 mb-3 leading-relaxed">
        Apply Newton&apos;s 2nd law (F = ma) to a water element of length Δx. Forces acting: hydrostatic
        pressure difference between faces, gravity down the slope, friction opposing flow. After
        expanding, dividing by ρ·A·Δx, and rearranging:
      </p>

      {/* Equation display */}
      <div className="overflow-x-auto mb-4">
        <div className="bg-slate-50 border border-slate-200 rounded-lg px-4 py-3 font-mono text-sm text-slate-800 whitespace-nowrap">
          ① ∂Q/∂t &nbsp;+&nbsp; ② ∂(Q²/A)/∂x &nbsp;+&nbsp; ③ gA·∂h/∂x &nbsp;+&nbsp; gA·(④ S_f &nbsp;−&nbsp; ⑤ S₀) &nbsp;=&nbsp; 0
        </div>
      </div>

      {/* Two-column layout */}
      <div className="flex flex-col lg:flex-row gap-6 items-start">
        {/* LEFT: Term boxes */}
        <div className="flex flex-col gap-4 shrink-0">
          {TERMS.map((term, i) => (
            <React.Fragment key={term.id}>
              <TermBox
                term={term}
                isSelected={selectedTerm === i}
                isDropped={dropped.has(i)}
                onSelect={() => handleSelect(i)}
                onToggleDrop={() => toggle(i)}
              />
              {i < TERMS.length - 1 && (
                <span className="text-slate-400 font-mono text-sm select-none pl-1">
                  {i === 3 ? '−' : '+'}
                </span>
              )}
            </React.Fragment>
          ))}
          <span className="text-slate-600 font-mono text-sm select-none pl-1">= 0</span>

          {/* Banner */}
          <SimplificationBanner dropped={dropped} />

          {/* Action buttons */}
          <div className="flex flex-wrap gap-2">
            <button
              onClick={reset}
              className="text-xs px-3 py-1.5 rounded-md border border-slate-300 bg-white text-slate-600 hover:bg-slate-50 transition-colors font-medium"
            >
              Reset
            </button>
            <button
              onClick={jumpToKinematic}
              className="text-xs px-3 py-1.5 rounded-md border border-green-300 bg-green-50 text-green-700 hover:bg-green-100 transition-colors font-medium"
            >
              Jump to Kinematic
            </button>
          </div>
        </div>

        {/* RIGHT: Explanation panel */}
        <div className="flex-1 min-w-0 lg:min-w-[300px]">
          <ExplanationPanel selectedTerm={selectedTerm} />
        </div>
      </div>

      {/* Kinematic derivation (below the two-col layout) */}
      <div className="mt-6">
        <h5 className="font-semibold text-slate-700 text-sm mb-3 tracking-tight">
          Kinematic Wave Derivation (from S_f = S₀)
        </h5>
        <KinematicDerivation isKinematic={isKinematic} />
      </div>

      {/* Simplification table */}
      <div className="mt-6">
        <SimplificationTable />
      </div>
    </section>
  );
}

// ─── Root widget ──────────────────────────────────────────────────────────────

export default function SaintVenantWidget() {
  return (
    <div className="not-prose my-8 rounded-2xl border border-slate-200 bg-white shadow-xl overflow-hidden font-sans">
      {/* Header */}
      <div className="bg-gradient-to-r from-sky-700 to-blue-900 px-6 py-4">
        <h3 className="text-white font-bold text-lg tracking-tight">The Saint-Venant Equations</h3>
        <p className="text-sky-200 text-sm mt-0.5">
          Click any term to understand it — then decide what to keep or drop
        </p>
      </div>

      {/* Body */}
      <div className="p-6 space-y-8">
        <ContinuitySection />
        <hr className="border-slate-200" />
        <MomentumSection />
      </div>
    </div>
  );
}
