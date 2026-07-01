# PROJECT REQUIREMENTS — OPM Interactive Hydrology Course

Derived from user prompts. Follow these rules exactly when generating new modules.

---

## SYSTEM ROLE
Expert Educational Platform Developer building a **static, open-source interactive textbook**
deployed to GitHub Pages. No backend. No Python-based compute. All simulations run in the browser.

---

## TECH STACK (non-negotiable)
| Layer | Choice |
|-------|--------|
| Framework | **Next.js 14** (App Router) with `@next/mdx` |
| Content | `.mdx` files — Markdown + LaTeX via **KaTeX** (`remark-math` + `rehype-katex` + `remark-gfm`) |
| Components | **React (TSX)**, self-contained with `'use client'` |
| Visualisation | **HTML5 Canvas** for 3-D and animation; **SVG** for 2-D grids and arrows |
| Styling | **Tailwind CSS** + `@tailwindcss/typography` |
| Deploy | `output: 'export'` → static GitHub Pages |
| No D3 needed | Pure React state + Canvas/SVG is sufficient and preferred |

---

## AUDIENCE & PEDAGOGY

Dual audience — **both must be served in every module**:

| Tier | What they need |
|------|---------------|
| Undergrad | Physical analogy (2–3 sentences max), colorful interactive widget, formula in plain English |
| Grad student | Rigorous PDE / algorithm derivation, complexity analysis, failure-mode analysis |

---

## MODULE STRUCTURE (mandatory for every chapter page)

Every chapter must follow this exact sequence:

1. **Hook** — 1–3 sentence real-world analogy. No math yet.
2. **2-D Interactive Widget** — placed immediately, before any prose explanation.
3. **3-D View Widget** — drag-to-rotate Canvas rendering of the same concept.
4. **Core Math** — KaTeX equations, concise.
5. **More Interactive Calculation Widgets** — calculations revealed interactively, not in prose.
6. **Grad-level section** — algorithms, pseudocode, complexity, failure modes.
7. **Summary table** — one-liner per concept.

**Rule: widgets carry the teaching burden. Prose is framing only.**
**Rule: every prose paragraph should be replaceable by a widget. If it isn't, cut it.**

---

## WIDGET DESIGN RULES

### Interaction model
- **Left-click** on a grid cell → raise elevation (+1, max 15)
- **Right-click** on a grid cell → lower elevation (−1, min 1)
- **Drag** on a 3-D canvas → rotate view (horizontal drag = azimuth, vertical = tilt)
- **Hover** on a cell → show tooltip with coordinates and value
- All controls visible without scrolling (compact layout)

### Required tabs / modes for the DEM widget
| Tab | Behaviour |
|-----|-----------|
| 🗺 Terrain | Editable grid, elevation colormap, ± click |
| ➡ Flow Direction | D8 arrows + optional ESRI code labels + pit markers |
| 📐 Slope Calculator | Click any cell → live 8-row table (dir, code, neighbour z, distance, Δz, slope m/m, status) + 3×3 neighbourhood mini-grid. Ties highlighted amber. Winner highlighted green. |
| 💧 Raindrop | Animated flow path, step-by-step, outlet vs pit outcome |

### 3-D Widget requirements
- HTML5 Canvas, painter's-algorithm depth sort (ascending depth = far-first)
- **Correct orthographic projection**:
  - `sx = (wx·cosφ − wz·sinφ) · scale + W/2`
  - `sy = (wx·sinφ·sinθ − wy·cosθ + wz·cosφ·sinθ) · scale + H/2`
  - `depth = wx·sinφ + wz·cosφ`
  - where wx = col − (C−1)/2, wy = elev × hScale, wz = row − (R−1)/2
- Draw south and east walls where a cell is taller than its neighbour
- Hillshading: NW sun, Lambert diffuse on per-cell gradient normal
- Controls: height exaggeration slider, azimuth slider, tilt slider, flow-arrows toggle, hillshading toggle, preset buttons
- Hover shows `(row, col) z = X m` in HUD

### Flow Accumulation Widget
- Animated step-by-step: yellow = current cell, purple = downstream recipient
- FA coloring: log-scale white → sky-blue → dark navy
- Stream threshold slider reveals channel network
- Speed slider + pause + step-forward + skip-to-end controls

### Pit Fill Widget (Wang & Liu 2006)
- Side-by-side Before | After grids
- Animated heap-pop / fill steps with text log
- Low-outlet cell marked in blue
- Back / Forward step buttons (not just autoplay)
- After completion: flat-area warning

### Fail-safe rule
Every widget must catch numerical blow-up (NaN / Infinity) and display
a visible "Model Unstable" warning instead of crashing.

---

## CONTENT RULES

### Text density
- **Maximum 3 sentences of prose per section** before a widget appears.
- No long explanatory paragraphs — the widget IS the explanation.
- Callout boxes inside widgets replace sidebar text.

### Tables
- Use GFM pipe tables (`remark-gfm` is configured). Always use them for structured data.
- Never render tables as prose.

### Math
- All equations in KaTeX via `$...$` (inline) or `$$...$$` (display).
- Always show the formula first, then reference the widget for the numerical example.

### Worked examples
- Numerical worked examples belong INSIDE the widget's "Slope Calculator" or equivalent tab.
- Do NOT write static numeric examples in MDX prose.

---

## COLOUR CONVENTIONS (use consistently across all widgets)

| Meaning | Colour |
|---------|--------|
| Elevation low | `rgb(67,117,180)` — blue |
| Elevation high | `rgb(245,245,244)` — near-white/snow |
| Currently processing cell | Amber / `#fbbf24` |
| Flow path / animated raindrop | Sky blue `#93c5fd` → `#0284c7` (head) |
| Flow-to / downstream | Purple `#a78bfa` |
| Winner direction | Emerald green |
| Tie direction | Amber |
| Uphill direction | Red-tinted |
| Raised (pit-filled) cell | Orange `#fdba74` |
| Low outlet | Blue `#3b82f6` |
| Stream channel | Dark navy (high FA) |

---

## FILE STRUCTURE
```
study/
  src/
    app/
      layout.tsx          ← sticky nav, KaTeX CSS import
      page.tsx            ← chapter index with status badges
      chapter/
        01-dem-flow-direction/page.tsx
    components/
      dem/
        DEMFlowWidget.tsx   ← 2-D grid, 4 tabs
        DEM3DWidget.tsx     ← 3-D canvas, drag-rotate
        FlowAccumWidget.tsx ← FA animation
        PitFillWidget.tsx   ← Wang & Liu animation
    content/
      01-dem-and-flow-direction.mdx
  mdx-components.tsx        ← register all widgets here
  next.config.mjs           ← remark-math, remark-gfm, rehype-katex
  tailwind.config.ts
```

---

## CHAPTER ROADMAP

| # | Title | Status |
|---|-------|--------|
| 1 | Digital Elevation Models & Flow Direction | ✅ Done |
| 2 | Watershed Delineation | Planned |
| 3 | Rainfall–Runoff (VSA + Green-Ampt) | Planned |
| 4 | Kinematic-Wave & Diffusive-Wave Routing | Planned |
| 5 | SERVES/GEE Integration & Dynamic Parameters | Planned |
