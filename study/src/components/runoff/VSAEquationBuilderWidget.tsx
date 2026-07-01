'use client';

import React, { useMemo, useState } from 'react';

// ─────────────────────────────────────────────────────────────────────────
// OPM's Variable Source Area (VSA) scheme — Pradhan & Ogden (2010), ported
// verbatim from vsa_opm.py's equation numbering (comments there read
// "Eq 10", "Eq 4", "Eq 12", "Eq 5", "Eq 9" in exactly this order):
//
//   Eq.10  A_t^(0) = A_outlet / (1 - ln(Qmin/Qmax))      — initial threshold
//   Eq.4   H_a = [A_t0/(A_t0-A1)] * ln(SDmin/SDmax0)      — constant, once
//   Eq.12  sandbox water balance (forward Euler, lateral Darcy drainage)
//   Eq.5   A_t(t) = Ha*A1 / (Ha - ln(SDmin/SDmax(t)))     — dynamic threshold
//   Eq.9   cell in VSA  <=>  upslope_area > A_t(t)
//
// Fixed global floors (not sliders): Qmin = 0.001 m^3/s, SDmin = 0.001 m.
// Worked example defaults below are verified against the project's LaTeX
// documentation / vsa_opm.py to the displayed precision.
// ─────────────────────────────────────────────────────────────────────────

const Q_MIN = 0.001; // m^3/s, fixed floor
const SD_MIN = 0.001; // m, fixed floor
const N_CELLS = 10;

function fmt(n: number, d = 2): string {
  return n.toFixed(d);
}

function fmtSci(n: number, d = 2): string {
  if (n === 0) return '0';
  return n.toExponential(d);
}

// ─────────────────────────────────────────────────────────────────────────
// Physics — mirrors vsa_opm.py exactly
// ─────────────────────────────────────────────────────────────────────────

function computeAtInit(aOutlet: number, qMax: number): number {
  return aOutlet / (1 - Math.log(Q_MIN / qMax));
}

function computeHa(atInit: number, a1: number, sdMax0: number): number {
  const ratio = atInit / (atInit - a1);
  return ratio * Math.log(SD_MIN / sdMax0);
}

interface SandboxState {
  z: number; // water-table height above impervious base, m
  sdMax: number; // current root-zone deficit, m
  qb: number; // lateral drainage this step, m^3/s
  dV: number; // net volume change this step, m^3
}

function sandboxStep(
  prev: SandboxState,
  klat: number,
  sDiv: number,
  dx: number,
  a1: number,
  phi: number,
  pDiv: number,
  dt: number,
  sdMax0: number
): SandboxState {
  const qb = klat * sDiv * prev.z * dx; // Eq.12, lateral Darcy drainage
  const dV = (pDiv * a1 - qb) * dt; // net volume in this step
  const zNew = Math.max(0, prev.z + dV / (a1 * phi)); // water-table rise
  const sdMaxNew = Math.max(SD_MIN, sdMax0 - zNew); // remaining deficit
  return { z: zNew, sdMax: sdMaxNew, qb, dV };
}

function computeAtDynamic(ha: number, a1: number, sdMax: number, aOutlet: number): number {
  const denom = ha - Math.log(SD_MIN / sdMax);
  const atRaw = (ha * a1) / denom;
  return Math.min(Math.max(atRaw, a1), aOutlet);
}

// ─────────────────────────────────────────────────────────────────────────
// Small presentational helpers (conventions matched to
// routing/FiniteDiffBuildupWidget.tsx and routing/ManningCelerityWidget.tsx)
// ─────────────────────────────────────────────────────────────────────────

function Badge({ n }: { n: number }) {
  const circled = ['①', '②', '③', '④', '⑤'][n - 1];
  return (
    <span className="inline-flex items-center justify-center w-8 h-8 rounded-full bg-sky-700 text-white font-bold text-base mr-2 shrink-0">
      {circled}
    </span>
  );
}

function StepHeader({ n, eq, title }: { n: number; eq: string; title: string }) {
  return (
    <div className="flex items-center mb-3 flex-wrap gap-2">
      <Badge n={n} />
      <h4 className="text-lg font-bold text-slate-800">{title}</h4>
      <span className="text-xs font-mono bg-slate-100 text-slate-500 border border-slate-200 rounded-full px-2 py-0.5">
        {eq}
      </span>
    </div>
  );
}

interface SliderProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  display: string;
  onChange: (v: number) => void;
}

function LabeledSlider({ label, value, min, max, step, display, onChange }: SliderProps) {
  return (
    <div className="flex flex-col gap-0.5">
      <div className="flex items-center justify-between">
        <label className="text-xs font-semibold text-slate-500">{label}</label>
        <span className="text-xs font-mono text-slate-700 tabular-nums">{display}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full h-1.5 rounded-full accent-sky-600 cursor-pointer"
      />
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────
// Root widget
// ─────────────────────────────────────────────────────────────────────────

export default function VSAEquationBuilderWidget() {
  // Progressive reveal of the 5 numbered equations
  const [visibleSteps, setVisibleSteps] = useState(1);

  // Inputs (all adjustable, defaulted to the verified worked example)
  const [aOutlet] = useState(100000); // m^2, fixed for this toy catchment
  const [a1] = useState(10000); // m^2, one cell's area (dx = 100 m)
  const [qMax, setQMax] = useState(1.0); // m^3/s, current/recent baseflow peak
  const [sdMax0, setSdMax0] = useState(0.1); // m, initial root-zone deficit
  const [phi, setPhi] = useState(0.35); // drainable porosity
  const [klatDayInput, setKlatDayInput] = useState(44); // m/day, slider unit
  const [sDiv, setSDiv] = useState(0.05); // m/m, divide-cell slope
  const [rainMmHr, setRainMmHr] = useState(20); // mm/hr, rain at divide

  // Timestep stepper
  const [timestep, setTimestep] = useState(0); // 0 = pre-storm, n = after n forward-Euler steps

  const dx = Math.sqrt(a1); // = 100 m at defaults
  const dt = 60; // s, fixed forward-Euler step (matches worked example)
  const klat = (klatDayInput / 86400); // m/day -> m/s
  const pDiv = (rainMmHr / 1000) / 3600; // mm/hr -> m/s

  // Step 1 — Eq.10
  const atInit = useMemo(() => computeAtInit(aOutlet, qMax), [aOutlet, qMax]);

  // Step 2 — Eq.4 (computed once from the initial state; never changes again)
  const ha = useMemo(() => computeHa(atInit, a1, sdMax0), [atInit, a1, sdMax0]);

  // Step 3 — Eq.12, march the sandbox forward `timestep` times from z=0
  const MAX_TIMESTEP = 8;
  const sandboxTrace = useMemo(() => {
    const trace: SandboxState[] = [{ z: 0, sdMax: sdMax0, qb: 0, dV: 0 }];
    let cur = trace[0];
    for (let n = 0; n < MAX_TIMESTEP; n++) {
      cur = sandboxStep(cur, klat, sDiv, dx, a1, phi, pDiv, dt, sdMax0);
      trace.push(cur);
    }
    return trace;
  }, [sdMax0, klat, sDiv, dx, a1, phi, pDiv, dt]);

  const current = sandboxTrace[timestep];
  const prevState = sandboxTrace[Math.max(timestep - 1, 0)];

  // Step 4 — Eq.5, dynamic threshold using the CURRENT sdMax
  const atDynamic = useMemo(
    () => computeAtDynamic(ha, a1, current.sdMax, aOutlet),
    [ha, a1, current.sdMax, aOutlet]
  );
  // At timestep 0 (pre-storm), the active threshold is the initial one (Eq.10);
  // Eq.5 takes over once the sandbox has run at least one step.
  const activeAt = timestep === 0 ? atInit : atDynamic;

  // Step 5 — Eq.9, the 10-cell VSA mask. Upslope areas = (i+1)*A1, i=0..9,
  // sorted upstream (i=0, the divide cell) -> downstream (i=9, the outlet).
  const upslopeAreas = useMemo(
    () => Array.from({ length: N_CELLS }, (_, i) => (i + 1) * a1),
    [a1]
  );
  const vsaMask = useMemo(
    () => upslopeAreas.map((up) => up > activeAt),
    [upslopeAreas, activeAt]
  );
  const nInVsa = vsaMask.filter(Boolean).length;
  const pctSaturatedAtStart = useMemo(() => {
    const mask0 = upslopeAreas.map((up) => up > atInit);
    return (mask0.filter(Boolean).length / N_CELLS) * 100;
  }, [upslopeAreas, atInit]);

  const isDefaultSettings =
    qMax === 1.0 && sdMax0 === 0.1 && phi === 0.35 && klatDayInput === 44 && sDiv === 0.05 && rainMmHr === 20;

  return (
    <div className="not-prose my-8 rounded-2xl border border-slate-200 bg-white shadow-xl overflow-hidden font-sans">
      {/* Header */}
      <div className="bg-gradient-to-r from-sky-700 to-emerald-800 px-6 py-4">
        <h3 className="text-white font-bold text-lg tracking-tight">
          Building the VSA Equations, One at a Time
        </h3>
        <p className="text-sky-200 text-sm mt-0.5">
          OPM&apos;s one-parameter saturation scheme — five numbered equations (Pradhan &amp;
          Ogden, 2010), revealed in the order OPM&apos;s code computes them
        </p>
      </div>

      <div className="p-6 space-y-8">
        {/* ── Global inputs (shared across all 5 steps) ───────────────── */}
        <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
          <div className="text-xs font-bold text-slate-500 uppercase mb-3">
            Catchment &amp; storm inputs — adjust to explore beyond the worked example
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <LabeledSlider
              label="Q_max (recent baseflow peak)"
              value={qMax}
              min={0.05}
              max={3}
              step={0.05}
              display={`${fmt(qMax, 2)} m³/s`}
              onChange={setQMax}
            />
            <LabeledSlider
              label="SD_max⁽⁰⁾ (initial deficit)"
              value={sdMax0}
              min={0.02}
              max={0.3}
              step={0.005}
              display={`${fmt(sdMax0, 3)} m`}
              onChange={setSdMax0}
            />
            <LabeledSlider
              label="Rain at divide P_div"
              value={rainMmHr}
              min={1}
              max={60}
              step={1}
              display={`${rainMmHr} mm/hr`}
              onChange={setRainMmHr}
            />
            <LabeledSlider
              label="Drainable porosity φ"
              value={phi}
              min={0.1}
              max={0.5}
              step={0.01}
              display={fmt(phi, 2)}
              onChange={setPhi}
            />
            <LabeledSlider
              label="K_lat (lateral hydraulic conductivity)"
              value={klatDayInput}
              min={5}
              max={150}
              step={1}
              display={`${klatDayInput} m/day`}
              onChange={setKlatDayInput}
            />
            <LabeledSlider
              label="S_div (divide-cell slope)"
              value={sDiv}
              min={0.005}
              max={0.3}
              step={0.005}
              display={`${fmt(sDiv, 3)} m/m`}
              onChange={setSDiv}
            />
          </div>
          <div className="flex flex-wrap gap-2 mt-3 text-xs font-mono text-slate-500">
            <span className="bg-white border border-slate-200 rounded-full px-2 py-0.5">
              A_outlet = {aOutlet.toLocaleString()} m²
            </span>
            <span className="bg-white border border-slate-200 rounded-full px-2 py-0.5">
              A₁ = {a1.toLocaleString()} m² (Δx = {dx} m)
            </span>
            <span className="bg-white border border-slate-200 rounded-full px-2 py-0.5">
              Q_min = {Q_MIN} m³/s (fixed)
            </span>
            <span className="bg-white border border-slate-200 rounded-full px-2 py-0.5">
              SD_min = {SD_MIN} m (fixed)
            </span>
            <span className="bg-white border border-slate-200 rounded-full px-2 py-0.5">
              Δt = {dt} s
            </span>
          </div>
          {!isDefaultSettings && (
            <button
              onClick={() => {
                setQMax(1.0);
                setSdMax0(0.1);
                setPhi(0.35);
                setKlatDayInput(44);
                setSDiv(0.05);
                setRainMmHr(20);
                setTimestep(0);
              }}
              className="mt-3 text-xs font-semibold px-3 py-1.5 rounded-full bg-slate-200 text-slate-700 hover:bg-slate-300 transition"
            >
              ⟲ Reset to worked-example defaults
            </button>
          )}
        </div>

        {/* ── Step 1 — Eq.10 ────────────────────────────────────────── */}
        {visibleSteps >= 1 && (
          <div>
            <StepHeader n={1} eq="Eq. 10" title="The Initial Threshold Area" />
            <p className="text-sm text-slate-700 mb-3">
              Before the storm starts, OPM calibrates a single number — the threshold upslope
              area <span className="font-mono">A_t⁽⁰⁾</span> — from one pre-storm measurement:
              the recent baseflow peak <span className="font-mono">Q_max</span>. Any cell whose
              upslope area already exceeds this threshold is treated as saturated before a drop
              of rain falls.
            </p>
            <div className="bg-white border border-slate-200 rounded-lg p-3 font-mono text-sm text-center overflow-x-auto mb-3">
              A_t⁽⁰⁾ = A_outlet / (1 − ln(Q_min / Q_max))
            </div>
            <div className="bg-sky-50 border border-sky-200 rounded-lg p-3 font-mono text-xs sm:text-sm overflow-x-auto">
              <div>
                A_t⁽⁰⁾ = {aOutlet.toLocaleString()} / (1 − ln({Q_MIN} / {fmt(qMax, 2)}))
              </div>
              <div>
                &nbsp;&nbsp;&nbsp;&nbsp;= {aOutlet.toLocaleString()} / (1 − (
                {fmt(Math.log(Q_MIN / qMax), 3)}))
              </div>
              <div>
                &nbsp;&nbsp;&nbsp;&nbsp;= {aOutlet.toLocaleString()} / {fmt(1 - Math.log(Q_MIN / qMax), 3)}
              </div>
              <div className="font-bold text-sky-800 mt-1">
                &nbsp;&nbsp;&nbsp;&nbsp;= {fmt(atInit, 0)} m²
              </div>
            </div>
            <div className="bg-amber-50 border border-amber-200 rounded-lg p-3 text-sm text-amber-900 mt-3">
              <span className="font-semibold">Physical intuition: </span>
              a larger <span className="font-mono">Q_max</span> (wetter catchment right now)
              makes <span className="font-mono">ln(Q_min/Q_max)</span> more negative, which
              makes the denominator bigger, which makes{' '}
              <span className="font-mono">A_t⁽⁰⁾</span> <strong>smaller</strong> — so{' '}
              <strong>more</strong> cells already qualify as saturated. Wet catchment → big
              initial VSA.
            </div>
          </div>
        )}

        {/* ── Step 2 — Eq.4 ─────────────────────────────────────────── */}
        {visibleSteps >= 2 && (
          <div>
            <StepHeader n={2} eq="Eq. 4" title="The Constant H_a" />
            <p className="text-sm text-slate-700 mb-3">
              Next, OPM computes a scaling constant <span className="font-mono">H_a</span> from
              the initial state. This number is computed <strong>once</strong>, at the start of
              the storm, and never changes again — it calibrates how strongly the threshold area
              will respond as the soil wets up.
            </p>
            <div className="bg-white border border-slate-200 rounded-lg p-3 font-mono text-sm text-center overflow-x-auto mb-3">
              H_a = [A_t⁽⁰⁾ / (A_t⁽⁰⁾ − A₁)] · ln(SD_min / SD_max⁽⁰⁾)
            </div>
            <div className="bg-sky-50 border border-sky-200 rounded-lg p-3 font-mono text-xs sm:text-sm overflow-x-auto">
              <div>
                A_t⁽⁰⁾/(A_t⁽⁰⁾−A₁) = {fmt(atInit, 0)} / {fmt(atInit - a1, 0)} ={' '}
                {fmt(atInit / (atInit - a1), 3)}
              </div>
              <div>
                ln(SD_min/SD_max⁽⁰⁾) = ln({SD_MIN}/{fmt(sdMax0, 3)}) ={' '}
                {fmt(Math.log(SD_MIN / sdMax0), 3)}
              </div>
              <div className="font-bold text-sky-800 mt-1">
                H_a = {fmt(atInit / (atInit - a1), 3)} × ({fmt(Math.log(SD_MIN / sdMax0), 3)}) ={' '}
                {fmt(ha, 2)}
              </div>
            </div>
            <div className="bg-amber-50 border border-amber-200 rounded-lg p-3 text-sm text-amber-900 mt-3">
              <span className="font-semibold">Why H_a is always negative: </span>
              since <span className="font-mono">SD_min/SD_max⁽⁰⁾ &lt; 1</span> the log term is
              negative, and the leading ratio{' '}
              <span className="font-mono">A_t⁽⁰⁾/(A_t⁽⁰⁾−A₁)</span> exceeds 1 — a negative number
              times something &gt; 1 stays negative. So{' '}
              <span className="font-mono">H_a &lt; 0</span> always.
            </div>
          </div>
        )}

        {/* ── Step 3 — Eq.12 ────────────────────────────────────────── */}
        {visibleSteps >= 3 && (
          <div>
            <StepHeader n={3} eq="Eq. 12" title="The Sandbox Water Balance" />
            <p className="text-sm text-slate-700 mb-3">
              As the storm proceeds, a tiny &ldquo;sandbox&rdquo; at the watershed divide tracks
              one number — <span className="font-mono">z</span>, the height of a water table
              above an impervious base — using exactly the same forward-Euler step used
              elsewhere in this course for explicit routing.
            </p>
            <div className="bg-white border border-slate-200 rounded-lg p-3 font-mono text-xs sm:text-sm text-center overflow-x-auto mb-3 space-y-1">
              <div>q_b^n = K_lat · S_div · z^n · Δx &nbsp;(lateral Darcy drainage, m³/s)</div>
              <div>ΔV = (P_div · A₁ − q_b^n) · Δt &nbsp;(net volume this step, m³)</div>
              <div>z^(n+1) = max(0, z^n + ΔV/(A₁·φ)) &nbsp;(water-table rise)</div>
              <div>SD_max^(n+1) = max(SD_min, SD_max⁽⁰⁾ − z^(n+1)) &nbsp;(deficit shrinks)</div>
            </div>

            <div className="flex flex-wrap items-center gap-2 mb-3">
              <button
                onClick={() => setTimestep(0)}
                className="text-xs font-semibold px-3 py-1.5 rounded-full bg-slate-100 text-slate-500 hover:bg-slate-200 transition"
              >
                ⏮ Reset (t=0)
              </button>
              <button
                onClick={() => setTimestep((t) => Math.max(t - 1, 0))}
                disabled={timestep <= 0}
                className="text-xs font-semibold px-3 py-1.5 rounded-full bg-slate-200 text-slate-700 hover:bg-slate-300 disabled:opacity-40 disabled:cursor-not-allowed transition"
              >
                ◀ Step back
              </button>
              <button
                onClick={() => setTimestep((t) => Math.min(t + 1, MAX_TIMESTEP))}
                disabled={timestep >= MAX_TIMESTEP}
                className="text-xs font-semibold px-3 py-1.5 rounded-full bg-sky-600 text-white hover:bg-sky-700 disabled:opacity-40 disabled:cursor-not-allowed transition"
              >
                Step → (advance one Δt)
              </button>
              <span className="text-xs text-slate-500 font-mono">
                timestep n = {timestep} (t = {timestep * dt}s)
              </span>
            </div>

            {timestep === 0 ? (
              <div className="bg-slate-50 border border-slate-200 rounded-lg p-3 text-sm text-slate-600">
                Pre-storm: z⁽⁰⁾ = 0, SD_max⁽⁰⁾ = {fmt(sdMax0, 3)} m. Press &ldquo;Step
                →&rdquo; to run the first forward-Euler update.
              </div>
            ) : (
              <div className="bg-sky-50 border border-sky-200 rounded-lg p-3 font-mono text-xs sm:text-sm overflow-x-auto space-y-1">
                <div>
                  q_b = K_lat·S_div·z^{timestep - 1}·Δx = {fmtSci(prevState.qb, 3)} m³/s
                  {prevState.z === 0 && (
                    <span className="text-slate-500"> (z=0 so far → no wedge yet)</span>
                  )}
                </div>
                <div>
                  ΔV = (P_div·A₁ − q_b)·Δt = {fmt(current.dV, 4)} m³
                </div>
                <div>
                  Δz = ΔV/(A₁·φ) = {fmtSci(current.dV / (a1 * phi), 3)} m ⇒ z^{timestep} ={' '}
                  {fmtSci(current.z, 3)} m ({fmt(current.z * 1000, 3)} mm)
                </div>
                <div className="font-bold text-sky-800">
                  SD_max^{timestep} = {fmt(sdMax0, 3)} − {fmt(current.z, 5)} ={' '}
                  {fmt(current.sdMax, 5)} m ({fmt(current.sdMax * 100, 3)} cm)
                </div>
              </div>
            )}
          </div>
        )}

        {/* ── Step 4 — Eq.5 ─────────────────────────────────────────── */}
        {visibleSteps >= 4 && (
          <div>
            <StepHeader n={4} eq="Eq. 5" title="The Dynamic Threshold Area" />
            <p className="text-sm text-slate-700 mb-3">
              Using the constant <span className="font-mono">H_a</span> from Step 2 and
              whatever <span className="font-mono">SD_max(t)</span> the sandbox just produced in
              Step 3, OPM recomputes the threshold area <strong>every timestep</strong>.
            </p>
            <div className="bg-white border border-slate-200 rounded-lg p-3 font-mono text-sm text-center overflow-x-auto mb-3">
              A_t(t) = H_a·A₁ / (H_a − ln(SD_min/SD_max(t))) &nbsp;,&nbsp; clipped to [A₁,
              A_outlet]
            </div>
            <div className="bg-sky-50 border border-sky-200 rounded-lg p-3 font-mono text-xs sm:text-sm overflow-x-auto">
              <div>
                R_f = SD_min/SD_max(t) = {SD_MIN}/{fmt(current.sdMax, 5)} ={' '}
                {fmt(SD_MIN / current.sdMax, 5)}
              </div>
              <div>
                denom = H_a − ln(R_f) = {fmt(ha, 2)} − ({fmt(Math.log(SD_MIN / current.sdMax), 3)})
                = {fmt(ha - Math.log(SD_MIN / current.sdMax), 2)}
              </div>
              <div className="font-bold text-sky-800 mt-1">
                A_t(t) = ({fmt(ha, 2)} × {a1.toLocaleString()}) /{' '}
                {fmt(ha - Math.log(SD_MIN / current.sdMax), 2)} = {fmt(atDynamic, 0)} m²
              </div>
            </div>
            <div className="bg-amber-50 border border-amber-200 rounded-lg p-3 text-sm text-amber-900 mt-3">
              <span className="font-semibold">As the storm soaks in: </span>
              <span className="font-mono">SD_max(t)</span> falls → <span className="font-mono">A_t(t)</span>{' '}
              decreases → more cells clear the bar → <strong>the VSA expands</strong>.
              {timestep > 0 && (
                <span>
                  {' '}
                  Here A_t fell from {fmt(atInit, 0)} m² (Eq.10, pre-storm) to {fmt(atDynamic, 0)} m²
                  after {timestep} step{timestep === 1 ? '' : 's'}.
                </span>
              )}
            </div>
          </div>
        )}

        {/* ── Step 5 — Eq.9 ─────────────────────────────────────────── */}
        {visibleSteps >= 5 && (
          <div>
            <StepHeader n={5} eq="Eq. 9" title="The VSA Mask — Who's In, Who's Out" />
            <p className="text-sm text-slate-700 mb-3">
              The final step is a simple comparison, rebuilt every timestep: any cell whose
              upslope area exceeds the current threshold is part of the variable source area.
            </p>
            <div className="bg-white border border-slate-200 rounded-lg p-3 font-mono text-sm text-center overflow-x-auto mb-4">
              cell i ∈ VSA &nbsp;⟺&nbsp; upslope_area(i) &gt; A_t(t)
            </div>

            <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
              <div className="text-xs font-bold text-slate-500 uppercase mb-2">
                10-cell toy catchment — i=0 (divide, most upstream) → i=9 (outlet, most
                downstream)
              </div>
              <div className="flex gap-1.5 mb-2 flex-wrap">
                {upslopeAreas.map((up, i) => {
                  const inVsa = vsaMask[i];
                  return (
                    <div
                      key={i}
                      className={`flex flex-col items-center justify-center w-16 h-16 rounded-lg border-2 text-center transition-colors ${
                        inVsa
                          ? 'bg-sky-500 border-sky-700 text-white'
                          : 'bg-amber-50 border-amber-300 text-amber-800'
                      }`}
                      title={`upslope area = ${up.toLocaleString()} m²`}
                    >
                      <span className="text-[10px] font-mono opacity-80">i={i}</span>
                      <span className="text-[10px] font-mono">{(up / 1000).toFixed(0)}k m²</span>
                      <span className="text-xs font-bold">{inVsa ? 'wet' : 'dry'}</span>
                    </div>
                  );
                })}
              </div>
              <div className="text-sm text-slate-700">
                At A_t(t) = <span className="font-mono font-semibold">{fmt(activeAt, 0)} m²</span>
                : <strong className="text-sky-700">{nInVsa} of {N_CELLS}</strong> cells are
                saturated (timestep {timestep}).
              </div>
            </div>

            {isDefaultSettings && (
              <div className="bg-amber-50 border border-amber-200 rounded-lg p-3 text-sm text-amber-900 mt-3">
                At this baseflow the catchment starts already {fmt(pctSaturatedAtStart, 0)}%
                saturated (only the most-upstream divide cell, with upslope area = 10,000 m²,
                is excluded at t=0). Try lowering Q_max above to see the VSA actually grow from
                a smaller seed.
              </div>
            )}
          </div>
        )}

        {/* ── Reveal controls ──────────────────────────────────────── */}
        <div className="flex gap-2 pt-2 border-t border-slate-200">
          <button
            onClick={() => setVisibleSteps((s) => Math.min(s + 1, 5))}
            disabled={visibleSteps >= 5}
            className="text-xs font-semibold px-3 py-1.5 rounded-full bg-sky-600 text-white hover:bg-sky-700 disabled:opacity-40 disabled:cursor-not-allowed transition"
          >
            Reveal next equation ▶
          </button>
          <button
            onClick={() => setVisibleSteps(5)}
            className="text-xs font-semibold px-3 py-1.5 rounded-full bg-slate-200 text-slate-700 hover:bg-slate-300 transition"
          >
            Show all
          </button>
          <button
            onClick={() => {
              setVisibleSteps(1);
              setTimestep(0);
            }}
            className="text-xs font-semibold px-3 py-1.5 rounded-full bg-slate-100 text-slate-500 hover:bg-slate-200 transition"
          >
            Reset
          </button>
        </div>
      </div>
    </div>
  );
}
