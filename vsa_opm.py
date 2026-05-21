"""
vsa_opm.py
==========
Standalone One-Parameter Model (OPM) for Variable Source Area (VSA) runoff.

Implements Pradhan & Ogden (2010):
  - Eq 10: initial threshold area A_t from single discharge measurement
  - Eq  4: constant scaling factor H_a
  - Eq 12: Sandbox water balance at the catchment divide (Darcy drainage)
  - Eq  5: dynamic threshold area A_t(t)
  - Eq  9: VSA = cells with upslope_area > A_t

Usage
-----
    python vsa_opm.py

Reads:  output/clipped_dem.tif, output/clipped_flow_accumulation.tif,
        output/flow_direction.tif, output/watershed.tif,
        precipitation/gauges.csv, precipitation/timeseries.csv  (via config.py)
Writes: output/vsa_opm_results.csv

All parameters are read from config.py:
    OPM_SD_MAX_INITIAL  [m]      initial max soil moisture deficit
    OPM_Q_MAX           [m³/s]   initial observed outlet discharge
    OPM_PHI             [-]      drainable porosity (default 0.35)
"""

import os
import sys
import time
import math

import numpy as np
import pandas as pd

# ── Ensure project root is on the path so imports work from any directory ────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import routing_utils as ru
from precip_input import PrecipEngine

# ── OPM physical constants ────────────────────────────────────────────────────
SD_MIN = 0.001          # m   minimum saturation deficit
Q_MIN  = 0.001          # m³/s  minimum discharge (Eq 10)
K_MS   = 44.0 / 86400.0  # m/s  saturated hydraulic conductivity (44 m/day)


# ─────────────────────────────────────────────────────────────────────────────
def run_opm(cfg):
    """
    Run the OPM/VSA loop and return a DataFrame of results.

    Parameters
    ----------
    cfg : module  (imported config)

    Returns
    -------
    df : pd.DataFrame  with columns: time_s, SD_max_t, A_t_m2, VSA_m2
    """
    print("=" * 60)
    print("OPM / VSA  —  Variable Source Area Runoff Generation")
    print("    Pradhan & Ogden (2010)")
    print("=" * 60)

    # ── 1. Load rasters (reuses routing_utils) ───────────────────────────────
    dem, fdir, faccum, ws_mask, transform, nodata_dem, cell_size = \
        ru.load_rasters(cfg)
    nrows, ncols = dem.shape
    cell_area    = cell_size ** 2

    # ── 2. Slope grid ────────────────────────────────────────────────────────
    print("  Computing slope grid...")
    slope_2d = ru.compute_slope_grid(
        dem, fdir, ws_mask, cell_size, cfg.MIN_SLOPE, nodata_dem
    )

    # ── 3. Topological order ─────────────────────────────────────────────────
    print("  Building topological order...")
    s_rows, s_cols, outlet_rc = ru.topological_order(faccum, fdir, ws_mask)
    n_cells = len(s_rows)

    # ── 4. Extract 1-D arrays ────────────────────────────────────────────────
    slope_1d  = slope_2d[s_rows, s_cols]
    faccum_1d = faccum[s_rows, s_cols].astype(np.float64)  # cell counts

    # Divide cell: first in topological order (minimum faccum ≈ 1)
    slope_divide = float(slope_1d[0])

    # Upslope contributing area per cell [m²]
    upslope_area_1d = faccum_1d * cell_area

    # ── 5. Build minimal grid_data for PrecipEngine ──────────────────────────
    grid_data = {
        "s_rows"   : s_rows,
        "s_cols"   : s_cols,
        "nrows"    : nrows,
        "ncols"    : ncols,
        "n_cells"  : n_cells,
        "ws_mask"  : ws_mask,
        "transform": transform,
    }
    print("  Building precipitation engine...")
    precip_engine = PrecipEngine(cfg, grid_data)

    # ── 6. OPM Initialisation ─────────────────────────────────────────────────

    SD_max_initial = float(cfg.OPM_SD_MAX_INITIAL)
    Q_max          = float(cfg.OPM_Q_MAX)
    phi            = float(getattr(cfg, 'OPM_PHI', 0.35))

    if Q_max <= Q_MIN:
        raise ValueError(
            f"OPM_Q_MAX={Q_max} m³/s must be > {Q_MIN} m³/s (Q_min constant)."
        )

    # A_1: upslope area of single divide cell [m²]
    A_1 = cell_area

    # A_outlet: total catchment area [m²] (outlet cell has highest faccum)
    A_outlet = float(faccum[outlet_rc]) * cell_area

    # Eq 10 — initial threshold contributing area (single discharge measurement)
    # A_t_init = A_outlet / (1 - ln(Q_min / Q_max))
    A_t_init = A_outlet / (1.0 - math.log(Q_MIN / Q_max))

    # Eq 4 — scaling factor H_a (constant, negative)
    Rf_init = SD_MIN / SD_max_initial
    ratio   = A_t_init / (A_t_init - A_1)
    H_a     = ratio * math.log(Rf_init)   # < 0

    # Verify self-consistency: Eq 5 with Rf_init should recover A_t_init
    _denom_check = H_a - math.log(Rf_init)
    if abs(_denom_check) > 1e-6:
        # For large watersheds (A_t_init >> A_1) the denominator is small but
        # nonzero; A_t computed from Eq 5 equals A_t_init up to floating-point.
        pass   # expected; singularity only at denom=0

    print()
    print(f"  OPM parameters:")
    print(f"    SD_max_initial = {SD_max_initial:.4f} m")
    print(f"    Q_max          = {Q_max:.4f} m³/s")
    print(f"    phi (porosity) = {phi:.3f}")
    print(f"    K (hyd. cond.) = {K_MS*86400:.1f} m/day = {K_MS:.2e} m/s")
    print(f"    A_1            = {A_1:.1f} m²  (single cell)")
    print(f"    A_outlet       = {A_outlet:.3e} m²  (total catchment)")
    print(f"    A_t_init [Eq10]= {A_t_init:.3e} m²")
    print(f"    Rf_init        = {Rf_init:.6f}")
    print(f"    H_a    [Eq 4]  = {H_a:.6f}  (negative)")
    print(f"    slope_divide   = {slope_divide:.6f} m/m")
    print()

    # ── 7. Time loop ──────────────────────────────────────────────────────────
    dt      = float(cfg.TIME_STEP_SECONDS)
    n_steps = int(cfg.TOTAL_SIMULATION_TIME_HOURS * 3600.0 / dt)

    _out_interval = getattr(cfg, 'OUTPUT_INTERVAL_SECONDS', None)
    write_every   = max(1, round((_out_interval or dt) / dt))

    print(f"  Time loop: {n_steps:,} steps × {dt} s  "
          f"(recording every {write_every} steps)\n")

    # State
    z        = 0.0            # saturated zone thickness [m]
    A_t      = A_t_init       # current threshold contributing area [m²]
    SD_max_t = SD_max_initial  # current max deficit [m]

    results     = []
    t_wall_start = time.time()

    for step in range(n_steps):
        t_s = step * dt   # simulation time at start of step

        # ── Get rainfall for this step ─────────────────────────────────────
        rain_1d  = precip_engine.get_field_1d(t_s)    # [m/s], (n_cells,)
        P_divide = float(rain_1d[0])                   # rainfall at divide cell

        # ── Compute VSA from CURRENT A_t (forward Euler) ──────────────────
        vsa_mask = upslope_area_1d > A_t               # (n_cells,) bool
        VSA_m2   = float(vsa_mask.sum()) * cell_area   # [m²]

        # ── Record at output interval ──────────────────────────────────────
        if (step + 1) % write_every == 0 or step == n_steps - 1:
            results.append({
                "time_s"  : t_s + dt,
                "SD_max_t": SD_max_t,
                "A_t_m2"  : A_t,
                "VSA_m2"  : VSA_m2,
            })

        # ── Sandbox water balance (Eq 12) ──────────────────────────────────
        # Darcy lateral outflow from saturated zone at the divide cell:
        #   q_b = K * i * z * b  [m³/s]  where i=slope, b=cell_size
        q_b = K_MS * slope_divide * z * cell_size       # [m³/s]

        # Net volume change; divide by (cell_area × phi) to get dz [m]
        dV  = (P_divide * cell_area - q_b) * dt         # [m³]
        dz  = dV / (cell_area * phi)
        z   = max(0.0, z + dz)                          # z ≥ 0

        # Updated maximum deficit (Eq 12), floored at SD_min
        SD_max_t = max(SD_MIN, SD_max_initial - z)

        # ── Eq 5 — updated threshold area ─────────────────────────────────
        Rf_t  = SD_MIN / SD_max_t
        denom = H_a - math.log(Rf_t)
        if abs(denom) < 1e-12:
            # Singularity at t=0 (Rf_t == Rf_init); keep current value
            pass
        else:
            new_A_t = H_a * A_1 / denom
            A_t     = float(np.clip(new_A_t, A_1, A_outlet))

        # Progress reporting
        if (step + 1) % max(1, n_steps // 10) == 0:
            elapsed = time.time() - t_wall_start
            pct     = 100.0 * (step + 1) / n_steps
            print(f"  {pct:5.1f}%  t={t_s/3600:.3f}h  "
                  f"SD_max={SD_max_t:.5f} m  "
                  f"A_t={A_t:.3e} m²  "
                  f"VSA={VSA_m2/1e6:.2f} km²  "
                  f"wall={elapsed:.1f}s")

    elapsed_total = time.time() - t_wall_start
    print(f"\n  OPM loop finished in {elapsed_total:.1f}s  "
          f"({len(results)} records)\n")

    # ── 8. Write output CSV ───────────────────────────────────────────────────
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(cfg.OUTPUT_DIR, "vsa_opm_results.csv")
    df = pd.DataFrame(results, columns=["time_s", "SD_max_t", "A_t_m2", "VSA_m2"])
    df.to_csv(out_path, index=False, float_format="%.6g")

    # Summary
    print(f"  Results written → {out_path}")
    print(f"  Final state  :  z={z:.4f} m  "
          f"SD_max={SD_max_t:.5f} m  "
          f"A_t={A_t:.3e} m²  "
          f"VSA={VSA_m2/1e6:.3f} km²")
    print(f"  Max VSA: {df['VSA_m2'].max()/1e6:.3f} km²  "
          f"at t={df.loc[df['VSA_m2'].idxmax(), 'time_s']/3600:.2f} h")
    print(f"  Min A_t: {df['A_t_m2'].min():.3e} m²  "
          f"Min SD_max: {df['SD_max_t'].min():.5f} m")
    print()

    return df


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = run_opm(config)
