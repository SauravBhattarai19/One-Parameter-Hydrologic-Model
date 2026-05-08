"""
kinematic_wave_router.py
========================
Explicit, grid-based Kinematic Wave routing model.

Algorithm (per time step)
-------------------------
1. Build a rainfall 2-D array for the current time.
2. Iterate over ALL active cells in topological (upstream-first) order.
3. For each cell i:
      a. Compute rainfall volume added this step.
      b. Add Q_in arriving from upstream cells (accumulated in inflow buffer).
      c. Compute depth from stored volume.
      d. Compute slope-driven Manning velocity → Q_out.
      e. Solve continuity: new_volume = old_volume + rain_vol + Q_in*dt - Q_out*dt
      f. Pass Q_out to the downstream neighbour's inflow buffer.
4. Record Q at the outlet cell → hydrograph.
5. Write hydrograph to CSV.

All physical parameters and paths come from config.py.
"""

import os
import time
import numpy as np
import pandas as pd

import config
import routing_utils as ru


# ─────────────────────────────────────────────────────────────────────────────
# Grid initialisation
# ─────────────────────────────────────────────────────────────────────────────

def initialise_grid(cfg):
    """
    Load rasters, compute slopes, build topological order and downstream map.

    Returns a dict ('grid_data') with every array the time loop needs.
    """
    print("=" * 60)
    print("KINEMATIC WAVE ROUTER  –  Grid Initialisation")
    print("=" * 60)

    # --- Load rasters ---
    dem, fdir, faccum, ws_mask, transform, nodata_dem = ru.load_rasters(cfg)
    nrows, ncols = dem.shape

    # --- Slope grid ---
    print("  Computing slope grid...")
    slope_2d = ru.compute_slope_grid(
        dem, fdir, ws_mask, cfg.CELL_SIZE, cfg.MIN_SLOPE, nodata_dem
    )

    # --- Topological order ---
    print("  Building topological order...")
    s_rows, s_cols, outlet_rc = ru.topological_order(faccum, fdir, ws_mask)
    n_cells = len(s_rows)
    print(f"  Total active cells: {n_cells:,}")

    # --- Downstream neighbour map ---
    print("  Building downstream neighbour map...")
    ds_idx = ru.build_downstream_map(
        s_rows, s_cols, fdir, ws_mask, nrows, ncols
    )

    # --- Extract 1-D arrays in topological order (fast indexing) ---
    slope_1d  = slope_2d[s_rows, s_cols]       # slope at each active cell [m/m]
    cell_area = cfg.CELL_SIZE ** 2              # [m²]  (same for every cell)

    # --- Index of outlet in the sorted list ---
    outlet_pos = n_cells - 1  # last element (highest accumulation = downstream-most)

    print("  Initialisation complete.\n")

    return {
        "dem"        : dem,
        "fdir"       : fdir,
        "ws_mask"    : ws_mask,
        "s_rows"     : s_rows,        # 1-D topologically sorted row indices
        "s_cols"     : s_cols,        # 1-D topologically sorted col indices
        "slope_1d"   : slope_1d,      # [m/m]
        "ds_idx"     : ds_idx,        # downstream position index (-1 = outlet/off-mask)
        "n_cells"    : n_cells,
        "nrows"      : nrows,
        "ncols"      : ncols,
        "cell_area"  : cell_area,
        "outlet_pos" : outlet_pos,
        "outlet_rc"  : outlet_rc,
        "transform"  : transform,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Time loop
# ─────────────────────────────────────────────────────────────────────────────

def run_time_loop(grid_data, cfg):
    """
    Core explicit kinematic-wave routing loop.

    Continuity per cell per step:
        V_new = V_old + (rain [m/s] * cell_area * dt)   <- rain input
                      + Q_in  * dt                       <- upstream inflow
                      - Q_out * dt                       <- Manning outflow
        V_new = max(V_new, 0)                            <- no negative storage
        depth = V_new / cell_area

    Returns
    -------
    hydrograph : list of (time_s, Q_out_m3s) tuples
    """
    dt        = cfg.TIME_STEP_SECONDS
    n_steps   = int(cfg.TOTAL_SIMULATION_TIME_HOURS * 3600.0 / dt)
    n         = cfg.MANNINGS_N
    cell_area = grid_data["cell_area"]
    dx        = cfg.CELL_SIZE

    s_rows     = grid_data["s_rows"]
    s_cols     = grid_data["s_cols"]
    slope_1d   = grid_data["slope_1d"]
    ds_idx     = grid_data["ds_idx"]
    n_cells    = grid_data["n_cells"]
    outlet_pos = grid_data["outlet_pos"]
    ws_mask    = grid_data["ws_mask"]
    grid_shape = (grid_data["nrows"], grid_data["ncols"])

    # State arrays (1-D over active cells, topological order)
    volume_1d = np.zeros(n_cells, dtype=np.float64)   # [m³]  water stored per cell
    Q_out_1d  = np.zeros(n_cells, dtype=np.float64)   # [m³/s] outflow at previous step

    hydrograph = []  # list of (time_seconds, Q_m3s)

    print("=" * 60)
    print(f"TIME LOOP  |  steps={n_steps:,}  dt={dt}s  "
          f"sim={cfg.TOTAL_SIMULATION_TIME_HOURS}h")
    print("=" * 60)

    # ── Courant number check ─────────────────────────────────────────────────
    # For a 1 m deep cell at the steepest slope, compute V and the Courant number.
    # If C > 1, the explicit scheme is conditionally unstable and the flux
    # limiter is critical. Print an advisory so the user can reduce dt if needed.
    V_at_1m_max_slope = (1.0 / n) * (1.0 ** (2.0 / 3.0)) * (slope_1d.max() ** 0.5)
    C_indicator       = V_at_1m_max_slope * dt / dx
    safe_dt           = dx / V_at_1m_max_slope
    if C_indicator > 1.0:
        print(f"  [CFL WARNING] Courant indicator C={C_indicator:.2f} > 1 at steepest slope.")
        print(f"  Flux limiter is ACTIVE. For unconditionally stable runs, "
              f"set TIME_STEP_SECONDS ≤ {safe_dt:.1f} s")
    else:
        print(f"  [CFL OK] Courant indicator C={C_indicator:.2f} at steepest slope.")
    print()

    t_wall_start = time.time()

    for step in range(n_steps):
        t_seconds = step * dt  # simulation time at the START of this step

        # ── 1. Rainfall array for this step ──────────────────────────────────
        rain_2d = ru.build_rainfall_array(
            grid_shape,
            cfg.RAIN_INTENSITY_MM_HR,
            cfg.RAIN_DURATION_HOURS,
            dt,
            t_seconds
        )
        # Extract rainfall rate [m/s] at each active cell (1-D, topo order)
        rain_1d = rain_2d[s_rows, s_cols]

        # ── 2. Inflow buffer: accumulates Q_out from upstream cells ──────────
        # Reset to zero at the beginning of each step; then each cell adds
        # its Q_out to its downstream neighbour's slot.
        inflow_1d = np.zeros(n_cells, dtype=np.float64)

        # Pass upstream outflow to downstream cells.
        # Because we process cells in topological order (upstream → downstream),
        # by the time we reach cell i the inflow_1d[i] already contains the
        # contributions of ALL upstream cells processed before it.
        # Here we pre-fill using the PREVIOUS step's Q_out values.
        valid_ds     = ds_idx >= 0                    # mask: has a valid downstream
        ds_positions = ds_idx[valid_ds]               # positions of downstream cells
        np.add.at(inflow_1d, ds_positions, Q_out_1d[valid_ds])

        # ── 3. Update each cell (vectorised where possible) ──────────────────
        #
        # Volume balance:
        #   V_new = V_old + rain*cell_area*dt + Q_in*dt - Q_out*dt
        #
        # Because Q_out depends on depth which depends on V_new, we use
        # the EXPLICIT (forward-Euler) scheme:
        #   - Q_out is computed from the CURRENT depth (V_old / cell_area)
        #   - Then V is advanced with that Q_out
        # This is the standard explicit kinematic-wave approach.

        depth_1d = volume_1d / cell_area                         # [m]
        depth_1d = np.clip(depth_1d, cfg.MIN_DEPTH_M, cfg.MAX_DEPTH_M)

        velocity_1d = ru.mannings_velocity(depth_1d, slope_1d, n)      # [m/s]
        Q_out_1d    = ru.cell_discharge(depth_1d, velocity_1d, dx)     # [m³/s]
        # Apply volume-conservative CFL limiter: a cell cannot eject more
        # water than it stores in one time step (prevents Courant runaway).
        Q_out_1d    = ru.flux_limiter(Q_out_1d, volume_1d, dt)         # [m³/s]

        # Volume advance
        rain_vol   = rain_1d * cell_area * dt        # [m³] rainfall added
        volume_1d  = (volume_1d
                      + rain_vol
                      + inflow_1d * dt
                      - Q_out_1d  * dt)
        volume_1d  = np.maximum(volume_1d, 0.0)     # no negative storage

        # ── 4. Record outlet hydrograph ───────────────────────────────────────
        Q_outlet = Q_out_1d[outlet_pos]
        hydrograph.append((t_seconds + dt, Q_outlet))  # record end-of-step time

        # Progress reporting every 10 % of simulation
        if (step + 1) % max(1, n_steps // 10) == 0:
            elapsed = time.time() - t_wall_start
            pct     = 100.0 * (step + 1) / n_steps
            print(f"  {pct:5.1f}%  |  t={t_seconds/3600:.3f}h  "
                  f"|  Q_outlet={Q_outlet:.4f} m³/s  "
                  f"|  wall={elapsed:.1f}s")

    print(f"\n  Simulation finished in {time.time()-t_wall_start:.1f}s")
    return hydrograph


# ─────────────────────────────────────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────────────────────────────────────

def save_hydrograph(hydrograph, cfg):
    """
    Save the hydrograph as a CSV with columns:
        time_s   – simulation time in seconds
        time_hr  – simulation time in hours
        Q_m3s    – discharge at the outlet [m³/s]
    Also prints peak flow and time-to-peak.
    """
    times_s  = np.array([h[0] for h in hydrograph])
    Q_values = np.array([h[1] for h in hydrograph])

    df = pd.DataFrame({
        "time_s"  : times_s,
        "time_hr" : times_s / 3600.0,
        "Q_m3s"   : Q_values,
    })

    df.to_csv(cfg.HYDROGRAPH_CSV, index=False)
    print(f"\n  Hydrograph saved → {cfg.HYDROGRAPH_CSV}")

    peak_Q  = Q_values.max()
    peak_t  = times_s[Q_values.argmax()] / 3600.0
    print(f"  Peak discharge : {peak_Q:.4f} m³/s  at t={peak_t:.2f} h")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # --- Initialise ---
    grid_data  = initialise_grid(config)

    # --- Run ---
    hydrograph = run_time_loop(grid_data, config)

    # --- Save ---
    df = save_hydrograph(hydrograph, config)

    # --- Quick console summary ---
    print("\n" + "=" * 60)
    print("DONE")
    print(f"  CSV rows       : {len(df):,}")
    print(f"  Output file    : {config.HYDROGRAPH_CSV}")
    print("=" * 60)


if __name__ == "__main__":
    main()
