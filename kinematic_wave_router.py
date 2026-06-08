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
import routing_utils as ru          # default CPU module; overridden per-call in GPU mode
import precip_input as pi
import runoff_input as ri
import gpu_utils


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

    # ── Backend selection ─────────────────────────────────────────────────────
    # Select routing/precip/runoff modules before any computation so the
    # vectorized GPU variants of compute_slope_grid and build_downstream_map
    # are used when BACKEND='gpu'.
    _backend = getattr(cfg, 'BACKEND', 'cpu').lower()
    _use_gpu  = (_backend == 'gpu') and gpu_utils.cupy_available()

    if _use_gpu:
        import cupy as cp
        import routing_utils_gpu as _ru
        from precip_input_gpu import PrecipEngineGPU  as _PrecipEngine
        from runoff_input_gpu  import RunoffEngineGPU  as _RunoffEngine
        xp = cp
        print("  Backend: GPU (CuPy)")
    else:
        _ru = ru   # module-level routing_utils
        from precip_input import PrecipEngine as _PrecipEngine
        from runoff_input  import RunoffEngine as _RunoffEngine
        xp = np
        if _backend == 'gpu':
            print("  [WARNING] BACKEND='gpu' requested but CuPy is unavailable "
                  "or no CUDA device found — falling back to CPU.")
        else:
            print("  Backend: CPU (NumPy)")

    # --- Load rasters ---
    dem, fdir, faccum, ws_mask, transform, nodata_dem, cell_size = _ru.load_rasters(cfg)
    nrows, ncols = dem.shape

    # --- Slope grid ---
    print("  Computing slope grid...")
    slope_2d = _ru.compute_slope_grid(
        dem, fdir, ws_mask, cell_size, cfg.MIN_SLOPE, nodata_dem
    )

    # --- Topological order ---
    print("  Building topological order...")
    s_rows, s_cols, outlet_rc = _ru.topological_order(faccum, fdir, ws_mask)
    n_cells = len(s_rows)
    print(f"  Total active cells: {n_cells:,}")

    # --- Downstream neighbour map ---
    print("  Building downstream neighbour map...")
    ds_idx = _ru.build_downstream_map(
        s_rows, s_cols, fdir, ws_mask, nrows, ncols
    )

    # --- Extract 1-D arrays in topological order (fast indexing) ---
    slope_1d  = slope_2d[s_rows, s_cols]       # slope at each active cell [m/m]
    faccum_1d = faccum[s_rows, s_cols]         # flow accumulation [cell count] for VSA/OPM
    cell_area = cell_size ** 2                  # [m²]  (same for every cell)

    # --- Index of outlet in the sorted list ---
    outlet_pos = n_cells - 1  # last element (highest accumulation = downstream-most)

    print("  Initialisation complete.\n")

    grid_data = {
        "dem"        : dem,
        "fdir"       : fdir,
        "ws_mask"    : ws_mask,
        "s_rows"     : s_rows,        # 1-D topologically sorted row indices
        "s_cols"     : s_cols,        # 1-D topologically sorted col indices
        "slope_1d"   : slope_1d,      # [m/m]
        "faccum_1d"  : faccum_1d,     # [cell count] flow accumulation in topo order
        "ds_idx"     : ds_idx,        # downstream position index (-1 = outlet/off-mask)
        "n_cells"    : n_cells,
        "nrows"      : nrows,
        "ncols"      : ncols,
        "cell_size"  : cell_size,
        "cell_area"  : cell_area,
        "outlet_pos" : outlet_pos,
        "outlet_rc"  : outlet_rc,
        "transform"  : transform,
    }

    # ── Resolve spatially variable Manning's n ──────────────────────────────
    print("  Resolving Manning's n...")
    n_1d = ru.resolve_mannings_n(cfg, grid_data)
    grid_data["n_1d"] = n_1d

    # ── Transfer hot arrays to GPU (must happen BEFORE engine construction) ───
    # Engines read grid_data['faccum_1d'] and grid_data['slope_1d'] during
    # __init__; they must already be CuPy arrays so engine state is on GPU.
    if _use_gpu:
        _dtype    = gpu_utils.get_dtype(cfg)
        slope_1d  = gpu_utils.to_device(slope_1d.astype(_dtype),  xp)
        ds_idx    = gpu_utils.to_device(ds_idx,                    xp)
        faccum_1d = gpu_utils.to_device(faccum_1d.astype(_dtype),  xp)
        n_1d      = gpu_utils.to_device(n_1d.astype(_dtype),       xp)
        grid_data["slope_1d"]  = slope_1d
        grid_data["ds_idx"]    = ds_idx
        grid_data["faccum_1d"] = faccum_1d
        grid_data["n_1d"]      = n_1d

    grid_data["xp"] = xp   # carried into run_time_loop

    # Build precipitation engine (uses grid_data for spatial weight construction)
    print("  Building precipitation engine...")
    grid_data["precip_engine"] = _PrecipEngine(cfg, grid_data)

    # Expose per-cell zone assignment for per-polygon VSA sandbox
    grid_data["cell_polygon"] = grid_data["precip_engine"].cell_polygon

    # Build runoff generation engine (optional; None when RUNOFF_SOURCE='none')
    _rsrc = getattr(cfg, 'RUNOFF_SOURCE', 'none').lower()
    if _rsrc != 'none':
        print("  Building runoff generation engine...")
        grid_data["runoff_engine"] = _RunoffEngine(cfg, grid_data)
    else:
        grid_data["runoff_engine"] = None

    return grid_data


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
    n         = grid_data.get("n_1d", cfg.MANNINGS_N)

    # Output write frequency: every N steps (rounded up to at least 1)
    _out_interval = getattr(cfg, 'OUTPUT_INTERVAL_SECONDS', None)
    write_every   = max(1, round((_out_interval or dt) / dt))
    cell_area = grid_data["cell_area"]
    dx        = grid_data["cell_size"]

    s_rows         = grid_data["s_rows"]
    s_cols         = grid_data["s_cols"]
    slope_1d       = grid_data["slope_1d"]
    ds_idx         = grid_data["ds_idx"]
    n_cells        = grid_data["n_cells"]
    outlet_pos     = grid_data["outlet_pos"]
    ws_mask        = grid_data["ws_mask"]
    precip_engine  = grid_data["precip_engine"]
    runoff_engine  = grid_data.get("runoff_engine")   # None when RUNOFF_SOURCE='none'

    # ── Array module (numpy or cupy) ─────────────────────────────────────────
    xp     = grid_data.get("xp", np)
    _dtype = gpu_utils.get_dtype(cfg)

    # State arrays on the correct device (CPU or GPU)
    volume_1d = xp.zeros(n_cells, dtype=_dtype)   # [m³]  water stored per cell
    Q_out_1d  = xp.zeros(n_cells, dtype=_dtype)   # [m³/s] outflow at previous step
    inflow_1d = xp.zeros(n_cells, dtype=_dtype)   # [m³/s] upstream inflow buffer

    hydrograph = []  # list of (time_seconds, Q_m3s) — Python floats

    print("=" * 60)
    print(f"TIME LOOP  |  steps={n_steps:,}  dt={dt}s  "
          f"sim={cfg.TOTAL_SIMULATION_TIME_HOURS}h")
    print("=" * 60)

    # ── Courant number check ─────────────────────────────────────────────────
    # For a 1 m deep cell at the steepest slope, compute V and the Courant number.
    # If C > 1, the explicit scheme is conditionally unstable and the flux
    # limiter is critical. Print an advisory so the user can reduce dt if needed.
    # Use .item() so the comparison works for both CuPy and NumPy scalars.
    max_slope         = float(slope_1d.max().item())
    n_min             = float(n.min().item()) if hasattr(n, 'min') else float(n)
    V_at_1m_max_slope = (1.0 / n_min) * (1.0 ** (2.0 / 3.0)) * (max_slope ** 0.5)
    C_indicator       = V_at_1m_max_slope * dt / dx
    safe_dt           = dx / V_at_1m_max_slope
    if C_indicator > 1.0:
        print(f"  [CFL WARNING] Courant indicator C={C_indicator:.2f} > 1 at steepest slope.")
        print(f"  Flux limiter is ACTIVE. For unconditionally stable runs, "
              f"set TIME_STEP_SECONDS ≤ {safe_dt:.1f} s")
    else:
        print(f"  [CFL OK] Courant indicator C={C_indicator:.2f} at steepest slope.")
    print()

    # ── Pre-compute static scatter-add masks (ds_idx never changes) ──────────
    valid_ds     = ds_idx >= 0          # mask: cell has a valid downstream neighbour
    ds_positions = ds_idx[valid_ds]     # downstream position indices

    Q_outlet     = 0.0                  # initialise for progress reporting
    t_wall_start = time.time()

    for step in range(n_steps):
        t_seconds = step * dt  # simulation time at the START of this step

        # ── 1. Rainfall array for this step ──────────────────────────────────
        rain_1d = precip_engine.get_field_1d(t_seconds)   # [m/s], (n_cells,)

        # ── 2. Inflow buffer: accumulates Q_out from upstream cells ──────────
        # Reuse the pre-allocated buffer — fill(0) avoids per-step allocation.
        # gpu_utils.scatter_add dispatches to cupyx.scatter_add (GPU) or
        # numpy.add.at (CPU); both are atomic-safe.
        inflow_1d.fill(0)
        gpu_utils.scatter_add(inflow_1d, ds_positions, Q_out_1d[valid_ds])

        # ── 3. Update each cell (vectorised) ─────────────────────────────────
        #
        # Volume balance:
        #   V_new = V_old + rain*cell_area*dt + Q_in*dt - Q_out*dt
        #
        # Because Q_out depends on depth which depends on V_new, we use
        # the EXPLICIT (forward-Euler) scheme:
        #   - Q_out is computed from the CURRENT depth (V_old / cell_area)
        #   - Then V is advanced with that Q_out
        # This is the standard explicit kinematic-wave approach.

        depth_1d = xp.maximum(volume_1d / cell_area, cfg.MIN_DEPTH_M)   # [m]
        # Floor only — no ceiling. A depth cap freezes Manning Q at the cap
        # value while volume keeps growing, creating a permanent flat plateau.
        # The flux limiter already prevents numerical runaway; deeper cells
        # simply get larger Q_out and drain faster (physically correct).

        velocity_1d = ru.mannings_velocity(depth_1d, slope_1d, n)      # [m/s]
        Q_out_1d    = ru.cell_discharge(depth_1d, velocity_1d, dx)     # [m³/s]
        # Apply volume-conservative CFL limiter: a cell cannot eject more
        # water than it stores in one time step (prevents Courant runaway).
        # Inlined with xp.minimum/xp.maximum so it works on both CPU and GPU.
        Q_out_1d = xp.minimum(Q_out_1d, xp.maximum(volume_1d, 0.0) / dt)

        # Volume advance
        # If a RunoffEngine is active, convert rainfall to effective runoff first
        # (forward Euler: query current VSA/mask, then advance sandbox state).
        # With RUNOFF_SOURCE='none', source_1d == rain_1d (bit-identical to old code).
        if runoff_engine is not None:
            source_1d = runoff_engine.get_effective_1d(t_seconds, rain_1d)  # [m/s]
            runoff_engine.update_state(rain_1d, dt)
        else:
            source_1d = rain_1d

        rain_vol   = source_1d * cell_area * dt      # [m³] effective runoff added
        volume_1d  = (volume_1d
                      + rain_vol
                      + inflow_1d * dt
                      - Q_out_1d  * dt)
        volume_1d  = xp.maximum(volume_1d, 0.0)     # no negative storage

        # ── 4. Record outlet hydrograph (at output interval, not every step) ───
        # Evaluate Q_outlet only when needed to minimise GPU→CPU transfers.
        _need_q = (
            (step + 1) % write_every == 0
            or step == n_steps - 1
            or (step + 1) % max(1, n_steps // 10) == 0
        )
        if _need_q:
            # .item() converts CuPy 0-d → Python float (8-byte D→H); no-op on NumPy.
            Q_outlet = float(Q_out_1d[outlet_pos].item())

        if (step + 1) % write_every == 0 or step == n_steps - 1:
            hydrograph.append((t_seconds + dt, Q_outlet))

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
