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

    # Static arrays for the diffusive-wave scheme (water-surface slope needs bed
    # elevation and the true flow-path length; harmless to build even when kinematic).
    dem_1d  = dem[s_rows, s_cols].astype(np.float64)        # bed elevation [m]
    fdir_1d = fdir[s_rows, s_cols]
    dist_1d = cell_size * np.where(                          # flow-path length to downstream [m]
        np.isin(fdir_1d, list(ru.D8_DIAGONAL)), np.sqrt(2.0), 1.0
    ).astype(np.float64)

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
        "dem_1d"     : dem_1d,        # [m] bed elevation in topo order (diffusive wave)
        "dist_1d"    : dist_1d,       # [m] flow-path length to downstream (diffusive wave)
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

    # ── Channel (river) cross-section geometry ──────────────────────────────
    # Wide-sheet defaults (cell_size / cell_area) unless CHANNEL_ROUTING is on, in
    # which case high-faccum channel cells get a confined rectangular section.
    chan_mask_1d, width_1d, store_area_1d = ru.build_channel_geometry(cfg, grid_data)
    grid_data["chan_mask_1d"]  = chan_mask_1d
    grid_data["width_1d"]      = width_1d
    grid_data["store_area_1d"] = store_area_1d

    # ── Transfer hot arrays to GPU (must happen BEFORE engine construction) ───
    # Engines read grid_data['faccum_1d'] and grid_data['slope_1d'] during
    # __init__; they must already be CuPy arrays so engine state is on GPU.
    if _use_gpu:
        _dtype    = gpu_utils.get_dtype(cfg)
        slope_1d  = gpu_utils.to_device(slope_1d.astype(_dtype),  xp)
        ds_idx    = gpu_utils.to_device(ds_idx,                    xp)
        faccum_1d = gpu_utils.to_device(faccum_1d.astype(_dtype),  xp)
        n_1d      = gpu_utils.to_device(n_1d.astype(_dtype),       xp)
        dem_1d    = gpu_utils.to_device(dem_1d.astype(_dtype),     xp)
        dist_1d   = gpu_utils.to_device(dist_1d.astype(_dtype),    xp)
        width_1d      = gpu_utils.to_device(width_1d.astype(_dtype),      xp)
        store_area_1d = gpu_utils.to_device(store_area_1d.astype(_dtype), xp)
        chan_mask_1d  = gpu_utils.to_device(chan_mask_1d,                 xp)
        grid_data["slope_1d"]  = slope_1d
        grid_data["ds_idx"]    = ds_idx
        grid_data["faccum_1d"] = faccum_1d
        grid_data["n_1d"]      = n_1d
        grid_data["dem_1d"]    = dem_1d
        grid_data["dist_1d"]   = dist_1d
        grid_data["width_1d"]      = width_1d
        grid_data["store_area_1d"] = store_area_1d
        grid_data["chan_mask_1d"]  = chan_mask_1d

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
    n         = grid_data.get("n_1d", cfg.MANNINGS_N)
    cell_area = grid_data["cell_area"]
    dx        = grid_data["cell_size"]
    T         = cfg.TOTAL_SIMULATION_TIME_HOURS * 3600.0

    # Adaptive CFL parameters
    adaptive      = getattr(cfg, 'ADAPTIVE_TIMESTEP', True)
    cfl_target    = float(getattr(cfg, 'CFL_TARGET', 0.8))
    _cfl_dt_max   = getattr(cfg, 'CFL_DT_MAX', None)
    _out_interval = float(getattr(cfg, 'OUTPUT_INTERVAL_SECONDS', None)
                          or cfg.TIME_STEP_SECONDS)
    cfl_dt_max    = float(_cfl_dt_max) if _cfl_dt_max is not None else _out_interval
    cfl_dt_min    = float(getattr(cfg, 'CFL_DT_MIN', 0.01))
    cfl_dt_grow   = float(getattr(cfg, 'CFL_DT_GROW', 1.5))
    dt            = cfg.TIME_STEP_SECONDS   # legacy dt / adaptive seed
    n_steps       = int(T / dt)             # reference value for legacy-mode header

    s_rows         = grid_data["s_rows"]
    s_cols         = grid_data["s_cols"]
    slope_1d       = grid_data["slope_1d"]
    dem_1d         = grid_data["dem_1d"]
    dist_1d        = grid_data["dist_1d"]
    width_1d       = grid_data["width_1d"]        # flow width [m] (channel cross-section)
    store_area_1d  = grid_data["store_area_1d"]   # depth-from-volume footprint [m²]
    chan_mask_1d   = grid_data["chan_mask_1d"]    # True on confined channel cells
    ds_idx         = grid_data["ds_idx"]
    n_cells        = grid_data["n_cells"]
    outlet_pos     = grid_data["outlet_pos"]
    ws_mask        = grid_data["ws_mask"]
    precip_engine  = grid_data["precip_engine"]
    runoff_engine  = grid_data.get("runoff_engine")   # None when RUNOFF_SOURCE='none'

    # ── Routing scheme ────────────────────────────────────────────────────────
    scheme = getattr(cfg, 'ROUTING_SCHEME', 'kinematic').lower()
    theta  = float(getattr(cfg, 'DIFFUSION_THETA', 1.0))
    if scheme == 'diffusive':
        print(f"  Routing scheme: DIFFUSIVE wave (water-surface slope, θ={theta:g})")
    else:
        print("  Routing scheme: KINEMATIC wave (bed slope)")

    # ── Array module (numpy or cupy) ─────────────────────────────────────────
    xp     = grid_data.get("xp", np)
    _dtype = gpu_utils.get_dtype(cfg)

    # State arrays on the correct device (CPU or GPU)
    volume_1d     = xp.zeros(n_cells, dtype=_dtype)   # [m³]   water stored per cell
    Q_out_1d      = xp.zeros(n_cells, dtype=_dtype)   # [m³/s] outflow rate (for hydrograph)
    Q_out_vol_1d  = xp.zeros(n_cells, dtype=_dtype)   # [m³]   outflow VOLUME last step
    inflow_vol_1d = xp.zeros(n_cells, dtype=_dtype)   # [m³]   upstream inflow volume this step

    hydrograph = []  # list of (time_seconds, Q_m3s) — Python floats

    # Baseflow offset: add the steady pre-storm discharge (OPM_Q_MAX) so the
    # reported outlet hydrograph starts at baseflow instead of zero, making it
    # directly comparable to observed (gauged) discharge.  Pure additive offset
    # on the routed stormflow — does not affect routing/runoff generation.
    q_base = (float(getattr(cfg, 'OPM_Q_MAX', 0.0))
              if getattr(cfg, 'OPM_BASEFLOW', False) else 0.0)
    if q_base:
        print(f"  Baseflow offset added to outlet: {q_base:.3f} m³/s")

    print("=" * 60)
    if adaptive:
        print(f"TIME LOOP  |  adaptive CFL  |  C_target={cfl_target}  "
              f"dt_max={cfl_dt_max:.0f}s  dt_min={cfl_dt_min}s  "
              f"dt_grow={cfl_dt_grow:.1f}×  sim={cfg.TOTAL_SIMULATION_TIME_HOURS}h")
    else:
        print(f"TIME LOOP  |  static dt  |  steps={n_steps:,}  dt={dt}s  "
              f"sim={cfg.TOTAL_SIMULATION_TIME_HOURS}h")
    print("=" * 60)

    # ── One-time CFL diagnostic (celerity = 5/3·V for Manning wide channel) ──
    # The kinematic wave speed is c = (5/3)·V, not V — using V alone under-counts
    # the Courant number by 5/3.  This corrected check is informational only;
    # the adaptive loop tracks actual instantaneous celerity at each step.
    max_slope  = float(slope_1d.max().item())
    n_min      = float(n.min().item()) if hasattr(n, 'min') else float(n)
    V_at_1m    = (1.0 / n_min) * (1.0 ** (2.0 / 3.0)) * (max_slope ** 0.5)
    c_at_1m    = (5.0 / 3.0) * V_at_1m          # wave celerity at 1 m depth [m/s]
    c_safe_dt  = dx / c_at_1m
    if adaptive:
        print(f"  [CFL] Steepest-slope celerity (1 m depth): c={c_at_1m:.2f} m/s  "
              f"→ CFL-safe dt={c_safe_dt:.2f}s  (adaptive loop will track this)")
    else:
        c_fixed = c_at_1m * dt / dx
        if c_fixed > 1.0:
            print(f"  [CFL WARNING] Celerity-based Courant C={c_fixed:.2f} > 1.  "
                  f"Flux limiter is ACTIVE.  Set TIME_STEP_SECONDS ≤ {c_safe_dt:.2f}s")
        else:
            print(f"  [CFL OK] Celerity-based Courant C={c_fixed:.2f} ≤ 1.")
    print()

    # ── Adaptive-loop state (also used for step-tracking in legacy mode) ──────
    _eps         = 1e-9    # time-axis guard: stop before T + _eps
    _eps_div     = 1e-12   # division guard for celerity (depth · dx)
    t_seconds    = 0.0
    step_count   = 0
    cfl_min_bind = 0
    _dt_sum      = 0.0
    _dt_min_seen = float('inf')
    _dt_max_seen  = 0.0
    _dt_cfl_prev  = cfl_dt_max   # unclamped CFL dt from previous step (growth-limiter seed)
    next_output_t   = _out_interval
    next_progress_t = T / 10.0

    # ── Pre-compute static scatter-add masks (ds_idx never changes) ──────────
    valid_ds      = ds_idx >= 0         # mask: cell has a valid downstream neighbour
    ds_positions  = ds_idx[valid_ds]    # downstream position indices
    ds_safe       = xp.where(valid_ds, ds_idx, 0)   # gather-safe index (diffusive scheme)
    boundary_mask = ~valid_ds           # cells whose Q_out leaves the domain (outlet + off-mask)
    boundary_f    = boundary_mask.astype(_dtype)    # float mask for hot-loop reduction

    # ── Mass-balance accumulators (always on; routing is exactly conservative,
    #    so |error/input| should be ~machine precision — any departure flags a bug) ──
    mb_in   = xp.zeros((), dtype=_dtype)   # Σ effective-runoff volume entering routing [m³]
    mb_out  = xp.zeros((), dtype=_dtype)   # Σ volume leaving the domain at boundary cells [m³]
    mb_rain = xp.zeros((), dtype=_dtype)   # Σ gross rainfall volume [m³] (for runoff ratio)

    # Flux-limiter engagement diagnostic: max per-step fraction of wet cells clipped.
    # Device 0-d scalar (xp.maximum each step), transferred to host once at the end.
    _frac_clip_max_dev = xp.zeros((), dtype=_dtype)

    # ── Runoff-mechanism partition (vsa_opm only) ────────────────────────────
    # Σ effective-runoff volume by generating mechanism [m³].  The three sum
    # EXACTLY to mb_in (effective runoff IN) by construction in the engine.
    _partition = (runoff_engine is not None
                  and getattr(runoff_engine, '_mode', None) == 'vsa_opm')
    mb_dunne  = xp.zeros((), dtype=_dtype)   # Σ Dunne / saturation-excess [m³]
    mb_horton = xp.zeros((), dtype=_dtype)   # Σ Horton / infiltration-excess [m³]
    mb_imperv = xp.zeros((), dtype=_dtype)   # Σ impervious (urban) shedding [m³]
    partition_series = []   # (time_hr, cum_dunne_m3, cum_horton_m3, cum_imperv_m3)

    Q_outlet     = 0.0                  # initialise for progress reporting
    # Interval-averaged outlet hydrograph: accumulate the volume that leaves the
    # outlet each step (device scalar, no per-step host sync) and report
    # accumulated-volume / interval at each output time.  This is the physically
    # correct hydrograph quantity (mean flux = ΔV/Δt, mass-consistent) and removes
    # the aliasing of sub-step dispersive ripples / adaptive-dt jitter that makes a
    # point-sampled instantaneous Q_out look like a saw-tooth at the output cadence.
    _out_vol_dev = xp.zeros((), dtype=_dtype)   # Σ outlet outflow volume this interval [m³]
    _last_out_t  = 0.0                          # interval start time [s]
    t_wall_start = time.time()

    while t_seconds < T - _eps:

        # Non-adaptive: reset dt to the configured value each iteration so that
        # output-boundary clamping (below) only shortens THIS step, not all future
        # ones.  Without this, dt would "stick" at the clamped value permanently
        # (e.g. 0.9 → 0.6 after the first 600s boundary → 50% more steps).
        if not adaptive:
            dt = cfg.TIME_STEP_SECONDS

        # ── 1. Rainfall array for this step ──────────────────────────────────
        rain_1d = precip_engine.get_field_1d(t_seconds)   # [m/s], (n_cells,)

        # ── 2. Inflow buffer: accumulates VOLUME (not rate) from upstream cells ─
        # We scatter-add Q_out_vol_1d [m³] — the volume that left each upstream
        # cell in the previous step — not the rate Q_out [m³/s].  This is the
        # critical invariant for variable-dt correctness: inflow volume is fixed
        # at what the upstream step actually computed, independent of dt_current.
        # (If we scattered rates and multiplied by dt_current, a dt jump from
        # 0.01s to 7s would inject 700× a cell's water into its downstream
        # neighbour, causing the observed blow-up to ~500K m³/s.)
        inflow_vol_1d.fill(0)
        gpu_utils.scatter_add(inflow_vol_1d, ds_positions, Q_out_vol_1d[valid_ds])

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

        # Depth from stored volume over the cell's storage footprint.  Overland:
        # store_area = cell_area → depth = V/cell_area (unchanged).  Channel cells:
        # store_area = B·L → depth = V/(B·L) = channel-reach depth (confined, deeper).
        depth_1d = xp.maximum(volume_1d / store_area_1d, cfg.MIN_DEPTH_M)   # [m]
        # Floor only — no ceiling. A depth cap freezes Manning Q at the cap
        # value while volume keeps growing, creating a permanent flat plateau.
        # The flux limiter already prevents numerical runaway; deeper cells
        # simply get larger Q_out and drain faster (physically correct).

        if scheme == 'diffusive':
            # Diffusion wave: Manning on the water-surface slope along the flow path,
            # with conveyance on the flow-depth-over-the-higher-bed (CASC2D/GSSHA-style).
            # Returns (Q, A_xs, S_eff); channel cells use a confined rectangular section.
            Q_out_1d, A_xs_1d, S_eff_1d = ru.diffusive_wave_discharge(
                depth_1d, dem_1d, dist_1d, slope_1d, n, ds_safe, valid_ds,
                theta, dx, xp, cfg.MIN_DEPTH_M, width_1d, chan_mask_1d,
            )                                                          # [m³/s], [m²], [m/m]
        else:
            # Kinematic wave: Manning on the bed slope.  Channel cells use the
            # confined rectangular section (R=A/P); overland cells reproduce the
            # original mannings_velocity·cell_discharge bit-for-bit.
            Q_out_1d, A_xs_1d = ru.mannings_discharge(
                depth_1d, slope_1d, n, width_1d, chan_mask_1d, dx, xp)  # [m³/s], [m²]
            S_eff_1d    = slope_1d   # bed slope; makes the combined limit reduce to pure CFL
        # ── Adaptive CFL dt (state^n → dt^n, computed before volume advance) ──
        if adaptive:
            # Wave celerity c = (5/3)·Q/A_xs, using the flow cross-section area A_xs
            # returned by the discharge function (overland: h_flow·cell_size; channel:
            # h_flow·B).  Using A_xs — not depth_1d·dx — keeps the celerity consistent
            # with the conveyance section: for the diffusive scheme A_xs is built from
            # h_flow = h_higher (depth over the higher bed), and for channel cells from
            # the confined width B, so dt tracks the true wave speed in both.
            c_1d   = (5.0 / 3.0) * Q_out_1d / xp.maximum(A_xs_1d, _eps_div)
            # Advective CFL only.  A von Neumann diffusion-number limit
            # (dt ≤ ½dx²/D, D = Q/(2·dx·S_eff) = h_flow^(5/3)/(2n·√S_eff)) was tried
            # and REMOVED: D ∝ S_eff^(-1/2) blows up on the always-present flat-water
            # cells (ponding / backwater / saturated VSA), pinning dt at CFL_DT_MIN
            # for the whole run with NO accuracy benefit.  Stability of this scheme
            # comes from the nonlinear volume flux limiter (Q ≤ V/dt) + the
            # S_eff=max(·,0) clamp, NOT from a linear-stability dt: static dt=0.9s
            # violates the diffusion-number bound yet is stable and accurate because
            # the limiter bounds the solution.  For the diffusive scheme dt is an
            # ACCURACY knob — prefer ADAPTIVE_TIMESTEP=False with a modest TIME_STEP.
            inv_dt     = c_1d / dx
            inv_dt_max = float(inv_dt.max().item())
            dt_new     = cfl_target / max(inv_dt_max, _eps_div)
            dt_new     = min(dt_new, cfl_dt_max)
            # Growth limiter (GSSHA-style): dt may shrink instantly but can grow by at
            # most cfl_dt_grow per step.  Without this, a sudden c_max drop (e.g. after
            # a CFL_DT_MIN-bound peak) causes dt to jump 700× in one step, dumping a
            # huge volume pulse downstream and creating oscillatory instability.
            # From dt=0.01s to dt=7s with grow=1.5 takes ~17 steps — smooth ramp-up.
            dt_new = min(dt_new, _dt_cfl_prev * cfl_dt_grow)
            if dt_new < cfl_dt_min:
                cfl_min_bind += 1
                if cfl_min_bind == 1:
                    print(f"  [CFL_DT_MIN] First bind at t={t_seconds/3600:.3f}h "
                          f"(dt_adaptive={dt_new:.4f}s < {cfl_dt_min}s) — "
                          f"flux limiter engaged for fast cells")
                dt_new = cfl_dt_min
            _dt_cfl_prev = dt_new   # save post-floor, pre-clamp CFL dt for next growth limit
            dt = dt_new

        # Clamp to land exactly on output-record boundaries and simulation end.
        # This keeps the hydrograph on a clean regular time axis and ensures the
        # final step reaches T without overshoot.
        dt = min(dt, next_output_t - t_seconds, T - t_seconds)
        dt = max(dt, _eps)

        # Apply volume-conservative CFL limiter: a cell cannot eject more
        # water than it stores in one time step (prevents Courant runaway).
        # Inlined with xp.minimum/xp.maximum so it works on both CPU and GPU.
        # Diagnostic: the limiter is meant to be a RARE safety net.  When the dt
        # controller keeps the scheme inside its stability envelope only a handful
        # of pathological cells should ever clip; a large clipped fraction signals
        # dt is too aggressive (artificial diffusion / outlet ringing).
        # On-device scalars (transferred to host once at the end, like the
        # mass-balance accumulators) so the hot loop stays sync-free on GPU.
        _q_cap     = xp.maximum(volume_1d, 0.0) / dt
        _wet       = volume_1d > 0.0
        _n_wet     = _wet.sum()
        _n_clipped = ((Q_out_1d > _q_cap) & _wet).sum()
        _frac_clip = _n_clipped / xp.maximum(_n_wet, 1)
        _frac_clip_max_dev = xp.maximum(_frac_clip_max_dev, _frac_clip)
        Q_out_1d = xp.minimum(Q_out_1d, _q_cap)

        # Convert outflow rate → volume for this step.  This is the value that
        # the NEXT step's scatter-add will use — decoupled from dt_next.
        Q_out_vol_1d = Q_out_1d * dt                 # [m³] outflow volume this step

        # Volume advance
        # If a RunoffEngine is active, convert rainfall to effective runoff first
        # (forward Euler: query current VSA/mask, then advance sandbox state).
        # With RUNOFF_SOURCE='none', source_1d == rain_1d (bit-identical to old code).
        if runoff_engine is not None:
            source_1d = runoff_engine.get_effective_1d(t_seconds, rain_1d)  # [m/s]
            if _partition:
                # Component rates [m/s] stashed by _opm_effective_runoff this step.
                mb_dunne  += runoff_engine._last_dunne_rate.sum()  * (cell_area * dt)
                mb_horton += runoff_engine._last_horton_rate.sum() * (cell_area * dt)
                mb_imperv += runoff_engine._last_imperv_rate.sum() * (cell_area * dt)
            runoff_engine.update_state(rain_1d, dt)
        else:
            source_1d = rain_1d

        rain_vol   = source_1d * cell_area * dt      # [m³] effective runoff added
        volume_1d  = (volume_1d
                      + rain_vol
                      + inflow_vol_1d               # [m³] pre-computed upstream volume
                      - Q_out_vol_1d)               # [m³] pre-computed outflow volume
        volume_1d  = xp.maximum(volume_1d, 0.0)     # no negative storage

        # ── Mass-balance accumulation (device reductions; transferred once at end) ──
        mb_in   += rain_vol.sum()                              # effective runoff entering routing
        mb_out  += (Q_out_vol_1d * boundary_f).sum()          # volume leaving the domain [m³]
        mb_rain += rain_1d.sum() * (cell_area * dt)           # gross rainfall (for runoff ratio)
        _out_vol_dev += Q_out_vol_1d[outlet_pos]              # outlet outflow volume this step [m³]

        # ── Advance simulation time and accumulate step statistics ───────────
        t_seconds    += dt
        step_count   += 1
        _dt_sum      += dt
        _dt_min_seen  = min(_dt_min_seen, dt)
        _dt_max_seen  = max(_dt_max_seen, dt)

        # ── 4. Record outlet hydrograph (at output interval, not every step) ───
        # Both triggers share the single D→H transfer when both fire on one step.
        _at_output   = t_seconds >= next_output_t - _eps or t_seconds >= T - _eps
        _at_progress = t_seconds >= next_progress_t - _eps
        if _at_output:
            # Interval-mean discharge = Σ outlet volume this interval / interval length.
            # .item() converts CuPy 0-d → Python float (8-byte D→H); no-op on NumPy.
            _interval = t_seconds - _last_out_t
            Q_outlet  = float(_out_vol_dev.item()) / max(_interval, _eps) + q_base
            _out_vol_dev.fill(0)            # reset accumulator for the next interval
            _last_out_t = t_seconds
        elif _at_progress:
            # Console-only: instantaneous rate is fine for the progress line.
            Q_outlet = float(Q_out_1d[outlet_pos].item()) + q_base

        if _at_output:
            hydrograph.append((t_seconds, Q_outlet))
            if _partition:
                # Cumulative mechanism volumes [m³] at the hydrograph cadence —
                # one D→H transfer per recorded row (cheap, same rate as Q).
                partition_series.append((
                    t_seconds / 3600.0,
                    float(mb_dunne.item()),
                    float(mb_horton.item()),
                    float(mb_imperv.item()),
                ))
            next_output_t += _out_interval

        # Progress reporting every 10 % of simulation
        if _at_progress:
            elapsed = time.time() - t_wall_start
            pct     = 100.0 * t_seconds / T
            print(f"  {pct:5.1f}%  |  t={t_seconds/3600:.3f}h  "
                  f"|  Q_outlet={Q_outlet:.4f} m³/s  "
                  f"|  wall={elapsed:.1f}s")
            next_progress_t += T / 10.0

    _wall_total = time.time() - t_wall_start
    _dt_mean    = _dt_sum / step_count if step_count > 0 else 0.0
    print(f"\n  Simulation finished in {_wall_total:.1f}s  |  "
          f"{step_count:,} steps  |  "
          f"dt mean={_dt_mean:.2f}s  min={_dt_min_seen:.3f}s  max={_dt_max_seen:.1f}s")
    if adaptive and cfl_min_bind > 0:
        print(f"  [CFL_DT_MIN] Bound {cfl_min_bind} times — "
              f"flux limiter engaged for pathological fast cells")
    # Flux-limiter engagement: a stable dt keeps this near zero; a large peak
    # fraction means the limiter (not the wave equation) is doing the routing,
    # i.e. dt is too aggressive for the chosen scheme.
    _frac_clip_max = float(_frac_clip_max_dev.item())
    print(f"  Flux limiter   |  peak {100.0*_frac_clip_max:.2f}% of wet cells clipped in a step")
    if _frac_clip_max > 0.02:
        print(f"  [WARNING] Flux limiter clipped >2% of wet cells — dt is marginal; "
              f"lower CFL_TARGET or (static mode) TIME_STEP_SECONDS.")

    # ── Mass balance ─────────────────────────────────────────────────────────
    # Budget on the routed water:  INPUT − OUTFLOW − STORAGE = ERROR.
    # STORAGE includes water still in cells PLUS the final step's interior outflow,
    # which is "in transit" (subtracted from upstream cells but, due to the one-step
    # routing lag, not yet scattered downstream).  Accounting for it makes the budget
    # close to machine precision when the scheme is conservative — so any non-trivial
    # error is a genuine red flag rather than a loop-boundary artefact.
    inflight   = float((Q_out_vol_1d[valid_ds].sum()).item())   # already [m³]
    storage    = float(volume_1d.sum().item()) + inflight
    input_m3   = float(mb_in.item())
    outflow_m3 = float(mb_out.item())
    rain_m3    = float(mb_rain.item())
    error_m3   = input_m3 - outflow_m3 - storage
    rel_error  = error_m3 / input_m3 if input_m3 > 0 else 0.0
    runoff_ratio = input_m3 / rain_m3 if rain_m3 > 0 else 0.0
    status     = "PASS" if abs(rel_error) < 1e-6 else "WARN"

    print("\n" + "=" * 60)
    print("MASS BALANCE  (routed water budget)")
    print("=" * 60)
    print(f"  Gross rainfall     : {rain_m3:16.3f} m³")
    print(f"  Effective runoff IN: {input_m3:16.3f} m³   (runoff ratio {runoff_ratio:.3f})")
    print(f"  Outflow at boundary: {outflow_m3:16.3f} m³")
    print(f"  Storage (end+transit): {storage:14.3f} m³")
    print(f"  Closure error      : {error_m3:16.3f} m³   ({100.0*rel_error:+.2e} % of input)  [{status}]")
    if status == "WARN":
        print("  [WARNING] Mass balance error exceeds 1e-6 — investigate routing/runoff.")

    # ── Runoff-mechanism partition (vsa_opm only) ────────────────────────────
    partition = None
    if _partition:
        dunne_m3  = float(mb_dunne.item())
        horton_m3 = float(mb_horton.item())
        imperv_m3 = float(mb_imperv.item())
        comp_sum  = dunne_m3 + horton_m3 + imperv_m3
        f_dunne   = dunne_m3  / input_m3 if input_m3 > 0 else 0.0
        f_horton  = horton_m3 / input_m3 if input_m3 > 0 else 0.0
        f_imperv  = imperv_m3 / input_m3 if input_m3 > 0 else 0.0
        # Closure check: the three components must sum to the effective runoff IN.
        part_err  = (comp_sum - input_m3) / input_m3 if input_m3 > 0 else 0.0
        partition = dict(dunne_m3=dunne_m3, horton_m3=horton_m3, imperv_m3=imperv_m3,
                         dunne_frac=f_dunne, horton_frac=f_horton, imperv_frac=f_imperv)
        print("-" * 60)
        print("RUNOFF PARTITION  (by generating mechanism)")
        print(f"  Dunne  (saturation-excess): {dunne_m3:16.3f} m³   ({100*f_dunne:5.1f} %)")
        print(f"  Horton (infiltration-exc.): {horton_m3:16.3f} m³   ({100*f_horton:5.1f} %)")
        print(f"  Impervious (urban shed)   : {imperv_m3:16.3f} m³   ({100*f_imperv:5.1f} %)")
        print(f"  Σ components vs runoff IN  : {part_err:+.2e}  rel  "
              f"[{'PASS' if abs(part_err) < 1e-6 else 'WARN'}]")
        _write_partition_series(cfg, partition_series)

    if getattr(cfg, 'MASS_BALANCE_REPORT', True):
        _append_mass_balance_csv(cfg, scheme, theta, rain_m3, input_m3,
                                 outflow_m3, storage, error_m3, rel_error,
                                 runoff_ratio, partition)

    return hydrograph


def _write_partition_series(cfg, series):
    """Write the cumulative Dunne/Horton/Impervious time series to
    {OUTPUT_DIR}/partition_{RUN_TAG}.csv (one file per event)."""
    if not series:
        return
    import csv
    tag = getattr(cfg, 'RUN_TAG', None)
    out_dir = getattr(cfg, 'OUTPUT_DIR', 'output/')
    fname = f"partition_{tag}.csv" if tag else "partition.csv"
    path = os.path.join(out_dir, fname)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['time_hr', 'dunne_m3', 'horton_m3', 'imperv_m3'])
        for t_hr, d, h, i in series:
            w.writerow([f"{t_hr:.4f}", f"{d:.3f}", f"{h:.3f}", f"{i:.3f}"])
    print(f"  Partition time series  → {path}")


def _append_mass_balance_csv(cfg, scheme, theta, rain_m3, input_m3, outflow_m3,
                             storage_m3, error_m3, rel_error, runoff_ratio,
                             partition=None):
    """Append one row of the mass-balance budget to MASS_BALANCE_CSV (header on create).
    Appends so successive runs at different configs accumulate for side-by-side review.
    The row also carries the run tag, the key runoff knobs, and (when available) the
    Dunne/Horton/Impervious mechanism partition so a sweep is self-describing."""
    import csv
    from datetime import datetime

    path = getattr(cfg, 'MASS_BALANCE_CSV', None) or (
        getattr(cfg, 'OUTPUT_DIR', 'output/') + 'mass_balance.csv')
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    p = partition or {}
    header = ['timestamp', 'run_tag', 'scheme', 'theta', 'runoff_source',
              'sd_source', 'sd_max', 'ksat_scale', 'infiltration', 'impervious',
              'rain_m3', 'input_m3', 'outflow_m3', 'storage_m3',
              'error_m3', 'rel_error', 'runoff_ratio',
              'dunne_m3', 'horton_m3', 'imperv_m3',
              'dunne_frac', 'horton_frac', 'imperv_frac']
    def _f(key, fmt="{}"):
        v = p.get(key)
        return "" if v is None else fmt.format(v)
    row = [datetime.now().isoformat(timespec='seconds'),
           getattr(cfg, 'RUN_TAG', ''),
           scheme, theta, getattr(cfg, 'RUNOFF_SOURCE', 'none'),
           getattr(cfg, 'OPM_SD_SOURCE', ''),
           getattr(cfg, 'OPM_SD_MAX_INITIAL', ''),
           getattr(cfg, 'OPM_GA_KSAT_SCALE', ''),
           getattr(cfg, 'OPM_INFILTRATION', ''),
           getattr(cfg, 'IMPERVIOUS_SOURCE', ''),
           f"{rain_m3:.3f}", f"{input_m3:.3f}", f"{outflow_m3:.3f}",
           f"{storage_m3:.3f}", f"{error_m3:.6e}", f"{rel_error:.6e}",
           f"{runoff_ratio:.4f}",
           _f('dunne_m3', "{:.3f}"), _f('horton_m3', "{:.3f}"), _f('imperv_m3', "{:.3f}"),
           _f('dunne_frac', "{:.4f}"), _f('horton_frac', "{:.4f}"), _f('imperv_frac', "{:.4f}")]
    write_header = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow(row)
    print(f"  Mass-balance row appended → {path}")


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
