"""
runoff_input.py
===============
Modular runoff generation engine for the kinematic-wave router.

Sits between PrecipEngine (rainfall in [m/s]) and the time loop (volume update),
transforming raw rainfall into effective surface runoff based on the selected mode.

Modes (set via config.RUNOFF_SOURCE):
  'none'        – all rainfall is direct runoff (default, backward compatible)
  'coefficient' – multiply rainfall by static spatial Cf raster [0–1]
  'raster'      – read pre-computed runoff raster time series [m/s]
  'scs_cn'      – SCS Curve Number method with per-cell CN raster
  'vsa_opm'     – Variable Source Area: Pradhan & Ogden (2010) OPM

Usage in the time loop (forward Euler):
    source_1d = runoff_engine.get_effective_1d(t_s, rain_1d)   # current state
    runoff_engine.update_state(rain_1d, dt)                     # advance state
    rain_vol = source_1d * cell_area * dt

Reference:
    Pradhan, N.R. and Ogden, F.L. (2010). Development of a one-parameter variable
    source area runoff model for ungauged basins. Advances in Water Resources,
    33(5), pp.572–584.
"""

import numpy as np
import pandas as pd
import rasterio
import os

import gpu_utils

# ── OPM physical constants (Pradhan & Ogden 2010) ────────────────────────────
_OPM_SD_MIN = 0.001          # m   minimum saturation deficit
_OPM_Q_MIN  = 0.001          # m³/s  minimum discharge (Eq 10)


def _resolve_sd_params(cfg, cell_size):
    """
    Resolve OPM soil parameters from config (manual) or GEE.

    Returns dict with keys: sd_max, sd_min, phi, sd_max_per_polygon, ksat_ms,
    deficit_raster.  Per-zone SD is reduced from 'deficit_raster' in
    _init_vsa_opm using the rainfall partition (cell_polygon), not here.
    """
    sd_source   = getattr(cfg, 'OPM_SD_SOURCE', 'manual').lower()
    ksat_ms     = float(getattr(cfg, 'OPM_K_SAT', 44.0)) / 86400.0

    _manual_params = {
        'sd_max': float(cfg.OPM_SD_MAX_INITIAL),
        'sd_min': _OPM_SD_MIN,
        'phi': float(getattr(cfg, 'OPM_PHI', 0.10)),
        'sd_max_per_polygon': None,
        'ksat_ms': ksat_ms,
        'deficit_raster': None,
    }

    if sd_source != 'gee':
        return _manual_params

    # Primary: EVENT_START_UTC is the event start — its date IS the SERVES target date.
    target_date = None
    evt = getattr(cfg, 'EVENT_START_UTC', None)
    if evt:
        target_date = str(evt).split()[0]   # 'YYYY-MM-DD'
        print(f"  SERVES target date → {target_date}  (from EVENT_START_UTC)")

    # Legacy fallback: IMERG_START_LOCAL (for old configs that predate EVENT_START_UTC)
    if target_date is None:
        method = getattr(cfg, 'PRECIP_METHOD', '').lower()
        start  = getattr(cfg, 'IMERG_START_LOCAL', None)
        if method.startswith('imerg') and start:
            target_date = str(start).split()[0]
            print(f"  SERVES target date → {target_date}  "
                  f"(legacy: derived from IMERG_START_LOCAL)")

    if target_date is None:
        print("  [WARN] EVENT_START_UTC not set; using manual SD/phi values.")
        return _manual_params

    try:
        from serves_gee import compute_opm_params, download_deficit_raster
    except ImportError:
        print("  [WARN] earthengine-api not installed; using manual values.")
        return _manual_params

    geojson = getattr(cfg, 'OPM_WATERSHED_GEOJSON', 'output/watershed.geojson')
    sat     = getattr(cfg, 'SERVES_SATELLITE', 'landsat')
    window  = getattr(cfg, 'SERVES_SEARCH_WINDOW', 16)
    band    = getattr(cfg, 'OPM_SOILGRIDS_DEPTH', 'b30')
    project = getattr(cfg, 'GEE_PROJECT', None)

    # Select lookup CSV and GEE land cover source to match MANNINGS_N_SOURCE.
    _n_source = getattr(cfg, 'MANNINGS_N_SOURCE', 'lulc').lower()
    if _n_source == 'lcz':
        lut          = getattr(cfg, 'LCZ_LOOKUP_CSV',  'lcz_lookup.csv')
        gee_lc_source = 'lcz'
    else:
        lut          = getattr(cfg, 'LULC_LOOKUP_CSV', 'lulc_lookup.csv')
        gee_lc_source = 'worldcover'

    # Watershed-level SD_max + phi + diagnostics (scalars; no partition).
    gee_result = compute_opm_params(
        watershed_geojson_path=geojson, cell_size=cell_size,
        lookup_csv_path=lut, target_date=target_date,
        satellite=sat, search_window=window, soil_depth_band=band,
        project=project,
        sd_reducer=getattr(cfg, 'OPM_SD_REDUCER', 'mean'),
        lulc_source=gee_lc_source,
    )

    if gee_result is not None:
        sd_max = gee_result['sd_max']
        phi    = gee_result['phi']
    else:
        sd_max = float(cfg.OPM_SD_MAX_INITIAL)
        phi    = float(getattr(cfg, 'OPM_PHI', 0.10))

    # Per-cell deficit raster (aligned to the routing grid, clipped to the
    # watershed).  The engine reduces it per precipitation zone (cell_polygon),
    # so the SD partition is identical to the rainfall partition.
    #
    # Date-stamp the filename so each event caches its own raster:
    #   deficit_serves_2024-09-26.tif vs deficit_serves_2024-10-05.tif
    # Re-runs of the same event reuse the cached file (no re-download).
    # Falls back to the plain name when target_date is None (manual SD mode).
    out_path = getattr(cfg, 'OPM_DEFICIT_RASTER', None) \
        or (getattr(cfg, 'OUTPUT_DIR', 'output/') + 'deficit_serves.tif')
    if target_date:
        import os as _os
        _base, _ext = _os.path.splitext(out_path)
        out_path = f"{_base}_{target_date}{_ext}"
    deficit_raster = None
    try:
        deficit_raster = download_deficit_raster(
            dem_path=cfg.ROUTING_DEM_PATH, watershed_geojson_path=geojson,
            output_path=out_path, lookup_csv_path=lut, target_date=target_date,
            satellite=sat, search_window=window, soil_depth_band=band,
            project=project, lulc_source=gee_lc_source,
        )
    except Exception as exc:
        print(f"  [WARN] deficit raster download failed: {exc}")

    # ── Print diagnostics ────────────────────────────────────────────────
    print(f"  OPM params    |  SD_max(ws)={sd_max:.4f} m  (source={sd_source})"
          f"  phi={phi:.4f}  K_sat={ksat_ms * 86400:.2f} m/day")
    if gee_result is not None:
        print(f"                |  theta=[{gee_result.get('theta_min', 0):.3f},"
              f" {gee_result.get('theta_max', 0):.3f}]"
              f"  Z_r_mean={gee_result.get('root_depth_mean', 0):.2f} m")
    print(f"                |  deficit raster: "
          f"{'ready' if deficit_raster else 'unavailable → per-zone SD = watershed SD'}")

    return {
        'sd_max': sd_max,
        'sd_min': _OPM_SD_MIN,
        'phi': phi,
        'sd_max_per_polygon': None,   # resolved per-zone from the raster instead
        'ksat_ms': ksat_ms,
        'deficit_raster': deficit_raster,
    }


def _per_zone_sd_from_raster(deficit_path, cell_polygon, n_polygons,
                             s_rows, s_cols, reducer, sd_min, ws_default):
    """
    Reduce the per-cell deficit raster into one SD_max per precipitation zone.

    For each zone p, SD_max[p] = mean|max of the deficit over ONLY the watershed
    cells that zone owns (cell_polygon == p) — the exact same nearest-station
    partition the rainfall uses.  Zones with no deficit data (or out-of-basin
    stations that own no cells) get *ws_default* (the watershed SD_max).  The
    result is therefore watershed-bounded and consistent with the rain zoning.
    """
    with rasterio.open(deficit_path) as src:
        deficit2d = src.read(1).astype(np.float64)
        nodata    = src.nodata
    if nodata is not None:
        deficit2d[deficit2d == nodata] = np.nan

    _to_np = lambda a: a.get() if hasattr(a, 'get') else np.asarray(a)
    sr = _to_np(s_rows); sc = _to_np(s_cols)

    # Guard against a cached raster on a different grid than the routing DEM.
    if sr.max() >= deficit2d.shape[0] or sc.max() >= deficit2d.shape[1]:
        print("  [WARN] deficit raster grid ≠ routing grid; "
              "per-zone SD → watershed SD.")
        return np.full(n_polygons, ws_default, dtype=np.float64)

    deficit_1d = deficit2d[sr, sc]                 # (n_cells,) deficit per cell
    finite     = np.isfinite(deficit_1d)
    redux      = np.max if reducer == 'max' else np.mean

    sd = np.full(n_polygons, ws_default, dtype=np.float64)
    for p in range(n_polygons):
        m = (cell_polygon == p) & finite
        if m.any():
            sd[p] = float(redux(deficit_1d[m]))
    return np.maximum(sd, sd_min)


def _deficit_1d_from_raster(deficit_path, s_rows, s_cols):
    """
    Per-cell SERVES deficit [(porosity−θ)·Z_r, in m] over the routing grid.

    Returns (n_cells,) float64 with nodata/out-of-grid cells set to NaN, or
    None if the raster grid does not match the routing grid.
    """
    with rasterio.open(deficit_path) as src:
        deficit2d = src.read(1).astype(np.float64)
        nodata    = src.nodata
    if nodata is not None:
        deficit2d[deficit2d == nodata] = np.nan

    _to_np = lambda a: a.get() if hasattr(a, 'get') else np.asarray(a)
    sr = _to_np(s_rows); sc = _to_np(s_cols)
    if sr.max() >= deficit2d.shape[0] or sc.max() >= deficit2d.shape[1]:
        return None
    return deficit2d[sr, sc]


# ─────────────────────────────────────────────────────────────────────────────
class RunoffEngine:
    """
    Unified runoff generation engine.

    Parameters
    ----------
    cfg       : config module
    grid_data : dict returned by kinematic_wave_router.initialise_grid()
                Must contain 'faccum_1d' for 'vsa_opm' mode.
    """

    def __init__(self, cfg, grid_data):
        mode = getattr(cfg, 'RUNOFF_SOURCE', 'none').lower()
        self._mode    = mode
        self._n_cells = grid_data['n_cells']
        self._s_rows  = grid_data['s_rows']
        self._s_cols  = grid_data['s_cols']
        self._nrows   = grid_data['nrows']
        self._ncols   = grid_data['ncols']

        if mode == 'none':
            pass
        elif mode == 'coefficient':
            self._init_coefficient(cfg, grid_data)
        elif mode == 'raster':
            self._init_raster(cfg, grid_data)
        elif mode == 'scs_cn':
            self._init_scs_cn(cfg, grid_data)
        elif mode == 'vsa_opm':
            self._init_vsa_opm(cfg, grid_data)
        else:
            raise ValueError(
                f"RUNOFF_SOURCE='{mode}' is not recognised. "
                "Valid options: 'none', 'coefficient', 'raster', 'scs_cn', 'vsa_opm'."
            )

        print(f"  RunoffEngine    |  mode='{mode}'")

    # ── Public interface ──────────────────────────────────────────────────────

    def update_state(self, rain_1d, dt):
        """
        Advance internal state by one timestep.

        Called AFTER get_effective_1d (forward Euler): the current state
        determines this step's runoff, then state advances for the next step.

        Parameters
        ----------
        rain_1d : (n_cells,) float64 [m/s]  raw rainfall from PrecipEngine
        dt      : float  timestep [s]
        """
        if self._mode == 'vsa_opm':
            self._update_opm_sandbox(rain_1d, dt)
        elif self._mode == 'scs_cn':
            self._update_scs_cn(rain_1d, dt)
        # 'none', 'coefficient', 'raster': stateless → no-op

    def get_effective_1d(self, t_seconds, rain_1d):
        """
        Effective runoff rate [m/s] per active cell, shape (n_cells,).

        Uses the state set by the PREVIOUS update_state call (forward Euler).
        At t=0 uses the initial state set during __init__.

        Parameters
        ----------
        t_seconds : float  current simulation time [s]
        rain_1d   : (n_cells,) float64 [m/s]  raw rainfall from PrecipEngine
        """
        if self._mode == 'none':
            return rain_1d
        elif self._mode == 'coefficient':
            return rain_1d * self._Cf_1d
        elif self._mode == 'raster':
            return self._interp_raster(t_seconds)
        elif self._mode == 'scs_cn':
            return self._get_scs_rate()
        elif self._mode == 'vsa_opm':
            return self._opm_effective_runoff(rain_1d)

    def _opm_effective_runoff(self, rain_1d):
        """
        Unified per-cell effective runoff [m/s] for vsa_opm mode:

            runoff = rain · [ Imp + (1 − Imp) · max(in_VSA, infil_excess_frac) ]

        - Impervious fraction Imp always sheds rain (urban contributes regardless
          of A_t).
        - Pervious fraction sheds rain if the cell is in the VSA (saturation
          excess) OR rainfall beats the Green-Ampt infiltration capacity
          (Hortonian / infiltration excess).  `max` caps shedding at 100 % of rain.
        With OPM_INFILTRATION='none' and no impervious layer this reduces exactly
        to the original `rain · in_VSA`.
        """
        xp = self._xp
        if self._infiltration == 'green_ampt':
            f_p = self._ga_ksat * (1.0 + self._ga_psi * self._ga_dtheta0
                                   / xp.maximum(self._ga_F, self._GA_F_FLOOR))
            excess      = xp.maximum(rain_1d - f_p, 0.0)
            excess_frac = xp.where(rain_1d > 0.0,
                                   excess / xp.maximum(rain_1d, 1e-30), 0.0)
        else:
            excess_frac = 0.0
        pervious_frac = xp.where(self._vsa_mask, 1.0, excess_frac)
        return rain_1d * (self._imperv_1d
                          + (1.0 - self._imperv_1d) * pervious_frac)

    def get_effective_2d(self, t_seconds, rain_1d):
        """
        2-D runoff map (nrows × ncols), NaN outside watershed.
        Used by the animation script for overlay rendering.
        """
        eff_1d = self.get_effective_1d(t_seconds, rain_1d)
        out = np.full((self._nrows, self._ncols), np.nan, dtype=np.float64)
        out[self._s_rows, self._s_cols] = eff_1d
        return out

    def is_active(self, t_seconds):
        """True if any cell is generating non-zero runoff this step."""
        if self._mode == 'none':
            return False
        if self._mode == 'vsa_opm':
            # Runoff can come from the VSA, impervious cells, or Green-Ampt
            # infiltration-excess — any of which makes the engine "active".
            return (bool(self._vsa_mask.any())
                    or bool((self._imperv_1d > 0).any())
                    or self._infiltration == 'green_ampt')
        if self._mode == 'coefficient':
            return bool((self._Cf_1d > 0).any())
        if self._mode == 'raster':
            return bool((self._interp_raster(t_seconds) > 0).any())
        if self._mode == 'scs_cn':
            return bool((self._delta_Pe_m > 0).any())
        return False

    def get_opm_diagnostics(self):
        """
        Return current OPM state as a dict (vsa_opm mode only).

        Per-polygon mode: z_m, SD_max_t, A_t_m2 are (n_polygons,) arrays.
        Single mode:      z_m, SD_max_t, A_t_m2 are scalars.
        VSA_m2 and VSA_fraction are always scalar (catchment-wide totals).
        """
        if self._mode != 'vsa_opm':
            return {}
        d = {
            "z_m":          self._opm_z.copy() if self._per_polygon else self._opm_z,
            "SD_max_t":     self._opm_SD_max.copy() if self._per_polygon else self._opm_SD_max,
            "A_t_m2":       self._opm_A_t.copy() if self._per_polygon else self._opm_A_t,
            "VSA_m2":       float(self._vsa_mask.sum()) * self._cell_area,
            "VSA_fraction": float(self._vsa_mask.sum()) / self._n_cells,
            "per_polygon":  self._per_polygon,
        }
        if self._per_polygon:
            d["n_polygons"] = self._n_polygons
        return d

    # ── Mode: 'coefficient' ───────────────────────────────────────────────────

    def _init_coefficient(self, cfg, grid_data):
        path = cfg.RUNOFF_COEFFICIENT_PATH
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"RUNOFF_COEFFICIENT_PATH '{path}' not found. "
                "Generate it with tools/generate_runoff_examples.py."
            )
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float64)
            nd  = src.nodata
        if nd is not None:
            arr[arr == nd] = 0.0
        arr = np.clip(arr, 0.0, 1.0)
        s_rows, s_cols = grid_data['s_rows'], grid_data['s_cols']
        self._Cf_1d = arr[s_rows, s_cols]

    # ── Mode: 'raster' ────────────────────────────────────────────────────────

    def _init_raster(self, cfg, grid_data):
        manifest_path = cfg.RUNOFF_RASTER_MANIFEST
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(
                f"RUNOFF_RASTER_MANIFEST '{manifest_path}' not found."
            )
        mf = pd.read_csv(manifest_path)
        s_rows, s_cols = grid_data['s_rows'], grid_data['s_cols']

        self._raster_times = mf['time_s'].values.astype(np.float64)
        self._raster_cache = {}
        for _, row in mf.iterrows():
            t_s  = float(row['time_s'])
            fpath = row['filepath']
            with rasterio.open(fpath) as src:
                arr = src.read(1).astype(np.float64)
                nd  = src.nodata
            if nd is not None:
                arr[arr == nd] = 0.0
            arr = np.maximum(arr, 0.0)
            self._raster_cache[t_s] = arr[s_rows, s_cols]

    def _interp_raster(self, t_seconds):
        times = self._raster_times
        idx   = np.searchsorted(times, t_seconds, side='right') - 1
        idx   = int(np.clip(idx, 0, len(times) - 2))
        t0, t1 = times[idx], times[idx + 1]
        r0 = self._raster_cache[t0]
        r1 = self._raster_cache[t1]
        w  = (t_seconds - t0) / (t1 - t0) if t1 > t0 else 0.0
        return r0 + w * (r1 - r0)

    # ── Mode: 'scs_cn' ────────────────────────────────────────────────────────

    def _init_scs_cn(self, cfg, grid_data):
        path = cfg.RUNOFF_CN_PATH
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"RUNOFF_CN_PATH '{path}' not found. "
                "Generate it with tools/generate_runoff_examples.py."
            )
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float64)
            nd  = src.nodata
        if nd is not None:
            arr[arr == nd] = 75.0
        arr = np.clip(arr, 1.0, 100.0)
        s_rows, s_cols = grid_data['s_rows'], grid_data['s_cols']
        CN_1d = arr[s_rows, s_cols]

        Ia_factor     = getattr(cfg, 'RUNOFF_SCS_Ia_FACTOR', 0.2)
        S_1d          = 25400.0 / CN_1d - 254.0          # [mm] max retention
        self._Ia_1d   = Ia_factor * S_1d                  # [mm] initial abstraction
        self._S_1d    = S_1d

        self._cumrain_mm  = np.zeros(self._n_cells, dtype=np.float64)
        self._Pe_mm_old   = np.zeros(self._n_cells, dtype=np.float64)
        self._scs_rate_ms = np.zeros(self._n_cells, dtype=np.float64)  # [m/s]

    def _scs_formula(self, P_mm):
        """SCS-CN accumulated effective rainfall [mm] from cumulative P [mm]."""
        excess = P_mm - self._Ia_1d
        return np.where(
            excess > 0,
            (excess ** 2) / (excess + self._S_1d),
            0.0,
        )

    def _update_scs_cn(self, rain_1d, dt):
        """Advance SCS-CN state and store instantaneous runoff rate [m/s]."""
        self._cumrain_mm += rain_1d * dt * 1000.0      # m/s → mm
        Pe_new = self._scs_formula(self._cumrain_mm)
        delta  = np.maximum(Pe_new - self._Pe_mm_old, 0.0)   # [mm] this step
        self._scs_rate_ms = (delta / 1000.0) / dt if dt > 0 \
            else np.zeros(self._n_cells)                       # [m/s]
        self._Pe_mm_old = Pe_new

    def _get_scs_rate(self):
        """Return SCS effective runoff rate [m/s] computed by last update_state."""
        return self._scs_rate_ms

    # ── Mode: 'vsa_opm' ──────────────────────────────────────────────────────

    def _init_vsa_opm(self, cfg, grid_data):
        """
        Initialise the OPM Variable Source Area engine.

        Implements Pradhan & Ogden (2010) Equations 4, 5, 10, 12.
        H_a is computed from Eq 4 using the initial A_t from Eq 10.

        Per-polygon mode (Thiessen / IDW):
            When cell_polygon is available in grid_data, each precipitation zone
            gets its own sandbox (z, SD_max, A_t).  The divide cell per zone is
            the cell with minimum flow accumulation within that zone.
            Catchment-wide constants (H_a, A_1, A_outlet) remain shared.
        """
        Q_max = float(cfg.OPM_Q_MAX)
        if Q_max <= _OPM_Q_MIN:
            raise ValueError(
                f"OPM_Q_MAX={Q_max} m³/s must be > {_OPM_Q_MIN} m³/s (Q_min)."
            )

        cell_area = float(grid_data['cell_area'])
        cell_size = float(grid_data['cell_size'])
        slope_1d  = grid_data['slope_1d']
        faccum_1d = grid_data['faccum_1d']   # (n_cells,) cell counts

        self._cell_area    = cell_area
        self._cell_size    = cell_size

        # Upslope contributing area per cell [m²] — computed once
        self._upslope_area = faccum_1d * cell_area   # (n_cells,) [m²]

        # ── OPM scalars ───────────────────────────────────────────────────────
        params = _resolve_sd_params(cfg, cell_size)
        SD_max_initial  = params['sd_max']
        sd_min          = params['sd_min']
        phi             = params['phi']
        ksat_ms         = params['ksat_ms']

        self._sd_min  = sd_min
        self._phi     = phi
        self._ksat_ms = ksat_ms

        # A_1: upslope area of a single divide cell [m²]
        A_1 = cell_area

        # A_outlet: total catchment area [m²]
        A_outlet = float(faccum_1d[-1]) * cell_area

        # Eq 10 — initial threshold contributing area from single discharge measurement
        A_t_init = A_outlet / (1.0 - np.log(_OPM_Q_MIN / Q_max))

        # Eq 4 ratio (shared — A_t_init comes from watershed-level Q_max)
        ratio = A_t_init / (A_t_init - A_1)

        # Store catchment-wide constants
        self._opm_A_1      = A_1
        self._opm_A_outlet = A_outlet
        self._opm_A_t_init = A_t_init

        # Extensibility hook: records which A_t initialisation method was used.
        self._at_method = 'single_measurement'

        # ── Backend (numpy / cupy) for the vectorised hot path ────────────────
        # _upslope_area is already CuPy in GPU mode (built from CuPy faccum_1d),
        # so get_xp picks the right module for every per-cell/per-zone op below.
        self._xp = gpu_utils.get_xp(self._upslope_area)
        xp = self._xp

        # ── Impervious fraction (urban areas shed rain regardless of A_t) ─────
        import routing_utils as _ru
        imperv_np = _ru.resolve_impervious_fraction(cfg, grid_data)   # (n_cells,)
        self._imperv_1d = xp.asarray(imperv_np)

        # ── Green-Ampt per-cell infiltration setup ────────────────────────────
        self._infiltration = getattr(cfg, 'OPM_INFILTRATION', 'none').lower()
        self._GA_F_FLOOR   = 1e-9        # m — floor on F so f_p is finite at F=0
        if self._infiltration == 'green_ampt':
            self._ga_psi = float(getattr(cfg, 'OPM_GA_SUCTION_M', 0.15))
            # Vertical surface infiltration capacity (NOT the lateral OPM_K_SAT).
            kv_ms = float(getattr(cfg, 'OPM_GA_KSAT_MMHR', 12.0)) / 1000.0 / 3600.0

            s_rows = grid_data['s_rows']
            s_cols = grid_data['s_cols']

            # Root-zone depth Z_r per cell from the SAME land-cover source that
            # built the SERVES deficit raster, so Δθ₀ = deficit / Z_r is consistent.
            _n_source = getattr(cfg, 'MANNINGS_N_SOURCE', 'lulc').lower()
            _lc_src   = 'lcz' if _n_source == 'lcz' else 'lulc'
            zr_default = float(getattr(cfg, 'OPM_SD_MAX_INITIAL', 0.5)) or 0.5
            zr_np = np.maximum(
                _ru.resolve_lulc_field(cfg, grid_data, 'root_zone_depth_m',
                                       zr_default, _lc_src), 1e-3)

            # Δθ₀ = initial moisture deficit (porosity − θ) per cell.
            #   From the SERVES deficit raster: Δθ₀ = deficit / Z_r.
            #   Fallback (no raster / nodata): watershed-scalar SD_max ÷ mean Z_r.
            _fallback = float(SD_max_initial) / float(zr_np.mean())
            dtheta0_np = None
            deficit_raster = params.get('deficit_raster')
            if deficit_raster:
                dfc = _deficit_1d_from_raster(deficit_raster, s_rows, s_cols)
                if dfc is not None:
                    with np.errstate(invalid='ignore', divide='ignore'):
                        dtheta0_np = dfc / zr_np
            if dtheta0_np is None:
                dtheta0_np = np.full(self._n_cells, _fallback, dtype=np.float64)
            _bad = ~np.isfinite(dtheta0_np)
            if _bad.any():
                dtheta0_np[_bad] = _fallback
            dtheta0_np = np.clip(dtheta0_np, 0.0, 1.0)

            self._ga_ksat    = xp.asarray(
                np.full(self._n_cells, kv_ms, dtype=np.float64))
            self._ga_dtheta0 = xp.asarray(dtheta0_np)
            self._ga_F       = xp.zeros(self._n_cells, dtype=np.float64)
            print(f"  Green-Ampt    |  K_v={kv_ms*1000*3600:.1f} mm/hr"
                  f"  psi={self._ga_psi} m"
                  f"  dtheta0=[{dtheta0_np.min():.3f}, {dtheta0_np.max():.3f}]"
                  f"  mean={dtheta0_np.mean():.3f}")
        else:
            self._ga_ksat = self._ga_dtheta0 = self._ga_F = None

        # ── Per-polygon vs single-sandbox branching ───────────────────────────
        cell_polygon = grid_data.get('cell_polygon')
        use_per_polygon = getattr(cfg, 'OPM_PER_POLYGON', True)

        if cell_polygon is not None and use_per_polygon:
            # ── Per-polygon mode ──────────────────────────────────────────────
            # Each precipitation zone (nearest-gauge region) gets its own
            # sandbox state so that spatially variable rainfall drives local
            # VSA expansion independently.
            cell_polygon = np.asarray(cell_polygon).ravel()
            # One zone per precipitation gauge — not cell_polygon.max()+1, which
            # undercounts when trailing gauges have no nearest cell (e.g. IMERG
            # edge pixels).  Aligning n_polygons with the gauge count keeps the
            # per-polygon state arrays in step with sd_max_per_polygon (one value
            # per gauge from SERVES) and with downstream consumers that index by
            # gauge id.  Empty zones are handled by the fallback divide below.
            precip_engine = grid_data.get('precip_engine')
            n_gauges      = getattr(precip_engine, '_n_gauges', None)
            n_polygons    = int(n_gauges) if n_gauges \
                else int(cell_polygon.max()) + 1

            # Ensure NumPy for the init loop (faccum_1d / slope_1d may be CuPy)
            _to_np = lambda a: a.get() if hasattr(a, 'get') else np.asarray(a)
            faccum_np = _to_np(faccum_1d)
            slope_np  = _to_np(slope_1d)
            dem = grid_data['dem']
            s_rows = grid_data['s_rows']
            s_cols = grid_data['s_cols']

            divide_idx     = np.empty(n_polygons, dtype=np.intp)
            slope_divide   = np.empty(n_polygons, dtype=np.float64)

            # Catchment-wide fallback divide (min faccum, highest elevation on tie).
            # Used for zones that contain no watershed cells — possible when the
            # precipitation stations come from a gridded product (e.g. IMERG) whose
            # buffered footprint includes pixels with no cell nearest to them.  Such
            # zones never index any real cell, so the fallback only avoids the
            # empty-array reduction; it does not affect results.
            g_cand = np.where(faccum_np == faccum_np.min())[0]
            g_elev = dem[s_rows[g_cand], s_cols[g_cand]]
            global_divide = int(g_cand[g_elev.argmax()])

            for p in range(n_polygons):
                local_idx = np.where(cell_polygon == p)[0]
                if local_idx.size == 0:
                    divide_idx[p]   = global_divide
                    slope_divide[p] = float(slope_np[global_divide])
                    continue
                local_fa  = faccum_np[local_idx]
                candidates = local_idx[local_fa == local_fa.min()]
                elev = dem[s_rows[candidates], s_cols[candidates]]
                best = candidates[elev.argmax()]
                divide_idx[p]   = best
                slope_divide[p] = float(slope_np[best])

            self._per_polygon          = True
            self._n_polygons           = n_polygons
            self._cell_polygon         = cell_polygon
            self._polygon_divide_idx   = divide_idx
            self._polygon_slope_divide = slope_divide

            # Per-zone SD_max from the deficit raster, reduced over each zone's
            # OWN watershed cells (same partition as rainfall).  Zones with no
            # deficit data fall back to the watershed SD_max.
            deficit_raster = params.get('deficit_raster')
            reducer = getattr(cfg, 'OPM_SD_REDUCER', 'mean').lower()
            if deficit_raster:
                sd_init_arr = _per_zone_sd_from_raster(
                    deficit_raster, cell_polygon, n_polygons,
                    s_rows, s_cols, reducer, sd_min, SD_max_initial)
                n_real = int((np.abs(sd_init_arr - SD_max_initial) > 1e-9).sum())
                print(f"                |  Per-zone SD ({reducer}) over watershed "
                      f"cells: {n_real}/{n_polygons} zones populated, "
                      f"range=[{sd_init_arr.min():.3f}, {sd_init_arr.max():.3f}] m")
            else:
                sd_init_arr = np.full(n_polygons, SD_max_initial,
                                      dtype=np.float64)
            self._SD_max_initial = sd_init_arr

            self._ksat_ms = np.full(n_polygons, ksat_ms, dtype=np.float64)

            Rf_init_arr = sd_min / sd_init_arr
            H_a_arr = ratio * np.log(Rf_init_arr)
            self._opm_H_a = H_a_arr

            # Per-polygon state arrays
            self._opm_z      = np.zeros(n_polygons, dtype=np.float64)
            self._opm_SD_max = sd_init_arr.copy()
            self._opm_A_t    = np.full(n_polygons, A_t_init, dtype=np.float64)

            # Initial VSA mask
            A_t_per_cell = self._opm_A_t[cell_polygon]
            if hasattr(self._upslope_area, '__cuda_array_interface__'):
                import cupy as _cp
                A_t_per_cell = _cp.asarray(A_t_per_cell)
            self._vsa_mask = self._upslope_area > A_t_per_cell

            print(f"  OPM           |  A_outlet={A_outlet:.3e} m²"
                  f"  A_t_init={A_t_init:.3e} m²")
            print(f"                |  H_a={H_a_arr}  phi={phi}  Q_max={Q_max} m³/s")
            print(f"                |  SD_max_init={sd_init_arr}")
            print(f"                |  Per-polygon mode: {n_polygons} zones")
            print(f"                |  Initial VSA={self._vsa_mask.sum():,} cells"
                  f" ({100*self._vsa_mask.mean():.1f}% of watershed)")

            # Move per-zone state + indices onto the active backend so the
            # vectorised sandbox update runs entirely in numpy OR cupy.
            self._cell_polygon         = xp.asarray(cell_polygon)
            self._polygon_divide_idx   = xp.asarray(divide_idx)
            self._polygon_slope_divide = xp.asarray(slope_divide)
            self._SD_max_initial       = xp.asarray(sd_init_arr)
            self._ksat_ms              = xp.asarray(self._ksat_ms)
            self._opm_H_a              = xp.asarray(H_a_arr)
            self._opm_z                = xp.asarray(self._opm_z)
            self._opm_SD_max           = xp.asarray(self._opm_SD_max)
            self._opm_A_t              = xp.asarray(self._opm_A_t)
        else:
            # ── Single-sandbox mode (uniform rainfall) ────────────────────────
            self._per_polygon  = False
            _to_np = lambda a: a.get() if hasattr(a, 'get') else np.asarray(a)
            faccum_np = _to_np(faccum_1d)
            slope_np  = _to_np(slope_1d)
            dem = grid_data['dem']
            s_rows = grid_data['s_rows']
            s_cols = grid_data['s_cols']
            min_fa = faccum_np.min()
            candidates = np.where(faccum_np == min_fa)[0]
            elev = dem[s_rows[candidates], s_cols[candidates]]
            divide_cell = candidates[elev.argmax()]
            self._slope_divide = float(slope_np[divide_cell])
            self._divide_cell  = int(divide_cell)

            self._SD_max_initial = SD_max_initial  # scalar
            Rf_init = sd_min / SD_max_initial
            H_a = ratio * np.log(Rf_init)
            self._opm_H_a = H_a  # scalar

            # Scalar state variables
            self._opm_z      = 0.0
            self._opm_SD_max = SD_max_initial
            self._opm_A_t    = A_t_init

            self._vsa_mask = self._upslope_area > A_t_init

            print(f"  OPM           |  A_outlet={A_outlet:.3e} m²"
                  f"  A_t_init={A_t_init:.3e} m²")
            print(f"                |  H_a={H_a:.4f}  SD_max_init={SD_max_initial} m"
                  f"  phi={phi}  Q_max={Q_max} m³/s")
            print(f"                |  Initial VSA={self._vsa_mask.sum():,} cells"
                  f" ({100*self._vsa_mask.mean():.1f}% of watershed)")

    # ── OPM sandbox update ────────────────────────────────────────────────────

    def _update_opm_sandbox(self, rain_1d, dt):
        """Advance OPM sandbox state by one timestep (dispatches to mode)."""
        if self._per_polygon:
            self._update_opm_sandbox_per_polygon(rain_1d, dt)
        else:
            self._update_opm_sandbox_single(rain_1d, dt)
        # Advance per-cell Green-Ampt cumulative infiltration AFTER the sandbox
        # has used the current-step F (forward Euler — matches get_effective_1d).
        self._update_ga_F(rain_1d, dt)

    def _update_ga_F(self, rain_1d, dt):
        """Advance per-cell Green-Ampt cumulative infiltration F [m]."""
        if self._infiltration != 'green_ampt':
            return
        xp  = self._xp
        f_p = self._ga_ksat * (1.0 + self._ga_psi * self._ga_dtheta0
                               / xp.maximum(self._ga_F, self._GA_F_FLOOR))
        f = xp.minimum(rain_1d, f_p)          # infiltration rate on pervious soil
        self._ga_F = self._ga_F + f * dt

    def _divide_infiltration(self, rain_1d):
        """
        Effective infiltration rate [m/s] feeding each zone's sandbox at its
        divide cell.  With Green-Ampt, rainfall is capped by the local
        infiltration capacity and scaled by the pervious fraction (the
        impervious fraction sheds water and does not recharge the water table).
        Returns a (n_polygons,) array indexed by polygon_divide_idx.
        """
        xp    = self._xp
        idx   = self._polygon_divide_idx
        P_div = rain_1d[idx]
        if self._infiltration != 'green_ampt':
            return P_div
        f_p = self._ga_ksat[idx] * (1.0 + self._ga_psi * self._ga_dtheta0[idx]
                                    / xp.maximum(self._ga_F[idx], self._GA_F_FLOOR))
        f_div = xp.minimum(P_div, f_p)
        return (1.0 - self._imperv_1d[idx]) * f_div

    def _update_opm_sandbox_per_polygon(self, rain_1d, dt):
        """
        Per-polygon sandbox update — fully vectorised over zones (numpy / cupy).

        Each precipitation zone has its own divide cell, saturated-zone thickness
        z[p], SD_max[p] and threshold area A_t[p].  Only the *infiltrated* depth
        at the divide raises z (Green-Ampt); the VSA mask is then rebuilt from
        per-cell polygon assignments.
        """
        xp    = self._xp
        f_div = self._divide_infiltration(rain_1d)             # (n_polygons,)

        q_b = (self._ksat_ms * self._polygon_slope_divide
               * self._opm_z * self._cell_size)
        dV  = (f_div * self._cell_area - q_b) * dt
        dz  = dV / (self._cell_area * self._phi)
        self._opm_z = xp.maximum(0.0, self._opm_z + dz)

        self._opm_SD_max = xp.maximum(self._sd_min,
                                      self._SD_max_initial - self._opm_z)

        Rf_t  = self._sd_min / self._opm_SD_max
        denom = self._opm_H_a - xp.log(Rf_t)
        # Guard the near-zero denominator before dividing (xp.where evaluates
        # both branches, so the divisor must be finite even where unused).
        denom_safe = xp.where(xp.abs(denom) < 1e-12, 1.0, denom)
        new_A_t    = xp.where(xp.abs(denom) < 1e-12, self._opm_A_t_init,
                              self._opm_H_a * self._opm_A_1 / denom_safe)
        self._opm_A_t = xp.clip(new_A_t, self._opm_A_1, self._opm_A_outlet)

        # Vectorised VSA mask rebuild: each cell uses its polygon's A_t
        A_t_per_cell   = self._opm_A_t[self._cell_polygon]
        self._vsa_mask = self._upslope_area > A_t_per_cell

    def _update_opm_sandbox_single(self, rain_1d, dt):
        """
        Single-sandbox update (original Pradhan & Ogden 2010, Eq 12), with
        optional Green-Ampt limiting of the divide-cell infiltration.
        """
        di    = self._divide_cell
        P_div = float(rain_1d[di])
        if self._infiltration == 'green_ampt':
            F_d  = float(self._ga_F[di])
            kv   = float(self._ga_ksat[di])
            dth0 = float(self._ga_dtheta0[di])
            f_p  = kv * (1.0 + self._ga_psi * dth0 / max(F_d, self._GA_F_FLOOR))
            f_div = (1.0 - float(self._imperv_1d[di])) * min(P_div, f_p)
        else:
            f_div = P_div

        q_b = self._ksat_ms * self._slope_divide * self._opm_z * self._cell_size
        dV  = (f_div * self._cell_area - q_b) * dt
        dz  = dV / (self._cell_area * self._phi)
        self._opm_z = max(0.0, self._opm_z + dz)

        self._opm_SD_max = max(self._sd_min, self._SD_max_initial - self._opm_z)

        Rf_t  = self._sd_min / self._opm_SD_max
        denom = self._opm_H_a - np.log(Rf_t)
        if abs(denom) < 1e-12:
            new_A_t = self._opm_A_t_init
        else:
            new_A_t = self._opm_H_a * self._opm_A_1 / denom

        self._opm_A_t = float(np.clip(new_A_t, self._opm_A_1, self._opm_A_outlet))
        self._vsa_mask = self._upslope_area > self._opm_A_t
