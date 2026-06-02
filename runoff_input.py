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

# ── OPM physical constants (Pradhan & Ogden 2010) ────────────────────────────
_OPM_SD_MIN = 0.001          # m   minimum saturation deficit
_OPM_Q_MIN  = 0.001          # m³/s  minimum discharge (Eq 10)


def _resolve_ksat(cfg, cell_size, gauge_csv, gee_result):
    """
    Resolve K_sat from config or HiHydroSoil via GEE.

    Returns (ksat_ms, ksat_per_polygon_ms_or_None).
    """
    ksat_manual = float(getattr(cfg, 'OPM_K_SAT', 44.0)) / 86400.0
    ksat_source = getattr(cfg, 'OPM_KSAT_SOURCE', 'manual').lower()

    if ksat_source != 'gee' or gee_result is None:
        return ksat_manual, None

    ksat_gee = gee_result.get('ksat_m_day')
    ksat_ms = ksat_gee / 86400.0 if ksat_gee is not None else ksat_manual

    ksat_pp = gee_result.get('ksat_per_polygon')
    ksat_pp_ms = [v / 86400.0 for v in ksat_pp] if ksat_pp else None

    return ksat_ms, ksat_pp_ms


def _resolve_sd_params(cfg, cell_size):
    """
    Resolve OPM soil parameters from config (manual) or GEE.

    Returns dict with keys: sd_max, sd_min, phi, sd_max_per_polygon,
    ksat_ms, ksat_per_polygon_ms.
    """
    sd_source   = getattr(cfg, 'OPM_SD_SOURCE', 'manual').lower()
    ksat_source = getattr(cfg, 'OPM_KSAT_SOURCE', 'manual').lower()
    ksat_manual = float(getattr(cfg, 'OPM_K_SAT', 44.0)) / 86400.0

    _manual_params = {
        'sd_max': float(cfg.OPM_SD_MAX_INITIAL),
        'sd_min': _OPM_SD_MIN,
        'phi': float(getattr(cfg, 'OPM_PHI', 0.10)),
        'sd_max_per_polygon': None,
        'ksat_ms': ksat_manual,
        'ksat_per_polygon_ms': None,
    }

    needs_gee = sd_source == 'gee' or ksat_source == 'gee'
    if not needs_gee:
        return _manual_params

    target_date = getattr(cfg, 'SERVES_TARGET_DATE', None)
    if target_date is None and sd_source == 'gee':
        print("  [WARN] SERVES_TARGET_DATE not set; using manual SD/phi values.")

    try:
        from serves_gee import compute_opm_params
    except ImportError:
        print("  [WARN] earthengine-api not installed; using manual values.")
        return _manual_params

    gauge_csv = getattr(cfg, 'PRECIP_GAUGE_FILE', None)
    precip_method = getattr(cfg, 'PRECIP_METHOD', 'uniform').lower()
    if precip_method not in ('thiessen', 'idw'):
        gauge_csv = None

    gee_result = compute_opm_params(
        watershed_geojson_path=getattr(cfg, 'OPM_WATERSHED_GEOJSON',
                                       'output/watershed.geojson'),
        cell_size=cell_size,
        lookup_csv_path=getattr(cfg, 'LULC_LOOKUP_CSV', 'lulc_lookup.csv'),
        target_date=target_date,
        satellite=getattr(cfg, 'SERVES_SATELLITE', 'landsat'),
        search_window=getattr(cfg, 'SERVES_SEARCH_WINDOW', 16),
        soil_depth_band=getattr(cfg, 'OPM_SOILGRIDS_DEPTH', 'b30'),
        project=getattr(cfg, 'GEE_PROJECT', None),
        gauge_csv_path=gauge_csv,
        target_crs=getattr(cfg, 'TARGET_CRS_EPSG', 'EPSG:32645'),
    )

    # ── SD / phi: from GEE or manual ─────────────────────────────────────
    if sd_source == 'gee' and gee_result is not None and target_date is not None:
        sd_max  = gee_result['sd_max']
        phi     = gee_result['phi']
        per_poly = gee_result.get('sd_max_per_polygon')
    else:
        sd_max  = float(cfg.OPM_SD_MAX_INITIAL)
        phi     = float(getattr(cfg, 'OPM_PHI', 0.10))
        per_poly = None

    # ── K_sat: from GEE or manual ────────────────────────────────────────
    ksat_ms, ksat_pp_ms = _resolve_ksat(cfg, cell_size, gauge_csv, gee_result)

    # ── Print diagnostics ────────────────────────────────────────────────
    print(f"  OPM params    |  SD_max={sd_max:.4f} m  (source={sd_source})"
          f"  phi={phi:.4f}"
          f"  K_sat={ksat_ms * 86400:.2f} m/day (source={ksat_source})")
    if gee_result is not None:
        print(f"                |  theta=[{gee_result.get('theta_min', 0):.3f},"
              f" {gee_result.get('theta_max', 0):.3f}]"
              f"  Z_r_mean={gee_result.get('root_depth_mean', 0):.2f} m")
    if per_poly:
        print(f"                |  Per-polygon SD_max: "
              f"{[f'{v:.3f}' for v in per_poly]}")
    if ksat_pp_ms:
        ksat_pp_mday = [v * 86400 for v in ksat_pp_ms]
        print(f"                |  Per-polygon K_sat:  "
              f"{[f'{v:.2f}' for v in ksat_pp_mday]} m/day")

    return {
        'sd_max': sd_max,
        'sd_min': _OPM_SD_MIN,
        'phi': phi,
        'sd_max_per_polygon': per_poly,
        'ksat_ms': ksat_ms,
        'ksat_per_polygon_ms': ksat_pp_ms,
    }


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
            return rain_1d * self._vsa_mask.astype(np.float64)

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
            return bool(self._vsa_mask.any())
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
        sd_max_per_poly = params['sd_max_per_polygon']
        ksat_ms         = params['ksat_ms']
        ksat_pp_ms      = params['ksat_per_polygon_ms']

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

        # ── Per-polygon vs single-sandbox branching ───────────────────────────
        cell_polygon = grid_data.get('cell_polygon')
        use_per_polygon = getattr(cfg, 'OPM_PER_POLYGON', True)

        if cell_polygon is not None and use_per_polygon:
            # ── Per-polygon mode ──────────────────────────────────────────────
            # Each precipitation zone (nearest-gauge region) gets its own
            # sandbox state so that spatially variable rainfall drives local
            # VSA expansion independently.
            cell_polygon = np.asarray(cell_polygon).ravel()
            n_polygons   = int(cell_polygon.max()) + 1

            # Ensure NumPy for the init loop (faccum_1d / slope_1d may be CuPy)
            _to_np = lambda a: a.get() if hasattr(a, 'get') else np.asarray(a)
            faccum_np = _to_np(faccum_1d)
            slope_np  = _to_np(slope_1d)

            divide_idx     = np.empty(n_polygons, dtype=np.intp)
            slope_divide   = np.empty(n_polygons, dtype=np.float64)

            for p in range(n_polygons):
                local_idx = np.where(cell_polygon == p)[0]
                best      = local_idx[faccum_np[local_idx].argmin()]
                divide_idx[p]   = best
                slope_divide[p] = float(slope_np[best])

            self._per_polygon          = True
            self._n_polygons           = n_polygons
            self._cell_polygon         = cell_polygon
            self._polygon_divide_idx   = divide_idx
            self._polygon_slope_divide = slope_divide

            # Per-polygon SD_max_initial and H_a
            if (sd_max_per_poly is not None
                    and len(sd_max_per_poly) == n_polygons):
                sd_init_arr = np.array(sd_max_per_poly, dtype=np.float64)
            else:
                sd_init_arr = np.full(n_polygons, SD_max_initial,
                                      dtype=np.float64)
            self._SD_max_initial = sd_init_arr

            # Per-polygon K_sat
            if (ksat_pp_ms is not None
                    and len(ksat_pp_ms) == n_polygons):
                self._ksat_ms = np.array(ksat_pp_ms, dtype=np.float64)
            else:
                self._ksat_ms = np.full(n_polygons, ksat_ms,
                                         dtype=np.float64)

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
        else:
            # ── Single-sandbox mode (uniform rainfall) ────────────────────────
            self._per_polygon  = False
            self._slope_divide = float(slope_1d[0])

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

    def _update_opm_sandbox_single(self, rain_1d, dt):
        """
        Single-sandbox update (original Pradhan & Ogden 2010, Eq 12).

        Rainfall at the catchment divide (index 0) drives the water balance.
        """
        P_divide = float(rain_1d[0])

        q_b = self._ksat_ms * self._slope_divide * self._opm_z * self._cell_size
        dV  = (P_divide * self._cell_area - q_b) * dt
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

    def _update_opm_sandbox_per_polygon(self, rain_1d, dt):
        """
        Per-polygon sandbox update.

        Each precipitation zone has its own divide cell, saturated zone
        thickness z[p], SD_max[p], and threshold area A_t[p].  The VSA mask
        is rebuilt from per-cell polygon assignments after all zones update.
        """
        for p in range(self._n_polygons):
            P_div = float(rain_1d[self._polygon_divide_idx[p]])

            z_p = self._opm_z[p]
            q_b = self._ksat_ms[p] * self._polygon_slope_divide[p] * z_p * self._cell_size
            dV  = (P_div * self._cell_area - q_b) * dt
            dz  = dV / (self._cell_area * self._phi)
            z_p = max(0.0, z_p + dz)
            self._opm_z[p] = z_p

            SD_max_p = max(self._sd_min, self._SD_max_initial[p] - z_p)
            self._opm_SD_max[p] = SD_max_p

            Rf_t  = self._sd_min / SD_max_p
            H_a_p = self._opm_H_a[p]
            denom = H_a_p - np.log(Rf_t)
            if abs(denom) < 1e-12:
                self._opm_A_t[p] = self._opm_A_t_init
            else:
                new_A_t = H_a_p * self._opm_A_1 / denom
                self._opm_A_t[p] = float(np.clip(new_A_t, self._opm_A_1,
                                                  self._opm_A_outlet))

        # Vectorised VSA mask rebuild: each cell uses its polygon's A_t
        A_t_per_cell   = self._opm_A_t[self._cell_polygon]
        self._vsa_mask = self._upslope_area > A_t_per_cell
