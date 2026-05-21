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
_OPM_K_MS   = 44.0 / 86400.0  # m/s  saturated hydraulic conductivity (44 m/day)


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
        Keys: z_m, SD_max_t, A_t_m2, VSA_m2, VSA_fraction.
        """
        if self._mode != 'vsa_opm':
            return {}
        return {
            "z_m":         self._opm_z,
            "SD_max_t":    self._opm_SD_max,
            "A_t_m2":      self._opm_A_t,
            "VSA_m2":      float(self._vsa_mask.sum()) * self._cell_area,
            "VSA_fraction": float(self._vsa_mask.sum()) / self._n_cells,
        }

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
        self._slope_divide = float(slope_1d[0])   # divide cell = index 0 (min faccum)

        # Upslope contributing area per cell [m²] — computed once
        self._upslope_area = faccum_1d * cell_area   # (n_cells,) [m²]

        # ── OPM scalars ───────────────────────────────────────────────────────
        SD_max_initial = float(cfg.OPM_SD_MAX_INITIAL)
        phi            = float(getattr(cfg, 'OPM_PHI', 0.35))

        self._SD_max_initial = SD_max_initial
        self._phi            = phi

        # A_1: upslope area of a single divide cell [m²]
        A_1 = cell_area

        # A_outlet: total catchment area [m²]
        # faccum_1d[-1] == faccum[outlet_rc] because topological_order puts
        # the outlet last (highest accumulation).
        A_outlet = float(faccum_1d[-1]) * cell_area

        # Eq 10 — initial threshold contributing area from single discharge measurement
        # A_t_init = A_outlet / (1 - ln(Q_min / Q_max))
        # Derivation in Pradhan & Ogden (2010) implicitly sets H_a=1 for this equation.
        A_t_init = A_outlet / (1.0 - np.log(_OPM_Q_MIN / Q_max))

        # Initial range factor
        Rf_init = _OPM_SD_MIN / SD_max_initial

        # Eq 4 — scaling factor H_a (constant for all timesteps)
        # H_a = (A_t_init / (A_t_init - A_1)) * ln(Rf_init)
        # H_a < 0 since ln(Rf_init) < 0 (Rf_init < 1)
        # For large watersheds (A_t_init >> A_1): H_a ≈ ln(Rf_init)
        # Self-consistency: plugging A_t_init and Rf_init into Eq 5 recovers A_t_init exactly.
        ratio = A_t_init / (A_t_init - A_1)
        H_a   = ratio * np.log(Rf_init)   # negative

        # Store for dynamic updates
        self._opm_A_1      = A_1
        self._opm_A_outlet = A_outlet
        self._opm_A_t_init = A_t_init
        self._opm_H_a      = H_a

        # State variables (updated each timestep by _update_opm_sandbox)
        self._opm_z      = 0.0            # saturated zone thickness [m]
        self._opm_SD_max = SD_max_initial  # current max deficit [m]
        self._opm_A_t    = A_t_init       # current threshold area [m²]

        # Initial VSA mask: cells with upslope_area > A_t_init
        self._vsa_mask = self._upslope_area > A_t_init

        # Extensibility hook: records which A_t initialisation method was used.
        # Future options: 'single_measurement', 'regression', 'baseflow_recession'.
        self._at_method = 'single_measurement'

        print(f"  OPM           |  A_outlet={A_outlet:.3e} m²"
              f"  A_t_init={A_t_init:.3e} m²")
        print(f"                |  H_a={H_a:.4f}  SD_max_init={SD_max_initial} m"
              f"  phi={phi}  Q_max={Q_max} m³/s")
        print(f"                |  Initial VSA={self._vsa_mask.sum():,} cells"
              f" ({100*self._vsa_mask.mean():.1f}% of watershed)")

    def _update_opm_sandbox(self, rain_1d, dt):
        """
        Advance OPM sandbox state by one timestep.

        Physics (Pradhan & Ogden 2010, Eq 12):
          - Rainfall infiltrates at the catchment divide (zero upstream area).
          - Water table z rises under precipitation; lateral Darcy flow drains it.
          - SD_max(t) = SD_max(1) - z(t)   [deficit decreases as table rises]
          - A_t(t) from Eq 5 decreases as SD_max decreases → VSA expands.

        Units:
          P_divide  [m/s]
          q_b = K * i * z * b  →  [m/s][m/m][m][m] = [m³/s]
          dV  [m³],  dz [m]
        """
        P_divide = float(rain_1d[0])   # m/s — rainfall at divide cell (index 0)

        # Darcy lateral outflow from saturated zone
        q_b = _OPM_K_MS * self._slope_divide * self._opm_z * self._cell_size  # [m³/s]

        # Net volume change in divide cell
        dV  = (P_divide * self._cell_area - q_b) * dt                         # [m³]

        # Update saturated zone thickness (bounded below by 0)
        dz            = dV / (self._cell_area * self._phi)
        self._opm_z   = max(0.0, self._opm_z + dz)

        # Eq 12: updated maximum deficit (bounded below by SD_min)
        self._opm_SD_max = max(_OPM_SD_MIN, self._SD_max_initial - self._opm_z)

        # Dynamic range factor
        Rf_t  = _OPM_SD_MIN / self._opm_SD_max

        # Eq 5: updated threshold contributing area
        # A_t(t) = H_a * A_1 / (H_a - ln(Rf_t))
        # As rain falls: SD_max ↓ → Rf_t ↑ → |ln(Rf_t)| ↓ → |denom| ↑ → A_t ↓ → VSA ↑
        denom = self._opm_H_a - np.log(Rf_t)
        if abs(denom) < 1e-12:
            # Singularity guard: occurs only when Rf_t == Rf_init (at t=0).
            # The initial condition A_t_init is already set; no update needed.
            new_A_t = self._opm_A_t_init
        else:
            new_A_t = self._opm_H_a * self._opm_A_1 / denom

        # Physical bounds: A_t in [A_1, A_outlet]
        self._opm_A_t = float(np.clip(new_A_t, self._opm_A_1, self._opm_A_outlet))

        # Recompute VSA mask for the NEXT get_effective_1d call
        self._vsa_mask = self._upslope_area > self._opm_A_t
