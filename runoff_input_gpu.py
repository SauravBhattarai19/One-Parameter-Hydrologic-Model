"""
runoff_input_gpu.py
===================
GPU-accelerated RunoffEngine subclass.

Strategy per mode
-----------------
vsa_opm     – _upslope_area and _vsa_mask are already CuPy when constructed:
               the parent _init_vsa_opm reads grid_data['faccum_1d'] which
               is a CuPy array in GPU mode.  _update_opm_sandbox is entirely
               scalar math + one CuPy comparison → no override needed.

scs_cn      – _Ia_1d, _S_1d, _cumrain_mm, _Pe_mm_old, _scs_rate_ms are
               built from rasterio (numpy) in the parent → transfer to GPU
               in _gpu_transfer; override _scs_formula and _update_scs_cn
               to use cp.where / cp.maximum.

coefficient – _Cf_1d built from rasterio (numpy) → transfer to GPU.

raster      – raster_cache values are numpy → transfer each slice to GPU.

none        – stateless; nothing to transfer.

Overridden methods
------------------
_gpu_transfer    – moves numpy state arrays to the GPU after parent __init__
_scs_formula     – replace np.where  with cp.where
_update_scs_cn   – replace np.maximum / np.zeros  with cp equivalents
get_effective_2d – GPU → CPU transfer before the 2-D write-back
"""

import math

import numpy as np
import cupy as cp

import gpu_utils
from runoff_input import RunoffEngine


class RunoffEngineGPU(RunoffEngine):
    """
    Drop-in GPU replacement for RunoffEngine.

    The parent __init__ builds all state on CPU (reads rasters, computes
    scalars).  _gpu_transfer() then moves the relevant arrays to the GPU.
    """

    def __init__(self, cfg, grid_data):
        # Build everything on CPU first.
        # For vsa_opm mode, _init_vsa_opm reads grid_data['faccum_1d'] and
        # grid_data['slope_1d'] — which are already CuPy arrays in GPU mode,
        # so _upslope_area and _vsa_mask come out as CuPy automatically.
        super().__init__(cfg, grid_data)
        self._gpu_transfer()

    # ── GPU transfer ─────────────────────────────────────────────────────────

    def _gpu_transfer(self):
        """Transfer numpy state arrays to GPU.  Skips already-CuPy arrays."""
        mode = self._mode

        if mode == 'scs_cn':
            self._Ia_1d       = cp.asarray(self._Ia_1d)
            self._S_1d        = cp.asarray(self._S_1d)
            self._cumrain_mm  = cp.asarray(self._cumrain_mm)
            self._Pe_mm_old   = cp.asarray(self._Pe_mm_old)
            self._scs_rate_ms = cp.asarray(self._scs_rate_ms)

        elif mode == 'coefficient':
            self._Cf_1d = cp.asarray(self._Cf_1d)

        elif mode == 'raster':
            # Transfer each cached raster slice to GPU
            self._raster_cache = {
                t: cp.asarray(arr)
                for t, arr in self._raster_cache.items()
            }

        elif mode == 'vsa_opm':
            # _upslope_area and _vsa_mask are already CuPy (built from CuPy
            # faccum_1d in grid_data).  Per-polygon mode needs cell_polygon
            # on GPU for vectorised mask rebuild.
            if getattr(self, '_per_polygon', False):
                self._cell_polygon = cp.asarray(self._cell_polygon)

        # 'none': stateless, nothing to do.

    # ── Overrides for SCS-CN (require cp.* ops) ───────────────────────────────

    def _scs_formula(self, P_mm):
        """SCS-CN accumulated effective rainfall [mm] — CuPy version."""
        excess = P_mm - self._Ia_1d
        return cp.where(
            excess > 0,
            (excess ** 2) / (excess + self._S_1d),
            0.0,
        )

    def _update_scs_cn(self, rain_1d, dt):
        """Advance SCS-CN state — CuPy version."""
        self._cumrain_mm += rain_1d * dt * 1000.0          # m/s → mm
        Pe_new = self._scs_formula(self._cumrain_mm)
        delta  = cp.maximum(Pe_new - self._Pe_mm_old, 0.0) # [mm] this step
        self._scs_rate_ms = (delta / 1000.0) / dt if dt > 0 \
            else cp.zeros(self._n_cells, dtype=np.float64)  # [m/s]
        self._Pe_mm_old = Pe_new

    # ── Per-polygon sandbox override (GPU) ──────────────────────────────────

    def _update_opm_sandbox_per_polygon(self, rain_1d, dt):
        """
        Per-polygon sandbox update — GPU version.

        The per-polygon loop is tiny (n_gauges iterations) so it runs with
        scalar Python math.  rain_1d is a CuPy array; we extract scalars
        via .item().  After the loop, sync A_t to GPU and rebuild the mask.
        """
        for p in range(self._n_polygons):
            P_div = float(rain_1d[self._polygon_divide_idx[p]].item())

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
            denom = H_a_p - math.log(Rf_t)
            if abs(denom) < 1e-12:
                self._opm_A_t[p] = self._opm_A_t_init
            else:
                new_A_t = H_a_p * self._opm_A_1 / denom
                self._opm_A_t[p] = max(self._opm_A_1,
                                       min(new_A_t, self._opm_A_outlet))

        # Sync updated A_t to GPU and rebuild VSA mask
        A_t_gpu        = cp.asarray(self._opm_A_t)
        A_t_per_cell   = A_t_gpu[self._cell_polygon]
        self._vsa_mask = self._upslope_area > A_t_per_cell

    # ── 2-D output override (GPU → CPU) ──────────────────────────────────────

    def get_effective_2d(self, t_seconds, rain_1d):
        """
        2-D runoff map (nrows × ncols), NaN outside watershed.
        Returns a NumPy array (for animation / plotting scripts).
        """
        eff_1d_cpu = gpu_utils.to_cpu(
            self.get_effective_1d(t_seconds, rain_1d)
        )
        out = np.full((self._nrows, self._ncols), np.nan, dtype=np.float64)
        out[self._s_rows, self._s_cols] = eff_1d_cpu
        return out
