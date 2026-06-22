# -*- coding: utf-8 -*-
"""
config_bridge.py
================
Provides OpmConfig — a plain Python object that mirrors every attribute in
config.py but is populated from the plugin UI (or programmatically).

DESIGN RULE: config.py on disk is NEVER read or mutated by the plugin.
The simulation functions (initialise_grid, run_opm, etc.) accept any object
with the required attributes, so OpmConfig is a perfect drop-in.

Usage
-----
    from qgis_plugin.bridge.config_bridge import OpmConfig

    cfg = OpmConfig()
    cfg.DEM_PATH = "/path/to/dem.tif"
    cfg.BACKEND  = "gpu"
    ...
    grid_data = kinematic_wave_router.initialise_grid(cfg)
"""

import os


class OpmConfig:
    """
    Mirror of config.py as a mutable plain object.

    Default values match the shipped config.py so the model runs
    "out of the box" without any UI interaction.  All values can be
    overridden individually from the plugin UI or from Processing
    algorithm parameters.
    """

    # ── Pre-processing ────────────────────────────────────────────────────────
    DEM_PATH: str = ""
    TARGET_CRS_EPSG: str = "EPSG:32645"
    OUTPUT_POINT: tuple = (27.632222, 85.293333)   # (lat, lon)
    OUTPUT_DIR: str = "output/"

    # ── Routing raster paths (set to OUTPUT_DIR outputs after process_dem) ───
    ROUTING_DEM_PATH: str = "output/clipped_dem.tif"
    ROUTING_FLOW_DIR_PATH: str = "output/flow_direction.tif"
    ROUTING_FLOW_ACCUM_PATH: str = "output/clipped_flow_accumulation.tif"
    ROUTING_WATERSHED_MASK_PATH: str = "output/watershed.tif"

    # ── Grid geometry ─────────────────────────────────────────────────────────
    CELL_SIZE = None    # auto-detected from ROUTING_DEM_PATH

    # ── Manning's roughness ───────────────────────────────────────────────────
    MANNINGS_N_SOURCE: str = "scalar"       # 'scalar' | 'lulc' | 'raster'
    MANNINGS_N: float = 0.09
    MANNINGS_N_LULC_PATH: str = "gee"
    MANNINGS_N_RASTER_PATH: str = None
    MANNINGS_N_CHANNEL = {1: 0.10, 2: 0.06, 3: 0.045, 4: 0.035}
    CHANNEL_FACCUM_THRESHOLD: int = None

    # ── Time stepping ────────────────────────────────────────────────────────
    TIME_STEP_SECONDS: int = 5
    TOTAL_SIMULATION_TIME_HOURS: float = 144.0
    OUTPUT_INTERVAL_SECONDS: int = 600

    # ── Uniform rainfall (used when PRECIP_METHOD='uniform') ─────────────────
    RAIN_INTENSITY_MM_HR: float = 20.0
    RAIN_DURATION_HOURS: float = 3.0

    # ── Gauge-based precipitation ─────────────────────────────────────────────
    PRECIP_METHOD: str = "uniform"          # 'uniform' | 'thiessen' | 'idw'
    PRECIP_GAUGE_FILE: str = ""
    PRECIP_TIMESERIES_FILE: str = ""
    PRECIP_IDW_POWER: float = 2.0

    # ── Runoff generation ─────────────────────────────────────────────────────
    RUNOFF_SOURCE: str = "none"            # 'none'|'coefficient'|'raster'|'scs_cn'|'vsa_opm'
    RUNOFF_COEFFICIENT_PATH: str = ""
    RUNOFF_RASTER_MANIFEST: str = ""
    RUNOFF_CN_PATH: str = ""
    RUNOFF_SCS_Ia_FACTOR: float = 0.2

    # ── OPM / VSA parameters ──────────────────────────────────────────────────
    OPM_SD_MAX_INITIAL: float = 1.0    # root zone depth D [m] (physical height)
    OPM_Q_MAX: float = 0.50
    OPM_PHI: float = 0.10              # drainable porosity (porosity − FC)
    OPM_K_SAT: float = 44.0
    OPM_PER_POLYGON: bool = True

    # ── GEE / SERVES integration ──────────────────────────────────────────────
    OPM_SD_SOURCE: str = "manual"          # 'manual' | 'gee'
    OPM_SD_REDUCER: str = "mean"          # 'max' | 'mean'
    LULC_LOOKUP_CSV: str = "lulc_lookup.csv"
    SERVES_TARGET_DATE: str = None
    SERVES_SATELLITE: str = "landsat"
    SERVES_SEARCH_WINDOW: int = 16
    OPM_SOILGRIDS_DEPTH: str = "b30"
    OPM_WATERSHED_GEOJSON: str = "output/watershed.geojson"
    GEE_PROJECT: str = None

    # ── Numerical stability ───────────────────────────────────────────────────
    MIN_SLOPE: float = 1e-4
    MIN_DEPTH_M: float = 1e-6
    MAX_DEPTH_M: float = 10.0      # display use only

    # ── Output ────────────────────────────────────────────────────────────────
    HYDROGRAPH_CSV: str = "output/hydrograph.csv"

    # ── Backend ───────────────────────────────────────────────────────────────
    BACKEND: str = "cpu"           # 'cpu' | 'gpu'
    GPU_PRECISION: str = "float64" # 'float32' | 'float64'

    # ─────────────────────────────────────────────────────────────────────────

    def __init__(self, **kwargs):
        """
        Create an OpmConfig, optionally overriding defaults via keyword args.

        Example
        -------
            cfg = OpmConfig(DEM_PATH="/data/dem.tif", BACKEND="gpu")
        """
        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise AttributeError(
                    f"OpmConfig has no attribute '{key}'.  "
                    f"Check config.py for valid parameter names."
                )
            setattr(self, key, value)

    def update_output_paths(self):
        """
        Convenience: sync ROUTING_* and HYDROGRAPH_CSV to point inside OUTPUT_DIR.
        Call this after setting OUTPUT_DIR.
        """
        d = self.OUTPUT_DIR
        self.ROUTING_DEM_PATH = os.path.join(d, "clipped_dem.tif")
        self.ROUTING_FLOW_DIR_PATH = os.path.join(d, "flow_direction.tif")
        self.ROUTING_FLOW_ACCUM_PATH = os.path.join(d, "clipped_flow_accumulation.tif")
        self.ROUTING_WATERSHED_MASK_PATH = os.path.join(d, "watershed.tif")
        self.HYDROGRAPH_CSV = os.path.join(d, "hydrograph.csv")

    def validate(self):
        """
        Basic sanity checks before starting a run.

        Raises
        ------
        ValueError  with a descriptive message on the first failed check.
        """
        errors = []

        if not self.DEM_PATH or not os.path.exists(self.DEM_PATH):
            errors.append(f"DEM_PATH not found: '{self.DEM_PATH}'")

        if self.TIME_STEP_SECONDS <= 0:
            errors.append(f"TIME_STEP_SECONDS must be > 0 (got {self.TIME_STEP_SECONDS})")

        if self.TOTAL_SIMULATION_TIME_HOURS <= 0:
            errors.append(f"TOTAL_SIMULATION_TIME_HOURS must be > 0")

        if self.MANNINGS_N <= 0:
            errors.append(f"MANNINGS_N must be > 0 (got {self.MANNINGS_N})")

        if self.PRECIP_METHOD in ("thiessen", "idw"):
            if not self.PRECIP_GAUGE_FILE or not os.path.exists(self.PRECIP_GAUGE_FILE):
                errors.append(f"PRECIP_GAUGE_FILE not found: '{self.PRECIP_GAUGE_FILE}'")
            if not self.PRECIP_TIMESERIES_FILE or not os.path.exists(self.PRECIP_TIMESERIES_FILE):
                errors.append(f"PRECIP_TIMESERIES_FILE not found: '{self.PRECIP_TIMESERIES_FILE}'")

        if self.RUNOFF_SOURCE == "vsa_opm":
            if self.OPM_Q_MAX <= 0.001:
                errors.append(f"OPM_Q_MAX must be > 0.001 m³/s (got {self.OPM_Q_MAX})")
            if not (0 < self.OPM_PHI < 1):
                errors.append(f"OPM_PHI must be in (0, 1) (got {self.OPM_PHI})")

        if self.BACKEND not in ("cpu", "gpu"):
            errors.append(f"BACKEND must be 'cpu' or 'gpu' (got '{self.BACKEND}')")

        if errors:
            raise ValueError("OpmConfig validation failed:\n" + "\n".join(f"  • {e}" for e in errors))

    def to_dict(self):
        """Return all public attributes as a plain dict (for logging/saving)."""
        return {
            k: getattr(self, k)
            for k in vars(OpmConfig)
            if not k.startswith("_") and not callable(getattr(OpmConfig, k))
        }

    def __repr__(self):
        items = ", ".join(f"{k}={v!r}" for k, v in self.to_dict().items())
        return f"OpmConfig({items})"
