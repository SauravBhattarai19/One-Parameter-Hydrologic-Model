# -*- coding: utf-8 -*-
"""
config_bridge.py
================
Provides OpmConfig — a plain Python object that mirrors EVERY attribute in
config.py but is populated from the plugin UI (or programmatically).

DESIGN RULE: config.py on disk is NEVER read or mutated by the plugin.
The simulation functions (initialise_grid, run_time_loop, run_opm, …) accept
any object with the required attributes, so OpmConfig is a perfect drop-in.

This mirror is kept in 1:1 sync with config.py.  The section headers below
match config.py so a diff against it is easy.  When a new knob is added to
config.py, add it here (matching default) and — if user-facing — expose it in
the relevant UI tab.

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
    "out of the box".  All values can be overridden individually from the
    plugin UI or from Processing algorithm parameters.
    """

    # ═════════════════════════════════════════════════════════════════════════
    # 1.  EVENT & SCENARIO
    # ═════════════════════════════════════════════════════════════════════════
    DEM_PATH: str = ""
    TARGET_CRS_EPSG: str = "EPSG:32645"
    OUTPUT_POINT: tuple = (27.632222, 85.293333)   # (lat, lon)
    OUTPUT_DIR: str = "output/"

    # EVENT_START_UTC: "YYYY-MM-DD HH:MM" UTC — single source of truth for the
    # event date (SERVES antecedent query + IMERG download window).
    EVENT_START_UTC = None
    TOTAL_SIMULATION_TIME_HOURS: float = 96.0
    IMERG_UTC_OFFSET_HOURS: float = 5.75           # NPT = UTC+5:45

    # Land-cover lookup CSVs (root-zone depth, Manning's n, impervious fraction).
    LULC_LOOKUP_CSV: str = "lulc_lookup.csv"       # ESA WorldCover
    LCZ_LOOKUP_CSV: str = "lcz_lookup.csv"         # WUDAPT LCZ

    # Google Earth Engine cloud project (IMERG / SERVES / LULC / LCZ download).
    GEE_PROJECT = None

    # ═════════════════════════════════════════════════════════════════════════
    # 2.  WATERSHED PRE-PROCESSING OUTPUTS  (written by process_dem.py)
    # ═════════════════════════════════════════════════════════════════════════
    ROUTING_DEM_PATH: str = "output/clipped_dem.tif"
    ROUTING_FLOW_DIR_PATH: str = "output/flow_direction.tif"
    ROUTING_FLOW_ACCUM_PATH: str = "output/clipped_flow_accumulation.tif"
    ROUTING_WATERSHED_MASK_PATH: str = "output/watershed.tif"
    OPM_WATERSHED_GEOJSON: str = "output/watershed.geojson"

    # ═════════════════════════════════════════════════════════════════════════
    # 3.  PRECIPITATION
    # ═════════════════════════════════════════════════════════════════════════
    RAIN_INTENSITY_MM_HR: float = 20.0
    RAIN_DURATION_HOURS: float = 3.0

    # 'uniform' | 'thiessen' | 'idw' | 'imerg_thiessen' | 'imerg_idw'
    PRECIP_METHOD: str = "uniform"
    PRECIP_GAUGE_FILE: str = ""
    PRECIP_TIMESERIES_FILE: str = ""
    PRECIP_IDW_POWER: float = 2.0
    PRECIP_EXCLUDE_OUTSIDE_STATIONS: bool = False

    # IMERG source (used when PRECIP_METHOD='imerg_thiessen'/'imerg_idw').
    IMERG_START_LOCAL = None       # None → auto from EVENT_START_UTC
    IMERG_END_LOCAL = None
    PRECIP_IMERG_DIR: str = "output/imerg/"
    IMERG_DATASET: str = "NASA/GPM_L3/IMERG_V07"
    IMERG_BAND: str = "precipitation"
    PRECIP_IMERG_FORCE_DOWNLOAD: bool = False
    IMERG_BBOX_BUFFER_M: float = 11132.0

    # ═════════════════════════════════════════════════════════════════════════
    # 4.  RUNOFF GENERATION
    # ═════════════════════════════════════════════════════════════════════════
    # 'none'|'coefficient'|'raster'|'scs_cn'|'vsa_opm'
    RUNOFF_SOURCE: str = "none"
    RUNOFF_COEFFICIENT_PATH: str = ""
    RUNOFF_RASTER_MANIFEST: str = ""
    RUNOFF_CN_PATH: str = ""
    RUNOFF_SCS_Ia_FACTOR: float = 0.2

    # ═════════════════════════════════════════════════════════════════════════
    # 5.  OPM / VSA PARAMETERS  (used when RUNOFF_SOURCE='vsa_opm')
    # ═════════════════════════════════════════════════════════════════════════
    # Which runoff-generation mechanisms are active (orthogonal subset).
    #   'vsa' | 'horton' | 'impervious'
    RUNOFF_MECHANISMS = ["vsa", "horton", "impervious"]

    OPM_SD_MAX_INITIAL: float = 0.10   # root zone depth D [m] (physical height)
    OPM_Q_MAX: float = 100.0           # observed baseflow / initial discharge [m³/s]
    OPM_PHI: float = 0.35              # drainable porosity [-]
    OPM_K_SAT: float = 44.0           # lateral saturated conductivity [m/day]
    OPM_PER_POLYGON: bool = True

    # ── Infiltration (Green-Ampt) ────────────────────────────────────────────
    OPM_INFILTRATION: str = "none"          # 'none' | 'green_ampt'
    OPM_GA_SUCTION_SOURCE: str = "scalar"   # 'scalar' | 'texture'
    OPM_GA_SUCTION_M: float = 0.15          # wetting-front suction head ψ [m]
    OPM_GA_KSAT_SOURCE: str = "scalar"      # 'scalar' | 'gee' | 'raster'
    OPM_GA_KSAT_MMHR: float = 12.0          # vertical surface Ksat [mm/hr]
    OPM_GA_KSAT_RASTER = None               # None → auto {OUTPUT_DIR}/ksat_hihydro.tif
    OPM_GA_KSAT_SCALE: float = 1.0

    # ── Impervious fraction (urban shedding) ─────────────────────────────────
    IMPERVIOUS_SOURCE: str = "none"         # 'none'|'lcz'|'lulc'|'raster'
    IMPERVIOUS_RASTER_PATH = None

    # ── Baseflow ─────────────────────────────────────────────────────────────
    OPM_BASEFLOW: bool = False

    # ═════════════════════════════════════════════════════════════════════════
    # 6.  SERVES / GEE SOIL-MOISTURE DEFICIT  (SD_max & phi from satellite)
    # ═════════════════════════════════════════════════════════════════════════
    OPM_SD_SOURCE: str = "manual"           # 'manual' | 'gee'
    OPM_SD_REDUCER: str = "mean"            # 'mean' | 'max' | 'divide'
    OPM_DEFICIT_RASTER = None               # None → auto {OUTPUT_DIR}/deficit_serves_{date}.tif
    SERVES_SATELLITE: str = "landsat"       # 'landsat' | 'sentinel2' | 'modis'
    SERVES_SEARCH_WINDOW: int = 30          # days backward from EVENT_START_UTC
    OPM_SOILGRIDS_DEPTH: str = "b30"        # 'b0' 'b10' 'b30' 'b60' 'b100' 'b200'
    # Legacy / backward-compat (older model builds read this if present).
    SERVES_TARGET_DATE = None

    # ═════════════════════════════════════════════════════════════════════════
    # 7.  MANNING'S ROUGHNESS  (kinematic-wave routing)
    # ═════════════════════════════════════════════════════════════════════════
    MANNINGS_N_SOURCE: str = "scalar"       # 'scalar'|'lulc'|'lcz'|'raster'
    MANNINGS_N: float = 0.09                # uniform fallback / nodata default
    MANNINGS_N_LULC_PATH: str = "gee"       # 'gee' → download ESA WorldCover
    MANNINGS_N_RASTER_PATH = None
    # Channel roughness override for cells above CHANNEL_FACCUM_THRESHOLD.
    # float → uniform channel n | dict{order:n} → per Strahler order | None → off.
    MANNINGS_N_CHANNEL = 0.035
    CHANNEL_FACCUM_THRESHOLD = None         # None → auto (top 1% of cells)

    # ═════════════════════════════════════════════════════════════════════════
    # 8.  GRID & NUMERICAL LIMITS
    # ═════════════════════════════════════════════════════════════════════════
    CELL_SIZE = None                        # None → auto-detect from DEM

    # ── Routing scheme ───────────────────────────────────────────────────────
    ROUTING_SCHEME: str = "kinematic"       # 'kinematic' | 'diffusive'
    DIFFUSION_THETA: float = 1.0            # diffusion weight θ∈[0,1]

    # ── Channel (river) cross-section routing ────────────────────────────────
    CHANNEL_ROUTING: bool = False
    CHANNEL_WIDTH_BY_ORDER = {1: 3.0, 2: 5.0, 3: 8.0, 4: 12.0,
                              5: 18.0, 6: 28.0, 7: 45.0, 8: 70.0}   # m

    # ── Time stepping ────────────────────────────────────────────────────────
    TIME_STEP_SECONDS: float = 2.0
    OUTPUT_INTERVAL_SECONDS: int = 600

    # ── Adaptive CFL timestep ────────────────────────────────────────────────
    ADAPTIVE_TIMESTEP: bool = False
    CFL_TARGET: float = 0.85
    CFL_DT_MAX = 5.0                        # None → OUTPUT_INTERVAL_SECONDS
    CFL_DT_MIN: float = 0.01
    CFL_DT_GROW: float = 1.5

    # ── Numerical floors ─────────────────────────────────────────────────────
    MIN_SLOPE: float = 1e-4
    MIN_DEPTH_M: float = 1e-6
    MAX_DEPTH_M: float = 10.0               # display use only

    # ═════════════════════════════════════════════════════════════════════════
    # 9.  OUTPUTS
    # ═════════════════════════════════════════════════════════════════════════
    HYDROGRAPH_CSV: str = "output/hydrograph.csv"
    MASS_BALANCE_REPORT: bool = True
    MASS_BALANCE_CSV: str = "output/mass_balance.csv"

    # ═════════════════════════════════════════════════════════════════════════
    # 10. COMPUTE BACKEND
    # ═════════════════════════════════════════════════════════════════════════
    BACKEND: str = "cpu"                    # 'cpu' | 'gpu'
    GPU_PRECISION: str = "float64"          # 'float32' | 'float64'

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
        Sync every OUTPUT_DIR-derived path to the current OUTPUT_DIR.
        Call this after setting OUTPUT_DIR.
        """
        d = self.OUTPUT_DIR
        self.ROUTING_DEM_PATH = os.path.join(d, "clipped_dem.tif")
        self.ROUTING_FLOW_DIR_PATH = os.path.join(d, "flow_direction.tif")
        self.ROUTING_FLOW_ACCUM_PATH = os.path.join(d, "clipped_flow_accumulation.tif")
        self.ROUTING_WATERSHED_MASK_PATH = os.path.join(d, "watershed.tif")
        self.OPM_WATERSHED_GEOJSON = os.path.join(d, "watershed.geojson")
        self.PRECIP_IMERG_DIR = os.path.join(d, "imerg/")
        self.HYDROGRAPH_CSV = os.path.join(d, "hydrograph.csv")
        self.MASS_BALANCE_CSV = os.path.join(d, "mass_balance.csv")

    def validate(self):
        """
        Basic sanity checks before starting a run.

        Raises
        ------
        ValueError  with a descriptive message listing all failed checks.
        """
        errors = []

        if not self.DEM_PATH or not os.path.exists(self.DEM_PATH):
            errors.append(f"DEM_PATH not found: '{self.DEM_PATH}'")

        if self.TIME_STEP_SECONDS <= 0:
            errors.append(f"TIME_STEP_SECONDS must be > 0 (got {self.TIME_STEP_SECONDS})")

        if self.TOTAL_SIMULATION_TIME_HOURS <= 0:
            errors.append("TOTAL_SIMULATION_TIME_HOURS must be > 0")

        if self.MANNINGS_N <= 0:
            errors.append(f"MANNINGS_N must be > 0 (got {self.MANNINGS_N})")

        # Gauge CSVs are only required for the file-based interpolation methods.
        if self.PRECIP_METHOD in ("thiessen", "idw"):
            if not self.PRECIP_GAUGE_FILE or not os.path.exists(self.PRECIP_GAUGE_FILE):
                errors.append(f"PRECIP_GAUGE_FILE not found: '{self.PRECIP_GAUGE_FILE}'")
            if not self.PRECIP_TIMESERIES_FILE or not os.path.exists(self.PRECIP_TIMESERIES_FILE):
                errors.append(f"PRECIP_TIMESERIES_FILE not found: '{self.PRECIP_TIMESERIES_FILE}'")

        # IMERG / GEE methods need a project and an event date to derive the window.
        if self.PRECIP_METHOD in ("imerg_thiessen", "imerg_idw"):
            if not (self.GEE_PROJECT or os.environ.get("GEE_PROJECT")):
                errors.append("IMERG precipitation needs GEE_PROJECT (or the GEE_PROJECT env var).")
            if not (self.EVENT_START_UTC or self.IMERG_START_LOCAL):
                errors.append("IMERG precipitation needs EVENT_START_UTC (or IMERG_START_LOCAL).")

        if self.RUNOFF_SOURCE == "vsa_opm":
            if self.OPM_Q_MAX <= 0.001:
                errors.append(f"OPM_Q_MAX must be > 0.001 m³/s (got {self.OPM_Q_MAX})")
            if not (0 < self.OPM_PHI < 1):
                errors.append(f"OPM_PHI must be in (0, 1) (got {self.OPM_PHI})")

        # GEE-backed parameter sources need a project.
        _needs_gee = (
            self.OPM_SD_SOURCE == "gee"
            or self.OPM_GA_KSAT_SOURCE == "gee"
            or self.OPM_GA_SUCTION_SOURCE == "texture"
            or self.MANNINGS_N_SOURCE in ("lulc", "lcz")
            or self.IMPERVIOUS_SOURCE in ("lulc", "lcz")
        )
        if _needs_gee and self.RUNOFF_SOURCE == "vsa_opm":
            if not (self.GEE_PROJECT or os.environ.get("GEE_PROJECT")):
                errors.append(
                    "A GEE-backed option is selected (SERVES SD, gridded Ksat, "
                    "texture suction, or LULC/LCZ Manning's/impervious) but "
                    "GEE_PROJECT is not set."
                )

        if self.BACKEND not in ("cpu", "gpu"):
            errors.append(f"BACKEND must be 'cpu' or 'gpu' (got '{self.BACKEND}')")

        if errors:
            raise ValueError(
                "OpmConfig validation failed:\n" + "\n".join(f"  • {e}" for e in errors)
            )

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
