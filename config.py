# config.py

# ─────────────────────────────────────────────────────────────────────────────
# PRE-PROCESSING  (existing)
# ─────────────────────────────────────────────────────────────────────────────

# Path to the Digital Elevation Model (DEM) file
DEM_PATH = "./dem.tif"

# Target Coordinate Reference System (CRS) for reprojection.
TARGET_CRS_EPSG = "EPSG:32645"

# Output point for watershed delineation (latitude, longitude)
OUTPUT_POINT = (27.632222, 85.293333)  # (lat, lon)

# Output directory for all generated files
OUTPUT_DIR = "output/"

# ─────────────────────────────────────────────────────────────────────────────
# KINEMATIC WAVE ROUTING  (new)
# ─────────────────────────────────────────────────────────────────────────────

# --- Input raster paths (outputs from process_dem.py) ---
ROUTING_DEM_PATH            = "output/clipped_dem.tif"
ROUTING_FLOW_DIR_PATH       = "output/flow_direction.tif"
ROUTING_FLOW_ACCUM_PATH     = "output/clipped_flow_accumulation.tif"
ROUTING_WATERSHED_MASK_PATH = "output/watershed.tif"

# --- Grid geometry ---
# Pixel size of the routing rasters in metres.
# None  → auto-detected from ROUTING_DEM_PATH at run-time (recommended).
# Set a float to override (e.g. if you want to document the exact value).
CELL_SIZE = None

# --- Manning's roughness ---
# Representative n for the overland/channel surface.
# Typical values: 0.04–0.10 for natural channels / hillslopes.
MANNINGS_N = 0.09

# --- Time stepping ---
# Δt must satisfy the Courant criterion: Δt ≤ CELL_SIZE / V_max
# Start conservatively (e.g., 30–60 s) and decrease if the model goes unstable.
TIME_STEP_SECONDS = 5           # seconds

# Total length of the simulation
TOTAL_SIMULATION_TIME_HOURS = 144  # hours  (FLOOD_03: 2024-09-26 to 2024-10-01)

# --- Output write interval ---
# How often to record a row in hydrograph.csv (seconds of simulation time).
# Should be >= TIME_STEP_SECONDS. With small dt (e.g. 5 s) writing every step
# produces millions of rows — set this to 300 s (5 min) or 3600 s (1 hr).
# None → write every time step (only sensible for large dt).
OUTPUT_INTERVAL_SECONDS = 600   # seconds  (5-minute hydrograph output)

# --- Rainfall input ---
# Uniform rainfall intensity applied over the whole watershed
RAIN_INTENSITY_MM_HR = 20.0         # mm / hour
RAIN_DURATION_HOURS  = 3.0          # hours  (rainfall stops after this; remainder is recession)

# --- Modular precipitation ---
# Method: 'uniform'  → uses RAIN_INTENSITY_MM_HR / RAIN_DURATION_HOURS above
#          'thiessen' → Voronoi nearest-gauge spatial weighting
#          'idw'      → Inverse Distance Weighting (exponent = PRECIP_IDW_POWER)
# When using 'thiessen' or 'idw' the rainfall comes from the CSV files below.
PRECIP_METHOD          = 'thiessen'
PRECIP_GAUGE_FILE      = "test_data/opm_format/FLOOD_03/gauges.csv"
PRECIP_TIMESERIES_FILE = "test_data/opm_format/FLOOD_03/timeseries.csv"
PRECIP_IDW_POWER       = 2.0        # IDW distance exponent (p=2 is standard)

# ── Runoff generation engine ─────────────────────────────────────────────────
# Controls whether a runoff-generation model transforms rainfall into effective
# runoff before it enters the kinematic-wave router.
#   'none'        → all rainfall is direct runoff (default, backward compatible)
#   'coefficient' → multiply by static spatial Cf raster (0–1)
#   'raster'      → pre-computed runoff raster time series
#   'scs_cn'      → SCS Curve Number: per-cell CN raster + cumulative rainfall
#   'vsa_opm'     → Variable Source Area: Pradhan & Ogden (2010) OPM
RUNOFF_SOURCE           = 'vsa_opm'
RUNOFF_COEFFICIENT_PATH = "runoff/runoff_coefficient.tif"
RUNOFF_RASTER_MANIFEST  = "runoff/manifest.csv"
RUNOFF_CN_PATH          = "runoff/curve_number.tif"
RUNOFF_SCS_Ia_FACTOR    = 0.2       # SCS initial abstraction ratio (standard 0.2)

# ── OPM / VSA parameters (used when RUNOFF_SOURCE = 'vsa_opm') ──────────────
# OPM_SD_MAX_INITIAL: initial unsaturated-zone storage capacity at the
#   catchment divide [m].  Represents max soil moisture deficit from field
#   capacity.  Typical range: 0.05–0.10 m (shallow soils) to 0.20–0.30 m
#   (deep forest soils).
OPM_SD_MAX_INITIAL = 0.10    # m

# OPM_Q_MAX: observed baseflow / initial discharge at the outlet [m³/s].
#   Used in Eq 10 (Pradhan & Ogden 2010) to calibrate the initial threshold
#   area A_t.  Must be > 0.001 m³/s (the model's Q_min constant).
OPM_Q_MAX          = 0.50    # m³/s  — set to observed pre-storm discharge

# OPM_PHI: drainable porosity (specific yield) of the soil [-].
#   Relates volume change of groundwater to saturated-zone thickness change.
#   Typical values: sandy soils 0.25–0.35; loamy soils 0.15–0.25; clay 0.05–0.15.
OPM_PHI            = 0.35    # dimensionless

# OPM_K_SAT: saturated hydraulic conductivity of the soil [m/day].
#   Used in Darcy lateral drainage at the catchment divide (sandbox water
#   balance, Eq 12).  Assumed uniform over the catchment.
#   Typical values: clay 0.01–0.1; loam 0.1–1; sandy loam 1–10; sand 10–100.
OPM_K_SAT          = 44.0   # m/day  (≈ sandy loam / gravelly soil)

# --- Numerical stability / physical limits ---
# Minimum slope used in Manning's equation to avoid division-by-zero.
# At 1e-5, a single flat cell (dx≈94 m) takes ~30 min to drain 1 m of water;
# a 10-cell flat path = 5 h pooling delay → artificial recession surge at t≈14 h.
# 1e-4 (10 cm per 1 km) is a defensible physical floor for Himalayan valleys
# and cuts per-cell drainage time to ~9 min.
MIN_SLOPE       = 1e-4              # m/m

# Minimum water depth kept to avoid numerical issues (wet/dry front treatment)
MIN_DEPTH_M     = 1e-6              # metres

# NOTE: MAX_DEPTH_M (depth ceiling) has been removed from the router.
# Capping depth before Manning's equation freezes Q_out at a fixed value while
# volume keeps growing → permanent flat plateau in the hydrograph.
# The flux limiter (Q_out ≤ volume/dt) is the correct stability mechanism;
# deeper cells simply produce larger Q_out and drain faster (self-correcting).
# MAX_DEPTH_M is kept here only for the animation colour-scale display.
MAX_DEPTH_M     = 10.0             # metres  (display use only)

# --- Output ---
HYDROGRAPH_CSV  = "output/hydrograph.csv"  # Time-series Q at the outlet

# ── Backend selection ─────────────────────────────────────────────────────────
# 'cpu' → NumPy only (default; always works, no extra dependencies)
# 'gpu' → CuPy/CUDA acceleration (falls back to CPU with a warning if CuPy is
#          not installed or no CUDA GPU is found)
BACKEND = 'cpu'

# Floating-point precision for GPU state arrays.
#   'float64' → full double precision; results match CPU bit-for-bit (default)
#   'float32' → halves GPU memory footprint and memory-bandwidth cost;
#               introduces ~1e-7 relative error per Manning step — validate
#               hydrograph output before enabling in production runs.
GPU_PRECISION = 'float64'

