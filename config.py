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
# Spatial resolution of the reprojected DEM in metres.
# Must match the pixel size of the rasters above.
CELL_SIZE = 93.94351273152283  # metres  (adjust to actual pixel size if different)

# --- Manning's roughness ---
# Representative n for the overland/channel surface.
# Typical values: 0.04–0.10 for natural channels / hillslopes.
MANNINGS_N = 0.06

# --- Time stepping ---
# Δt must satisfy the Courant criterion: Δt ≤ CELL_SIZE / V_max
# Start conservatively (e.g., 30–60 s) and decrease if the model goes unstable.
TIME_STEP_SECONDS = 30.0            # seconds

# Total length of the simulation
TOTAL_SIMULATION_TIME_HOURS = 36   # hours

# --- Rainfall input ---
# Uniform rainfall intensity applied over the whole watershed
RAIN_INTENSITY_MM_HR = 20.0         # mm / hour
RAIN_DURATION_HOURS  = 3.0          # hours  (rainfall stops after this; remainder is recession)

# --- Numerical stability / physical limits ---
# Minimum slope used in Manning's equation to avoid division-by-zero
MIN_SLOPE       = 1e-5              # m/m

# Minimum water depth kept to avoid numerical issues (wet/dry front treatment)
MIN_DEPTH_M     = 1e-6              # metres

# Maximum allowable depth per cell (safety cap)
MAX_DEPTH_M     = 10.0             # metres

# --- Output ---
HYDROGRAPH_CSV  = "output/hydrograph.csv"  # Time-series Q at the outlet

