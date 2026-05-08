"""
tests/02_test_routing_animation.py
===================================
Visualises the kinematic-wave routing model spatially AND temporally by:

  LEFT  panel – 2-D map of water depth [m] over the watershed at each frame
                (draped on a shaded DEM background)
  RIGHT panel – Hydrograph (Q at outlet) building up in real time,
                with a red vertical cursor showing the current time step

The script imports directly from the parent-directory modules
(kinematic_wave_router, routing_utils, config) so it re-runs the
simulation itself rather than reading a pre-existing CSV.

Output:  output/routing_animation.gif
"""

import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")          # no display needed – we write to file
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
from matplotlib.colors import LightSource
import rasterio

# ── Make parent directory importable ──────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
import routing_utils as ru
from kinematic_wave_router import initialise_grid

# ─────────────────────────────────────────────────────────────────────────────
# SETTINGS  (tweak without touching the main model)
# ─────────────────────────────────────────────────────────────────────────────
FRAME_EVERY_N_STEPS = 10        # capture a frame every N time steps
GIF_FPS             = 8         # frames per second in the output GIF
GIF_DPI             = 110       # resolution of each frame
OUTPUT_GIF          = os.path.join(config.OUTPUT_DIR, "routing_animation.gif")

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Initialise grid (load rasters, build topo order, slopes, etc.)
# ─────────────────────────────────────────────────────────────────────────────
print("Initialising routing grid …")
grid_data = initialise_grid(config)

s_rows     = grid_data["s_rows"]
s_cols     = grid_data["s_cols"]
slope_1d   = grid_data["slope_1d"]
ds_idx     = grid_data["ds_idx"]
n_cells    = grid_data["n_cells"]
outlet_pos = grid_data["outlet_pos"]
cell_area  = grid_data["cell_area"]
nrows      = grid_data["nrows"]
ncols      = grid_data["ncols"]
ws_mask    = grid_data["ws_mask"]
dem        = grid_data["dem"]

dx = config.CELL_SIZE
dt = config.TIME_STEP_SECONDS
n  = config.MANNINGS_N
grid_shape = (nrows, ncols)
n_steps    = int(config.TOTAL_SIMULATION_TIME_HOURS * 3600.0 / dt)

print(f"Total steps : {n_steps:,}  |  frames to capture : {n_steps // FRAME_EVERY_N_STEPS}")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Load DEM for shaded-relief background
# ─────────────────────────────────────────────────────────────────────────────
with rasterio.open(config.ROUTING_DEM_PATH) as src:
    dem_bg     = src.read(1).astype(float)
    transform  = src.transform
    nodata_dem = src.nodata

if nodata_dem is not None:
    dem_bg[dem_bg == nodata_dem] = np.nan

# Shaded relief (hillshade) – static background
ls        = LightSource(azdeg=315, altdeg=45)
dem_valid = np.where(np.isnan(dem_bg), 0, dem_bg)
hillshade = ls.hillshade(dem_valid, vert_exag=2, dx=dx, dy=dx)   # [0,1]

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Run the time loop, capturing frames
# ─────────────────────────────────────────────────────────────────────────────
print("Running simulation and capturing frames …")

volume_1d   = np.zeros(n_cells, dtype=np.float64)
Q_out_1d    = np.zeros(n_cells, dtype=np.float64)

# Storage for animation
frame_times_hr  = []   # simulation time [hours] for each captured frame
frame_depths    = []   # 2-D depth arrays, one per frame
hydro_times_hr  = []   # full hydrograph time axis
hydro_Q         = []   # full hydrograph Q values

t_wall = time.time()

for step in range(n_steps):
    t_seconds = step * dt

    # ── Rainfall ────────────────────────────────────────────────────────────
    rain_2d = ru.build_rainfall_array(
        grid_shape, config.RAIN_INTENSITY_MM_HR,
        config.RAIN_DURATION_HOURS, dt, t_seconds
    )
    rain_1d = rain_2d[s_rows, s_cols]

    # ── Inflow from upstream cells (previous step Q_out) ────────────────────
    inflow_1d = np.zeros(n_cells, dtype=np.float64)
    valid_ds  = ds_idx >= 0
    np.add.at(inflow_1d, ds_idx[valid_ds], Q_out_1d[valid_ds])

    # ── Manning's physics ────────────────────────────────────────────────────
    depth_1d    = np.clip(volume_1d / cell_area, config.MIN_DEPTH_M, config.MAX_DEPTH_M)
    velocity_1d = ru.mannings_velocity(depth_1d, slope_1d, n)
    Q_out_1d    = ru.cell_discharge(depth_1d, velocity_1d, dx)
    Q_out_1d    = ru.flux_limiter(Q_out_1d, volume_1d, dt)

    # ── Volume advance ───────────────────────────────────────────────────────
    volume_1d = np.maximum(
        volume_1d + rain_1d * cell_area * dt + inflow_1d * dt - Q_out_1d * dt,
        0.0
    )

    # ── Record hydrograph ────────────────────────────────────────────────────
    t_end_hr = (t_seconds + dt) / 3600.0
    hydro_times_hr.append(t_end_hr)
    hydro_Q.append(Q_out_1d[outlet_pos])

    # ── Capture frame ────────────────────────────────────────────────────────
    if step % FRAME_EVERY_N_STEPS == 0 or step == n_steps - 1:
        # Map 1-D depths back to 2-D spatial grid
        depth_2d            = np.full((nrows, ncols), np.nan)
        depth_2d[s_rows, s_cols] = volume_1d / cell_area
        depth_2d[~ws_mask]  = np.nan   # mask outside watershed

        frame_depths.append(depth_2d.copy())
        frame_times_hr.append(t_end_hr)

        if (step // FRAME_EVERY_N_STEPS) % 10 == 0:
            print(f"  Frame {len(frame_depths):3d}  |  t={t_end_hr:.2f}h  "
                  f"|  Q={hydro_Q[-1]:.3f} m³/s  "
                  f"|  wall={time.time()-t_wall:.1f}s")

print(f"Captured {len(frame_depths)} frames in {time.time()-t_wall:.1f}s")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Build animation
# ─────────────────────────────────────────────────────────────────────────────
print("Rendering animation …")

hydro_times_hr = np.array(hydro_times_hr)
hydro_Q        = np.array(hydro_Q)
n_frames       = len(frame_depths)

# Colour scale: fix to the 99th-percentile depth across all frames
max_depth_vis = max(
    np.nanpercentile(d, 99) for d in frame_depths if np.any(~np.isnan(d))
)
max_depth_vis = max(max_depth_vis, 1e-4)   # safety floor

# Peak Q for y-axis
peak_Q = max(hydro_Q.max(), 1e-3)

# ── Figure layout ─────────────────────────────────────────────────────────
fig, (ax_map, ax_hyd) = plt.subplots(
    1, 2,
    figsize=(14, 6),
    gridspec_kw={"width_ratios": [1.2, 1]}
)
fig.patch.set_facecolor("#0f1117")
for ax in (ax_map, ax_hyd):
    ax.set_facecolor("#0f1117")
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

# ── LEFT: spatial map ─────────────────────────────────────────────────────
# Hillshade background (static)
ax_map.imshow(
    hillshade,
    cmap="gray", vmin=0, vmax=1,
    aspect="auto", alpha=0.55,
    extent=[0, ncols, nrows, 0]
)

# Watershed outline
ws_outline = np.where(ws_mask, 0.5, np.nan)
ax_map.imshow(ws_outline, cmap="Greys", vmin=0, vmax=1,
              aspect="auto", alpha=0.15, extent=[0, ncols, nrows, 0])

# Water depth layer (updated each frame)
depth_im = ax_map.imshow(
    frame_depths[0],
    cmap="YlGnBu",
    vmin=0, vmax=max_depth_vis,
    aspect="auto",
    alpha=0.85,
    extent=[0, ncols, nrows, 0]
)

# Outlet marker
ax_map.plot(
    grid_data["outlet_rc"][1],
    grid_data["outlet_rc"][0],
    marker="*", color="#FF4444", markersize=12,
    zorder=10, label="Outlet"
)
ax_map.legend(loc="lower right", facecolor="#1a1a2e", labelcolor="white",
              fontsize=8, framealpha=0.7)

cbar = fig.colorbar(depth_im, ax=ax_map, fraction=0.035, pad=0.02)
cbar.set_label("Water Depth  [m]", color="white", fontsize=9)
cbar.ax.yaxis.set_tick_params(color="white")
plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

ax_map.set_title("Watershed Water Depth", color="white", fontsize=11, pad=6)
ax_map.set_xlabel("Column (cells)", color="#aaa", fontsize=8)
ax_map.set_ylabel("Row (cells)", color="#aaa", fontsize=8)
time_text = ax_map.text(
    0.03, 0.97, "", transform=ax_map.transAxes,
    color="white", fontsize=10, va="top",
    bbox=dict(boxstyle="round,pad=0.3", fc="#1a1a2e", alpha=0.8)
)
rain_text = ax_map.text(
    0.03, 0.88, "", transform=ax_map.transAxes,
    color="#FFD700", fontsize=9, va="top",
    bbox=dict(boxstyle="round,pad=0.3", fc="#1a1a2e", alpha=0.8)
)

# ── RIGHT: hydrograph ─────────────────────────────────────────────────────
ax_hyd.set_xlim(0, config.TOTAL_SIMULATION_TIME_HOURS)
ax_hyd.set_ylim(0, peak_Q * 1.15)
ax_hyd.set_xlabel("Time  [hours]", color="#aaa", fontsize=9)
ax_hyd.set_ylabel("Q at Outlet  [m³/s]", color="#aaa", fontsize=9)
ax_hyd.set_title("Outlet Hydrograph", color="white", fontsize=11, pad=6)
ax_hyd.grid(True, color="#333", linewidth=0.5, linestyle="--")

# Rainfall duration shading
rain_end_hr = config.RAIN_DURATION_HOURS
ax_hyd.axvspan(0, rain_end_hr, alpha=0.12, color="#4fc3f7",
               label=f"Rainfall ({config.RAIN_INTENSITY_MM_HR} mm/hr)")
ax_hyd.legend(loc="upper right", facecolor="#1a1a2e", labelcolor="white",
              fontsize=8, framealpha=0.7)

# Growing hydrograph line
hyd_line, = ax_hyd.plot([], [], color="#00e5ff", linewidth=1.8, zorder=4)

# Peak annotation (will be updated)
peak_annot = ax_hyd.annotate(
    "", xy=(0, 0), xytext=(0, 0),
    color="#FF4444", fontsize=8,
    arrowprops=dict(arrowstyle="->", color="#FF4444")
)

# Vertical time cursor
vline = ax_hyd.axvline(0, color="#FF4444", linewidth=1.2, linestyle="--", zorder=5)

plt.tight_layout(pad=1.5)

# ─────────────────────────────────────────────────────────────────────────────
# 5.  Animation update function
# ─────────────────────────────────────────────────────────────────────────────
def update(frame_idx):
    t_hr       = frame_times_hr[frame_idx]
    depth_2d   = frame_depths[frame_idx]

    # --- spatial map ---
    depth_2d_clipped = np.where(np.isnan(depth_2d), np.nan,
                                np.clip(depth_2d, 0, max_depth_vis))
    depth_im.set_data(depth_2d_clipped)
    time_text.set_text(f"t = {t_hr:.2f} h")

    is_raining = t_hr <= config.RAIN_DURATION_HOURS
    rain_text.set_text("☂ Raining" if is_raining else "☀ Dry period")

    # --- hydrograph (show data up to current frame time) ---
    mask = hydro_times_hr <= t_hr
    hyd_line.set_data(hydro_times_hr[mask], hydro_Q[mask])

    # Move cursor
    vline.set_xdata([t_hr, t_hr])

    # Annotate running peak
    if mask.any():
        local_peak_idx = np.argmax(hydro_Q[mask])
        local_peak_Q   = hydro_Q[mask][local_peak_idx]
        local_peak_t   = hydro_times_hr[mask][local_peak_idx]
        if local_peak_Q > 0.01:
            peak_annot.xy          = (local_peak_t, local_peak_Q)
            peak_annot.xytext      = (local_peak_t + 0.1,
                                      local_peak_Q * 0.85)
            peak_annot.set_text(f"Peak\n{local_peak_Q:.2f} m³/s")
        else:
            peak_annot.set_text("")

    return depth_im, hyd_line, vline, time_text, rain_text, peak_annot


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Render and save GIF
# ─────────────────────────────────────────────────────────────────────────────
anim = animation.FuncAnimation(
    fig,
    update,
    frames=n_frames,
    interval=1000 // GIF_FPS,
    blit=True
)

writer = animation.PillowWriter(fps=GIF_FPS)
anim.save(OUTPUT_GIF, writer=writer, dpi=GIF_DPI,
          savefig_kwargs={"facecolor": fig.get_facecolor()})

plt.close(fig)
print(f"\n✓  Animation saved → {OUTPUT_GIF}")
print(f"   Frames : {n_frames}   FPS : {GIF_FPS}   DPI : {GIF_DPI}")
