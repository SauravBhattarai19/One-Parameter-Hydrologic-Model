# -*- coding: utf-8 -*-
"""
hydraulics.py — per-step flow kernels shared by both compute backends.

Manning velocity/discharge (sheet and confined rectangular channel),
the diffusive-wave (CASC2D/GSSHA-style water-surface-slope) discharge,
the volume flux limiter and the legacy uniform-rainfall array builder.
All array functions are backend-agnostic (NumPy or CuPy via the xp arg).
"""

import numpy as np




# ---------------------------------------------------------------------------
# 5.  Manning's equation (vectorised over all active cells)
# ---------------------------------------------------------------------------

def mannings_velocity(depth, slope, n):
    """
    V = (1/n) * depth^(2/3) * slope^(1/2)    [m/s]

    Parameters are 1-D arrays (one value per active cell).
    """
    return (1.0 / n) * (depth ** (2.0 / 3.0)) * (slope ** 0.5)


def cell_discharge(depth, velocity, cell_size):
    """
    Q = V * width * depth   [m³/s]
    Assumes wide rectangular cross-section → width ≈ cell_size.
    """
    return velocity * cell_size * depth


def mannings_discharge(depth, slope, n, width, chan_mask, cell_size, xp):
    """
    Kinematic Manning discharge with an optional CONFINED rectangular channel
    section on channel cells.

    Overland cells (``chan_mask`` False) use the wide-channel shortcut R ≈ depth
    and a flow width of ``cell_size`` — reproducing ``mannings_velocity`` followed
    by ``cell_discharge`` *bit-for-bit* (same arithmetic, same grouping).  Channel
    cells use a true rectangular cross-section of width ``B`` (≪ cell_size):

        A   = B · h                       (cross-section area)
        P   = B + 2·h                     (wetted perimeter)
        R   = A / P = B·h/(B+2h)          (hydraulic radius; < h when B is finite)
        Q   = (1/n) · R^(2/3) · √S · A    [m³/s]

    Confining the flow to ``B`` instead of spreading it across the whole DEM cell
    makes channel cells run deeper and faster (correct celerity / attenuation),
    which the wide-sheet ``R ≈ depth`` assumption cannot capture.

    Parameters
    ----------
    depth     : (n,) array – flow depth [m] (channel cells: depth over B·L footprint)
    slope     : (n,) array – friction slope [m/m]
    n         : scalar or (n,) array – Manning's n
    width     : (n,) array – flow width [m]: cell_size overland, B on channel cells
    chan_mask : (n,) bool array – True where the rectangular section applies
    cell_size : float – DEM cell size [m] (overland width)
    xp        : array module (numpy or cupy)

    Returns
    -------
    Q    : (n,) array [m³/s]  – Manning discharge (pre flux-limiter)
    A_xs : (n,) array [m²]    – flow cross-section area; celerity uses c = 5/3·Q/A_xs
    """
    # Overland (wide sheet): identical arithmetic to the original kinematic path.
    velocity   = (1.0 / n) * (depth ** (2.0 / 3.0)) * (slope ** 0.5)
    Q_overland = velocity * cell_size * depth
    A_overland = depth * cell_size

    # Channel (rectangular): true hydraulic radius R = A/P.
    A_chan = depth * width
    R_chan = A_chan / (width + 2.0 * depth)
    Q_chan = (1.0 / n) * (R_chan ** (2.0 / 3.0)) * (slope ** 0.5) * A_chan

    Q    = xp.where(chan_mask, Q_chan, Q_overland)
    A_xs = xp.where(chan_mask, A_chan, A_overland)
    return Q, A_xs


def diffusive_wave_discharge(depth, dem, dist, slope_bnd, n, ds_safe, valid_ds,
                             theta, cell_size, xp, min_depth, width, chan_mask):
    """
    CASC2D / GSSHA-style diffusion-wave cell discharge [m³/s].

    Replaces the pure-kinematic ``mannings_velocity``→``cell_discharge`` pair when
    ``ROUTING_SCHEME='diffusive'``.  The friction slope becomes the *water-surface*
    slope along the D8 flow path, which lets the wave attenuate (peak flattening) and
    slow under an adverse gradient — physics the bed-slope kinematic wave cannot capture.

        S_w    = slope_bnd  +  θ · (h_i − h_ds)/dist               (water-surface slope)
        S_eff  = max(S_w, 0)                                        (adverse grad → no flow)
        h_hb   = max(WSE_i, WSE_ds) − max(z_i, z_ds)               (depth over higher bed)
        h_flow = max( (1−θ)·h_i + θ·h_hb , min_depth )            (kinematic↔diffusion blend)
        Q      = (1/n) · h_flow^(5/3) · S_eff^(1/2) · cell_size

    θ blends BOTH the slope and the conveyance depth between the two coherent endpoints:
    θ=0 → own depth + bed slope = the kinematic scheme *exactly*; θ=1 → flow-depth-over-the-
    higher-bed + water-surface slope = the full CASC2D/GSSHA-style diffusion wave.  The bed
    term is the existing ``slope_bnd`` (= ``slope_1d`` = (z_i−z_ds)/dist, already floored at
    MIN_SLOPE with watershed-boundary handling), so steep cells get the true bed slope while
    flat cells keep draining; the depth-gradient term can still drive S_w below zero → the
    clamp reproduces backwater slowdown.  Cells with no valid downstream neighbour
    (``~valid_ds`` — the outlet and any cell draining off-mask) keep ``slope_bnd`` and their
    own depth, i.e. free outflow identical to the kinematic scheme.

    All arithmetic is via ``xp`` (NumPy or CuPy) so the helper runs on CPU and GPU alike.

    Parameters
    ----------
    depth     : (n,) array  – current flow depth per cell [m]
    dem       : (n,) array  – bed elevation per cell [m]
    dist      : (n,) array  – flow-path length to the downstream cell [m] (dx or dx·√2)
    slope_bnd : (n,) array  – bed slope (used only for ~valid_ds free-outflow cells)
    n         : scalar or (n,) array – Manning's n
    ds_safe   : (n,) int array – downstream index, clamped to 0 where invalid
    valid_ds  : (n,) bool array – True where the cell has a real downstream neighbour
    theta     : float – diffusion weight θ∈[0,1]
    cell_size : float – flow width ≈ cell size [m]
    xp        : array module (numpy or cupy)
    min_depth : float – wet/dry conveyance-depth floor [m]
    width     : (n,) array – flow width [m]: cell_size overland, channel width B on
                             channel cells (CONFINED rectangular conveyance).
    chan_mask : (n,) bool array – True where the rectangular section R=A/P applies.

    Returns
    -------
    Q_out  : (n,) array  [m³/s]  (NOT yet flux-limited — caller applies the CFL limiter)
    A_xs   : (n,) array  [m²]    conveyance cross-section area; celerity denominator
                                 (c = 5/3·Q/A_xs).  Overland: h_flow·cell_size.
    S_eff  : (n,) array  [m/m]   effective (clamped) friction slope used for Q.
    """
    depth_ds = depth[ds_safe]
    dem_ds   = dem[ds_safe]

    # Water-surface slope along the flow path: floored bed slope + θ depth-gradient term.
    S_w = slope_bnd + theta * (depth - depth_ds) / dist
    # Free-outflow cells (no downstream) fall back to the kinematic bed slope.
    S_eff = xp.where(valid_ds, S_w, slope_bnd)
    S_eff = xp.maximum(S_eff, 0.0)                       # adverse gradient → no discharge

    # Conveyance depth: blend own depth (kinematic) with flow-depth-over-the-higher-bed
    # (LISFLOOD-FP diffusion-wave convention) by the SAME θ as the slope, so the two terms
    # stay a coherent pair — θ=0 → own depth + bed slope = kinematic exactly; θ=1 → higher-
    # bed depth + water-surface slope = full diffusion wave.  In the normal downhill case
    # the higher-bed depth already equals the upstream cell's own depth.
    wse      = dem + depth
    wse_ds   = wse[ds_safe]
    h_higher = xp.maximum(wse, wse_ds) - xp.maximum(dem, dem_ds)
    h_flow   = (1.0 - theta) * depth + theta * h_higher
    h_flow   = xp.where(valid_ds, h_flow, depth)          # free-outflow cells: own depth
    h_flow   = xp.maximum(h_flow, min_depth)

    # Conveyance discharge.  Overland (chan_mask False): wide sheet R≈h_flow,
    # width=cell_size — identical arithmetic to the original diffusive path.
    # Channel: confined rectangular section, true hydraulic radius R=A/P.
    Q_overland = (1.0 / n) * (h_flow ** (5.0 / 3.0)) * (S_eff ** 0.5) * cell_size
    A_overland = h_flow * cell_size

    A_chan = h_flow * width
    R_chan = A_chan / (width + 2.0 * h_flow)
    Q_chan = (1.0 / n) * (R_chan ** (2.0 / 3.0)) * (S_eff ** 0.5) * A_chan

    Q    = xp.where(chan_mask, Q_chan, Q_overland)
    A_xs = xp.where(chan_mask, A_chan, A_overland)
    # A_xs exposed so callers use the correct celerity denominator (c=5/3·Q/A_xs);
    # S_eff exposed for diagnostics.
    return Q, A_xs, S_eff


def flux_limiter(Q_out, volume, dt):
    """
    Volume-conservative CFL limiter.

    Caps Q_out so that a cell can never drain more water than it currently
    stores in a single time step:

        Q_out_limited = min(Q_out, volume / dt)

    This prevents the positive-feedback runaway that occurs in the explicit
    kinematic-wave scheme when the Courant number C = V * dt / dx > 1.
    The fix is mass-conservative: the downstream cell simply receives less
    inflow, which is physically correct (there is no more water to give).

    Parameters
    ----------
    Q_out  : 1-D float array  – Manning's discharge [m³/s] for each cell
    volume : 1-D float array  – current stored volume [m³] for each cell
    dt     : float            – time step [s]

    Returns
    -------
    Q_out_limited : 1-D float array  [m³/s]
    """
    return np.minimum(Q_out, np.maximum(volume, 0.0) / dt)


# ---------------------------------------------------------------------------
# 6.  Rainfall array builder
# ---------------------------------------------------------------------------

def build_rainfall_array(shape, intensity_mm_hr, duration_hours, dt_seconds, t_seconds):
    """
    Return a 2-D rainfall array (m/s) for the current simulation time.

    For a spatially uniform event:
        - intensity_mm_hr converted to m/s = intensity / (1000 * 3600)
        - Applied only while t_seconds < duration_hours * 3600

    The function signature accepts `shape` so it can later be replaced by a
    spatially variable (e.g., radar) array without changing the router logic.

    Parameters
    ----------
    shape           : (nrows, ncols) of the grid
    intensity_mm_hr : uniform rainfall rate [mm/hr]
    duration_hours  : rainfall duration [hr]
    dt_seconds      : time step [s]  (unused here; kept for API consistency)
    t_seconds       : current simulation time [s]

    Returns
    -------
    rain_ms : 2-D float64 array  (m/s)
    """
    rain_ms_value = (intensity_mm_hr / (1000.0 * 3600.0)
                     if t_seconds < duration_hours * 3600.0
                     else 0.0)
    return np.full(shape, rain_ms_value, dtype=np.float64)
