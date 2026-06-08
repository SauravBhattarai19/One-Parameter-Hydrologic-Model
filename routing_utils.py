"""
routing_utils.py
================
Helper functions for the explicit kinematic-wave routing model.

Responsibilities:
  - Load and validate all input rasters.
  - Build a flattened, topologically-ordered list of active (watershed) cells.
  - Compute per-cell slope from the DEM + flow-direction.
  - Manning's equation physics (velocity, discharge).
  - Rainfall array construction.
"""

import numpy as np
import rasterio

# ---------------------------------------------------------------------------
# D8 directional look-up (pysheds convention)
# Encoding:  64 128  1
#            32   *  2
#            16   8  4
# ---------------------------------------------------------------------------
# Maps encoded value -> (row_delta, col_delta)
D8_MOVE = {
    64:  (-1,  0),   # N
    128: (-1,  1),   # NE
    1:   ( 0,  1),   # E
    2:   ( 1,  1),   # SE
    4:   ( 1,  0),   # S
    8:   ( 1, -1),   # SW
    16:  ( 0, -1),   # W
    32:  (-1, -1),   # NW
}

# Diagonal directions (distance = cell_size * sqrt(2))
D8_DIAGONAL = {128, 2, 8, 32}


# ---------------------------------------------------------------------------
# 1.  Raster loading
# ---------------------------------------------------------------------------

def load_rasters(cfg):
    """
    Load all four input rasters and return numpy arrays plus spatial metadata.

    Parameters
    ----------
    cfg : module
        The imported config module.

    Returns
    -------
    dem        : 2-D float64 array  – clipped elevation (m)
    fdir       : 2-D int array      – D8 flow direction
    faccum     : 2-D float array    – clipped flow accumulation
    ws_mask    : 2-D bool array     – True where cell is inside the watershed
    transform  : affine.Affine      – rasterio transform of the rasters
    nodata_dem : float              – nodata value from the DEM raster
    """
    with rasterio.open(cfg.ROUTING_DEM_PATH) as src:
        dem        = src.read(1).astype(np.float64)
        nodata_dem = src.nodata
        transform  = src.transform
        # Cell size: use config override if set, otherwise derive from transform.
        # transform.a = pixel width (x), transform.e = pixel height (negative).
        # For square pixels in a projected CRS both magnitudes are equal.
        if getattr(cfg, 'CELL_SIZE', None) is not None:
            cell_size = float(cfg.CELL_SIZE)
        else:
            cell_size = float(abs(transform.a))

    with rasterio.open(cfg.ROUTING_FLOW_DIR_PATH) as src:
        fdir = src.read(1)

    with rasterio.open(cfg.ROUTING_FLOW_ACCUM_PATH) as src:
        faccum     = src.read(1).astype(np.float64)
        nodata_fa  = src.nodata

    with rasterio.open(cfg.ROUTING_WATERSHED_MASK_PATH) as src:
        ws_raw = src.read(1)

    # Active cells: inside watershed AND dem has valid data
    ws_mask = (ws_raw > 0)
    if nodata_dem is not None:
        ws_mask &= (dem != nodata_dem)
    if nodata_fa is not None:
        ws_mask &= (faccum != nodata_fa)

    print(f"  Rasters loaded  |  shape={dem.shape}  |  active cells={ws_mask.sum():,}"
          f"  |  cell_size={cell_size:.2f} m")
    return dem, fdir, faccum, ws_mask, transform, nodata_dem, cell_size


# ---------------------------------------------------------------------------
# 2.  Slope calculation
# ---------------------------------------------------------------------------

def compute_slope_grid(dem, fdir, ws_mask, cell_size, min_slope, nodata_dem=None):
    """
    For every active cell compute slope along its D8 flow direction.

        S0 = max( (elev_current - elev_downstream) / distance , min_slope )

    Diagonal neighbours use distance = cell_size * sqrt(2).

    Boundary handling
    -----------------
    Cells whose downstream neighbour is out-of-bounds *or* carries a nodata
    value are treated as watershed-boundary cells.  Their slope is estimated
    from the *backward* gradient (upstream cell → current cell), which is the
    best available proxy for the local channel slope.  Two common failure modes
    are guarded against:

    1. Out-of-bounds downstream (raster edge) — handled explicitly.
    2. In-bounds but nodata downstream — the value may be NaN or the DEM's
       sentinel (e.g. -9999, -32768).  Python's built-in max() silently returns
       min_slope when the first argument is NaN (max(nan, x) == x), so this
       case must be detected *before* attempting arithmetic.

    Parameters
    ----------
    dem       : 2-D float64 array
    fdir      : 2-D int array  (D8 encoding)
    ws_mask   : 2-D bool array
    cell_size : float  [m]
    min_slope : float  [m/m]  – floor applied everywhere
    nodata_dem: float or None – nodata sentinel in *dem*

    Returns
    -------
    slope : 2-D float64 array  (same shape as dem, zero for inactive cells)
    """
    nrows, ncols = dem.shape
    slope = np.zeros_like(dem, dtype=np.float64)

    # Pre-compute nodata sentinel as float64 for reliable comparison
    _nodata = float(nodata_dem) if nodata_dem is not None else None

    def _is_nodata(val):
        """True if val is NaN or equals the DEM nodata sentinel."""
        if np.isnan(val):
            return True
        if _nodata is not None and not np.isnan(_nodata) and val == _nodata:
            return True
        return False

    rows, cols = np.where(ws_mask)

    for r, c in zip(rows, cols):
        d = int(fdir[r, c])
        if d not in D8_MOVE:
            slope[r, c] = min_slope
            continue

        dr, dc = D8_MOVE[d]
        rn, cn = r + dr, c + dc
        dist   = cell_size * (2 ** 0.5 if d in D8_DIAGONAL else 1.0)

        # ── Case 1: valid downstream cell (in-bounds, not nodata) ───────────
        if (0 <= rn < nrows and 0 <= cn < ncols and not _is_nodata(dem[rn, cn])):
            dz = dem[r, c] - dem[rn, cn]
            slope[r, c] = max(dz / dist, min_slope)
            continue

        # ── Case 2: boundary cell (downstream out-of-bounds or nodata) ──────
        # Use the backward gradient (upstream cell → this cell) as proxy for
        # the local channel slope.
        rb, cb = r - dr, c - dc
        if (0 <= rb < nrows and 0 <= cb < ncols and not _is_nodata(dem[rb, cb])):
            dz_back = dem[rb, cb] - dem[r, c]
            slope[r, c] = max(dz_back / dist, min_slope)
            continue

        slope[r, c] = min_slope

    return slope


# ---------------------------------------------------------------------------
# 3.  Topological ordering
# ---------------------------------------------------------------------------

def topological_order(faccum, fdir, ws_mask):
    """
    Return the (row, col) indices of all active watershed cells sorted from
    upstream → downstream (ascending flow accumulation).

    Using flow accumulation as a proxy for topological rank is valid because
    a downstream cell always has a higher accumulation than any of its upstream
    contributors.

    Returns
    -------
    sorted_rows : 1-D int array
    sorted_cols : 1-D int array
    outlet_rc   : (int, int) – row/col of the outlet (highest accumulation)
    """
    rows, cols = np.where(ws_mask)
    accum_vals = faccum[rows, cols]

    # Sort ascending: upstream first
    order       = np.argsort(accum_vals)
    sorted_rows = rows[order]
    sorted_cols = cols[order]

    # Outlet = last cell (maximum accumulation)
    outlet_rc = (int(sorted_rows[-1]), int(sorted_cols[-1]))
    print(f"  Outlet cell     |  row={outlet_rc[0]}  col={outlet_rc[1]}"
          f"  accum={faccum[outlet_rc]:.0f}")

    return sorted_rows, sorted_cols, outlet_rc


# ---------------------------------------------------------------------------
# 4.  Downstream neighbour lookup (vectorised)
# ---------------------------------------------------------------------------

def build_downstream_map(sorted_rows, sorted_cols, fdir, ws_mask, nrows, ncols):
    """
    For each active cell (indexed in topological order) find the flat index
    of its downstream neighbour (or -1 if the neighbour is out-of-bounds or
    outside the watershed).

    Returns
    -------
    ds_idx : 1-D int array  (same length as sorted_rows)
        Flat index into sorted_rows/sorted_cols of the downstream cell,
        or -1 for the outlet / cells that drain off-mask.
    """
    n_cells = len(sorted_rows)

    # Build a (nrows*ncols) -> position-in-sorted-list lookup
    flat_to_pos = np.full(nrows * ncols, -1, dtype=np.int64)
    flat_ids    = sorted_rows * ncols + sorted_cols
    flat_to_pos[flat_ids] = np.arange(n_cells, dtype=np.int64)

    ds_idx = np.full(n_cells, -1, dtype=np.int64)

    for i, (r, c) in enumerate(zip(sorted_rows, sorted_cols)):
        d = int(fdir[r, c])
        if d not in D8_MOVE:
            continue
        dr, dc = D8_MOVE[d]
        rn, cn = r + dr, c + dc
        if 0 <= rn < nrows and 0 <= cn < ncols and ws_mask[rn, cn]:
            ds_flat     = rn * ncols + cn
            ds_idx[i]   = flat_to_pos[ds_flat]

    return ds_idx


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


# ---------------------------------------------------------------------------
# 7.  Raster alignment
# ---------------------------------------------------------------------------

def align_raster_to_dem(src_path, dem_path, resampling='nearest'):
    """
    Reproject/resample *src_path* to exactly match the routing DEM grid.

    Returns a 2-D numpy array with the same (height, width) as the DEM.
    """
    from rasterio.warp import reproject, Resampling

    METHODS = {
        'nearest':  Resampling.nearest,
        'bilinear': Resampling.bilinear,
    }

    with rasterio.open(dem_path) as dem:
        dst_crs       = dem.crs
        dst_transform = dem.transform
        dst_shape     = (dem.height, dem.width)

    with rasterio.open(src_path) as src:
        dst_array = np.empty(dst_shape, dtype=src.dtypes[0])
        reproject(
            source=rasterio.band(src, 1),
            destination=dst_array,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=METHODS.get(resampling, Resampling.nearest),
        )
    return dst_array


# ---------------------------------------------------------------------------
# 8.  Spatially variable Manning's n
# ---------------------------------------------------------------------------

def resolve_mannings_n(cfg, grid_data):
    """
    Build a per-cell Manning's n array from config.

    Supports three sources (``MANNINGS_N_SOURCE``):
      scalar  – uniform value from ``MANNINGS_N``
      lulc    – LULC class codes remapped via ``LULC_LOOKUP_CSV``
      raster  – pre-computed Manning's n GeoTIFF

    In all modes, cells whose flow accumulation exceeds a threshold
    are overridden with ``MANNINGS_N_CHANNEL`` (if not None).

    Returns 1-D float64 array of shape ``(n_cells,)``.
    """
    import os
    import pandas as pd

    source     = getattr(cfg, 'MANNINGS_N_SOURCE', 'scalar').lower()
    n_fallback = float(cfg.MANNINGS_N)
    s_rows     = grid_data['s_rows']
    s_cols     = grid_data['s_cols']
    n_cells    = len(s_rows)

    # ── Source: scalar ────────────────────────────────────────────────────
    if source == 'scalar':
        n_1d = np.full(n_cells, n_fallback, dtype=np.float64)

    # ── Source: pre-computed raster ───────────────────────────────────────
    elif source == 'raster':
        path = getattr(cfg, 'MANNINGS_N_RASTER_PATH', None)
        if path is None or not os.path.isfile(path):
            raise FileNotFoundError(
                f"MANNINGS_N_SOURCE='raster' but MANNINGS_N_RASTER_PATH "
                f"not found: {path}"
            )
        dem_path = cfg.ROUTING_DEM_PATH
        n_2d = align_raster_to_dem(path, dem_path, resampling='bilinear')
        n_1d = n_2d[s_rows, s_cols].astype(np.float64)
        bad = (n_1d <= 0) | ~np.isfinite(n_1d)
        if bad.any():
            n_1d[bad] = n_fallback

    # ── Source: LULC class codes → lookup CSV ────────────────────────────
    elif source == 'lulc':
        lulc_path = getattr(cfg, 'MANNINGS_N_LULC_PATH', 'gee')
        dem_path  = cfg.ROUTING_DEM_PATH

        if str(lulc_path).lower() == 'gee':
            try:
                from serves_gee import download_lulc_raster
            except ImportError:
                print("  [WARN] earthengine-api not installed; "
                      f"using scalar n={n_fallback}")
                return np.full(n_cells, n_fallback, dtype=np.float64)

            output_dir = getattr(cfg, 'OUTPUT_DIR', 'output/')
            cached = os.path.join(output_dir, 'lulc_mannings.tif')
            result = download_lulc_raster(
                dem_path=dem_path,
                watershed_geojson_path=getattr(
                    cfg, 'OPM_WATERSHED_GEOJSON', 'output/watershed.geojson'),
                output_path=cached,
                project=getattr(cfg, 'GEE_PROJECT', None),
            )
            if result is None:
                print("  [WARN] GEE LULC download failed; "
                      f"using scalar n={n_fallback}")
                return np.full(n_cells, n_fallback, dtype=np.float64)
            lulc_2d = align_raster_to_dem(cached, dem_path,
                                          resampling='nearest')
        else:
            if not os.path.isfile(lulc_path):
                raise FileNotFoundError(
                    f"MANNINGS_N_LULC_PATH not found: {lulc_path}"
                )
            lulc_2d = align_raster_to_dem(lulc_path, dem_path,
                                          resampling='nearest')

        csv_path = getattr(cfg, 'LULC_LOOKUP_CSV', 'lulc_lookup.csv')
        lut = pd.read_csv(csv_path)
        code_to_n = dict(zip(lut['class_code'].astype(int),
                             lut['mannings_n'].astype(float)))
        lulc_1d = lulc_2d[s_rows, s_cols]
        n_1d = np.full(n_cells, n_fallback, dtype=np.float64)
        for code, nval in code_to_n.items():
            n_1d[lulc_1d == code] = nval

    # ── Source: WUDAPT Local Climate Zones ───────────────────────────────
    elif source == 'lcz':
        try:
            from serves_gee import download_lcz_raster
        except ImportError:
            print("  [WARN] earthengine-api not installed; "
                  f"using scalar n={n_fallback}")
            return np.full(n_cells, n_fallback, dtype=np.float64)

        output_dir = getattr(cfg, 'OUTPUT_DIR', 'output/')
        cached = os.path.join(output_dir, 'lulc_mannings_lcz.tif')
        dem_path = cfg.ROUTING_DEM_PATH
        result = download_lcz_raster(
            dem_path=dem_path,
            watershed_geojson_path=getattr(
                cfg, 'OPM_WATERSHED_GEOJSON', 'output/watershed.geojson'),
            output_path=cached,
            project=getattr(cfg, 'GEE_PROJECT', None),
        )
        if result is None:
            print("  [WARN] GEE LCZ download failed; "
                  f"using scalar n={n_fallback}")
            return np.full(n_cells, n_fallback, dtype=np.float64)

        lcz_2d = align_raster_to_dem(cached, dem_path, resampling='nearest')
        csv_path = getattr(cfg, 'LCZ_LOOKUP_CSV', 'lcz_lookup.csv')
        lut = pd.read_csv(csv_path)
        code_to_n = dict(zip(lut['class_code'].astype(int),
                             lut['mannings_n'].astype(float)))
        lulc_1d = lcz_2d[s_rows, s_cols]
        n_1d = np.full(n_cells, n_fallback, dtype=np.float64)
        for code, nval in code_to_n.items():
            n_1d[lulc_1d == code] = nval

    else:
        raise ValueError(f"Unknown MANNINGS_N_SOURCE: '{source}'")

    # ── Channel override (all modes) ─────────────────────────────────────
    n_channel_cfg = getattr(cfg, 'MANNINGS_N_CHANNEL', None)
    if n_channel_cfg is not None:
        faccum_1d = grid_data['faccum_1d']
        fa = faccum_1d.get() if hasattr(faccum_1d, 'get') else np.asarray(
            faccum_1d)
        threshold = getattr(cfg, 'CHANNEL_FACCUM_THRESHOLD', None)
        if threshold is None:
            threshold = max(1, n_cells // 100)
        channel_mask = fa > threshold

        if isinstance(n_channel_cfg, dict):
            ds_idx = grid_data['ds_idx']
            ds_np = ds_idx.get() if hasattr(ds_idx, 'get') else np.asarray(
                ds_idx)
            strahler = compute_strahler_order(ds_np, n_cells)
            max_order = max(n_channel_cfg.keys())
            for ci in np.where(channel_mask)[0]:
                so = min(int(strahler[ci]), max_order)
                n_1d[ci] = n_channel_cfg.get(so, n_channel_cfg[max_order])
            order_dist = {o: int((strahler[channel_mask] == o).sum())
                          for o in sorted(set(strahler[channel_mask]))}
            print(f"  Manning's n   |  channel cells: "
                  f"{int(channel_mask.sum()):,} / {n_cells:,}  "
                  f"(threshold={threshold})")
            print(f"  Manning's n   |  Strahler order distribution: "
                  f"{order_dist}")
        else:
            n_1d[channel_mask] = float(n_channel_cfg)
            print(f"  Manning's n   |  channel cells: "
                  f"{int(channel_mask.sum()):,} / {n_cells:,}  "
                  f"(threshold={threshold})")

    print(f"  Manning's n   |  source={source}"
          f"  range=[{n_1d.min():.4f}, {n_1d.max():.4f}]"
          f"  mean={n_1d.mean():.4f}")
    return n_1d


# ---------------------------------------------------------------------------
# 8b. Per-cell land-cover field lookup (impervious fraction, root depth, …)
# ---------------------------------------------------------------------------

def _lulc_class_1d(cfg, grid_data, source):
    """
    Per-cell land-cover class codes from the cached LCZ/LULC raster.

    Reuses the SAME cache file as ``resolve_mannings_n``
    (``lulc_mannings_lcz.tif`` / ``lulc_mannings.tif``) so no extra GEE download
    happens when Manning's n already pulled the layer.

    Returns
    -------
    (class_1d, lookup_csv_path) : (n_cells,) int-ish array + CSV path,
        or (None, None) when the raster is unavailable.
    """
    import os

    s_rows = grid_data['s_rows']
    s_cols = grid_data['s_cols']
    dem_path   = cfg.ROUTING_DEM_PATH
    output_dir = getattr(cfg, 'OUTPUT_DIR', 'output/')
    geojson    = getattr(cfg, 'OPM_WATERSHED_GEOJSON', 'output/watershed.geojson')
    project    = getattr(cfg, 'GEE_PROJECT', None)

    if source == 'lcz':
        try:
            from serves_gee import download_lcz_raster
        except ImportError:
            return None, None
        cached = os.path.join(output_dir, 'lulc_mannings_lcz.tif')
        result = download_lcz_raster(dem_path=dem_path,
                                     watershed_geojson_path=geojson,
                                     output_path=cached, project=project)
        csv_path = getattr(cfg, 'LCZ_LOOKUP_CSV', 'lcz_lookup.csv')
    else:
        try:
            from serves_gee import download_lulc_raster
        except ImportError:
            return None, None
        cached = os.path.join(output_dir, 'lulc_mannings.tif')
        result = download_lulc_raster(dem_path=dem_path,
                                      watershed_geojson_path=geojson,
                                      output_path=cached, project=project)
        csv_path = getattr(cfg, 'LULC_LOOKUP_CSV', 'lulc_lookup.csv')

    if result is None:
        return None, None

    arr2d = align_raster_to_dem(cached, dem_path, resampling='nearest')
    _to_np = lambda a: a.get() if hasattr(a, 'get') else np.asarray(a)
    return arr2d[_to_np(s_rows), _to_np(s_cols)], csv_path


def resolve_lulc_field(cfg, grid_data, column, default, source):
    """
    Build a per-cell field by remapping land-cover class codes through *column*
    of the LCZ/LULC lookup CSV.  Used for impervious fraction and root-zone
    depth.  Cells with an unmapped class (or when the raster/column is missing)
    get *default*.  Returns (n_cells,) float64.
    """
    import pandas as pd

    n_cells = len(grid_data['s_rows'])
    class_1d, csv_path = _lulc_class_1d(cfg, grid_data, source)
    if class_1d is None:
        print(f"  [WARN] {source} raster unavailable; "
              f"'{column}' → {default} everywhere")
        return np.full(n_cells, float(default), dtype=np.float64)

    lut = pd.read_csv(csv_path)
    if column not in lut.columns:
        print(f"  [WARN] column '{column}' missing from {csv_path}; "
              f"using {default} everywhere")
        return np.full(n_cells, float(default), dtype=np.float64)

    code_to_v = dict(zip(lut['class_code'].astype(int),
                         lut[column].astype(float)))
    out = np.full(n_cells, float(default), dtype=np.float64)
    for code, v in code_to_v.items():
        out[class_1d == code] = v
    return out


def resolve_impervious_fraction(cfg, grid_data):
    """
    Per-cell impervious fraction Imp ∈ [0,1] from ``IMPERVIOUS_SOURCE``.

      'lcz' / 'lulc' → impervious_fraction column of the matching lookup CSV
      'raster'       → continuous GeoTIFF at IMPERVIOUS_RASTER_PATH (bilinear)
      'none'         → zeros

    Returns (n_cells,) float64, clipped to [0,1].
    """
    import os

    source  = getattr(cfg, 'IMPERVIOUS_SOURCE', 'none').lower()
    n_cells = len(grid_data['s_rows'])
    s_rows  = grid_data['s_rows']
    s_cols  = grid_data['s_cols']

    if source == 'none':
        return np.zeros(n_cells, dtype=np.float64)

    if source == 'raster':
        path = getattr(cfg, 'IMPERVIOUS_RASTER_PATH', None)
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(
                f"IMPERVIOUS_SOURCE='raster' but IMPERVIOUS_RASTER_PATH "
                f"not found: {path}"
            )
        arr2d = align_raster_to_dem(path, cfg.ROUTING_DEM_PATH,
                                    resampling='bilinear')
        _to_np = lambda a: a.get() if hasattr(a, 'get') else np.asarray(a)
        imp = arr2d[_to_np(s_rows), _to_np(s_cols)].astype(np.float64)
        imp[~np.isfinite(imp)] = 0.0
    elif source in ('lcz', 'lulc'):
        imp = resolve_lulc_field(cfg, grid_data, 'impervious_fraction',
                                 0.0, source)
    else:
        raise ValueError(f"Unknown IMPERVIOUS_SOURCE: '{source}'")

    imp = np.clip(imp, 0.0, 1.0)
    print(f"  Impervious    |  source={source}"
          f"  range=[{imp.min():.2f}, {imp.max():.2f}]"
          f"  mean={imp.mean():.2f}  (>0: {int((imp > 0).sum()):,}/{n_cells:,} cells)")
    return imp


# ---------------------------------------------------------------------------
# 9.  Strahler stream order
# ---------------------------------------------------------------------------

def compute_strahler_order(ds_idx, n_cells):
    """
    Compute Strahler stream order for each cell.

    Cells must be in topological order (upstream first).  Single O(n) pass:
    headwater cells (no upstream) get order 1; when two streams of the same
    order merge, the result is order + 1; otherwise the higher order continues.

    Parameters
    ----------
    ds_idx  : (n_cells,) int array — downstream neighbour index (-1 = outlet)
    n_cells : int

    Returns
    -------
    order : (n_cells,) int array — Strahler order per cell
    """
    order     = np.ones(n_cells, dtype=np.int32)
    max_order = np.zeros(n_cells, dtype=np.int32)
    max_count = np.zeros(n_cells, dtype=np.int32)

    for i in range(n_cells):
        if max_order[i] == 0:
            order[i] = 1
        elif max_count[i] >= 2:
            order[i] = max_order[i] + 1
        else:
            order[i] = max_order[i]

        ds = int(ds_idx[i])
        if ds >= 0:
            if order[i] > max_order[ds]:
                max_order[ds] = order[i]
                max_count[ds] = 1
            elif order[i] == max_order[ds]:
                max_count[ds] += 1

    return order
