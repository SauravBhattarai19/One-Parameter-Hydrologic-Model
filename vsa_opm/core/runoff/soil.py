# -*- coding: utf-8 -*-
"""
soil.py — OPM soil-parameter resolution (SD_max, phi, K_sat, suction).

Resolves the saturation-deficit / porosity parameters either from the manual
config values or from the SERVES/SoilGrids rasters (via vsa_opm.gee), plus the
Rawls (1983) Green-Ampt suction lookup by USDA texture class.
"""

import numpy as np
import rasterio


# ── OPM physical constants (Pradhan & Ogden 2010) ────────────────────────────
OPM_SD_MIN = 0.001          # m   minimum saturation deficit
OPM_Q_MIN  = 0.001          # m³/s  minimum discharge (Eq 10)


def resolve_sd_params(cfg, cell_size):
    """
    Resolve OPM soil parameters from config (manual) or GEE.

    Returns dict with keys: sd_max, sd_min, phi, sd_max_per_polygon, ksat_ms,
    deficit_raster.  Per-zone SD is reduced from 'deficit_raster' in
    _init_vsa_opm using the rainfall partition (cell_polygon), not here.
    """
    sd_source   = getattr(cfg, 'OPM_SD_SOURCE', 'manual').lower()
    ksat_ms     = float(getattr(cfg, 'OPM_K_SAT', 44.0)) / 86400.0

    _manual_params = {
        'sd_max': float(cfg.OPM_SD_MAX_INITIAL),
        'sd_min': OPM_SD_MIN,
        'phi': float(getattr(cfg, 'OPM_PHI', 0.10)),
        'sd_max_per_polygon': None,
        'ksat_ms': ksat_ms,
        'deficit_raster': None,
    }

    if sd_source != 'gee':
        return _manual_params

    # Primary: EVENT_START_UTC is the event start — its date IS the SERVES target date.
    target_date = None
    evt = getattr(cfg, 'EVENT_START_UTC', None)
    if evt:
        target_date = str(evt).split()[0]   # 'YYYY-MM-DD'
        print(f"  SERVES target date → {target_date}  (from EVENT_START_UTC)")

    # Legacy fallback: IMERG_START_LOCAL (for old configs that predate EVENT_START_UTC)
    if target_date is None:
        method = getattr(cfg, 'PRECIP_METHOD', '').lower()
        start  = getattr(cfg, 'IMERG_START_LOCAL', None)
        if method.startswith('imerg') and start:
            target_date = str(start).split()[0]
            print(f"  SERVES target date → {target_date}  "
                  f"(legacy: derived from IMERG_START_LOCAL)")

    if target_date is None:
        print("  [WARN] EVENT_START_UTC not set; using manual SD/phi values.")
        return _manual_params

    try:
        from ...gee.serves_gee import compute_opm_params, download_deficit_raster
    except ImportError:
        print("  [WARN] earthengine-api not installed; using manual values.")
        return _manual_params

    geojson = getattr(cfg, 'OPM_WATERSHED_GEOJSON', 'output/watershed.geojson')
    sat     = getattr(cfg, 'SERVES_SATELLITE', 'landsat')
    window  = getattr(cfg, 'SERVES_SEARCH_WINDOW', 16)
    band    = getattr(cfg, 'OPM_SOILGRIDS_DEPTH', 'b30')
    project = getattr(cfg, 'GEE_PROJECT', None)

    # Select lookup CSV and GEE land cover source to match MANNINGS_N_SOURCE.
    _n_source = getattr(cfg, 'MANNINGS_N_SOURCE', 'lulc').lower()
    if _n_source == 'lcz':
        lut          = getattr(cfg, 'LCZ_LOOKUP_CSV',  'lcz_lookup.csv')
        gee_lc_source = 'lcz'
    else:
        lut          = getattr(cfg, 'LULC_LOOKUP_CSV', 'lulc_lookup.csv')
        gee_lc_source = 'worldcover'

    # Watershed-level SD_max + phi + diagnostics (scalars; no partition).
    gee_result = compute_opm_params(
        watershed_geojson_path=geojson, cell_size=cell_size,
        lookup_csv_path=lut, target_date=target_date,
        satellite=sat, search_window=window, soil_depth_band=band,
        project=project,
        sd_reducer=getattr(cfg, 'OPM_SD_REDUCER', 'mean'),
        lulc_source=gee_lc_source,
    )

    if gee_result is not None:
        sd_max = gee_result['sd_max']
        phi    = gee_result['phi']
    else:
        sd_max = float(cfg.OPM_SD_MAX_INITIAL)
        phi    = float(getattr(cfg, 'OPM_PHI', 0.10))

    # Per-cell deficit raster (aligned to the routing grid, clipped to the
    # watershed).  The engine reduces it per precipitation zone (cell_polygon),
    # so the SD partition is identical to the rainfall partition.
    #
    # Date-stamp the filename so each event caches its own raster:
    #   deficit_serves_2024-09-26.tif vs deficit_serves_2024-10-05.tif
    # Re-runs of the same event reuse the cached file (no re-download).
    # Falls back to the plain name when target_date is None (manual SD mode).
    out_path = getattr(cfg, 'OPM_DEFICIT_RASTER', None) \
        or (getattr(cfg, 'OUTPUT_DIR', 'output/') + 'deficit_serves.tif')
    if target_date:
        import os as _os
        _base, _ext = _os.path.splitext(out_path)
        out_path = f"{_base}_{target_date}{_ext}"
    deficit_raster = None
    try:
        deficit_raster = download_deficit_raster(
            dem_path=cfg.ROUTING_DEM_PATH, watershed_geojson_path=geojson,
            output_path=out_path, lookup_csv_path=lut, target_date=target_date,
            satellite=sat, search_window=window, soil_depth_band=band,
            project=project, lulc_source=gee_lc_source,
        )
    except Exception as exc:
        print(f"  [WARN] deficit raster download failed: {exc}")

    # ── Print diagnostics ────────────────────────────────────────────────
    print(f"  OPM params    |  SD_max(ws)={sd_max:.4f} m  (source={sd_source})"
          f"  phi={phi:.4f}  K_sat={ksat_ms * 86400:.2f} m/day")
    if gee_result is not None:
        print(f"                |  theta=[{gee_result.get('theta_min', 0):.3f},"
              f" {gee_result.get('theta_max', 0):.3f}]"
              f"  Z_r_mean={gee_result.get('root_depth_mean', 0):.2f} m")
    print(f"                |  deficit raster: "
          f"{'ready' if deficit_raster else 'unavailable → per-zone SD = watershed SD'}")

    return {
        'sd_max': sd_max,
        'sd_min': OPM_SD_MIN,
        'phi': phi,
        'sd_max_per_polygon': None,   # resolved per-zone from the raster instead
        'ksat_ms': ksat_ms,
        'deficit_raster': deficit_raster,
    }


def per_zone_sd_from_raster(deficit_path, cell_polygon, n_polygons,
                             s_rows, s_cols, reducer, sd_min, ws_default,
                             divide_idx=None):
    """
    Reduce the per-cell deficit raster into one SD_max per precipitation zone.

    Reducer modes (``OPM_SD_REDUCER``):
      'mean'   – zone-average deficit (representative; outlier-robust).
      'max'    – largest deficit in the zone (max soil-STORAGE-capacity cell;
                 biased toward deep-rooted cells via Z_r, not pure dryness).
      'divide' – deficit sampled AT the zone's divide cell (``divide_idx[p]``), so
                 the SD_max ceiling is measured where the OPM sandbox actually runs.
                 Falls back to that zone's max-finite deficit, then *ws_default*,
                 when the divide cell has no SERVES data (cloud gap).

    For 'mean'/'max', SD_max[p] is reduced over ONLY the watershed cells that zone
    owns (cell_polygon == p) — the exact same nearest-station partition the
    rainfall uses.  Zones with no deficit data (or out-of-basin stations that own
    no cells) get *ws_default* (the watershed SD_max).  The result is therefore
    watershed-bounded and consistent with the rain zoning.
    """
    with rasterio.open(deficit_path) as src:
        deficit2d = src.read(1).astype(np.float64)
        nodata    = src.nodata
    if nodata is not None:
        deficit2d[deficit2d == nodata] = np.nan

    _to_np = lambda a: a.get() if hasattr(a, 'get') else np.asarray(a)
    sr = _to_np(s_rows); sc = _to_np(s_cols)

    # Guard against a cached raster on a different grid than the routing DEM.
    if sr.max() >= deficit2d.shape[0] or sc.max() >= deficit2d.shape[1]:
        print("  [WARN] deficit raster grid ≠ routing grid; "
              "per-zone SD → watershed SD.")
        return np.full(n_polygons, ws_default, dtype=np.float64)

    deficit_1d = deficit2d[sr, sc]                 # (n_cells,) deficit per cell
    finite     = np.isfinite(deficit_1d)

    if reducer == 'divide' and divide_idx is not None:
        div = _to_np(divide_idx).astype(np.intp)
        sd  = np.full(n_polygons, ws_default, dtype=np.float64)
        for p in range(n_polygons):
            v = deficit_1d[div[p]]
            if np.isfinite(v):
                sd[p] = float(v)                       # deficit at the divide cell
            else:
                m = (cell_polygon == p) & finite       # fallback: zone max-finite
                if m.any():
                    sd[p] = float(np.max(deficit_1d[m]))
                # else keep ws_default
        return np.maximum(sd, sd_min)

    redux = np.max if reducer == 'max' else np.mean
    sd = np.full(n_polygons, ws_default, dtype=np.float64)
    for p in range(n_polygons):
        m = (cell_polygon == p) & finite
        if m.any():
            sd[p] = float(redux(deficit_1d[m]))
    return np.maximum(sd, sd_min)


# Rawls, Brakensiek & Miller (1983) Green-Ampt wetting-front suction head [cm]
# by USDA texture class.  (Silt → silt-loam value; Rawls has no silt class.)
RAWLS_PSI_CM = {
    'sand': 4.95, 'loamy_sand': 6.13, 'sandy_loam': 11.01, 'loam': 8.89,
    'silt_loam': 16.68, 'silt': 16.68, 'sandy_clay_loam': 21.85,
    'clay_loam': 20.88, 'silty_clay_loam': 27.30, 'sandy_clay': 23.90,
    'silty_clay': 29.22, 'clay': 31.63,
}


def usda_psi_m(sand, clay):
    """
    Per-cell Green-Ampt suction ψ [m] from sand% / clay% via the USDA texture
    triangle → Rawls (1983) table.  Vectorised (numpy).  `silt = 100−sand−clay`.
    First-matching class wins (np.select), default = loam.
    """
    sand = np.asarray(sand, dtype=np.float64)
    clay = np.asarray(clay, dtype=np.float64)
    silt = 100.0 - sand - clay
    P = RAWLS_PSI_CM
    conds = [
        (clay >= 40) & (sand <= 45) & (silt < 40),                  # clay
        (clay >= 40) & (silt >= 40),                                # silty clay
        (clay >= 35) & (sand >= 45),                                # sandy clay
        (clay >= 27) & (clay < 40) & (sand > 20) & (sand <= 45),    # clay loam
        (clay >= 27) & (clay < 40) & (sand <= 20),                  # silty clay loam
        (clay >= 20) & (clay < 35) & (silt < 28) & (sand > 45),     # sandy clay loam
        (clay >= 7) & (clay < 27) & (silt >= 28) & (silt < 50) & (sand <= 52),  # loam
        ((silt >= 50) & (clay >= 12) & (clay < 27))
        | ((silt >= 50) & (silt < 80) & (clay < 12)),               # silt loam
        (silt >= 80) & (clay < 12),                                 # silt
        ((clay >= 7) & (clay < 20) & (sand > 52) & (silt + 2 * clay >= 30))
        | ((clay < 7) & (silt < 50) & (silt + 2 * clay >= 30)),     # sandy loam
        (silt + 1.5 * clay >= 15) & (silt + 1.5 * clay < 30),       # loamy sand
        (silt + 1.5 * clay < 15),                                   # sand
    ]
    psi_cm = [
        P['clay'], P['silty_clay'], P['sandy_clay'], P['clay_loam'],
        P['silty_clay_loam'], P['sandy_clay_loam'], P['loam'], P['silt_loam'],
        P['silt'], P['sandy_loam'], P['loamy_sand'], P['sand'],
    ]
    return np.select(conds, psi_cm, default=P['loam']) / 100.0   # cm → m
