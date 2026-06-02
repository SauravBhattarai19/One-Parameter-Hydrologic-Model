"""
serves_gee.py
=============
SERVES + LULC + SoilGrids pipeline on Google Earth Engine for OPM parameters.

Computes SD_max_initial from the SERVES soil-moisture deficit formula:
    SM_deficit = (porosity − θ_SERVES) × Z_r
where
    porosity   = 1 − bulk_density/particle_density (from OpenLandMap)
    θ_SERVES   = SERVES volumetric soil moisture [WP, FC]
    Z_r        = root zone depth [m] from ESA WorldCover LULC + lookup CSV
    SD_max     = max(SM_deficit) across the watershed

Also derives drainable porosity φ = mean(porosity − FC) from SoilGrids,
and K_sat from HiHydroSoil v2.0.

GEE datasets
------------
    ESA/WorldCover/v200/2021                             — LULC 10m
    LANDSAT/LC08/C02/T1_L2  +  LANDSAT/LC09/C02/T1_L2   — NDVI 30m
    COPERNICUS/S2_SR_HARMONIZED                          — NDVI 10m (alt)
    MODIS/061/MOD13A2                                    — NDVI 1km (alt)
    ISRIC/SoilGrids250m/v2_0/wv0033                      — field capacity
    ISRIC/SoilGrids250m/v2_0/wv1500                      — wilting point
    OpenLandMap/SOL/SOL_BULKDENS-FINEEARTH_USDA-4A1H_M/v02 — bulk density
    projects/sat-io/open-datasets/HiHydroSoilv2_0/ksat   — K_sat 250m

Authentication
--------------
    Priority: GOOGLE_APPLICATION_CREDENTIALS → GEE_SERVICE_ACCOUNT_KEY
              → ee.Initialize() default → ee.Authenticate()
"""

import os
import json
import math
import logging

import pandas as pd

logger = logging.getLogger(__name__)

try:
    import ee
    GEE_AVAILABLE = True
except ImportError:
    GEE_AVAILABLE = False

_DEPTH_BANDS = {
    'b0':   'val_0_5cm_mean',
    'b10':  'val_5_15cm_mean',
    'b30':  'val_15_30cm_mean',
    'b60':  'val_30_60cm_mean',
    'b100': 'val_60_100cm_mean',
    'b200': 'val_100_200cm_mean',
}

_PARTICLE_DENSITY = 2650.0

# SERVES coefficients (matches serves.js CONFIG)
_NDVI_COEFFICIENT = 1.33
_NDVI_INTERCEPT = -0.049
_LANDSAT_SCALE = 0.0000275
_LANDSAT_OFFSET = -0.2


def _authenticate(project=None):
    """Initialize GEE with the best available credentials."""
    proj = project or os.environ.get('GEE_PROJECT')
    init_kw = {'project': proj} if proj else {}

    sa_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    if sa_path and os.path.isfile(sa_path):
        try:
            credentials = ee.ServiceAccountCredentials(None, sa_path)
            ee.Initialize(credentials, **init_kw)
            logger.info("GEE authenticated via GOOGLE_APPLICATION_CREDENTIALS")
            return True
        except Exception as exc:
            logger.warning("Service account auth failed: %s", exc)

    sa_json = os.environ.get('GEE_SERVICE_ACCOUNT_KEY')
    if sa_json:
        try:
            key_data = json.loads(sa_json)
            credentials = ee.ServiceAccountCredentials(
                key_data['client_email'], key_data=sa_json
            )
            ee.Initialize(credentials, **init_kw)
            logger.info("GEE authenticated via GEE_SERVICE_ACCOUNT_KEY")
            return True
        except Exception as exc:
            logger.warning("Inline service account auth failed: %s", exc)

    try:
        ee.Initialize(**init_kw)
        logger.info("GEE authenticated via default credentials")
        return True
    except Exception:
        try:
            ee.Authenticate()
            ee.Initialize(**init_kw)
            logger.info("GEE authenticated via interactive flow")
            return True
        except Exception as exc:
            logger.warning("GEE authentication failed: %s", exc)
            return False


def _load_watershed_geometry(geojson_path, simplify_tolerance_m=None):
    """Load watershed GeoJSON → EPSG:4326 → ee.Geometry."""
    import geopandas as gpd

    gdf = gpd.read_file(geojson_path)
    gdf_4326 = gdf.to_crs("EPSG:4326")
    dissolved = gdf_4326.dissolve()
    geom = dissolved.geometry.iloc[0]

    if simplify_tolerance_m is not None and simplify_tolerance_m > 0:
        centroid = geom.centroid
        deg_per_m = 1.0 / (111320.0 * math.cos(math.radians(centroid.y)))
        tol_deg = simplify_tolerance_m * deg_per_m
        geom = geom.simplify(tol_deg, preserve_topology=True)

    if geom.geom_type == 'MultiPolygon':
        coords = [list(p.exterior.coords) for p in geom.geoms]
        return ee.Geometry.MultiPolygon(coords)
    else:
        coords = list(geom.exterior.coords)
        return ee.Geometry.Polygon(coords)


# ── NDVI retrieval (mirrors serves.js Sections 3–5) ─────────────────────────

def _get_ndvi_landsat(geometry, target_date, search_window):
    """Landsat 8/9 NDVI composite around target_date ± search_window days."""
    target = ee.Date(target_date)
    start = target.advance(-search_window, 'day')
    end = target.advance(search_window, 'day')

    def _mask_and_ndvi(image):
        qa = image.select('QA_PIXEL')
        mask = (qa.bitwiseAnd(1 << 1).eq(0)
                .And(qa.bitwiseAnd(1 << 3).eq(0))
                .And(qa.bitwiseAnd(1 << 4).eq(0))
                .And(qa.bitwiseAnd(1 << 5).eq(0)))
        nir = image.select('SR_B5').multiply(_LANDSAT_SCALE).add(_LANDSAT_OFFSET)
        red = image.select('SR_B4').multiply(_LANDSAT_SCALE).add(_LANDSAT_OFFSET)
        ndvi = nir.subtract(red).divide(nir.add(red)).clamp(-1, 1).rename('NDVI')
        return ndvi.updateMask(mask)

    l8 = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
          .filterBounds(geometry).filterDate(start, end)
          .filter(ee.Filter.lt('CLOUD_COVER', 100))
          .map(_mask_and_ndvi))
    l9 = (ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
          .filterBounds(geometry).filterDate(start, end)
          .filter(ee.Filter.lt('CLOUD_COVER', 100))
          .map(_mask_and_ndvi))

    return l8.merge(l9).median().clip(geometry)


def _get_ndvi_sentinel2(geometry, target_date, search_window):
    """Sentinel-2 NDVI composite."""
    target = ee.Date(target_date)
    start = target.advance(-search_window, 'day')
    end = target.advance(search_window, 'day')

    def _mask_and_ndvi(image):
        scl = image.select('SCL')
        mask = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7))
        nir = image.select('B8').divide(10000)
        red = image.select('B4').divide(10000)
        ndvi = nir.subtract(red).divide(nir.add(red)).clamp(-1, 1).rename('NDVI')
        return ndvi.updateMask(mask)

    s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
          .filterBounds(geometry).filterDate(start, end)
          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 100))
          .map(_mask_and_ndvi))

    return s2.median().clip(geometry)


def _get_ndvi_modis(geometry, target_date):
    """MODIS NDVI closest 16-day composite."""
    target = ee.Date(target_date)
    modis = (ee.ImageCollection('MODIS/061/MOD13A2')
             .filterBounds(geometry)
             .filterDate(target.advance(-16, 'day'), target.advance(16, 'day')))

    closest = ee.Image(
        modis.map(lambda img: img.set(
            'date_diff',
            ee.Date(img.get('system:time_start')).difference(target, 'day').abs()
        )).sort('date_diff').first()
    )
    return closest.select('NDVI').multiply(0.0001).rename('NDVI').clip(geometry)


def _get_ndvi(geometry, target_date, satellite, search_window):
    """Dispatch to the appropriate NDVI retriever."""
    if satellite == 'sentinel2':
        return _get_ndvi_sentinel2(geometry, target_date, search_window)
    elif satellite == 'modis':
        return _get_ndvi_modis(geometry, target_date)
    else:
        return _get_ndvi_landsat(geometry, target_date, search_window)


# ── Voronoi polygon builder ──────────────────────────────────────────────────

def _build_voronoi_polygons(gauge_csv_path, watershed_geojson_path, target_crs):
    """
    Build Voronoi (Thiessen) polygons from gauge locations, clip to watershed.

    Returns a GeoDataFrame in EPSG:4326 with one polygon per gauge (in gauge
    CSV order), or None on failure.
    """
    import geopandas as gpd
    from shapely.geometry import Point, MultiPoint
    from shapely.ops import voronoi_diagram

    gauges = pd.read_csv(gauge_csv_path)
    points = [Point(x, y) for x, y
              in zip(gauges['easting_m'], gauges['northing_m'])]

    ws = gpd.read_file(watershed_geojson_path)
    if ws.crs and str(ws.crs).upper() != target_crs.upper():
        ws = ws.to_crs(target_crs)
    ws_geom = ws.dissolve().geometry.iloc[0]

    regions = voronoi_diagram(MultiPoint(points), envelope=ws_geom.envelope)

    polygons = [None] * len(points)
    for region in regions.geoms:
        clipped = region.intersection(ws_geom)
        if clipped.is_empty:
            continue
        for i, pt in enumerate(points):
            if clipped.contains(pt):
                polygons[i] = clipped
                break

    if any(p is None for p in polygons):
        for i in range(len(polygons)):
            if polygons[i] is None:
                polygons[i] = points[i].buffer(1.0).intersection(ws_geom)

    gdf = gpd.GeoDataFrame(geometry=polygons, crs=target_crs)
    return gdf.to_crs("EPSG:4326")


# ── Main pipeline ────────────────────────────────────────────────────────────

def compute_opm_params(
    watershed_geojson_path,
    cell_size,
    lookup_csv_path,
    target_date,
    satellite='landsat',
    search_window=16,
    soil_depth_band='b30',
    project=None,
    gauge_csv_path=None,
    target_crs='EPSG:32645',
):
    """
    Single GEE pipeline: SERVES + LULC + SoilGrids → OPM parameters.

    SM_deficit = (porosity − θ_SERVES) × Z_r
    SD_max     = max(SM_deficit) over the watershed
    φ          = mean(porosity − FC)

    When gauge_csv_path is provided, also computes per-polygon max(deficit)
    using Voronoi (Thiessen) polygons built from gauge locations.

    Returns
    -------
    dict or None
        {sd_max, sd_min, phi, sd_max_per_polygon (if gauges), ...}
        sd_min is a fixed OPM floor of 0.001 m, not a SERVES-derived value.
    """
    if not GEE_AVAILABLE:
        logger.warning("earthengine-api not installed")
        return None

    if project and not os.environ.get('GEE_PROJECT'):
        os.environ['GEE_PROJECT'] = project

    if not _authenticate(project):
        return None

    try:
        # ── Load lookup CSV ──────────────────────────────────────────────
        lut = pd.read_csv(lookup_csv_path)
        from_codes = lut['class_code'].tolist()
        to_depths = lut['root_zone_depth_m'].tolist()

        band = _DEPTH_BANDS.get(soil_depth_band, 'val_15_30cm_mean')
        bdod_band = soil_depth_band

        geometry = _load_watershed_geometry(
            watershed_geojson_path,
            simplify_tolerance_m=cell_size,
        )

        # ── LULC → root zone depth Z_r ───────────────────────────────────
        lulc = ee.Image('ESA/WorldCover/v200/2021').select('Map')
        root_depth = lulc.remap(from_codes, to_depths).rename('root_depth')

        # ── NDVI ─────────────────────────────────────────────────────────
        ndvi = _get_ndvi(geometry, target_date, satellite, search_window)

        # ── Soil properties from SoilGrids ───────────────────────────────
        fc = ee.Image('ISRIC/SoilGrids250m/v2_0/wv0033').select(band) \
            .rename('fc')
        wp = ee.Image('ISRIC/SoilGrids250m/v2_0/wv1500').select(band) \
            .rename('wp')

        # ── Porosity from bulk density (OpenLandMap) ─────────────────────
        bdod = ee.Image(
            'OpenLandMap/SOL/SOL_BULKDENS-FINEEARTH_USDA-4A1H_M/v02'
        ).select(bdod_band)
        porosity = ee.Image(1).subtract(
            bdod.multiply(10).divide(_PARTICLE_DENSITY)
        ).rename('porosity')

        # ── SERVES θ (mirrors serves.js:452–466) ────────────────────────
        et_frac = ndvi.multiply(_NDVI_COEFFICIENT).add(_NDVI_INTERCEPT) \
            .clamp(0, 1).rename('et_frac')

        paw = fc.subtract(wp)   # plant available water
        theta = et_frac.multiply(paw).add(wp).rename('theta')
        theta = theta.where(theta.lt(wp), wp)
        theta = theta.where(theta.gt(fc), fc)

        # Handle water bodies: assign θ = FC (serves.js:518)
        water = ee.Image("JRC/GSW1_4/GlobalSurfaceWater") \
            .select('occurrence').gt(0)
        theta = theta.where(water, fc)

        # Handle negative NDVI: assign θ = FC (serves.js:522)
        neg_ndvi = ndvi.lt(0)
        theta = theta.where(neg_ndvi, fc)

        # ── SM_deficit = (porosity − θ) × Z_r ───────────────────────────
        deficit = porosity.subtract(theta).max(0) \
            .multiply(root_depth).rename('deficit')

        # ── Reduce over watershed ────────────────────────────────────────
        scale = max(float(cell_size), 30.0)  # floor at Landsat 30m

        deficit_stats = deficit.reduceRegion(
            reducer=ee.Reducer.max(),
            geometry=geometry,
            scale=scale,
            bestEffort=True,
            maxPixels=int(1e9),
        ).getInfo()

        sd_max_raw = deficit_stats.get('deficit')
        if sd_max_raw is None:
            logger.warning(
                "SERVES deficit returned null — check NDVI coverage for "
                "target date %s", target_date
            )
            return None

        sd_max = float(sd_max_raw)
        sd_min = 0.001

        if sd_max <= 0:
            logger.warning(
                "Computed SD_max=%.4f m is non-positive; "
                "soil may be fully saturated at target date.", sd_max
            )
            return None

        # ── θ diagnostics ────────────────────────────────────────────────
        theta_stats = theta.reduceRegion(
            reducer=ee.Reducer.minMax(),
            geometry=geometry,
            scale=scale,
            bestEffort=True,
            maxPixels=int(1e9),
        ).getInfo()

        # ── Root depth diagnostics ───────────────────────────────────────
        root_stats = root_depth.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geometry,
            scale=10.0,  # WorldCover native
            bestEffort=True,
            maxPixels=int(1e9),
        ).getInfo()

        # ── LULC class distribution ──────────────────────────────────────
        lulc_hist = lulc.reduceRegion(
            reducer=ee.Reducer.frequencyHistogram(),
            geometry=geometry,
            scale=10.0,
            bestEffort=True,
            maxPixels=int(1e9),
        ).getInfo()
        lulc_fractions = lulc_hist.get('Map', lulc_hist.get('remapped', {}))

        # ── φ = porosity − FC (porosity already computed above) ──────────
        phi_img = porosity.subtract(fc).rename('phi')

        phi_stats = phi_img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geometry,
            scale=max(float(cell_size), 250.0),
            bestEffort=True,
            maxPixels=int(1e9),
        ).getInfo()

        phi_mean = float(phi_stats.get('phi', 0.10))
        if phi_mean <= 0:
            logger.warning(
                "Computed φ=%.4f is non-positive; using default 0.10", phi_mean
            )
            phi_mean = 0.10

        # ── K_sat from HiHydroSoil v2.0 ─────────────────────────────────
        ksat_m_day = None
        ksat_per_polygon = None
        try:
            ksat_col = ee.ImageCollection(
                "projects/sat-io/open-datasets/HiHydroSoilv2_0/ksat"
            )
            # Raw values are int × 10000; multiply by 0.0001 to get cm/day
            ksat_img = ksat_col.mosaic().multiply(0.0001).rename('ksat')

            ksat_stats = ksat_img.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry,
                scale=250.0,
                bestEffort=True,
                maxPixels=int(1e9),
            ).getInfo()

            ksat_cm_day = ksat_stats.get('ksat')
            if ksat_cm_day is not None and float(ksat_cm_day) > 0:
                ksat_m_day = float(ksat_cm_day) / 100.0  # cm/day → m/day
            else:
                logger.warning("HiHydroSoil K_sat returned null/zero; "
                               "using config value")
        except Exception as exc:
            logger.warning("HiHydroSoil K_sat query failed: %s", exc)

        # ── Per-polygon stats via Voronoi ────────────────────────────────
        sd_max_per_polygon = None
        if gauge_csv_path:
            try:
                voronoi_gdf = _build_voronoi_polygons(
                    gauge_csv_path, watershed_geojson_path, target_crs
                )
                features = []
                for i, geom in enumerate(voronoi_gdf.geometry):
                    if geom.geom_type == 'MultiPolygon':
                        coords = [list(p.exterior.coords) for p in geom.geoms]
                        ee_geom = ee.Geometry.MultiPolygon(coords)
                    else:
                        coords = list(geom.exterior.coords)
                        ee_geom = ee.Geometry.Polygon(coords)
                    features.append(ee.Feature(ee_geom, {'zone': i}))

                voronoi_fc = ee.FeatureCollection(features)
                n_zones = len(voronoi_gdf)

                # Per-polygon max deficit
                per_poly = deficit.reduceRegions(
                    collection=voronoi_fc,
                    reducer=ee.Reducer.max(),
                    scale=scale,
                ).getInfo()

                zone_max = {}
                for feat in per_poly.get('features', []):
                    props = feat['properties']
                    zone_max[int(props['zone'])] = float(props.get('max', 0))

                sd_max_per_polygon = [
                    max(zone_max.get(i, sd_max), 0.001)
                    for i in range(n_zones)
                ]
                logger.info("Per-polygon SD_max: %s", sd_max_per_polygon)

                # Per-polygon mean K_sat
                if ksat_m_day is not None:
                    try:
                        ksat_poly = ksat_img.reduceRegions(
                            collection=voronoi_fc,
                            reducer=ee.Reducer.mean(),
                            scale=250.0,
                        ).getInfo()

                        zone_ksat = {}
                        for feat in ksat_poly.get('features', []):
                            props = feat['properties']
                            val = props.get('mean')
                            if val is not None and float(val) > 0:
                                zone_ksat[int(props['zone'])] = \
                                    float(val) / 100.0  # cm/day → m/day

                        ksat_per_polygon = [
                            zone_ksat.get(i, ksat_m_day)
                            for i in range(n_zones)
                        ]
                        logger.info("Per-polygon K_sat (m/day): %s",
                                    [f'{v:.2f}' for v in ksat_per_polygon])
                    except Exception as exc:
                        logger.warning("Per-polygon K_sat failed: %s", exc)

            except Exception as exc:
                logger.warning("Per-polygon Voronoi failed: %s — "
                               "using watershed-level values", exc)

        result = {
            'sd_max':              sd_max,
            'sd_min':              sd_min,
            'phi':                 phi_mean,
            'ksat_m_day':          ksat_m_day,
            'ksat_per_polygon':    ksat_per_polygon,
            'sd_max_per_polygon':  sd_max_per_polygon,
            'theta_min':           float(theta_stats.get('theta_min', 0)),
            'theta_max':           float(theta_stats.get('theta_max', 0)),
            'root_depth_mean':     float(root_stats.get('root_depth', 0)),
            'lulc_fractions':      lulc_fractions,
            'target_date':         target_date,
            'satellite':           satellite,
            'soil_depth_band':     soil_depth_band,
            'source':              'serves_lulc_gee',
        }
        return result

    except Exception as exc:
        logger.warning("GEE SERVES/LULC query failed: %s", exc)
        return None
