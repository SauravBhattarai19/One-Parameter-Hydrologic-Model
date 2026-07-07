import os
import rasterio
import rasterio.features
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pyproj import CRS, Transformer
import numpy as np
import geopandas as gpd
from shapely.geometry import shape

# Monkey-patch np.in1d for pysheds compatibility with numpy 2.0+
if not hasattr(np, 'in1d'):
    np.in1d = np.isin

import warnings
from pysheds.grid import Grid

# Suppress numba warnings from pysheds
warnings.filterwarnings("ignore", message="The TBB threading layer requires TBB version")

def reproject_dem(input_dem_path, output_dem_path, target_crs_epsg):
    """
    Reprojects a DEM to a target CRS.
    """
    print(f"Reprojecting DEM: {input_dem_path} to {target_crs_epsg}")
    with rasterio.open(input_dem_path) as src:
        # Determine the target CRS
        target_crs = CRS(target_crs_epsg)

        # Calculate the transform and dimensions for the reprojected DEM
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds
        )

        profile = src.profile
        profile.update({
            'crs': target_crs,
            'transform': transform,
            'width': width,
            'height': height,
            'nodata': src.nodata if src.nodata is not None else -9999 # Ensure nodata is set
        })

        with rasterio.open(output_dem_path, 'w', **profile) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=target_crs,
                resampling=Resampling.bilinear,
                num_threads=os.cpu_count()
            )
    print(f"Reprojected DEM saved to: {output_dem_path}")
    return output_dem_path

def _terrain(grid, dem, profile, output_dir):
    """
    Outlet-independent terrain analysis: fill sinks, resolve flats, flow
    direction and flow accumulation.  Writes filled_dem.tif, inflated_dem.tif,
    flow_direction.tif and flow_accumulation.tif.

    Returns (filled_dem, inflated_dem, flow_direction, flow_accumulation).

    This is the first half of the DEM stage — it needs no pour point, so the
    plugin can run it, show the stream network, and let the user pick an outlet
    against it before delineation (see ``analyze_terrain``).
    """
    # 1. Fill sinks
    print("Filling sinks...")
    filled_dem = grid.fill_depressions(dem)
    # Resolve flats to ensure all areas drain
    inflated_dem = grid.resolve_flats(filled_dem)

    with rasterio.open(os.path.join(output_dir,"filled_dem.tif"), 'w', **profile) as dst:
        dst.write(np.asarray(filled_dem), 1)
    print(f"Filled DEM saved to: {os.path.join(output_dir,'filled_dem.tif')}")

    # Save the hydrologically conditioned DEM (fill + resolve_flats).
    #
    # Why float32 matters for INTEGER source DEMs (e.g. SRTM int16):
    #   resolve_flats adds sub-metre increments (e.g. +0.001 m) to flat cells
    #   so the D8 algorithm can assign unambiguous flow directions.  If the file
    #   is written back as int16, those increments round to zero and the flat
    #   areas look slope-less again in the router → water pools → delayed surge.
    #   Saving as float32 preserves the increments exactly.
    #
    # If the source DEM is already floating-point the dtype is kept as-is;
    # the float32 upgrade only activates for integer-typed inputs.
    src_dtype = profile.get('dtype', 'float32')
    is_int_dem = np.dtype(src_dtype).kind in ('i', 'u')   # signed/unsigned int
    save_dtype = 'float32' if is_int_dem else src_dtype
    inflated_arr = np.asarray(inflated_dem).astype(save_dtype)
    # Replace any NaN or original nodata with a safe sentinel
    orig_nodata = profile.get('nodata')
    if orig_nodata is not None:
        inflated_arr[inflated_arr == np.array(orig_nodata, dtype=save_dtype)] = -9999.0
    inflated_arr[~np.isfinite(inflated_arr)] = -9999.0
    inflated_profile = profile.copy()
    inflated_profile.update(dtype=save_dtype, nodata=-9999.0)
    with rasterio.open(os.path.join(output_dir,"inflated_dem.tif"), 'w', **inflated_profile) as dst:
        dst.write(inflated_arr, 1)
    print(f"Inflated DEM (float32) saved to: {os.path.join(output_dir,'inflated_dem.tif')}")

    # 2. Flow direction
    print("Calculating flow direction...")
    # Specify directional mapping (N, NE, E, SE, S, SW, W, NW)
    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
    flow_direction = grid.flowdir(inflated_dem, dirmap=dirmap)

    fd_profile = profile.copy()
    fd_profile.update(dtype=flow_direction.dtype, nodata=None)
    with rasterio.open(os.path.join(output_dir,"flow_direction.tif"), 'w', **fd_profile) as dst:
        dst.write(np.asarray(flow_direction), 1)
    print(f"Flow direction saved to: {os.path.join(output_dir,'flow_direction.tif')}")

    # 3. Flow accumulation (contributing area)
    print("Calculating flow accumulation (contributing area)...")
    flow_accumulation = grid.accumulation(flow_direction, dirmap=dirmap)

    fa_profile = profile.copy()
    fa_profile.update(dtype=flow_accumulation.dtype, nodata=None)
    with rasterio.open(os.path.join(output_dir,"flow_accumulation.tif"), 'w', **fa_profile) as dst:
        dst.write(np.asarray(flow_accumulation), 1)
    print(f"Flow accumulation saved to: {os.path.join(output_dir,'flow_accumulation.tif')}")

    return filled_dem, inflated_dem, flow_direction, flow_accumulation


def _stream_mask(flow_accumulation):
    """The stream cells used both for outlet snapping and for the display layer.

    Top 1 % of cells by accumulation, minimum 1.  A fixed value (e.g. 1000)
    fails on coarse or small DEMs where the total cell count is below the
    threshold, leaving the stream mask empty.
    """
    n_cells_total = int(flow_accumulation.size)
    stream_threshold = max(1, n_cells_total // 100)
    return flow_accumulation > stream_threshold, stream_threshold, n_cells_total


def _delineate(grid, flow_direction, flow_accumulation, profile,
               output_point_latlon, target_crs_epsg, output_dir):
    """
    Outlet-dependent watershed delineation: transform the pour point, snap it to
    the stream network, run the catchment, and write watershed.tif /
    watershed.geojson.  Returns the watershed raster.

    This is the second half of the DEM stage; it consumes the terrain products
    from ``_terrain`` (in-memory in the all-in-one path, or reloaded from disk in
    ``delineate_from_outlet``).
    """
    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)

    # Transform output point to DEM's CRS
    src_crs_latlon = CRS("EPSG:4326") # Assuming input point is always lat/lon
    target_crs = CRS(target_crs_epsg)
    transformer = Transformer.from_crs(src_crs_latlon, target_crs, always_xy=True)

    # Convert lat/lon to projected coordinates
    output_point_x, output_point_y = transformer.transform(output_point_latlon[1], output_point_latlon[0])
    print(f"Output point (lat/lon): {output_point_latlon}")
    print(f"Output point (projected): ({output_point_x}, {output_point_y})")

    # Check bounds
    bounds = grid.bbox
    print(f"DEM Bounds (projected): {bounds}")
    if not (bounds[0] <= output_point_x <= bounds[2] and bounds[1] <= output_point_y <= bounds[3]):
        print("Warning: Output point is outside the DEM's bounds. Watershed delineation might fail or be empty.")

    print("Snapping outlet point to nearest stream cell...")
    streams, stream_threshold, n_cells_total = _stream_mask(flow_accumulation)
    print(f"  Stream threshold: {stream_threshold} cells  (1 % of {n_cells_total} total)")

    # Use pysheds built-in snap_to_mask: works in projected coordinate space
    # Returns (x, y) i.e. (easting, northing) of the snapped cell
    snap_x, snap_y = grid.snap_to_mask(streams, (output_point_x, output_point_y))
    print(f"Output point (projected): ({output_point_x:.2f}, {output_point_y:.2f})")
    print(f"Snapped outlet (projected): ({snap_x:.2f}, {snap_y:.2f})")
    print(f"Snapped cell accumulation: {flow_accumulation[grid.nearest_cell(snap_x, snap_y)[1], grid.nearest_cell(snap_x, snap_y)[0]]}")

    print("Delineating watershed...")
    # Use xytype='coordinate' so pysheds handles the col/row conversion internally
    watershed = grid.catchment(x=snap_x, y=snap_y, fdir=flow_direction, dirmap=dirmap, xytype='coordinate')

    ws_profile = profile.copy()
    watershed_uint8 = np.asarray(watershed).astype('uint8')
    ws_profile.update(dtype=watershed_uint8.dtype, nodata=0)
    with rasterio.open(os.path.join(output_dir,"watershed.tif"), 'w', **ws_profile) as dst:
        dst.write(watershed_uint8, 1)
    print(f"Watershed saved to: {os.path.join(output_dir,'watershed.tif')}")

    print("Exporting watershed to GeoJSON...")
    # Generate vector polygons from the raster
    shapes = rasterio.features.shapes(watershed_uint8, transform=profile['transform'])
    polygons = []
    for geom, val in shapes:
        if val == 1:
            polygons.append(shape(geom))

    if polygons:
        gdf = gpd.GeoDataFrame({'geometry': polygons}, crs=target_crs_epsg)
        gdf.to_file(os.path.join(output_dir,'watershed.geojson'), driver='GeoJSON')
        print(f"Watershed vector saved to: {os.path.join(output_dir,'watershed.geojson')}")

    return watershed


def _write_streams_geojson(flow_accumulation, profile, target_crs_epsg, output_dir):
    """Vectorise the stream mask to streams.geojson for on-canvas outlet picking.

    Additive display aid only — uses the exact same stream threshold as outlet
    snapping (``_stream_mask``) so what the user sees is what the snap targets.
    Does not touch any existing output file.
    """
    streams, stream_threshold, n_cells_total = _stream_mask(flow_accumulation)
    print(f"Vectorising streams for display (threshold {stream_threshold} cells "
          f"= 1 % of {n_cells_total})...")
    stream_uint8 = np.asarray(streams).astype('uint8')
    shapes = rasterio.features.shapes(stream_uint8, transform=profile['transform'])
    polygons = [shape(geom) for geom, val in shapes if val == 1]
    out_path = os.path.join(output_dir, "streams.geojson")
    if polygons:
        gdf = gpd.GeoDataFrame({'geometry': polygons}, crs=target_crs_epsg)
        gdf.to_file(out_path, driver='GeoJSON')
        print(f"Stream network saved to: {out_path}")
        return out_path
    print("No stream cells above threshold — streams.geojson not written.")
    return None


def perform_hydrological_analysis(dem_path, output_point_latlon, target_crs_epsg, output_dir):
    """
    Performs hydrological analysis: fill, flow direction, flow accumulation,
    and watershed delineation using pysheds.

    Behaviour is unchanged from before the terrain/delineation split — it simply
    runs ``_terrain`` then ``_delineate`` in the same order, writing the same
    files.  Kept as the single-shot entry point for the CLI / API / batch path.
    """
    print(f"Starting hydrological analysis on: {dem_path}")

    grid = Grid.from_raster(dem_path)
    dem = grid.read_raster(dem_path).astype(np.float64)

    # Read profile for saving later
    with rasterio.open(dem_path) as src:
        profile = src.profile.copy()

    filled_dem, inflated_dem, flow_direction, flow_accumulation = _terrain(
        grid, dem, profile, output_dir)

    watershed = _delineate(grid, flow_direction, flow_accumulation, profile,
                           output_point_latlon, target_crs_epsg, output_dir)

    return filled_dem, watershed, flow_accumulation, profile


def analyze_terrain(dem_path, target_crs_epsg, output_dir):
    """
    Phase 1 of the DEM stage for interactive use: reproject the DEM and run the
    outlet-independent terrain analysis, then vectorise the stream network so a
    user can pick the outlet against it on the map canvas.

    Writes reprojected_dem.tif, filled_dem.tif, inflated_dem.tif,
    flow_direction.tif, flow_accumulation.tif and streams.geojson.  No pour point
    required.  Returns a dict of output paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(dem_path):
        raise FileNotFoundError(
            f"DEM file not found at {dem_path}. Please provide a valid DEM_PATH.")

    # Reproject DEM (same first step as main())
    reprojected_dem_path = os.path.join(output_dir, "reprojected_dem.tif")
    reproject_dem(dem_path, reprojected_dem_path, target_crs_epsg)

    grid = Grid.from_raster(reprojected_dem_path)
    dem = grid.read_raster(reprojected_dem_path).astype(np.float64)
    with rasterio.open(reprojected_dem_path) as src:
        profile = src.profile.copy()

    _filled, _inflated, _fdir, flow_accumulation = _terrain(grid, dem, profile, output_dir)
    streams_path = _write_streams_geojson(flow_accumulation, profile, target_crs_epsg, output_dir)

    print("\n--- Terrain analysis complete ---")
    print("Pick an outlet on the stream network, then delineate the watershed.")
    return {
        "reprojected_dem":   reprojected_dem_path,
        "filled_dem":        os.path.join(output_dir, "filled_dem.tif"),
        "inflated_dem":      os.path.join(output_dir, "inflated_dem.tif"),
        "flow_direction":    os.path.join(output_dir, "flow_direction.tif"),
        "flow_accumulation": os.path.join(output_dir, "flow_accumulation.tif"),
        "streams":           streams_path,
    }


def delineate_from_outlet(output_dir, output_point_latlon, target_crs_epsg):
    """
    Phase 2 of the DEM stage for interactive use: given the terrain products
    already written by ``analyze_terrain``, snap the picked outlet to the stream
    network, delineate the watershed and clip the DEM / flow accumulation to it.

    Reloads flow_direction, flow_accumulation and the reprojected-DEM profile
    from disk (the same rasters ``analyze_terrain`` wrote), so the result is
    identical to the all-in-one ``perform_hydrological_analysis`` path.  Returns a
    dict of output paths.
    """
    reprojected_dem_path = os.path.join(output_dir, "reprojected_dem.tif")
    fdir_path = os.path.join(output_dir, "flow_direction.tif")
    facc_path = os.path.join(output_dir, "flow_accumulation.tif")
    inflated_dem_path = os.path.join(output_dir, "inflated_dem.tif")
    for p in (reprojected_dem_path, fdir_path, facc_path, inflated_dem_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing terrain product {p}. Run analyze_terrain first.")

    # Reload the grid and terrain rasters from disk.
    grid = Grid.from_raster(fdir_path)
    flow_direction = grid.read_raster(fdir_path)
    flow_accumulation = grid.read_raster(facc_path)
    with rasterio.open(reprojected_dem_path) as src:
        profile = src.profile.copy()

    watershed = _delineate(grid, flow_direction, flow_accumulation, profile,
                           output_point_latlon, target_crs_epsg, output_dir)

    # Clip DEM by watershed — use the inflated (fill + resolve_flats, float32)
    # DEM so slopes preserve the sub-metre gradients from resolve_flats.
    clipped_dem_path = os.path.join(output_dir, "clipped_dem.tif")
    clip_dem_by_watershed(inflated_dem_path, watershed, clipped_dem_path,
                          profile, nodata_fill=-9999.0)

    clipped_fa_path = os.path.join(output_dir, "clipped_flow_accumulation.tif")
    clip_flow_accumulation_by_watershed(flow_accumulation, watershed, clipped_fa_path, profile)

    print("\n--- Watershed delineation complete ---")
    return {
        "watershed_tif":     os.path.join(output_dir, "watershed.tif"),
        "watershed_geojson": os.path.join(output_dir, "watershed.geojson"),
        "clipped_dem":       clipped_dem_path,
        "clipped_flow_accumulation": clipped_fa_path,
    }

def clip_dem_by_watershed(original_dem_path, watershed_raster, output_clipped_dem_path,
                          dem_profile, nodata_fill=None):
    """
    Clips the original DEM by the delineated watershed boundary.

    nodata_fill : value written to outside-watershed pixels.  Defaults to
                  dem_profile['nodata'].  Pass explicitly when the source file
                  uses a different nodata convention (e.g. float32 with -9999).
    """
    print(f"Clipping DEM: {original_dem_path} by watershed...")
    if nodata_fill is None:
        nodata_fill = dem_profile['nodata']

    with rasterio.open(original_dem_path) as src:
        dem_data = src.read(1)
        clipped_dem_data = np.where(np.asarray(watershed_raster) > 0, dem_data, nodata_fill)

        clipped_profile = dem_profile.copy()
        clipped_profile.update(
            dtype=clipped_dem_data.dtype,
            nodata=nodata_fill,
        )

        with rasterio.open(output_clipped_dem_path, 'w', **clipped_profile) as dst:
            dst.write(clipped_dem_data, 1)
    print(f"Clipped DEM saved to: {output_clipped_dem_path}")
    return output_clipped_dem_path

def clip_flow_accumulation_by_watershed(flow_accumulation, watershed_raster, output_path, dem_profile):
    """
    Clips the flow accumulation raster by the delineated watershed boundary.
    Pixels outside the watershed are set to nodata (-1).
    The output represents contributing area (number of upstream cells) within
    the watershed only — i.e. the spatially distributed contributing area.
    """
    print("Clipping flow accumulation by watershed...")
    fa_array = np.asarray(flow_accumulation).astype(np.float32)
    ws_mask  = np.asarray(watershed_raster) > 0
    nodata_val = -1.0
    clipped_fa = np.where(ws_mask, fa_array, nodata_val)

    fa_profile = dem_profile.copy()
    fa_profile.update(dtype='float32', nodata=nodata_val)

    with rasterio.open(output_path, 'w', **fa_profile) as dst:
        dst.write(clipped_fa, 1)
    print(f"Clipped flow accumulation saved to: {output_path}")
    return output_path

def main(cfg):
    """
    Run the full DEM pre-processing pipeline.

    Parameters
    ----------
    cfg : object
        Any object exposing DEM_PATH, TARGET_CRS_EPSG, OUTPUT_POINT and
        OUTPUT_DIR attributes (vsa_opm.config.OpmConfig or equivalent).
    """
    dem_path            = cfg.DEM_PATH
    target_crs_epsg     = cfg.TARGET_CRS_EPSG
    output_point_latlon = cfg.OUTPUT_POINT
    output_dir          = cfg.OUTPUT_DIR

    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(dem_path):
        raise FileNotFoundError(
            f"DEM file not found at {dem_path}. Please provide a valid DEM_PATH.")

    # Reproject DEM
    reprojected_dem_path = os.path.join(output_dir, "reprojected_dem.tif")
    reprojected_dem_path = reproject_dem(dem_path, reprojected_dem_path, target_crs_epsg)

    # Perform hydrological analysis
    filled_dem, watershed, flow_accumulation, dem_profile = perform_hydrological_analysis(
        reprojected_dem_path, output_point_latlon, target_crs_epsg, output_dir)

    # Clip DEM by watershed — use the inflated (fill + resolve_flats, float32) DEM
    # so that slopes preserve the sub-metre gradients from resolve_flats.
    # The integer filled_dem rounds those increments to zero, collapsing flat
    # areas to slope=0 and causing a delayed drainage surge in the router.
    inflated_dem_path = os.path.join(output_dir, "inflated_dem.tif")
    clipped_dem_path  = os.path.join(output_dir, "clipped_dem.tif")
    clip_dem_by_watershed(inflated_dem_path, watershed, clipped_dem_path,
                          dem_profile, nodata_fill=-9999.0)

    # Clip flow accumulation (contributing area) by watershed
    clipped_fa_path = os.path.join(output_dir, "clipped_flow_accumulation.tif")
    clip_flow_accumulation_by_watershed(flow_accumulation, watershed, clipped_fa_path, dem_profile)

    print("""
--- Processing Complete ---""")
    print(f"All output files are saved in the '{output_dir}' directory.")
    print("Files include: filled_dem.tif, flow_direction.tif, flow_accumulation.tif, watershed.tif, clipped_dem.tif, clipped_flow_accumulation.tif")
