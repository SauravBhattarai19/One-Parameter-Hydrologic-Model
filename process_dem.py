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

# Create output directory if it doesn't exist
if not os.path.exists("output"):
    os.makedirs("output")

# Import configuration
try:
    import config
    DEM_PATH = config.DEM_PATH
    TARGET_CRS_EPSG = config.TARGET_CRS_EPSG
    OUTPUT_POINT_LATLON = config.OUTPUT_POINT
    OUTPUT_DIR = config.OUTPUT_DIR
except ImportError:
    print("Error: config.py not found or invalid. Please ensure it exists and is correctly configured.")
    exit()
except AttributeError as e:
    print(f"Error in config.py: {e}. Please ensure all required variables are defined.")
    exit()

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

def perform_hydrological_analysis(dem_path, output_point_latlon, target_crs_epsg):
    """
    Performs hydrological analysis: fill, flow direction, flow accumulation,
    and watershed delineation using pysheds.
    """
    print(f"Starting hydrological analysis on: {dem_path}")
    
    grid = Grid.from_raster(dem_path)
    dem = grid.read_raster(dem_path).astype(np.float64)

    # Read profile for saving later
    with rasterio.open(dem_path) as src:
        profile = src.profile.copy()

    # 1. Fill sinks
    print("Filling sinks...")
    filled_dem = grid.fill_depressions(dem)
    # Resolve flats to ensure all areas drain
    inflated_dem = grid.resolve_flats(filled_dem)
    
    with rasterio.open(os.path.join(OUTPUT_DIR, "filled_dem.tif"), 'w', **profile) as dst:
        dst.write(np.asarray(filled_dem), 1)
    print(f"Filled DEM saved to: {os.path.join(OUTPUT_DIR, 'filled_dem.tif')}")

    # 2. Flow direction
    print("Calculating flow direction...")
    # Specify directional mapping (N, NE, E, SE, S, SW, W, NW)
    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
    flow_direction = grid.flowdir(inflated_dem, dirmap=dirmap)
    
    fd_profile = profile.copy()
    fd_profile.update(dtype=flow_direction.dtype, nodata=None)
    with rasterio.open(os.path.join(OUTPUT_DIR, "flow_direction.tif"), 'w', **fd_profile) as dst:
        dst.write(np.asarray(flow_direction), 1)
    print(f"Flow direction saved to: {os.path.join(OUTPUT_DIR, 'flow_direction.tif')}")

    # 3. Flow accumulation (contributing area)
    print("Calculating flow accumulation (contributing area)...")
    flow_accumulation = grid.accumulation(flow_direction, dirmap=dirmap)
    
    fa_profile = profile.copy()
    fa_profile.update(dtype=flow_accumulation.dtype, nodata=None)
    with rasterio.open(os.path.join(OUTPUT_DIR, "flow_accumulation.tif"), 'w', **fa_profile) as dst:
        dst.write(np.asarray(flow_accumulation), 1)
    print(f"Flow accumulation saved to: {os.path.join(OUTPUT_DIR, 'flow_accumulation.tif')}")

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
    # Define stream mask: cells with high accumulation
    stream_threshold = 1000
    streams = flow_accumulation > stream_threshold
    
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
    with rasterio.open(os.path.join(OUTPUT_DIR, "watershed.tif"), 'w', **ws_profile) as dst:
        dst.write(watershed_uint8, 1)
    print(f"Watershed saved to: {os.path.join(OUTPUT_DIR, 'watershed.tif')}")

    print("Exporting watershed to GeoJSON...")
    # Generate vector polygons from the raster
    shapes = rasterio.features.shapes(watershed_uint8, transform=profile['transform'])
    polygons = []
    for geom, val in shapes:
        if val == 1:
            polygons.append(shape(geom))

    if polygons:
        gdf = gpd.GeoDataFrame({'geometry': polygons}, crs=target_crs_epsg)
        gdf.to_file(os.path.join(OUTPUT_DIR, 'watershed.geojson'), driver='GeoJSON')
        print(f"Watershed vector saved to: {os.path.join(OUTPUT_DIR, 'watershed.geojson')}")

    return filled_dem, watershed, flow_accumulation, profile

def clip_dem_by_watershed(original_dem_path, watershed_raster, output_clipped_dem_path, dem_profile):
    """
    Clips the original DEM by the delineated watershed boundary.
    """
    print(f"Clipping DEM: {original_dem_path} by watershed...")
    with rasterio.open(original_dem_path) as src:
        dem_data = src.read(1)
        # Set pixels outside the watershed to nodata
        clipped_dem_data = np.where(np.asarray(watershed_raster) > 0, dem_data, dem_profile['nodata'])
        
        # Update profile for clipped DEM
        clipped_profile = dem_profile.copy()
        clipped_profile.update(
            dtype=clipped_dem_data.dtype,
            nodata=dem_profile['nodata']
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

def main():
    if not os.path.exists(DEM_PATH):
        print(f"Error: DEM file not found at {DEM_PATH}. Please provide a valid path in config.py.")
        exit()

    # Reproject DEM
    reprojected_dem_path = os.path.join(OUTPUT_DIR, "reprojected_dem.tif")
    reprojected_dem_path = reproject_dem(DEM_PATH, reprojected_dem_path, TARGET_CRS_EPSG)

    # Perform hydrological analysis
    filled_dem, watershed, flow_accumulation, dem_profile = perform_hydrological_analysis(reprojected_dem_path, OUTPUT_POINT_LATLON, TARGET_CRS_EPSG)

    # Clip DEM by watershed — use the filled (hydrologically conditioned) DEM
    # so that slopes are consistent with the flow direction / accumulation grids.
    filled_dem_path  = os.path.join(OUTPUT_DIR, "filled_dem.tif")
    clipped_dem_path = os.path.join(OUTPUT_DIR, "clipped_dem.tif")
    clip_dem_by_watershed(filled_dem_path, watershed, clipped_dem_path, dem_profile)

    # Clip flow accumulation (contributing area) by watershed
    clipped_fa_path = os.path.join(OUTPUT_DIR, "clipped_flow_accumulation.tif")
    clip_flow_accumulation_by_watershed(flow_accumulation, watershed, clipped_fa_path, dem_profile)

    print("""
--- Processing Complete ---""")
    print(f"All output files are saved in the '{OUTPUT_DIR}' directory.")
    print("Files include: filled_dem.tif, flow_direction.tif, flow_accumulation.tif, watershed.tif, clipped_dem.tif, clipped_flow_accumulation.tif")

if __name__ == "__main__":
    main()
