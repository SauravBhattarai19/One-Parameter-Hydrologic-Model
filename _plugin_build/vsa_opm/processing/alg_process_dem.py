# -*- coding: utf-8 -*-
"""
alg_process_dem.py
==================
QGIS Processing Algorithm: DEM Pre-processing.

Wraps process_dem.main() — reprojects DEM, fills sinks, computes flow
direction and accumulation, delineates watershed.

Inputs
------
  INPUT_DEM       : raster layer or file path
  TARGET_CRS      : target coordinate reference system
  OUTLET_LAT      : outlet latitude  (WGS-84 decimal degrees)
  OUTLET_LON      : outlet longitude (WGS-84 decimal degrees)
  OUTPUT_DIR      : folder for output rasters

Outputs
-------
  WATERSHED_TIF   : output/watershed.tif
  CLIPPED_DEM     : output/clipped_dem.tif
  FLOW_DIRECTION  : output/flow_direction.tif
  FLOW_ACCUM      : output/clipped_flow_accumulation.tif
"""

import os
import sys

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterCrs,
    QgsProcessingParameterNumber,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterRasterDestination,
    QgsProcessingOutputRasterLayer,
    QgsProcessingOutputVectorLayer,
    QgsProcessingContext,
    QgsProcessingFeedback,
)

# Plugin root is 2 levels up from this file: vsa_opm/processing/ → vsa_opm/
_OPM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _ensure_opm_path():
    if _OPM_ROOT not in sys.path:
        sys.path.insert(0, _OPM_ROOT)


class ProcessDemAlgorithm(QgsProcessingAlgorithm):
    """QGIS Processing wrapper for process_dem.main()."""

    # Parameter IDs
    INPUT_DEM = "INPUT_DEM"
    TARGET_CRS = "TARGET_CRS"
    OUTLET_LAT = "OUTLET_LAT"
    OUTLET_LON = "OUTLET_LON"
    OUTPUT_DIR = "OUTPUT_DIR"

    def createInstance(self):  # noqa: N802
        return ProcessDemAlgorithm()

    def name(self):
        return "process_dem"

    def displayName(self):  # noqa: N802
        return "1. DEM Pre-processing"

    def group(self):
        return "VSA-OPM Pipeline"

    def groupId(self):  # noqa: N802
        return "vsaopm_pipeline"

    def shortHelpString(self):  # noqa: N802
        return (
            "Reprojects the DEM to the target CRS, fills sinks, computes D8 "
            "flow direction and accumulation, snaps the outlet point to the "
            "nearest stream cell, and delineates the watershed using pysheds.\n\n"
            "Outputs: watershed.tif, clipped_dem.tif, flow_direction.tif, "
            "clipped_flow_accumulation.tif, watershed.geojson."
        )

    def initAlgorithm(self, config=None):  # noqa: N802
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_DEM, "DEM raster"
            )
        )
        self.addParameter(
            QgsProcessingParameterCrs(
                self.TARGET_CRS, "Target CRS", defaultValue="EPSG:32645"
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.OUTLET_LAT, "Outlet latitude (WGS-84)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=27.632222, minValue=-90.0, maxValue=90.0
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.OUTLET_LON, "Outlet longitude (WGS-84)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=85.293333, minValue=-180.0, maxValue=180.0
            )
        )
        self.addParameter(
            QgsProcessingParameterFolderDestination(
                self.OUTPUT_DIR, "Output directory"
            )
        )

    def processAlgorithm(self, parameters, context, feedback):  # noqa: N802
        # QGIS Processing sets sys.stderr/stdout to None;
        # numpy's C extension needs them writable during import.
        import io
        if sys.stderr is None:
            sys.stderr = io.StringIO()
        if sys.stdout is None:
            sys.stdout = io.StringIO()

        _ensure_opm_path()

        from ..bridge.config_bridge import OpmConfig
        import process_dem as pd_mod

        # Build config
        dem_layer = self.parameterAsRasterLayer(parameters, self.INPUT_DEM, context)
        dem_path = dem_layer.source() if dem_layer else parameters[self.INPUT_DEM]
        crs = self.parameterAsCrs(parameters, self.TARGET_CRS, context)
        lat = self.parameterAsDouble(parameters, self.OUTLET_LAT, context)
        lon = self.parameterAsDouble(parameters, self.OUTLET_LON, context)
        out_dir = self.parameterAsString(parameters, self.OUTPUT_DIR, context)

        os.makedirs(out_dir, exist_ok=True)

        # Monkey-patch module globals (process_dem reads them at module level)
        pd_mod.DEM_PATH = dem_path
        pd_mod.TARGET_CRS_EPSG = crs.authid()
        pd_mod.OUTPUT_POINT_LATLON = (lat, lon)
        pd_mod.OUTPUT_DIR = out_dir

        feedback.setProgress(5)
        feedback.pushInfo("Running DEM pre-processing …")

        pd_mod.main()

        feedback.setProgress(95)

        # ── Auto-load output layers into the QGIS map canvas ─────────────────
        outputs = {
            "WATERSHED_TIF":  (os.path.join(out_dir, "watershed.tif"),        "Watershed",             "raster"),
            "CLIPPED_DEM":    (os.path.join(out_dir, "clipped_dem.tif"),       "Clipped DEM",           "raster"),
            "FLOW_DIRECTION": (os.path.join(out_dir, "flow_direction.tif"),    "Flow Direction",        "raster"),
            "FLOW_ACCUM":     (os.path.join(out_dir, "clipped_flow_accumulation.tif"), "Flow Accumulation", "raster"),
        }

        # Also load watershed GeoJSON vector if it was created
        geojson_path = os.path.join(out_dir, "watershed.geojson")
        if os.path.exists(geojson_path):
            outputs["WATERSHED_VEC"] = (geojson_path, "Watershed Boundary", "vector")

        from qgis.core import QgsProcessingContext, QgsProject

        results = {}
        for key, (path, name, layer_type) in outputs.items():
            if not os.path.exists(path):
                feedback.pushInfo(f"  Skipping {name} — file not found: {path}")
                continue

            feedback.pushInfo(f"  Loading layer: {name}")
            details = QgsProcessingContext.LayerDetails(name, QgsProject.instance(), key)
            context.addLayerToLoadOnCompletion(path, details)
            results[key] = path

        feedback.setProgress(100)
        feedback.pushInfo(
            f"\nDEM Pre-processing complete. {len(results)} layers added to the map."
        )
        return results
