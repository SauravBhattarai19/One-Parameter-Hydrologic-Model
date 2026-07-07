# -*- coding: utf-8 -*-
"""
vsa_opm — Variable Source Area / One-Parameter Model (Pradhan & Ogden 2010)
distributed hydrologic model.

Subpackages
-----------
core   : the science — DEM preprocessing, precipitation, runoff generation
         (VSA / Green-Ampt / impervious), kinematic/diffusive-wave routing.
         Pure NumPy/SciPy/rasterio; no QGIS, no Qt.
gee    : Google Earth Engine integrations (IMERG rainfall, SERVES deficit,
         SoilGrids, LULC/LCZ).  Optional; requires earthengine-api.
utils  : shared helpers (CPU/GPU backend selection).
cli    : the ``vsa-opm`` command-line interface (config-file driven runs).

Quick start
-----------
    from vsa_opm import OpmConfig, run_pipeline

    cfg = OpmConfig(DEM_PATH="dem.tif", OUTPUT_DIR="results/")
    cfg.update_output_paths()
    results = run_pipeline(cfg, stages=("process_dem", "routing"))
"""

__version__ = "2.0.0"

from .config import OpmConfig
from .pipeline import run_pipeline, DEFAULT_STAGES

__all__ = ["OpmConfig", "run_pipeline", "DEFAULT_STAGES", "__version__"]
