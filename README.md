# OPM — One-Parameter Hydrologic Model

A distributed, physics-based rainfall–runoff and flood-routing model built around
a **Variable Source Area (VSA)** runoff scheme, Green-Ampt infiltration, and
explicit kinematic/diffusive-wave channel routing — driven by open satellite
data (SERVES soil moisture, IMERG precipitation, SoilGrids, LULC/LCZ) via
Google Earth Engine.

## 📖 Learn the model — interactive course

**[sauravbhattarai19.github.io/One-Parameter-Hydrologic-Model](https://sauravbhattarai19.github.io/One-Parameter-Hydrologic-Model/)**

This is the place to actually understand how the model works. It's a free,
open-source interactive textbook (built from the [`study/`](study) folder) that
walks through the physics from the ground up, with in-browser simulations —
no installation required:

| Chapter | Topic |
|---|---|
| 1 | Digital Elevation Models & Flow Direction (D8, flow accumulation, pit filling) |
| 2 | Watershed Delineation (catchment boundaries, stream networks, pour points) |
| 3 | Rainfall–Runoff Generation (VSA, Green-Ampt, impervious shedding, satellite pipeline) |
| 4 | Kinematic-Wave Routing (Saint-Venant simplification, Manning's equation) |
| 5 | Diffusive-Wave Routing (backwater effects, GSSHA-style conveyance, numerical diffusion) |

If you're new to this repo, start there before digging into the code below.

## Installation

The model is a pip-installable package (`vsa_opm`):

```bash
pip install .            # core (CPU)
pip install .[gpu]       # + CuPy/CUDA acceleration
pip install .[gee]       # + Google Earth Engine (IMERG, SERVES, SoilGrids, LULC/LCZ)
```

## Running the model

One core drives three interfaces:

**1. Python API**

```python
from vsa_opm import OpmConfig, run_pipeline

cfg = OpmConfig(DEM_PATH="dem.tif", OUTPUT_DIR="results/")
cfg.update_output_paths()
results = run_pipeline(cfg, stages=("process_dem", "routing"))
```

**2. CLI** — config-file driven (YAML, JSON, or a legacy flat `.py` module):

```bash
vsa-opm init-config -o my_run.yaml     # template with every parameter
vsa-opm validate -c my_run.yaml        # pre-flight checks
vsa-opm run -c my_run.yaml             # process_dem + routing
```

See [`configs/example_config.yaml`](configs/example_config.yaml) for the
repository's research scenario in CLI form.

**3. QGIS plugin** — see [`qgis_plugin/README.md`](qgis_plugin/README.md); the
plugin imports the same `vsa_opm` package (pip-installed or vendored in the
plugin zip).

For the repository's batch research workflows, configure
[`config.py`](config.py) (legacy scenario module) and run
[`tools/runner.py`](tools/runner.py), which dispatches by `PRECIP_METHOD`
(Thiessen, IDW, or IMERG-driven variants).

Outputs (hydrograph, mass-balance diagnostics, rasters) are written under the
scenario's `OUTPUT_DIR`.  Optional Earth Engine integration
(`OPM_SD_SOURCE='gee'`, IMERG precipitation) needs a GEE service account — see
[`vsa_opm/gee/serves_gee.py`](vsa_opm/gee/serves_gee.py) and `test_ee_auth.sh`
for setup/verification.

## Repository layout

- `vsa_opm/` — the pip-installable model package
  - `core/` — the science (QGIS-free)
    - `routing/` — `terrain.py` (D8/slopes/topological order), `hydraulics.py`
      (Manning + diffusive-wave kernels), `surface.py` (Manning's n, channel
      geometry, impervious), `router.py` (the time loop), `reporting.py`
      (hydrograph/mass balance), `gpu.py` (CuPy variants)
    - `runoff/` — `engine.py` (RunoffEngine dispatcher), `vsa.py` (VSA-OPM /
      Green-Ampt / impervious mechanics), `soil.py` (SD_max/phi/suction
      resolution), `gpu.py`
    - `precip/` — `engine.py` (uniform/Thiessen/IDW/IMERG), `gpu.py`
    - `dem_processing.py` — watershed preprocessing (pysheds)
    - `opm.py` — standalone OPM runner; `io_utils.py` — shared raster helpers
  - `gee/` — Earth Engine integrations (`auth.py`, `serves_gee.py`, `imerg_gee.py`)
  - `cli/` — the `vsa-opm` command-line interface
  - `config.py` — `OpmConfig`, the single configuration object
  - `pipeline.py` — stage orchestration shared by API, CLI and plugin
- `config.py` — legacy research-scenario settings used by `tools/` and `tests/`
- `configs/` — example CLI config files
- `tools/` — batch runners, sensitivity/OFAT sweeps, experiment combinations
- `study/` — source for the interactive course site above
- `docs/` — LaTeX lecture notes companion to the course
- `qgis_plugin/` — QGIS front-end (UI + QThread worker importing `vsa_opm`)
