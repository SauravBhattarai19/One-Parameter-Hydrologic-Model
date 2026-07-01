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

## Running the model

```bash
pip install -r requirements.txt
```

1. Configure a scenario in [`config.py`](config.py) — DEM path, outlet point, event
   window, and which runoff/routing options to use.
2. Run via [`tools/runners/`](tools/runners) (dispatches by `PRECIP_METHOD` — Thiessen,
   IDW, or IMERG-driven variants) or directly with `vsa_opm.py`.
3. Outputs (hydrograph, mass-balance diagnostics, rasters) are written under the
   scenario's `OUTPUT_DIR`.

Optional Earth Engine integration (`OPM_SD_SOURCE='gee'`, IMERG precipitation)
needs a GEE service account — see [`serves_gee.py`](serves_gee.py) and
`test_ee_auth.sh` for setup/verification.

## Repository layout

- `vsa_opm.py`, `runoff_input.py` — VSA/Green-Ampt runoff generation
- `kinematic_wave_router.py`, `routing_utils.py` — channel/overland routing
- `serves_gee.py`, `imerg_gee.py`, `process_dem.py` — satellite data & DEM pre-processing
- `config.py` — single source of truth for all scenario parameters
- `tools/` — batch runners, sensitivity/OFAT sweeps, experiment combinations
- `study/` — source for the interactive course site above
- `docs/` — LaTeX lecture notes companion to the course
- `qgis_plugin/` — QGIS front-end for the model
