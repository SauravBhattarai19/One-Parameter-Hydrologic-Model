# VSA-OPM QGIS Plugin

A QGIS 3.x plugin that wraps the **Variable Source Area – One Parameter Model
(VSA-OPM)** distributed hydrologic pipeline (Pradhan & Ogden 2010), enabling full
model runs from within QGIS with optional GPU (CuPy/CUDA) acceleration.

---

## Features

| Feature | Detail |
|---|---|
| DEM pre-processing | Reproject → fill sinks → D8 flow direction/accumulation → watershed delineation (pysheds) |
| Precipitation engines | Uniform · Thiessen · IDW · **NASA GPM IMERG V07 satellite** (Thiessen/IDW) via Google Earth Engine |
| Runoff generation | None · Runoff coefficient · Pre-computed raster series · SCS-CN · **VSA-OPM** |
| OPM mechanisms | Composable: **VSA** saturation-excess + **Green-Ampt** infiltration-excess (Horton) + **impervious** urban shedding |
| Satellite parameters | **SERVES** soil-moisture deficit, SoilGrids porosity/texture suction, HiHydroSoil vertical Ksat (all via GEE, optional) |
| Routing | **Kinematic** or **diffusive-wave** (CASC2D/GSSHA), confined channel cross-sections, **adaptive CFL** time-stepping |
| Spatial roughness | Manning's n from scalar · ESA WorldCover (LULC) · WUDAPT LCZ · raster, with channel-cell override |
| Diagnostics | Per-run mass-balance report |
| GPU acceleration | CuPy/CUDA backend (optional; falls back to CPU automatically) |
| QGIS Processing | DEM pre-processing + routing exposed as Processing Algorithms (Graphical Modeller compatible) |
| Results viewer | Embedded hydrograph plot · one-click layer loading · CSV/PNG export |

---

## Installation

### A. Prebuilt folder (recommended — no build step)

The repository ships a self-contained plugin folder at
`_plugin_build/vsa_opm_plugin/` that bundles the plugin **and** a vendored
copy of the `vsa_opm` core package (under `_vendor/`).

1. Pull the repository.
2. **Remove any old `vsa_opm` plugin folder** (the plugin was renamed so it no
   longer shadows the core package's import name), then copy
   `_plugin_build/vsa_opm_plugin/` into your QGIS plugins directory:
   - **Windows:** `%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\`
   - **Linux:** `~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/`
   - **macOS:** `~/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/`
3. QGIS → **Plugins → Manage and Install Plugins → Installed → ✓ VSA-OPM Hydrological Model**

### B. Rebuild the package yourself

```bash
cd /path/to/OPM
./build_windows_plugin.sh      # regenerates _plugin_build/vsa_opm_plugin/ and vsa_opm_windows.zip
```

### C. Development symlink (Linux/macOS)

```bash
./install_plugin.sh            # symlinks qgis_plugin/ → the QGIS plugins dir
./install_plugin.sh --remove   # uninstall
```
> In symlink mode the plugin finds the core package in the repository root
> automatically (`bridge.ensure_core()`); pip-installing `vsa-opm` into the
> QGIS interpreter also works for any install mode.

---

## Google Earth Engine (for IMERG / SERVES / LULC / LCZ)

The satellite-backed options (IMERG rainfall, SERVES deficit, gridded Ksat,
texture suction, LULC/LCZ Manning's & impervious) need Earth Engine access.

- Set the **GEE project** on the *DEM & Watershed* tab (or the `GEE_PROJECT`
  environment variable).
- Authenticate once in the QGIS Python console:
  ```python
  import ee
  ee.Authenticate()
  ```
- Or place a service-account `key.json` next to `serves_gee.py` inside the
  installed plugin's `_vendor/vsa_opm/gee/` folder. **`key.json` is never
  committed to git** — you must supply your own.
- These options are entirely optional: scalar/manual/gauge settings run fully
  offline.

---

## Dependencies

All dependencies must be importable from **QGIS's bundled Python** (NOT your
system Python — a normal `pip install` in a terminal targets the wrong
interpreter and QGIS will still fail with `ModuleNotFoundError`).

### Easiest: the built-in Dependencies manager

Open the plugin dialog and click **🔧 Dependencies**. It shows which packages
are present in QGIS's Python and installs the missing ones into that same
interpreter with one click (with a live log and a copyable manual command if an
install needs elevated permissions). The dialog also pops up automatically if
you press **Run** while something required is missing.

| Package | Purpose | Required? |
|---|---|---|
| `numpy`, `pandas` | Array math / CSV I/O | required (usually pre-installed) |
| `scipy` | Spatial weights (KDTree) | required |
| `rasterio` | Raster read/write | required |
| `pysheds` | DEM watershed delineation | required |
| `matplotlib` | Embedded hydrograph plot | optional |
| `earthengine-api` | GEE / IMERG / SERVES | optional (satellite features) |
| `cupy` | GPU acceleration | optional (`pip install cupy-cuda12x`) |

### Manual fallback (Windows)

If a one-click install hits a permission error, open **Start → OSGeo4W →
OSGeo4W Shell** and run the command the Dependencies dialog shows, e.g.:
```
python -m pip install rasterio pysheds scipy pandas
```
then restart QGIS. On Linux/macOS run the same command against the QGIS Python.

**Check inside the QGIS Python console:**
```python
import rasterio, pysheds, scipy, numpy, pandas, matplotlib
import ee      # optional (satellite features)
import cupy    # optional (GPU)
```

---

## The 5-tab dialog

1. **DEM & Watershed** — DEM, CRS, outlet point (or pick from map), output dir,
   event start (UTC), UTC offset, GEE project.
2. **Precipitation** — uniform / Thiessen / IDW / IMERG (Thiessen/IDW).
3. **Runoff** — source + full OPM panel: mechanisms, core parameters, SERVES/GEE
   soil moisture, Green-Ampt infiltration, impervious, baseflow.
4. **Routing** — Manning's source, kinematic/diffusive scheme, confined channel
   routing, static or adaptive-CFL time-stepping, compute backend, mass balance.
5. **Results** — hydrograph plot, layer loading, CSV/PNG export.

Every knob maps 1:1 to an attribute on `OpmConfig` — defined once in the core
package (`vsa_opm/config.py`) and re-exported by `bridge/config_bridge.py` for
backward compatibility. **Save Config** exports a `config.py`-compatible file
with all parameters, which also loads via `vsa-opm run -c <file>`.

---

## GPU / CPU Notes

- The plugin detects CuPy at load time (`gpu_utils.cupy_available()`); the GPU
  radio button is disabled with an install hint when CuPy is absent.
- Runs execute in a `QThread` — the QGIS GUI stays responsive.
- GPU memory pools are freed after each run; a VRAM check warns if < 0.5 GB free.

---

## Running Tests

```bash
cd /path/to/OPM
pytest qgis_plugin/tests/test_config_bridge.py -v   # no QGIS required
pytest qgis_plugin/tests/test_runner.py -v          # needs output/ rasters
```

---

## Reference

Pradhan, N.R. and Ogden, F.L. (2010). *Development of a one-parameter variable
source area runoff model for ungauged basins.* Advances in Water Resources,
33(5), pp. 572–584.
