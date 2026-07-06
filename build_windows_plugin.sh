#!/usr/bin/env bash
# =============================================================================
# build_windows_plugin.sh
# =============================================================================
# Builds a self-contained vsa_opm_windows.zip.
# When extracted, it produces a single "vsa_opm/" folder that the user
# drops directly into their QGIS plugins directory. No extra steps needed.
#
# QGIS plugins dir on Windows:
#   %APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\
#
# Usage:
#   chmod +x build_windows_plugin.sh
#   ./build_windows_plugin.sh
# =============================================================================

set -euo pipefail

OPM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$OPM_ROOT/_plugin_build"
PLUGIN_DIR="$BUILD_DIR/vsa_opm"
TARGET_ZIP="$OPM_ROOT/vsa_opm_windows.zip"

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║   VSA-OPM — Windows Plugin Package Builder      ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# ── Clean ─────────────────────────────────────────────────────────────────────
echo "▶ Cleaning old build..."
rm -rf "$BUILD_DIR" "$TARGET_ZIP"
mkdir -p "$PLUGIN_DIR"

# ── Copy the QGIS plugin structure ────────────────────────────────────────────
echo "▶ Copying plugin structure (bridge/, ui/, processing/, resources/)..."
cp -r "$OPM_ROOT/qgis_plugin/bridge"      "$PLUGIN_DIR/bridge"
cp -r "$OPM_ROOT/qgis_plugin/ui"          "$PLUGIN_DIR/ui"
cp -r "$OPM_ROOT/qgis_plugin/processing"  "$PLUGIN_DIR/processing"
cp -r "$OPM_ROOT/qgis_plugin/resources"   "$PLUGIN_DIR/resources"
cp    "$OPM_ROOT/qgis_plugin/__init__.py"  "$PLUGIN_DIR/__init__.py"
cp    "$OPM_ROOT/qgis_plugin/opm_plugin.py" "$PLUGIN_DIR/opm_plugin.py"
cp    "$OPM_ROOT/qgis_plugin/metadata.txt" "$PLUGIN_DIR/metadata.txt"

# ── Copy core OPM model files into the plugin root ────────────────────────────
echo "▶ Copying core OPM model files into plugin root..."
for f in \
    config.py \
    process_dem.py \
    kinematic_wave_router.py \
    vsa_opm.py \
    routing_utils.py \
    routing_utils_gpu.py \
    precip_input.py \
    precip_input_gpu.py \
    runoff_input.py \
    runoff_input_gpu.py \
    gpu_utils.py \
    serves_gee.py \
    imerg_gee.py \
    lulc_lookup.csv \
    lcz_lookup.csv; do
    if [[ -f "$OPM_ROOT/$f" ]]; then
        cp "$OPM_ROOT/$f" "$PLUGIN_DIR/$f"
        echo "  + $f"
    else
        echo "  ! WARNING: $f not found in OPM root, skipping"
    fi
done

# Optional GEE service-account key: bundle it ONLY if the user placed one next
# to the repo.  It is git-ignored, so packaging it is a deliberate local choice.
if [[ -f "$OPM_ROOT/key.json" ]]; then
    cp "$OPM_ROOT/key.json" "$PLUGIN_DIR/key.json"
    echo "  + key.json  (GEE service-account credentials — bundled locally, NOT from git)"
else
    echo "  ℹ key.json not found — GEE features will need interactive auth or the GEE_PROJECT env var"
fi

# ── Fix _OPM_ROOT paths ────────────────────────────────────────────────────────
# Files in processing/ are 2 dirs deep (vsa_opm/processing/file.py)
# So dirname(dirname(abspath(__file__))) = vsa_opm/  ✓
# Files in bridge/ are also 2 dirs deep (vsa_opm/bridge/file.py)
# So dirname(dirname(abspath(__file__))) = vsa_opm/  ✓
# The source files already have 2 dirnames after our fix.
# This sed is kept as a safety net in case the source files get reset.
echo "▶ Verifying _OPM_ROOT paths..."
for f in \
    "$PLUGIN_DIR/processing/alg_process_dem.py" \
    "$PLUGIN_DIR/processing/alg_router.py" \
    "$PLUGIN_DIR/bridge/runner.py"; do
    # If file still has old 3-dirname pattern, fix it to 2
    sed -i 's|os\.path\.dirname(os\.path\.dirname(os\.path\.dirname(os\.path\.abspath(__file__))))|os.path.dirname(os.path.dirname(os.path.abspath(__file__)))|g' "$f"
    # Verify result
    if grep -q "_OPM_ROOT" "$f"; then
        echo "  ✓ $(basename $f): $(grep '_OPM_ROOT = ' "$f" | head -1 | xargs)"
    fi
done

# Fix relative imports in processing algorithms
echo "▶ Verifying relative imports..."
for f in \
    "$PLUGIN_DIR/processing/alg_process_dem.py" \
    "$PLUGIN_DIR/processing/alg_router.py"; do
    # Replace absolute "from bridge.config_bridge" with relative "from ..bridge.config_bridge"
    sed -i 's|from bridge\.config_bridge import|from ..bridge.config_bridge import|g' "$f"
    echo "  ✓ $(basename $f): $(grep 'config_bridge import' "$f" | head -1 | xargs)"
done

# ── Remove __pycache__ ────────────────────────────────────────────────────────
echo "▶ Removing __pycache__ directories..."
find "$PLUGIN_DIR" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# ── Show final structure ──────────────────────────────────────────────────────
echo ""
echo "▶ Final plugin structure:"
find "$PLUGIN_DIR" -type f | sort | sed "s|$PLUGIN_DIR/||"

# ── Zip ──────────────────────────────────────────────────────────────────────
echo ""
echo "▶ Creating zip: $TARGET_ZIP"
cd "$BUILD_DIR"
zip -r "$TARGET_ZIP" vsa_opm/

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                  BUILD COMPLETE ✅                           ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "  Zip file: $TARGET_ZIP"
echo "  $(du -sh "$TARGET_ZIP" | cut -f1) total size"
echo ""
echo "  To install on Windows:"
echo "  1. Download: vsa_opm_windows.zip"
echo "  2. Extract — you will get a single folder called 'vsa_opm'"
echo "  3. Copy 'vsa_opm' into:"
echo "     %%APPDATA%%\\QGIS\\QGIS3\\profiles\\default\\python\\plugins\\"
echo "  4. Open QGIS → Plugins → Manage and Install Plugins"
echo "     → Installed → tick 'VSA-OPM Hydrological Model'"
echo ""
