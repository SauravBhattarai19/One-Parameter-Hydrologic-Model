#!/usr/bin/env bash
# =============================================================================
# install_plugin.sh
# =============================================================================
# Installs (or refreshes) the VSA-OPM QGIS plugin by creating a symlink
# from the QGIS user plugin directory to this repository's qgis_plugin/ folder.
#
# A symlink means any edit you make to the source files is reflected
# immediately in QGIS — no zip-and-reinstall cycle needed during development.
#
# Usage
# -----
#   chmod +x install_plugin.sh
#   ./install_plugin.sh
#
# Then in QGIS:
#   Plugins → Manage and Install Plugins → Installed → enable "VSA-OPM …"
#   (If it doesn't appear, run: Plugins → Reload all plugin icons)
#
# To uninstall
# ------------
#   ./install_plugin.sh --remove
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_SRC="$SCRIPT_DIR/qgis_plugin"
PLUGIN_NAME="vsa_opm_plugin"   # renamed: must not shadow the core "vsa_opm" package import   # must match the folder name QGIS sees

# ── Detect QGIS plugin directory ─────────────────────────────────────────────
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    QGIS_PLUGIN_DIR="$HOME/.local/share/QGIS/QGIS3/profiles/default/python/plugins"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    QGIS_PLUGIN_DIR="$HOME/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins"
else
    # Windows (Git Bash / WSL)
    QGIS_PLUGIN_DIR="$APPDATA/QGIS/QGIS3/profiles/default/python/plugins"
fi

TARGET="$QGIS_PLUGIN_DIR/$PLUGIN_NAME"

# ── Remove mode ───────────────────────────────────────────────────────────────
if [[ "${1:-}" == "--remove" ]]; then
    if [[ -L "$TARGET" ]]; then
        rm "$TARGET"
        echo "✅  Symlink removed: $TARGET"
    elif [[ -d "$TARGET" ]]; then
        echo "⚠️   $TARGET is a real directory (not a symlink). Remove it manually."
        exit 1
    else
        echo "ℹ️   Plugin not installed at $TARGET — nothing to do."
    fi
    exit 0
fi

# ── Install ───────────────────────────────────────────────────────────────────
mkdir -p "$QGIS_PLUGIN_DIR"

if [[ -L "$TARGET" ]]; then
    echo "ℹ️   Existing symlink found. Updating …"
    rm "$TARGET"
elif [[ -d "$TARGET" ]]; then
    echo "⚠️   $TARGET is a real directory. Please remove it first:"
    echo "    rm -rf $TARGET"
    exit 1
fi

ln -s "$PLUGIN_SRC" "$TARGET"

echo ""
echo "✅  Plugin installed (symlink):"
echo "    $TARGET → $PLUGIN_SRC"
echo ""
echo "Next steps:"
echo "  1. Open QGIS."
echo "  2. Plugins → Manage and Install Plugins → Installed tab."
echo "  3. Find 'VSA-OPM Hydrological Model' and tick the checkbox."
echo "  4. A toolbar button and Plugins menu entry will appear."
echo ""
echo "Dependencies check (run in QGIS Python console or your env):"
echo "  import rasterio, pysheds, scipy, numpy, pandas"
echo "  # Optional GPU: import cupy"
