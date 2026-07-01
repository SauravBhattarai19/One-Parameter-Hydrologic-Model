#!/usr/bin/env bash
# test_ee_auth.sh — verify Earth Engine service-account authentication
set -euo pipefail

KEY="/home/sauravbhattarai/Documents/ORISE/OPM/key.json"
PROJECT="ee-sauravbhattarai1999"
CONDA_ENV="opm"

# ── colours ──────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; NC='\033[0m'
pass() { echo -e "${GREEN}[PASS]${NC} $1"; }
fail() { echo -e "${RED}[FAIL]${NC} $1"; FAILED=$((FAILED+1)); }
info() { echo -e "${YELLOW}[INFO]${NC} $1"; }
FAILED=0

echo "======================================================"
echo "  Earth Engine Service Account Auth Test"
echo "======================================================"

# ── 1. Key file checks ────────────────────────────────────────────────────────
info "Checking key file..."

if [[ ! -f "$KEY" ]]; then
    fail "Key file not found: $KEY"
    echo "Place your service-account JSON at that path and re-run."
    exit 1
fi

if [[ ! -s "$KEY" ]]; then
    fail "Key file is empty: $KEY"
    echo "Paste your service-account JSON content into the file and re-run."
    exit 1
fi

pass "Key file exists and is non-empty"

# ── 2. Valid JSON? ────────────────────────────────────────────────────────────
SA_EMAIL=$(python3 -c "
import json, sys
try:
    d = json.load(open('$KEY'))
    print(d['client_email'])
except Exception as e:
    print('ERROR:' + str(e), file=sys.stderr)
    sys.exit(1)
" 2>&1) || { fail "Key file is not valid JSON — $SA_EMAIL"; exit 1; }

pass "Key file is valid JSON"
info "Service account: $SA_EMAIL"

# ── 3. ee Python package installed? ──────────────────────────────────────────
info "Checking earthengine-api..."
EE_VER=$(conda run -n "$CONDA_ENV" python3 -c "import ee; print(ee.__version__)" 2>&1) \
    && pass "earthengine-api $EE_VER installed" \
    || { fail "earthengine-api not found in conda env '$CONDA_ENV'"; exit 1; }

# ── 4. Initialize with service account ───────────────────────────────────────
info "Initialising Earth Engine with service account..."
conda run -n "$CONDA_ENV" python3 - <<PYEOF
import ee, sys

key  = "$KEY"
sa   = "$SA_EMAIL"
proj = "$PROJECT"

try:
    creds = ee.ServiceAccountCredentials(sa, key)
    ee.Initialize(creds, project=proj)
    print("  ee.Initialize() succeeded")
except Exception as e:
    print(f"  ERROR: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF
[[ $? -eq 0 ]] && pass "ee.Initialize() with service account" \
               || { fail "ee.Initialize() failed (see error above)"; FAILED=$((FAILED+1)); }

# ── 5. Simple EE API call ─────────────────────────────────────────────────────
info "Running a simple EE API call (SRTM DEM band names)..."
conda run -n "$CONDA_ENV" python3 - <<PYEOF
import ee, sys

key  = "$KEY"
sa   = "$SA_EMAIL"
proj = "$PROJECT"

try:
    creds = ee.ServiceAccountCredentials(sa, key)
    ee.Initialize(creds, project=proj)
    bands = ee.Image("USGS/SRTMGL1_003").bandNames().getInfo()
    print(f"  Band names: {bands}")
except Exception as e:
    print(f"  ERROR: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF
[[ $? -eq 0 ]] && pass "EE API call (SRTM band names)" \
               || { fail "EE API call failed (see error above)"; FAILED=$((FAILED+1)); }

# ── 6. IMERG dataset accessible? ─────────────────────────────────────────────
info "Checking IMERG dataset access..."
conda run -n "$CONDA_ENV" python3 - <<PYEOF
import ee, sys

key  = "$KEY"
sa   = "$SA_EMAIL"
proj = "$PROJECT"

try:
    creds = ee.ServiceAccountCredentials(sa, key)
    ee.Initialize(creds, project=proj)
    info = ee.ImageCollection("NASA/GPM_L3/IMERG_V07").limit(1).getInfo()
    n = len(info.get('features', []))
    print(f"  IMERG collection reachable, sampled {n} image(s)")
except Exception as e:
    print(f"  ERROR: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF
[[ $? -eq 0 ]] && pass "IMERG dataset accessible" \
               || { fail "IMERG dataset not accessible (see error above)"; FAILED=$((FAILED+1)); }

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "======================================================"
if [[ $FAILED -eq 0 ]]; then
    echo -e "${GREEN}All tests passed.${NC}"
    echo ""
    echo "Add these lines to ~/.bashrc to make it permanent:"
    echo "  export GOOGLE_APPLICATION_CREDENTIALS=\"$KEY\""
    echo "  export EE_SERVICE_ACCOUNT=\"$SA_EMAIL\""
else
    echo -e "${RED}$FAILED test(s) failed.${NC}"
    echo ""
    echo "If EE rejects the service account, register it at:"
    echo "  https://signup.earthengine.google.com/#!/service_accounts"
    exit 1
fi
echo "======================================================"
