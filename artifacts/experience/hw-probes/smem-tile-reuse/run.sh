#!/bin/bash
# run.sh -- on h200_ncu: build, lock clocks (if permitted), run probe, capture
# NCU section-level profile. The .ncu-rep binary stays on the host; only the
# .txt dump lands in the artifacts subtree (per probe-artifact convention (.ncu-rep stays on host)).
set -euo pipefail
cd "$(dirname "$0")"
export PATH=/usr/local/cuda-12.9/bin:$PATH

HOST_PROBE_ROOT="/inspire/hdd/project/qianghuaxuexi/public/kernel_pilot_public/kp-probe-artifacts/smem-tile-reuse"
DATE_TAG="$(date +%Y-%m-%d)"
HOST_DIR="${HOST_PROBE_ROOT}/${DATE_TAG}"
mkdir -p "${HOST_DIR}"

# 1. Build
bash build.sh

# 2. Device introspection (once; small file, commits with the repo)
if [[ ! -f device.json ]]; then
    nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total,compute_cap,clocks.current.graphics,clocks.max.graphics \
        --format=csv > device.csv
    python3 - <<'PY' > device.json
import csv, json, sys
with open('device.csv') as f:
    r = list(csv.DictReader(f))
print(json.dumps(r, indent=2))
PY
fi

# 3. Lock clocks (requires privileges; skip gracefully if not allowed).
CLOCK_POLICY="unlocked"
if nvidia-smi -lgc 1590,1590 >/dev/null 2>&1; then
    CLOCK_POLICY="locked-1590MHz"
elif nvidia-smi --query-gpu=clocks.gr --format=csv,noheader,nounits >/dev/null 2>&1; then
    CLOCK_POLICY="unlocked-logged-only"
fi
echo "clock_policy: ${CLOCK_POLICY}" | tee "${HOST_DIR}/clock_policy.txt"

# 4. Timing run -- dump to repo artifacts
./smem_tile_reuse_probe 2>&1 | tee run.log

# 5. Unlock clocks
if [[ "${CLOCK_POLICY}" == locked-* ]]; then
    nvidia-smi -rgc >/dev/null 2>&1 || true
fi

# 6. NCU section-level profile for each kernel. .ncu-rep goes to host-only
# path; .txt dump goes into the repo for later diffs.
if command -v ncu >/dev/null 2>&1; then
    ncu --section SpeedOfLight --section WarpStateStats \
        --section MemoryWorkloadAnalysis \
        --log-file "${HOST_DIR}/ncu.txt" \
        -o "${HOST_DIR}/smem_tile_reuse" \
        --force-overwrite \
        ./smem_tile_reuse_probe >/dev/null 2>&1 || true
    if [[ -f "${HOST_DIR}/ncu.txt" ]]; then
        cp "${HOST_DIR}/ncu.txt" ncu.txt
        echo "NCU report (binary) saved to: ${HOST_DIR}/smem_tile_reuse.ncu-rep"
        echo "NCU text dump copied to: ${PWD}/ncu.txt"
    fi
fi

echo ""
echo "run.log and ncu.txt are committed artifacts."
echo "Binary NCU report lives on host at: ${HOST_DIR}/"
