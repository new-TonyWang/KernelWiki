#!/bin/bash
# run.sh -- on h200_ncu: build, run, capture NCU. .ncu-rep/.log/.csv
# stay host-only under /inspire/hdd/project/qianghuaxuexi/public/kernel_pilot_public/kp-probe-artifacts/warp-divergence-cost/<date>/
# per probe-artifact convention.
set -euo pipefail
cd "$(dirname "$0")"
export PATH=/usr/local/cuda-12.9/bin:$PATH

DATE_TAG="$(date +%Y-%m-%d)"
HOST_DIR="/inspire/hdd/project/qianghuaxuexi/public/kernel_pilot_public/kp-probe-artifacts/warp-divergence-cost/${DATE_TAG}"
mkdir -p "${HOST_DIR}"

bash build.sh

if [[ ! -f device.json ]]; then
    nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total,compute_cap,clocks.current.graphics,clocks.max.graphics \
        --format=csv --id=0 > device.csv
    python3 - <<'PY' > device.json
import csv, json
with open('device.csv') as f:
    r = list(csv.DictReader(f))
print(json.dumps(r, indent=2))
PY
    rm -f device.csv
fi

./divergence_cost_probe 2>&1 | tee "${HOST_DIR}/run.log"

# NCU focuses on SIMT efficiency metric plus stall analysis.
if command -v ncu >/dev/null 2>&1; then
    ncu --section SpeedOfLight --section WarpStateStats \
        --metrics smsp__thread_inst_executed_per_inst_executed.ratio,smsp__sass_average_branch_targets_threads_uniform.pct \
        --csv ./divergence_cost_probe 2>/dev/null > "${HOST_DIR}/ncu_metrics.csv"
    ncu --section SpeedOfLight --section WarpStateStats \
        -o "${HOST_DIR}/warp_divergence_cost" --force-overwrite \
        --log-file "${HOST_DIR}/ncu.txt" \
        ./divergence_cost_probe >/dev/null 2>&1 || true
fi

echo ""
echo "Repo-committed artifacts : .cu / .sh / .md / .json"
echo "Host-retained artifacts  : ${HOST_DIR}/"
