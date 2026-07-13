#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
export PATH=/usr/local/cuda-12.9/bin:$PATH

DATE_TAG="$(date +%Y-%m-%d)"
HOST_DIR="/inspire/hdd/project/qianghuaxuexi/public/kernel_pilot_public/kp-probe-artifacts/barrier-cost/${DATE_TAG}"
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

./barrier_cost_probe 2>&1 | tee "${HOST_DIR}/run.log"

if command -v ncu >/dev/null 2>&1; then
    ncu --launch-skip 0 --launch-count 12 \
        --section SpeedOfLight --section WarpStateStats \
        --csv ./barrier_cost_probe 2>/dev/null > "${HOST_DIR}/ncu_metrics.csv"
    ncu --section SpeedOfLight --section WarpStateStats \
        -o "${HOST_DIR}/barrier_cost" --force-overwrite \
        --log-file "${HOST_DIR}/ncu.txt" \
        ./barrier_cost_probe >/dev/null 2>&1 || true
fi

echo ""
echo "Repo-committed artifacts : .cu / .sh / .md / .json"
echo "Host-retained artifacts  : ${HOST_DIR}/"
