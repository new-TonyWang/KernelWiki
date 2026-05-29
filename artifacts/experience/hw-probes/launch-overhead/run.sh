#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
export PATH=/usr/local/cuda-12.9/bin:$PATH

DATE_TAG="$(date +%Y-%m-%d)"
HOST_DIR="/inspire/hdd/project/qianghuaxuexi/public/kernel_pilot_public/kp-probe-artifacts/launch-overhead/${DATE_TAG}"
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

./launch_overhead_probe 2>&1 | tee "${HOST_DIR}/run.log"

# NCU has minimal value for launch-overhead measurement because the
# interesting signal is on the host side; skip NCU entirely.

echo ""
echo "Repo-committed artifacts : .cu / .sh / .md / .json"
echo "Host-retained artifacts  : ${HOST_DIR}/"
