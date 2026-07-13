#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
export PATH=/usr/local/cuda-12.9/bin:$PATH

DATE_TAG="$(date +%Y-%m-%d)"
HOST_DIR="/inspire/hdd/project/qianghuaxuexi/public/kernel_pilot_public/kp-probe-artifacts/half2-throughput/${DATE_TAG}"
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

./half2_throughput_probe 2>&1 | tee "${HOST_DIR}/run.log"

if command -v ncu >/dev/null 2>&1; then
    # 5 variants * (5 warmup + 20 iters) = 125 launches total.
    # Cover the first timed launch of each variant (launch index 5, 30,
    # 55, 80, 105). Use generous launch-count and launch-skip stride so
    # each variant's first timed iteration is profiled.
    ncu --launch-skip 5 --launch-count 1 \
        --kernel-name regex:fma_bench_ \
        --section SpeedOfLight \
        --section WarpStateStats \
        --section InstructionStats \
        --csv ./half2_throughput_probe 2>/dev/null > "${HOST_DIR}/ncu_metrics.csv"
    ncu --launch-skip 5 --launch-count 1 \
        --kernel-name regex:fma_bench_ \
        --section SpeedOfLight \
        --section WarpStateStats \
        --section InstructionStats \
        -o "${HOST_DIR}/half2_throughput" --force-overwrite \
        --log-file "${HOST_DIR}/ncu.txt" \
        ./half2_throughput_probe >/dev/null 2>&1 || true
fi

echo ""
echo "Repo-committed artifacts : .cu / .sh / .md / .json"
echo "Host-retained artifacts  : ${HOST_DIR}/"
