#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
export PATH=/usr/local/cuda-12.9/bin:$PATH

DATE_TAG="$(date +%Y-%m-%d)"
HOST_DIR="/inspire/hdd/project/qianghuaxuexi/public/kernel_pilot_public/kp-probe-artifacts/l2-residency/${DATE_TAG}"
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

./l2_residency_probe 2>&1 | tee "${HOST_DIR}/run.log"

if command -v ncu >/dev/null 2>&1; then
    # L2 hit-rate + SOL + warp stalls for the first 9 launches (3 WS x 3 policies).
    ncu --launch-skip 0 --launch-count 9 \
        --section SpeedOfLight \
        --section WarpStateStats \
        --section MemoryWorkloadAnalysis \
        --metrics lts__t_sectors_srcunit_tex_op_read.sum,\
lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum,\
lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum,\
lts__t_bytes.sum.per_second \
        --csv ./l2_residency_probe 2>/dev/null > "${HOST_DIR}/ncu_metrics.csv"
    ncu --launch-skip 0 --launch-count 9 \
        --section SpeedOfLight \
        --section WarpStateStats \
        --section MemoryWorkloadAnalysis \
        -o "${HOST_DIR}/l2_residency" --force-overwrite \
        --log-file "${HOST_DIR}/ncu.txt" \
        ./l2_residency_probe >/dev/null 2>&1 || true
fi

echo ""
echo "Repo-committed artifacts : .cu / .sh / .md / .json"
echo "Host-retained artifacts  : ${HOST_DIR}/"
