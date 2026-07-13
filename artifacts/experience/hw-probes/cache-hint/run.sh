#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
export PATH=/usr/local/cuda-12.9/bin:$PATH

DATE_TAG="$(date +%Y-%m-%d)"
HOST_DIR="/inspire/hdd/project/qianghuaxuexi/public/kernel_pilot_public/kp-probe-artifacts/cache-hint/${DATE_TAG}"
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

./cache_hint_probe 2>&1 | tee "${HOST_DIR}/run.log"

if command -v ncu >/dev/null 2>&1; then
    # 12 launches total (2 regimes × 6 variants). Profile all.
    ncu --launch-skip 0 --launch-count 12 \
        --section SpeedOfLight \
        --section MemoryWorkloadAnalysis \
        --section WarpStateStats \
        --metrics l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,\
lts__t_sectors_srcunit_tex_op_read.sum,\
lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum,\
lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum,\
dram__bytes_read.sum \
        --csv ./cache_hint_probe 2>/dev/null > "${HOST_DIR}/ncu_metrics.csv"
    ncu --launch-skip 0 --launch-count 12 \
        --section SpeedOfLight \
        --section MemoryWorkloadAnalysis \
        --section WarpStateStats \
        -o "${HOST_DIR}/cache_hint" --force-overwrite \
        --log-file "${HOST_DIR}/ncu.txt" \
        ./cache_hint_probe >/dev/null 2>&1 || true
fi

echo ""
echo "Repo-committed artifacts : .cu / .sh / .md / .json"
echo "Host-retained artifacts  : ${HOST_DIR}/"
