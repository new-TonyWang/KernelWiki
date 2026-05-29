#!/bin/bash
# run.sh -- on h200_ncu: build, run, capture NCU. .ncu-rep/.log/.csv
# stay host-only under /inspire/hdd/project/qianghuaxuexi/public/kernel_pilot_public/kp-probe-artifacts/aos-vs-soa/<date>/ per probe-artifact convention.
set -euo pipefail
cd "$(dirname "$0")"
export PATH=/usr/local/cuda-12.9/bin:$PATH

DATE_TAG="$(date +%Y-%m-%d)"
HOST_DIR="/inspire/hdd/project/qianghuaxuexi/public/kernel_pilot_public/kp-probe-artifacts/aos-vs-soa/${DATE_TAG}"
mkdir -p "${HOST_DIR}"

bash build.sh

# Device snapshot
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

# Clean run (host-retained log only)
./aos_vs_soa_probe 2>&1 | tee "${HOST_DIR}/run.log"

# NCU section-level profile; .ncu-rep to host, .csv to host.
if command -v ncu >/dev/null 2>&1; then
    ncu --launch-skip 0 --launch-count 4 \
        --section SpeedOfLight --section MemoryWorkloadAnalysis --section WarpStateStats \
        --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum \
        --csv ./aos_vs_soa_probe 2>/dev/null > "${HOST_DIR}/ncu_metrics.csv"
    ncu --section SpeedOfLight --section MemoryWorkloadAnalysis --section WarpStateStats \
        -o "${HOST_DIR}/aos_vs_soa" --force-overwrite \
        --log-file "${HOST_DIR}/ncu.txt" \
        ./aos_vs_soa_probe >/dev/null 2>&1 || true
fi

echo ""
echo "Repo-committed artifacts : .cu / .sh / .md / .json"
echo "Host-retained artifacts  : ${HOST_DIR}/"
