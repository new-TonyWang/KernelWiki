#!/usr/bin/env bash
# AC-3 (wgmma shape sweep) probe driver. Runs the AC-2 binary at five
# representative problem sizes (small / medium / medium-rectangular / large)
# and emits a CSV that becomes the table in the AC-3 record.
#
# This sweep varies the gemm *problem* size (M, N, K) at the canonical
# atom shape MMA_64x128x8_F32TF32TF32_SS_TN. Sweeping the wgmma atom N
# itself (8/16/32/64/96/128/192/256) requires recompiling cutlass with
# different template arguments — that wider sweep is documented but
# deferred; this script captures the lower-bound AC-3 evidence (≥3 shapes).
set -euo pipefail
cd "$(dirname "$0")"
export PATH=/usr/local/cuda-12.9/bin:$PATH

DATE_TAG="$(date +%Y-%m-%d)"
HOST_DIR="/inspire/hdd/project/qianghuaxuexi/public/kernel_pilot_public/kp-probe-artifacts/cutlass-cute-wgmma-shape-sweep/${DATE_TAG}"
mkdir -p "${HOST_DIR}" profiles

bash build.sh

CSV="profiles/${DATE_TAG}-wgmma-shape-sweep.csv"
{
    echo "category,M,N,K,dtype,atom,kernel_us,TFLOPS,disposition"
} > "${CSV}"

run_one () {
    local cat=$1 m=$2 n=$3 k=$4
    local out
    out="$(./wgmma_min_viable --m=$m --n=$n --k=$k --iterations=20 2>&1)"
    local ms=$(echo "$out" | grep "Avg runtime" | awk -F': ' '{print $2}' | awk '{print $1}')
    local gflops=$(echo "$out" | grep "GFLOPS" | awk -F': ' '{print $2}')
    local disp=$(echo "$out" | grep "Disposition" | awk -F': ' '{print $2}' | tr -d '[:space:]')
    local us=$(python3 -c "print(f'{${ms}*1000:.2f}')")
    local tfl=$(python3 -c "print(f'{${gflops}/1000:.1f}')")
    echo "${cat},${m},${n},${k},TF32_F32,MMA_64x128x8_F32TF32TF32_SS_TN,${us},${tfl},${disp}" \
        | tee -a "${CSV}"
    echo "[${cat}] ${m}x${n}x${k}: ${us}us, ${tfl} TFLOPS, ${disp}" \
        | tee -a "${HOST_DIR}/sweep.log"
}

run_one "small"        512 512  512
run_one "medium"      2048 2048 2048
run_one "medium-rect" 4096 4096 4096
run_one "rectangular" 5120 4096 4096
run_one "large"       8192 8192 8192

echo ""
echo "Repo-committed artifact : ${CSV}"
echo "Host-retained log       : ${HOST_DIR}/sweep.log"
