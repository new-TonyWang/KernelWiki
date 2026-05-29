#!/usr/bin/env bash
CUDA_REPO_ROOT=${CUDA_REPO_ROOT:-/path/to/your/cuda_repo}
CUTLASS_DIR=${CUTLASS_DIR:-${CUDA_REPO_ROOT}/cutlass}
# AC-3 (wgmma atom-shape sweep) probe driver.
# Builds three TileShape variants of cutlass example 48 (which differ only in
# the second dimension of TileShape and, for N=256, a reduced ClusterShape so
# the cluster covers fewer CTAs in N). cute's CollectiveBuilder picks the
# wgmma atom from TileShape.N; this is the lever that selects the atom we
# want to characterize.
#
# Variants:
#   wgmma_atom_n64.cu  -> TileShape <128, 64,32>, Cluster <4,2,1> -> MMA_64x64x8_F32TF32TF32_SS_TN
#   wgmma_atom_n128.cu -> TileShape <128,128,32>, Cluster <4,2,1> -> MMA_64x128x8_F32TF32TF32_SS_TN  (cutlass example-48 default)
#   wgmma_atom_n256.cu -> TileShape <128,256,32>, Cluster <2,2,1> -> MMA_64x256x8_F32TF32TF32_SS_TN
#
# Same problem size (4096x4096x4096) for all three, so the only varying
# dimension is the wgmma atom N.
set -euo pipefail
cd "$(dirname "$0")"
export PATH=/usr/local/cuda-12.9/bin:$PATH

CUTLASS_ROOT="${CUTLASS_ROOT:-${CUTLASS_DIR}}"
DATE_TAG="$(date +%Y-%m-%d)"
HOST_DIR="/inspire/hdd/project/qianghuaxuexi/public/kernel_pilot_public/kp-probe-artifacts/cutlass-cute-wgmma-atom-sweep/${DATE_TAG}"
mkdir -p "${HOST_DIR}" profiles

# Per-variant TileShape and ClusterShape, kept in sync with the .cu sources.
declare -A TILE_FOR=( [64]="(128;64;32)" [128]="(128;128;32)" [256]="(128;256;32)" )
declare -A CLUSTER_FOR=( [64]="(4;2;1)" [128]="(4;2;1)" [256]="(2;2;1)" )
declare -A ATOM_FOR=(
    [64]="MMA_64x64x8_F32TF32TF32_SS_TN"
    [128]="MMA_64x128x8_F32TF32TF32_SS_TN"
    [256]="MMA_64x256x8_F32TF32TF32_SS_TN"
)

build_variant () {
    local n=$1
    nvcc -std=c++17 -O3 -arch=sm_90a -lineinfo \
         -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 \
         --expt-relaxed-constexpr --expt-extended-lambda \
         -I"${CUTLASS_ROOT}/include" \
         -I"${CUTLASS_ROOT}/tools/util/include" \
         -I"$(pwd)" \
         "wgmma_atom_n${n}.cu" \
         -o "wgmma_atom_n${n}"
}

run_variant () {
    local n=$1 m_dim=$2 n_dim=$3 k_dim=$4
    "./wgmma_atom_n${n}" --m=${m_dim} --n=${n_dim} --k=${k_dim} --iterations=20
}

profile_variant () {
    local n=$1 m_dim=$2 n_dim=$3 k_dim=$4
    ncu --kernel-name regex:device_kernel --launch-count 1 \
        --metrics sm__cycles_active.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
sm__inst_executed_pipe_tensor_op_hmma.sum,\
l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum \
        --csv "./wgmma_atom_n${n}" --m=${m_dim} --n=${n_dim} --k=${k_dim} \
        > "profiles/${DATE_TAG}-wgmma_atom_n${n}.csv" 2>"${HOST_DIR}/ncu_n${n}.stderr"
}

# Append one row to the aggregate CSV by combining the variant's run output
# (cutlass writes "Avg runtime" and "GFLOPS" lines) with the per-variant ncu
# CSV (which carries sm__cycles_active.avg.pct_of_peak_sustained_elapsed and
# sm__warps_active.avg.pct_of_peak_sustained_active).
append_row () {
    local n=$1 m_dim=$2 n_dim=$3 k_dim=$4 run_log=$5
    local ms gflops disposition tfl us
    ms=$(grep -E "Avg runtime"  "${run_log}" | awk -F': ' '{print $2}' | awk '{print $1}')
    gflops=$(grep -E "^  GFLOPS" "${run_log}" | awk -F': ' '{print $2}' | tr -d '[:space:]')
    disposition=$(grep -E "Disposition" "${run_log}" | awk -F': ' '{print $2}' | tr -d '[:space:]')
    us=$(python3 -c "print(f'{${ms}*1000:.2f}')")
    tfl=$(python3 -c "print(f'{${gflops}/1000:.1f}')")
    local cycles_pct warps_pct
    cycles_pct=$(grep -F "sm__cycles_active.avg.pct_of_peak_sustained_elapsed" \
                  "profiles/${DATE_TAG}-wgmma_atom_n${n}.csv" \
                  | head -1 | awk -F'","' '{print $(NF-1)}' | tr -d '"')
    warps_pct=$(grep -F "sm__warps_active.avg.pct_of_peak_sustained_active" \
                  "profiles/${DATE_TAG}-wgmma_atom_n${n}.csv" \
                  | head -1 | awk -F'","' '{print $(NF-1)}' | tr -d '"')
    # Grid string from the same ncu CSV (column "Grid Size").
    local grid
    grid=$(grep -F "sm__cycles_active.avg.pct_of_peak_sustained_elapsed" \
            "profiles/${DATE_TAG}-wgmma_atom_n${n}.csv" \
            | head -1 | awk -F'","' '{print $9}' | tr -d '"' \
            | sed 's/, /;/g; s/(//; s/)//')
    echo "${ATOM_FOR[$n]},${TILE_FOR[$n]},${CLUSTER_FOR[$n]},${m_dim},${n_dim},${k_dim},TF32_F32,${us},${tfl},${cycles_pct},${warps_pct},(${grid}),${disposition}" \
        >> "${CSV}"
}

# Build all three.
for n in 64 128 256; do
    echo "=== build wgmma_atom_n${n} ==="
    build_variant ${n} 2>&1 | tee -a "${HOST_DIR}/build.log" | tail -3
done

# Run + profile each at 4096x4096x4096; emit CSV with all three rows.
CSV="profiles/${DATE_TAG}-wgmma-atom-sweep.csv"
echo "atom,TileShape,ClusterShape,M,N,K,dtype,kernel_us,TFLOPS,sm_cycles_active_pct,sm_warps_active_pct,grid,disposition" > "${CSV}"

for n in 64 128 256; do
    echo "=== run + profile n=${n} ==="
    RUN_LOG="${HOST_DIR}/run_n${n}.log"
    run_variant  ${n} 4096 4096 4096 | tee "${RUN_LOG}" >> "${HOST_DIR}/run.log"
    profile_variant ${n} 4096 4096 4096
    append_row ${n} 4096 4096 4096 "${RUN_LOG}"
done

echo "=== aggregate CSV ==="
cat "${CSV}"

echo ""
echo "Repo-committed artifacts : wgmma_atom_n{64,128,256}.cu / build script / run.sh / profiles/*.csv"
echo "Host-retained artifacts  : ${HOST_DIR}/  (build.log, run.log, ncu_n*.stderr)"
