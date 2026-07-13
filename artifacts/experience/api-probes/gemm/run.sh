#!/usr/bin/env bash
CUDA_REPO_ROOT=${CUDA_REPO_ROOT:-/path/to/your/cuda_repo}
CUTLASS_DIR=${CUTLASS_DIR:-${CUDA_REPO_ROOT}/cutlass}
# AC-4 (aligned GEMM) probe driver. Round-4+ canonical workflow: the AC-4
# correctness oracle is the direct cutlass-vs-cuBLAS comparator
# `gemm_compare.cu`, which runs cutlass and cublasGemmEx on the same A/B
# tensors in one binary and emits per-shape `max_abs_diff` + `max_rel_diff`
# + mismatch count. cublas_reference.cu is the cuBLAS-only TFLOPS harness
# kept for the side-by-side TFLOPS table; it can also be invoked with
# `--ac5` to dump cuBLAS data at the AC-5 non-aligned shapes.
#
# Sequence:
#   1. Build gemm_aligned, gemm_compare, cublas_reference.
#   2. Run gemm_compare at 512^3 / 2048^3 / 8192^3 — direct numeric diff.
#   3. Run cublas_reference at the same three shapes — TFLOPS table.
#   4. Capture ncu CSV at the 4096^3 anchor (`profiles/${DATE_TAG}-gemm-aligned.csv`).
#   5. compute-sanitizer memcheck on gemm_compare at 2048^3.
set -euo pipefail
cd "$(dirname "$0")"
export PATH=/usr/local/cuda-12.9/bin:$PATH

CUTLASS_ROOT="${CUTLASS_ROOT:-${CUTLASS_DIR}}"
DATE_TAG="$(date +%Y-%m-%d)"
HOST_DIR="/inspire/hdd/project/qianghuaxuexi/public/kernel_pilot_public/kp-probe-artifacts/cutlass-cute-gemm-aligned/${DATE_TAG}"
mkdir -p "${HOST_DIR}" profiles

# Capture device.json once.
if [[ ! -f device.json ]]; then
    nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total,compute_cap,clocks.current.graphics,clocks.max.graphics \
        --format=csv --id=0 2>/dev/null > device.csv
    python3 -c "import csv,json; r=list(csv.DictReader(open('device.csv'))); print(json.dumps(r, indent=2))" > device.json
    rm -f device.csv
fi

# Build all three binaries.
echo "=== build gemm_aligned (cutlass example 48 path, timing only) ==="
nvcc -std=c++17 -O3 -arch=sm_90a -lineinfo \
     -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 \
     --expt-relaxed-constexpr --expt-extended-lambda \
     -I"${CUTLASS_ROOT}/include" \
     -I"${CUTLASS_ROOT}/tools/util/include" \
     -I"$(pwd)" \
     gemm_aligned.cu -o gemm_aligned

echo "=== build gemm_compare (cutlass + cublasGemmEx side-by-side, direct numeric diff) ==="
nvcc -std=c++17 -O3 -arch=sm_90a -lineinfo \
     -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 \
     --expt-relaxed-constexpr --expt-extended-lambda \
     -I"${CUTLASS_ROOT}/include" \
     -I"${CUTLASS_ROOT}/tools/util/include" \
     -I"$(pwd)" \
     gemm_compare.cu -o gemm_compare -lcublas

echo "=== build cublas_reference (cuBLAS-only TF32 timing harness) ==="
nvcc -std=c++17 -O3 -arch=sm_90a \
     cublas_reference.cu -o cublas_reference -lcublas

# Run the direct cutlass-vs-cuBLAS comparator at the three AC-4 aligned shapes.
# This is the canonical AC-4 oracle: cutlass + cublasGemmEx in the same binary
# with a numeric diff (max_abs, max_rel, mismatch count) per shape.
echo "=== cutlass-vs-cuBLAS direct comparator ==="
for shape in "512 512 512" "2048 2048 2048" "8192 8192 8192"; do
    read m n k <<< "$shape"
    ./gemm_compare --m=$m --n=$n --k=$k --iterations=20 | tee -a "${HOST_DIR}/compare.log"
done

# Run the cuBLAS-only timing harness for the side-by-side TFLOPS table.
echo "=== cuBLAS-only TFLOPS at AC-4 shapes ==="
./cublas_reference 20 | tee "${HOST_DIR}/cublas.log"

# ncu metric capture at the 4096^3 dense anchor. Filename matches the probe's
# artifacts.profile pointer.
NCU_CSV="profiles/${DATE_TAG}-gemm-aligned.csv"
ncu --kernel-name regex:device_kernel --launch-count 1 \
    --metrics sm__cycles_active.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum,\
sm__inst_executed_pipe_tensor_op_hmma.sum \
    --csv ./gemm_aligned --m=4096 --n=4096 --k=4096 \
    > "${NCU_CSV}" 2>"${HOST_DIR}/ncu.stderr"

# compute-sanitizer: AC-V Stage-3.5 sanitizer pass on the direct comparator at
# 2048^3, matching the probe's documented memcheck command. Completes the
# canonical verification contract (build → exec → ncu → sanitizer).
echo "=== compute-sanitizer memcheck (gemm_compare at 2048^3) ==="
if command -v compute-sanitizer >/dev/null 2>&1; then
    compute-sanitizer --tool memcheck \
        ./gemm_compare --m=2048 --n=2048 --k=2048 --iterations=2 \
        2>&1 | tee "${HOST_DIR}/compute-sanitizer.log" | tail -10
fi

echo ""
echo "Repo-committed artifacts : gemm_aligned.cu / gemm_compare.cu / cublas_reference.cu / helper.h / build.sh / run.sh / device.json / profiles/*.csv"
echo "Host-retained artifacts  : ${HOST_DIR}/  (compare.log, cublas.log, ncu.stderr, compute-sanitizer.log)"
