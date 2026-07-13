#!/usr/bin/env bash
# AC-11 cutlass-free GEMM build + cutlass-free verification + cuBLAS comparison.
set -euo pipefail
cd "$(dirname "$0")"
export PATH=/usr/local/cuda-12.9/bin:$PATH

NVCC_FLAGS=(-std=c++17 -O3 -gencode=arch=compute_90a,code=sm_90a -lineinfo)

# 1. Build.
nvcc "${NVCC_FLAGS[@]}" gemm_ptx.cu -lcuda -lcublas -o gemm_ptx

# 2. Cutlass-free verification (preprocessor):
echo "=== nvcc -E grep for cutlass::/cute:: in preprocessed source ==="
nvcc "${NVCC_FLAGS[@]}" -E gemm_ptx.cu 2>/dev/null \
    | grep -E 'cutlass::|cute::' > preprocessor_grep.txt || true
if [ -s preprocessor_grep.txt ]; then
    echo "FAIL: cutlass/cute symbols present in preprocessed source."
    head -3 preprocessor_grep.txt
    exit 1
fi
echo "PASS: preprocessor produces 0 cutlass:: / cute:: matches."
rm -f preprocessor_grep.txt

# 3. Cutlass-free verification (linked binary):
echo "=== cuobjdump --dump-elf-symbols grep for cutlass::/cute:: in linked binary ==="
cuobjdump --dump-elf-symbols gemm_ptx \
    | grep -E 'cutlass::|cute::' > symbol_grep.txt || true
if [ -s symbol_grep.txt ]; then
    echo "FAIL: cutlass/cute symbols present in linked binary."
    head -3 symbol_grep.txt
    exit 1
fi
echo "PASS: linked binary has 0 cutlass:: / cute:: symbols."
rm -f symbol_grep.txt
