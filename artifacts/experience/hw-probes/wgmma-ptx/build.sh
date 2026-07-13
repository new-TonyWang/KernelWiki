#!/usr/bin/env bash
# wgmma-ptx build + cutlass-free verification.
set -euo pipefail
cd "$(dirname "$0")"
export PATH=/usr/local/cuda-12.9/bin:$PATH

# NOTE: -arch=sm_90a alone is insufficient for wgmma inline asm because nvcc 12.9
# generates an intermediate compute_90 (no `a`) PTX that ptxas rejects with
# "Instruction 'wgmma.fence' not supported on .target 'sm_90'". Use the explicit
# -gencode form to force compute_90a as the intermediate target.
NVCC_FLAGS=(-std=c++17 -O3 -gencode=arch=compute_90a,code=sm_90a -lineinfo)

# 1. Build.
nvcc "${NVCC_FLAGS[@]}" wgmma_hello.cu -o wgmma_hello

# 2. Cutlass-free verification (preprocessor check):
echo "=== nvcc -E grep for cutlass::/cute:: in preprocessed source ==="
nvcc "${NVCC_FLAGS[@]}" -E wgmma_hello.cu 2>/dev/null \
    | grep -E 'cutlass::|cute::' > preprocessor_grep.txt || true
if [ -s preprocessor_grep.txt ]; then
    echo "FAIL: cutlass/cute symbols present in preprocessed source. See preprocessor_grep.txt"
    head -3 preprocessor_grep.txt
    exit 1
fi
echo "PASS: preprocessor produces 0 cutlass:: / cute:: matches."
rm -f preprocessor_grep.txt

# 3. Cutlass-free verification (linked-symbol check):
echo "=== cuobjdump --dump-elf-symbols grep for cutlass::/cute:: in linked binary ==="
cuobjdump --dump-elf-symbols wgmma_hello \
    | grep -E 'cutlass::|cute::' > symbol_grep.txt || true
if [ -s symbol_grep.txt ]; then
    echo "FAIL: cutlass/cute symbols present in linked binary. See symbol_grep.txt"
    head -3 symbol_grep.txt
    exit 1
fi
echo "PASS: linked binary has 0 cutlass:: / cute:: symbols."
rm -f symbol_grep.txt
