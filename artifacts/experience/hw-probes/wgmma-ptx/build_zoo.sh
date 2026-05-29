#!/usr/bin/env bash
# Build the cutlass-free wgmma multi-config harness + verify cutlass-free gates.
set -euo pipefail
SRC=wgmma_zoo.cu
BIN=wgmma_zoo
NVCC=${NVCC:-/usr/local/cuda-12.9/bin/nvcc}

"$NVCC" -std=c++17 -O3 -gencode=arch=compute_90a,code=sm_90a -lineinfo \
        "$SRC" -o "$BIN"

echo "=== nvcc -E grep for cutlass::/cute:: in preprocessed source ==="
PRE=$("$NVCC" -E -gencode=arch=compute_90a,code=sm_90a "$SRC" 2>/dev/null \
        | grep -cE 'cutlass::|cute::' || true)
[ "$PRE" = "0" ] && echo "PASS: 0 cutlass:: / cute:: matches." || { echo "FAIL: $PRE matches."; exit 1; }

echo "=== cuobjdump --dump-elf-symbols grep for cutlass::/cute:: in linked binary ==="
SYM=$(/usr/local/cuda-12.9/bin/cuobjdump --dump-elf-symbols "$BIN" 2>/dev/null \
        | grep -cE 'cutlass::|cute::' || true)
[ "$SYM" = "0" ] && echo "PASS: 0 cutlass:: / cute:: symbols." || { echo "FAIL: $SYM symbols."; exit 1; }
