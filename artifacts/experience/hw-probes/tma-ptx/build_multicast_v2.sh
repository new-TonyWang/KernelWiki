#!/usr/bin/env bash
set -euo pipefail
SRC=tma_multicast_v2.cu
BIN=tma_multicast_v2
NVCC=${NVCC:-/usr/local/cuda-12.9/bin/nvcc}

"$NVCC" -std=c++17 -O3 -gencode=arch=compute_90a,code=sm_90a -lineinfo \
        "$SRC" -lcuda -o "$BIN"

echo "=== nvcc -E grep for cutlass::/cute:: ==="
PRE=$("$NVCC" -E -gencode=arch=compute_90a,code=sm_90a "$SRC" 2>/dev/null \
        | grep -cE 'cutlass::|cute::' || true)
[ "$PRE" = "0" ] && echo "PASS preprocessor" || { echo "FAIL: $PRE"; exit 1; }

echo "=== cuobjdump --dump-elf-symbols grep ==="
SYM=$(/usr/local/cuda-12.9/bin/cuobjdump --dump-elf-symbols "$BIN" 2>/dev/null \
        | grep -cE 'cutlass::|cute::' || true)
[ "$SYM" = "0" ] && echo "PASS binary" || { echo "FAIL: $SYM"; exit 1; }
