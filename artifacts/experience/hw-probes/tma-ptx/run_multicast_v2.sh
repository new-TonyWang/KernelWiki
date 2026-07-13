#!/usr/bin/env bash
# Build + run the v2 multicast follow-up probe; persist CSV + ncu metrics.
set -euo pipefail
DATE=$(date +%Y-%m-%d)
PROFILES_DIR=profiles
mkdir -p "$PROFILES_DIR"
OUT="$PROFILES_DIR/${DATE}-tma-multicast-v2.csv"
NCU_DRAM="$PROFILES_DIR/${DATE}-tma-multicast-v2-ncu-dram.csv"
NCU_TMA="$PROFILES_DIR/${DATE}-tma-multicast-v2-ncu-tma.csv"

bash build_multicast_v2.sh
./tma_multicast_v2 2>&1 | tee "$OUT"

NVCC=${NVCC:-/usr/local/cuda-12.9/bin/nvcc}
NCU=$(dirname "$NVCC")/ncu

# DRAM-side metrics
"$NCU" --kernel-name "regex:tma_v2_kernel" \
       --metrics dram__bytes_read.sum,sm__cycles_active.avg.pct_of_peak_sustained_elapsed \
       --csv ./tma_multicast_v2 2>/dev/null \
       | awk '/^"ID"/,EOF' > "$NCU_DRAM"

# TMA / multicast counters
"$NCU" --kernel-name "regex:tma_v2_kernel" \
       --metrics l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum,l1tex__m_l1tex2xbar_read_requests_mem_global_op_tma_ld_dest_multicast.sum \
       --csv ./tma_multicast_v2 2>/dev/null \
       | awk '/^"ID"/,EOF' > "$NCU_TMA"

echo "wrote $OUT $NCU_DRAM $NCU_TMA"
