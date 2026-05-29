#!/usr/bin/env bash
# Build + run the TMA throughput sweep on H200 and persist the CSV.
set -euo pipefail
DATE=$(date +%Y-%m-%d)
PROFILES_DIR=profiles
mkdir -p "$PROFILES_DIR"
OUT="$PROFILES_DIR/${DATE}-tma-throughput.csv"

bash build_throughput.sh
./tma_throughput_probe | tee "$OUT"
echo "wrote $OUT"
