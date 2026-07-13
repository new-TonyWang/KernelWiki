#!/usr/bin/env bash
# Build + run the TMA cluster-multicast probe; persist CSV.
set -euo pipefail
DATE=$(date +%Y-%m-%d)
PROFILES_DIR=profiles
mkdir -p "$PROFILES_DIR"
OUT="$PROFILES_DIR/${DATE}-tma-multicast.csv"

bash build_multicast.sh
./tma_multicast_probe | tee "$OUT"
echo "wrote $OUT"
