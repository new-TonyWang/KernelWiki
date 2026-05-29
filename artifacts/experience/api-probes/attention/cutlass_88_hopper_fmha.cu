// Repo-local extracted wrapper for cutlass 88_hopper_fmha.
// This file includes the upstream cutlass example directly.
// Build requires the cutlass checkout at CUTLASS_DIR (see build_cutlass_fmha.sh).
//
// Upstream source: cutlass/examples/88_hopper_fmha/88_hopper_fmha.cu
// Upstream commit: cutlass main (or cutlass@f74fea9c+)
//
// Build:
//   CUTLASS_DIR={{CUTLASS_REPO_REF}} bash build_cutlass_fmha.sh
//
// Run:
//   ./88_hopper_fmha --b=2 --h=16 --q=1024 --k=1024 --d=128 --verify

// Include the upstream example source directly.
// The cutlass include tree is required at compile time via -I flags.
#include "88_hopper_fmha.cu"
