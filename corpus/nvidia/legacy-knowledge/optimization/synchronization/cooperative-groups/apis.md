# Cooperative Groups -- Related APIs


## Core APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `barrier{.cta}.sync[.aligned]` | PTX ISA | CTA barrier with alignment control (non-aligned supported sm_70+) |
| `cooperative_groups::*` | Runtime API | Cooperative Groups API (thread_block, grid_group, tiled_partition, etc.) |
| `cudaLaunchCooperativeKernel` | Runtime API | Launches a kernel where thread blocks can cooperate and synchronize across grid |
