# Pooling Pattern

| Skill | Focus | Key Optimization |
|-------|-------|-----------------|
| Thread-to-Output Mapping | Each thread computes one output element | Simple grid mapping: (W_out, H_out, N*C) |
| Shared Memory Tiling | Load input tile including halo into SMEM | Reuse overlapping input regions across output elements |
| Elements-Per-Thread (EPT) | Each thread computes multiple output elements along width | Amortize SMEM load cost; increase arithmetic intensity |
| Specialized Stride-1 Kernel | Template on kernel_size for stride=1, dilation=1 | Fully unrolled loops; compile-time tile dimensions |
| NCHW vs NHWC Layout | Choose layout for coalesced access in pooling dimension | NHWC enables channel-wise vectorized loads |
| Global Average Pooling | Reduce entire spatial dimensions to single value | Use warp/block reduction instead of sliding window |
| Boundary Handling | Padding and edge cases without branch divergence | Unsigned comparison trick: (unsigned)idx < (unsigned)dim |
| General Kernel Fallback | Arbitrary stride/dilation with __ldg cache hints | Per-element computation with read-only cache path |
