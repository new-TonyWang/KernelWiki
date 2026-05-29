# GEMM Pattern

| Skill / Sub-Topic | Focus | Key Optimization |
|-------------------|-------|-----------------|
| [Tiling Strategy](tiling-strategy/) | CTA tile dimensions (bM, bN, bK) selection | Balance arithmetic intensity vs SMEM usage vs occupancy |
| [Pipeline Overlap](pipeline-overlap/) | Multi-stage global-to-shared copy overlapping MMA | Hide memory latency with N-buffered pipeline stages |
| [Warp Specialization](warp-specialization/) | Producer warps for data movement, consumer warps for MMA | Asymmetric register allocation; full TC utilization |
| [cuBLAS Integration](cublas-integration/) | Vendor library for standard GEMM shapes | cublasGemmEx / cublasLtMatmul with epilogue fusion |
| Decision Framework | Custom kernel vs cuBLAS selection | Custom for non-standard epilogues, unusual shapes, arch-specific tuning |
