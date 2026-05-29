# Pipeline Overlap -- Related APIs


## Core APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `cp.async.bulk.tensor[.dim]` | PTX ISA | TMA tensor async copy for pipeline stages (1d-5d addressing) |
| `wgmma.commit_group.sync.aligned` | PTX ISA | Commit outstanding wgmma.mma_async ops into a group |
| `wgmma.wait_group.sync.aligned N` | PTX ISA | Wait for wgmma group N to complete |
| `mbarrier.arrive[.shared]` | PTX ISA | Signal arrival at mbarrier; returns arrival token for pipeline stage sync |
| `mbarrier.try_wait[.acquire][.shared]` | PTX ISA | Blocking try-wait with timeout on mbarrier phase |
| `fence.proxy.async[.space]` | PTX ISA | Proxy fence for TMA-WGMMA async pipeline |
| `cublasLtMatmul()` | cuBLAS | D = alpha*op(A)*op(B) + beta*C with epilogue fusion |
| `cublasLtMatmulAlgoCapGetAttribute()` | cuBLAS | Query algorithm capabilities (tile size, stages, math mode) |
| `cublasLtMatmulAlgoConfigSetAttribute()` | cuBLAS | Set algorithm configuration (tile, stages, splitK, etc.) |
