# Warp Specialization -- Related APIs


## Core APIs

| API / Instruction | Source | Description |
|-------------------|--------|-------------|
| `setmaxnreg.{inc/dec}.sync.aligned.u32` | PTX ISA | Dynamically adjust per-warp register count (re-partition between producer/consumer warps) |
| `bar{.cta}.sync a[, b]` | PTX ISA | CTA-level barrier sync for producer-consumer synchronization |
| `bar{.cta}.arrive a, b` | PTX ISA | CTA-level barrier arrive (producer signals without waiting) |
| `elect.sync` | PTX ISA | Elect a single leader thread from active mask (single-thread TMA issue) |
| `cublasLtMatmul()` | cuBLAS | D = alpha*op(A)*op(B) + beta*C with epilogue fusion |
