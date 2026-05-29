# Compute Optimization Techniques

| Technique | When to Consider | Key Indicator |
|-----------|-----------------|---------------|
| [Warp Primitives](warp-primitives/) | Warp-level reduction/broadcast needed without shared memory; shuffle-based communication | NCU: `smsp__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active` high from SMEM ops |
| [Operator Fusion](operator-fusion/) | FMA patterns (a*b+c); multiple element-wise ops that can merge into fewer instructions | NCU: `smsp__inst_executed.avg.pct_of_peak_sustained_active` low; excess instruction count |
| [Fast Math](fast-math/) | Standard math functions (sin, cos, exp, log) are throughput bottlenecks; reduced accuracy OK | NCU: `smsp__inst_executed_pipe_xu.avg.pct_of_peak_sustained_active` high (SFU bound) |
| [Half-Precision Math](half-precision-math/) | Element-wise ops on fp16 data; packed __half2 doubles throughput | NCU: `sm__inst_executed_pipe_fp16.avg.pct_of_peak_sustained_active` low vs potential |
| [Instruction-Level Parallelism](instruction-level-parallelism/) | Loop body has independent ops but compiler unrolling is insufficient; known trip count | NCU: `smsp__warps_issue_stalled_long_scoreboard.avg.pct_of_peak_sustained_active` high |
| [Tensor Core](tensor-core/) | Matrix multiply-accumulate on fp16/bf16/tf32/fp64; SM 7.0+ | NCU: `sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active` at 0% (unused) |
| [Compiler Hints](compiler-hints/) | Pointers alias prevents optimization; loop unrolling needed; launch bounds missing | NCU: `smsp__inst_executed.avg.pct_of_peak_sustained_active` below expected |
| [Warp Specialization](warp-specialization/) | Overlap data movement with computation; producer/consumer pattern; asymmetric resource needs | NCU: `smsp__warps_issue_stalled_mio_throttle.avg.pct_of_peak_sustained_active` high |
