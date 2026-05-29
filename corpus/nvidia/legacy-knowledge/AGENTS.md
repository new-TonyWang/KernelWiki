# KernelPilot Knowledge Base Index

Top-level index, always loaded into Agent context.

Two entry points:
1. **From profiling data**: use NCU metrics to identify which optimization category is most relevant
2. **From operator type**: go directly to the pattern/ section for operator-specific techniques

---

## Optimization Strategies

### Memory Optimization

Techniques to improve memory access efficiency. Profiling hint: NCU `dram_throughput > sm_throughput` suggests memory is the primary bottleneck.

- [Vectorized Access](optimization/memory/vectorized-access/) — float4 aligned loads, unaligned fallback
- [Shared Memory Cache](optimization/memory/shared-memory-cache/) — cache repeatedly accessed data in shared memory; L1/shared config
- [Bank Conflict Avoidance](optimization/memory/bank-conflict-avoidance/) — shared memory padding, swizzle to eliminate bank conflicts
- [Double Buffering / Prefetch](optimization/memory/data-prefetch/) — overlap compute and memory access, hide latency
- [Memory Coalescing](optimization/memory/coalescing/) — consecutive threads access consecutive addresses; alignment
- [Data Layout Transform](optimization/memory/layout-transform/) — AoS→SoA and other layout adjustments
- [L2 Cache Control](optimization/memory/l2-cache-control/) — persistence/streaming access policy, hitRatio tuning
- [Register Pressure](optimization/memory/register-pressure/) — avoid spilling to local memory; __launch_bounds__, __maxnreg__
- [Cache Load Hints](optimization/memory/cache-load-hints/) — __ldg(), explicit cache operators (__ldcg/__ldca/__ldcs)
- [Host-Device Transfer](optimization/memory/host-device-transfer/) — pinned memory, async transfer, zero-copy, batched transfer
- [Unified Memory](optimization/memory/unified-memory/) — prefetch/advise hints, migration overhead
- [NUMA Binding](optimization/memory/numa-binding/) — memory affinity for multi-socket GPU servers

See [optimization/memory/INDEX.md](optimization/memory/INDEX.md)

### Compute Optimization

Techniques to improve compute throughput. Profiling hint: NCU `sm_throughput > dram_throughput` suggests compute is the primary bottleneck.

- [Tensor Core](optimization/compute/tensor-core/) — WMMA/MMA matrix operations
- [Instruction-Level Parallelism](optimization/compute/instruction-level-parallelism/) — loop unrolling, reducing dependency chains
- [Fast Math](optimization/compute/fast-math/) — `__expf`/`__fdividef`, specialized functions (exp2/sinpi/cbrt), integer strength reduction
- [Operator Fusion](optimization/compute/operator-fusion/) — reduce kernel launches and intermediate memory access
- [Compiler Hints](optimization/compute/compiler-hints/) — __launch_bounds__, __restrict__, __grid_constant__, precision flags (-ftz, -prec-div), inline PTX
- [Half-Precision Math](optimization/compute/half-precision-math/) — half2/bfloat162 packed arithmetic (__hadd2, __hfma2)
- [Warp Primitives](optimization/compute/warp-primitives/) — __shfl_sync, __ballot_sync, __reduce_sync for register-to-register communication
- [Warp Specialization](optimization/compute/warp-specialization/) — dedicate warps to producer/consumer roles for pipeline overlap

See [optimization/compute/INDEX.md](optimization/compute/INDEX.md)

### Latency Optimization

Techniques to reduce launch overhead and improve GPU utilization. Profiling hint: NCU shows both `dram_throughput` and `sm_throughput` < 60%.

- [Occupancy Tuning](optimization/latency/occupancy-tuning/) — block size (multiples of 32), register/shared memory limits
- [Warp Divergence](optimization/latency/warp-divergence/) — branch elimination, predication, reduce divergent execution
- [Launch Overhead](optimization/latency/kernel-launch-overhead/) — persistent kernels, launch configuration
- [Stream Concurrency](optimization/latency/stream-concurrency/) — streams, events, priorities, async overlap, implicit sync pitfalls
- [CUDA Graphs](optimization/latency/cuda-graphs/) — define-once execute-many, stream capture, graph update
- [Programmatic Dependent Launch](optimization/latency/programmatic-dependent-launch/) — partial kernel overlap in same stream
- [Dynamic Parallelism](optimization/latency/dynamic-parallelism/) — device-side kernel launch (CDP2) for recursive/irregular workloads
- [Context Management](optimization/latency/context-management/) — avoid multiple CUDA contexts per GPU
- [Green Contexts](optimization/latency/green-contexts/) — SM and work queue partitioning for latency-sensitive work
- [Work Stealing](optimization/latency/work-stealing/) — Cluster Launch Control (Blackwell), dynamic load balancing
- [GPU Library Usage](optimization/latency/gpu-library-usage/) — cuBLAS/cuFFT/Thrust instead of custom kernels

See [optimization/latency/INDEX.md](optimization/latency/INDEX.md)

### Synchronization Optimization

Techniques to reduce synchronization overhead. Profiling hint: NCU shows warp stalls primarily from `barrier` or `atomicCAS`.

- [Barrier Optimization](optimization/synchronization/barrier-optimization/) — reduce sync points, async barriers
- [Atomic Reduction](optimization/synchronization/atomic-reduction/) — warp shuffle instead of atomics, hierarchical reduction
- [Cooperative Groups](optimization/synchronization/cooperative-groups/) — fine-grained synchronization, partitioning, collective ops
- [Thread Scopes](optimization/synchronization/thread-scopes/) — block/cluster/device/system scope performance; use narrowest scope
- [Memory Sync Domains](optimization/synchronization/memory-sync-domains/) — CC 9.0+ fence isolation between compute and communication

See [optimization/synchronization/INDEX.md](optimization/synchronization/INDEX.md)

---

## Operator Patterns → Specialized Knowledge

When optimizing a specific operator type, consult the corresponding pattern:

- [GEMM](pattern/gemm/) — tiling strategy, warp specialization, compute-memory pipeline overlap, cuBLAS integration
- [Convolution](pattern/convolution/) — cuDNN algorithm selection, depthwise custom kernel, depthwise-separable fusion, NCHW/NHWC layout, Winograd
- [Reduction](pattern/reduction/) — warp shuffle, multi-level reduction, atomics
- [Elementwise](pattern/elementwise/) — vectorization, grid-stride loop
- [Attention](pattern/attention/) — online softmax, tiling, FlashAttention patterns
- [Normalization](pattern/normalization/) — LayerNorm/RMSNorm/BatchNorm optimization
- [Pooling](pattern/pooling/) — sliding window, shared memory tiling
- [Scan / Prefix Sum](pattern/scan-cumulative/) — Blelloch scan, decoupled lookback

---

## Advanced Optimization Patterns

**Consult this section when standard techniques are exhausted or deep search stalls.**

- [Resource Trade-offs](advanced/resource-tradeoff/) — registers vs occupancy, shared mem allocation, cross-warp register redistribution
- [Pipeline Design](advanced/pipeline-design/) — compute-memory overlap, multi-stage pipeline, stage restructuring, warp role assignment
- [Branch Elimination](advanced/branch-elimination/) — predicated execution, branchless patterns, uniform control flow
- [Cross-Component Cooperation](advanced/cross-component/) — warp group scheduling, barrier placement, fence strength selection

See [advanced/INDEX.md](advanced/INDEX.md)

### When to consult advanced/

The following signals suggest optimization beyond standard bottleneck categories may be needed:

- 5+ consecutive rounds with no improvement, and standard techniques (vectorization, shared memory, etc.) already attempted
- Profiling shows contradictory metrics (e.g., memory-bound but vectorization ineffective)
- Local memory spills observed but occupancy is not high (→ possible register imbalance across warp groups)
- Warp stalls observed but not typical memory/compute stalls (→ possible barrier/fence overhead)
- Warp divergence observed but branch logic appears necessary (→ may be convertible to predicated execution)

---

## Reasoning Patterns

Meta-knowledge teaching the Agent "how to think", for inspiring creative optimization:

- [Resource Freed, Then What?](reasoning-pattern/resource-freed-then-what.md) — after freeing a resource, check for cascading opportunities
- [Idle Resource Detection](reasoning-pattern/idle-resource-detection.md) — what useful work can idle hardware units do?
- [Constraint Relaxation](reasoning-pattern/constraint-relaxation.md) — when one constraint is lifted, can others be relaxed too?
- [Cross-Warp Thinking](reasoning-pattern/cross-warp-thinking.md) — don't just look at a single warp, look at inter-warp cooperation
- [Trade-off Rebalancing](reasoning-pattern/tradeoff-rebalancing.md) — when resources are unevenly distributed, consider cross-component rebalancing

---

## Hardware Reference

Consult on demand, not loaded by default:

- [H200 Specs](hardware/h200-specs.md) — key parameters (HBM bandwidth, SM count, shared mem size, etc.)
- [Tensor Core Specs](hardware/tensor-core-specs.md) — MMA instruction shapes, precision combinations, throughput per precision
- [Memory Hierarchy](hardware/memory-hierarchy.md) — registers → shared → L1 → L2 → HBM
- [Warp Execution Model](hardware/warp-execution-model.md) — warp scheduling, register allocation, warp groups
- [Hopper Features](hardware/hopper-features/) — TMA, Thread Block Cluster, DSM, async pipeline

---

## API Reference

Breadth fallback layer — search here when the knowledge tree above doesn't cover the relevant API:

- [CUDA Runtime API Index](api-reference/runtime-api-index.md)
- [PTX Instruction Index](api-reference/ptx-instruction-index.md)
- [cuBLAS API Index](api-reference/cublas-api-index.md)
- [Math Intrinsics Index](api-reference/math-intrinsics-index.md)

---

## Experience Log

Level 3 sandbox verification results, archived by date:

→ [experience/](experience/)
