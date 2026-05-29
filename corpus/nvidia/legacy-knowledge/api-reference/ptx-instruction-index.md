# PTX Instruction Index

> Source: PTX ISA 9.2 (CUDA Toolkit 13.2)
> Focus: instructions relevant to CUDA kernel optimization

---

## 1. Data Movement and Conversion Instructions

These instructions are the most critical for memory-bound kernel optimization. They control how data moves between global memory, shared memory, registers, and caches.

### 1.1 Load / Store / Prefetch

| Instruction | Description | SM | Knowledge Node | Relevance |
|---|---|---|---|---|
| ld.global[.vec][.type] | Global memory load; supports .v2/.v4 vector variants | all | memory-bound/vectorized-access, memory-bound/coalescing | core |
| ld.global.nc[.vec][.type] | Non-coherent global load via read-only texture cache path | sm_35+ | memory-bound/cache-load-hints | core |
| ld.shared[.vec][.type] | Shared memory load | all | memory-bound/shared-memory-cache, memory-bound/bank-conflict-avoidance | core |
| ld.shared::cta[.vec][.type] | Shared memory load (CTA scope, explicit) | sm_30+ | memory-bound/shared-memory-cache | core |
| ld.shared::cluster[.vec][.type] | Distributed shared memory load across cluster | sm_90+ (Hopper) | memory-bound/shared-memory-cache | core |
| ld.local[.vec][.type] | Local (thread-private) memory load | all | memory-bound/register-pressure | related |
| ld.const[.vec][.type] | Constant memory load | all | memory-bound/cache-load-hints | related |
| ld.param[.type] | Kernel parameter load | all | memory-bound/cache-load-hints | low-relevance |
| ldu.global[.vec][.type] | Uniform global load (same address across warp) | sm_20+ | memory-bound/coalescing | related |
| st.global[.vec][.type] | Global memory store; supports .v2/.v4 | all | memory-bound/vectorized-access, memory-bound/coalescing | core |
| st.shared[.vec][.type] | Shared memory store | all | memory-bound/shared-memory-cache, memory-bound/bank-conflict-avoidance | core |
| st.shared::cluster[.vec][.type] | Distributed shared memory store across cluster | sm_90+ (Hopper) | memory-bound/shared-memory-cache | core |
| st.local[.vec][.type] | Local memory store | all | memory-bound/register-pressure | related |
| st.async[.type] | Asynchronous store to shared memory; .b128 variant in PTX 9.2 | sm_90+ (Hopper) | memory-bound/data-prefetch | core |
| st.bulk[.type] | Bulk store to shared memory (zeros or initial values) | sm_90+ (Hopper) | memory-bound/shared-memory-cache | related |
| prefetch{.space} | Prefetch data into cache (.L1/.L2) | sm_20+ | memory-bound/data-prefetch, memory-bound/cache-load-hints | core |
| prefetchu.L1 | Prefetch uniform address into L1 | sm_20+ | memory-bound/data-prefetch | related |

### 1.2 Cache Operators and Eviction Hints

| Instruction / Qualifier | Description | SM | Knowledge Node | Relevance |
|---|---|---|---|---|
| .ca (cache-all) | Cache at all levels (L1+L2), default for ld | sm_20+ | memory-bound/cache-load-hints | core |
| .cg (cache-global) | Cache in L2 only, bypass L1 | sm_20+ | memory-bound/cache-load-hints, memory-bound/l2-cache-control | core |
| .cs (cache-streaming) | Streaming access, evict-first policy | sm_20+ | memory-bound/cache-load-hints, memory-bound/l2-cache-control | core |
| .lu (last-use) | Hint that data will not be reused | sm_20+ | memory-bound/cache-load-hints | related |
| .cv (cache-volatile) | Invalidate + refetch from memory | sm_20+ | memory-bound/cache-load-hints | related |
| .wb (write-back) | Default store caching, write-back all levels | sm_20+ | memory-bound/cache-load-hints | related |
| .wt (write-through) | Write-through to system memory | sm_20+ | memory-bound/cache-load-hints | related |
| evict_normal / evict_first / evict_last / no_allocate | Cache eviction priority hints | sm_70+ | memory-bound/l2-cache-control | core |
| .L2::cache_hint | L2 cache policy hint (used with createpolicy) | sm_80+ | memory-bound/l2-cache-control | core |
| createpolicy | Create L2 cache access policy | sm_80+ | memory-bound/l2-cache-control | core |
| applypriority.global.L2 | Apply eviction priority to L2 cache range | sm_80+ | memory-bound/l2-cache-control | related |
| discard.global.L2 | Discard L2 cache line (no write-back) | sm_80+ | memory-bound/l2-cache-control | related |

### 1.3 Asynchronous Copy Instructions

| Instruction | Description | SM | Knowledge Node | Relevance |
|---|---|---|---|---|
| cp.async.ca.shared.global[.vec] | Async copy global->shared (bypasses registers) | sm_80+ | memory-bound/data-prefetch, memory-bound/shared-memory-cache | core |
| cp.async.cg.shared.global[.vec] | Async copy global->shared, L2 cache only | sm_80+ | memory-bound/data-prefetch, memory-bound/cache-load-hints | core |
| cp.async.commit_group | Commit outstanding async copies into a group | sm_80+ | memory-bound/data-prefetch | core |
| cp.async.wait_group N | Wait for async copy group N to complete | sm_80+ | memory-bound/data-prefetch | core |
| cp.async.wait_all | Wait for all async copy groups | sm_80+ | memory-bound/data-prefetch | core |
| cp.async.bulk[.dst][.src] | Bulk async copy (TMA-style, large transfers) | sm_90+ (Hopper) | memory-bound/data-prefetch, memory-bound/shared-memory-cache | core |
| cp.reduce.async.bulk[.op] | Async bulk copy with reduction | sm_90+ (Hopper) | memory-bound/data-prefetch, synchronization-bound/atomic-reduction | core |
| cp.async.bulk.prefetch | Bulk async prefetch to L2 cache | sm_90+ (Hopper) | memory-bound/data-prefetch, memory-bound/l2-cache-control | core |
| cp.async.bulk.tensor[.dim] | TMA tensor async copy (1d-5d addressing) | sm_90+ (Hopper) | memory-bound/data-prefetch, memory-bound/layout-transform | core |
| cp.reduce.async.bulk.tensor[.op] | TMA tensor async copy with reduction | sm_90+ (Hopper) | memory-bound/data-prefetch, synchronization-bound/atomic-reduction | core |
| cp.async.bulk.prefetch.tensor | TMA tensor prefetch to L2 | sm_90+ (Hopper) | memory-bound/data-prefetch, memory-bound/l2-cache-control | core |
| cp.async.bulk.commit_group | Commit bulk async copies into group | sm_90+ (Hopper) | memory-bound/data-prefetch | core |
| cp.async.bulk.wait_group N | Wait for bulk async group N | sm_90+ (Hopper) | memory-bound/data-prefetch | core |
| tensormap.replace | Dynamically modify tensor map parameters | sm_90+ (Hopper) | memory-bound/layout-transform | related |

### 1.4 Multi-Memory (multimem) Instructions

| Instruction | Description | SM | Knowledge Node | Relevance |
|---|---|---|---|---|
| multimem.ld_reduce[.op][.type] | Load+reduce from multiple memory locations | sm_90+ (Hopper) | synchronization-bound/atomic-reduction | related |
| multimem.st[.type] | Multi-memory store | sm_90+ (Hopper) | synchronization-bound/atomic-reduction | related |
| multimem.red[.op][.type] | Multi-memory reduction | sm_90+ (Hopper) | synchronization-bound/atomic-reduction | related |
| multimem.cp.async.bulk | Multi-memory bulk async copy | sm_90+ (Hopper) | memory-bound/data-prefetch | related |
| multimem.cp.reduce.async.bulk | Multi-memory bulk async copy+reduce | sm_90+ (Hopper) | synchronization-bound/atomic-reduction | related |

### 1.5 Register Movement and Conversion

| Instruction | Description | SM | Knowledge Node | Relevance |
|---|---|---|---|---|
| mov[.type] | Register-to-register move; also pack/unpack vectors (.b32, .b64, .b128) | all | compute-bound/instruction-level-parallelism | related |
| shfl.sync[.mode].b32 | Warp shuffle: exchange registers between lanes (.up/.down/.bfly/.idx) | sm_30+ | compute-bound/warp-primitives, pattern/reduction | core |
| prmt.b32[.mode] | Byte permute across two 32-bit registers | sm_20+ | memory-bound/layout-transform | related |
| cvt[.rnd][.sat].dtype.stype | Type conversion (int/float, narrowing/widening, rounding modes) | all | compute-bound/half-precision-math, compute-bound/fast-math | core |
| cvt.pack[.type] | Pack multiple values into a register | sm_72+ | compute-bound/half-precision-math | related |
| cvta.space.size | Convert between generic and space-specific addresses | all | memory-bound/shared-memory-cache | related |
| isspacep.space | Test if generic address is in given state space | all | memory-bound/shared-memory-cache | low-relevance |
| mapa.space | Map shared memory address to another CTA in cluster | sm_90+ (Hopper) | memory-bound/shared-memory-cache | related |
| getctarank | Get CTA rank from shared memory address in cluster | sm_90+ (Hopper) | memory-bound/shared-memory-cache | related |

---

## 2. Integer Arithmetic Instructions

### 2.1 Basic Integer Arithmetic

| Instruction | Description | SM | Knowledge Node | Relevance |
|---|---|---|---|---|
| add[.type] | Integer add; .u16x2/.s16x2 SIMD variants (sm_90+); .u8x4/.s8x4 (sm_120f+) | all | compute-bound/instruction-level-parallelism | related |
| sub[.type] | Integer subtract; .u8x4/.s8x4 SIMD variants (sm_120f+) | all | compute-bound/instruction-level-parallelism | related |
| mul[.mode][.type] | Integer multiply (.hi/.lo/.wide) | all | compute-bound/instruction-level-parallelism | related |
| mad[.mode][.type] | Multiply-add (fused mul+add, integer) | all | compute-bound/instruction-level-parallelism, compute-bound/operator-fusion | related |
| mul24[.mode].type | 24-bit integer multiply (legacy) | all | compute-bound/fast-math | low-relevance |
| mad24[.mode].type | 24-bit integer multiply-add (legacy) | all | compute-bound/fast-math | low-relevance |
| div[.type] | Integer divide (slow) | all | compute-bound/fast-math | related |
| rem[.type] | Integer remainder | all | compute-bound/fast-math | low-relevance |
| abs[.type] | Integer absolute value | all | compute-bound/instruction-level-parallelism | low-relevance |
| neg[.type] | Integer negate; .s8x4 SIMD variant (sm_120f+) | all | compute-bound/instruction-level-parallelism | low-relevance |
| min[.type] | Integer min; SIMD .u16x2/.s16x2 (sm_90+), .relu variants; .u8x4/.s8x4 (sm_120f+) | all | compute-bound/instruction-level-parallelism | related |
| max[.type] | Integer max; SIMD .u16x2/.s16x2 (sm_90+), .relu variants; .u8x4/.s8x4 (sm_120f+) | all | compute-bound/instruction-level-parallelism | related |
| sad[.type] | Sum of absolute differences | all | pattern/reduction | low-relevance |
| dp4a.atype.btype | 4-way byte dot-product-accumulate (INT8) | sm_61+ | compute-bound/tensor-core, pattern/gemm | core |
| dp2a.mode.atype.btype | 2-way dot-product-accumulate (INT16xINT8) | sm_61+ | compute-bound/tensor-core, pattern/gemm | related |

### 2.2 Bit Manipulation

| Instruction | Description | SM | Knowledge Node | Relevance |
|---|---|---|---|---|
| popc.type | Population count (count set bits) | sm_20+ | compute-bound/instruction-level-parallelism | low-relevance |
| clz.type | Count leading zeros | sm_20+ | compute-bound/instruction-level-parallelism | low-relevance |
| bfind[.shiftamt].type | Find most significant non-sign bit | sm_20+ | compute-bound/instruction-level-parallelism | low-relevance |
| brev.type | Bit reverse | sm_20+ | compute-bound/instruction-level-parallelism | low-relevance |
| bfe.type | Bit field extract | sm_20+ | compute-bound/instruction-level-parallelism | low-relevance |
| bfi.type | Bit field insert | sm_20+ | compute-bound/instruction-level-parallelism | low-relevance |
| bmsk.mode.b32 | Bit field mask generation | sm_70+ | compute-bound/instruction-level-parallelism | low-relevance |
| szext.mode.type | Sign/zero extend | sm_70+ | compute-bound/instruction-level-parallelism | low-relevance |
| fns.b32 | Find n-th set bit | sm_30+ | compute-bound/instruction-level-parallelism | low-relevance |

### 2.3 Extended-Precision Integer Arithmetic

| Instruction | Description | SM | Knowledge Node | Relevance |
|---|---|---|---|---|
| add.cc[.type] | Add with carry-out | all | compute-bound/instruction-level-parallelism | low-relevance |
| addc[.cc][.type] | Add with carry-in and optional carry-out | all | compute-bound/instruction-level-parallelism | low-relevance |
| sub.cc[.type] | Subtract with borrow-out | all | compute-bound/instruction-level-parallelism | low-relevance |
| subc[.cc][.type] | Subtract with borrow-in and optional borrow-out | all | compute-bound/instruction-level-parallelism | low-relevance |
| mad.cc[.mode][.type] | Multiply-add with carry-out | sm_20+ | compute-bound/instruction-level-parallelism | low-relevance |
| madc[.cc][.mode][.type] | Multiply-add with carry-in/out (extended precision mul) | sm_20+ | compute-bound/instruction-level-parallelism | low-relevance |

---

## 3. Floating-Point Instructions (f32/f64)

### 3.1 Basic FP Arithmetic

| Instruction | Description | SM | Knowledge Node | Relevance |
|---|---|---|---|---|
| add.rnd[.ftz][.sat].f32 | FP32 add; .f32x2 SIMD variant (sm_100+, **Blackwell**) | all | compute-bound/fast-math | related |
| add.rnd.f64 | FP64 add | sm_13+ | compute-bound/fast-math | related |
| sub.rnd[.ftz][.sat].f32 | FP32 subtract | all | compute-bound/fast-math | related |
| sub.rnd.f64 | FP64 subtract | sm_13+ | compute-bound/fast-math | related |
| mul.rnd[.ftz][.sat].f32 | FP32 multiply; .f32x2 SIMD variant (sm_100+, **Blackwell**) | all | compute-bound/fast-math | related |
| mul.rnd.f64 | FP64 multiply | sm_13+ | compute-bound/fast-math | related |
| fma.rnd[.ftz][.sat].f32 | FP32 fused multiply-add; .f32x2 variant (sm_100+, **Blackwell**) | sm_20+ | compute-bound/fast-math, compute-bound/operator-fusion | core |
| fma.rnd.f64 | FP64 fused multiply-add | sm_13+ | compute-bound/fast-math, compute-bound/operator-fusion | core |
| mad[.rnd][.ftz][.sat].f32 | FP32 multiply-add (equiv to fma on sm_20+) | all | compute-bound/operator-fusion | related |
| div.approx[.ftz].f32 | Fast approximate FP32 divide | all | compute-bound/fast-math | core |
| div.full[.ftz].f32 | Full-range approximate FP32 divide | all | compute-bound/fast-math | related |
| div.rnd[.ftz].f32 | IEEE-754 compliant FP32 divide | sm_20+ | compute-bound/fast-math | related |
| div.rnd.f64 | IEEE-754 compliant FP64 divide | sm_13+ | compute-bound/fast-math | related |

### 3.2 FP Transcendentals and Special Functions

| Instruction | Description | SM | Knowledge Node | Relevance |
|---|---|---|---|---|
| rcp.approx[.ftz].f32 | Fast reciprocal (1/x), max 1 ulp error | all | compute-bound/fast-math | core |
| rcp.rnd[.ftz].f32 | IEEE-compliant reciprocal | sm_20+ | compute-bound/fast-math | related |
| rcp.approx.ftz.f64 | Fast approximate f64 reciprocal | sm_20+ | compute-bound/fast-math | related |
| sqrt.approx[.ftz].f32 | Fast approximate square root | all | compute-bound/fast-math | core |
| sqrt.rnd[.ftz].f32 | IEEE-compliant square root | sm_20+ | compute-bound/fast-math | related |
| rsqrt.approx[.ftz].f32 | Fast reciprocal square root (1/sqrt(x)) | all | compute-bound/fast-math | core |
| rsqrt.approx.f64 | Approximate f64 reciprocal square root (emulated, slow) | sm_13+ | compute-bound/fast-math | related |
| sin.approx[.ftz].f32 | Fast approximate sine | all | compute-bound/fast-math | core |
| cos.approx[.ftz].f32 | Fast approximate cosine | all | compute-bound/fast-math | core |
| lg2.approx[.ftz].f32 | Fast approximate log2 | all | compute-bound/fast-math | core |
| ex2.approx[.ftz].f32 | Fast approximate 2^x | all | compute-bound/fast-math | core |
| tanh.approx.f32 | Fast approximate tanh (activation function) | sm_75+ | compute-bound/fast-math, pattern/attention | core |

### 3.3 FP Comparison and Min/Max

| Instruction | Description | SM | Knowledge Node | Relevance |
|---|---|---|---|---|
| min[.ftz][.NaN].f32 | FP32 min; .xorsign.abs variants (sm_86+); 3-input (sm_100+, **Blackwell**) | all | compute-bound/fast-math | related |
| max[.ftz][.NaN].f32 | FP32 max; .xorsign.abs variants (sm_86+); 3-input (sm_100+, **Blackwell**) | all | compute-bound/fast-math | related |
| min.f64 / max.f64 | FP64 min/max | sm_13+ | compute-bound/fast-math | low-relevance |
| testp.op.f32 / .f64 | Test FP property (finite, infinite, nan, normal, subnormal, number) | all | compute-bound/fast-math | low-relevance |
| copysign.f32 / .f64 | Copy sign bit from one value to another | all | compute-bound/fast-math | low-relevance |
| abs[.ftz].f32 / abs.f64 | FP absolute value | all | compute-bound/fast-math | low-relevance |
| neg[.ftz].f32 / neg.f64 | FP negate | all | compute-bound/fast-math | low-relevance |

---

## 4. Half Precision and BFloat16 Instructions

| Instruction | Description | SM | Knowledge Node | Relevance |
|---|---|---|---|---|
| add[.rnd][.ftz][.sat].f16 / .f16x2 | FP16 add; SIMD 2-wide on .f16x2 | sm_53+ | compute-bound/half-precision-math | core |
| add[.rnd].bf16 / .bf16x2 | BF16 add; SIMD 2-wide | sm_90+ (Hopper) | compute-bound/half-precision-math | core |
| sub[.rnd][.ftz][.sat].f16 / .f16x2 | FP16 subtract; SIMD 2-wide | sm_53+ | compute-bound/half-precision-math | core |
| sub[.rnd].bf16 / .bf16x2 | BF16 subtract; SIMD 2-wide | sm_90+ (Hopper) | compute-bound/half-precision-math | core |
| mul[.rnd][.ftz][.sat].f16 / .f16x2 | FP16 multiply; SIMD 2-wide | sm_53+ | compute-bound/half-precision-math | core |
| mul[.rnd].bf16 / .bf16x2 | BF16 multiply; SIMD 2-wide | sm_90+ (Hopper) | compute-bound/half-precision-math | core |
| fma.rnd[.ftz][.sat][.relu].f16 / .f16x2 | FP16 fused multiply-add; relu/oob variants | sm_53+ (.relu: sm_80+, .oob: sm_90+) | compute-bound/half-precision-math, compute-bound/operator-fusion | core |
| fma.rnd[.relu].bf16 / .bf16x2 | BF16 fused multiply-add; .oob variant (sm_90+) | sm_80+ | compute-bound/half-precision-math, compute-bound/operator-fusion | core |
| neg/abs .f16/.f16x2/.bf16/.bf16x2 | Half-precision negate/absolute value | sm_53+ (.bf16: sm_80+) | compute-bound/half-precision-math | related |
| min/max [.NaN][.xorsign.abs] .f16/.f16x2/.bf16/.bf16x2 | Half-precision min/max with NaN handling | sm_80+ (.xorsign: sm_86+) | compute-bound/half-precision-math | related |
| tanh.approx .f16/.f16x2/.bf16/.bf16x2 | Half-precision approximate tanh | sm_75+ (.bf16: sm_90+) | compute-bound/half-precision-math, pattern/attention | core |
| ex2.approx .f16/.f16x2/.bf16/.bf16x2 | Half-precision approximate 2^x | sm_75+ (.bf16: sm_90+) | compute-bound/half-precision-math | related |

### 4.1 Mixed Precision Floating-Point Instructions

| Instruction | Description | SM | Knowledge Node | Relevance |
|---|---|---|---|---|
| add.rnd[.sat].f32.{f16/bf16} | Mixed-precision add: f16/bf16 input -> f32 output | sm_100+ (**Blackwell**) | compute-bound/half-precision-math | related |
| sub.rnd[.sat].f32.{f16/bf16} | Mixed-precision sub: f16/bf16 input -> f32 output | sm_100+ (**Blackwell**) | compute-bound/half-precision-math | related |
| fma.rnd[.sat].f32.{f16/bf16} | Mixed-precision fma: f16/bf16 inputs -> f32 output | sm_100+ (**Blackwell**) | compute-bound/half-precision-math, compute-bound/operator-fusion | core |

---

## 5. Comparison and Selection Instructions

| Instruction | Description | SM | Knowledge Node | Relevance |
|---|---|---|---|---|
| setp.CmpOp[.BoolOp][.ftz].type | Set predicate from comparison; dual-predicate output (p\|q) | all | latency-bound/warp-divergence | related |
| set.CmpOp[.BoolOp][.ftz].dtype.stype | Compare and write typed result (not predicate) | all | latency-bound/warp-divergence | low-relevance |
| selp.type | Select based on predicate (ternary/conditional move) | all | latency-bound/warp-divergence | related |
| slct[.ftz].dtype.{s32/f32} | Select based on sign of third operand | all | latency-bound/warp-divergence | low-relevance |
| setp.CmpOp .f16/.f16x2/.bf16/.bf16x2 | Half-precision predicate comparison | sm_53+ (.bf16: sm_90+) | compute-bound/half-precision-math | related |
| set.CmpOp .f16/.f16x2/.bf16/.bf16x2 | Half-precision typed comparison | sm_53+ (.bf16: sm_90+) | compute-bound/half-precision-math | related |

---

## 6. Logic and Shift Instructions

| Instruction | Description | SM | Knowledge Node | Relevance |
|---|---|---|---|---|
| and.type | Bitwise AND (.pred/.b16/.b32/.b64) | all | compute-bound/instruction-level-parallelism | low-relevance |
| or.type | Bitwise OR | all | compute-bound/instruction-level-parallelism | low-relevance |
| xor.type | Bitwise XOR | all | compute-bound/instruction-level-parallelism | low-relevance |
| not.type | Bitwise NOT | all | compute-bound/instruction-level-parallelism | low-relevance |
| cnot.type | C-style logical negation | all | compute-bound/instruction-level-parallelism | low-relevance |
| lop3.b32 | Arbitrary 3-input logical operation (256 possible ops via LUT) | sm_50+ | compute-bound/instruction-level-parallelism, compute-bound/operator-fusion | related |
| shf.{l/r}.{clamp/wrap}.type | Funnel shift (double-width shift across two registers) | sm_35+ | compute-bound/instruction-level-parallelism | low-relevance |
| shl.type | Shift left | all | compute-bound/instruction-level-parallelism | low-relevance |
| shr.type | Shift right (arithmetic or logical) | all | compute-bound/instruction-level-parallelism | low-relevance |

---

## 7. Control Flow Instructions

| Instruction | Description | SM | Knowledge Node | Relevance |
|---|---|---|---|---|
| @{!}p | Predicated execution (all instructions) | all | latency-bound/warp-divergence | core |
| bra[.uni] | Branch (conditional/unconditional; .uni = uniform) | all | latency-bound/warp-divergence | related |
| brx.idx[.uni] | Indexed branch from label list | sm_30+ | latency-bound/warp-divergence | low-relevance |
| call[.uni] | Function call (direct/indirect) | all | latency-bound/kernel-launch-overhead | low-relevance |
| ret[.uni] | Return from function | all | latency-bound/kernel-launch-overhead | low-relevance |
| exit | Terminate thread | all | latency-bound/warp-divergence | related |

---

## 8. Parallel Synchronization and Communication Instructions

### 8.1 Barriers

| Instruction | Description | SM | Knowledge Node | Relevance |
|---|---|---|---|---|
| bar{.cta}.sync a[, b] | CTA-level barrier sync (16 named barriers) | all | synchronization-bound/barrier-optimization | core |
| bar{.cta}.arrive a, b | CTA-level barrier arrive (no wait) | sm_20+ | synchronization-bound/barrier-optimization | core |
| bar{.cta}.red.{popc/and/or} | CTA-level barrier with reduction | sm_20+ | synchronization-bound/barrier-optimization, pattern/reduction | core |
| barrier{.cta}.sync[.aligned] | CTA barrier with alignment control (non-aligned supported sm_70+) | sm_30+ | synchronization-bound/barrier-optimization, synchronization-bound/cooperative-groups | core |
| bar.warp.sync membermask | Warp-level barrier sync (subset of threads via bitmask) | sm_30+ | synchronization-bound/barrier-optimization, compute-bound/warp-primitives | core |
| barrier.cluster.arrive[.sem][.aligned] | Cluster-level barrier arrive (.release/.relaxed) | sm_90+ (Hopper) | synchronization-bound/barrier-optimization, synchronization-bound/thread-scopes | core |
| barrier.cluster.wait[.acquire][.aligned] | Cluster-level barrier wait | sm_90+ (Hopper) | synchronization-bound/barrier-optimization, synchronization-bound/thread-scopes | core |

### 8.2 Memory Barriers and Fences

| Instruction | Description | SM | Knowledge Node | Relevance |
|---|---|---|---|---|
| membar.cta / .gl / .sys | Memory barrier at CTA/global/system level (old-style) | all (.sys: sm_20+) | synchronization-bound/thread-scopes, synchronization-bound/memory-sync-domains | core |
| fence[.sem].scope | Thread fence (.sc/.acq_rel/.acquire/.release at .cta/.cluster/.gpu/.sys) | sm_70+ (.cluster: sm_90+) | synchronization-bound/thread-scopes, synchronization-bound/memory-sync-domains | core |
| fence.proxy.alias | Proxy fence for aliased memory accesses | sm_70+ | synchronization-bound/memory-sync-domains | related |
| fence.proxy.async[.space] | Proxy fence for async operations (.global, .shared::cta, .shared::cluster) | sm_90+ (Hopper) | synchronization-bound/memory-sync-domains | core |
| fence.mbarrier_init.release.cluster | Fence for mbarrier initialization | sm_90+ (Hopper) | synchronization-bound/barrier-optimization | related |
| fence.proxy.tensormap::generic | Fence for tensormap updates | sm_90+ (Hopper) | synchronization-bound/memory-sync-domains | related |

### 8.3 mbarrier (Asynchronous Barrier) Instructions

| Instruction | Description | SM | Knowledge Node | Relevance |
|---|---|---|---|---|
| mbarrier.init[.shared::cta].b64 | Initialize mbarrier object with expected arrival count | sm_80+ | synchronization-bound/barrier-optimization | core |
| mbarrier.inval[.shared::cta].b64 | Invalidate mbarrier object | sm_80+ | synchronization-bound/barrier-optimization | related |
| mbarrier.arrive[.shared{::cta/::cluster}].b64 | Signal arrival at mbarrier; returns arrival token | sm_80+ (.cluster: sm_90+) | synchronization-bound/barrier-optimization | core |
| mbarrier.arrive.expect_tx | Arrive with expected async transaction byte count | sm_90+ (Hopper) | synchronization-bound/barrier-optimization, memory-bound/data-prefetch | core |
| mbarrier.arrive_drop[.shared::cta].b64 | Arrive and drop participation from future phases | sm_80+ | synchronization-bound/barrier-optimization | related |
| mbarrier.test_wait[.acquire][.shared::cta].b64 | Non-blocking test if mbarrier phase is complete | sm_80+ | synchronization-bound/barrier-optimization | core |
| mbarrier.try_wait[.acquire][.shared{::cta/::cluster}].b64 | Blocking try-wait with timeout on mbarrier phase | sm_80+ (.try: sm_90+) | synchronization-bound/barrier-optimization | core |
| mbarrier.pending_count.b64 | Get pending arrival count from mbarrier state | sm_80+ | synchronization-bound/barrier-optimization | related |
| cp.async.mbarrier.arrive | Signal mbarrier from async copy completion | sm_80+ | synchronization-bound/barrier-optimization, memory-bound/data-prefetch | core |

### 8.4 Atomic and Reduction Operations

| Instruction | Description | SM | Knowledge Node | Relevance |
|---|---|---|---|---|
| atom[.sem][.scope][.space].op.type | Atomic read-modify-write (.add/.min/.max/.inc/.dec/.and/.or/.xor/.exch/.cas) | all (scoped: sm_70+) | synchronization-bound/atomic-reduction | core |
| atom[.sem][.scope].add.noftz.f16/bf16 | Atomic FP16/BF16 add | sm_70+ (.bf16: sm_90+) | synchronization-bound/atomic-reduction, compute-bound/half-precision-math | core |
| atom.cas.b128 / atom.exch.b128 | 128-bit atomic CAS and exchange | sm_90+ (Hopper) | synchronization-bound/atomic-reduction | related |
| atom[.scope].add.vec.f32 | Vectorized atomic FP32 add (.v2/.v4) | sm_90+ (Hopper) | synchronization-bound/atomic-reduction, memory-bound/vectorized-access | core |
| red[.sem][.scope][.space].op.type | Reduction (no return value); same ops as atom | all (scoped: sm_70+) | synchronization-bound/atomic-reduction, pattern/reduction | core |
| red.async[.sem].scope.space.op.type | Asynchronous reduction | sm_90+ (Hopper) | synchronization-bound/atomic-reduction | related |

### 8.5 Warp-Level Communication

| Instruction | Description | SM | Knowledge Node | Relevance |
|---|---|---|---|---|
| vote.sync.{all/any/uni/ballot}.b32 | Warp vote with sync (all/any-true, uniform, ballot mask) | sm_30+ | compute-bound/warp-primitives, latency-bound/warp-divergence | core |
| match.sync.{any/all}.b32/b64 | Warp match: find threads with matching values | sm_70+ | compute-bound/warp-primitives | related |
| activemask.b32 | Get bitmask of active threads in warp | sm_20+ | compute-bound/warp-primitives, latency-bound/warp-divergence | related |
| redux.sync.op.type | Warp-level reduction (.add/.min/.max/.and/.or/.xor) | sm_80+ | compute-bound/warp-primitives, pattern/reduction | core |
| elect.sync | Elect a single leader thread from active mask | sm_80+ | compute-bound/warp-primitives | related |

### 8.6 Grid-Level Control

| Instruction | Description | SM | Knowledge Node | Relevance |
|---|---|---|---|---|
| griddepcontrol.launch_dependents | Signal dependent grids may launch (programmatic dependent launch) | sm_90+ (Hopper) | latency-bound/programmatic-dependent-launch | core |
| griddepcontrol.wait | Wait for prerequisite grid completion | sm_90+ (Hopper) | latency-bound/programmatic-dependent-launch | core |
| clusterlaunchcontrol.try_cancel | Try to cancel a pending cluster launch | sm_100+ (**Blackwell**) | latency-bound/programmatic-dependent-launch | related |
| clusterlaunchcontrol.query_cancel | Query if a cluster launch was cancelled | sm_100+ (**Blackwell**) | latency-bound/programmatic-dependent-launch | related |

### 8.7 Tensormap Fence

| Instruction | Description | SM | Knowledge Node | Relevance |
|---|---|---|---|---|
| tensormap.cp_fenceproxy | Fence for tensormap copy operations between proxies | sm_90+ (Hopper) | synchronization-bound/memory-sync-domains | related |

---

## 9. Warp-Level Matrix Multiply-Accumulate (MMA) Instructions

### 9.1 WMMA (Warp Matrix Multiply-Accumulate)

| Instruction | Description | SM | Knowledge Node | Relevance |
|---|---|---|---|---|
| wmma.load.a/b/c.sync[.aligned].shape.layout[.type] | Load matrix fragment from memory to registers | sm_70+ | compute-bound/tensor-core, pattern/gemm | core |
| wmma.store.d.sync[.aligned].shape.layout[.type] | Store matrix fragment from registers to memory | sm_70+ | compute-bound/tensor-core, pattern/gemm | core |
| wmma.mma.sync[.aligned].shape.dtype.atype.btype | Warp-level MxNxK matrix multiply-accumulate | sm_70+ | compute-bound/tensor-core, pattern/gemm | core |

Supported shapes and types (wmma):
- `.f16` inputs: m16n16k16, m8n32k16, m32n8k16 (sm_70+)
- `.bf16` inputs: m16n16k16, m8n32k16, m32n8k16 (sm_80+)
- `.tf32` inputs: m16n16k8 (sm_80+)
- `.u8/.s8` inputs: m16n16k16, m8n32k16, m32n8k16 (sm_72+)
- `.u4/.s4` inputs: m8n8k32 (sm_75+)
- `.b1` inputs: m8n8k128 (sm_75+)

### 9.2 MMA (Low-Level Matrix Multiply-Accumulate)

| Instruction | Description | SM | Knowledge Node | Relevance |
|---|---|---|---|---|
| mma.sync.aligned.shape.row.col.dtype.atype.btype.ctype | Warp-level MMA with explicit register mapping | sm_75+ | compute-bound/tensor-core, pattern/gemm | core |
| mma.sp.sync.aligned.shape.row.col.dtype.atype.btype.ctype | Sparse MMA (structured sparsity 2:4) | sm_80+ | compute-bound/tensor-core, pattern/gemm | core |

Supported shapes and types (mma):
- `.f16` inputs: m8n8k4 (sm_70+), m16n8k8 (sm_75+), m16n8k16 (sm_80+)
- `.bf16` inputs: m16n8k8, m16n8k16 (sm_80+)
- `.tf32` inputs: m16n8k4, m16n8k8 (sm_80+)
- `.f64` inputs: m8n8k4 (sm_80+), m16n8k4/k8/k16 (sm_90+, Hopper)
- `.u8/.s8` inputs: m8n8k16 (sm_75+), m16n8k16/k32 (sm_80+)
- `.u4/.s4` inputs: m8n8k32 (sm_75+), m16n8k32/k64 (sm_80+)
- `.b1` inputs: m8n8k128 (sm_75+), m16n8k128/k256 (sm_80+)
- `.e4m3/.e5m2` (FP8) inputs: m16n8k32 (sm_89+), m16n8k16 (sm_100+, **Blackwell**)
- `.e3m2/.e2m3/.e2m1` inputs: m16n8k32 (sm_100+, **Blackwell**)

---

## 10. Asynchronous Warpgroup MMA (WGMMA) Instructions

All WGMMA instructions require sm_90+ (Hopper).

| Instruction | Description | SM | Knowledge Node | Relevance |
|---|---|---|---|---|
| wgmma.mma_async.sync.aligned.shape.dtype.atype.btype | Warpgroup-level async MMA (4 warps = 128 threads) | sm_90+ (Hopper) | compute-bound/tensor-core, pattern/gemm, pattern/attention | core |
| wgmma.mma_async.sp.sync.aligned.shape... | Sparse warpgroup MMA (structured 2:4 sparsity) | sm_90+ (Hopper) | compute-bound/tensor-core, pattern/gemm | core |
| wgmma.fence.sync.aligned | Fence before wgmma (declare register/smem readiness) | sm_90+ (Hopper) | compute-bound/tensor-core, synchronization-bound/barrier-optimization | core |
| wgmma.commit_group.sync.aligned | Commit outstanding wgmma.mma_async ops into a group | sm_90+ (Hopper) | compute-bound/tensor-core | core |
| wgmma.wait_group.sync.aligned N | Wait for wgmma group N to complete | sm_90+ (Hopper) | compute-bound/tensor-core | core |

Supported shapes (wgmma, M=64 fixed):
- `.f16/.bf16` inputs: m64nNk16 (N=8..256 step 8), dense and sparse
- `.tf32` inputs: m64nNk8 (dense) and m64nNk16 (sparse)
- `.e4m3/.e5m2` inputs: m64nNk32 (dense) and m64nNk64 (sparse)
- `.u8/.s8` inputs: m64nNk32 (dense) and m64nNk64 (sparse)
- `.b1` inputs: m64nNk256 (dense)

Matrix A source: registers or shared memory.
Matrix B source: shared memory only.

---

## 11. TensorCore 5th Generation (tcgen05) Instructions

All tcgen05 instructions require sm_100+ (**Blackwell**).

| Instruction | Description | SM | Knowledge Node | Relevance |
|---|---|---|---|---|
| tcgen05.alloc | Allocate Tensor Memory (dedicated on-chip memory for TC5) | sm_100+ (**Blackwell**) | compute-bound/tensor-core | core |
| tcgen05.dealloc | Deallocate Tensor Memory | sm_100+ (**Blackwell**) | compute-bound/tensor-core | core |
| tcgen05.relinquish_alloc_permit | Relinquish allocation permit for Tensor Memory | sm_100+ (**Blackwell**) | compute-bound/tensor-core | related |
| tcgen05.ld[.shape].tmem | Load from Tensor Memory to registers | sm_100+ (**Blackwell**) | compute-bound/tensor-core | core |
| tcgen05.st[.shape].tmem | Store from registers to Tensor Memory | sm_100+ (**Blackwell**) | compute-bound/tensor-core | core |
| tcgen05.cp[.shape] | Copy data between shared memory and Tensor Memory | sm_100+ (**Blackwell**) | compute-bound/tensor-core, memory-bound/data-prefetch | core |
| tcgen05.mma[.kind][.shape] | 5th-gen TensorCore MMA (supports M=64/128/256 via CTA groups) | sm_100+ (**Blackwell**) | compute-bound/tensor-core, pattern/gemm, pattern/attention | core |
| tcgen05.mma.sp[.kind][.shape] | 5th-gen TensorCore sparse MMA | sm_100+ (**Blackwell**) | compute-bound/tensor-core, pattern/gemm | core |
| tcgen05.fence | TC5 fence (before/after operations) | sm_100+ (**Blackwell**) | compute-bound/tensor-core, synchronization-bound/barrier-optimization | core |
| tcgen05.commit | Commit outstanding TC5 async operations | sm_100+ (**Blackwell**) | compute-bound/tensor-core | core |
| tcgen05.wait | Wait for TC5 operations to complete | sm_100+ (**Blackwell**) | compute-bound/tensor-core | core |

Supported types (tcgen05.mma):
- kind::f16: .f16/.bf16 -> .f16/.f32 accumulator
- kind::tf32: .tf32 -> .f32 accumulator
- kind::f8f6f4: .e4m3/.e5m2/.e3m2/.e2m3/.e2m1 -> .f32 accumulator
- kind::i8: .u8/.s8 -> .s32 accumulator
- kind::mxf8f6f4: microscaled FP8/FP6/FP4 (with scale factors)

Key feature: dedicated Tensor Memory (512 cols x 128 lanes per CTA, 32-bit cells).

---

## 12. Texture and Surface Instructions

| Instruction | Description | SM | Knowledge Node | Relevance |
|---|---|---|---|---|
| tex.{1d/2d/3d/a1d/a2d/cube/acube}[.level][.type] | Texture fetch with various addressing modes | all | memory-bound/cache-load-hints | low-relevance |
| tld4.{comp}.2d.{type} | Texture load 4 (gather single component from 4 texels) | sm_20+ | memory-bound/cache-load-hints | low-relevance |
| txq.{query}.b32 | Query texture attributes (width, height, etc.) | all | memory-bound/cache-load-hints | low-relevance |
| suld.b.{dim}.{vec}.{type} | Surface load | sm_20+ | memory-bound/cache-load-hints | low-relevance |
| sust.b.{dim}.{vec}.{type} | Surface store | sm_20+ | memory-bound/cache-load-hints | low-relevance |
| sured.b.{dim}.{op}.{type} | Surface reduction | sm_20+ | synchronization-bound/atomic-reduction | low-relevance |
| suq.{query}.b32 | Query surface attributes | sm_20+ | memory-bound/cache-load-hints | low-relevance |

---

## 13. Video Instructions

| Instruction | Description | SM | Knowledge Node | Relevance |
|---|---|---|---|---|
| vadd/vsub/vmad/vmin/vmax | Scalar video instructions (byte/half-word extract + arithmetic) | sm_20+ | compute-bound/fast-math | low-relevance |
| vadd2/vadd4/vsub2/vsub4 | SIMD 2-way/4-way video add/sub | sm_30+ | compute-bound/fast-math | low-relevance |
| vabsdiff/vabsdiff2/vabsdiff4 | SIMD absolute difference | sm_20+ | compute-bound/fast-math | low-relevance |
| vset/vset2/vset4 | SIMD video comparison | sm_20+ | compute-bound/fast-math | low-relevance |

---

## 14. Miscellaneous Instructions

| Instruction | Description | SM | Knowledge Node | Relevance |
|---|---|---|---|---|
| setmaxnreg.{inc/dec}.sync.aligned.u32 | Dynamically adjust per-warp register count (warpgroup-level) | sm_90a+ (Hopper) | memory-bound/register-pressure, latency-bound/occupancy-tuning | core |
| nanosleep.u32 | Suspend thread for ~N nanoseconds (backoff/polling) | sm_70+ | latency-bound/occupancy-tuning | related |
| brkpt | Breakpoint (debugging) | sm_11+ | (debugging) | low-relevance |
| trap | Abort execution, signal host | all | (debugging) | low-relevance |
| pmevent | Trigger performance monitor event | all | (profiling) | low-relevance |

---

## 15. Stack Manipulation Instructions

| Instruction | Description | SM | Knowledge Node | Relevance |
|---|---|---|---|---|
| stacksave.type | Save stack pointer to register | sm_52+ | latency-bound/dynamic-parallelism | low-relevance |
| stackrestore.type | Restore stack pointer from register | sm_52+ | latency-bound/dynamic-parallelism | low-relevance |
| alloca.type | Dynamically allocate stack memory | sm_52+ | latency-bound/dynamic-parallelism | low-relevance |

---

## Summary

### Instruction Count by Category

| Category | Total Families | Core | Related | Low-Relevance |
|---|---|---|---|---|
| Data Movement & Conversion | 42 | 28 | 12 | 2 |
| Integer Arithmetic | 30 | 2 | 9 | 19 |
| Float Arithmetic (f32/f64) | 22 | 11 | 8 | 3 |
| Half/BF16 Precision | 14 | 9 | 4 | 1 |
| Mixed Precision FP | 3 | 1 | 2 | 0 |
| Comparison & Selection | 6 | 0 | 4 | 2 |
| Logic & Shift | 9 | 0 | 1 | 8 |
| Control Flow | 6 | 1 | 2 | 3 |
| Synchronization & Communication | 34 | 22 | 9 | 3 |
| MMA (wmma/mma) | 5 | 5 | 0 | 0 |
| WGMMA | 5 | 5 | 0 | 0 |
| TensorCore 5th Gen (tcgen05) | 11 | 9 | 1 | 1 |
| Texture & Surface | 7 | 0 | 0 | 7 |
| Video | 4 | 0 | 0 | 4 |
| Miscellaneous | 5 | 1 | 1 | 3 |
| Stack Manipulation | 3 | 0 | 0 | 3 |
| **Total** | **206** | **94** | **53** | **59** |

### Architecture Generation Summary

| Generation | SM | Key Instruction Additions |
|---|---|---|
| Volta | sm_70 | mma (tensor core), shfl.sync, fence, independent thread scheduling |
| Turing | sm_75 | mma int8/int4/b1, tanh.approx.f32 |
| Ampere | sm_80 | mma bf16/tf32/f64, cp.async, mbarrier, redux.sync, fma.relu.f16/bf16 |
| Ada | sm_89 | mma FP8 (e4m3/e5m2) |
| Hopper | sm_90 | wgmma, cp.async.bulk.tensor (TMA), barrier.cluster, griddepcontrol, setmaxnreg, bf16 arithmetic, .cluster scope, st.async, red.async |
| Blackwell | sm_100 | tcgen05 (5th-gen TC with Tensor Memory), f32x2 SIMD, mixed-precision fma, 3-input min/max, mma e3m2/e2m3/e2m1, clusterlaunchcontrol |

### Top Knowledge Node Coverage

| Knowledge Node | Core Instructions |
|---|---|
| **compute-bound/tensor-core** | mma, wmma, wgmma, tcgen05, dp4a |
| **memory-bound/data-prefetch** | cp.async, cp.async.bulk, cp.async.bulk.tensor, prefetch, mbarrier.arrive.expect_tx |
| **memory-bound/vectorized-access** | ld.global.v2/v4, st.global.v2/v4, atom.add.vec.f32 |
| **memory-bound/shared-memory-cache** | ld.shared, st.shared, cp.async.*.shared, ld/st.shared::cluster |
| **synchronization-bound/barrier-optimization** | bar.sync, barrier.cluster, mbarrier.*, wgmma.fence/commit/wait |
| **synchronization-bound/atomic-reduction** | atom.*, red.*, redux.sync, cp.reduce.async.bulk |
| **compute-bound/fast-math** | div.approx, rcp.approx, rsqrt.approx, sin/cos/lg2/ex2.approx, fma |
| **compute-bound/half-precision-math** | f16/bf16 add/sub/mul/fma, tanh.approx.f16, cvt f16/bf16 |
| **compute-bound/warp-primitives** | shfl.sync, vote.sync, match.sync, redux.sync, bar.warp.sync, elect.sync |
| **memory-bound/l2-cache-control** | eviction hints, createpolicy, .L2::cache_hint, cp.async.bulk.prefetch |
| **latency-bound/warp-divergence** | @p predication, bra.uni, vote.sync, setp |
| **compute-bound/operator-fusion** | fma (all types), mad, fma.relu, fma.oob |
| **latency-bound/programmatic-dependent-launch** | griddepcontrol.launch_dependents, griddepcontrol.wait |
| **memory-bound/register-pressure** | setmaxnreg |
