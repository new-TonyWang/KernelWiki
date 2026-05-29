---
title: Vectorized Memory Access - Pitfalls
status: draft
evidence_level: spec
applies_to_pattern_class:
- cuda-core
applies_to_ops:
- elementwise
- reduction
- normalization
- scan
requires_sm: '>=6.0'
single_kernel_useful: true
source:
- path: spec
  anchor: Reference
id: pitfall-vectorized-access
type: pitfall
vendor: nvidia
source_refs:
- source_id: cuda-official/toolkit-docs-13.2
  path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L22574-L22702
- source_id: cuda-official/toolkit-docs-13.2
  path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-c-best-practices-guide/cuda_cuda-c-best-practices-guide_index.html.md
  anchor: L1062
- source_id: cuda-official/toolkit-docs-13.2
  path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L22735-L22753
---
## P1: Misaligned pointer for vector load

**Symptom**: Runtime crash (misaligned address fault) or silent corruption when loading float4 from a pointer that is not 16-byte aligned.

**Cause**: `float4` has a 16-byte alignment requirement (programming guide Table 42, L22578). If `reinterpret_cast<const float4*>(ptr)` is used on a pointer `ptr` that is not 16-byte aligned, the resulting load triggers an alignment fault. This can happen when:
- The base pointer has an offset applied (e.g., `ptr + 1` shifts by 4 bytes, breaking 16-byte alignment).
- The data starts at a struct member with insufficient alignment.
- A sub-array slice begins at a non-aligned offset.

**Example**:
```cuda
// BAD: ptr+1 is 4-byte aligned, not 16-byte aligned
float* ptr = d_array;  // 256-byte aligned from cudaMalloc
float4 v = reinterpret_cast<const float4*>(ptr + 1)[tid];  // CRASH

// GOOD: ensure offset is a multiple of 4 floats (16 bytes)
float4 v = reinterpret_cast<const float4*>(ptr)[tid];  // OK
// or with aligned offset:
float4 v = reinterpret_cast<const float4*>(ptr + 4 * k)[tid];  // OK if k is integer
```

**Fix**: Ensure the pointer passed to `reinterpret_cast<float4*>()` is 16-byte aligned. Base pointers from `cudaMalloc` are 256-byte aligned, so they are safe. Any offset from the base must be a multiple of 4 elements (for float) or 16 bytes in general. If alignment cannot be guaranteed, fall back to scalar loads for the misaligned portion.

**Detection**: The kernel will abort with `misaligned address` if CUDA error checking is enabled. In debug mode (`-G` flag), the CUDA runtime detects the fault immediately.

## P2: Out-of-bounds read when N is not a multiple of 4

**Symptom**: Reading beyond the allocated buffer, causing incorrect results or memory access violations.

**Cause**: When using `float4` loads with `n` total floats and `n` is not divisible by 4, the last `float4` load at index `n/4` reads `n/4 * 4` through `n/4 * 4 + 3`, which may extend past the end of the allocation.

**Example**:
```cuda
// BAD: n=1025, n/4=256, thread 256 reads bytes 4096-4111 but only 4100 allocated
int tid = blockIdx.x * blockDim.x + threadIdx.x;
if (tid < (n + 3) / 4) {  // This allows index 256 which overflows
    float4 v = reinterpret_cast<const float4*>(input)[tid];
}
```

**Fix**: Two approaches:
1. **Pad the allocation**: allocate `((n + 3) / 4) * 4` floats and zero-pad the tail. This is the simplest and most efficient approach.
2. **Separate tail loop**: process the bulk with float4 loads and handle the remaining `n % 4` elements with scalar loads:
```cuda
int n4 = n / 4;
if (tid < n4) {
    float4 v = reinterpret_cast<const float4*>(input)[tid];
    // ... process v
}
// Tail: only first (n % 4) threads handle remainder
int tail_start = n4 * 4;
if (tid < (n - tail_start)) {
    float s = input[tail_start + tid];
    // ... process s
}
```

## P3: Incorrect reinterpret_cast with non-float types

**Symptom**: Garbage values or type-aliasing bugs when vector-loading data of a different base type.

**Cause**: `reinterpret_cast<const float4*>(half_ptr)` will load 16 bytes (4 floats = 128 bits), but if the data is actually `__half` (2 bytes each), those 16 bytes correspond to 8 half-precision values, not 4 floats. The values will be misinterpreted.

**Example**:
```cuda
// BAD: half data reinterpreted as float4
__half* h_ptr = ...;
float4 v = reinterpret_cast<const float4*>(h_ptr)[tid];  // Wrong types!

// GOOD: use the correct vector width for the data type
// For __half, use __half2 (loads 2 halfs = 4 bytes) or load via uint4
// and manually unpack:
uint4 raw = reinterpret_cast<const uint4*>(h_ptr)[tid];  // 16 bytes = 8 halfs
__half2* h2 = reinterpret_cast<__half2*>(&raw);  // 4 half2 values
```

**Fix**: Match the vector type to the underlying data type. For float data, use `float4`. For half-precision, use `__half2` for 2-element loads or load raw bytes via `uint4` and manually unpack. For `int` data, use `int4`. The general rule: the vector type's base must match the data element type.

## P4: Increased register pressure from vector loads

**Symptom**: Kernel occupancy drops after switching to vectorized loads, negating the bandwidth benefit.

**Cause**: Each `float4` variable occupies 4 registers (one per component: `.x`, `.y`, `.z`, `.w`). If a kernel loads multiple `float4` values simultaneously (e.g., two inputs for a fused operation), that is 8 registers just for the loaded data. The best-practices guide (L1062) notes that "there is no register-related reason to pack data into vector data types such as `float4`" -- the register cost is the same as 4 separate float registers. However, if the kernel was already near the register limit, the additional temporary registers needed during the unpack/compute/repack sequence may push it over the threshold.

**Example**: A kernel using `__launch_bounds__(256, 4)` (256 threads/block, 4 blocks/SM, requiring <= 64 registers/thread on H200) may exceed 64 regs when switching from scalar to float4 loads if the compute phase is complex.

**Fix**: Check register usage with `--ptxas-options=-v` after adding vectorized loads. If register count increases significantly:
- Reduce the vector width from float4 to float2.
- Use `__launch_bounds__` to constrain registers.
- Manually schedule the unpack/compute sequence to reduce live registers.

**Detection**: Compile with `nvcc --ptxas-options=-v` and compare register count before and after vectorization.

## P5: Assuming vectorized loads help compute-bound kernels

**Symptom**: No performance improvement (or even slight regression) after adding vectorized loads.

**Cause**: Vectorized loads reduce the number of memory instructions, which only helps when the kernel is memory-instruction-bound or memory-bandwidth- bound. If the kernel is compute-bound (the arithmetic pipeline is the bottleneck, not the memory subsystem), reducing load instructions has no effect on overall throughput.

**Fix**: Before vectorizing, determine the kernel's bottleneck:
- If `ncu` shows memory throughput close to peak and compute throughput well below peak, the kernel is memory-bound -- vectorize.
- If compute throughput is close to peak and memory throughput is low, the kernel is compute-bound -- vectorization will not help.
- Without `ncu`, use a rough estimate: if the kernel does less than ~10 FLOPs per byte loaded, it is likely memory-bound.

## P6: Using float3 / int3 for "vectorized" loads

**Symptom**: No performance improvement or even slower execution when using 3-component vector types.

**Cause**: `float3` has size 12 bytes and alignment 4 bytes (programming guide Table 42, L22695-L22698). Unlike `float4` (16 bytes, alignment 16) and `float2` (8 bytes, alignment 8), `float3` does not map to a single wide load instruction. The compiler typically generates three separate 32-bit loads for a `float3`. The odd size also means that consecutive `float3` values are not naturally aligned for wide loads.

**Fix**: Use `float4` with an ignored `.w` component, or use `float2` plus a scalar load. Never use `float3` / `int3` for performance-critical memory access.

## P7: Forgetting to vectorize the store side

**Symptom**: Kernel shows only partial improvement after vectorizing loads.

**Cause**: Vectorized loads halve or quarter the load instruction count, but if the stores remain scalar, the store side still issues 4x the instructions. In memory-bound kernels where the write bandwidth is significant (e.g., elementwise ops with 1:1 read:write ratio), un-vectorized stores leave performance on the table.

**Fix**: Apply the same `float4` pattern to outputs:
```cuda
float4 vc;
vc.x = va.x + vb.x;
vc.y = va.y + vb.y;
vc.z = va.z + vb.z;
vc.w = va.w + vb.w;
reinterpret_cast<float4*>(c)[tid] = vc;  // STG.128 -- vectorized store
```

## P8: Breaking coalescing with strided vector access

**Symptom**: Vectorized loads show no improvement or performance regression compared to scalar loads.

**Cause**: If consecutive threads do not access consecutive `float4` elements -- for example, `((float4*)ptr)[tid * 2]` with stride 2 between threads -- the loads are not coalesced. Each `float4` access is 16 bytes wide, so a stride-2 pattern means the warp touches addresses 32 bytes apart per thread, wasting half the fetched data. Vectorized access must be used together with coalesced access patterns to achieve full benefit.

**Fix**: Ensure the access index is `tid` (linear, stride-1 across threads within a warp). The vectorization-plus-coalescing pattern is: `reinterpret_cast<const float4*>(ptr)[blockIdx.x * blockDim.x + threadIdx.x]`.
