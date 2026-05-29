---
api: cudaFuncSetAttribute
namespace: runtime
probe_slug: runtime-cuda-func-set-attribute
status: verified
kind: documented
trigger: init-sweep
evidence_level: measured
clock_policy: unknown
measured_on:
  device: NVIDIA H200
  sm: 9.0a
  gpu_uuid: GPU-fbaa167f4646b8b9c4f5a4c4734ebc25
  cuda_runtime: '12.9'
  driver: 570.124.06
artifacts:
  code: sources/experience/api-probes/artifacts/cudaFuncSetAttribute_probe.cu
  build: nvcc -arch=sm_90a -O3 -std=c++17 -lineinfo -o cudaFuncSetAttribute_probe
    cudaFuncSetAttribute_probe.cu
  introspection: ''
  profile: ''
referenced_in_corpus:
- path: corpus/nvidia/cuda-official/cuda-toolkit-documentation-13.2/CUDA API References/cuda-runtime-api/cuda_cuda-runtime-api_index.html.md
  line_range: L3372-L3440
- path: corpus/nvidia/cuda-official/cuda-toolkit-documentation-13.2/CUDA API References/cuda-runtime-api/cuda_cuda-runtime-api_index.html.md
  line_range: L15838-L15870
- path: corpus/nvidia/source-code/cuda-samples/Samples/3_CUDA_Features/bf16TensorCoreGemm/bf16TensorCoreGemm.cu
  line_range: L774-L785
source:
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA API References/cuda-runtime-api/cuda_cuda-runtime-api_index.html.md
  anchor: L3375-L3375
  excerpt: __host__ cudaError_t cudaFuncSetAttribute ( const void* func, cudaFuncAttribute
    attr, int value )
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA API References/cuda-runtime-api/cuda_cuda-runtime-api_index.html.md
  anchor: L3395-L3398
  excerpt: cudaFuncAttributeMaxDynamicSharedMemorySize - The requested maximum size
    in bytes of dynamically-allocated shared memory. The sum of this value and the
    function attribute sharedSizeBytes cannot exceed the device attribute cudaDevAttrMaxSharedMemoryPerBlockOptin.
- path: cuda-official/cuda-toolkit-documentation-13.2/CUDA API References/cuda-runtime-api/cuda_cuda-runtime-api_index.html.md
  anchor: L15841-L15841
  excerpt: template < class T > __host__ cudaError_t cudaFuncSetAttribute ( T* func,
    cudaFuncAttribute attr, int value ) [inline]
conclusions:
  max_abs_err: 0.0
  latency_ms_median: 0.006624
  latency_ms_p10: 0.006432
  latency_ms_p90: 0.006976
  baseline_name: pre-optin-launch (cudaErrorInvalidValue)
  baseline_ms: null
  ratio: null
back_filled_into:
- wiki/nvidia/api-definitions/runtime/cudaFuncSetAttribute.md
open_questions:
- clock_policy is unknown — GPU clocks were not locked during the post-optin kernel
  launch; the kernel is short enough (6.6 µs median) that free-running clocks could
  jitter it noticeably.
- Only the MaxDynamicSharedMemorySize attribute was exercised; PreferredSharedMemoryCarveout
  and the four cluster-related attributes (RequiredClusterWidth/Height/Depth, NonPortableClusterSizeAllowed,
  ClusterSchedulingPolicyPreference) were not probed.
- The probe does not measure the maximum achievable opt-in size. H200 sharedMemPerBlockOptin
  is 232 448 bytes; only 200 KB was tested. The 232 KB edge and over-the-limit (>232
  448) behavior are not characterized here.
id: exp-2026-04-17-runtime-cuda-func-set-attribute
type: experience
vendor: nvidia
title: 2026 04 17 Runtime Cuda Func Set Attribute
---
## Summary

End-to-end probe of `cudaFuncSetAttribute` on H200 (sm_90a, CUDA 12.9, driver 570.124.06). The probe targets the most common use case: opting into dynamic shared memory beyond the default 48 KB cap via `cudaFuncAttributeMaxDynamicSharedMemorySize`. Observed behavior in sequence: (1) launching the kernel with 200 KB dynamic shmem BEFORE any opt-in returns `cudaErrorInvalidValue` at launch, as expected (the default `sharedMemPerBlock` cap is 49 152 bytes); (2) one call to `cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 200*1024)` returns `cudaSuccess` (host-side call latency ~0.05 µs amortized over 1000 iters); (3) a follow-up `cudaFuncGetAttributes` shows `maxDynamicSharedSizeBytes = 204 800` confirming the attribute was applied; (4) the re-launch with `<<<1, 256, 204800>>>` succeeds and produces numerically correct output (`max_abs_err = 0`, tol = 1e-5).

## Minimal Kernel

```cuda
// Probe: cudaFuncSetAttribute opting into large dynamic shared memory.
//
// Signature:
//   __host__ cudaError_t cudaFuncSetAttribute(
//       const void* func, cudaFuncAttribute attr, int value);
//
// Flow:
//   1) Query initial cudaFuncAttributes.
//   2) Launch with dynSmem=200 KB → expect cudaErrorInvalidValue.
//   3) cudaFuncSetAttribute(kernel,
//        cudaFuncAttributeMaxDynamicSharedMemorySize, 200*1024).
//   4) Re-query → maxDynamicSharedSizeBytes should be 204 800.
//   5) Re-launch → expect success + correct output.
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { cudaError_t e = (call); \
    if (e != cudaSuccess) { fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, \
    __LINE__, cudaGetErrorString(e)); exit(1); } } while(0)

__global__ void big_smem_kernel(float* out, int n_floats_per_block) {
    extern __shared__ float smem[];
    int t = threadIdx.x;
    int blk = blockDim.x;
    for (int i = t; i < n_floats_per_block; i += blk) {
        smem[i] = (float)(i * (t + 1));
    }
    __syncthreads();
    if (t < n_floats_per_block) out[t] = smem[t];
}

// Host flow:
//   - cudaMalloc(&dOut, 256*sizeof(float));
//   - big_smem_kernel<<<1, 256, 200*1024>>>(dOut, 51200); // expect fail
//   - cudaFuncSetAttribute(big_smem_kernel,
//       cudaFuncAttributeMaxDynamicSharedMemorySize, 200*1024);
//   - big_smem_kernel<<<1, 256, 200*1024>>>(dOut, 51200); // now succeeds
//   - verify: out[t] == t*(t+1) for all t in [0,256).
// (Full source under artifacts/; this snippet preserves the core probe.)
```

## Build

```bash
nvcc -arch=sm_90a -O3 -std=c++17 -lineinfo \
     -o cudaFuncSetAttribute_probe \
     sources/experience/api-probes/artifacts/cudaFuncSetAttribute_probe.cu
```

## Measurement

Two separate measurements:

1. **Host-side API call latency** — `cudaFuncSetAttribute` itself, averaged over 1000 back-to-back calls with `std::chrono::high_resolution_clock`: **0.050 µs per call** (amortized). Single-shot latency is below std::chrono resolution, so only the amortized value is reported.

2. **Kernel latency after opt-in** — `<<<1, 256, 204800>>>` launched with CUDA events, 5 warmup (discarded) + 20 measured per `benchmark-protocol.md`:

| shape | dtype | latency_ms_median | latency_ms_p10 | latency_ms_p90 | baseline_name | baseline_ms | ratio | clock_policy | reproduce_cmd |
|---|---|---|---|---|---|---|---|---|---|
| 1 block × 256 threads × 200 KB dynSmem | fp32 | 0.006624 | 0.006432 | 0.006976 | pre-optin-launch (cudaErrorInvalidValue) | N/A | N/A | unknown | `nvcc -arch=sm_90a -O3 -std=c++17 -o /tmp/p sources/experience/api-probes/artifacts/cudaFuncSetAttribute_probe.cu && /tmp/p` |

The baseline here is semantic rather than numeric: the same launch config **fails** before `cudaFuncSetAttribute` is called (returning `cudaErrorInvalidValue` at launch time). This is the "before/after" baseline the API exists to serve.

### State transitions observed

| Step | cudaFuncAttributes.maxDynamicSharedSizeBytes | Launch <<<1,256,200KB>>> result |
|---|---|---|
| Initial                                                             | 49 152   | `cudaErrorInvalidValue` |
| After `cudaFuncSetAttribute(..., MaxDynamicSharedMemorySize, 200*1024)` | 204 800  | `cudaSuccess` |

### Device properties at probe time

| prop | value |
|---|---|
| `sharedMemPerBlock` (default cap)         | 49 152 bytes |
| `sharedMemPerBlockOptin` (opt-in ceiling) | 232 448 bytes |
| `sharedMemPerMultiprocessor`              | 233 472 bytes |

The opt-in value (200 KB = 204 800 bytes) is below the 232 448-byte ceiling, so the set succeeds. Attempting to set beyond `sharedMemPerBlockOptin` would return `cudaErrorInvalidValue` per the API reference (not exercised in this probe).

## Introspection

No `kp_introspect kernel-static` bundle was generated (cuda-python not available in the H200 shell environment). Device-static facts used by this probe come from `cudaGetDeviceProperties` at runtime.

## Notes

- **Signature**. Per the CUDA Runtime API reference (L3375): `__host__ cudaError_t cudaFuncSetAttribute(const void* func, cudaFuncAttribute attr, int value)`. A templated inline wrapper exists at L15841 accepting any `T* func`, which is just a convenience cast to `const void*`.

- **Typical attr values** (L3395-L3410):
  - `cudaFuncAttributeMaxDynamicSharedMemorySize` — the requested maximum size in bytes of dynamically-allocated shared memory. Sum with static `sharedSizeBytes` must not exceed `cudaDevAttrMaxSharedMemoryPerBlockOptin` (232 448 bytes on H200).
  - `cudaFuncAttributePreferredSharedMemoryCarveout` — hint, percent of total shared memory for L1 vs shmem split; driver may override.
  - `cudaFuncAttributeRequiredClusterWidth / Height / Depth` — Hopper cluster grid; setting at runtime returns `cudaErrorNotPermitted` if already set at compile time.
  - `cudaFuncAttributeNonPortableClusterSizeAllowed`, `cudaFuncAttributeClusterSchedulingPolicyPreference` — Hopper CGA tuning; not exercised here.

- **Return codes observed**:
  - `cudaSuccess` on the opt-in call with a valid 200 KB value.
  - `cudaErrorInvalidValue` on the pre-opt-in launch (the kernel-launch function call reports this via `cudaGetLastError()` — NOT `cudaFuncSetAttribute` itself).

- **Non-sticky error**. The pre-opt-in `cudaErrorInvalidValue` from the failed launch is returned by the launch-wrapper and cleared by the subsequent `cudaGetLastError()` call. Later `cudaFuncSetAttribute` and the post-opt-in launch proceed normally — consistent with the CUDA runtime's rule that "invalid argument" from a launch is non-sticky.

- **Host call latency**. Amortized ~0.05 µs per call on H200 / CUDA 12.9. This is cheap enough to call unconditionally at kernel-init time. Repeating the same set is idempotent (observed no state change across 1000 iters).

- **Kernel latency**. 6.6 µs median for a 1-block, 256-thread kernel that touches 50 200 floats of shared memory twice (one init loop + one read). This includes launch overhead; the actual shmem traffic is small relative to launch cost.

- **Correctness**. `out[t] = smem[t]` where `smem[i] = i*(t_writer+1)` and `t_writer = i % BLK`. For `i == t` the writer is the reader, so `out[t] == t*(t+1)` exactly. The probe observes `max_abs_err = 0` against this scalar reference for all t in [0, 256).

- **Reference use in CUDA Samples**. `bf16TensorCoreGemm.cu` (L774-L785) is a canonical example of this pattern: it checks the device `sharedMemPerBlockOptin`, then calls `cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ)` before its first launch. The `SHMEM_SZ` there is well over 48 KB (≈ 65 KB for the bf16 tensor-core tiles).

- **Not measured / out of scope**:
  - Over-the-limit behavior (value > `sharedMemPerBlockOptin`).
  - Other attr values (carveout, cluster attrs).
  - Interaction with `cudaLibraryGetKernel` / `cudaKernel_t`.
  - Persistence across contexts — the attribute is per-function in the current context; cross-context lifecycle is not probed.
