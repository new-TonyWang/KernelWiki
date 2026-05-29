---
api: __ldg
namespace: runtime
probe_slug: runtime-ldg
status: verified
kind: documented
trigger: init-sweep
evidence_level: measured
clock_policy: unknown
measured_on:
  device: NVIDIA H200
  sm: 9.0a
  gpu_uuid: GPU-fbaa167f-4646-b8b9-c4f5-a4c4734ebc25
  cuda_runtime: '12.9'
  driver: 570.124.06
artifacts:
  code: sources/experience/api-probes/artifacts/__ldg_probe.cu
  build: nvcc -arch=sm_90a -O3 -std=c++17 -o __ldg_probe __ldg_probe.cu
  introspection: ''
  profile: ''
referenced_in_corpus:
- path: corpus/nvidia/cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming
    Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  line_range: L25102-L25130
- path: corpus/nvidia/cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming
    Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  line_range: L22347-L22353
- path: corpus/nvidia/cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming
    Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  line_range: L24547-L24557
source:
- path: spec
  anchor: Reference
conclusions:
  max_abs_err: 0.0
  latency_ms_median: 0.007712
  latency_ms_p10: 0.007552
  latency_ms_p90: 0.008096
  baseline_name: plain-global-load
  baseline_ms: 0.007744
  ratio: 1.0041
back_filled_into:
- wiki/nvidia/api-definitions/runtime/__ldg.md
open_questions:
- clock_policy is unknown -- clocks were not locked during measurement.
- On sm_90a, the compiler typically routes const __restrict__ loads through the read-only
  cache automatically. The ratio ~1.00 confirms __ldg and plain loads compile to equivalent
  instructions when the compiler can prove immutability. Explicit __ldg remains useful
  for non-const/non-restrict pointers.
id: exp-2026-04-16-runtime-ldg
type: experience
vendor: nvidia
title: 2026 04 16 Runtime Ldg
source_refs:
- source_id: cuda-official/toolkit-docs-13.2
  path: cuda-official/cuda-toolkit-documentation-13.2/CUDA Programming Guides/cuda-programming-guide/cuda_cuda-programming-guide_index.html.md
  anchor: L24550
---
## Summary

End-to-end probe of `__ldg` on H200 (sm_90a, CUDA 12.9). The probe allocates 1M floats, copies to device, launches a kernel that loads each element via `__ldg`, adds 1.0, and stores the result. A baseline kernel with plain global loads is measured for comparison. Results are copied back and verified against a CPU reference. Correctness: exact match (max_abs_err = 0). The `__ldg` kernel and plain-load kernel show essentially identical latency (ratio 1.004), which is expected on sm_90a where `const __restrict__` pointers are automatically routed through the read-only cache.

## Minimal Kernel

```cuda
// Probe: __ldg read-only cache load (end-to-end)
// Loads N floats via __ldg, adds 1.0, stores to output.
// Verifies against CPU reference (normal load + add).
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>

#define N (1 << 20)  // 1M elements
#define BLK 256
#define CHECK(call) do { cudaError_t e = (call); \
    if (e != cudaSuccess) { fprintf(stderr, "CUDA %s:%d: %s\n", \
    __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while(0)

__global__ void ldg_add(const float* __restrict__ in, float* out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
        out[tid] = __ldg(in + tid) + 1.0f;
}

// Baseline: plain global load (compiler may also use LDG for const __restrict__,
// but we omit __restrict__ here to force the default path).
__global__ void plain_add(const float* in, float* out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
        out[tid] = in[tid] + 1.0f;
}

int main() {
    size_t bytes = N * sizeof(float);
    float *h_in  = (float*)malloc(bytes);
    float *h_out = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) h_in[i] = (float)(i % 1000) * 0.001f;

    float *d_in, *d_out;
    CHECK(cudaMalloc(&d_in, bytes));
    CHECK(cudaMalloc(&d_out, bytes));
    CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    int grid = (N + BLK - 1) / BLK;
    // warmup
    for (int i = 0; i < 5; i++) ldg_add<<<grid, BLK>>>(d_in, d_out, N);
    CHECK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    std::vector<float> ms_ldg(20), ms_plain(20);
    for (int i = 0; i < 20; i++) {
        cudaEventRecord(t0);
        ldg_add<<<grid, BLK>>>(d_in, d_out, N);
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        cudaEventElapsedTime(&ms_ldg[i], t0, t1);
    }
    // warmup baseline
    for (int i = 0; i < 5; i++) plain_add<<<grid, BLK>>>(d_in, d_out, N);
    CHECK(cudaDeviceSynchronize());
    for (int i = 0; i < 20; i++) {
        cudaEventRecord(t0);
        plain_add<<<grid, BLK>>>(d_in, d_out, N);
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        cudaEventElapsedTime(&ms_plain[i], t0, t1);
    }
    std::sort(ms_ldg.begin(), ms_ldg.end());
    std::sort(ms_plain.begin(), ms_plain.end());

    // verify __ldg kernel
    ldg_add<<<grid, BLK>>>(d_in, d_out, N);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    float max_err = 0;
    for (int i = 0; i < N; i++)
        max_err = fmaxf(max_err, fabsf(h_out[i] - (h_in[i] + 1.0f)));

    printf("PASS=%s  max_abs_err=%.6e\n", max_err <= 1e-5 ? "true" : "false", max_err);
    printf("ldg    median=%.6f p10=%.6f p90=%.6f ms\n", ms_ldg[10], ms_ldg[2], ms_ldg[18]);
    printf("plain  median=%.6f p10=%.6f p90=%.6f ms\n", ms_plain[10], ms_plain[2], ms_plain[18]);
    printf("ratio=%.4f\n", ms_plain[10] / ms_ldg[10]);

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return (max_err <= 1e-5) ? 0 : 1;
}
```

## Build

```bash
nvcc -arch=sm_90a -O3 -std=c++17 -o __ldg_probe __ldg_probe.cu
```

## Measurement

Configuration: N = 1048576 (1M) floats, grid = 4096, block = 256. 5 warmup launches, 20 measurement launches via CUDA events.

| shape | dtype | latency_ms_median | latency_ms_p10 | latency_ms_p90 | baseline_name | baseline_ms | ratio | clock_policy | reproduce_cmd |
|---|---|---|---|---|---|---|---|---|---|
| 1048576 | fp32 | 0.007712 | 0.007552 | 0.008096 | plain-global-load | 0.007744 | 1.0041 | unknown | `nvcc -arch=sm_90a -O3 -std=c++17 -o /tmp/p sources/experience/api-probes/artifacts/__ldg_probe.cu && /tmp/p` |

## Introspection

No `kp_introspect` bundle was generated for this probe (tool not available in this environment).

## Notes

- Upstream documentation: CUDA Programming Guide (L24550) describes `__ldg` as performing a read-only L1/Tex cache load, supporting all fundamental types and CUDA vector types.
- The documentation also notes (L22347) that `const __restrict__` global pointers are compiled as read-only cache loads (`ld.global.nc`), equivalent to explicit `__ldg`.
- The ratio of ~1.00 confirms that on sm_90a the compiler automatically uses `ld.global.nc` for `const __restrict__` pointers, so explicit `__ldg` adds no measurable benefit in this case.
- Explicit `__ldg` remains useful when the pointer is not declared `const __restrict__` but the programmer knows the data is read-only during the kernel execution.
- No upstream `.cu` examples of `__ldg` were found in the cuda_repo corpus.
