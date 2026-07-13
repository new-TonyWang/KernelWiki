// __pipeline_memcpy_async_probe.cu
// Probe for the LDGSTS primitives API trio:
//   __pipeline_memcpy_async(dst, src, size)
//   __pipeline_commit()
//   __pipeline_wait_prior(N)
//
// Target: H200 (sm_90a), CUDA 12.9. These three primitives only do useful
// work when composed: memcpy_async issues async global-to-shared copies,
// commit groups the preceding issues into one pipeline stage, and
// wait_prior(N) blocks until at most N stages are still pending.
//
// This probe validates three behaviours:
//   (a) Single-stage: 16-byte-per-thread L1-bypass copy via float4,
//       commit once, wait_prior(0), verify data in shared memory.
//   (b) Multi-stage ordering: issue 3 commit stages back-to-back, then
//       wait_prior(2), wait_prior(1), wait_prior(0) and verify that at
//       each step the corresponding oldest stage is complete.
//   (c) Correctness of the primitive vs. a CPU reference on an
//       elementwise pass-through kernel (no compute beyond the copy).
//
// Build:
//   nvcc -arch=sm_90a -O3 -std=c++17 -o /tmp/pipeline_probe \
//        knowledge/80-experience/api-probes/artifacts/__pipeline_memcpy_async_probe.cu
// Run:
//   /tmp/pipeline_probe

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>

#define CHECK(call) do {                                                \
    cudaError_t e = (call);                                             \
    if (e != cudaSuccess) {                                             \
        fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__,         \
                cudaGetErrorString(e));                                  \
        exit(1);                                                        \
    }                                                                   \
} while (0)

constexpr int BLOCK_DIM = 256;
constexpr int VEC = 4;   // float4 => 16B per thread => L1 BYPASS eligible

// --------- Kernel (a+c): single-stage pass-through using float4 copy ---------
// Each block copies BLOCK_DIM float4 elements (= BLOCK_DIM*16 bytes) from
// global to shared via __pipeline_memcpy_async, commits once, waits for 0
// pending stages (i.e. wait for everything), then writes back to global.
__global__ void kernel_single_stage(const float4* __restrict__ in,
                                    float4* __restrict__ out,
                                    int n_vec) {
    __shared__ __align__(16) float4 smem[BLOCK_DIM];
    const int tid = threadIdx.x;
    const int g = blockIdx.x * BLOCK_DIM + tid;

    if (g < n_vec) {
        // 16-byte async copy, L1 BYPASS path (dst and src both 16B aligned).
        __pipeline_memcpy_async(&smem[tid], &in[g], sizeof(float4));
    }
    // Commit the pending copies as one stage.
    __pipeline_commit();
    // Wait until 0 stages remain pending (i.e. this single stage completes).
    __pipeline_wait_prior(0);
    // Block-wide sync: other threads may also have issued copies that we
    // are about to read, and __pipeline_wait_prior is thread-scoped.
    __syncthreads();

    if (g < n_vec) {
        out[g] = smem[tid];
    }
}

// --------- Kernel (b): multi-stage ordering probe ---------
// Issues three stages and then drains them one by one with
// wait_prior(2), wait_prior(1), wait_prior(0). After each wait, the oldest
// stage's data must be visible in shared memory; we record a checksum of
// each stage after its corresponding wait into `stage_check[3]`.
//
// Each thread uses one float element per stage (L1 ACCESS path, 4B).
__global__ void kernel_multi_stage(const float* __restrict__ in,
                                   unsigned int* __restrict__ stage_check,
                                   int n) {
    __shared__ float smem[3][BLOCK_DIM];
    const int tid = threadIdx.x;
    const int base = blockIdx.x * BLOCK_DIM;

    // Issue three commit-stages, each copying BLOCK_DIM floats from a
    // distinct slab of `in` into smem[s].
    for (int s = 0; s < 3; ++s) {
        int g = base + s * BLOCK_DIM + tid;
        if (g < n) {
            __pipeline_memcpy_async(&smem[s][tid], &in[g], sizeof(float));
        }
        __pipeline_commit();
    }

    // Drain in FIFO order. After wait_prior(N), at most N of the issued
    // commit stages remain pending; the others (oldest first) are complete.
    unsigned int local_sum = 0;

    __pipeline_wait_prior(2);              // stage 0 must now be done
    __syncthreads();
    local_sum = __float_as_uint(smem[0][tid]);
    atomicAdd(&stage_check[0], local_sum);

    __pipeline_wait_prior(1);              // stage 1 must now also be done
    __syncthreads();
    local_sum = __float_as_uint(smem[1][tid]);
    atomicAdd(&stage_check[1], local_sum);

    __pipeline_wait_prior(0);              // stage 2 must now be done
    __syncthreads();
    local_sum = __float_as_uint(smem[2][tid]);
    atomicAdd(&stage_check[2], local_sum);
}

int main() {
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s  SM: %d.%d  CUDA runtime: %d.%d\n",
           prop.name, prop.major, prop.minor,
           CUDART_VERSION / 1000, (CUDART_VERSION % 1000) / 10);
    int driver_version = 0;
    CHECK(cudaDriverGetVersion(&driver_version));
    printf("Driver: %d.%d\n", driver_version / 1000, (driver_version % 1000) / 10);

    // -------------- (a+c) single-stage correctness + timing --------------
    constexpr int N_VEC = 1 << 20;          // 1M float4 = 16 MiB
    constexpr int N     = N_VEC * VEC;      // 4M floats
    const size_t bytes4 = (size_t)N_VEC * sizeof(float4);

    std::vector<float> h_in(N), h_ref(N), h_out(N);
    for (int i = 0; i < N; ++i) {
        h_in[i]  = 1.0f + (i % 4096) * 0.00025f;
        h_ref[i] = h_in[i];                 // pass-through
    }

    float4 *d_in4, *d_out4;
    CHECK(cudaMalloc(&d_in4, bytes4));
    CHECK(cudaMalloc(&d_out4, bytes4));
    CHECK(cudaMemcpy(d_in4, h_in.data(), bytes4, cudaMemcpyHostToDevice));

    int blocks = (N_VEC + BLOCK_DIM - 1) / BLOCK_DIM;

    // Correctness
    CHECK(cudaMemset(d_out4, 0, bytes4));
    kernel_single_stage<<<blocks, BLOCK_DIM>>>(d_in4, d_out4, N_VEC);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h_out.data(), d_out4, bytes4, cudaMemcpyDeviceToHost));

    float max_err = 0.0f;
    for (int i = 0; i < N; ++i) {
        float e = fabsf(h_out[i] - h_ref[i]);
        if (e > max_err) max_err = e;
    }
    const bool single_ok = (max_err <= 1e-5f);
    printf("[single-stage] N_VEC=%d (%d floats)  max_abs_err=%.3e  %s\n",
           N_VEC, N, max_err, single_ok ? "PASS" : "FAIL");

    // Warmup + 20 measurements
    for (int i = 0; i < 5; ++i)
        kernel_single_stage<<<blocks, BLOCK_DIM>>>(d_in4, d_out4, N_VEC);
    CHECK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    std::vector<float> ms(20);
    for (int i = 0; i < 20; ++i) {
        cudaEventRecord(t0);
        kernel_single_stage<<<blocks, BLOCK_DIM>>>(d_in4, d_out4, N_VEC);
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        cudaEventElapsedTime(&ms[i], t0, t1);
    }
    std::sort(ms.begin(), ms.end());
    printf("[single-stage] median=%.6f p10=%.6f p90=%.6f ms\n",
           ms[10], ms[2], ms[18]);

    // -------------- (b) multi-stage ordering --------------
    constexpr int M_BLOCKS = 4;
    constexpr int M_N = 3 * BLOCK_DIM * M_BLOCKS;     // 3 stages per block
    std::vector<float> h_min(M_N);
    for (int i = 0; i < M_N; ++i) {
        h_min[i] = 0.5f + i * 0.125f;
    }
    float *d_min;
    unsigned int *d_check;
    CHECK(cudaMalloc(&d_min, M_N * sizeof(float)));
    CHECK(cudaMalloc(&d_check, 3 * sizeof(unsigned int)));
    CHECK(cudaMemcpy(d_min, h_min.data(), M_N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_check, 0, 3 * sizeof(unsigned int)));

    kernel_multi_stage<<<M_BLOCKS, BLOCK_DIM>>>(d_min, d_check, M_N);
    CHECK(cudaDeviceSynchronize());

    unsigned int h_check[3];
    CHECK(cudaMemcpy(h_check, d_check, 3 * sizeof(unsigned int),
                     cudaMemcpyDeviceToHost));

    // CPU reference: compute the same atomic-sum of float-bit-patterns.
    unsigned int ref_check[3] = {0, 0, 0};
    for (int b = 0; b < M_BLOCKS; ++b) {
        int base = b * BLOCK_DIM;
        for (int s = 0; s < 3; ++s) {
            for (int t = 0; t < BLOCK_DIM; ++t) {
                int g = base + s * BLOCK_DIM + t;
                if (g < M_N) {
                    unsigned int u;
                    std::memcpy(&u, &h_min[g], sizeof(u));
                    ref_check[s] += u;
                }
            }
        }
    }

    bool multi_ok = true;
    for (int s = 0; s < 3; ++s) {
        if (h_check[s] != ref_check[s]) multi_ok = false;
        printf("[multi-stage] stage %d  gpu=0x%08x  cpu=0x%08x  %s\n",
               s, h_check[s], ref_check[s],
               (h_check[s] == ref_check[s]) ? "ok" : "MISMATCH");
    }
    printf("[multi-stage] %s\n", multi_ok ? "PASS" : "FAIL");

    printf("\nSUMMARY: single=%s multi=%s\n",
           single_ok ? "PASS" : "FAIL", multi_ok ? "PASS" : "FAIL");

    CHECK(cudaFree(d_in4));
    CHECK(cudaFree(d_out4));
    CHECK(cudaFree(d_min));
    CHECK(cudaFree(d_check));
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return (single_ok && multi_ok) ? 0 : 1;
}
