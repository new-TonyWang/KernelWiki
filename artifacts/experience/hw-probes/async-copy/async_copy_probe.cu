// async_copy_probe.cu -- compare vanilla elementwise vs 2-stage LDGSTS prefetch
// Target: H200 (sm_90a), CUDA 12.9
// Task: 2026-04-16-async-copy
//
// Vanilla kernel: c[i] = a[i] * b[i]  (grid-stride, no unrolling)
// 2-stage kernel: same computation, but uses __pipeline primitives with 2
//                 shared-memory stages to prefetch data from global memory.
//
// Build:
//   nvcc -arch=sm_90a -O3 -std=c++17 -lineinfo -o async_copy_probe async_copy_probe.cu
// Run:
//   ./async_copy_probe

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>

#define CHECK(call)                                                          \
  do {                                                                       \
    cudaError_t err = (call);                                                \
    if (err != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,       \
              cudaGetErrorString(err));                                       \
      exit(1);                                                               \
    }                                                                        \
  } while (0)

constexpr int BLOCK_DIM = 256;
constexpr int NUM_STAGES = 2;

// ---------- Vanilla kernel ----------
__global__ void kernel_vanilla(const float* __restrict__ a,
                               const float* __restrict__ b,
                               float* __restrict__ c,
                               int N) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < N;
         i += gridDim.x * blockDim.x) {
        c[i] = a[i] * b[i];
    }
}

// ---------- 2-stage LDGSTS prefetch kernel (primitives API) ----------
// Pattern based on GTC25-S72683 and the CUDA Programming Guide
// Section 4.11 "Asynchronous Data Copies" + Best Practices Guide L940-L999.
//
// Each thread copies sizeof(float) per async op. Uses L1 ACCESS mode (4-byte).
// The pipeline has 2 stages; each iteration commits one batch and waits for
// the oldest batch to complete.
__global__ void kernel_async_2stage(const float* __restrict__ a,
                                    const float* __restrict__ b,
                                    float* __restrict__ c,
                                    int N) {
    __shared__ float a_buf[NUM_STAGES][BLOCK_DIM];
    __shared__ float b_buf[NUM_STAGES][BLOCK_DIM];

    const int tid = threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const int base = blockIdx.x * blockDim.x;

    // Prologue: submit first NUM_STAGES batches
    for (int s = 0; s < NUM_STAGES; ++s) {
        int idx = base + s * stride + tid;
        if (idx < N) {
            __pipeline_memcpy_async(&a_buf[s][tid], &a[idx], sizeof(float));
            __pipeline_memcpy_async(&b_buf[s][tid], &b[idx], sizeof(float));
        }
        __pipeline_commit();
    }

    // Main loop
    for (int iter = 0; ; ++iter) {
        int stage = iter % NUM_STAGES;
        long long compute_base = (long long)base + (long long)iter * stride;

        // If nothing left to compute, break
        if (compute_base >= N) break;

        // Wait for this stage to be ready (wait_prior(NUM_STAGES - 1) means
        // "wait until at most NUM_STAGES - 1 stages are pending")
        __pipeline_wait_prior(NUM_STAGES - 1);

        // Compute from shared memory
        long long compute_idx = compute_base + tid;
        if (compute_idx < N) {
            c[compute_idx] = a_buf[stage][tid] * b_buf[stage][tid];
        }

        // Submit the next batch into the stage we just consumed
        long long prefetch_base = (long long)base + (long long)(iter + NUM_STAGES) * stride;
        if (prefetch_base < N) {
            long long pidx = prefetch_base + tid;
            if (pidx < N) {
                __pipeline_memcpy_async(&a_buf[stage][tid], &a[pidx], sizeof(float));
                __pipeline_memcpy_async(&b_buf[stage][tid], &b[pidx], sizeof(float));
            }
        }
        __pipeline_commit();
    }
}

int main() {
    constexpr int N = 256 * 1024 * 1024;  // 256M floats = 1 GiB
    constexpr size_t bytes = (size_t)N * sizeof(float);

    // Print device info
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s  SM: %d.%d  CUDA runtime: %d.%d\n",
           prop.name, prop.major, prop.minor,
           CUDART_VERSION / 1000, (CUDART_VERSION % 1000) / 10);

    int driver_version = 0;
    CHECK(cudaDriverGetVersion(&driver_version));
    printf("Driver version: %d.%d\n", driver_version / 1000,
           (driver_version % 1000) / 10);

    // Allocate
    float *d_a, *d_b, *d_c;
    CHECK(cudaMalloc(&d_a, bytes));
    CHECK(cudaMalloc(&d_b, bytes));
    CHECK(cudaMalloc(&d_c, bytes));

    // Initialize with known pattern
    std::vector<float> h_a(N), h_b(N), h_ref(N);
    for (int i = 0; i < N; ++i) {
        h_a[i] = 1.0f + (i % 1000) * 0.001f;
        h_b[i] = 2.0f - (i % 500) * 0.002f;
        h_ref[i] = h_a[i] * h_b[i];
    }
    CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    int sm_count;
    CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0));
    int blocks = sm_count * 8;  // target 100% occupancy with 256 threads/block

    auto measure = [&](const char* name, auto launch_fn) {
        // Correctness check
        CHECK(cudaMemset(d_c, 0, bytes));
        launch_fn();
        CHECK(cudaDeviceSynchronize());

        std::vector<float> h_c(N);
        CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));
        float max_err = 0.0f;
        for (int i = 0; i < N; ++i) {
            float err = fabsf(h_c[i] - h_ref[i]);
            if (err > max_err) max_err = err;
        }
        if (max_err > 1e-5f) {
            printf("[%s] CORRECTNESS FAILED: max_abs_err = %e\n", name, max_err);
            return;
        }

        // Warmup
        for (int i = 0; i < 5; ++i) launch_fn();
        CHECK(cudaDeviceSynchronize());

        // Measure 20 runs
        cudaEvent_t e_start, e_end;
        CHECK(cudaEventCreate(&e_start));
        CHECK(cudaEventCreate(&e_end));

        std::vector<float> times_ms;
        for (int i = 0; i < 20; ++i) {
            CHECK(cudaEventRecord(e_start));
            launch_fn();
            CHECK(cudaEventRecord(e_end));
            CHECK(cudaEventSynchronize(e_end));
            float ms;
            CHECK(cudaEventElapsedTime(&ms, e_start, e_end));
            times_ms.push_back(ms);
        }
        std::sort(times_ms.begin(), times_ms.end());
        float median = times_ms[10];
        float p10 = times_ms[2];
        float p90 = times_ms[18];

        // 3 arrays * N * 4 bytes (2 reads + 1 write)
        double bw_gb = 3.0 * N * sizeof(float) / (median * 1e-3) / 1e9;
        printf("[%s] median=%.4f ms  p10=%.4f ms  p90=%.4f ms  BW=%.2f GB/s  max_err=%e\n",
               name, median, p10, p90, bw_gb, max_err);

        CHECK(cudaEventDestroy(e_start));
        CHECK(cudaEventDestroy(e_end));
    };

    printf("N = %d  blocks = %d  threads/block = %d\n\n", N, blocks, BLOCK_DIM);

    measure("vanilla", [&]() {
        kernel_vanilla<<<blocks, BLOCK_DIM>>>(d_a, d_b, d_c, N);
    });

    measure("async-2stage", [&]() {
        kernel_async_2stage<<<blocks, BLOCK_DIM>>>(d_a, d_b, d_c, N);
    });

    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));
    CHECK(cudaFree(d_c));
    return 0;
}
