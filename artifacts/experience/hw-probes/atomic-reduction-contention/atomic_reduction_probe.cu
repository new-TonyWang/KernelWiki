// atomic_reduction_probe.cu -- Microbenchmark: naive global atomicAdd vs
// hierarchical reduction (warp shuffle + shared memory + one atomic per block)
// for sum-reduction.
//
// Target: H200 (sm_90a), CUDA 12.9
// Validates the S1 technique from knowledge/30-skill/sync/atomic-reduction/skill.md
//
// Three kernels, same problem (float sum over N elements):
//   A. naive_atomic     : every thread --> global atomicAdd(&out, x[i])
//   B. hierarchical_s1  : warp shuffle -> shmem -> ONE global atomicAdd per block
//   C. grid_stride_s1   : variant of B with grid-stride loop so each block
//                         folds many elements before its single atomic
//
// Protocol: warmup 5, measure 20, CUDA events, median/p10/p90.
// Also reports correctness vs a double-precision CPU reference.

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <vector>
#include <numeric>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do {                                       \
    cudaError_t err = (call);                                       \
    if (err != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                \
                __FILE__, __LINE__, cudaGetErrorString(err));       \
        exit(1);                                                    \
    }                                                               \
} while (0)

// ---------------------------------------------------------------------------
// Kernel A. Naive: every thread issues one global atomicAdd to a single addr.
// Pathological contention — all N atomics serialize at the L2 cache.
// ---------------------------------------------------------------------------
__global__ void naive_atomic(const float* __restrict__ in,
                             float* __restrict__ out,
                             int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        atomicAdd(out, in[tid]);
    }
}

// ---------------------------------------------------------------------------
// Kernel B. S1 hierarchical: warp shuffle -> per-warp shmem -> first warp
// reduces across warps -> one atomicAdd per block.
// ---------------------------------------------------------------------------
__global__ void hierarchical_s1(const float* __restrict__ in,
                                float* __restrict__ out,
                                int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (tid < n) ? in[tid] : 0.0f;

    // Warp-level reduction via shuffle (no memory traffic).
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFFu, val, offset);

    __shared__ float warpSums[32];
    int lane   = threadIdx.x & 31;
    int warpId = threadIdx.x >> 5;
    if (lane == 0) warpSums[warpId] = val;
    __syncthreads();

    // First warp reduces the per-warp partials.
    if (warpId == 0) {
        int nWarps = (blockDim.x + 31) >> 5;
        val = (lane < nWarps) ? warpSums[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xFFFFFFFFu, val, offset);
        if (lane == 0) atomicAdd(out, val);   // ONE atomic per block
    }
}

// ---------------------------------------------------------------------------
// Kernel C. Grid-stride loop + S1. Each block chews many elements before its
// single atomic. Minimizes both contention and block-launch overhead.
// ---------------------------------------------------------------------------
__global__ void grid_stride_s1(const float* __restrict__ in,
                               float* __restrict__ out,
                               int n) {
    float val = 0.0f;
    int gridStride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += gridStride) {
        val += in[i];
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFFu, val, offset);

    __shared__ float warpSums[32];
    int lane   = threadIdx.x & 31;
    int warpId = threadIdx.x >> 5;
    if (lane == 0) warpSums[warpId] = val;
    __syncthreads();

    if (warpId == 0) {
        int nWarps = (blockDim.x + 31) >> 5;
        val = (lane < nWarps) ? warpSums[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xFFFFFFFFu, val, offset);
        if (lane == 0) atomicAdd(out, val);
    }
}

// ---------------------------------------------------------------------------
// Measurement harness
// ---------------------------------------------------------------------------
struct Stats {
    float median_ms, p10_ms, p90_ms, min_ms;
};

template <typename Launcher>
Stats time_kernel(Launcher launcher, int warmup, int iters) {
    cudaEvent_t e0, e1;
    CHECK_CUDA(cudaEventCreate(&e0));
    CHECK_CUDA(cudaEventCreate(&e1));
    for (int i = 0; i < warmup; ++i) launcher();
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> ms(iters);
    for (int i = 0; i < iters; ++i) {
        CHECK_CUDA(cudaEventRecord(e0));
        launcher();
        CHECK_CUDA(cudaEventRecord(e1));
        CHECK_CUDA(cudaEventSynchronize(e1));
        CHECK_CUDA(cudaEventElapsedTime(&ms[i], e0, e1));
    }
    std::sort(ms.begin(), ms.end());
    Stats s;
    s.min_ms    = ms.front();
    s.median_ms = ms[iters / 2];
    s.p10_ms    = ms[(int)(iters * 0.1)];
    s.p90_ms    = ms[(int)(iters * 0.9)];
    CHECK_CUDA(cudaEventDestroy(e0));
    CHECK_CUDA(cudaEventDestroy(e1));
    return s;
}

int main(int argc, char** argv) {
    const int N       = (argc > 1) ? std::atoi(argv[1]) : (32 * 1024 * 1024);  // 32M floats = 128 MB
    const int BLOCK   = 256;
    const int WARMUP  = 5;
    const int ITERS   = 20;

    printf("Problem size : N = %d floats (%.1f MB)\n", N, N * sizeof(float) / 1048576.0);
    printf("Block size   : %d\n", BLOCK);
    printf("Warmup/iters : %d / %d\n", WARMUP, ITERS);
    printf("\n");

    // Host data: use a deterministic but non-trivial pattern.
    std::vector<float> h_in(N);
    for (int i = 0; i < N; ++i) h_in[i] = (i % 31) * 1e-4f;

    // Double-precision reference sum on the host.
    double ref = 0.0;
    for (int i = 0; i < N; ++i) ref += (double)h_in[i];

    float *d_in = nullptr, *d_out = nullptr;
    CHECK_CUDA(cudaMalloc(&d_in,  N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out,     sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    int gridFull = (N + BLOCK - 1) / BLOCK;

    // Persistent grid for kernel C (grid-stride). Size = 2048 blocks (~heavy enough).
    int deviceId = 0;
    CHECK_CUDA(cudaGetDevice(&deviceId));
    int numSMs = 0;
    CHECK_CUDA(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, deviceId));
    int gridPersist = numSMs * 16;     // 16 blocks per SM as a cheap target.

    auto run_with_reset = [&](auto kernel_launcher) {
        return time_kernel([&]() {
            CHECK_CUDA(cudaMemsetAsync(d_out, 0, sizeof(float)));
            kernel_launcher();
        }, WARMUP, ITERS);
    };

    // ---- Correctness check for each kernel ----
    auto check = [&](const char* name, float got) {
        double rel_err = std::abs((double)got - ref) / std::max(std::abs(ref), 1e-9);
        printf("%-22s result=%.6f  ref=%.6f  rel_err=%.3e  %s\n",
               name, got, (float)ref, rel_err,
               (rel_err < 1e-3) ? "OK" : "FAIL");
        return rel_err < 1e-3;
    };

    float h_out = 0.0f;

    CHECK_CUDA(cudaMemset(d_out, 0, sizeof(float)));
    naive_atomic<<<gridFull, BLOCK>>>(d_in, d_out, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost));
    check("naive_atomic", h_out);

    CHECK_CUDA(cudaMemset(d_out, 0, sizeof(float)));
    hierarchical_s1<<<gridFull, BLOCK>>>(d_in, d_out, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost));
    check("hierarchical_s1", h_out);

    CHECK_CUDA(cudaMemset(d_out, 0, sizeof(float)));
    grid_stride_s1<<<gridPersist, BLOCK>>>(d_in, d_out, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost));
    check("grid_stride_s1", h_out);

    printf("\n");

    // ---- Timing ----
    Stats s_naive = run_with_reset([&]() {
        naive_atomic<<<gridFull, BLOCK>>>(d_in, d_out, N);
    });
    Stats s_s1    = run_with_reset([&]() {
        hierarchical_s1<<<gridFull, BLOCK>>>(d_in, d_out, N);
    });
    Stats s_gs    = run_with_reset([&]() {
        grid_stride_s1<<<gridPersist, BLOCK>>>(d_in, d_out, N);
    });

    auto bw = [&](float ms) {
        // effective GB/s for reading N floats from HBM
        return (double)N * sizeof(float) / (ms * 1.0e-3) / 1.0e9;
    };

    printf("Kernel                  median_ms   p10_ms    p90_ms    eff_BW_GB/s\n");
    printf("--------------------------------------------------------------------\n");
    printf("naive_atomic           %9.4f %9.4f %9.4f   %8.2f\n",
           s_naive.median_ms, s_naive.p10_ms, s_naive.p90_ms, bw(s_naive.median_ms));
    printf("hierarchical_s1        %9.4f %9.4f %9.4f   %8.2f\n",
           s_s1.median_ms, s_s1.p10_ms, s_s1.p90_ms, bw(s_s1.median_ms));
    printf("grid_stride_s1         %9.4f %9.4f %9.4f   %8.2f\n",
           s_gs.median_ms, s_gs.p10_ms, s_gs.p90_ms, bw(s_gs.median_ms));
    printf("\n");
    printf("Speedup S1 / naive       : %.2fx\n", s_naive.median_ms / s_s1.median_ms);
    printf("Speedup grid_stride / naive: %.2fx\n", s_naive.median_ms / s_gs.median_ms);
    printf("Speedup grid_stride / S1   : %.2fx\n", s_s1.median_ms / s_gs.median_ms);

    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    return 0;
}
