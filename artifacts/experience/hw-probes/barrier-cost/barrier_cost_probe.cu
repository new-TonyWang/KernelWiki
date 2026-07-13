// barrier_cost_probe.cu -- Microbenchmark: per-barrier cycle cost of
// __syncwarp / __syncthreads / mbarrier arrive+wait, swept over block
// sizes {128, 256, 512, 1024}. Validates the barrier cost model for
// knowledge/30-skill/sync/barrier-optimization/.
//
// Target: H200 (sm_90a), CUDA 12.9
//
// Each kernel runs an inner loop of N_ITERS barrier calls back-to-back
// (with a tiny volatile-smem payload so the compiler cannot delete the
// loop) and reports average cycles per call.
//
// Protocol:
//   - Single block per kernel launch (block size varies).
//   - Inner loop N_ITERS = 10,000 barriers.
//   - Wall-clock via CUDA events (5 warmup + 20 iters, median).
//   - Per-call cycle cost = median_ms * clockRateHz / N_ITERS.
//   - Also reports clock64() delta captured in-kernel for reference.

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>
#include <cuda/barrier>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define CHECK_CUDA(call) do {                                       \
    cudaError_t err = (call);                                       \
    if (err != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                \
                __FILE__, __LINE__, cudaGetErrorString(err));       \
        exit(1);                                                    \
    }                                                               \
} while (0)

constexpr int N_ITERS = 10000;

// ---------------------------------------------------------------------------
// Kernel A. __syncwarp loop. One warp per block is exercised; all warps
// call __syncwarp in lockstep so the measurement is honest.
// ---------------------------------------------------------------------------
__global__ void syncwarp_loop(unsigned long long* out_cycles) {
    unsigned long long t0 = clock64();
    #pragma unroll 1
    for (int i = 0; i < N_ITERS; ++i) {
        __syncwarp(0xFFFFFFFFu);
    }
    unsigned long long t1 = clock64();
    if (threadIdx.x == 0) out_cycles[blockIdx.x] = t1 - t0;
}

// ---------------------------------------------------------------------------
// Kernel B. __syncthreads loop. Block-wide barrier N_ITERS times.
// ---------------------------------------------------------------------------
__global__ void syncthreads_loop(unsigned long long* out_cycles) {
    __shared__ volatile int dummy;                // kept live to prevent DCE
    if (threadIdx.x == 0) dummy = 0;
    __syncthreads();

    unsigned long long t0 = clock64();
    #pragma unroll 1
    for (int i = 0; i < N_ITERS; ++i) {
        __syncthreads();
        if (threadIdx.x == 0) dummy += 1;         // cheap payload, keeps loop alive
    }
    unsigned long long t1 = clock64();
    if (threadIdx.x == 0) out_cycles[blockIdx.x] = t1 - t0;
}

// ---------------------------------------------------------------------------
// Kernel C. mbarrier arrive+wait via cuda::barrier. One barrier, re-used
// across iterations (each arrive/wait advances a phase).
// ---------------------------------------------------------------------------
__global__ void mbarrier_loop(unsigned long long* out_cycles) {
    __shared__ cuda::barrier<cuda::thread_scope_block> bar;
    auto block = cg::this_thread_block();
    if (block.thread_rank() == 0) init(&bar, block.size());
    block.sync();

    unsigned long long t0 = clock64();
    #pragma unroll 1
    for (int i = 0; i < N_ITERS; ++i) {
        auto token = bar.arrive();
        bar.wait(std::move(token));
    }
    unsigned long long t1 = clock64();
    if (threadIdx.x == 0) out_cycles[blockIdx.x] = t1 - t0;
}

// ---------------------------------------------------------------------------
// Measurement harness
// ---------------------------------------------------------------------------
struct Stats { float median_ms, p10_ms, p90_ms; };

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
    s.median_ms = ms[iters / 2];
    s.p10_ms    = ms[(int)(iters * 0.1)];
    s.p90_ms    = ms[(int)(iters * 0.9)];
    CHECK_CUDA(cudaEventDestroy(e0));
    CHECK_CUDA(cudaEventDestroy(e1));
    return s;
}

int main() {
    const int block_sizes[] = {128, 256, 512, 1024};
    const int n_sizes = sizeof(block_sizes) / sizeof(block_sizes[0]);
    const int WARMUP = 5;
    const int ITERS  = 20;

    // Query SM clock rate for cycle -> wall-time conversion (reference).
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s  (sm %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("SM clock rate: %d kHz\n", prop.clockRate);
    printf("Barriers per run : %d\n", N_ITERS);
    printf("Warmup / iters   : %d / %d\n", WARMUP, ITERS);
    printf("\n");

    unsigned long long* d_cycles = nullptr;
    CHECK_CUDA(cudaMalloc(&d_cycles, sizeof(unsigned long long)));

    printf("%-6s %18s %18s %18s\n",
           "block", "__syncwarp ns/call", "__syncthreads ns/call", "mbarrier ns/call");
    printf("-------------------------------------------------------------------------\n");

    for (int bi = 0; bi < n_sizes; ++bi) {
        int B = block_sizes[bi];

        Stats s_warp = time_kernel([&](){
            syncwarp_loop<<<1, B>>>(d_cycles);
        }, WARMUP, ITERS);
        Stats s_th = time_kernel([&](){
            syncthreads_loop<<<1, B>>>(d_cycles);
        }, WARMUP, ITERS);
        Stats s_mb = time_kernel([&](){
            mbarrier_loop<<<1, B>>>(d_cycles);
        }, WARMUP, ITERS);

        // ns per call = median_ms * 1e6 / N_ITERS
        auto ns = [](float ms){ return ms * 1e6 / N_ITERS; };

        printf("%-6d %18.2f %18.2f %18.2f\n",
               B, ns(s_warp.median_ms), ns(s_th.median_ms), ns(s_mb.median_ms));
    }

    printf("\n");
    printf("%-6s %18s %18s %18s\n",
           "block", "__syncwarp total_ms", "__syncthreads total_ms", "mbarrier total_ms");
    printf("-------------------------------------------------------------------------\n");
    for (int bi = 0; bi < n_sizes; ++bi) {
        int B = block_sizes[bi];

        Stats s_warp = time_kernel([&](){ syncwarp_loop<<<1, B>>>(d_cycles); }, WARMUP, ITERS);
        Stats s_th   = time_kernel([&](){ syncthreads_loop<<<1, B>>>(d_cycles); }, WARMUP, ITERS);
        Stats s_mb   = time_kernel([&](){ mbarrier_loop<<<1, B>>>(d_cycles); }, WARMUP, ITERS);

        printf("%-6d %18.4f %18.4f %18.4f\n",
               B, s_warp.median_ms, s_th.median_ms, s_mb.median_ms);
    }
    printf("\n");
    printf("Note: single-block kernels; per-call cost shown.\n");
    printf("      mbarrier includes arrive + wait round-trip per iter.\n");

    CHECK_CUDA(cudaFree(d_cycles));
    return 0;
}
