// launch_overhead_probe.cu -- measure the per-launch floor on H200 sm_9.0a
// and the workload-too-small crossover point.
//
// Scope: SINGLE-KERNEL launch cost only. Per the migration plan, stream /
// graph / persistent-kernel variants are deliberately out of scope — they
// belong to 90-system-level (bucket I) not here.
//
// Three measurements:
//
//   Part A. Per-launch floor, varying launch shape:
//     - empty kernel launched as <<<1, 1>>>         (smallest possible)
//     - empty kernel launched as <<<1, 256>>>       (one block, full warp)
//     - empty kernel launched as <<<528, 256>>>     (full H200 grid: 132 SMs x 4 blocks)
//     - each above also via cudaLaunchKernel (driver API, distinct from <<< >>>)
//     - N = 10000 launches, CUDA events around the whole batch, divide by N.
//
//   Part B. Workload crossover: FMA per thread from 0 to 4096 steps on a
//     full-grid launch. At what per-thread work does the kernel leave the
//     "launch overhead dominates" regime and enter "GPU-work dominates"?
//     Reports both median ms per launch and effective-work TFLOPS to
//     visualise the crossover.
//
//   Part C. Timing method cost: CUDA-event bracket cost itself (empty
//     `cudaEventRecord / cudaEventSynchronize` pair with no kernel
//     between). Establishes the noise floor of our timing.

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>

#define CHECK(call) do {                                          \
    cudaError_t e = (call);                                       \
    if (e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                 \
                __FILE__, __LINE__, cudaGetErrorString(e));       \
        exit(1);                                                  \
    }                                                             \
} while (0)

// ------------------------------------------------------------------
// Kernels
// ------------------------------------------------------------------
__global__ void empty_k() {}

// Parametric "small compute" kernel: WORK_STEPS FMA per thread.
// Used for Part B (workload crossover).
template<int WORK_STEPS>
__global__ void work_k(float* __restrict__ sink, size_t N) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    float a = __int_as_float((unsigned)tid & 0xFFFF);
    #pragma unroll 1
    for (int i = 0; i < WORK_STEPS; ++i) {
        a = __fmaf_rn(a, 0.99999f, 1e-7f);
    }
    if (a == -1.0f) sink[tid] = a;  // DCE guard
}

// ------------------------------------------------------------------
// Timing harness — CUDA events around a batch of N launches.
// We time the WHOLE batch and divide by N to amortise event overhead.
// ------------------------------------------------------------------
template <typename Body>
float time_batch_ms(Body body, int warmup, int iters, int N_launch) {
    cudaEvent_t e0, e1;
    CHECK(cudaEventCreate(&e0));
    CHECK(cudaEventCreate(&e1));
    // Warmup batches
    for (int w = 0; w < warmup; ++w) {
        for (int i = 0; i < N_launch; ++i) body();
        CHECK(cudaDeviceSynchronize());
    }
    std::vector<float> ms(iters);
    for (int it = 0; it < iters; ++it) {
        CHECK(cudaEventRecord(e0));
        for (int i = 0; i < N_launch; ++i) body();
        CHECK(cudaEventRecord(e1));
        CHECK(cudaEventSynchronize(e1));
        CHECK(cudaEventElapsedTime(&ms[it], e0, e1));
    }
    std::sort(ms.begin(), ms.end());
    CHECK(cudaEventDestroy(e0));
    CHECK(cudaEventDestroy(e1));
    return ms[iters / 2];  // median
}

// cudaLaunchKernel wrapper for an empty kernel
static cudaError_t launch_empty_api(dim3 grid, dim3 block) {
    return cudaLaunchKernel(
        reinterpret_cast<const void*>(empty_k),
        grid, block, nullptr, 0, nullptr);
}

// cudaLaunchKernelEx with NO attributes attached — isolates the attribute-
// parsing cost from the base launch cost.
static cudaError_t launch_empty_ex_noattr(dim3 grid, dim3 block) {
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = 0;
    cfg.stream = nullptr;
    cfg.attrs = nullptr;
    cfg.numAttrs = 0;
    return cudaLaunchKernelEx(&cfg, empty_k);
}

// cudaLaunchKernelEx with ONE access-policy-window attribute attached.
// This is a realistic "why you would pick Ex over plain cudaLaunchKernel"
// scenario — the attribute requires the Ex path to apply the window to
// this launch. Same API call otherwise; measures the per-attribute cost.
static cudaError_t launch_empty_ex_apw(dim3 grid, dim3 block,
                                       void* apw_buffer, size_t apw_bytes) {
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = 0;
    cfg.stream = nullptr;

    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeAccessPolicyWindow;
    attr.val.accessPolicyWindow.base_ptr  = apw_buffer;
    attr.val.accessPolicyWindow.num_bytes = apw_bytes;
    attr.val.accessPolicyWindow.hitRatio  = 1.0f;
    attr.val.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
    attr.val.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;

    cfg.attrs    = &attr;
    cfg.numAttrs = 1;
    return cudaLaunchKernelEx(&cfg, empty_k);
}

int main() {
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device  : %s (sm %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("SMs     : %d\n", prop.multiProcessorCount);

    const int WARMUP = 3;
    const int ITERS  = 10;
    const int N_BATCH = 10000;      // launches per batch

    printf("\n=== Part A. Per-launch floor (N_BATCH = %d launches, median of %d timed batches) ===\n",
           N_BATCH, ITERS);
    printf("%-30s %-18s %14s %12s\n",
           "shape", "mechanism", "batch_ms", "us/launch");
    printf("----------------------------------------------------------------------------\n");

    struct Shape { const char* name; dim3 grid; dim3 block; };
    Shape shapes[] = {
        { "<<<1, 1>>>",            dim3(1),        dim3(1)   },
        { "<<<1, 256>>>",          dim3(1),        dim3(256) },
        { "<<<528, 256>>> (H200)", dim3(132*4),    dim3(256) },
    };

    // 4 KiB dummy buffer for the access-policy-window attribute — just
    // needs a valid device pointer; the empty kernel never touches it.
    const size_t APW_BYTES = 4096;
    void* d_apw = nullptr;
    CHECK(cudaMalloc(&d_apw, APW_BYTES));

    for (const auto& s : shapes) {
        // (i) triple-chevron
        float ms_chev = time_batch_ms([&]() {
            empty_k<<<s.grid, s.block>>>();
        }, WARMUP, ITERS, N_BATCH);
        printf("%-30s %-18s %14.4f %12.3f\n",
               s.name, "<<< >>>", ms_chev, ms_chev * 1000.0 / N_BATCH);

        // (ii) cudaLaunchKernel
        float ms_api = time_batch_ms([&]() {
            launch_empty_api(s.grid, s.block);
        }, WARMUP, ITERS, N_BATCH);
        printf("%-30s %-18s %14.4f %12.3f\n",
               s.name, "cudaLaunchKernel", ms_api, ms_api * 1000.0 / N_BATCH);

        // (iii) cudaLaunchKernelEx with no attributes — isolates the Ex
        //        path's fixed cost vs plain cudaLaunchKernel.
        float ms_ex0 = time_batch_ms([&]() {
            launch_empty_ex_noattr(s.grid, s.block);
        }, WARMUP, ITERS, N_BATCH);
        printf("%-30s %-18s %14.4f %12.3f\n",
               s.name, "LaunchKernelEx(0)", ms_ex0, ms_ex0 * 1000.0 / N_BATCH);

        // (iv) cudaLaunchKernelEx with one access-policy-window attr —
        //        measures the per-attribute incremental cost.
        float ms_ex1 = time_batch_ms([&]() {
            launch_empty_ex_apw(s.grid, s.block, d_apw, APW_BYTES);
        }, WARMUP, ITERS, N_BATCH);
        printf("%-30s %-18s %14.4f %12.3f\n",
               s.name, "LaunchKernelEx(+APW)", ms_ex1, ms_ex1 * 1000.0 / N_BATCH);
    }
    CHECK(cudaFree(d_apw));

    printf("\n=== Part B. Workload crossover (full-grid, WORK_STEPS sweep) ===\n");
    printf("%-12s %14s %12s %14s\n",
           "WORK_STEPS", "batch_ms", "us/launch", "GFLOPS_eff");
    printf("-----------------------------------------------------------------\n");

    const int GRID = 132 * 4;
    const int BLOCK = 256;
    const size_t N_THREADS = (size_t)GRID * BLOCK;
    float* d_sink = nullptr;
    CHECK(cudaMalloc(&d_sink, N_THREADS * sizeof(float)));

    // Each FMA step = 2 FP ops per thread.
    auto run_work = [&](auto work_launch, int work_steps) {
        float ms = time_batch_ms(work_launch, WARMUP, ITERS, N_BATCH);
        double flops_per_launch = 2.0 * (double)work_steps * (double)N_THREADS;
        double gflops_eff = flops_per_launch / (ms * 1e-3 / N_BATCH) / 1e9;
        printf("%-12d %14.4f %12.3f %14.2f\n",
               work_steps, ms, ms * 1000.0 / N_BATCH, gflops_eff);
    };

    run_work([&](){ work_k<0>   <<<GRID, BLOCK>>>(d_sink, N_THREADS); }, 0);
    run_work([&](){ work_k<1>   <<<GRID, BLOCK>>>(d_sink, N_THREADS); }, 1);
    run_work([&](){ work_k<4>   <<<GRID, BLOCK>>>(d_sink, N_THREADS); }, 4);
    run_work([&](){ work_k<16>  <<<GRID, BLOCK>>>(d_sink, N_THREADS); }, 16);
    run_work([&](){ work_k<64>  <<<GRID, BLOCK>>>(d_sink, N_THREADS); }, 64);
    run_work([&](){ work_k<256> <<<GRID, BLOCK>>>(d_sink, N_THREADS); }, 256);
    run_work([&](){ work_k<1024><<<GRID, BLOCK>>>(d_sink, N_THREADS); }, 1024);
    run_work([&](){ work_k<4096><<<GRID, BLOCK>>>(d_sink, N_THREADS); }, 4096);

    printf("\n=== Part C. Timing noise floor — empty event-pair cost ===\n");
    // An empty batch: N CUDA event pairs with no kernel between.
    // Reveals how much of the per-launch us comes from the events themselves.
    cudaEvent_t e0, e1;
    CHECK(cudaEventCreate(&e0));
    CHECK(cudaEventCreate(&e1));
    for (int w = 0; w < WARMUP; ++w) {
        for (int i = 0; i < 1000; ++i) {
            CHECK(cudaEventRecord(e0));
            CHECK(cudaEventRecord(e1));
        }
        CHECK(cudaDeviceSynchronize());
    }
    float ev_noise = 0.0f;
    {
        std::vector<float> ms(ITERS);
        for (int it = 0; it < ITERS; ++it) {
            cudaEvent_t outer_s, outer_e;
            CHECK(cudaEventCreate(&outer_s));
            CHECK(cudaEventCreate(&outer_e));
            CHECK(cudaEventRecord(outer_s));
            for (int i = 0; i < N_BATCH; ++i) {
                CHECK(cudaEventRecord(e0));
                CHECK(cudaEventRecord(e1));
            }
            CHECK(cudaEventRecord(outer_e));
            CHECK(cudaEventSynchronize(outer_e));
            CHECK(cudaEventElapsedTime(&ms[it], outer_s, outer_e));
            CHECK(cudaEventDestroy(outer_s));
            CHECK(cudaEventDestroy(outer_e));
        }
        std::sort(ms.begin(), ms.end());
        ev_noise = ms[ITERS / 2];
    }
    CHECK(cudaEventDestroy(e0));
    CHECK(cudaEventDestroy(e1));
    printf("Empty event-pair batch: %.4f ms total → %.3f us per event pair\n",
           ev_noise, ev_noise * 1000.0 / N_BATCH);
    printf("(Subtract this floor from Part A us/launch figures for a\n"
           " 'pure launch overhead' estimate.)\n");

    CHECK(cudaFree(d_sink));
    return 0;
}
