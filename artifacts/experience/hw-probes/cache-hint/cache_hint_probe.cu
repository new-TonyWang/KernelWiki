// cache_hint_probe.cu -- Microbenchmark: effective bandwidth of a
// grid-stride read kernel under 6 load-hint variants (default / __ldg /
// __ldca / __ldcg / __ldcs / __ldcv), at two working-set sizes:
//   - DRAM regime (256 MiB, >> 60 MiB L2): expect all 5 cache variants
//     to collapse to the DRAM bandwidth floor; __ldcv bypasses every
//     level and should be visibly slower.
//   - L2 regime (8 MiB, well below L2) with N_REPEATS=16 inner passes:
//     expect reuse-friendly variants (default / __ldg / __ldca) to win,
//     evict-first variants (__ldcs) to degrade, __ldcv to be worst.
//
// Target: H200 (sm_90a), CUDA 12.9.
// Backs 30-skill/memory/cache-load-hints/ and answers open question
// the cache-load-hints skill's open question: "is __ldg still meaningful on H200's unified cache?"

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do {                                       \
    cudaError_t err = (call);                                       \
    if (err != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                \
                __FILE__, __LINE__, cudaGetErrorString(err));       \
        exit(1);                                                    \
    }                                                               \
} while (0)

// -------------------------------------------------------------------------
// Load variants (template specialization). V encodes the cache operator.
// -------------------------------------------------------------------------
enum : int {
    V_DEFAULT = 0,   // plain *p
    V_LDG     = 1,   // __ldg
    V_LDCA    = 2,   // __ldca (cache-all, L1+L2)
    V_LDCG    = 3,   // __ldcg (cache-global, L2 only)
    V_LDCS    = 4,   // __ldcs (cache-streaming, evict-first)
    V_LDCV    = 5,   // __ldcv (cache-volatile, bypass caches)
    V_COUNT   = 6
};

template<int V>
__device__ __forceinline__ float load_v(const float* p) {
    if constexpr (V == V_DEFAULT) return *p;
    else if constexpr (V == V_LDG)  return __ldg(p);
    else if constexpr (V == V_LDCA) return __ldca(p);
    else if constexpr (V == V_LDCG) return __ldcg(p);
    else if constexpr (V == V_LDCS) return __ldcs(p);
    else if constexpr (V == V_LDCV) return __ldcv(p);
    else return 0.0f;
}

static const char* variant_name(int v) {
    switch (v) {
        case V_DEFAULT: return "default ";
        case V_LDG:     return "__ldg   ";
        case V_LDCA:    return "__ldca  ";
        case V_LDCG:    return "__ldcg  ";
        case V_LDCS:    return "__ldcs  ";
        case V_LDCV:    return "__ldcv  ";
    }
    return "?      ";
}

// -------------------------------------------------------------------------
// One-shot streaming kernel. Each element is read once per launch; the
// outer N_REPEATS loop is `1` in DRAM regime. Single register accumulator
// with a sentinel-write sink prevents DCE. `acc *= 0.999f` between repeats
// blocks the compiler from hoisting the outer loop when N_REPEATS > 1.
// -------------------------------------------------------------------------
template<int V>
__global__ void stream_read(const float* __restrict__ in,
                            size_t N, int N_REPEATS,
                            float* __restrict__ sink) {
    size_t tid    = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    float acc = 0.0f;
    #pragma unroll 1
    for (int r = 0; r < N_REPEATS; ++r) {
        for (size_t i = tid; i < N; i += stride) {
            acc += load_v<V>(&in[i]);
        }
        acc *= 0.999f;
    }
    if (acc == -1.0f) sink[tid] = acc;
}

// -------------------------------------------------------------------------
// Timing harness: CUDA-events median / p10 / p90 over N_ITERS launches.
// -------------------------------------------------------------------------
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
    Stats s{ ms[iters / 2], ms[(int)(iters * 0.1)], ms[(int)(iters * 0.9)] };
    CHECK_CUDA(cudaEventDestroy(e0));
    CHECK_CUDA(cudaEventDestroy(e1));
    return s;
}

// -------------------------------------------------------------------------
// Dispatch helper: select a compiled template specialization by runtime v.
// -------------------------------------------------------------------------
static void launch_variant(int v, const float* in, size_t N, int N_REPEATS,
                           float* sink, dim3 grid, dim3 block,
                           cudaStream_t stream) {
    switch (v) {
        case V_DEFAULT:
            stream_read<V_DEFAULT><<<grid, block, 0, stream>>>(in, N, N_REPEATS, sink); break;
        case V_LDG:
            stream_read<V_LDG>    <<<grid, block, 0, stream>>>(in, N, N_REPEATS, sink); break;
        case V_LDCA:
            stream_read<V_LDCA>   <<<grid, block, 0, stream>>>(in, N, N_REPEATS, sink); break;
        case V_LDCG:
            stream_read<V_LDCG>   <<<grid, block, 0, stream>>>(in, N, N_REPEATS, sink); break;
        case V_LDCS:
            stream_read<V_LDCS>   <<<grid, block, 0, stream>>>(in, N, N_REPEATS, sink); break;
        case V_LDCV:
            stream_read<V_LDCV>   <<<grid, block, 0, stream>>>(in, N, N_REPEATS, sink); break;
    }
}

int main() {
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("Device           : %s (sm %d.%d)\n",
           prop.name, prop.major, prop.minor);
    printf("L2 total         : %.2f MiB\n", prop.l2CacheSize / 1048576.0);
    printf("Mem clock        : %.2f MHz\n", prop.memoryClockRate / 1000.0);
    printf("Mem bus width    : %d bits\n", prop.memoryBusWidth);
    // H200 HBM3e peak ≈ memoryClockRate * 2 (DDR) * (busWidth/8) / 1e6 GB/s,
    // but prop.memoryClockRate is often stale; just report for reference.
    printf("\n");

    const size_t MiB = 1 << 20;

    // Regime A: DRAM-bound (256 MiB > 60 MiB L2). Single read per element.
    const size_t N_dram      = 256ULL * MiB / sizeof(float);
    const int    REPEATS_drm = 1;
    // Regime B: L2-resident (8 MiB). Multiple inner passes so data is warm.
    const size_t N_l2        = 8ULL * MiB / sizeof(float);
    const int    REPEATS_l2  = 16;

    const int BLOCK = 256;
    const int GRID  = 132 * 4;       // 132 SMs × 4 blocks
    const int WARMUP = 5;
    const int ITERS  = 20;

    // Allocate worst-case buffer once.
    size_t buf_bytes = std::max(N_dram, N_l2) * sizeof(float);
    float* d_in   = nullptr;
    float* d_sink = nullptr;
    CHECK_CUDA(cudaMalloc(&d_in, buf_bytes));
    CHECK_CUDA(cudaMalloc(&d_sink, sizeof(float) * GRID * BLOCK));

    // Fill with 1.0f for deterministic reads.
    std::vector<float> host_init(buf_bytes / sizeof(float), 1.0f);
    CHECK_CUDA(cudaMemcpy(d_in, host_init.data(), buf_bytes,
                          cudaMemcpyHostToDevice));
    host_init.clear(); host_init.shrink_to_fit();

    cudaStream_t stream = nullptr; // default stream OK; no policy window here.

    struct Regime {
        const char* label;
        size_t N;
        int    N_REPEATS;
        double bytes_per_launch;
    };
    Regime regimes[] = {
        { "DRAM  256 MiB x1 ",  N_dram, REPEATS_drm, (double)N_dram * sizeof(float) * REPEATS_drm },
        { "L2    8 MiB x16  ",  N_l2,   REPEATS_l2,  (double)N_l2   * sizeof(float) * REPEATS_l2  },
    };
    const int n_reg = sizeof(regimes) / sizeof(regimes[0]);

    printf("%-18s %-10s %11s %11s %11s %12s\n",
           "regime", "variant", "median_ms", "p10_ms", "p90_ms", "GB/s_eff");
    printf("---------------------------------------------------------------------------------\n");

    for (int ri = 0; ri < n_reg; ++ri) {
        Regime R = regimes[ri];
        for (int v = 0; v < V_COUNT; ++v) {
            Stats s = time_kernel([&]() {
                launch_variant(v, d_in, R.N, R.N_REPEATS, d_sink,
                               dim3(GRID), dim3(BLOCK), stream);
            }, WARMUP, ITERS);
            double gbps = R.bytes_per_launch / (s.median_ms * 1e-3) / 1e9;
            printf("%-18s %-10s %11.4f %11.4f %11.4f %12.2f\n",
                   R.label, variant_name(v),
                   s.median_ms, s.p10_ms, s.p90_ms, gbps);
        }
        printf("\n");
    }

    printf("Notes:\n");
    printf("  DRAM regime : 256 MiB buffer >> 60 MiB L2, each element read once.\n");
    printf("                Expect all cache variants to collapse to DRAM BW.\n");
    printf("                __ldcv bypasses caches so should be slower.\n");
    printf("  L2 regime   : 8 MiB fits L2; 16 inner repeats expose cache reuse.\n");
    printf("                Cache-all / __ldg / __ldca should dominate; __ldcs\n");
    printf("                (evict-first) and __ldcv should degrade.\n");

    CHECK_CUDA(cudaFree(d_sink));
    CHECK_CUDA(cudaFree(d_in));
    return 0;
}
