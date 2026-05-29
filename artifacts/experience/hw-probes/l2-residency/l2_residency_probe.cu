// l2_residency_probe.cu -- Microbenchmark: effective bandwidth of
// `repeat_read_sum` under three L2 policies, across three working-set
// sizes. Backs the cost model for
// knowledge/30-skill/memory/l2-access-policy/.
//
// Target: H200 (sm_90a, 60 MiB L2), CUDA 12.9
//
// Each (working-set, policy) pair runs a grid-stride-loop kernel that
// sums the buffer `N_REPEATS` times within one launch. After the first
// repeat the buffer is (at best) resident in L2 and subsequent repeats
// expose the L2 policy's effect on effective bandwidth.
//
// Policies:
//   P0. none        -- no accessPolicyWindow, default L2 behavior
//   P1. persist@1.0 -- accessPolicyWindow(hitRatio=1.0, hitProp=Persisting,
//                      num_bytes=WS), set-aside sized to 3/4 of l2CacheSize.
//   P2. persist@r   -- same, but hitRatio = min(1, set_aside / WS), so when
//                      WS > set-aside the hardware stochastically persists
//                      only the fitting fraction (legacy skill S2).
//
// Working sets (bytes):
//   WS0 =  4 MiB  (fits trivially in L2, control / noise floor)
//   WS1 = 40 MiB  (fits in H200 60 MiB L2, P1 should pin)
//   WS2 = 80 MiB  (exceeds L2, P1 should thrash, P2 should partially
//                  recover)
//
// Protocol:
//   - Warmup: 5 timed launches (ignored) to settle clock/L2 state.
//   - Timed: 20 launches with CUDA events; median / p10 / p90 reported.
//   - Between (WS, policy) pairs: disable the window, call
//     `cudaCtxResetPersistingL2Cache` so the next policy starts clean.

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
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
// Repeat-read kernel. `acc *= 0.999f` per outer iteration prevents the
// compiler from hoisting the N_REPEATS loop after realizing the inner pass
// computes the same value each time.
// -------------------------------------------------------------------------
__global__ void repeat_read_sum(const float* __restrict__ buf,
                                size_t N, int N_REPEATS,
                                float* __restrict__ out) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    float acc = 0.0f;
    #pragma unroll 1
    for (int r = 0; r < N_REPEATS; ++r) {
        for (size_t i = tid; i < N; i += stride) {
            acc += buf[i];
        }
        acc *= 0.999f;
    }
    if (acc == -1.0f) out[tid] = acc;  // sink; prevents DCE
}

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

enum Policy { POL_NONE = 0, POL_PERSIST_1 = 1, POL_PERSIST_R = 2 };

static const char* policy_label(Policy p) {
    switch (p) {
        case POL_NONE:      return "none         ";
        case POL_PERSIST_1: return "persist@1.0  ";
        case POL_PERSIST_R: return "persist@tuned";
    }
    return "?";
}

int main() {
    // Query device + set set-aside capacity.
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));

    size_t l2_size            = prop.l2CacheSize;
    size_t l2_persist_max     = prop.persistingL2CacheMaxSize;
    size_t l2_window_max      = prop.accessPolicyMaxWindowSize;
    size_t set_aside          = std::min<size_t>(
        (size_t)((double)l2_size * 0.75), l2_persist_max);

    printf("Device              : %s (sm %d.%d)\n",
           prop.name, prop.major, prop.minor);
    printf("L2 total            : %.2f MiB (%zu B)\n",
           l2_size / 1048576.0, l2_size);
    printf("persistingL2Max     : %.2f MiB (%zu B)\n",
           l2_persist_max / 1048576.0, l2_persist_max);
    printf("accessPolicyWinMax  : %.2f MiB (%zu B)\n",
           l2_window_max / 1048576.0, l2_window_max);
    printf("Set-aside this run  : %.2f MiB (%zu B)\n",
           set_aside / 1048576.0, set_aside);

    CHECK_CUDA(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, set_aside));

    // Working sets: 4 / 40 / 80 MiB, in elements of float (4 B each).
    const size_t MiB = 1 << 20;
    struct Ws { const char* name; size_t bytes; };
    Ws wss[] = {
        { " 4 MiB", 4ULL  * MiB },
        { "40 MiB", 40ULL * MiB },
        { "80 MiB", 80ULL * MiB },
    };
    const int n_ws = sizeof(wss) / sizeof(wss[0]);

    // Kernel launch shape.
    const int BLOCK    = 256;
    const int GRID     = 132 * 4;      // 132 SMs × 4 blocks
    const int N_REPEATS = 32;          // inner repeats per launch
    const int WARMUP   = 5;
    const int ITERS    = 20;

    printf("\nKernel shape        : grid=%d, block=%d, N_REPEATS=%d\n",
           GRID, BLOCK, N_REPEATS);
    printf("Warmup / timed      : %d / %d\n\n", WARMUP, ITERS);

    // Allocate worst-case buffer once, reuse slices.
    size_t buf_bytes = wss[n_ws - 1].bytes;
    float* d_buf = nullptr;
    float* d_out = nullptr;
    CHECK_CUDA(cudaMalloc(&d_buf, buf_bytes));
    CHECK_CUDA(cudaMalloc(&d_out, sizeof(float) * GRID * BLOCK));
    // Fill with 1.0f so reads return deterministic non-zero.
    std::vector<float> host_init(buf_bytes / sizeof(float), 1.0f);
    CHECK_CUDA(cudaMemcpy(d_buf, host_init.data(), buf_bytes,
                          cudaMemcpyHostToDevice));
    host_init.clear();
    host_init.shrink_to_fit();

    // Dedicated stream for the probes so accessPolicyWindow is per-stream.
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    auto reset_policy = [&]() {
        cudaStreamAttrValue attr = {};
        attr.accessPolicyWindow.num_bytes = 0;
        CHECK_CUDA(cudaStreamSetAttribute(
            stream, cudaStreamAttributeAccessPolicyWindow, &attr));
        CHECK_CUDA(cudaCtxResetPersistingL2Cache());
    };

    auto apply_policy = [&](Policy pol, size_t ws_bytes) {
        cudaStreamAttrValue attr = {};
        switch (pol) {
            case POL_NONE:
                attr.accessPolicyWindow.num_bytes = 0;
                break;
            case POL_PERSIST_1: {
                size_t nb = std::min(ws_bytes, l2_window_max);
                attr.accessPolicyWindow.base_ptr  = d_buf;
                attr.accessPolicyWindow.num_bytes = nb;
                attr.accessPolicyWindow.hitRatio  = 1.0f;
                attr.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
                attr.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
                break;
            }
            case POL_PERSIST_R: {
                size_t nb = std::min(ws_bytes, l2_window_max);
                double r  = (double)set_aside / (double)ws_bytes;
                if (r > 1.0) r = 1.0;
                attr.accessPolicyWindow.base_ptr  = d_buf;
                attr.accessPolicyWindow.num_bytes = nb;
                attr.accessPolicyWindow.hitRatio  = (float)r;
                attr.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
                attr.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
                break;
            }
        }
        CHECK_CUDA(cudaStreamSetAttribute(
            stream, cudaStreamAttributeAccessPolicyWindow, &attr));
    };

    printf("%-10s %-14s %11s %11s %11s %12s\n",
           "WS", "policy", "median_ms", "p10_ms", "p90_ms", "GB/s_eff");
    printf("-------------------------------------------------------------------------------\n");

    const Policy pols[] = { POL_NONE, POL_PERSIST_1, POL_PERSIST_R };
    for (int wi = 0; wi < n_ws; ++wi) {
        size_t ws_bytes = wss[wi].bytes;
        size_t N_elems  = ws_bytes / sizeof(float);

        for (Policy pol : pols) {
            reset_policy();
            apply_policy(pol, ws_bytes);

            Stats s = time_kernel([&]() {
                repeat_read_sum<<<GRID, BLOCK, 0, stream>>>(
                    d_buf, N_elems, N_REPEATS, d_out);
            }, WARMUP, ITERS);

            // Effective BW = bytes read / wall-clock.
            double bytes    = (double)ws_bytes * (double)N_REPEATS;
            double gbps_eff = bytes / (s.median_ms * 1e-3) / 1e9;

            printf("%-10s %-14s %11.4f %11.4f %11.4f %12.2f\n",
                   wss[wi].name, policy_label(pol),
                   s.median_ms, s.p10_ms, s.p90_ms, gbps_eff);
        }
    }

    printf("\n");
    printf("Notes:\n");
    printf("  GB/s_eff = (WS * N_REPEATS) / median_ms; 1 GB = 1e9 B.\n");
    printf("  For WS <= set-aside, policy effect is usually within noise.\n");
    printf("  For WS near set-aside, persist@1.0 should lock the buffer.\n");
    printf("  For WS > set-aside, persist@1.0 thrashes; persist@tuned\n");
    printf("  keeps a (set_aside / WS) fraction hot.\n");

    reset_policy();
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(d_out));
    CHECK_CUDA(cudaFree(d_buf));
    return 0;
}
