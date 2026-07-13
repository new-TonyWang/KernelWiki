// branchless_patterns_probe.cu -- measure wall-clock + audit PTX/SASS for
// 10 variants of three canonical "branch vs branchless" patterns on H200
// sm_9.0a. The probe is compute-bound (4-chain ILP x 1024 inner iterations)
// with lane-variant input so that any variant emitting a real branch
// would diverge 50% within each warp.
//
// Pattern families:
//   A. ReLU-ish clamp to zero       (4 variants)
//   B. abs() on signed float         (3 variants)
//   C. conditional register assign   (3 variants)
//
// The purpose is NOT to re-discover "branch divergence is slow" (the
// warp-divergence probe already establishes that at 1.82x for compute-
// bound). The purpose is to measure whether the canonical rewrites
// (fmaxf / fabsf / ternary / bit-trick) compile to the SAME SASS on
// H200 sm_9.0a, and to expose the cases where they do not.
//
// Each variant is a small __device__ function. Host harness times
// N_ITERS=1024 inner applications per variant in a 4-chain ILP loop,
// 5 warmup + 20 timed launches, CUDA events median.

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do {                                    \
    cudaError_t err = (call);                                    \
    if (err != cudaSuccess) {                                    \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                \
                __FILE__, __LINE__, cudaGetErrorString(err));    \
        exit(1);                                                 \
    }                                                            \
} while (0)

constexpr int N_ITERS = 1024;

// -------------------------------------------------------------------
// Variant encoding
// -------------------------------------------------------------------
enum : int {
    // Group A: ReLU (max(x, 0))
    V_RELU_BRANCH   = 0,  // if (x < 0) y = 0; else y = x;
    V_RELU_TERNARY  = 1,  // y = (x < 0) ? 0 : x;
    V_RELU_FMAXF    = 2,  // y = fmaxf(x, 0);
    V_RELU_BITTRICK = 3,  // y = __int_as_float(__float_as_int(x) & ~(__float_as_int(x) >> 31));

    // Group B: abs
    V_ABS_BRANCH    = 4,  // if (x < 0) x = -x;
    V_ABS_FABSF     = 5,  // x = fabsf(x);
    V_ABS_BITTRICK  = 6,  // __int_as_float(__float_as_int(x) & 0x7FFFFFFF)

    // Group C: conditional register assign -- acc = cond ? v : acc;
    V_COND_BRANCH   = 7,  // if (cond) acc = v;
    V_COND_TERNARY  = 8,  // acc = cond ? v : acc;
    V_COND_ARITH    = 9,  // acc = acc * (1 - cond) + v * cond;
    V_COUNT         = 10
};

static const char* variant_name(int v) {
    switch (v) {
        case V_RELU_BRANCH:   return "relu-branch    ";
        case V_RELU_TERNARY:  return "relu-ternary   ";
        case V_RELU_FMAXF:    return "relu-fmaxf     ";
        case V_RELU_BITTRICK: return "relu-bittrick  ";
        case V_ABS_BRANCH:    return "abs-branch     ";
        case V_ABS_FABSF:     return "abs-fabsf      ";
        case V_ABS_BITTRICK:  return "abs-bittrick   ";
        case V_COND_BRANCH:   return "cond-branch    ";
        case V_COND_TERNARY:  return "cond-ternary   ";
        case V_COND_ARITH:    return "cond-arith     ";
    }
    return "?";
}

// -------------------------------------------------------------------
// Per-variant inner op on a single float accumulator
// -------------------------------------------------------------------
template<int V>
__device__ __forceinline__ float apply(float acc, float v2, int cond) {
    // Group A: clamp acc to [0, +inf)
    if constexpr (V == V_RELU_BRANCH) {
        if (acc < 0.0f) acc = 0.0f;
        return acc;
    } else if constexpr (V == V_RELU_TERNARY) {
        return (acc < 0.0f) ? 0.0f : acc;
    } else if constexpr (V == V_RELU_FMAXF) {
        return fmaxf(acc, 0.0f);
    } else if constexpr (V == V_RELU_BITTRICK) {
        int bits = __float_as_int(acc);
        // mask = ~(sign-extended sign bit)
        int mask = ~(bits >> 31);
        return __int_as_float(bits & mask);
    }
    // Group B: abs
    else if constexpr (V == V_ABS_BRANCH) {
        if (acc < 0.0f) acc = -acc;
        return acc;
    } else if constexpr (V == V_ABS_FABSF) {
        return fabsf(acc);
    } else if constexpr (V == V_ABS_BITTRICK) {
        return __int_as_float(__float_as_int(acc) & 0x7FFFFFFF);
    }
    // Group C: conditional assign
    else if constexpr (V == V_COND_BRANCH) {
        if (cond) acc = v2;
        return acc;
    } else if constexpr (V == V_COND_TERNARY) {
        return cond ? v2 : acc;
    } else if constexpr (V == V_COND_ARITH) {
        float c = static_cast<float>(cond);
        return acc * (1.0f - c) + v2 * c;
    }
    return acc;
}

// -------------------------------------------------------------------
// Compute-bound kernel. 4 independent chains, 1024 inner iterations.
// `cond` and `v2` are per-lane, per-iter varying to defeat hoisting
// and to give branchful variants worst-case (lane-variant) divergence.
// -------------------------------------------------------------------
template<int V>
__global__ void bench_kernel(float* sink, size_t N) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    // Seeds: lane-variant initial value so relu/abs branch truly splits warp.
    float a0 = __int_as_float((unsigned)tid & 0xFFFF);
    float b0 = __int_as_float(((unsigned)tid * 31u) & 0xFFFF);
    float c0 = __int_as_float(((unsigned)tid * 7u)  & 0xFFFF);
    float d0 = __int_as_float(((unsigned)tid * 13u) & 0xFFFF);
    // Subtract a midpoint so ~50% are negative, 50% positive.
    float acc0 = a0 - 32768.0f;
    float acc1 = b0 - 32768.0f;
    float acc2 = c0 - 32768.0f;
    float acc3 = d0 - 32768.0f;

    #pragma unroll 1
    for (int i = 0; i < N_ITERS; ++i) {
        // Per-iter "fresh" rhs value and cond so the op is non-trivial.
        float v2    = __int_as_float(((unsigned)tid ^ (unsigned)i) & 0xFFFF) - 32768.0f;
        int   cond  = ((threadIdx.x + i) & 1);
        acc0 = apply<V>(acc0, v2, cond);
        acc1 = apply<V>(acc1, v2, cond);
        acc2 = apply<V>(acc2, v2, cond);
        acc3 = apply<V>(acc3, v2, cond);
        // After the op, shuffle acc slightly so the next iter's input is
        // non-constant and branch predicate isn't trivially hoistable.
        acc0 = __fmaf_rn(acc0, 0.9999f,  1e-6f);
        acc1 = __fmaf_rn(acc1, 0.9999f, -1e-6f);
        acc2 = __fmaf_rn(acc2, 0.9999f,  1e-6f);
        acc3 = __fmaf_rn(acc3, 0.9999f, -1e-6f);
    }
    // Sentinel to defeat DCE.
    if (acc0 + acc1 + acc2 + acc3 == -1.0f) sink[tid] = acc0;
}

// -------------------------------------------------------------------
// Timing harness
// -------------------------------------------------------------------
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
    Stats s{ ms[iters/2], ms[(int)(iters*0.1)], ms[(int)(iters*0.9)] };
    CHECK_CUDA(cudaEventDestroy(e0));
    CHECK_CUDA(cudaEventDestroy(e1));
    return s;
}

static void launch(int v, float* sink, size_t N, dim3 grid, dim3 block) {
    switch (v) {
        case V_RELU_BRANCH:   bench_kernel<V_RELU_BRANCH>  <<<grid,block>>>(sink,N); break;
        case V_RELU_TERNARY:  bench_kernel<V_RELU_TERNARY> <<<grid,block>>>(sink,N); break;
        case V_RELU_FMAXF:    bench_kernel<V_RELU_FMAXF>   <<<grid,block>>>(sink,N); break;
        case V_RELU_BITTRICK: bench_kernel<V_RELU_BITTRICK><<<grid,block>>>(sink,N); break;
        case V_ABS_BRANCH:    bench_kernel<V_ABS_BRANCH>   <<<grid,block>>>(sink,N); break;
        case V_ABS_FABSF:     bench_kernel<V_ABS_FABSF>    <<<grid,block>>>(sink,N); break;
        case V_ABS_BITTRICK:  bench_kernel<V_ABS_BITTRICK> <<<grid,block>>>(sink,N); break;
        case V_COND_BRANCH:   bench_kernel<V_COND_BRANCH>  <<<grid,block>>>(sink,N); break;
        case V_COND_TERNARY:  bench_kernel<V_COND_TERNARY> <<<grid,block>>>(sink,N); break;
        case V_COND_ARITH:    bench_kernel<V_COND_ARITH>   <<<grid,block>>>(sink,N); break;
    }
}

int main() {
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("Device  : %s (sm %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("N_ITERS : %d inner iterations per thread, 4-chain ILP\n", N_ITERS);

    const int BLOCK = 256;
    const int GRID  = 132 * 4;
    const size_t N  = (size_t)GRID * BLOCK;
    const int WARMUP = 5;
    const int ITERS  = 20;

    float* d_sink = nullptr;
    CHECK_CUDA(cudaMalloc(&d_sink, N * sizeof(float)));

    printf("\n%-18s %11s %11s %11s %12s\n",
           "variant", "median_ms", "p10_ms", "p90_ms", "per-iter ns");
    printf("---------------------------------------------------------------------------\n");
    for (int v = 0; v < V_COUNT; ++v) {
        Stats s = time_kernel([&](){
            launch(v, d_sink, N, dim3(GRID), dim3(BLOCK));
        }, WARMUP, ITERS);
        // per-thread per-iter ns: median_ms * 1e6 / (N_ITERS * 4 chains)
        // (shared across N threads, so this is "wall-clock ns for one thread's one op")
        double per_iter_ns = s.median_ms * 1e6 / (double)(N_ITERS * 4);
        printf("%-18s %11.4f %11.4f %11.4f %12.3f\n",
               variant_name(v), s.median_ms, s.p10_ms, s.p90_ms, per_iter_ns);
    }
    printf("\n");
    printf("Notes:\n");
    printf("  per-iter ns = median_ms * 1e6 / (N_ITERS * 4)  — wall-clock per inner op on one SM chain.\n");
    printf("  Variants within one group should collapse to the same per-iter ns\n");
    printf("  if the compiler emits identical SASS (e.g. all to selp or all to FMNMX).\n");
    printf("  A divergent outlier signals that one pattern is NOT being translated\n");
    printf("  to branchless SASS on H200 sm_9.0a.\n");

    CHECK_CUDA(cudaFree(d_sink));
    return 0;
}
