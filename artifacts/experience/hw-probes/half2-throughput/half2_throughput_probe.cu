// half2_throughput_probe.cu -- FMA throughput across five precision
// variants on H200 sm_9.0a: fp32 scalar, fp16 scalar, fp16 packed
// (__hfma2), bf16 scalar, bf16 packed. Compute-bound harness using 4
// independent accumulator chains (to hide FMA pipeline latency, like
// the ILP skill), with N_FMA_ITERS inner FMAs per chain.
//
// Target: H200 (sm_90a), CUDA 12.9. Backs
// 30-skill/compute/half-precision-math/ and answers the migration plan's open
// question "bf16 packed op coverage on sm_9.0a".
//
// We report:
//   - median wall-clock per launch
//   - "scalar-equivalent GFLOPS": (4 chains * N_FMA_ITERS * 2 FLOPs/fma
//     * lanes_per_instr) * threads / time. Packed variants report 2x
//     because each instruction does 2 FP ops.
//
// A secondary goal is to log which bf16 intrinsics actually compile
// and execute on sm_9.0a (P6 of the legacy pitfalls is the coverage
// concern). All variants compile if this binary compiles; runtime
// non-zero values in the output confirm execution.

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define CHECK_CUDA(call) do {                                       \
    cudaError_t err = (call);                                       \
    if (err != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                \
                __FILE__, __LINE__, cudaGetErrorString(err));       \
        exit(1);                                                    \
    }                                                               \
} while (0)

constexpr int N_FMA_ITERS = 2048;        // FMAs per chain
constexpr int N_CHAINS    = 4;           // independent accumulator chains (ILP)

// -------------------------------------------------------------------------
// Variant encoding.
// -------------------------------------------------------------------------
enum : int {
    V_FP32          = 0,   // float, fmaf
    V_FP16_SCALAR   = 1,   // __half, __hfma
    V_FP16_PACKED   = 2,   // __half2, __hfma2  (2 ops/instr)
    V_BF16_SCALAR   = 3,   // __nv_bfloat16, __hfma
    V_BF16_PACKED   = 4,   // __nv_bfloat162, __hfma2  (2 ops/instr)
    V_COUNT         = 5
};

static const char* variant_name(int v) {
    switch (v) {
        case V_FP32:        return "fp32-scalar     ";
        case V_FP16_SCALAR: return "fp16-scalar     ";
        case V_FP16_PACKED: return "fp16-packed(h2) ";
        case V_BF16_SCALAR: return "bf16-scalar     ";
        case V_BF16_PACKED: return "bf16-packed(h2) ";
    }
    return "?";
}

static int lanes_per_instr(int v) {
    return (v == V_FP16_PACKED || v == V_BF16_PACKED) ? 2 : 1;
}

// -------------------------------------------------------------------------
// Kernels (one per variant). Pattern: 4 parallel FMA chains for ILP;
// N_FMA_ITERS inner iterations to make the kernel compute-bound.
// -------------------------------------------------------------------------
__global__ void fma_bench_fp32(float* out, size_t N) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    float a  = 1.0f + __uint_as_float((unsigned)tid & 0xFFFF);
    float b  = 0.9999f;
    float c0 = 0.1f, c1 = 0.2f, c2 = 0.3f, c3 = 0.4f;
    #pragma unroll 1
    for (int i = 0; i < N_FMA_ITERS; ++i) {
        c0 = fmaf(a, b, c0);
        c1 = fmaf(a, b, c1);
        c2 = fmaf(a, b, c2);
        c3 = fmaf(a, b, c3);
    }
    out[tid] = c0 + c1 + c2 + c3;
}

__global__ void fma_bench_fp16_scalar(__half* out, size_t N) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    __half a  = __float2half(1.0001f);
    __half b  = __float2half(0.9999f);
    __half c0 = __float2half(0.1f), c1 = __float2half(0.2f),
           c2 = __float2half(0.3f), c3 = __float2half(0.4f);
    #pragma unroll 1
    for (int i = 0; i < N_FMA_ITERS; ++i) {
        c0 = __hfma(a, b, c0);
        c1 = __hfma(a, b, c1);
        c2 = __hfma(a, b, c2);
        c3 = __hfma(a, b, c3);
    }
    out[tid] = __hadd(__hadd(c0, c1), __hadd(c2, c3));
}

__global__ void fma_bench_fp16_packed(__half2* out, size_t N) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    __half2 a  = __floats2half2_rn(1.0001f, 1.0001f);
    __half2 b  = __floats2half2_rn(0.9999f, 0.9999f);
    __half2 c0 = __floats2half2_rn(0.1f, 0.5f),
            c1 = __floats2half2_rn(0.2f, 0.6f),
            c2 = __floats2half2_rn(0.3f, 0.7f),
            c3 = __floats2half2_rn(0.4f, 0.8f);
    #pragma unroll 1
    for (int i = 0; i < N_FMA_ITERS; ++i) {
        c0 = __hfma2(a, b, c0);
        c1 = __hfma2(a, b, c1);
        c2 = __hfma2(a, b, c2);
        c3 = __hfma2(a, b, c3);
    }
    out[tid] = __hadd2(__hadd2(c0, c1), __hadd2(c2, c3));
}

__global__ void fma_bench_bf16_scalar(__nv_bfloat16* out, size_t N) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    __nv_bfloat16 a  = __float2bfloat16(1.0001f);
    __nv_bfloat16 b  = __float2bfloat16(0.9999f);
    __nv_bfloat16 c0 = __float2bfloat16(0.1f), c1 = __float2bfloat16(0.2f),
                  c2 = __float2bfloat16(0.3f), c3 = __float2bfloat16(0.4f);
    #pragma unroll 1
    for (int i = 0; i < N_FMA_ITERS; ++i) {
        c0 = __hfma(a, b, c0);
        c1 = __hfma(a, b, c1);
        c2 = __hfma(a, b, c2);
        c3 = __hfma(a, b, c3);
    }
    out[tid] = __hadd(__hadd(c0, c1), __hadd(c2, c3));
}

__global__ void fma_bench_bf16_packed(__nv_bfloat162* out, size_t N) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    __nv_bfloat162 a  = __floats2bfloat162_rn(1.0001f, 1.0001f);
    __nv_bfloat162 b  = __floats2bfloat162_rn(0.9999f, 0.9999f);
    __nv_bfloat162 c0 = __floats2bfloat162_rn(0.1f, 0.5f),
                   c1 = __floats2bfloat162_rn(0.2f, 0.6f),
                   c2 = __floats2bfloat162_rn(0.3f, 0.7f),
                   c3 = __floats2bfloat162_rn(0.4f, 0.8f);
    #pragma unroll 1
    for (int i = 0; i < N_FMA_ITERS; ++i) {
        c0 = __hfma2(a, b, c0);
        c1 = __hfma2(a, b, c1);
        c2 = __hfma2(a, b, c2);
        c3 = __hfma2(a, b, c3);
    }
    out[tid] = __hadd2(__hadd2(c0, c1), __hadd2(c2, c3));
}

// -------------------------------------------------------------------------
// Timing harness.
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

int main() {
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("Device    : %s (sm %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("N_FMA_ITERS=%d, N_CHAINS=%d\n", N_FMA_ITERS, N_CHAINS);

    // One element per thread. Large enough grid to saturate SMs.
    const int BLOCK = 256;
    const int GRID  = 132 * 4;        // 132 SMs x 4 blocks
    const size_t N  = (size_t)GRID * BLOCK;     // 135,168 threads

    const int WARMUP = 5;
    const int ITERS  = 20;

    // Allocate output for the widest type (bf16_packed is 4 B/elem).
    size_t out_bytes = N * sizeof(__half2);   // 4 B/elem covers all variants
    void* d_out = nullptr;
    CHECK_CUDA(cudaMalloc(&d_out, out_bytes));

    printf("\n%-20s %11s %11s %11s %12s %12s\n",
           "variant", "median_ms", "p10_ms", "p90_ms",
           "scalar GFLOPS", "per-thread");
    printf("-------------------------------------------------------------------------------------\n");

    // Total FLOPs (scalar) per launch:
    //   scalar variants : N_CHAINS * N_FMA_ITERS * 2 * N
    //   packed variants : N_CHAINS * N_FMA_ITERS * 2 * 2 * N   (2 lanes/instr)
    double base_flops = (double)N_CHAINS * N_FMA_ITERS * 2 * N;

    for (int v = 0; v < V_COUNT; ++v) {
        Stats s = time_kernel([&]() {
            switch (v) {
                case V_FP32:
                    fma_bench_fp32       <<<GRID, BLOCK>>>((float*)d_out, N); break;
                case V_FP16_SCALAR:
                    fma_bench_fp16_scalar<<<GRID, BLOCK>>>((__half*)d_out, N); break;
                case V_FP16_PACKED:
                    fma_bench_fp16_packed<<<GRID, BLOCK>>>((__half2*)d_out, N); break;
                case V_BF16_SCALAR:
                    fma_bench_bf16_scalar<<<GRID, BLOCK>>>((__nv_bfloat16*)d_out, N); break;
                case V_BF16_PACKED:
                    fma_bench_bf16_packed<<<GRID, BLOCK>>>((__nv_bfloat162*)d_out, N); break;
            }
        }, WARMUP, ITERS);

        double flops = base_flops * lanes_per_instr(v);
        double gflops = flops / (s.median_ms * 1e-3) / 1e9;
        double per_thread_ns = s.median_ms * 1e6 / N;
        printf("%-20s %11.4f %11.4f %11.4f %12.1f %12.2f\n",
               variant_name(v), s.median_ms, s.p10_ms, s.p90_ms,
               gflops, per_thread_ns);
    }

    printf("\n");
    printf("Notes:\n");
    printf("  Each thread runs %d chains x %d FMAs = %d FMAs.\n",
           N_CHAINS, N_FMA_ITERS, N_CHAINS * N_FMA_ITERS);
    printf("  Packed variants (h2) execute 2 FP ops per instruction;\n");
    printf("  'scalar GFLOPS' counts SCALAR ops (2 per packed instr) so\n");
    printf("  packed should ~2x the rate of its scalar counterpart.\n");

    CHECK_CUDA(cudaFree(d_out));
    return 0;
}
