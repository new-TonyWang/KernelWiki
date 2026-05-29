// divergence_cost_probe.cu -- Microbenchmark: cost of warp divergence vs
// probability p that each lane takes the "other" branch. Validates the
// serialization cost model from BP Guide §13.1 and the predication
// escape hatch from §13.2, for the skill at
// knowledge/30-skill/compute/warp-divergence/
//
// Target: H200 (sm_90a), CUDA 12.9
//
// Three kernels, all do the same logical work per lane (pick one of
// two polynomial expressions depending on a per-lane predicate), each
// with a different handling strategy:
//
//   A. branch_variant      -- if/else with the two bodies large enough
//                             that nvcc will NOT predicate. Measures the
//                             serialization cost. Expected: slow at
//                             p=0.5, fast at p=0 and p=1 (one-path).
//   B. predicated_variant  -- same math expressed as ternary/fmax so
//                             the compiler emits selp/predicated code.
//                             Expected: flat across p.
//   C. warp_uniform_variant -- condition lifted to warp-uniform using
//                             __any_sync so the whole warp takes one
//                             path. Expected: as fast as branch_variant
//                             at p=0 or p=1, slower at p=0.5 (since the
//                             warp ends up doing both bodies when ANY
//                             lane needs the other).
//
// Probability sweep: p in {0.00, 0.25, 0.50, 0.75, 1.00}.
// N = 2^24 lanes = 16,777,216 threads worth of independent work.

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <algorithm>
#include <cmath>
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

// Heavy body (~1024 FMAs) that nvcc will NOT predicate. Two distinct
// polynomials so the compiler cannot fold them. Sized to make the
// kernel compute-bound on H200 so divergence cost is visible (at
// lighter loads, memory-latency hides the serialization).
__device__ __forceinline__ float path_A(float x) {
    #pragma unroll 64
    for (int i = 0; i < 512; ++i) {
        x = fmaf(x, 1.0001f, 0.5f);
        x = fmaf(x, 0.9999f, -0.25f);
    }
    return x;
}
__device__ __forceinline__ float path_B(float x) {
    #pragma unroll 64
    for (int i = 0; i < 512; ++i) {
        x = fmaf(x, 1.0003f, -0.7f);
        x = fmaf(x, 0.9997f,  0.35f);
    }
    return x;
}

// Small bodies that nvcc WILL predicate (for the predicated variant).
__device__ __forceinline__ float small_A(float x) { return fmaf(x, 1.0001f, 0.5f); }
__device__ __forceinline__ float small_B(float x) { return fmaf(x, 1.0003f, -0.7f); }

// ---------------------------------------------------------------------------
// Kernel A. Actual branch: nvcc will emit bra.uni / bra based on each
// lane's predicate. At p=0.5, every warp diverges and both bodies run.
// ---------------------------------------------------------------------------
__global__ void branch_variant(const float* __restrict__ in,
                               const uint8_t* __restrict__ take_b,
                               float* __restrict__ out,
                               int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = in[i];
    if (take_b[i]) {
        x = path_B(x);
    } else {
        x = path_A(x);
    }
    out[i] = x;
}

// ---------------------------------------------------------------------------
// Kernel B. Predicated: small-body ternary. Compiler emits selp and
// runs BOTH expressions into registers, picking the result. No
// divergence cost regardless of p.
// ---------------------------------------------------------------------------
__global__ void predicated_variant(const float* __restrict__ in,
                                   const uint8_t* __restrict__ take_b,
                                   float* __restrict__ out,
                                   int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = in[i];
    // Match total compute of branch_variant: ~1024 FMAs effective
    // per lane, but structured as straight-line code with a selp
    // at each step. All lanes do the SAME work (no divergence).
    uint8_t b_flag = take_b[i];
    #pragma unroll 32
    for (int k = 0; k < 512; ++k) {
        float a = fmaf(x, 1.0001f, 0.5f);
        float bb = fmaf(x, 1.0003f, -0.7f);
        x = b_flag ? bb : a;
        a = fmaf(x, 0.9999f, -0.25f);
        bb = fmaf(x, 0.9997f,  0.35f);
        x = b_flag ? bb : a;
    }
    out[i] = x;
}

// ---------------------------------------------------------------------------
// Kernel C. Warp-uniform: the whole warp decides once via __any_sync.
// If any lane wants path_B, the entire warp runs path_B; otherwise
// the entire warp runs path_A. Incorrect in general (only valid when
// semantics allow) but serves as the "best case of S3" benchmark.
// ---------------------------------------------------------------------------
__global__ void warp_uniform_variant(const float* __restrict__ in,
                                     const uint8_t* __restrict__ take_b,
                                     float* __restrict__ out,
                                     int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = in[i];
    int any_b = __any_sync(0xFFFFFFFFu, take_b[i]);
    if (any_b) {
        x = path_B(x);           // whole warp takes path_B
    } else {
        x = path_A(x);           // whole warp takes path_A
    }
    out[i] = x;
}

// ---------------------------------------------------------------------------
// Measurement harness
// ---------------------------------------------------------------------------
struct Stats { float median_ms, p10_ms, p90_ms, min_ms; };

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

// Fill predicate array: each lane independently takes path_B with probability p.
// Seed is fixed so runs are reproducible.
static void fill_predicate(uint8_t* h, int n, float p) {
    // Very simple LCG so we don't spin on std::mt19937 for 16M elements.
    uint32_t state = 0xDEADBEEFu;
    uint32_t threshold = (uint32_t)(p * 4294967295.0f);
    for (int i = 0; i < n; ++i) {
        state = state * 1664525u + 1013904223u;
        h[i] = (state < threshold) ? 1u : 0u;
    }
}

int main(int argc, char** argv) {
    // Smaller N than the memory-focused probes: ~1024 FMAs per lane x
    // N lanes dominates wall-clock only if the total compute volume
    // outweighs the modest memory-traffic startup cost. 1M lanes is
    // enough to saturate all SMs; more would just scale up both
    // branch and predicated times linearly.
    const int N      = (argc > 1) ? std::atoi(argv[1]) : (1024 * 1024);
    const int BLOCK  = 256;
    const int WARMUP = 5;
    const int ITERS  = 20;

    size_t bytes_f = (size_t)N * sizeof(float);
    size_t bytes_b = (size_t)N * sizeof(uint8_t);
    printf("Problem size : N = %d lanes  (%.1f MB float input + %.1f MB predicate)\n",
           N, bytes_f / 1048576.0, bytes_b / 1048576.0);
    printf("Block size   : %d\n", BLOCK);
    printf("Warmup/iters : %d / %d\n", WARMUP, ITERS);
    printf("\n");

    // Host buffers
    std::vector<float>   h_in(N);
    std::vector<uint8_t> h_pred(N);
    for (int i = 0; i < N; ++i) h_in[i] = 0.5f + (float)(i % 17) * 0.01f;

    float   *d_in  = nullptr, *d_out = nullptr;
    uint8_t *d_pred = nullptr;
    CHECK_CUDA(cudaMalloc(&d_in,   bytes_f));
    CHECK_CUDA(cudaMalloc(&d_out,  bytes_f));
    CHECK_CUDA(cudaMalloc(&d_pred, bytes_b));
    CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), bytes_f, cudaMemcpyHostToDevice));

    int grid = (N + BLOCK - 1) / BLOCK;

    const float probabilities[] = {0.00f, 0.25f, 0.50f, 0.75f, 1.00f};
    const int   n_p = sizeof(probabilities) / sizeof(probabilities[0]);

    printf("%-22s %9s %9s %9s %9s %9s\n",
           "kernel \\ p", "p=0.00", "p=0.25", "p=0.50", "p=0.75", "p=1.00");
    printf("----------------------------------------------------------------------------------\n");

    float branch_ms[n_p], pred_ms[n_p], uni_ms[n_p];

    for (int pi = 0; pi < n_p; ++pi) {
        float p = probabilities[pi];
        fill_predicate(h_pred.data(), N, p);
        CHECK_CUDA(cudaMemcpy(d_pred, h_pred.data(), bytes_b, cudaMemcpyHostToDevice));

        Stats s_branch = time_kernel([&](){
            branch_variant<<<grid, BLOCK>>>(d_in, d_pred, d_out, N);
        }, WARMUP, ITERS);
        Stats s_pred = time_kernel([&](){
            predicated_variant<<<grid, BLOCK>>>(d_in, d_pred, d_out, N);
        }, WARMUP, ITERS);
        Stats s_uni = time_kernel([&](){
            warp_uniform_variant<<<grid, BLOCK>>>(d_in, d_pred, d_out, N);
        }, WARMUP, ITERS);

        branch_ms[pi] = s_branch.median_ms;
        pred_ms[pi]   = s_pred.median_ms;
        uni_ms[pi]    = s_uni.median_ms;
    }

    printf("%-22s %9.4f %9.4f %9.4f %9.4f %9.4f\n",
           "branch_variant (ms)",
           branch_ms[0], branch_ms[1], branch_ms[2], branch_ms[3], branch_ms[4]);
    printf("%-22s %9.4f %9.4f %9.4f %9.4f %9.4f\n",
           "predicated_variant (ms)",
           pred_ms[0], pred_ms[1], pred_ms[2], pred_ms[3], pred_ms[4]);
    printf("%-22s %9.4f %9.4f %9.4f %9.4f %9.4f\n",
           "warp_uniform (ms)",
           uni_ms[0], uni_ms[1], uni_ms[2], uni_ms[3], uni_ms[4]);
    printf("\n");

    // Ratios to p=0 baseline of the same kernel (the "divergence slowdown curve")
    printf("%-22s %9s %9s %9s %9s %9s\n",
           "slowdown vs p=0", "p=0.00", "p=0.25", "p=0.50", "p=0.75", "p=1.00");
    printf("----------------------------------------------------------------------------------\n");
    printf("%-22s %9.2f %9.2f %9.2f %9.2f %9.2f\n",
           "branch_variant",
           branch_ms[0]/branch_ms[0], branch_ms[1]/branch_ms[0],
           branch_ms[2]/branch_ms[0], branch_ms[3]/branch_ms[0],
           branch_ms[4]/branch_ms[0]);
    printf("%-22s %9.2f %9.2f %9.2f %9.2f %9.2f\n",
           "predicated_variant",
           pred_ms[0]/pred_ms[0], pred_ms[1]/pred_ms[0],
           pred_ms[2]/pred_ms[0], pred_ms[3]/pred_ms[0],
           pred_ms[4]/pred_ms[0]);
    printf("%-22s %9.2f %9.2f %9.2f %9.2f %9.2f\n",
           "warp_uniform",
           uni_ms[0]/uni_ms[0], uni_ms[1]/uni_ms[0],
           uni_ms[2]/uni_ms[0], uni_ms[3]/uni_ms[0],
           uni_ms[4]/uni_ms[0]);

    printf("\nNote: predicated_variant uses short bodies so nvcc emits selp;\n");
    printf("branch_variant uses long bodies (32-FMA loops) that force real branches.\n");
    printf("warp_uniform_variant matches branch compute; whole warp picks one path via __any_sync.\n");

    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    CHECK_CUDA(cudaFree(d_pred));
    return 0;
}
