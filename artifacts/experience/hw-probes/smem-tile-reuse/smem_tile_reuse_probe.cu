// smem_tile_reuse_probe.cu -- Microbenchmark: naive (global-only) matrix
// transpose vs shared-memory-staged transpose, with and without bank-conflict
// padding. Validates S2 ("coalescing transform via smem") from
// knowledge/30-skill/memory/shared-memory-cache/skill.md.
//
// Target: H200 (sm_90a), CUDA 12.9
//
// Three kernels, same problem (NxN float matrix transpose):
//   A. naive_transpose       : direct gmem read + transposed gmem write
//                              (write side is non-coalesced)
//   B. smem_tiled_conflict   : [TILE][TILE] smem tile (32-way bank conflicts
//                              on column read)
//   C. smem_tiled_padded     : [TILE][TILE+1] smem tile (conflict-free)
//
// Protocol: warmup 5, measure 20, CUDA events, median/p10/p90.
// Correctness check vs a CPU reference on a small subset.

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

constexpr int TILE = 32;

// ---------------------------------------------------------------------------
// Kernel A. Naive: gmem read + transposed gmem write. The write is
// non-coalesced (32 threads of a warp write to stride-N addresses).
// ---------------------------------------------------------------------------
__global__ void naive_transpose(const float* __restrict__ in,
                                float* __restrict__ out,
                                int N) {
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    if (x < N && y < N) {
        out[x * N + y] = in[y * N + x];   // coalesced read, non-coalesced write
    }
}

// ---------------------------------------------------------------------------
// Kernel B. Smem-staged transpose WITHOUT padding. Row reads and column
// reads both exist; the column read suffers 32-way bank conflicts.
// ---------------------------------------------------------------------------
__global__ void smem_tiled_conflict(const float* __restrict__ in,
                                    float* __restrict__ out,
                                    int N) {
    __shared__ float smem[TILE][TILE];      // no padding -> bank conflicts
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    if (x < N && y < N)
        smem[threadIdx.y][threadIdx.x] = in[y * N + x];   // coalesced load
    __syncthreads();
    x = blockIdx.y * TILE + threadIdx.x;    // swap block coords
    y = blockIdx.x * TILE + threadIdx.y;
    if (x < N && y < N)
        out[y * N + x] = smem[threadIdx.x][threadIdx.y];  // coalesced store,
                                                          // but column smem
                                                          // read hits banks
}

// ---------------------------------------------------------------------------
// Kernel C. Smem-staged transpose WITH [TILE][TILE+1] padding to break
// the 32-way column bank conflict.
// ---------------------------------------------------------------------------
__global__ void smem_tiled_padded(const float* __restrict__ in,
                                  float* __restrict__ out,
                                  int N) {
    __shared__ float smem[TILE][TILE + 1];  // +1 padding
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    if (x < N && y < N)
        smem[threadIdx.y][threadIdx.x] = in[y * N + x];
    __syncthreads();
    x = blockIdx.y * TILE + threadIdx.x;
    y = blockIdx.x * TILE + threadIdx.y;
    if (x < N && y < N)
        out[y * N + x] = smem[threadIdx.x][threadIdx.y];
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

int main(int argc, char** argv) {
    const int N      = (argc > 1) ? std::atoi(argv[1]) : 4096;  // 4096x4096 fp32 = 64 MB per matrix
    const int WARMUP = 5;
    const int ITERS  = 20;

    if (N % TILE != 0) {
        fprintf(stderr, "N must be a multiple of TILE=%d (got %d)\n", TILE, N);
        return 1;
    }

    size_t bytes = (size_t)N * N * sizeof(float);
    printf("Problem size : %d x %d fp32 (%.1f MB per matrix)\n", N, N, bytes / 1048576.0);
    printf("Tile         : %d x %d\n", TILE, TILE);
    printf("Warmup/iters : %d / %d\n", WARMUP, ITERS);
    printf("\n");

    // Host input: deterministic pattern so row-major index is recoverable.
    std::vector<float> h_in((size_t)N * N);
    for (size_t i = 0; i < h_in.size(); ++i) h_in[i] = (float)(i % 997);

    float *d_in = nullptr, *d_out = nullptr;
    CHECK_CUDA(cudaMalloc(&d_in,  bytes));
    CHECK_CUDA(cudaMalloc(&d_out, bytes));
    CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid(N / TILE, N / TILE);

    // ---- Correctness check: sample 1024 random positions and verify transpose.
    auto check = [&](const char* name) {
        std::vector<float> h_out((size_t)N * N);
        CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
        int errors = 0;
        for (int s = 0; s < 1024; ++s) {
            int y = (int)(((uint64_t)s * 73856093u) % (uint64_t)N);
            int x = (int)(((uint64_t)s * 19349663u) % (uint64_t)N);
            if (h_out[(size_t)x * N + y] != h_in[(size_t)y * N + x]) {
                errors++;
                if (errors <= 3) {
                    fprintf(stderr, "  %s mismatch at (y=%d,x=%d): got %g, want %g\n",
                            name, y, x, h_out[(size_t)x * N + y], h_in[(size_t)y * N + x]);
                }
            }
        }
        printf("%-22s correctness: %s (errors=%d / 1024 samples)\n",
               name, errors == 0 ? "OK" : "FAIL", errors);
    };

    CHECK_CUDA(cudaMemset(d_out, 0, bytes));
    naive_transpose<<<grid, block>>>(d_in, d_out, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    check("naive_transpose");

    CHECK_CUDA(cudaMemset(d_out, 0, bytes));
    smem_tiled_conflict<<<grid, block>>>(d_in, d_out, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    check("smem_tiled_conflict");

    CHECK_CUDA(cudaMemset(d_out, 0, bytes));
    smem_tiled_padded<<<grid, block>>>(d_in, d_out, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    check("smem_tiled_padded");

    printf("\n");

    // ---- Timing ----
    Stats s_naive    = time_kernel([&](){ naive_transpose<<<grid, block>>>(d_in, d_out, N); },
                                   WARMUP, ITERS);
    Stats s_conflict = time_kernel([&](){ smem_tiled_conflict<<<grid, block>>>(d_in, d_out, N); },
                                   WARMUP, ITERS);
    Stats s_padded   = time_kernel([&](){ smem_tiled_padded<<<grid, block>>>(d_in, d_out, N); },
                                   WARMUP, ITERS);

    auto bw = [&](float ms) {
        // effective GB/s = 2 * N*N*sizeof(float) (read + write) / ms
        return 2.0 * (double)N * N * sizeof(float) / (ms * 1.0e-3) / 1.0e9;
    };

    printf("Kernel                  median_ms   p10_ms    p90_ms    eff_BW_GB/s\n");
    printf("--------------------------------------------------------------------\n");
    printf("naive_transpose        %9.4f %9.4f %9.4f   %8.2f\n",
           s_naive.median_ms, s_naive.p10_ms, s_naive.p90_ms, bw(s_naive.median_ms));
    printf("smem_tiled_conflict    %9.4f %9.4f %9.4f   %8.2f\n",
           s_conflict.median_ms, s_conflict.p10_ms, s_conflict.p90_ms, bw(s_conflict.median_ms));
    printf("smem_tiled_padded      %9.4f %9.4f %9.4f   %8.2f\n",
           s_padded.median_ms, s_padded.p10_ms, s_padded.p90_ms, bw(s_padded.median_ms));
    printf("\n");
    printf("Speedup conflict / naive  : %.2fx\n", s_naive.median_ms / s_conflict.median_ms);
    printf("Speedup padded   / naive  : %.2fx\n", s_naive.median_ms / s_padded.median_ms);
    printf("Speedup padded   / conflict: %.2fx\n", s_conflict.median_ms / s_padded.median_ms);

    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    return 0;
}
