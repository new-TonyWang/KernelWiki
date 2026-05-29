// aos_vs_soa_probe.cu -- Microbenchmark: Array-of-Structures vs
// Structure-of-Arrays for the "read one hot field" pattern. Validates
// sub-skill S1 from
// knowledge/30-skill/memory/layout-transform/skill.md
//
// Target: H200 (sm_90a), CUDA 12.9
//
// Struct under test: Particle { float x, y, z, vx, vy, vz } = 24 B.
// N = 2^24 = 16,777,216 elements = 384 MB in AoS, 64 MB per SoA field.
//
// Kernels, all do the same logical work: out[i] = hot_field[i] * 2.0f:
//   A. aos_read_one_field    : reads .x from AoS, stride = 24 B (6-way
//                              under-utilization of each 32-B segment)
//   B. soa_read_one_field    : reads x[] from SoA, stride = 4 B (coalesced)
//   C. soa_vectorized        : reads x[] as float4 (coalesced + 128-bit/thread)
//   D. aos_to_soa_convert    : one-time conversion kernel (reads AoS
//                              coalesced, writes 6 SoA fields coalesced)
//
// Protocol: 5 warmup + 20 timed iters, CUDA events, median/p10/p90.
// Correctness: double-precision reference on the host.

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

struct Particle {
    float x, y, z, vx, vy, vz;          // 24 B
};

// ---------------------------------------------------------------------------
// Kernel A. AoS: each thread reads the .x field only. Natural stride = 24 B.
// ---------------------------------------------------------------------------
__global__ void aos_read_one_field(const Particle* __restrict__ in,
                                   float* __restrict__ out,
                                   int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = in[i].x * 2.0f;        // only .x is used
    }
}

// ---------------------------------------------------------------------------
// Kernel B. SoA scalar: each thread reads x[i] (stride 4 B, coalesced).
// ---------------------------------------------------------------------------
__global__ void soa_read_one_field(const float* __restrict__ x,
                                   float* __restrict__ out,
                                   int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = x[i] * 2.0f;
    }
}

// ---------------------------------------------------------------------------
// Kernel C. SoA vectorized: each thread reads a float4 (4 elements), so the
// kernel issues N/4 loads. Stride = 16 B per thread (coalesced 128-B warp).
// ---------------------------------------------------------------------------
__global__ void soa_vectorized(const float4* __restrict__ x4,
                               float4* __restrict__ out4,
                               int n4) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4) {
        float4 v = x4[i];
        v.x *= 2.0f; v.y *= 2.0f; v.z *= 2.0f; v.w *= 2.0f;
        out4[i] = v;
    }
}

// ---------------------------------------------------------------------------
// Kernel D. AoS -> SoA conversion. Each thread reads one 24-B struct
// (coalesced), writes into 6 SoA arrays (each stride-1 coalesced).
// ---------------------------------------------------------------------------
__global__ void aos_to_soa_convert(const Particle* __restrict__ in,
                                   float* __restrict__ x,
                                   float* __restrict__ y,
                                   float* __restrict__ z,
                                   float* __restrict__ vx,
                                   float* __restrict__ vy,
                                   float* __restrict__ vz,
                                   int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        Particle p = in[i];
        x[i]  = p.x;   y[i]  = p.y;   z[i]  = p.z;
        vx[i] = p.vx;  vy[i] = p.vy;  vz[i] = p.vz;
    }
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
    const int N      = (argc > 1) ? std::atoi(argv[1]) : (16 * 1024 * 1024);
    const int BLOCK  = 256;
    const int WARMUP = 5;
    const int ITERS  = 20;

    if (N % 4 != 0) {
        fprintf(stderr, "N must be divisible by 4 (got %d)\n", N);
        return 1;
    }

    size_t aos_bytes = (size_t)N * sizeof(Particle);
    size_t soa_bytes = (size_t)N * sizeof(float);
    printf("Problem size : N = %d particles  AoS=%.1f MB  SoA/field=%.1f MB\n",
           N, aos_bytes / 1048576.0, soa_bytes / 1048576.0);
    printf("Block size   : %d\n", BLOCK);
    printf("Warmup/iters : %d / %d\n", WARMUP, ITERS);
    printf("\n");

    // Host data
    std::vector<Particle> h_aos(N);
    for (int i = 0; i < N; ++i) {
        h_aos[i].x  = (float)(i % 1009);
        h_aos[i].y  = (float)(i % 1013) * 0.5f;
        h_aos[i].z  = (float)(i % 1019) * 0.25f;
        h_aos[i].vx = h_aos[i].vy = h_aos[i].vz = 0.0f;
    }

    // AoS device buffer
    Particle* d_aos = nullptr;
    CHECK_CUDA(cudaMalloc(&d_aos, aos_bytes));
    CHECK_CUDA(cudaMemcpy(d_aos, h_aos.data(), aos_bytes, cudaMemcpyHostToDevice));

    // SoA device buffers (6 fields; only x is exercised by read kernels)
    float *d_x = nullptr, *d_y = nullptr, *d_z = nullptr;
    float *d_vx = nullptr, *d_vy = nullptr, *d_vz = nullptr;
    CHECK_CUDA(cudaMalloc(&d_x,  soa_bytes));
    CHECK_CUDA(cudaMalloc(&d_y,  soa_bytes));
    CHECK_CUDA(cudaMalloc(&d_z,  soa_bytes));
    CHECK_CUDA(cudaMalloc(&d_vx, soa_bytes));
    CHECK_CUDA(cudaMalloc(&d_vy, soa_bytes));
    CHECK_CUDA(cudaMalloc(&d_vz, soa_bytes));

    // Output buffer
    float* d_out = nullptr;
    CHECK_CUDA(cudaMalloc(&d_out, soa_bytes));

    int grid = (N + BLOCK - 1) / BLOCK;
    int grid4 = (N / 4 + BLOCK - 1) / BLOCK;

    // ---- Correctness check ----
    auto check = [&](const char* name) {
        std::vector<float> h_out(N);
        CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, soa_bytes, cudaMemcpyDeviceToHost));
        int errs = 0;
        for (int i = 0; i < N; i += (N / 1024)) {
            float want = h_aos[i].x * 2.0f;
            if (std::abs(h_out[i] - want) > 1e-4f) {
                errs++;
                if (errs <= 3) fprintf(stderr, "  %s mismatch at %d: got %g, want %g\n",
                                        name, i, h_out[i], want);
            }
        }
        printf("%-24s correctness: %s (errors=%d / 1024 samples)\n",
               name, errs == 0 ? "OK" : "FAIL", errs);
    };

    // First, run the AoS-to-SoA converter once to populate the SoA buffers.
    aos_to_soa_convert<<<grid, BLOCK>>>(d_aos, d_x, d_y, d_z, d_vx, d_vy, d_vz, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemset(d_out, 0, soa_bytes));
    aos_read_one_field<<<grid, BLOCK>>>(d_aos, d_out, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    check("aos_read_one_field");

    CHECK_CUDA(cudaMemset(d_out, 0, soa_bytes));
    soa_read_one_field<<<grid, BLOCK>>>(d_x, d_out, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    check("soa_read_one_field");

    CHECK_CUDA(cudaMemset(d_out, 0, soa_bytes));
    soa_vectorized<<<grid4, BLOCK>>>(reinterpret_cast<const float4*>(d_x),
                                     reinterpret_cast<float4*>(d_out),
                                     N / 4);
    CHECK_CUDA(cudaDeviceSynchronize());
    check("soa_vectorized");

    printf("\n");

    // ---- Timing ----
    Stats s_aos = time_kernel([&](){ aos_read_one_field<<<grid, BLOCK>>>(d_aos, d_out, N); },
                              WARMUP, ITERS);
    Stats s_soa = time_kernel([&](){ soa_read_one_field<<<grid, BLOCK>>>(d_x, d_out, N); },
                              WARMUP, ITERS);
    Stats s_soa4 = time_kernel([&](){ soa_vectorized<<<grid4, BLOCK>>>(
                                        reinterpret_cast<const float4*>(d_x),
                                        reinterpret_cast<float4*>(d_out),
                                        N / 4); },
                               WARMUP, ITERS);
    Stats s_conv = time_kernel([&](){ aos_to_soa_convert<<<grid, BLOCK>>>(
                                         d_aos, d_x, d_y, d_z, d_vx, d_vy, d_vz, N); },
                               WARMUP, ITERS);

    // Effective BW. Each kernel has a different read/write volume.
    // aos: reads 32 B / thread effectively (32-B segment amortized over 6 stride-hit warps)
    // but the "useful" bytes are N * 4 in + N * 4 out. We report effective on
    // "useful bytes" (what the kernel would have moved with perfect layout).
    auto bw_useful = [&](float ms, double useful_bytes) {
        return useful_bytes / (ms * 1e-3) / 1e9;
    };
    auto bw_actual_aos = [&](float ms) {
        // AoS read: 32 B per thread due to stride; plus N*4 output write.
        return ((double)N * 32 + (double)N * 4) / (ms * 1e-3) / 1e9;
    };

    double useful_read = (double)N * 4 + (double)N * 4;      // N floats in + N floats out
    double conv_bytes  = (double)N * 24 + (double)N * 24;    // N Particles in + 6*N floats out

    printf("Kernel                    median_ms   p10_ms    p90_ms    useful_BW_GB/s  actual_BW_GB/s\n");
    printf("--------------------------------------------------------------------------------------\n");
    printf("aos_read_one_field       %9.4f %9.4f %9.4f   %11.2f   %11.2f\n",
           s_aos.median_ms, s_aos.p10_ms, s_aos.p90_ms,
           bw_useful(s_aos.median_ms, useful_read),
           bw_actual_aos(s_aos.median_ms));
    printf("soa_read_one_field       %9.4f %9.4f %9.4f   %11.2f   %11.2f\n",
           s_soa.median_ms, s_soa.p10_ms, s_soa.p90_ms,
           bw_useful(s_soa.median_ms, useful_read),
           bw_useful(s_soa.median_ms, useful_read));
    printf("soa_vectorized           %9.4f %9.4f %9.4f   %11.2f   %11.2f\n",
           s_soa4.median_ms, s_soa4.p10_ms, s_soa4.p90_ms,
           bw_useful(s_soa4.median_ms, useful_read),
           bw_useful(s_soa4.median_ms, useful_read));
    printf("aos_to_soa_convert       %9.4f %9.4f %9.4f   %11.2f   %11.2f\n",
           s_conv.median_ms, s_conv.p10_ms, s_conv.p90_ms,
           bw_useful(s_conv.median_ms, conv_bytes),
           bw_useful(s_conv.median_ms, conv_bytes));
    printf("\n");
    printf("Speedup soa / aos          : %.2fx\n", s_aos.median_ms / s_soa.median_ms);
    printf("Speedup soa4 / aos         : %.2fx\n", s_aos.median_ms / s_soa4.median_ms);
    printf("Speedup soa4 / soa         : %.2fx\n", s_soa.median_ms / s_soa4.median_ms);
    printf("\n");
    printf("Amortization: conversion costs %.4f ms. Each AoS-vs-SoA kernel saves\n",
           s_conv.median_ms);
    printf("  (aos - soa) = %.4f ms per call. Break-even after %.2f downstream calls.\n",
           s_aos.median_ms - s_soa.median_ms,
           s_conv.median_ms / (s_aos.median_ms - s_soa.median_ms));

    CHECK_CUDA(cudaFree(d_aos));
    CHECK_CUDA(cudaFree(d_x));  CHECK_CUDA(cudaFree(d_y));  CHECK_CUDA(cudaFree(d_z));
    CHECK_CUDA(cudaFree(d_vx)); CHECK_CUDA(cudaFree(d_vy)); CHECK_CUDA(cudaFree(d_vz));
    CHECK_CUDA(cudaFree(d_out));
    return 0;
}
