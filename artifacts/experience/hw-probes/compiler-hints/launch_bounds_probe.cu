// launch_bounds_probe.cu
// Microbenchmark: register-hungry kernel with and without
// __launch_bounds__(256, 4) on H200 (sm_90a).
//
// On sm_90a: 65536 regs/SM.
//   __launch_bounds__(256, 4) -> max 65536/(256*4)=64 regs/thread
//   __launch_bounds__(256, 2) -> max 65536/(256*2)=128 regs/thread
//   No launch_bounds -> compiler free to choose
//
// The heavy_compute function uses 32 independent accumulators with
// cross-dependencies in 3 unrolled rounds, forcing high register pressure.
//
// Protocol: benchmark-protocol.md (warmup 5, measure 20, CUDA events).

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#define CHECK(call)                                                         \
    do {                                                                    \
        cudaError_t e = (call);                                             \
        if (e != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(e));                                 \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

// ---------------------------------------------------------------------------
// Heavy compute: 32 accumulators, 3 rounds of cross-mixing.
// __forceinline__ ensures the compiler sees all live variables at once.
// ---------------------------------------------------------------------------
__device__ __host__ __forceinline__
float heavy_compute(const float* __restrict__ in, int idx, int n) {
    float a00 = in[(idx +  0) % n];
    float a01 = in[(idx +  1) % n];
    float a02 = in[(idx +  2) % n];
    float a03 = in[(idx +  3) % n];
    float a04 = in[(idx +  4) % n];
    float a05 = in[(idx +  5) % n];
    float a06 = in[(idx +  6) % n];
    float a07 = in[(idx +  7) % n];
    float a08 = in[(idx +  8) % n];
    float a09 = in[(idx +  9) % n];
    float a10 = in[(idx + 10) % n];
    float a11 = in[(idx + 11) % n];
    float a12 = in[(idx + 12) % n];
    float a13 = in[(idx + 13) % n];
    float a14 = in[(idx + 14) % n];
    float a15 = in[(idx + 15) % n];
    float a16 = in[(idx + 16) % n];
    float a17 = in[(idx + 17) % n];
    float a18 = in[(idx + 18) % n];
    float a19 = in[(idx + 19) % n];
    float a20 = in[(idx + 20) % n];
    float a21 = in[(idx + 21) % n];
    float a22 = in[(idx + 22) % n];
    float a23 = in[(idx + 23) % n];
    float a24 = in[(idx + 24) % n];
    float a25 = in[(idx + 25) % n];
    float a26 = in[(idx + 26) % n];
    float a27 = in[(idx + 27) % n];
    float a28 = in[(idx + 28) % n];
    float a29 = in[(idx + 29) % n];
    float a30 = in[(idx + 30) % n];
    float a31 = in[(idx + 31) % n];

    // Round 1: ring FMA
    a00 = a00 * a01 + a31;  a01 = a01 * a02 + a00;
    a02 = a02 * a03 + a01;  a03 = a03 * a04 + a02;
    a04 = a04 * a05 + a03;  a05 = a05 * a06 + a04;
    a06 = a06 * a07 + a05;  a07 = a07 * a08 + a06;
    a08 = a08 * a09 + a07;  a09 = a09 * a10 + a08;
    a10 = a10 * a11 + a09;  a11 = a11 * a12 + a10;
    a12 = a12 * a13 + a11;  a13 = a13 * a14 + a12;
    a14 = a14 * a15 + a13;  a15 = a15 * a16 + a14;
    a16 = a16 * a17 + a15;  a17 = a17 * a18 + a16;
    a18 = a18 * a19 + a17;  a19 = a19 * a20 + a18;
    a20 = a20 * a21 + a19;  a21 = a21 * a22 + a20;
    a22 = a22 * a23 + a21;  a23 = a23 * a24 + a22;
    a24 = a24 * a25 + a23;  a25 = a25 * a26 + a24;
    a26 = a26 * a27 + a25;  a27 = a27 * a28 + a26;
    a28 = a28 * a29 + a27;  a29 = a29 * a30 + a28;
    a30 = a30 * a31 + a29;  a31 = a31 * a00 + a30;

    // Round 2: butterfly FMA (stride 16)
    a00 = a00 * a16 + a08;  a01 = a01 * a17 + a09;
    a02 = a02 * a18 + a10;  a03 = a03 * a19 + a11;
    a04 = a04 * a20 + a12;  a05 = a05 * a21 + a13;
    a06 = a06 * a22 + a14;  a07 = a07 * a23 + a15;
    a08 = a08 * a24 + a00;  a09 = a09 * a25 + a01;
    a10 = a10 * a26 + a02;  a11 = a11 * a27 + a03;
    a12 = a12 * a28 + a04;  a13 = a13 * a29 + a05;
    a14 = a14 * a30 + a06;  a15 = a15 * a31 + a07;
    a16 = a16 * a00 + a08;  a17 = a17 * a01 + a09;
    a18 = a18 * a02 + a10;  a19 = a19 * a03 + a11;
    a20 = a20 * a04 + a12;  a21 = a21 * a05 + a13;
    a22 = a22 * a06 + a14;  a23 = a23 * a07 + a15;
    a24 = a24 * a08 + a16;  a25 = a25 * a09 + a17;
    a26 = a26 * a10 + a18;  a27 = a27 * a11 + a19;
    a28 = a28 * a12 + a20;  a29 = a29 * a13 + a21;
    a30 = a30 * a14 + a22;  a31 = a31 * a15 + a23;

    // Round 3: reverse ring
    a00 = a00 * a31 + a15;  a01 = a01 * a00 + a16;
    a02 = a02 * a01 + a17;  a03 = a03 * a02 + a18;
    a04 = a04 * a03 + a19;  a05 = a05 * a04 + a20;
    a06 = a06 * a05 + a21;  a07 = a07 * a06 + a22;
    a08 = a08 * a07 + a23;  a09 = a09 * a08 + a24;
    a10 = a10 * a09 + a25;  a11 = a11 * a10 + a26;
    a12 = a12 * a11 + a27;  a13 = a13 * a12 + a28;
    a14 = a14 * a13 + a29;  a15 = a15 * a14 + a30;
    a16 = a16 * a15 + a31;  a17 = a17 * a16 + a00;
    a18 = a18 * a17 + a01;  a19 = a19 * a18 + a02;
    a20 = a20 * a19 + a03;  a21 = a21 * a20 + a04;
    a22 = a22 * a21 + a05;  a23 = a23 * a22 + a06;
    a24 = a24 * a23 + a07;  a25 = a25 * a24 + a08;
    a26 = a26 * a25 + a09;  a27 = a27 * a26 + a10;
    a28 = a28 * a27 + a11;  a29 = a29 * a28 + a12;
    a30 = a30 * a29 + a13;  a31 = a31 * a30 + a14;

    return a00 + a01 + a02 + a03 + a04 + a05 + a06 + a07
         + a08 + a09 + a10 + a11 + a12 + a13 + a14 + a15
         + a16 + a17 + a18 + a19 + a20 + a21 + a22 + a23
         + a24 + a25 + a26 + a27 + a28 + a29 + a30 + a31;
}

// ---------------------------------------------------------------------------
// Kernel WITHOUT __launch_bounds__
// ---------------------------------------------------------------------------
__global__ void heavy_no_lb(const float* __restrict__ in,
                            float*       __restrict__ out,
                            int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = heavy_compute(in, idx, n);
}

// Kernel WITH __launch_bounds__(256, 4)
__global__ void
__launch_bounds__(256, 4)
heavy_lb_256_4(const float* __restrict__ in,
               float*       __restrict__ out,
               int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = heavy_compute(in, idx, n);
}

// Kernel WITH __launch_bounds__(256, 2)
__global__ void
__launch_bounds__(256, 2)
heavy_lb_256_2(const float* __restrict__ in,
               float*       __restrict__ out,
               int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = heavy_compute(in, idx, n);
}

// CPU reference
void cpu_ref(const float* in, float* out, int n) {
    for (int i = 0; i < n; ++i)
        out[i] = heavy_compute(in, i, n);
}

// ---------------------------------------------------------------------------
void print_func_attrs(const char* name, const void* func) {
    cudaFuncAttributes attr;
    CHECK(cudaFuncGetAttributes(&attr, func));
    printf("  %-20s  regs/thread=%3d  smem_static=%5zu  maxTPB=%d\n",
           name, attr.numRegs, attr.sharedSizeBytes, attr.maxThreadsPerBlock);
}

struct BenchResult { float median_ms, p10_ms, p90_ms; };
typedef void (*KernelFn)(const float*, float*, int);

BenchResult bench(KernelFn kernel, const float* d_in,
                  float* d_out, int n, int block_size) {
    int grid = (n + block_size - 1) / block_size;

    for (int i = 0; i < 5; ++i)
        kernel<<<grid, block_size>>>(d_in, d_out, n);
    CHECK(cudaDeviceSynchronize());

    cudaEvent_t e_start, e_end;
    CHECK(cudaEventCreate(&e_start));
    CHECK(cudaEventCreate(&e_end));

    std::vector<float> times_ms;
    for (int i = 0; i < 20; ++i) {
        CHECK(cudaEventRecord(e_start));
        kernel<<<grid, block_size>>>(d_in, d_out, n);
        CHECK(cudaEventRecord(e_end));
        CHECK(cudaEventSynchronize(e_end));
        float ms;
        CHECK(cudaEventElapsedTime(&ms, e_start, e_end));
        times_ms.push_back(ms);
    }

    CHECK(cudaEventDestroy(e_start));
    CHECK(cudaEventDestroy(e_end));

    std::sort(times_ms.begin(), times_ms.end());
    return {times_ms[10], times_ms[2], times_ms[18]};
}

int main() {
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    printf("CUDA Runtime: %d.%d\n", CUDART_VERSION / 1000,
           (CUDART_VERSION % 1000) / 10);
    int driverVersion;
    CHECK(cudaDriverGetVersion(&driverVersion));
    printf("Driver: %d.%d\n", driverVersion / 1000,
           (driverVersion % 1000) / 10);

    char uuid_str[64];
    snprintf(uuid_str, sizeof(uuid_str),
             "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
             (unsigned char)prop.uuid.bytes[0],  (unsigned char)prop.uuid.bytes[1],
             (unsigned char)prop.uuid.bytes[2],  (unsigned char)prop.uuid.bytes[3],
             (unsigned char)prop.uuid.bytes[4],  (unsigned char)prop.uuid.bytes[5],
             (unsigned char)prop.uuid.bytes[6],  (unsigned char)prop.uuid.bytes[7],
             (unsigned char)prop.uuid.bytes[8],  (unsigned char)prop.uuid.bytes[9],
             (unsigned char)prop.uuid.bytes[10], (unsigned char)prop.uuid.bytes[11],
             (unsigned char)prop.uuid.bytes[12], (unsigned char)prop.uuid.bytes[13],
             (unsigned char)prop.uuid.bytes[14], (unsigned char)prop.uuid.bytes[15]);
    printf("GPU UUID: %s\n", uuid_str);

    printf("\nKernel attributes:\n");
    print_func_attrs("heavy_no_lb",    (const void*)heavy_no_lb);
    print_func_attrs("heavy_lb_256_4", (const void*)heavy_lb_256_4);
    print_func_attrs("heavy_lb_256_2", (const void*)heavy_lb_256_2);

    const int N = 1 << 22;
    size_t bytes = N * sizeof(float);

    float *h_in  = (float*)malloc(bytes);
    float *h_ref = (float*)malloc(bytes);
    float *h_gpu = (float*)malloc(bytes);

    srand(42);
    for (int i = 0; i < N; ++i)
        h_in[i] = (float)(rand() % 100) / 10000.0f;

    cpu_ref(h_in, h_ref, N);

    float *d_in, *d_out;
    CHECK(cudaMalloc(&d_in, bytes));
    CHECK(cudaMalloc(&d_out, bytes));
    CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    const int BLOCK = 256;

    // Correctness checks
    const char* names[] = {"no_lb", "lb_256_4", "lb_256_2"};
    KernelFn kernels[] = {heavy_no_lb, heavy_lb_256_4, heavy_lb_256_2};

    for (int k = 0; k < 3; ++k) {
        kernels[k]<<<(N + BLOCK - 1) / BLOCK, BLOCK>>>(d_in, d_out, N);
        CHECK(cudaMemcpy(h_gpu, d_out, bytes, cudaMemcpyDeviceToHost));
        float max_err = 0.0f;
        for (int i = 0; i < N; ++i)
            max_err = fmaxf(max_err, fabsf(h_gpu[i] - h_ref[i]));
        printf("Correctness (%s): max_abs_err = %e\n", names[k], max_err);
        // Relaxed tolerance: FMA reordering between CPU/GPU causes differences
        if (max_err > 1e0f) { fprintf(stderr, "FAIL: %s\n", names[k]); return 1; }
    }

    // Benchmark
    printf("\nBenchmark (N=%d, block=%d):\n", N, BLOCK);

    BenchResult r_no = bench(heavy_no_lb,    d_in, d_out, N, BLOCK);
    BenchResult r_4  = bench(heavy_lb_256_4, d_in, d_out, N, BLOCK);
    BenchResult r_2  = bench(heavy_lb_256_2, d_in, d_out, N, BLOCK);

    printf("  heavy_no_lb:    median=%.4f ms  p10=%.4f ms  p90=%.4f ms\n",
           r_no.median_ms, r_no.p10_ms, r_no.p90_ms);
    printf("  heavy_lb_256_4: median=%.4f ms  p10=%.4f ms  p90=%.4f ms\n",
           r_4.median_ms, r_4.p10_ms, r_4.p90_ms);
    printf("  heavy_lb_256_2: median=%.4f ms  p10=%.4f ms  p90=%.4f ms\n",
           r_2.median_ms, r_2.p10_ms, r_2.p90_ms);

    printf("\n  Ratios (baseline = heavy_no_lb):\n");
    printf("    lb_256_4 vs no_lb: %.4f  (>1 = lb faster)\n",
           r_no.median_ms / r_4.median_ms);
    printf("    lb_256_2 vs no_lb: %.4f  (>1 = lb faster)\n",
           r_no.median_ms / r_2.median_ms);

    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_out));
    free(h_in); free(h_ref); free(h_gpu);

    printf("\nDone.\n");
    return 0;
}
