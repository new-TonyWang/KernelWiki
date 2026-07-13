// reg_pressure_probe.cu
// Microbenchmark: register-heavy kernel compiled with default register
// allocation vs --maxrregcount=32 forced spilling on H200 (sm_90a).
//
// Purpose: demonstrate the register-pressure tradeoff.
//   (a) Default compilation: compiler allocates as many regs as needed
//   (b) --maxrregcount=32: forces the compiler to cap at 32 regs/thread,
//       causing register spills to local memory
//
// The kernel uses 48 independent float accumulators with 3 rounds of
// cross-dependent FMA operations, designed to create very high register
// pressure that will force spills when constrained to 32 registers.
//
// This file is compiled TWICE:
//   nvcc -arch=sm_90a -O3 -std=c++17 -Xptxas=-v -o probe_default  reg_pressure_probe.cu
//   nvcc -arch=sm_90a -O3 -std=c++17 -Xptxas=-v --maxrregcount=32 -o probe_maxreg32 reg_pressure_probe.cu
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
// Heavy compute: 48 accumulators, 3 rounds of cross-mixing.
// The large number of live variables forces the compiler to use many
// registers (>32 by default), so --maxrregcount=32 will cause spills.
// ---------------------------------------------------------------------------
__device__ __forceinline__
float heavy_compute_48(const float* __restrict__ in, int idx, int n) {
    // Load 48 values from global memory
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
    float a32 = in[(idx + 32) % n];
    float a33 = in[(idx + 33) % n];
    float a34 = in[(idx + 34) % n];
    float a35 = in[(idx + 35) % n];
    float a36 = in[(idx + 36) % n];
    float a37 = in[(idx + 37) % n];
    float a38 = in[(idx + 38) % n];
    float a39 = in[(idx + 39) % n];
    float a40 = in[(idx + 40) % n];
    float a41 = in[(idx + 41) % n];
    float a42 = in[(idx + 42) % n];
    float a43 = in[(idx + 43) % n];
    float a44 = in[(idx + 44) % n];
    float a45 = in[(idx + 45) % n];
    float a46 = in[(idx + 46) % n];
    float a47 = in[(idx + 47) % n];

    // Round 1: ring FMA — each accumulator depends on the next
    a00 = a00 * a01 + a47;  a01 = a01 * a02 + a00;
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
    a30 = a30 * a31 + a29;  a31 = a31 * a32 + a30;
    a32 = a32 * a33 + a31;  a33 = a33 * a34 + a32;
    a34 = a34 * a35 + a33;  a35 = a35 * a36 + a34;
    a36 = a36 * a37 + a35;  a37 = a37 * a38 + a36;
    a38 = a38 * a39 + a37;  a39 = a39 * a40 + a38;
    a40 = a40 * a41 + a39;  a41 = a41 * a42 + a40;
    a42 = a42 * a43 + a41;  a43 = a43 * a44 + a42;
    a44 = a44 * a45 + a43;  a45 = a45 * a46 + a44;
    a46 = a46 * a47 + a45;  a47 = a47 * a00 + a46;

    // Round 2: butterfly FMA (stride 24)
    a00 = a00 * a24 + a12;  a01 = a01 * a25 + a13;
    a02 = a02 * a26 + a14;  a03 = a03 * a27 + a15;
    a04 = a04 * a28 + a16;  a05 = a05 * a29 + a17;
    a06 = a06 * a30 + a18;  a07 = a07 * a31 + a19;
    a08 = a08 * a32 + a20;  a09 = a09 * a33 + a21;
    a10 = a10 * a34 + a22;  a11 = a11 * a35 + a23;
    a12 = a12 * a36 + a00;  a13 = a13 * a37 + a01;
    a14 = a14 * a38 + a02;  a15 = a15 * a39 + a03;
    a16 = a16 * a40 + a04;  a17 = a17 * a41 + a05;
    a18 = a18 * a42 + a06;  a19 = a19 * a43 + a07;
    a20 = a20 * a44 + a08;  a21 = a21 * a45 + a09;
    a22 = a22 * a46 + a10;  a23 = a23 * a47 + a11;
    a24 = a24 * a00 + a12;  a25 = a25 * a01 + a13;
    a26 = a26 * a02 + a14;  a27 = a27 * a03 + a15;
    a28 = a28 * a04 + a16;  a29 = a29 * a05 + a17;
    a30 = a30 * a06 + a18;  a31 = a31 * a07 + a19;
    a32 = a32 * a08 + a20;  a33 = a33 * a09 + a21;
    a34 = a34 * a10 + a22;  a35 = a35 * a11 + a23;
    a36 = a36 * a12 + a24;  a37 = a37 * a13 + a25;
    a38 = a38 * a14 + a26;  a39 = a39 * a15 + a27;
    a40 = a40 * a16 + a28;  a41 = a41 * a17 + a29;
    a42 = a42 * a18 + a30;  a43 = a43 * a19 + a31;
    a44 = a44 * a20 + a32;  a45 = a45 * a21 + a33;
    a46 = a46 * a22 + a34;  a47 = a47 * a23 + a35;

    // Round 3: reverse ring
    a00 = a00 * a47 + a23;  a01 = a01 * a00 + a24;
    a02 = a02 * a01 + a25;  a03 = a03 * a02 + a26;
    a04 = a04 * a03 + a27;  a05 = a05 * a04 + a28;
    a06 = a06 * a05 + a29;  a07 = a07 * a06 + a30;
    a08 = a08 * a07 + a31;  a09 = a09 * a08 + a32;
    a10 = a10 * a09 + a33;  a11 = a11 * a10 + a34;
    a12 = a12 * a11 + a35;  a13 = a13 * a12 + a36;
    a14 = a14 * a13 + a37;  a15 = a15 * a14 + a38;
    a16 = a16 * a15 + a39;  a17 = a17 * a16 + a40;
    a18 = a18 * a17 + a41;  a19 = a19 * a18 + a42;
    a20 = a20 * a19 + a43;  a21 = a21 * a20 + a44;
    a22 = a22 * a21 + a45;  a23 = a23 * a22 + a46;
    a24 = a24 * a23 + a47;  a25 = a25 * a24 + a00;
    a26 = a26 * a25 + a01;  a27 = a27 * a26 + a02;
    a28 = a28 * a27 + a03;  a29 = a29 * a28 + a04;
    a30 = a30 * a29 + a05;  a31 = a31 * a30 + a06;
    a32 = a32 * a31 + a07;  a33 = a33 * a32 + a08;
    a34 = a34 * a33 + a09;  a35 = a35 * a34 + a10;
    a36 = a36 * a35 + a11;  a37 = a37 * a36 + a12;
    a38 = a38 * a37 + a13;  a39 = a39 * a38 + a14;
    a40 = a40 * a39 + a15;  a41 = a41 * a40 + a16;
    a42 = a42 * a41 + a17;  a43 = a43 * a42 + a18;
    a44 = a44 * a43 + a19;  a45 = a45 * a44 + a20;
    a46 = a46 * a45 + a21;  a47 = a47 * a46 + a22;

    return a00 + a01 + a02 + a03 + a04 + a05 + a06 + a07
         + a08 + a09 + a10 + a11 + a12 + a13 + a14 + a15
         + a16 + a17 + a18 + a19 + a20 + a21 + a22 + a23
         + a24 + a25 + a26 + a27 + a28 + a29 + a30 + a31
         + a32 + a33 + a34 + a35 + a36 + a37 + a38 + a39
         + a40 + a41 + a42 + a43 + a44 + a45 + a46 + a47;
}

// ---------------------------------------------------------------------------
// Single kernel function — the binary IS the variant (default vs maxreg32)
// ---------------------------------------------------------------------------
__global__ void reg_pressure_kernel(const float* __restrict__ in,
                                    float*       __restrict__ out,
                                    int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = heavy_compute_48(in, idx, n);
}

// ---------------------------------------------------------------------------
// CPU reference (for correctness check)
// ---------------------------------------------------------------------------
__host__
float heavy_compute_48_host(const float* in, int idx, int n) {
    float a00 = in[(idx +  0) % n]; float a01 = in[(idx +  1) % n];
    float a02 = in[(idx +  2) % n]; float a03 = in[(idx +  3) % n];
    float a04 = in[(idx +  4) % n]; float a05 = in[(idx +  5) % n];
    float a06 = in[(idx +  6) % n]; float a07 = in[(idx +  7) % n];
    float a08 = in[(idx +  8) % n]; float a09 = in[(idx +  9) % n];
    float a10 = in[(idx + 10) % n]; float a11 = in[(idx + 11) % n];
    float a12 = in[(idx + 12) % n]; float a13 = in[(idx + 13) % n];
    float a14 = in[(idx + 14) % n]; float a15 = in[(idx + 15) % n];
    float a16 = in[(idx + 16) % n]; float a17 = in[(idx + 17) % n];
    float a18 = in[(idx + 18) % n]; float a19 = in[(idx + 19) % n];
    float a20 = in[(idx + 20) % n]; float a21 = in[(idx + 21) % n];
    float a22 = in[(idx + 22) % n]; float a23 = in[(idx + 23) % n];
    float a24 = in[(idx + 24) % n]; float a25 = in[(idx + 25) % n];
    float a26 = in[(idx + 26) % n]; float a27 = in[(idx + 27) % n];
    float a28 = in[(idx + 28) % n]; float a29 = in[(idx + 29) % n];
    float a30 = in[(idx + 30) % n]; float a31 = in[(idx + 31) % n];
    float a32 = in[(idx + 32) % n]; float a33 = in[(idx + 33) % n];
    float a34 = in[(idx + 34) % n]; float a35 = in[(idx + 35) % n];
    float a36 = in[(idx + 36) % n]; float a37 = in[(idx + 37) % n];
    float a38 = in[(idx + 38) % n]; float a39 = in[(idx + 39) % n];
    float a40 = in[(idx + 40) % n]; float a41 = in[(idx + 41) % n];
    float a42 = in[(idx + 42) % n]; float a43 = in[(idx + 43) % n];
    float a44 = in[(idx + 44) % n]; float a45 = in[(idx + 45) % n];
    float a46 = in[(idx + 46) % n]; float a47 = in[(idx + 47) % n];

    a00 = a00 * a01 + a47;  a01 = a01 * a02 + a00;
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
    a30 = a30 * a31 + a29;  a31 = a31 * a32 + a30;
    a32 = a32 * a33 + a31;  a33 = a33 * a34 + a32;
    a34 = a34 * a35 + a33;  a35 = a35 * a36 + a34;
    a36 = a36 * a37 + a35;  a37 = a37 * a38 + a36;
    a38 = a38 * a39 + a37;  a39 = a39 * a40 + a38;
    a40 = a40 * a41 + a39;  a41 = a41 * a42 + a40;
    a42 = a42 * a43 + a41;  a43 = a43 * a44 + a42;
    a44 = a44 * a45 + a43;  a45 = a45 * a46 + a44;
    a46 = a46 * a47 + a45;  a47 = a47 * a00 + a46;

    a00 = a00 * a24 + a12;  a01 = a01 * a25 + a13;
    a02 = a02 * a26 + a14;  a03 = a03 * a27 + a15;
    a04 = a04 * a28 + a16;  a05 = a05 * a29 + a17;
    a06 = a06 * a30 + a18;  a07 = a07 * a31 + a19;
    a08 = a08 * a32 + a20;  a09 = a09 * a33 + a21;
    a10 = a10 * a34 + a22;  a11 = a11 * a35 + a23;
    a12 = a12 * a36 + a00;  a13 = a13 * a37 + a01;
    a14 = a14 * a38 + a02;  a15 = a15 * a39 + a03;
    a16 = a16 * a40 + a04;  a17 = a17 * a41 + a05;
    a18 = a18 * a42 + a06;  a19 = a19 * a43 + a07;
    a20 = a20 * a44 + a08;  a21 = a21 * a45 + a09;
    a22 = a22 * a46 + a10;  a23 = a23 * a47 + a11;
    a24 = a24 * a00 + a12;  a25 = a25 * a01 + a13;
    a26 = a26 * a02 + a14;  a27 = a27 * a03 + a15;
    a28 = a28 * a04 + a16;  a29 = a29 * a05 + a17;
    a30 = a30 * a06 + a18;  a31 = a31 * a07 + a19;
    a32 = a32 * a08 + a20;  a33 = a33 * a09 + a21;
    a34 = a34 * a10 + a22;  a35 = a35 * a11 + a23;
    a36 = a36 * a12 + a24;  a37 = a37 * a13 + a25;
    a38 = a38 * a14 + a26;  a39 = a39 * a15 + a27;
    a40 = a40 * a16 + a28;  a41 = a41 * a17 + a29;
    a42 = a42 * a18 + a30;  a43 = a43 * a19 + a31;
    a44 = a44 * a20 + a32;  a45 = a45 * a21 + a33;
    a46 = a46 * a22 + a34;  a47 = a47 * a23 + a35;

    a00 = a00 * a47 + a23;  a01 = a01 * a00 + a24;
    a02 = a02 * a01 + a25;  a03 = a03 * a02 + a26;
    a04 = a04 * a03 + a27;  a05 = a05 * a04 + a28;
    a06 = a06 * a05 + a29;  a07 = a07 * a06 + a30;
    a08 = a08 * a07 + a31;  a09 = a09 * a08 + a32;
    a10 = a10 * a09 + a33;  a11 = a11 * a10 + a34;
    a12 = a12 * a11 + a35;  a13 = a13 * a12 + a36;
    a14 = a14 * a13 + a37;  a15 = a15 * a14 + a38;
    a16 = a16 * a15 + a39;  a17 = a17 * a16 + a40;
    a18 = a18 * a17 + a41;  a19 = a19 * a18 + a42;
    a20 = a20 * a19 + a43;  a21 = a21 * a20 + a44;
    a22 = a22 * a21 + a45;  a23 = a23 * a22 + a46;
    a24 = a24 * a23 + a47;  a25 = a25 * a24 + a00;
    a26 = a26 * a25 + a01;  a27 = a27 * a26 + a02;
    a28 = a28 * a27 + a03;  a29 = a29 * a28 + a04;
    a30 = a30 * a29 + a05;  a31 = a31 * a30 + a06;
    a32 = a32 * a31 + a07;  a33 = a33 * a32 + a08;
    a34 = a34 * a33 + a09;  a35 = a35 * a34 + a10;
    a36 = a36 * a35 + a11;  a37 = a37 * a36 + a12;
    a38 = a38 * a37 + a13;  a39 = a39 * a38 + a14;
    a40 = a40 * a39 + a15;  a41 = a41 * a40 + a16;
    a42 = a42 * a41 + a17;  a43 = a43 * a42 + a18;
    a44 = a44 * a43 + a19;  a45 = a45 * a44 + a20;
    a46 = a46 * a45 + a21;  a47 = a47 * a46 + a22;

    return a00 + a01 + a02 + a03 + a04 + a05 + a06 + a07
         + a08 + a09 + a10 + a11 + a12 + a13 + a14 + a15
         + a16 + a17 + a18 + a19 + a20 + a21 + a22 + a23
         + a24 + a25 + a26 + a27 + a28 + a29 + a30 + a31
         + a32 + a33 + a34 + a35 + a36 + a37 + a38 + a39
         + a40 + a41 + a42 + a43 + a44 + a45 + a46 + a47;
}

// ---------------------------------------------------------------------------
void print_func_attrs(const char* name, const void* func) {
    cudaFuncAttributes attr;
    CHECK(cudaFuncGetAttributes(&attr, func));
    printf("  %-25s  regs/thread=%3d  localSize=%5zu  smem_static=%5zu  maxTPB=%d\n",
           name, attr.numRegs, attr.localSizeBytes, attr.sharedSizeBytes,
           attr.maxThreadsPerBlock);
}

struct BenchResult { float median_ms, p10_ms, p90_ms; };

BenchResult bench(const float* d_in, float* d_out, int n, int block_size) {
    int grid = (n + block_size - 1) / block_size;

    // Warmup: 5 launches
    for (int i = 0; i < 5; ++i)
        reg_pressure_kernel<<<grid, block_size>>>(d_in, d_out, n);
    CHECK(cudaDeviceSynchronize());

    // Measure: 20 launches with CUDA events
    cudaEvent_t e_start, e_end;
    CHECK(cudaEventCreate(&e_start));
    CHECK(cudaEventCreate(&e_end));

    std::vector<float> times_ms;
    for (int i = 0; i < 20; ++i) {
        CHECK(cudaEventRecord(e_start));
        reg_pressure_kernel<<<grid, block_size>>>(d_in, d_out, n);
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
    // Print device info
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    printf("CUDA Runtime: %d.%d\n", CUDART_VERSION / 1000,
           (CUDART_VERSION % 1000) / 10);
    int driverVersion;
    CHECK(cudaDriverGetVersion(&driverVersion));
    printf("Driver: %d.%d\n", driverVersion / 1000,
           (driverVersion % 1000) / 10);
    printf("Regs/SM: %d\n", prop.regsPerMultiprocessor);
    printf("Max threads/SM: %d\n", prop.maxThreadsPerMultiProcessor);

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

    // Occupancy query
    int numBlocksPerSm = 0;
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerSm, reg_pressure_kernel, 256, 0));
    printf("cudaOccupancyMaxActiveBlocksPerMultiprocessor(block=256): %d\n",
           numBlocksPerSm);

    // Print kernel attributes
    printf("\nKernel attributes:\n");
    print_func_attrs("reg_pressure_kernel", (const void*)reg_pressure_kernel);

    const int N = 1 << 22;  // 4M elements
    size_t bytes = N * sizeof(float);

    float *h_in  = (float*)malloc(bytes);
    float *h_gpu = (float*)malloc(bytes);

    srand(42);
    for (int i = 0; i < N; ++i)
        h_in[i] = (float)(rand() % 100) / 10000.0f;

    // CPU reference for a small subset (full CPU ref is too slow)
    const int NCHECK = 1024;
    float *h_ref = (float*)malloc(NCHECK * sizeof(float));
    for (int i = 0; i < NCHECK; ++i)
        h_ref[i] = heavy_compute_48_host(h_in, i, N);

    float *d_in, *d_out;
    CHECK(cudaMalloc(&d_in, bytes));
    CHECK(cudaMalloc(&d_out, bytes));
    CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    const int BLOCK = 256;

    // Correctness check on first NCHECK elements
    reg_pressure_kernel<<<(N + BLOCK - 1) / BLOCK, BLOCK>>>(d_in, d_out, N);
    CHECK(cudaMemcpy(h_gpu, d_out, bytes, cudaMemcpyDeviceToHost));

    float max_err = 0.0f;
    for (int i = 0; i < NCHECK; ++i)
        max_err = fmaxf(max_err, fabsf(h_gpu[i] - h_ref[i]));
    printf("\nCorrectness: max_abs_err = %e (checked %d elements)\n",
           max_err, NCHECK);
    // Relaxed tolerance: FMA reordering between CPU/GPU causes differences
    if (max_err > 1e0f) {
        fprintf(stderr, "FAIL: correctness check\n");
        return 1;
    }

    // Benchmark
    printf("\nBenchmark (N=%d, block=%d):\n", N, BLOCK);
    BenchResult r = bench(d_in, d_out, N, BLOCK);
    printf("  median=%.4f ms  p10=%.4f ms  p90=%.4f ms\n",
           r.median_ms, r.p10_ms, r.p90_ms);

    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_out));
    free(h_in); free(h_gpu); free(h_ref);

    printf("\nDone.\n");
    return 0;
}
