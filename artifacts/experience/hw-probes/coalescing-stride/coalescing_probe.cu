// coalescing_probe.cu — Microbenchmark: coalesced vs strided global memory access
// Target: H200 (sm_90a), CUDA 12.9
//
// Measures effective bandwidth for two access patterns:
//   (a) Coalesced (stride-1):  thread i reads array[i]
//   (b) Strided   (stride-32): thread i reads array[i * 32]
//
// Protocol: warmup 5, measure 20, CUDA events, median/p10/p90.

#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do {                                       \
    cudaError_t err = (call);                                       \
    if (err != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",               \
                __FILE__, __LINE__, cudaGetErrorString(err));       \
        exit(1);                                                    \
    }                                                               \
} while(0)

// Kernel: each thread reads one float from global memory and writes a
// reduced value to prevent dead-code elimination.
// stride == 1 → coalesced; stride == 32 → non-coalesced.
__global__ void read_kernel(const float* __restrict__ input,
                            float*       __restrict__ output,
                            int stride,
                            int n_elements) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = tid * stride;
    float val = 0.0f;
    if (idx < n_elements) {
        val = input[idx];
    }
    // Write to output to prevent DCE. Only lane 0 of each warp writes
    // the warp-reduced sum to avoid output-side bottleneck.
    // Use warp shuffle to reduce within the warp.
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    int lane = threadIdx.x % 32;
    int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (lane == 0) {
        output[warpId] = val;
    }
}

int main() {
    // Print device info
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("HBM bandwidth (theoretical): %.1f GB/s\n",
           2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);

    int driverVersion, runtimeVersion;
    CHECK_CUDA(cudaDriverGetVersion(&driverVersion));
    CHECK_CUDA(cudaRuntimeGetVersion(&runtimeVersion));
    printf("CUDA Runtime: %d.%d, Driver: %d.%d\n",
           runtimeVersion / 1000, (runtimeVersion % 100) / 10,
           driverVersion / 1000, (driverVersion % 100) / 10);

    // Configuration
    const int BLOCK_SIZE = 256;
    const int NUM_THREADS = 1 << 20;  // ~1M threads
    const int NUM_BLOCKS = NUM_THREADS / BLOCK_SIZE;
    const int WARMUP = 5;
    const int REPEATS = 20;

    // For stride-1:  need NUM_THREADS floats = 4 MB
    // For stride-32: need NUM_THREADS * 32 floats = 128 MB (fits in HBM, exceeds L2)
    const int N_COALESCED = NUM_THREADS;
    const int N_STRIDED   = NUM_THREADS * 32;

    printf("\nConfiguration:\n");
    printf("  Threads: %d (%d blocks x %d threads)\n", NUM_THREADS, NUM_BLOCKS, BLOCK_SIZE);
    printf("  Coalesced array: %d floats (%.2f MB)\n", N_COALESCED, N_COALESCED * 4.0 / (1 << 20));
    printf("  Strided array:   %d floats (%.2f MB)\n", N_STRIDED, N_STRIDED * 4.0 / (1 << 20));
    printf("  Warmup: %d, Repeats: %d\n", WARMUP, REPEATS);

    // Allocate device memory
    float *d_input_coal, *d_input_stride, *d_output;
    int n_warps = NUM_THREADS / 32;
    CHECK_CUDA(cudaMalloc(&d_input_coal,   N_COALESCED * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_input_stride, N_STRIDED   * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output,       n_warps     * sizeof(float)));

    // Initialize input with deterministic values on host, copy to device
    std::vector<float> h_input_coal(N_COALESCED);
    std::vector<float> h_input_stride(N_STRIDED);
    for (int i = 0; i < N_COALESCED; i++) h_input_coal[i] = 1.0f;
    for (int i = 0; i < N_STRIDED; i++)   h_input_stride[i] = 1.0f;
    CHECK_CUDA(cudaMemcpy(d_input_coal, h_input_coal.data(),
                           N_COALESCED * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_input_stride, h_input_stride.data(),
                           N_STRIDED * sizeof(float), cudaMemcpyHostToDevice));

    // CUDA events
    cudaEvent_t e_start, e_end;
    CHECK_CUDA(cudaEventCreate(&e_start));
    CHECK_CUDA(cudaEventCreate(&e_end));

    auto bench = [&](const char* label, const float* d_input, int stride, int n_elem) {
        // Warmup
        for (int i = 0; i < WARMUP; i++) {
            read_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_input, d_output, stride, n_elem);
        }
        CHECK_CUDA(cudaDeviceSynchronize());

        // Measure
        std::vector<float> times_ms(REPEATS);
        for (int i = 0; i < REPEATS; i++) {
            CHECK_CUDA(cudaEventRecord(e_start));
            read_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_input, d_output, stride, n_elem);
            CHECK_CUDA(cudaEventRecord(e_end));
            CHECK_CUDA(cudaEventSynchronize(e_end));
            float ms;
            CHECK_CUDA(cudaEventElapsedTime(&ms, e_start, e_end));
            times_ms[i] = ms;
        }
        std::sort(times_ms.begin(), times_ms.end());
        float median = times_ms[REPEATS / 2];
        float p10    = times_ms[REPEATS / 5];       // index 4
        float p90    = times_ms[REPEATS * 9 / 10];  // index 18

        // Each thread reads one float (4 bytes)
        double bytes_read = (double)NUM_THREADS * 4.0;
        double bw_gbps = bytes_read / (median * 1e-3) / 1e9;

        printf("\n=== %s ===\n", label);
        printf("  Latency (ms): median=%.4f  p10=%.4f  p90=%.4f\n", median, p10, p90);
        printf("  Effective BW: %.2f GB/s\n", bw_gbps);
        printf("  Bytes read (useful): %.2f MB\n", bytes_read / (1 << 20));
        return median;
    };

    float coal_ms   = bench("COALESCED (stride=1)", d_input_coal, 1, N_COALESCED);
    float stride_ms = bench("STRIDED (stride=32)",  d_input_stride, 32, N_STRIDED);

    printf("\n=== SUMMARY ===\n");
    printf("  Coalesced median:  %.4f ms\n", coal_ms);
    printf("  Strided median:    %.4f ms\n", stride_ms);
    printf("  Slowdown ratio:    %.2fx (strided / coalesced)\n", stride_ms / coal_ms);
    printf("  BW ratio:          %.2fx (coalesced / strided)\n", stride_ms / coal_ms);

    // Cleanup
    CHECK_CUDA(cudaEventDestroy(e_start));
    CHECK_CUDA(cudaEventDestroy(e_end));
    CHECK_CUDA(cudaFree(d_input_coal));
    CHECK_CUDA(cudaFree(d_input_stride));
    CHECK_CUDA(cudaFree(d_output));

    return 0;
}
