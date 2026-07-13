/*
 * Occupancy Sweep Probe
 *
 * Sweeps block sizes (64, 128, 256, 512, 1024) on two kernels:
 *   1. Simple vec_add (memory-bound, 12 regs/thread → 100% occupancy at all sizes)
 *   2. Register-heavy vec_add (40+ regs/thread → varying occupancy)
 *
 * For each block size, measures:
 *   1. Theoretical occupancy via cudaOccupancyMaxActiveBlocksPerMultiprocessor
 *   2. Kernel latency via CUDA events (5 warmup + 20 measured)
 *
 * Demonstrates:
 *   - All block sizes can achieve 100% occupancy for low-register kernels
 *   - Block size 256/512 is typically the latency sweet spot even at equal occupancy
 *   - cudaOccupancyMaxPotentialBlockSize suggests the occupancy-optimal config
 *   - The occupancy-optimal block size is not always the latency-optimal one
 *
 * Build: nvcc -arch=sm_90a -O3 -std=c++17 -lineinfo -Xptxas=-v \
 *        -o occupancy_sweep_probe occupancy_sweep_probe.cu
 */

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                   \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Simple vector-add kernel: C[i] = A[i] + B[i]
// Uses ~12 registers → 100% occupancy at all block sizes on H200
__global__ void vec_add(const float* __restrict__ A,
                        const float* __restrict__ B,
                        float*       __restrict__ C,
                        int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

// Register-heavy variant: each thread accumulates into 16 independent
// registers before writing, increasing register pressure.
// With -maxrregcount=48, this forces occupancy to vary by block size.
__global__ void __launch_bounds__(1024, 1)
vec_add_regpress(const float* __restrict__ A,
                 const float* __restrict__ B,
                 float*       __restrict__ C,
                 int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // 16 independent accumulators to increase register pressure
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    float a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;
    float a8 = 0.0f, a9 = 0.0f, a10 = 0.0f, a11 = 0.0f;
    float a12 = 0.0f, a13 = 0.0f, a14 = 0.0f, a15 = 0.0f;

    // Grid-stride loop: each thread processes multiple elements
    for (int i = idx; i + 15 < n; i += stride * 16) {
        a0  += A[i]      * B[i];      a1  += A[i+1]    * B[i+1];
        a2  += A[i+2]    * B[i+2];    a3  += A[i+3]    * B[i+3];
        a4  += A[i+4]    * B[i+4];    a5  += A[i+5]    * B[i+5];
        a6  += A[i+6]    * B[i+6];    a7  += A[i+7]    * B[i+7];
        a8  += A[i+8]    * B[i+8];    a9  += A[i+9]    * B[i+9];
        a10 += A[i+10]   * B[i+10];   a11 += A[i+11]   * B[i+11];
        a12 += A[i+12]   * B[i+12];   a13 += A[i+13]   * B[i+13];
        a14 += A[i+14]   * B[i+14];   a15 += A[i+15]   * B[i+15];
    }

    // Write back combined result
    float sum = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 +
                a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15;
    if (idx < n) {
        C[idx] = sum;
    }
}

struct SweepResult {
    int block_size;
    double occupancy;
    float latency_ms_median;
    float latency_ms_p10;
    float latency_ms_p90;
    int num_blocks_per_sm;
    int active_warps;
};

static SweepResult sweep_block_size(void* kernel,
                                     int blockSize, size_t dynamicSMem,
                                     const float* d_A, const float* d_B,
                                     float* d_C, int n) {
    SweepResult result;
    result.block_size = blockSize;

    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &result.num_blocks_per_sm, kernel, blockSize, dynamicSMem));

    result.active_warps = result.num_blocks_per_sm * blockSize / prop.warpSize;
    int maxWarps = prop.maxThreadsPerMultiProcessor / prop.warpSize;
    result.occupancy = (double)result.active_warps / (double)maxWarps;

    // Compute grid size: fill the device for maximum throughput
    int gridSize = result.num_blocks_per_sm * prop.multiProcessorCount;
    // But don't exceed what's needed for the data
    int minGridNeeded = (n + blockSize - 1) / blockSize;
    if (gridSize > minGridNeeded) gridSize = minGridNeeded;
    if (gridSize < 1) gridSize = 1;

    // Warmup: 5 launches
    for (int i = 0; i < 5; ++i) {
        vec_add<<<gridSize, blockSize, dynamicSMem>>>(d_A, d_B, d_C, n);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Measure: 20 launches with CUDA events
    cudaEvent_t e_start, e_end;
    CUDA_CHECK(cudaEventCreate(&e_start));
    CUDA_CHECK(cudaEventCreate(&e_end));

    std::vector<float> times_ms;
    for (int i = 0; i < 20; ++i) {
        CUDA_CHECK(cudaEventRecord(e_start));
        if (kernel == (void*)vec_add_regpress) {
            vec_add_regpress<<<gridSize, blockSize, dynamicSMem>>>(
                d_A, d_B, d_C, n);
        } else {
            vec_add<<<gridSize, blockSize, dynamicSMem>>>(
                d_A, d_B, d_C, n);
        }
        CUDA_CHECK(cudaEventRecord(e_end));
        CUDA_CHECK(cudaEventSynchronize(e_end));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, e_start, e_end));
        times_ms.push_back(ms);
    }

    std::sort(times_ms.begin(), times_ms.end());
    result.latency_ms_median = times_ms[10];
    result.latency_ms_p10    = times_ms[2];
    result.latency_ms_p90    = times_ms[18];

    CUDA_CHECK(cudaEventDestroy(e_start));
    CUDA_CHECK(cudaEventDestroy(e_end));

    return result;
}

int main() {
    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    printf("=== Device Info ===\n");
    printf("Device: %s\n", prop.name);
    printf("SM count: %d\n", prop.multiProcessorCount);
    printf("Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Max warps per SM: %d\n", prop.maxThreadsPerMultiProcessor / prop.warpSize);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Max blocks per SM: %d\n", prop.maxBlocksPerMultiProcessor);
    printf("Registers per SM: %d\n", prop.regsPerMultiprocessor);
    printf("Shared mem per SM: %zu bytes\n", prop.sharedMemPerMultiprocessor);
    printf("\n");

    // Allocate data: 64M elements (256 MB per buffer)
    const int N = 64 * 1024 * 1024;
    size_t bytes = N * sizeof(float);

    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    // Initialize with simple values for correctness check
    for (int i = 0; i < N; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Test cudaOccupancyMaxPotentialBlockSize for both kernels
    int autoMinGridSize, autoBlockSize;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(
        &autoMinGridSize, &autoBlockSize, vec_add, 0, 0));
    printf("cudaOccupancyMaxPotentialBlockSize(vec_add):\n");
    printf("  blockSize = %d, minGridSize = %d\n\n", autoBlockSize, autoMinGridSize);

    int autoMinGridSize2, autoBlockSize2;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(
        &autoMinGridSize2, &autoBlockSize2, vec_add_regpress, 0, 0));
    printf("cudaOccupancyMaxPotentialBlockSize(vec_add_regpress):\n");
    printf("  blockSize = %d, minGridSize = %d\n\n", autoBlockSize2, autoMinGridSize2);

    // Test cudaOccupancyAvailableDynamicSMemPerBlock
    int numBlocks256;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocks256, vec_add, 256, 0));
    size_t availDynSmem;
    CUDA_CHECK(cudaOccupancyAvailableDynamicSMemPerBlock(
        &availDynSmem, vec_add, numBlocks256, 256));
    printf("cudaOccupancyAvailableDynamicSMemPerBlock(vec_add, %d blocks, 256 threads):\n",
           numBlocks256);
    printf("  Available dynamic SMem = %zu bytes\n\n", availDynSmem);

    // Sweep block sizes for simple vec_add
    int blockSizes[] = {64, 128, 256, 512, 1024};
    int numBlockSizes = 5;

    printf("=== Simple vec_add kernel (12 regs/thread) ===\n");
    printf("%-12s %-12s %-12s %-12s %-12s %-12s %-12s\n",
           "block_size", "blocks/SM", "active_warps", "occupancy",
           "median_ms", "p10_ms", "p90_ms");
    printf("--------------------------------------------------------------------------------\n");

    for (int i = 0; i < numBlockSizes; ++i) {
        SweepResult r = sweep_block_size((void*)vec_add,
                                          blockSizes[i], 0,
                                          d_A, d_B, d_C, N);
        printf("%-12d %-12d %-12d %-12.1f%% %-12.4f %-12.4f %-12.4f\n",
               r.block_size, r.num_blocks_per_sm, r.active_warps,
               r.occupancy * 100.0,
               r.latency_ms_median, r.latency_ms_p10, r.latency_ms_p90);
    }

    printf("\n");

    // Sweep block sizes for register-heavy kernel
    printf("=== Register-heavy vec_add_regpress kernel ===\n");
    printf("%-12s %-12s %-12s %-12s %-12s %-12s %-12s\n",
           "block_size", "blocks/SM", "active_warps", "occupancy",
           "median_ms", "p10_ms", "p90_ms");
    printf("--------------------------------------------------------------------------------\n");

    for (int i = 0; i < numBlockSizes; ++i) {
        SweepResult r = sweep_block_size((void*)vec_add_regpress,
                                          blockSizes[i], 0,
                                          d_A, d_B, d_C, N);
        printf("%-12d %-12d %-12d %-12.1f%% %-12.4f %-12.4f %-12.4f\n",
               r.block_size, r.num_blocks_per_sm, r.active_warps,
               r.occupancy * 100.0,
               r.latency_ms_median, r.latency_ms_p10, r.latency_ms_p90);
    }

    // Verify correctness for simple vec_add
    int gridSize = (N + 256 - 1) / 256;
    vec_add<<<gridSize, 256>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
    float max_err = 0.0f;
    for (int i = 0; i < N; ++i) {
        float err = fabsf(h_C[i] - 3.0f);
        if (err > max_err) max_err = err;
    }
    printf("\nCorrectness check (vec_add): max_abs_err = %e (tolerance: 1e-5)\n", max_err);
    printf("%s\n", max_err <= 1e-5f ? "PASS" : "FAIL");

    // Cleanup
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}