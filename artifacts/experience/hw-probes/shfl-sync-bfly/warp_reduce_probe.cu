// warp_reduce_probe.cu — Microbench for warp-reduce butterfly pattern
// using __shfl_xor_sync with delta = 16, 8, 4, 2, 1.
//
// Protocol: hardware-microbench.md compute-latency probe.
// Each trial: 1 warp, thread 0 reports clock64() around an unrolled
// chain of 128 butterfly reductions (5 shfl_xor per reduction = 640 shfl).
// We report cycles per single 5-step butterfly reduction.
//
// Target: H200 (sm_90a), CUDA 12.9, nvcc -arch=sm_90a -O3 -std=c++17

#include <cstdio>
#include <cstdint>
#include <algorithm>
#include <vector>
#include <numeric>
#include <cmath>

// Single butterfly reduction: 5 dependent shfl_xor_sync calls.
// The result feeds the next reduction to form a dependent chain.
__device__ __forceinline__ float warp_reduce_sum(float val) {
    val += __shfl_xor_sync(0xFFFFFFFF, val, 16);
    val += __shfl_xor_sync(0xFFFFFFFF, val, 8);
    val += __shfl_xor_sync(0xFFFFFFFF, val, 4);
    val += __shfl_xor_sync(0xFFFFFFFF, val, 2);
    val += __shfl_xor_sync(0xFFFFFFFF, val, 1);
    return val;
}

// Kernel: unrolled chain of 128 butterfly reductions.
// val feeds forward to prevent dead-code elimination.
__global__ void probe_kernel(uint64_t* __restrict__ out_cycles,
                             float*    __restrict__ out_val,
                             int                    num_trials) {
    int lane = threadIdx.x % 32;
    float val = static_cast<float>(lane + 1);  // non-zero seed

    for (int trial = 0; trial < num_trials; ++trial) {
        uint64_t start = clock64();

        // 128 dependent butterfly reductions
        #pragma unroll
        for (int i = 0; i < 128; ++i) {
            val = warp_reduce_sum(val);
        }

        uint64_t end = clock64();

        if (lane == 0) {
            out_cycles[trial] = end - start;
        }
    }

    // Prevent DCE — store final val
    if (lane == 0) {
        out_val[0] = val;
    }
}

int main() {
    const int NUM_TRIALS = 4096;
    const int REDUCTIONS_PER_TRIAL = 128;
    const int SHFL_PER_REDUCTION = 5;

    uint64_t* d_cycles;
    float*    d_val;
    cudaMalloc(&d_cycles, NUM_TRIALS * sizeof(uint64_t));
    cudaMalloc(&d_val, sizeof(float));

    // Warmup
    probe_kernel<<<1, 32>>>(d_cycles, d_val, 5);
    cudaDeviceSynchronize();

    // Measurement
    probe_kernel<<<1, 32>>>(d_cycles, d_val, NUM_TRIALS);
    cudaDeviceSynchronize();

    // Copy back
    std::vector<uint64_t> h_cycles(NUM_TRIALS);
    cudaMemcpy(h_cycles.data(), d_cycles, NUM_TRIALS * sizeof(uint64_t),
               cudaMemcpyDeviceToHost);

    float h_val;
    cudaMemcpy(&h_val, d_val, sizeof(float), cudaMemcpyDeviceToHost);

    // Compute per-reduction and per-shfl cycles
    std::vector<double> per_reduction(NUM_TRIALS);
    std::vector<double> per_shfl(NUM_TRIALS);
    for (int i = 0; i < NUM_TRIALS; ++i) {
        per_reduction[i] = static_cast<double>(h_cycles[i]) / REDUCTIONS_PER_TRIAL;
        per_shfl[i] = static_cast<double>(h_cycles[i]) /
                      (REDUCTIONS_PER_TRIAL * SHFL_PER_REDUCTION);
    }

    std::sort(per_reduction.begin(), per_reduction.end());
    std::sort(per_shfl.begin(), per_shfl.end());

    double median_red  = per_reduction[NUM_TRIALS / 2];
    double p10_red     = per_reduction[NUM_TRIALS / 10];
    double p90_red     = per_reduction[NUM_TRIALS * 9 / 10];

    double median_shfl = per_shfl[NUM_TRIALS / 2];
    double p10_shfl    = per_shfl[NUM_TRIALS / 10];
    double p90_shfl    = per_shfl[NUM_TRIALS * 9 / 10];

    printf("=== Warp-Reduce Butterfly Probe (H200, sm_90a) ===\n");
    printf("Trials:              %d\n", NUM_TRIALS);
    printf("Reductions/trial:    %d\n", REDUCTIONS_PER_TRIAL);
    printf("shfl_xor/reduction:  %d\n", SHFL_PER_REDUCTION);
    printf("\n");
    printf("Per butterfly-reduction (5 shfl_xor_sync):\n");
    printf("  median: %.2f cycles\n", median_red);
    printf("  p10:    %.2f cycles\n", p10_red);
    printf("  p90:    %.2f cycles\n", p90_red);
    printf("\n");
    printf("Per shfl_xor_sync (within butterfly chain):\n");
    printf("  median: %.2f cycles\n", median_shfl);
    printf("  p10:    %.2f cycles\n", p10_shfl);
    printf("  p90:    %.2f cycles\n", p90_shfl);
    printf("\n");
    printf("Anti-DCE value: %f\n", h_val);

    cudaFree(d_cycles);
    cudaFree(d_val);
    return 0;
}
