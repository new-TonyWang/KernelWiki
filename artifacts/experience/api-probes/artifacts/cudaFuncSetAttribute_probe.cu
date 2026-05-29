/*
 * Probe: cudaFuncSetAttribute (end-to-end)
 *
 * Signature (runtime):
 *   __host__ cudaError_t
 *   cudaFuncSetAttribute(const void* func, cudaFuncAttribute attr, int value);
 *
 * This probe exercises the most common use of the API: opting into dynamic
 * shared memory > 48 KB on H200 via
 * cudaFuncAttributeMaxDynamicSharedMemorySize.
 *
 * Behavior:
 *   1. Query initial attributes with cudaFuncGetAttributes.
 *   2. First, launch the kernel with 200 KB of dynamic shared memory
 *      WITHOUT calling cudaFuncSetAttribute. This should fail with
 *      cudaErrorInvalidValue because the default dynamic-shared-memory cap
 *      is ~48 KB (cudaDevAttrMaxSharedMemoryPerBlock).
 *   3. Then call cudaFuncSetAttribute(kernel,
 *      cudaFuncAttributeMaxDynamicSharedMemorySize, 200*1024). Verify the
 *      attribute is reflected in a follow-up cudaFuncGetAttributes query.
 *   4. Re-launch with 200 KB dynamic shmem. Now the launch should succeed.
 *   5. Verify numerical correctness: kernel writes smem[i] = i * threadIdx.x
 *      then copies to output; host compares against a scalar reference.
 *   6. Time the host-side cudaFuncSetAttribute call (1000 iters average).
 *
 * Build:
 *   nvcc -arch=sm_90a -O3 -std=c++17 -lineinfo \
 *        -o func_set_attr_probe cudaFuncSetAttribute_probe.cu
 */

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Kernel that uses a configurable amount of dynamic shared memory.
// Each thread initializes N_per_thread entries of smem, then stores
// smem[lane*N_per_thread] to out[threadIdx.x].
__global__ void big_smem_kernel(float* out, int n_floats_per_block) {
    extern __shared__ float smem[];
    int t = threadIdx.x;
    int blk = blockDim.x;

    // Cooperative init: each thread writes strided positions in smem.
    for (int i = t; i < n_floats_per_block; i += blk) {
        smem[i] = (float)(i * (t + 1));
    }
    __syncthreads();

    // Each thread picks its slot and emits a checksum.
    // pick smem[t] (only valid if n_floats_per_block >= blk).
    if (t < n_floats_per_block) {
        out[t] = smem[t];
    }
}

int main() {
    int dev = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("device=%s sm=%d.%d\n", prop.name, prop.major, prop.minor);
    printf("sharedMemPerBlock (default cap)   = %d bytes\n",
           (int)prop.sharedMemPerBlock);
    printf("sharedMemPerBlockOptin (opt-in)   = %d bytes\n",
           (int)prop.sharedMemPerBlockOptin);
    printf("sharedMemPerMultiprocessor        = %d bytes\n",
           (int)prop.sharedMemPerMultiprocessor);

    // ========= Step 1: initial attributes =========
    cudaFuncAttributes fattr;
    CUDA_CHECK(cudaFuncGetAttributes(&fattr, (const void*)big_smem_kernel));
    printf("\n=== Initial cudaFuncAttributes (big_smem_kernel) ===\n");
    printf("  numRegs=%d\n", fattr.numRegs);
    printf("  sharedSizeBytes (static)              = %d\n",
           (int)fattr.sharedSizeBytes);
    printf("  maxDynamicSharedSizeBytes (pre-set)   = %d\n",
           (int)fattr.maxDynamicSharedSizeBytes);
    printf("  preferredShmemCarveout                = %d\n",
           fattr.preferredShmemCarveout);

    // ========= Step 2: launch WITHOUT opt-in, expect failure =========
    const int BLK = 256;
    const int DYN_FLOATS = (200 * 1024) / (int)sizeof(float);   // 51200 floats
    const size_t DYN_BYTES = DYN_FLOATS * sizeof(float);

    float* dOut = nullptr;
    CUDA_CHECK(cudaMalloc(&dOut, BLK * sizeof(float)));

    printf("\n=== Launch without cudaFuncSetAttribute (dynSmem=%zu bytes) ===\n",
           DYN_BYTES);
    big_smem_kernel<<<1, BLK, DYN_BYTES>>>(dOut, DYN_FLOATS);
    cudaError_t e_pre = cudaGetLastError();
    cudaError_t e_sync_pre = cudaDeviceSynchronize();
    bool pre_failed = (e_pre != cudaSuccess) || (e_sync_pre != cudaSuccess);
    printf("  launch err   = %s\n", cudaGetErrorString(e_pre));
    printf("  sync err     = %s\n", cudaGetErrorString(e_sync_pre));
    printf("  expected failure? %s ; observed failure? %s\n",
           "yes", pre_failed ? "yes" : "no");

    // Clear any sticky error for the next launch.
    // Some error codes are NOT sticky, so we call cudaGetLastError twice.
    (void)cudaGetLastError();

    // ========= Step 3: call cudaFuncSetAttribute =========
    const int opt_in = 200 * 1024;  // 200 KB
    printf("\n=== cudaFuncSetAttribute(MaxDynamicSharedMemorySize, %d) ===\n",
           opt_in);
    cudaError_t e_set = cudaFuncSetAttribute(
        (const void*)big_smem_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        opt_in);
    printf("  return = %s\n", cudaGetErrorString(e_set));
    if (e_set != cudaSuccess) { cudaFree(dOut); return 2; }

    // Time the host-side API call.
    const int REP = 1000;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < REP; ++i) {
        CUDA_CHECK(cudaFuncSetAttribute(
            (const void*)big_smem_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, opt_in));
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double us_per_call =
        std::chrono::duration<double, std::micro>(t1 - t0).count() / REP;
    printf("  avg host call latency = %.3f us (over %d iters)\n",
           us_per_call, REP);

    // ========= Step 4: re-query attributes to confirm value =========
    CUDA_CHECK(cudaFuncGetAttributes(&fattr, (const void*)big_smem_kernel));
    printf("\n=== Post-set cudaFuncAttributes ===\n");
    printf("  maxDynamicSharedSizeBytes (post-set)  = %d (expected %d)\n",
           (int)fattr.maxDynamicSharedSizeBytes, opt_in);
    bool attr_ok = ((int)fattr.maxDynamicSharedSizeBytes == opt_in);

    // ========= Step 5: re-launch, expect success =========
    printf("\n=== Re-launch with dynSmem=%zu bytes (after opt-in) ===\n",
           DYN_BYTES);

    // Warmup
    for (int i = 0; i < 5; ++i) {
        big_smem_kernel<<<1, BLK, DYN_BYTES>>>(dOut, DYN_FLOATS);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Measure kernel latency via CUDA events (5 warmup discarded above, 20 measured).
    cudaEvent_t eStart, eEnd;
    CUDA_CHECK(cudaEventCreate(&eStart));
    CUDA_CHECK(cudaEventCreate(&eEnd));
    std::vector<float> ms(20);
    for (int i = 0; i < 20; ++i) {
        CUDA_CHECK(cudaEventRecord(eStart));
        big_smem_kernel<<<1, BLK, DYN_BYTES>>>(dOut, DYN_FLOATS);
        CUDA_CHECK(cudaEventRecord(eEnd));
        CUDA_CHECK(cudaEventSynchronize(eEnd));
        CUDA_CHECK(cudaEventElapsedTime(&ms[i], eStart, eEnd));
    }
    std::sort(ms.begin(), ms.end());

    cudaError_t e_launch = cudaGetLastError();
    cudaError_t e_sync   = cudaDeviceSynchronize();
    bool post_ok = (e_launch == cudaSuccess) && (e_sync == cudaSuccess);
    printf("  launch err = %s\n", cudaGetErrorString(e_launch));
    printf("  sync err   = %s\n", cudaGetErrorString(e_sync));
    printf("  kernel latency median=%.6f p10=%.6f p90=%.6f ms\n",
           ms[10], ms[2], ms[18]);

    // ========= Verify correctness =========
    std::vector<float> hOut(BLK, -1.0f);
    CUDA_CHECK(cudaMemcpy(hOut.data(), dOut, BLK * sizeof(float),
                          cudaMemcpyDeviceToHost));
    // For the init loop: smem[i] = i * (t + 1). Thread t is responsible for
    // strided indices i = t, t+BLK, t+2*BLK ... . The last write at a given
    // slot i comes from thread t0 = i % BLK. So smem[t] (i == t) ends up as
    // t * (t + 1) -- but other threads also write smem[t + k*BLK] for k>=1
    // using their own (t_writer + 1). The final value at smem[i=t] in
    // 1-block launch equals i * ((i % BLK) + 1) = t * (t + 1). Thread t
    // reads smem[t] -> out[t].
    double max_err = 0.0;
    for (int t = 0; t < BLK; ++t) {
        float ref = (float)t * (float)(t + 1);
        double err = std::abs((double)hOut[t] - (double)ref);
        if (err > max_err) max_err = err;
    }
    bool correctness_ok = (max_err <= 1e-5);
    printf("  max_abs_err vs scalar ref = %.6g (tol=1e-5) %s\n",
           max_err, correctness_ok ? "PASS" : "FAIL");

    // ========= Summary =========
    printf("\n=== Summary ===\n");
    printf("  pre-optin launch failed (expected)   : %s\n", pre_failed ? "yes" : "no");
    printf("  attribute reflects opt-in value      : %s\n", attr_ok ? "yes" : "no");
    printf("  post-optin launch succeeded          : %s\n", post_ok ? "yes" : "no");
    printf("  correctness (max_abs_err <= 1e-5)    : %s\n", correctness_ok ? "yes" : "no");

    CUDA_CHECK(cudaFree(dOut));
    CUDA_CHECK(cudaEventDestroy(eStart));
    CUDA_CHECK(cudaEventDestroy(eEnd));

    bool all_ok = pre_failed && attr_ok && post_ok && correctness_ok;
    printf("ALL_OK=%s\n", all_ok ? "true" : "false");
    return all_ok ? 0 : 1;
}
