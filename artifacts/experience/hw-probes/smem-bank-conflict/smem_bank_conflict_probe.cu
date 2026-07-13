// smem_bank_conflict_probe.cu — Microbench for shared-memory bank conflicts
//
// Protocol: hardware-microbench.md (adapted for bank-conflict factor sweep).
//
// Measures shared-memory load latency under different access patterns.
// All 32 threads in a warp execute loads simultaneously.  The access pattern
// determines whether threads hit the same or different banks.
//
// Method: pure pointer-chasing with NO arithmetic between loads.
// smem[idx] = next_idx, the chain is pre-built so that every hop maintains
// the desired conflict pattern.  This eliminates ALU contamination.
//
// Patterns:
//   (a) conflict-free:  smem[tid] = tid (stride-1 self-loop, 0 conflicts)
//   (b) 2-way conflict: smem[tid*2] = tid*2 (stride-2 self-loop, 2-way)
//   (c) broadcast:      smem[0] = 0 (all read same addr, broadcast, 0 conflicts)
//   (d) stride-32:      smem[tid*32] = tid*32 (stride-32 self-loop, 32-way)
//   (e) padded:         smem[tid*33] = tid*33 (stride-33 self-loop, ~0 conflicts)
//
// Target: H200 (sm_90a), CUDA 12.9
// Build : nvcc -arch=sm_90a -O3 -std=c++17 -lineinfo -o smem_bank_conflict_probe smem_bank_conflict_probe.cu

#include <cstdio>
#include <cstdint>
#include <algorithm>
#include <vector>

static constexpr int CHAIN_LEN   = 4096;
static constexpr int NUM_TRIALS  = 64;
static constexpr int NUM_LAUNCHES = 20;
static constexpr int WARP_SIZE   = 32;

// ==========================================================================
// All kernels use the same structure: pointer-chasing self-loops.
// The key difference is where each thread's pointer points (which determines
// the bank-access pattern for the warp-wide load).
// NO arithmetic between loads — pure dependent ld.shared chain.
// ==========================================================================

// Pattern (a): conflict-free (stride-1)
// Thread i reads smem[i] which contains i.  Pure self-loop.
// Bank(i) = i % 32 — all 32 threads hit different banks.
__global__ void probe_conflict_free(uint64_t* __restrict__ out_cycles,
                                    uint32_t* __restrict__ out_val) {
    __shared__ int smem[1024];
    int tid = threadIdx.x;
    for (int i = tid; i < 1024; i += WARP_SIZE)
        smem[i] = i;  // self-loop: smem[i] = i
    __syncwarp();

    int idx = tid;
    for (int t = 0; t < NUM_TRIALS; ++t) {
        uint64_t s = clock64();
        #pragma unroll 1
        for (int h = 0; h < CHAIN_LEN; ++h)
            idx = smem[idx];
        uint64_t e = clock64();
        if (tid == 0) out_cycles[t] = e - s;
    }
    if (tid == 0) out_val[0] = (uint32_t)idx;
}

// Pattern (b): 2-way bank conflict (stride-2)
// Thread i reads smem[i*2] which contains i*2.  Self-loop.
// Bank(i*2) = (i*2) % 32.  Threads 0 and 16 → bank 0, etc. → 2-way.
__global__ void probe_2way(uint64_t* __restrict__ out_cycles,
                           uint32_t* __restrict__ out_val) {
    __shared__ int smem[1024];
    int tid = threadIdx.x;
    for (int i = tid; i < 1024; i += WARP_SIZE)
        smem[i] = i;  // self-loop
    __syncwarp();

    int idx = tid * 2;  // thread i starts at its stride-2 slot
    for (int t = 0; t < NUM_TRIALS; ++t) {
        uint64_t s = clock64();
        #pragma unroll 1
        for (int h = 0; h < CHAIN_LEN; ++h)
            idx = smem[idx];
        uint64_t e = clock64();
        if (tid == 0) out_cycles[t] = e - s;
    }
    if (tid == 0) out_val[0] = (uint32_t)idx;
}

// Pattern (c): broadcast (all threads read smem[0])
// smem[0] = 0, all threads start at 0.  All read same address → broadcast.
__global__ void probe_broadcast(uint64_t* __restrict__ out_cycles,
                                uint32_t* __restrict__ out_val) {
    __shared__ int smem[1024];
    int tid = threadIdx.x;
    for (int i = tid; i < 1024; i += WARP_SIZE)
        smem[i] = 0;  // everything points to 0
    __syncwarp();

    int idx = 0;
    for (int t = 0; t < NUM_TRIALS; ++t) {
        uint64_t s = clock64();
        #pragma unroll 1
        for (int h = 0; h < CHAIN_LEN; ++h)
            idx = smem[idx];
        uint64_t e = clock64();
        if (tid == 0) out_cycles[t] = e - s;
    }
    if (tid == 0) out_val[0] = (uint32_t)idx;
}

// Pattern (d): 32-way bank conflict (stride-32)
// Thread i reads smem[i*32].  Bank(i*32) = 0 for all i.  32-way conflict.
// smem[i*32] = i*32  (self-loop).
__global__ void probe_32way(uint64_t* __restrict__ out_cycles,
                            uint32_t* __restrict__ out_val) {
    __shared__ int smem[1024];
    int tid = threadIdx.x;
    for (int i = tid; i < 1024; i += WARP_SIZE)
        smem[i] = i;
    __syncwarp();

    int idx = tid * 32;  // thread 0 → 0, thread 1 → 32, ... thread 31 → 992
    for (int t = 0; t < NUM_TRIALS; ++t) {
        uint64_t s = clock64();
        #pragma unroll 1
        for (int h = 0; h < CHAIN_LEN; ++h)
            idx = smem[idx];
        uint64_t e = clock64();
        if (tid == 0) out_cycles[t] = e - s;
    }
    if (tid == 0) out_val[0] = (uint32_t)idx;
}

// Pattern (e): padded (stride-33 — the +1 fix)
// Thread i reads smem[i*33].  Bank(i*33) = (i*33)%32 = (i+i*32)%32 = i%32.
// All 32 threads hit different banks → conflict-free.
// smem[i*33] = i*33  (self-loop).
__global__ void probe_padded(uint64_t* __restrict__ out_cycles,
                             uint32_t* __restrict__ out_val) {
    __shared__ int smem[1088];  // need 31*33+1 = 1024 at most; round up
    int tid = threadIdx.x;
    for (int i = tid; i < 1088; i += WARP_SIZE)
        smem[i] = i;
    __syncwarp();

    int idx = tid * 33;
    for (int t = 0; t < NUM_TRIALS; ++t) {
        uint64_t s = clock64();
        #pragma unroll 1
        for (int h = 0; h < CHAIN_LEN; ++h)
            idx = smem[idx];
        uint64_t e = clock64();
        if (tid == 0) out_cycles[t] = e - s;
    }
    if (tid == 0) out_val[0] = (uint32_t)idx;
}

// ==========================================================================
// Host driver
// ==========================================================================
struct ProbeResult {
    double median, p10, p90;
};

ProbeResult run_probe(void (*kernel)(uint64_t*, uint32_t*), const char* label) {
    uint64_t* d_cycles;
    uint32_t* d_val;
    cudaMalloc(&d_cycles, NUM_TRIALS * sizeof(uint64_t));
    cudaMalloc(&d_val, sizeof(uint32_t));

    for (int w = 0; w < 5; ++w) {
        kernel<<<1, 32>>>(d_cycles, d_val);
        cudaDeviceSynchronize();
    }

    std::vector<double> all;
    for (int launch = 0; launch < NUM_LAUNCHES; ++launch) {
        kernel<<<1, 32>>>(d_cycles, d_val);
        cudaDeviceSynchronize();

        std::vector<uint64_t> h(NUM_TRIALS);
        cudaMemcpy(h.data(), d_cycles, NUM_TRIALS * sizeof(uint64_t),
                   cudaMemcpyDeviceToHost);
        for (int t = 0; t < NUM_TRIALS; ++t)
            all.push_back((double)h[t] / CHAIN_LEN);
    }

    std::sort(all.begin(), all.end());
    int N = (int)all.size();
    ProbeResult r = { all[N/2], all[N/10], all[N*9/10] };

    printf("  %-40s median=%7.2f  p10=%7.2f  p90=%7.2f  cycles/load\n",
           label, r.median, r.p10, r.p90);

    cudaFree(d_cycles);
    cudaFree(d_val);
    return r;
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int driver_ver = 0;
    cudaDriverGetVersion(&driver_ver);
    printf("Device: %s  SM: %d.%d  CUDA runtime: %d.%d  Driver: %d.%d\n",
           prop.name, prop.major, prop.minor,
           CUDART_VERSION / 1000, (CUDART_VERSION % 1000) / 10,
           driver_ver / 1000, (driver_ver % 1000) / 10);
    printf("CHAIN_LEN=%d  NUM_TRIALS=%d  NUM_LAUNCHES=%d  total_samples=%d\n\n",
           CHAIN_LEN, NUM_TRIALS, NUM_LAUNCHES, NUM_TRIALS * NUM_LAUNCHES);

    printf("=== Shared Memory Bank Conflict Probe (H200, sm_90a) ===\n\n");

    ProbeResult r_free = run_probe(probe_conflict_free, "(a) conflict-free (stride-1)");
    ProbeResult r_2way = run_probe(probe_2way,          "(b) 2-way conflict (stride-2)");
    ProbeResult r_bc   = run_probe(probe_broadcast,     "(c) broadcast (all smem[0])");
    ProbeResult r_32   = run_probe(probe_32way,         "(d) 32-way conflict (stride-32)");
    ProbeResult r_pad  = run_probe(probe_padded,        "(e) stride-32 padded (+1 fix)");

    printf("\n=== Summary ===\n");
    printf("  conflict-free baseline:   %.2f cycles/load\n", r_free.median);
    printf("  2-way / baseline:         %.2fx\n", r_2way.median / r_free.median);
    printf("  broadcast / baseline:     %.2fx\n", r_bc.median / r_free.median);
    printf("  32-way / baseline:        %.2fx\n", r_32.median / r_free.median);
    printf("  padded / baseline:        %.2fx\n", r_pad.median / r_free.median);

    return 0;
}
