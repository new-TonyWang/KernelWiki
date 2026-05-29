// Cutlass-free TMA cluster-multicast probe.
//
// Compares 3 modes on H200 sm_90a:
//   - C=1: no cluster, regular TMA load — baseline (DRAM-bound)
//   - C=2: cluster=<2,1,1>, multicast TMA — 2 CTAs receive the same tile from
//          one DRAM read; effective smem-delivery bandwidth = 2× DRAM bw
//   - C=4: cluster=<4,1,1>, multicast TMA — 4 CTAs share one DRAM read;
//          effective bandwidth = 4× DRAM bw
//
// Workload per launch:
//   - 128 MiB bf16 source (> 60 MiB H200 L2 → DRAM-bound)
//   - 132 CTAs (= H200 SM count) total, arranged as 132/C clusters
//   - Each cluster's leader (rank 0) issues all the TMA loads via multicast;
//     all C CTAs in the cluster receive the data into their own smem
//   - Pipeline depth=4 mbarrier ring buffer per CTA, 5 warmup + 20 timed launches
//
// Build:
//   nvcc -std=c++17 -O3 -gencode=arch=compute_90a,code=sm_90a -lineinfo \
//        tma_multicast_probe.cu -lcuda -o tma_multicast_probe

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <string>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda.h>

#define CUDA_CHECK(x) do { cudaError_t err = (x); if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
    std::exit(1); } } while (0)

#define CU_CHECK(x) do { CUresult err = (x); if (err != CUDA_SUCCESS) { \
    const char* msg = nullptr; cuGetErrorString(err, &msg); \
    fprintf(stderr, "CU error %s at %s:%d\n", msg ? msg : "(?)", __FILE__, __LINE__); \
    std::exit(1); } } while (0)

constexpr int SRC_ROWS = 16384;
constexpr int SRC_COLS = 4096;
constexpr size_t SRC_BYTES = size_t(SRC_ROWS) * SRC_COLS * sizeof(__nv_bfloat16);
constexpr int N_SMS = 132;
constexpr int BOX_ROWS = 64;
constexpr int BOX_COLS = 64;          // 128 B fast-axis (matches all swizzle modes)
constexpr int N_COLS_TILES = SRC_COLS / BOX_COLS;
constexpr int TILE_BYTES = BOX_ROWS * BOX_COLS * sizeof(__nv_bfloat16);   // 8192 B
constexpr int DEPTH = 4;

// ---------------------------------------------------------------------------
// PTX wrappers (cutlass-free).

__device__ __forceinline__ uint32_t get_cluster_rank() {
    uint32_t rank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
    return rank;
}

__device__ __forceinline__ void cluster_sync() {
    asm volatile("barrier.cluster.arrive.aligned;");
    asm volatile("barrier.cluster.wait.aligned;");
}

__device__ __forceinline__ void mbarrier_init(uint64_t* bar, uint32_t n_threads) {
    uint32_t b = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(b), "r"(n_threads));
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* bar, uint32_t expected_bytes) {
    uint32_t b = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"(b), "r"(expected_bytes));
}

__device__ __forceinline__ void mbarrier_wait(uint64_t* bar, uint32_t phase) {
    uint32_t b = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile(
        "{ .reg .pred P1;\n"
        "  LAB_W_%=:\n"
        "  mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
        "  @P1 bra DONE_W_%=;\n"
        "  bra LAB_W_%=;\n"
        "  DONE_W_%=: }"
        :: "r"(b), "r"(phase));
}

__device__ __forceinline__ void tma_load_2d(
    void* smem_dst, const CUtensorMap* tensor_map,
    int32_t coord_row, int32_t coord_col, uint64_t* mbar)
{
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
    uint32_t b = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    uint64_t map_addr = reinterpret_cast<uint64_t>(tensor_map);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%3, %4}], [%2];"
        :: "r"(s), "l"(map_addr), "r"(b), "r"(coord_col), "r"(coord_row));
}

__device__ __forceinline__ void tma_load_2d_multicast(
    void* smem_dst, const CUtensorMap* tensor_map,
    int32_t coord_row, int32_t coord_col, uint64_t* mbar, uint16_t cta_mask)
{
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
    uint32_t b = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    uint64_t map_addr = reinterpret_cast<uint64_t>(tensor_map);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster "
        "[%0], [%1, {%3, %4}], [%2], %5;"
        :: "r"(s), "l"(map_addr), "r"(b),
           "r"(coord_col), "r"(coord_row), "h"(cta_mask));
}

// ---------------------------------------------------------------------------
// Cluster-aware streaming kernel: each cluster's leader issues all TMA loads
// via multicast (or plain TMA if CLUSTER_SIZE==1); each CTA in the cluster
// receives the same tile sequence into its own smem ring buffer.

template <int CLUSTER_SIZE>
__global__ void __cluster_dims__(CLUSTER_SIZE, 1, 1)
tma_multicast_kernel(
    const __grid_constant__ CUtensorMap tensor_map,
    int n_tiles_per_cluster,
    uint64_t* sink)
{
    constexpr uint16_t CTA_MASK = (CLUSTER_SIZE > 1)
        ? static_cast<uint16_t>((1u << CLUSTER_SIZE) - 1u)
        : 0;

    extern __shared__ uint8_t smem_raw[];
    uint8_t*  smem_tiles = smem_raw;
    uint64_t* smem_bars  = reinterpret_cast<uint64_t*>(smem_raw + DEPTH * TILE_BYTES);

    int tid = threadIdx.x;
    uint32_t cluster_rank = (CLUSTER_SIZE > 1) ? get_cluster_rank() : 0u;
    int cluster_id = blockIdx.x / CLUSTER_SIZE;

    if (tid == 0) {
        #pragma unroll
        for (int d = 0; d < DEPTH; ++d) {
            mbarrier_init(&smem_bars[d], 1);
        }
    }
    __syncthreads();

    // Cluster-wide sync to ensure all CTAs in the cluster have their mbarriers
    // initialised before the leader starts issuing multicast loads.
    if constexpr (CLUSTER_SIZE > 1) cluster_sync();

    int tile_base = cluster_id * n_tiles_per_cluster;

    if (tid == 0) {
        for (int i = 0; i < n_tiles_per_cluster; ++i) {
            int slot = i % DEPTH;
            uint64_t* bar = &smem_bars[slot];
            uint8_t*  dst = smem_tiles + slot * TILE_BYTES;

            if (i >= DEPTH) {
                uint32_t prev_phase = ((i / DEPTH) - 1) & 1u;
                mbarrier_wait(bar, prev_phase);
            }

            mbarrier_arrive_expect_tx(bar, TILE_BYTES);

            int linear = tile_base + i;
            int row_t = linear / N_COLS_TILES;
            int col_t = linear % N_COLS_TILES;
            int coord_row = row_t * BOX_ROWS;
            int coord_col = col_t * BOX_COLS;

            if constexpr (CLUSTER_SIZE > 1) {
                if (cluster_rank == 0) {
                    tma_load_2d_multicast(dst, &tensor_map, coord_row, coord_col, bar, CTA_MASK);
                }
            } else {
                tma_load_2d(dst, &tensor_map, coord_row, coord_col, bar);
            }
        }
        // Drain.
        int drain_start = (n_tiles_per_cluster > DEPTH) ? (n_tiles_per_cluster - DEPTH) : 0;
        for (int i = drain_start; i < n_tiles_per_cluster; ++i) {
            int slot = i % DEPTH;
            uint32_t phase = (i / DEPTH) & 1u;
            mbarrier_wait(&smem_bars[slot], phase);
        }
    }
    __syncthreads();

    // Anti-DCE: hash one element of the last loaded tile.
    if (tid == 0 && blockIdx.x == 0 && n_tiles_per_cluster > 0) {
        const uint16_t* last = reinterpret_cast<const uint16_t*>(
            smem_tiles + ((n_tiles_per_cluster - 1) % DEPTH) * TILE_BYTES);
        atomicAdd((unsigned long long*)sink, (unsigned long long)last[0]);
    }
}

// ---------------------------------------------------------------------------
// Driver: build tensor map, time the kernel, report bandwidth.

template <int CLUSTER_SIZE>
double run_config(const __nv_bfloat16* d_src, uint64_t* d_sink,
                  int warmup, int timed,
                  size_t* out_dram_bytes, size_t* out_smem_bytes_delivered)
{
    static_assert(CLUSTER_SIZE == 1 || CLUSTER_SIZE == 2 || CLUSTER_SIZE == 4,
                  "supported cluster sizes: 1, 2, 4");

    CUtensorMap tensor_map = {};
    cuuint64_t global_dim[2]    = {SRC_COLS, SRC_ROWS};
    cuuint64_t global_stride[1] = {SRC_COLS * sizeof(__nv_bfloat16)};
    cuuint32_t box_dim[2]       = {BOX_COLS, BOX_ROWS};
    cuuint32_t elem_stride[2]   = {1, 1};

    CU_CHECK(cuTensorMapEncodeTiled(
        &tensor_map,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2,
        const_cast<__nv_bfloat16*>(d_src),
        global_dim, global_stride, box_dim, elem_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,        // matches BOX_COLS=64 bf16 = 128 B
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    // Hold per-CTA workload constant across cluster sizes so the multicast
    // benefit shows as a halving (or quartering) of DRAM bytes / time, not as
    // doubling of per-CTA work. Each cluster's leader issues
    // n_tiles_per_cluster = N_TILES_PER_CTA_TARGET unique tiles and they are
    // multicast to all CLUSTER_SIZE CTAs. DRAM bytes per launch then shrink
    // as CLUSTER_SIZE grows.
    constexpr int N_TILES_PER_CTA_TARGET = 124;       // matches the c=1 baseline
    int n_clusters = N_SMS / CLUSTER_SIZE;
    int n_tiles_per_cluster = N_TILES_PER_CTA_TARGET;
    (void)0;

    size_t smem_bytes = DEPTH * TILE_BYTES + DEPTH * sizeof(uint64_t);
    cudaFuncSetAttribute(
        (const void*)tma_multicast_kernel<CLUSTER_SIZE>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        int(smem_bytes));

    auto launch = [&]() {
        tma_multicast_kernel<CLUSTER_SIZE><<<N_SMS, 32, smem_bytes>>>(
            tensor_map, n_tiles_per_cluster, d_sink);
    };

    for (int i = 0; i < warmup; ++i) launch();
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    std::vector<float> ms;
    ms.reserve(timed);
    for (int i = 0; i < timed; ++i) {
        cudaEventRecord(e0);
        launch();
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float t = 0;
        cudaEventElapsedTime(&t, e0, e1);
        ms.push_back(t);
    }
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    std::sort(ms.begin(), ms.end());
    double median_ms = ms[ms.size() / 2];

    // DRAM bytes per launch: each cluster issues n_tiles_per_cluster unique
    // TMA loads regardless of cluster size — multicast is a fan-out at L2/L1TEX,
    // not at DRAM. So DRAM = n_clusters × n_tiles_per_cluster × tile_bytes.
    size_t dram_bytes = size_t(n_clusters) * n_tiles_per_cluster * TILE_BYTES;
    // smem-bytes delivered: each CTA in the cluster receives the same tiles, so
    // delivered = N_SMS × n_tiles_per_cluster × tile_bytes.
    size_t smem_delivered = size_t(N_SMS) * n_tiles_per_cluster * TILE_BYTES;
    *out_dram_bytes = dram_bytes;
    *out_smem_bytes_delivered = smem_delivered;
    return median_ms;
}

int main(int argc, char** argv) {
    CU_CHECK(cuInit(0));

    __nv_bfloat16* d_src;
    CUDA_CHECK(cudaMalloc(&d_src, SRC_BYTES));
    CUDA_CHECK(cudaMemset(d_src, 0x3F, SRC_BYTES));

    uint64_t* d_sink;
    CUDA_CHECK(cudaMalloc(&d_sink, sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(d_sink, 0, sizeof(uint64_t)));

    int warmup = 5, timed = 20;
    printf("config,cluster_size,n_clusters,n_tiles_per_cluster,dram_bytes,smem_bytes_delivered,median_ms,dram_gbps,effective_smem_gbps,multicast_speedup\n");

    auto run = [&](const char* name, int C, double ms, size_t dram, size_t smem) {
        double dram_gbps = double(dram) / (ms * 1e-3) / 1e9;
        double smem_gbps = double(smem) / (ms * 1e-3) / 1e9;
        double speedup = double(C);    // theoretical ideal
        printf("%s,%d,%d,%d,%zu,%zu,%.4f,%.2f,%.2f,%.2f\n",
               name, C, N_SMS / C,
               int(smem / (size_t(N_SMS) * TILE_BYTES)),
               dram, smem, ms, dram_gbps, smem_gbps, speedup);
    };

    size_t dram, smem;
    double ms;

    ms = run_config<1>(d_src, d_sink, warmup, timed, &dram, &smem);
    run("c1_baseline",    1, ms, dram, smem);

    ms = run_config<2>(d_src, d_sink, warmup, timed, &dram, &smem);
    run("c2_multicast",   2, ms, dram, smem);

    ms = run_config<4>(d_src, d_sink, warmup, timed, &dram, &smem);
    run("c4_multicast",   4, ms, dram, smem);

    cudaFree(d_src); cudaFree(d_sink);
    return 0;
}
