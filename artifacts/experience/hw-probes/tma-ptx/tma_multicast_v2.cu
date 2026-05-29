// Cutlass-free TMA cluster-multicast follow-up probe (v2).
//
// Addresses the 6 open_questions of the v1 probe (2026-04-30-tma-multicast.md):
//   Q1: ncu metric breakdown for sub-linear scaling (separate runner — ncu_multicast.sh)
//   Q2: cluster sizes C ∈ {1, 2, 4, 8, 16}
//   Q3: multi-producer-warp issue (1, 2, 4 producer warps per CTA, each with own ring)
//   Q4: L2 promotion sweep — NONE, 64B, 128B, 256B at C=2
//   Q5: cross-CTA empty-mbarrier signalling via mapa + mbarrier.arrive.shared::cluster.b64
//   Q6: numeric correctness — deterministic source + per-CTA tile dump + host compare
//
// Build:
//   nvcc -std=c++17 -O3 -gencode=arch=compute_90a,code=sm_90a -lineinfo \
//        tma_multicast_v2.cu -lcuda -o tma_multicast_v2

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
constexpr int BOX_COLS = 64;
constexpr int N_COLS_TILES = SRC_COLS / BOX_COLS;
constexpr int TILE_BYTES = BOX_ROWS * BOX_COLS * sizeof(__nv_bfloat16);
constexpr int DEPTH = 4;
constexpr int N_TILES_PER_CTA_TARGET = 124;

// ---------------------------------------------------------------------------
// PTX wrappers.

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
        "  L_W_%=:\n"
        "  mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
        "  @P1 bra D_W_%=;\n"
        "  bra L_W_%=;\n"
        "  D_W_%=: }"
        :: "r"(b), "r"(phase));
}

// Cross-CTA mbarrier arrive via DSMEM: maps a smem address into target CTA's
// cluster-shared address space, then arrives. The .release.cluster qualifier
// is required by PTX ISA for cluster-scope mbarrier arrives.
__device__ __forceinline__ void mbarrier_arrive_cluster_remote(uint64_t* local_bar_addr, uint32_t target_cta) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(local_bar_addr));
    uint32_t mapped;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
        : "=r"(mapped) : "r"(addr), "r"(target_cta));
    asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];"
        :: "r"(mapped));
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
// Q2 + Q3 + Q4: cluster size × producer-warp count × L2 promotion.
// Each producer warp w has its own ring buffer of DEPTH slots and processes
// tiles in stride PROD_WARPS.

template <int CLUSTER_SIZE, int PROD_WARPS>
__global__ void __cluster_dims__(CLUSTER_SIZE, 1, 1)
tma_v2_kernel(
    const __grid_constant__ CUtensorMap tensor_map,
    int n_tiles_per_cluster,
    uint64_t* sink)
{
    constexpr uint16_t CTA_MASK = (CLUSTER_SIZE > 1)
        ? static_cast<uint16_t>((1u << CLUSTER_SIZE) - 1u) : 0;
    constexpr int N_BARS = PROD_WARPS * DEPTH;
    constexpr int TILES_BUF_BYTES = N_BARS * TILE_BYTES;

    extern __shared__ uint8_t smem_raw[];
    uint8_t*  smem_tiles = smem_raw;
    uint64_t* smem_bars  = reinterpret_cast<uint64_t*>(smem_raw + TILES_BUF_BYTES);

    int warp_id = threadIdx.x >> 5;
    int lane    = threadIdx.x & 31;

    // Each producer warp's lane 0 inits its own DEPTH mbarriers.
    if (warp_id < PROD_WARPS && lane == 0) {
        #pragma unroll
        for (int d = 0; d < DEPTH; ++d) mbarrier_init(&smem_bars[warp_id * DEPTH + d], 1);
    }
    __syncthreads();
    if constexpr (CLUSTER_SIZE > 1) cluster_sync();

    uint32_t cluster_rank = (CLUSTER_SIZE > 1) ? get_cluster_rank() : 0u;
    int cluster_id = blockIdx.x / CLUSTER_SIZE;
    int tile_base = cluster_id * n_tiles_per_cluster;

    if (warp_id < PROD_WARPS && lane == 0) {
        int my_tiles = (n_tiles_per_cluster - warp_id + PROD_WARPS - 1) / PROD_WARPS;
        for (int i = 0; i < my_tiles; ++i) {
            int slot = i % DEPTH;
            int slot_idx = warp_id * DEPTH + slot;
            uint64_t* bar = &smem_bars[slot_idx];
            uint8_t*  dst = smem_tiles + slot_idx * TILE_BYTES;

            if (i >= DEPTH) {
                uint32_t prev_phase = ((i / DEPTH) - 1) & 1u;
                mbarrier_wait(bar, prev_phase);
            }
            mbarrier_arrive_expect_tx(bar, TILE_BYTES);

            int linear = tile_base + i * PROD_WARPS + warp_id;
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
        int drain_start = (my_tiles > DEPTH) ? (my_tiles - DEPTH) : 0;
        for (int i = drain_start; i < my_tiles; ++i) {
            int slot = i % DEPTH;
            uint32_t phase = (i / DEPTH) & 1u;
            mbarrier_wait(&smem_bars[warp_id * DEPTH + slot], phase);
        }
    }
    __syncthreads();

    if (threadIdx.x == 0 && blockIdx.x == 0 && n_tiles_per_cluster > 0) {
        const uint16_t* last = reinterpret_cast<const uint16_t*>(smem_tiles);
        atomicAdd((unsigned long long*)sink, (unsigned long long)last[0]);
    }
}

// ---------------------------------------------------------------------------
// Q5: cross-CTA empty-mbarrier producer/consumer pipeline.
//
// Each CTA has bar_full[D] (signaled by multicast TMA) and bar_empty[D]
// (signaled by all CLUSTER_SIZE CTAs after they consume; cross-CTA via
// mbarrier.arrive.shared::cluster). Leader waits on its local bar_empty[s]
// (which expects CLUSTER_SIZE arrivals from the cluster) before re-issuing.
template <int CLUSTER_SIZE>
__global__ void __cluster_dims__(CLUSTER_SIZE, 1, 1)
tma_v2_xcta_empty_kernel(
    const __grid_constant__ CUtensorMap tensor_map,
    int n_tiles_per_cluster,
    uint64_t* sink)
{
    constexpr uint16_t CTA_MASK = static_cast<uint16_t>((1u << CLUSTER_SIZE) - 1u);

    extern __shared__ uint8_t smem_raw[];
    uint8_t*  smem_tiles = smem_raw;
    uint64_t* bar_full   = reinterpret_cast<uint64_t*>(smem_raw + DEPTH * TILE_BYTES);
    uint64_t* bar_empty  = bar_full + DEPTH;

    int tid = threadIdx.x;
    uint32_t cluster_rank = get_cluster_rank();
    int cluster_id = blockIdx.x / CLUSTER_SIZE;
    int tile_base = cluster_id * n_tiles_per_cluster;

    if (tid == 0) {
        for (int d = 0; d < DEPTH; ++d) {
            mbarrier_init(&bar_full[d],  1);
            mbarrier_init(&bar_empty[d], CLUSTER_SIZE);
        }
    }
    __syncthreads();
    cluster_sync();

    if (tid == 0) {
        for (int i = 0; i < n_tiles_per_cluster; ++i) {
            int slot = i % DEPTH;

            // Each CTA waits on its local bar_empty[slot] before consuming a
            // refill into the same slot. expected_arrival was init'd to
            // CLUSTER_SIZE; each CTA arrives once per iteration via
            // mbarrier_arrive_cluster_remote on the LEADER's bar_empty.
            // The first DEPTH iterations skip this (no prior load to drain).
            if (i >= DEPTH) {
                uint32_t prev_phase = ((i / DEPTH) - 1) & 1u;
                if (cluster_rank == 0) {
                    mbarrier_wait(&bar_empty[slot], prev_phase);
                }
            }

            // Fence: leader's empty-wait must complete before its arrive_expect_tx
            // on bar_full re-uses the slot. Cluster-sync each iteration is too
            // expensive; rely on the per-iteration arrive_expect_tx ordering.
            mbarrier_arrive_expect_tx(&bar_full[slot], TILE_BYTES);

            int linear = tile_base + i;
            int row_t = linear / N_COLS_TILES;
            int col_t = linear % N_COLS_TILES;

            if (cluster_rank == 0) {
                tma_load_2d_multicast(smem_tiles + slot * TILE_BYTES, &tensor_map,
                                      row_t * BOX_ROWS, col_t * BOX_COLS,
                                      &bar_full[slot], CTA_MASK);
            }

            // Wait for the load on this slot (each CTA waits locally).
            uint32_t phase = (i / DEPTH) & 1u;
            mbarrier_wait(&bar_full[slot], phase);

            // After "consuming" (no actual consumer in this microbench), every
            // CTA arrives remotely on the leader's bar_empty[slot]. The leader
            // also arrives locally (rank 0 → target_cta 0).
            uint32_t my_cta_in_cluster = cluster_rank;
            // mapa to leader CTA (cluster-rank 0)
            mbarrier_arrive_cluster_remote(&bar_empty[slot], 0);
            (void)my_cta_in_cluster;
        }
    }
    __syncthreads();

    if (tid == 0 && blockIdx.x == 0 && n_tiles_per_cluster > 0) {
        const uint16_t* last = reinterpret_cast<const uint16_t*>(
            smem_tiles + ((n_tiles_per_cluster - 1) % DEPTH) * TILE_BYTES);
        atomicAdd((unsigned long long*)sink, (unsigned long long)last[0]);
    }
}

// ---------------------------------------------------------------------------
// Q6: numeric correctness — verify mode.
//
// Same kernel as tma_v2_kernel<C, 1>, but after each load completes, lane 0
// scans the whole loaded tile word-by-word and increments a per-CTA mismatch
// counter if any bf16 element doesn't match the expected source pattern. The
// source is initialised host-side to data[r * SRC_COLS + c] = bf16(r * 100 + c)
// (low-order 16 bits, masked to fit bf16's representable range).
template <int CLUSTER_SIZE>
__global__ void __cluster_dims__(CLUSTER_SIZE, 1, 1)
tma_v2_verify_kernel(
    const __grid_constant__ CUtensorMap tensor_map,
    int n_tiles_per_cluster,
    int* per_cta_mismatches,
    int* per_cta_total)
{
    constexpr uint16_t CTA_MASK = (CLUSTER_SIZE > 1)
        ? static_cast<uint16_t>((1u << CLUSTER_SIZE) - 1u) : 0;

    extern __shared__ uint8_t smem_raw[];
    uint8_t*  smem_tiles = smem_raw;
    uint64_t* smem_bars  = reinterpret_cast<uint64_t*>(smem_raw + DEPTH * TILE_BYTES);

    int tid = threadIdx.x;
    uint32_t cluster_rank = (CLUSTER_SIZE > 1) ? get_cluster_rank() : 0u;
    int cluster_id = blockIdx.x / CLUSTER_SIZE;

    if (tid == 0) {
        for (int d = 0; d < DEPTH; ++d) mbarrier_init(&smem_bars[d], 1);
    }
    __syncthreads();
    if constexpr (CLUSTER_SIZE > 1) cluster_sync();

    int tile_base = cluster_id * n_tiles_per_cluster;
    int local_mismatches = 0;
    int local_total = 0;

    for (int i = 0; i < n_tiles_per_cluster; ++i) {
        int slot = i % DEPTH;
        if (tid == 0 && i >= DEPTH) {
            uint32_t prev_phase = ((i / DEPTH) - 1) & 1u;
            mbarrier_wait(&smem_bars[slot], prev_phase);
        }
        if (tid == 0) {
            mbarrier_arrive_expect_tx(&smem_bars[slot], TILE_BYTES);
            int linear = tile_base + i;
            int row_t = linear / N_COLS_TILES;
            int col_t = linear % N_COLS_TILES;
            if constexpr (CLUSTER_SIZE > 1) {
                if (cluster_rank == 0) {
                    tma_load_2d_multicast(smem_tiles + slot * TILE_BYTES, &tensor_map,
                                          row_t * BOX_ROWS, col_t * BOX_COLS,
                                          &smem_bars[slot], CTA_MASK);
                }
            } else {
                tma_load_2d(smem_tiles + slot * TILE_BYTES, &tensor_map,
                            row_t * BOX_ROWS, col_t * BOX_COLS,
                            &smem_bars[slot]);
            }
        }
        // All threads wait
        if (tid == 0) {
            uint32_t phase = (i / DEPTH) & 1u;
            mbarrier_wait(&smem_bars[slot], phase);
        }
        __syncthreads();

        // Cooperative tile compare against expected source pattern.
        int linear = tile_base + i;
        int row_t = linear / N_COLS_TILES;
        int col_t = linear % N_COLS_TILES;
        int row0 = row_t * BOX_ROWS;
        int col0 = col_t * BOX_COLS;
        const __nv_bfloat16* tile = reinterpret_cast<const __nv_bfloat16*>(
            smem_tiles + slot * TILE_BYTES);
        for (int idx = tid; idx < BOX_ROWS * BOX_COLS; idx += blockDim.x) {
            int r = idx / BOX_COLS;
            int c = idx % BOX_COLS;
            int gr = row0 + r;
            int gc = col0 + c;
            __nv_bfloat16 expected = __float2bfloat16(float((gr * 31 + gc * 7) & 0x3FF));
            __nv_bfloat16 got = tile[r * BOX_COLS + c];
            if (__bfloat162float(got) != __bfloat162float(expected)) ++local_mismatches;
            ++local_total;
        }
        __syncthreads();
    }

    // Reduce per-CTA via atomic into the global per-CTA slots
    atomicAdd(&per_cta_mismatches[blockIdx.x], local_mismatches);
    atomicAdd(&per_cta_total[blockIdx.x], local_total);
}

// ---------------------------------------------------------------------------
// Driver helpers.

CUtensorMap make_tensor_map(const __nv_bfloat16* d_src, CUtensorMapL2promotion l2_promo,
                             CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_128B) {
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
        swizzle,
        l2_promo,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    return tensor_map;
}

const char* l2_promo_name(CUtensorMapL2promotion p) {
    switch (p) {
        case CU_TENSOR_MAP_L2_PROMOTION_NONE: return "NONE";
        case CU_TENSOR_MAP_L2_PROMOTION_L2_64B: return "64B";
        case CU_TENSOR_MAP_L2_PROMOTION_L2_128B: return "128B";
        case CU_TENSOR_MAP_L2_PROMOTION_L2_256B: return "256B";
        default: return "?";
    }
}

template <int CLUSTER_SIZE, int PROD_WARPS>
double run_v2(const __nv_bfloat16* d_src, uint64_t* d_sink,
              CUtensorMapL2promotion l2_promo,
              int warmup, int timed,
              size_t* out_dram_bytes, size_t* out_smem_bytes_delivered)
{
    auto tensor_map = make_tensor_map(d_src, l2_promo);

    // Grid must be a multiple of cluster size; round down.
    int n_clusters = N_SMS / CLUSTER_SIZE;
    int grid = n_clusters * CLUSTER_SIZE;
    int n_tiles_per_cluster = N_TILES_PER_CTA_TARGET;

    constexpr int N_BARS = PROD_WARPS * DEPTH;
    size_t smem_bytes = N_BARS * TILE_BYTES + N_BARS * sizeof(uint64_t);

    cudaFuncSetAttribute(
        (const void*)tma_v2_kernel<CLUSTER_SIZE, PROD_WARPS>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem_bytes));
    if (CLUSTER_SIZE > 8) {
        cudaFuncSetAttribute(
            (const void*)tma_v2_kernel<CLUSTER_SIZE, PROD_WARPS>,
            cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    }

    int n_threads = PROD_WARPS * 32;
    auto launch = [&]() {
        tma_v2_kernel<CLUSTER_SIZE, PROD_WARPS><<<grid, n_threads, smem_bytes>>>(
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

    size_t dram_bytes = size_t(n_clusters) * n_tiles_per_cluster * TILE_BYTES;
    size_t smem_delivered = size_t(grid) * n_tiles_per_cluster * TILE_BYTES;
    *out_dram_bytes = dram_bytes;
    *out_smem_bytes_delivered = smem_delivered;
    return median_ms;
}

template <int CLUSTER_SIZE>
double run_xcta(const __nv_bfloat16* d_src, uint64_t* d_sink,
                int warmup, int timed,
                size_t* out_dram_bytes, size_t* out_smem_bytes_delivered)
{
    auto tensor_map = make_tensor_map(d_src, CU_TENSOR_MAP_L2_PROMOTION_NONE);
    // Grid must be a multiple of cluster size; round down.
    int n_clusters = N_SMS / CLUSTER_SIZE;
    int grid = n_clusters * CLUSTER_SIZE;
    int n_tiles_per_cluster = N_TILES_PER_CTA_TARGET;

    size_t smem_bytes = DEPTH * TILE_BYTES + 2 * DEPTH * sizeof(uint64_t);
    cudaFuncSetAttribute(
        (const void*)tma_v2_xcta_empty_kernel<CLUSTER_SIZE>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem_bytes));
    if (CLUSTER_SIZE > 8) {
        cudaFuncSetAttribute(
            (const void*)tma_v2_xcta_empty_kernel<CLUSTER_SIZE>,
            cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    }

    auto launch = [&]() {
        tma_v2_xcta_empty_kernel<CLUSTER_SIZE><<<grid, 32, smem_bytes>>>(
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

    *out_dram_bytes = size_t(n_clusters) * n_tiles_per_cluster * TILE_BYTES;
    *out_smem_bytes_delivered = size_t(grid) * n_tiles_per_cluster * TILE_BYTES;
    return median_ms;
}

template <int CLUSTER_SIZE>
void verify_correctness(const __nv_bfloat16* d_src, int* out_total_mismatches, int* out_total_compared) {
    // Verify with SWIZZLE_NONE so the smem layout matches the source layout
    // element-by-element. SWIZZLE_128B (used by the throughput runs) permutes
    // smem positions and the verify kernel's direct compare would fail.
    auto tensor_map = make_tensor_map(d_src, CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                       CU_TENSOR_MAP_SWIZZLE_NONE);
    int grid = (N_SMS / CLUSTER_SIZE) * CLUSTER_SIZE;
    int n_tiles_per_cluster = N_TILES_PER_CTA_TARGET;

    int* d_mis;  int* d_tot;
    CUDA_CHECK(cudaMalloc(&d_mis, grid * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_tot, grid * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_mis, 0, grid * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_tot, 0, grid * sizeof(int)));

    size_t smem_bytes = DEPTH * TILE_BYTES + DEPTH * sizeof(uint64_t);
    cudaFuncSetAttribute(
        (const void*)tma_v2_verify_kernel<CLUSTER_SIZE>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem_bytes));
    if (CLUSTER_SIZE > 8) {
        cudaFuncSetAttribute(
            (const void*)tma_v2_verify_kernel<CLUSTER_SIZE>,
            cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    }

    tma_v2_verify_kernel<CLUSTER_SIZE><<<grid, 128, smem_bytes>>>(
        tensor_map, n_tiles_per_cluster, d_mis, d_tot);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int> h_mis(grid), h_tot(grid);
    CUDA_CHECK(cudaMemcpy(h_mis.data(), d_mis, grid * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_tot.data(), d_tot, grid * sizeof(int), cudaMemcpyDeviceToHost));

    long long total_mis = 0, total_cmp = 0;
    for (int i = 0; i < grid; ++i) { total_mis += h_mis[i]; total_cmp += h_tot[i]; }
    *out_total_mismatches = int(total_mis);
    *out_total_compared = int(total_cmp);
    cudaFree(d_mis); cudaFree(d_tot);
}

int main() {
    CU_CHECK(cuInit(0));

    // Initialise source with deterministic pattern that the verify kernel checks.
    __nv_bfloat16* d_src;
    CUDA_CHECK(cudaMalloc(&d_src, SRC_BYTES));
    {
        std::vector<__nv_bfloat16> h(SRC_ROWS * SRC_COLS);
        for (int r = 0; r < SRC_ROWS; ++r) {
            for (int c = 0; c < SRC_COLS; ++c) {
                h[r * SRC_COLS + c] = __float2bfloat16(float((r * 31 + c * 7) & 0x3FF));
            }
        }
        CUDA_CHECK(cudaMemcpy(d_src, h.data(), SRC_BYTES, cudaMemcpyHostToDevice));
    }

    uint64_t* d_sink;
    CUDA_CHECK(cudaMalloc(&d_sink, sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(d_sink, 0, sizeof(uint64_t)));

    int warmup = 5, timed = 20;

    printf("section,config,cluster_size,prod_warps,l2_promo,n_clusters,dram_bytes,smem_bytes_delivered,median_ms,dram_gbps,effective_smem_gbps\n");

    auto report = [&](const char* section, const char* name,
                      int C, int PW, const char* l2,
                      double ms, size_t dram, size_t smem) {
        int n_clusters = N_SMS / C;
        double dram_gbps = double(dram) / (ms * 1e-3) / 1e9;
        double smem_gbps = double(smem) / (ms * 1e-3) / 1e9;
        printf("%s,%s,%d,%d,%s,%d,%zu,%zu,%.4f,%.2f,%.2f\n",
               section, name, C, PW, l2, n_clusters, dram, smem, ms, dram_gbps, smem_gbps);
    };

    size_t dram, smem;  double ms;

    // ---- Q2: Cluster size sweep, PROD_WARPS=1, L2=NONE ----
    ms = run_v2<1, 1>(d_src, d_sink, CU_TENSOR_MAP_L2_PROMOTION_NONE, warmup, timed, &dram, &smem);
    report("cluster_sweep", "c1",  1,  1, "NONE", ms, dram, smem);
    ms = run_v2<2, 1>(d_src, d_sink, CU_TENSOR_MAP_L2_PROMOTION_NONE, warmup, timed, &dram, &smem);
    report("cluster_sweep", "c2",  2,  1, "NONE", ms, dram, smem);
    ms = run_v2<4, 1>(d_src, d_sink, CU_TENSOR_MAP_L2_PROMOTION_NONE, warmup, timed, &dram, &smem);
    report("cluster_sweep", "c4",  4,  1, "NONE", ms, dram, smem);
    ms = run_v2<8, 1>(d_src, d_sink, CU_TENSOR_MAP_L2_PROMOTION_NONE, warmup, timed, &dram, &smem);
    report("cluster_sweep", "c8",  8,  1, "NONE", ms, dram, smem);
    ms = run_v2<16, 1>(d_src, d_sink, CU_TENSOR_MAP_L2_PROMOTION_NONE, warmup, timed, &dram, &smem);
    report("cluster_sweep", "c16", 16, 1, "NONE", ms, dram, smem);

    // ---- Q3: Multi-producer-warp at C=2 ----
    ms = run_v2<2, 1>(d_src, d_sink, CU_TENSOR_MAP_L2_PROMOTION_NONE, warmup, timed, &dram, &smem);
    report("prod_warp_sweep", "c2_pw1", 2, 1, "NONE", ms, dram, smem);
    ms = run_v2<2, 2>(d_src, d_sink, CU_TENSOR_MAP_L2_PROMOTION_NONE, warmup, timed, &dram, &smem);
    report("prod_warp_sweep", "c2_pw2", 2, 2, "NONE", ms, dram, smem);
    ms = run_v2<2, 4>(d_src, d_sink, CU_TENSOR_MAP_L2_PROMOTION_NONE, warmup, timed, &dram, &smem);
    report("prod_warp_sweep", "c2_pw4", 2, 4, "NONE", ms, dram, smem);

    // ---- Q4: L2 promotion sweep at C=2 ----
    ms = run_v2<2, 1>(d_src, d_sink, CU_TENSOR_MAP_L2_PROMOTION_NONE,    warmup, timed, &dram, &smem);
    report("l2_promo_sweep", "c2_l2_NONE", 2, 1, "NONE", ms, dram, smem);
    ms = run_v2<2, 1>(d_src, d_sink, CU_TENSOR_MAP_L2_PROMOTION_L2_64B,  warmup, timed, &dram, &smem);
    report("l2_promo_sweep", "c2_l2_64B",  2, 1, "64B",  ms, dram, smem);
    ms = run_v2<2, 1>(d_src, d_sink, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, warmup, timed, &dram, &smem);
    report("l2_promo_sweep", "c2_l2_128B", 2, 1, "128B", ms, dram, smem);
    ms = run_v2<2, 1>(d_src, d_sink, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, warmup, timed, &dram, &smem);
    report("l2_promo_sweep", "c2_l2_256B", 2, 1, "256B", ms, dram, smem);

    // ---- Q5: Cross-CTA empty mbarrier at C=2 ----
    ms = run_xcta<2>(d_src, d_sink, warmup, timed, &dram, &smem);
    report("xcta_empty_sweep", "c2_xcta_empty", 2, 1, "NONE", ms, dram, smem);

    // ---- Q6: Numeric correctness ----
    fprintf(stderr, "\n=== Numeric correctness verification ===\n");
    int mis, cmp;
    verify_correctness<1>(d_src, &mis, &cmp);
    fprintf(stderr, "  C=1: %d / %d mismatches (%s)\n", mis, cmp,
            mis == 0 ? "PASS" : "FAIL");
    verify_correctness<2>(d_src, &mis, &cmp);
    fprintf(stderr, "  C=2 multicast: %d / %d mismatches (%s)\n", mis, cmp,
            mis == 0 ? "PASS" : "FAIL");
    verify_correctness<4>(d_src, &mis, &cmp);
    fprintf(stderr, "  C=4 multicast: %d / %d mismatches (%s)\n", mis, cmp,
            mis == 0 ? "PASS" : "FAIL");

    cudaFree(d_src); cudaFree(d_sink);
    return 0;
}
