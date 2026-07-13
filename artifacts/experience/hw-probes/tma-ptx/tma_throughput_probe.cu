// Cutlass-free TMA throughput probe.
//
// Sweeps cuTensorMapEncodeTiled parameter axes (swizzle mode, mbarrier-pipeline
// depth, box-row size) and reports per-config wall-clock + bandwidth.
//
// Workload per launch: each CTA streams ~1 MiB of bf16 through TMA into a smem
// ring buffer of depth D; 132 CTAs (= H200 SM count) cooperatively cover the
// 128 MiB source tensor exactly once. Source > L2 (~60 MiB on H200) so the test
// is DRAM-bandwidth-bound, not L2-resident.
//
// Build:
//   nvcc -std=c++17 -O3 -gencode=arch=compute_90a,code=sm_90a -lineinfo \
//        tma_throughput_probe.cu -lcuda -o tma_throughput_probe

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

// ---------------------------------------------------------------------------
// Source tensor geometry: 16384 rows x 4096 cols of bf16 = 128 MiB.
// L2 on H200 is ~60 MiB, so a full sweep exceeds L2 and forces DRAM traffic.
constexpr int SRC_ROWS = 16384;
constexpr int SRC_COLS = 4096;       // 4096 bf16 = 8192 B per row
constexpr size_t SRC_BYTES = size_t(SRC_ROWS) * SRC_COLS * sizeof(__nv_bfloat16);
constexpr int N_SMS = 132;           // H200 SM count

// Box columns are per-swizzle: SWIZZLE_NB requires fast-axis == N bytes exactly.
//   NONE  → any (we use 64 bf16 = 128 B for parity with 128B-swizzle)
//   32B   → 16 bf16 (= 32 B)
//   64B   → 32 bf16 (= 64 B)
//   128B  → 64 bf16 (= 128 B)

// ---------------------------------------------------------------------------
// Inline-PTX wrappers (cutlass-free).

__device__ __forceinline__ void mbarrier_init(uint64_t* bar, uint32_t n_threads) {
    uint32_t b = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" :: "r"(b), "r"(n_threads));
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* bar, uint32_t expected_bytes) {
    uint32_t b = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
        :: "r"(b), "r"(expected_bytes));
}

__device__ __forceinline__ void mbarrier_wait(uint64_t* bar, uint32_t phase) {
    uint32_t b = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile(
        "{ .reg .pred P1;\n"
        "  LAB_WAIT_%=: \n"
        "  mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
        "  @P1 bra DONE_WAIT_%=; \n"
        "  bra LAB_WAIT_%=; \n"
        "  DONE_WAIT_%=: }\n"
        :: "r"(b), "r"(phase));
}

__device__ __forceinline__ void tma_load_2d(
    void* smem_dst,
    const CUtensorMap* tensor_map,
    int32_t coord_row, int32_t coord_col,
    uint64_t* mbar)
{
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
    uint32_t b = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    uint64_t map_addr = reinterpret_cast<uint64_t>(tensor_map);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%3, %4}], [%2];\n"
        :: "r"(s), "l"(map_addr), "r"(b),
           "r"(coord_col), "r"(coord_row));
}

// ---------------------------------------------------------------------------
// Templated streaming kernel: each CTA loads its slice of tiles through a
// ring buffer of depth DEPTH. Box geometry BOX_ROWS x BOX_COLS bf16.

template <int DEPTH, int BOX_ROWS, int BOX_COLS>
__global__ void tma_stream_kernel(
    const __grid_constant__ CUtensorMap tensor_map,
    int n_tiles_per_cta,
    uint64_t* sink)
{
    constexpr int TILE_BYTES = BOX_ROWS * BOX_COLS * sizeof(__nv_bfloat16);
    constexpr int N_COLS_TILES = SRC_COLS / BOX_COLS;
    extern __shared__ uint8_t smem_raw[];

    // Layout: [DEPTH * tile_bytes][DEPTH * 8 B mbarrier]
    uint8_t*  smem_tiles = smem_raw;
    uint64_t* smem_bars  = reinterpret_cast<uint64_t*>(smem_raw + DEPTH * TILE_BYTES);

    int tid = threadIdx.x;

    if (tid == 0) {
        #pragma unroll
        for (int d = 0; d < DEPTH; ++d) {
            mbarrier_init(&smem_bars[d], 1);
        }
    }
    __syncthreads();

    int cta = blockIdx.x;
    int tile_base = cta * n_tiles_per_cta;

    if (tid == 0) {
        // Pipelined issue + wait. Caller guarantees tile_base + n_tiles_per_cta
        // is in-bounds for the source tensor.
        for (int i = 0; i < n_tiles_per_cta; ++i) {
            int slot = i % DEPTH;
            uint64_t* bar = &smem_bars[slot];
            uint8_t* dst = smem_tiles + slot * TILE_BYTES;

            if (i >= DEPTH) {
                uint32_t prev_phase = ((i / DEPTH) - 1) & 1u;
                mbarrier_wait(bar, prev_phase);
            }

            int linear = tile_base + i;
            int row_t = linear / N_COLS_TILES;
            int col_t = linear % N_COLS_TILES;
            int coord_row = row_t * BOX_ROWS;
            int coord_col = col_t * BOX_COLS;

            mbarrier_arrive_expect_tx(bar, TILE_BYTES);
            tma_load_2d(dst, &tensor_map, coord_row, coord_col, bar);
        }
        // Drain.
        int drain_start = (n_tiles_per_cta > DEPTH) ? (n_tiles_per_cta - DEPTH) : 0;
        for (int i = drain_start; i < n_tiles_per_cta; ++i) {
            int slot = i % DEPTH;
            uint32_t phase = (i / DEPTH) & 1u;
            mbarrier_wait(&smem_bars[slot], phase);
        }
    }
    __syncthreads();

    // Anti-DCE: hash one element of the last loaded tile into a global sink.
    if (tid == 0 && cta == 0 && n_tiles_per_cta > 0) {
        const uint16_t* last = reinterpret_cast<const uint16_t*>(
            smem_tiles + ((n_tiles_per_cta - 1) % DEPTH) * TILE_BYTES);
        atomicAdd((unsigned long long*)sink, (unsigned long long)last[0]);
    }
}

// ---------------------------------------------------------------------------
// Driver: encode tensor map with a given swizzle and box-row count, time the
// kernel for some (DEPTH, BOX_ROWS) instantiation.

template <int DEPTH, int BOX_ROWS, int BOX_COLS>
double run_config(const __nv_bfloat16* d_src, uint64_t* d_sink,
                  CUtensorMapSwizzle swizzle, int warmup, int timed)
{
    constexpr int TILE_BYTES = BOX_ROWS * BOX_COLS * sizeof(__nv_bfloat16);
    constexpr int N_COLS_TILES = SRC_COLS / BOX_COLS;

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
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    int n_row_tiles = SRC_ROWS / BOX_ROWS;
    int n_total_tiles = n_row_tiles * N_COLS_TILES;
    int n_tiles_per_cta = n_total_tiles / N_SMS;

    size_t smem_bytes = DEPTH * TILE_BYTES + DEPTH * sizeof(uint64_t);
    cudaFuncSetAttribute(
        (const void*)tma_stream_kernel<DEPTH, BOX_ROWS, BOX_COLS>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        int(smem_bytes));

    for (int i = 0; i < warmup; ++i) {
        tma_stream_kernel<DEPTH, BOX_ROWS, BOX_COLS><<<N_SMS, 32, smem_bytes>>>(
            tensor_map, n_tiles_per_cta, d_sink);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    std::vector<float> ms_per_launch;
    ms_per_launch.reserve(timed);
    for (int i = 0; i < timed; ++i) {
        cudaEventRecord(e0);
        tma_stream_kernel<DEPTH, BOX_ROWS, BOX_COLS><<<N_SMS, 32, smem_bytes>>>(
            tensor_map, n_tiles_per_cta, d_sink);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, e0, e1);
        ms_per_launch.push_back(ms);
    }
    cudaEventDestroy(e0); cudaEventDestroy(e1);

    std::sort(ms_per_launch.begin(), ms_per_launch.end());
    return ms_per_launch[ms_per_launch.size() / 2];
}

const char* swizzle_name(CUtensorMapSwizzle s) {
    switch (s) {
        case CU_TENSOR_MAP_SWIZZLE_NONE: return "NONE";
        case CU_TENSOR_MAP_SWIZZLE_32B:  return "32B";
        case CU_TENSOR_MAP_SWIZZLE_64B:  return "64B";
        case CU_TENSOR_MAP_SWIZZLE_128B: return "128B";
        default: return "?";
    }
}

int main(int argc, char** argv) {
    CU_CHECK(cuInit(0));

    // Allocate + fill source.
    __nv_bfloat16* d_src;
    CUDA_CHECK(cudaMalloc(&d_src, SRC_BYTES));
    CUDA_CHECK(cudaMemset(d_src, 0x3F, SRC_BYTES));   // arbitrary non-zero pattern

    uint64_t* d_sink;
    CUDA_CHECK(cudaMalloc(&d_sink, sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(d_sink, 0, sizeof(uint64_t)));

    int warmup = 5, timed = 20;

    printf("config,swizzle,depth,box_rows,box_cols,bytes,median_ms,gbps\n");

    auto report_inline = [&](const char* name, const char* sw, int depth,
                             int box_rows, int box_cols, double ms) {
        int n_row_tiles = SRC_ROWS / box_rows;
        int n_col_tiles = SRC_COLS / box_cols;
        int n_total_tiles = n_row_tiles * n_col_tiles;
        int n_tiles_per_cta = n_total_tiles / N_SMS;
        size_t tile_bytes = size_t(box_rows) * box_cols * sizeof(__nv_bfloat16);
        size_t bytes = size_t(N_SMS) * n_tiles_per_cta * tile_bytes;
        double gbps = double(bytes) / (ms * 1e-3) / 1e9;
        printf("%s,%s,%d,%d,%d,%zu,%.4f,%.2f\n",
               name, sw, depth, box_rows, box_cols, bytes, ms, gbps);
    };

    // ---- Main scan: 4 swizzles x 3 depths, box_cols matched to swizzle ----
    // For each swizzle the fast-axis tile (BOX_COLS) is fixed by the spec; the
    // slow-axis tile (BOX_ROWS) is held at 64 across the main scan.

#define DO(NAME, SW, DEPTH, BR, BC) \
    report_inline(NAME, swizzle_name(SW), DEPTH, BR, BC, \
        run_config<DEPTH, BR, BC>(d_src, d_sink, SW, warmup, timed))

    DO("swz-NONE_d1_r64",  CU_TENSOR_MAP_SWIZZLE_NONE,  1, 64, 64);
    DO("swz-NONE_d2_r64",  CU_TENSOR_MAP_SWIZZLE_NONE,  2, 64, 64);
    DO("swz-NONE_d4_r64",  CU_TENSOR_MAP_SWIZZLE_NONE,  4, 64, 64);

    DO("swz-32B_d1_r64",   CU_TENSOR_MAP_SWIZZLE_32B,   1, 64, 16);
    DO("swz-32B_d2_r64",   CU_TENSOR_MAP_SWIZZLE_32B,   2, 64, 16);
    DO("swz-32B_d4_r64",   CU_TENSOR_MAP_SWIZZLE_32B,   4, 64, 16);

    DO("swz-64B_d1_r64",   CU_TENSOR_MAP_SWIZZLE_64B,   1, 64, 32);
    DO("swz-64B_d2_r64",   CU_TENSOR_MAP_SWIZZLE_64B,   2, 64, 32);
    DO("swz-64B_d4_r64",   CU_TENSOR_MAP_SWIZZLE_64B,   4, 64, 32);

    DO("swz-128B_d1_r64",  CU_TENSOR_MAP_SWIZZLE_128B,  1, 64, 64);
    DO("swz-128B_d2_r64",  CU_TENSOR_MAP_SWIZZLE_128B,  2, 64, 64);
    DO("swz-128B_d4_r64",  CU_TENSOR_MAP_SWIZZLE_128B,  4, 64, 64);

    // ---- Box-row sweep at swizzle = 128B, depth = 4, box_cols = 64 ----
    DO("swz-128B_d4_r8",   CU_TENSOR_MAP_SWIZZLE_128B,  4,   8, 64);
    DO("swz-128B_d4_r16",  CU_TENSOR_MAP_SWIZZLE_128B,  4,  16, 64);
    DO("swz-128B_d4_r32",  CU_TENSOR_MAP_SWIZZLE_128B,  4,  32, 64);
    DO("swz-128B_d4_r128", CU_TENSOR_MAP_SWIZZLE_128B,  4, 128, 64);

#undef DO

    cudaFree(d_src); cudaFree(d_sink);
    return 0;
}
