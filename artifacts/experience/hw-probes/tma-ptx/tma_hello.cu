// Minimal cutlass-free TMA probe.
//
// Uses cuTensorMapEncodeTiled (CUDA Driver API) on the host to build a CUtensorMap,
// then issues `cp.async.bulk.tensor.2d.shared::cluster.global` from inline PTX on
// device, synchronizes via mbarrier, and verifies the loaded tile matches the host
// source via memcmp.
//
// What this proves:
//   * Hopper TMA load works without any cutlass/ or cute/ header.
//   * The cuTensorMapEncodeTiled + cp.async.bulk.tensor + mbarrier protocol is
//     fully exercised in cutlass-free C++ + inline PTX.
//   * `nvcc -E` and `cuobjdump --dump-elf-symbols` both report 0 cutlass:: / cute:: matches.
//
// Build:
//   nvcc -std=c++17 -O3 -gencode=arch=compute_90a,code=sm_90a -lineinfo \
//        tma_hello.cu -lcuda -o tma_hello

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda.h>

#define CUDA_CHECK(x) do { cudaError_t err = (x); if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
    std::exit(1); } } while (0)

#define CU_CHECK(x) do { CUresult err = (x); if (err != CUDA_SUCCESS) { \
    const char* msg = nullptr; cuGetErrorString(err, &msg); \
    fprintf(stderr, "CU error %s at %s:%d\n", msg ? msg : "(?)", __FILE__, __LINE__); \
    std::exit(1); } } while (0)

// Tile dimensions: load a 16 × 32 tile of float32 (= 64 bytes per row, 16 rows = 1024 B = 256 floats).
// Source tensor: 64 × 32 floats. The TMA map describes the source layout; the kernel loads tile (0,0).
constexpr int SRC_ROWS = 64;
constexpr int SRC_COLS = 32;
constexpr int TILE_ROWS = 16;
constexpr int TILE_COLS = 32;

// Inline-PTX wrappers for TMA + mbarrier instructions (sm_90a).
__device__ __forceinline__ void mbarrier_init(uint64_t* bar, uint32_t n_threads) {
    uint32_t bar_int = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" :: "r"(bar_int), "r"(n_threads));
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* bar, uint32_t expected_bytes) {
    uint32_t bar_int = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
        :: "r"(bar_int), "r"(expected_bytes));
}

__device__ __forceinline__ void mbarrier_wait(uint64_t* bar, uint32_t phase) {
    uint32_t bar_int = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile(
        "{ .reg .pred P1;\n"
        "  LAB_WAIT: \n"
        "  mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
        "  @P1 bra DONE_WAIT; \n"
        "  bra LAB_WAIT; \n"
        "  DONE_WAIT: }\n"
        :: "r"(bar_int), "r"(phase));
}

// Issue cp.async.bulk.tensor.2d.shared::cluster.global  (TMA tile load) via inline PTX.
__device__ __forceinline__ void tma_load_2d(
    void* smem_dst,
    const CUtensorMap* tensor_map,
    int32_t coord_row, int32_t coord_col,
    uint64_t* mbar)
{
    uint32_t smem_int = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
    uint32_t bar_int  = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    uint64_t map_addr = reinterpret_cast<uint64_t>(tensor_map);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%3, %4}], [%2];\n"
        :: "r"(smem_int), "l"(map_addr), "r"(bar_int),
           "r"(coord_col), "r"(coord_row));
}

__global__ void tma_load_kernel(const CUtensorMap* tensor_map, float* d_out) {
    extern __shared__ uint8_t smem_raw[];
    // Layout in smem: [tile data, 1024 bytes (256 floats)] [mbarrier, 8 bytes]
    float* smem_tile = reinterpret_cast<float*>(smem_raw);
    uint64_t* smem_bar = reinterpret_cast<uint64_t*>(smem_raw + TILE_ROWS * TILE_COLS * sizeof(float));

    int tid = threadIdx.x;

    if (tid == 0) {
        mbarrier_init(smem_bar, 1);  // one arrival expected (the TMA itself)
        constexpr uint32_t expected_bytes = TILE_ROWS * TILE_COLS * sizeof(float);  // 2048 B
        mbarrier_arrive_expect_tx(smem_bar, expected_bytes);
        tma_load_2d(smem_tile, tensor_map, /*row=*/0, /*col=*/0, smem_bar);
    }
    __syncthreads();
    if (tid == 0) {
        mbarrier_wait(smem_bar, /*phase=*/0);
    }
    __syncthreads();

    // Cooperatively copy the tile from smem to d_out for host verification.
    for (int i = tid; i < TILE_ROWS * TILE_COLS; i += blockDim.x) {
        d_out[i] = smem_tile[i];
    }
}

int main() {
    // 1. Build the source tensor on host with a deterministic pattern.
    size_t src_bytes = SRC_ROWS * SRC_COLS * sizeof(float);
    float* h_src = new float[SRC_ROWS * SRC_COLS];
    for (int r = 0; r < SRC_ROWS; ++r)
        for (int c = 0; c < SRC_COLS; ++c)
            h_src[r * SRC_COLS + c] = float(r * 100 + c);

    // 2. Allocate device memory and copy source.
    float* d_src;
    CUDA_CHECK(cudaMalloc(&d_src, src_bytes));
    CUDA_CHECK(cudaMemcpy(d_src, h_src, src_bytes, cudaMemcpyHostToDevice));

    // 3. Build a CUtensorMap describing the source via the CUDA Driver API.
    //    No cutlass include needed — pure driver API.
    CUtensorMap tensor_map = {};
    cuuint64_t global_dim[2]  = {SRC_COLS, SRC_ROWS};                  // {fastest-moving (col), slow (row)}
    cuuint64_t global_stride[1] = {SRC_COLS * sizeof(float)};          // bytes per row
    cuuint32_t box_dim[2]     = {TILE_COLS, TILE_ROWS};                // tile shape
    cuuint32_t elem_stride[2] = {1, 1};                                // element stride within the tile

    CUresult cu_init = cuInit(0);
    if (cu_init != CUDA_SUCCESS) {
        fprintf(stderr, "cuInit failed\n");
        return 1;
    }
    CU_CHECK(cuTensorMapEncodeTiled(
        &tensor_map,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
        2,
        d_src,
        global_dim,
        global_stride,           // strides for dims [1..rank-1] only (rank-1 entries)
        box_dim,
        elem_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    // 4. Allocate a device-side copy of the tensor map (kernel takes a __grid_constant__ pointer
    //    in cutlass; we simulate by allocating a device buffer and copying).
    CUtensorMap* d_tensor_map;
    CUDA_CHECK(cudaMalloc(&d_tensor_map, sizeof(CUtensorMap)));
    CUDA_CHECK(cudaMemcpy(d_tensor_map, &tensor_map, sizeof(CUtensorMap), cudaMemcpyHostToDevice));

    // 5. Allocate device output.
    size_t tile_bytes = TILE_ROWS * TILE_COLS * sizeof(float);
    float* d_out;
    CUDA_CHECK(cudaMalloc(&d_out, tile_bytes));
    CUDA_CHECK(cudaMemset(d_out, 0xFF, tile_bytes));

    // 6. Launch.
    size_t smem_bytes = tile_bytes + sizeof(uint64_t);
    tma_load_kernel<<<1, 32, smem_bytes>>>(d_tensor_map, d_out);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 7. Verify the loaded tile matches host source[0:TILE_ROWS, 0:TILE_COLS].
    float* h_out = new float[TILE_ROWS * TILE_COLS];
    CUDA_CHECK(cudaMemcpy(h_out, d_out, tile_bytes, cudaMemcpyDeviceToHost));

    int matched = 0;
    for (int r = 0; r < TILE_ROWS; ++r) {
        for (int c = 0; c < TILE_COLS; ++c) {
            float expected = h_src[r * SRC_COLS + c];
            float got = h_out[r * TILE_COLS + c];
            if (got == expected) ++matched;
        }
    }
    int total = TILE_ROWS * TILE_COLS;

    printf("=== tma_hello (cutlass-free TMA 2D tile load) ===\n");
    printf("  source:     %d x %d float (%zu B)\n", SRC_ROWS, SRC_COLS, src_bytes);
    printf("  tile:       %d x %d float (%zu B)\n", TILE_ROWS, TILE_COLS, tile_bytes);
    printf("  matched:    %d / %d\n", matched, total);
    printf("  h_out[0]=%.0f  h_out[1]=%.0f  h_out[%d]=%.0f  h_out[%d]=%.0f\n",
           h_out[0], h_out[1],
           TILE_COLS, h_out[TILE_COLS],
           TILE_ROWS * TILE_COLS - 1, h_out[TILE_ROWS * TILE_COLS - 1]);
    printf("  expected:   h_src[0]=%.0f  h_src[1]=%.0f  h_src[%d]=%.0f  h_src[%d]=%.0f\n",
           h_src[0], h_src[1],
           SRC_COLS, h_src[SRC_COLS],
           (TILE_ROWS - 1) * SRC_COLS + (TILE_COLS - 1),
           h_src[(TILE_ROWS - 1) * SRC_COLS + (TILE_COLS - 1)]);

    delete[] h_src; delete[] h_out;
    cudaFree(d_src); cudaFree(d_tensor_map); cudaFree(d_out);

    return matched == total ? 0 : 1;
}
