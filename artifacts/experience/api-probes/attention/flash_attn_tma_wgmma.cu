// TMA + wgmma flash-attention kernel for H200 (sm_90a).
//
// Computes O = softmax(Q @ K^T / sqrt(d)) @ V using:
//   - cp.async.bulk.tensor (TMA) for tile loads from HBM to smem
//   - wgmma.mma_async m64n16k16 for Q@K^T and P@V matmuls
//   - Thread-level online softmax (FlashAttention-2 algorithm)
//
// Fixed config: BLOCK_M=64, BLOCK_N=64, HEAD_DIM=64, 128 threads (1 warpgroup).
// Sequences must be multiples of 64 for this MVP.
//
// Build:
//   nvcc -std=c++17 -O3 -gencode=arch=compute_90a,code=sm_90a -lineinfo \
//        flash_attn_tma_wgmma.cu -lcuda -o flash_attn_tma_wgmma
//
// Run:  ./flash_attn_tma_wgmma

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda.h>

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e) { \
    fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    exit(1); } } while(0)
#define CU_CHECK(x) do { CUresult e = (x); if (e) { \
    const char* m=0; cuGetErrorString(e,&m); \
    fprintf(stderr, "CU %s at %s:%d\n", m?m:"?", __FILE__, __LINE__); exit(1); } } while(0)

static constexpr int BM = 64, BN = 64, HD = 64, NTH = 128;

// ---- smem layout (all 16-byte aligned) ----
static constexpr size_t Q_OFF     = 0;                    // 64*64*2 = 8192
static constexpr size_t KV_OFF    = 8192;                 // 64*64*2 = 8192
static constexpr size_t S_OFF     = 16384;                // 64*64*4 = 16384
static constexpr size_t P_OFF     = 32768;                // 64*64*2 = 8192
static constexpr size_t RMAX_OFF  = 40960;                // 64*4 = 256
static constexpr size_t RSUM_OFF  = 41216;                // 64*4 = 256
static constexpr size_t RSCL_OFF  = 41472;                // 64*4 = 256
static constexpr size_t MBAR_OFF  = 41728;                // 8
static constexpr size_t SMEM_SZ   = 41736;                // ~40.8 KB

// ---- device helpers ----

// Swizzle<3,4,3>: XOR bits[4:7] with bits[7:10] of byte address
__device__ __forceinline__ int swz128(int byte_addr) {
    return byte_addr ^ (((byte_addr >> 7) & 7) << 4);
}

// GmmaDescriptor with SWIZZLE_128B (layout_type=1)
// Per CUTLASS mma_sm90_desc.hpp, 2 unused bits between each field.
__device__ __forceinline__ uint64_t mk_desc(const void* p, uint32_t ld, uint32_t sd) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    uint64_t d = 0;
    d |= ((uint64_t)(a >> 4)) & 0x3FFF;           // bits [0,14)
    d |= ((uint64_t)((ld >> 4) & 0x3FFF)) << 16;  // bits [16,30)
    d |= ((uint64_t)((sd >> 4) & 0x3FFF)) << 32;  // bits [32,46)
    d |= ((uint64_t)1) << 62;                      // SWIZZLE_128B
    return d;
}

__device__ __forceinline__ void mbar_init(uint64_t* b, uint32_t n) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(b);
    asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" :: "r"(a), "r"(n));
}
__device__ __forceinline__ void mbar_tx(uint64_t* b, uint32_t bytes) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(b);
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(a), "r"(bytes));
}
__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t ph) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(b);
    asm volatile(
        "{ .reg .pred P1;\n"
        " RETRY%=: mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
        " @P1 bra DONE%=;\n bra RETRY%=;\n DONE%=: }\n" :: "r"(a), "r"(ph));
}
__device__ __forceinline__ void tma2d(void* dst, const CUtensorMap* m,
    int32_t c0, int32_t c1, uint64_t* bar) {
    uint32_t s = (uint32_t)__cvta_generic_to_shared(dst);
    uint32_t b = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%3, %4}], [%2];\n"
        :: "r"(s), "l"((uint64_t)m), "r"(b), "r"(c0), "r"(c1));
}

// wgmma m64n16k16 f32.f16.f16 — matches CUTLASS MMA_64x16x16_F32F16F16_SS operand form.
// Per CUTLASS GmmaDescriptor comments in mma_sm90_desc.hpp:
//   trans=0 (N-mode/K-major): LBO unused for SWIZZLE_NONE, SBO = 8 * row_stride
//   trans=1 (T-mode/MN-major): LBO = 8 * row_stride, SBO = 8 * element_size
// Q@K^T: both K-major → trans=0,0.  P@V: A K-major, B N-major → trans=0,1.
__device__ __forceinline__ void wgmma_nn(float* d, uint64_t dA, uint64_t dB) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %10, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, %8, %9, p, 1, 1, 0, 0;\n"
        "}\n"
        : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),
          "+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7])
        : "l"(dA), "l"(dB), "r"(1));
}
__device__ __forceinline__ void wgmma_nt(float* d, uint64_t dA, uint64_t dB) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %10, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, %8, %9, p, 1, 1, 0, 1;\n"
        "}\n"
        : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),
          "+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7])
        : "l"(dA), "l"(dB), "r"(1));
}

// Scatter m64n16 wgmma output (8 f32/thread) to S_smem at column offset n_off
__device__ __forceinline__ void scatter_s(float* S, const float* d, int n_off) {
    int w = threadIdx.x / 32, l = threadIdx.x % 32;
    int r0 = w*16 + l/4, c0 = (l%4)*2 + n_off;
    S[r0*BN + c0]         = d[0]; S[r0*BN + c0+1]       = d[1];
    S[(r0+8)*BN + c0]     = d[2]; S[(r0+8)*BN + c0+1]   = d[3];
    S[r0*BN + c0+8]       = d[4]; S[r0*BN + c0+9]       = d[5];
    S[(r0+8)*BN + c0+8]   = d[6]; S[(r0+8)*BN + c0+9]   = d[7];
}

// Store m64n16 O accumulator to global O at column offset n_off, with 1/l normalization
__device__ __forceinline__ void store_o(half* O_row, const float* d, int n_off,
    const float* rsum) {
    int w = threadIdx.x / 32, l = threadIdx.x % 32;
    int r0 = w*16 + l/4, c0 = (l%4)*2 + n_off;
    float l0 = rsum[r0],  l1 = rsum[r0+8];
    if (l0 == 0.f) l0 = 1.f; if (l1 == 0.f) l1 = 1.f;
    O_row[(r0)*HD + c0]     = __float2half(d[0]/l0);
    O_row[(r0)*HD + c0+1]   = __float2half(d[1]/l0);
    O_row[(r0+8)*HD + c0]   = __float2half(d[2]/l1);
    O_row[(r0+8)*HD + c0+1] = __float2half(d[3]/l1);
    O_row[(r0)*HD + c0+8]   = __float2half(d[4]/l0);
    O_row[(r0)*HD + c0+9]   = __float2half(d[5]/l0);
    O_row[(r0+8)*HD + c0+8] = __float2half(d[6]/l1);
    O_row[(r0+8)*HD + c0+9] = __float2half(d[7]/l1);
}

// ---- main kernel ----

__global__ void flash_attn_tma_wgmma_kernel(
    const CUtensorMap* __restrict__ tma_q,
    const CUtensorMap* __restrict__ tma_k,
    const CUtensorMap* __restrict__ tma_v,
    half* __restrict__ O, int S, float scale)
{
    extern __shared__ uint8_t smem[];
    half*     Q_s   = (half*)(smem + Q_OFF);
    half*     KV_s  = (half*)(smem + KV_OFF);
    float*    S_s   = (float*)(smem + S_OFF);
    half*     P_s   = (half*)(smem + P_OFF);
    float*    rmax  = (float*)(smem + RMAX_OFF);
    float*    rsum  = (float*)(smem + RSUM_OFF);
    float*    rscl  = (float*)(smem + RSCL_OFF);
    uint64_t* mbar  = (uint64_t*)(smem + MBAR_OFF);

    int tid = threadIdx.x;
    int q_start = blockIdx.x * BM;
    int bh = blockIdx.y;
    int row_off = bh * S;  // linear row offset in (BH*S, HD) view

    // wgmma fragment ownership: thread owns rows (r0, r0+8)
    int warp = tid / 32, lane = tid % 32;
    int r0 = warp*16 + lane/4;

    // Init
    if (tid == 0) mbar_init(mbar, 1);
    for (int i = tid; i < BM; i += NTH) { rmax[i] = -INFINITY; rsum[i] = 0.f; }
    __syncthreads();

    // TMA load Q
    uint32_t ph = 0;
    if (tid == 0) {
        mbar_tx(mbar, BM*HD*sizeof(half));
        tma2d(Q_s, tma_q, 0, row_off + q_start, mbar);
    }
    __syncthreads();
    if (tid == 0) mbar_wait(mbar, ph);
    ph ^= 1;
    __syncthreads();

    // O accumulator: 4 n-groups × 8 regs
    float Oa[4][8] = {};

    int nkv = S / BN;
    for (int t = 0; t < nkv; t++) {
        int kv_start = t * BN;

        // TMA load K
        if (tid == 0) {
            mbar_tx(mbar, BN*HD*sizeof(half));
            tma2d(KV_s, tma_k, 0, row_off + kv_start, mbar);
        }
        __syncthreads();
        if (tid == 0) mbar_wait(mbar, ph);
        ph ^= 1;
        __syncthreads();

        // wgmma: S = Q @ K^T  (4 n-groups × 4 k-iters)
        // Per CUTLASS GmmaDescriptor (mma_sm90_desc.hpp):
        //   N-mode (trans=0, K-major): LBO unused for SWIZZLE_NONE, SBO = 8 * row_stride
        // A (Q, trans=0): LBO=16 (unused), SBO = 8*HD*2 = 1024
        // B (K^T, trans=0): LBO=16 (unused), SBO = 8*HD*2 = 1024
        //   K^T has K as fast dim in K_smem (row-major BN×HD, HD contiguous)
        float Sa[4][8] = {};
        asm volatile("wgmma.fence.sync.aligned;\n");
        for (int ng = 0; ng < BN/16; ng++) {
            for (int k = 0; k < HD/16; k++) {
                uint64_t dA = mk_desc(Q_s + k*16, 16, 8*HD*2);
                uint64_t dB = mk_desc(KV_s + ng*16*HD + k*16, 16, 8*HD*2);
                wgmma_nn(Sa[ng], dA, dB);
            }
        }
        asm volatile("wgmma.commit_group.sync.aligned;\n");
        asm volatile("wgmma.wait_group.sync.aligned 0;\n");

        // Scale and scatter S to smem
        for (int ng = 0; ng < BN/16; ng++) {
            for (int i = 0; i < 8; i++) Sa[ng][i] *= scale;
            scatter_s(S_s, Sa[ng], ng*16);
        }
        __syncthreads();

        // Softmax (thread-level, one thread per row for tid < BM)
        if (tid < BM) {
            int qi = tid;
            float m_old = rmax[qi], m_new = m_old;
            for (int ki = 0; ki < BN; ki++)
                m_new = fmaxf(m_new, S_s[qi*BN + ki]);
            float rs = (m_old == -INFINITY) ? 0.f : expf(m_old - m_new);
            rscl[qi] = rs;
            rsum[qi] *= rs;
            for (int ki = 0; ki < BN; ki++) {
                float p = expf(S_s[qi*BN + ki] - m_new);
                // Write P with swizzle for second wgmma (K-major, 128-byte rows)
                int byte_off = qi * 128 + ki * 2;
                *(half*)((uint8_t*)P_s + swz128(byte_off)) = __float2half(p);
                rsum[qi] += p;
            }
            rmax[qi] = m_new;
        }
        __syncthreads();

        // Rescale O_acc per owned rows
        float rs0 = rscl[r0], rs1 = rscl[r0+8];
        for (int ng = 0; ng < HD/16; ng++) {
            Oa[ng][0] *= rs0; Oa[ng][1] *= rs0;
            Oa[ng][2] *= rs1; Oa[ng][3] *= rs1;
            Oa[ng][4] *= rs0; Oa[ng][5] *= rs0;
            Oa[ng][6] *= rs1; Oa[ng][7] *= rs1;
        }

        // TMA load V (reuse KV_s)
        if (tid == 0) {
            mbar_tx(mbar, BN*HD*sizeof(half));
            tma2d(KV_s, tma_v, 0, row_off + kv_start, mbar);
        }
        __syncthreads();
        if (tid == 0) mbar_wait(mbar, ph);
        ph ^= 1;
        __syncthreads();

        // wgmma: O += P @ V  (4 n-groups × 4 k-iters)
        // Both A(P) and B(V) use SWIZZLE_128B with 128-byte rows → LBO=16, SBO=1024
        // A (P, trans=0 K-major): P written with swz128, 128-byte rows
        // B (V, trans=1 MN-major): TMA-loaded with SWIZZLE_128B, 128-byte rows (HD=64 f16)
        asm volatile("wgmma.fence.sync.aligned;\n");
        for (int ng = 0; ng < HD/16; ng++) {
            for (int k = 0; k < BN/16; k++) {
                uint64_t dA = mk_desc(P_s + k*16, 16, 8*BN*2);
                uint64_t dB = mk_desc(KV_s + k*16*HD + ng*16, 16, 8*HD*2);
                wgmma_nt(Oa[ng], dA, dB);
            }
        }
        asm volatile("wgmma.commit_group.sync.aligned;\n");
        asm volatile("wgmma.wait_group.sync.aligned 0;\n");
        __syncthreads();
    }

    // Writeback O with 1/l normalization
    half* O_bh = O + bh * S * HD;
    for (int ng = 0; ng < HD/16; ng++)
        store_o(O_bh + q_start * HD, Oa[ng], ng*16, rsum);
}

// ---- CPU reference ----

void cpu_attention(const half* Q, const half* K, const half* V, half* O,
    int BH, int S, int D) {
    float sc = 1.f / sqrtf((float)D);
    for (int bh = 0; bh < BH; bh++) {
        const half *Qp=Q+bh*S*D, *Kp=K+bh*S*D, *Vp=V+bh*S*D;
        half* Op = O + bh*S*D;
        for (int qi = 0; qi < S; qi++) {
            float m = -INFINITY;
            float* s = new float[S];
            for (int ki = 0; ki < S; ki++) {
                float dot = 0.f;
                for (int d = 0; d < D; d++)
                    dot += __half2float(Qp[qi*D+d]) * __half2float(Kp[ki*D+d]);
                s[ki] = dot * sc; m = fmaxf(m, s[ki]);
            }
            float sum = 0.f;
            for (int ki = 0; ki < S; ki++) { s[ki] = expf(s[ki]-m); sum += s[ki]; }
            for (int ki = 0; ki < S; ki++) s[ki] /= sum;
            for (int d = 0; d < D; d++) {
                float v = 0.f;
                for (int ki = 0; ki < S; ki++) v += s[ki] * __half2float(Vp[ki*D+d]);
                Op[qi*D+d] = __float2half(v);
            }
            delete[] s;
        }
    }
}

// ---- host: create tensor map ----

CUtensorMap make_tma_map(void* d_ptr, int S, int D, int BH, int tile_rows) {
    CUtensorMap map = {};
    cuuint64_t gdim[2] = {(cuuint64_t)D, (cuuint64_t)(BH * S)};
    cuuint64_t gstride[1] = {(cuuint64_t)(D * sizeof(half))};
    cuuint32_t bdim[2] = {(cuuint32_t)D, (cuuint32_t)tile_rows};
    cuuint32_t estride[2] = {1, 1};
    CU_CHECK(cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        2, d_ptr, gdim, gstride, bdim, estride,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    return map;
}

int main() {
    CU_CHECK(cuInit(0));
    int B=1, H=2, S=128, D=HD;
    int BH = B*H;
    size_t elems = (size_t)BH*S*D, bytes = elems*sizeof(half);

    printf("TMA+wgmma flash-attention: B=%d H=%d S=%d D=%d\n", B,H,S,D);
    printf("Config: BM=%d BN=%d HD=%d NTH=%d smem=%.1f KB\n", BM,BN,HD,NTH, SMEM_SZ/1024.f);

    // Host data
    half *hQ=new half[elems], *hK=new half[elems], *hV=new half[elems];
    half *hO_gpu=new half[elems], *hO_cpu=new half[elems];
    srand(42);
    for (size_t i = 0; i < elems; i++) {
        hQ[i] = __float2half((float)(rand()%100-50)/100.f);
        hK[i] = __float2half((float)(rand()%100-50)/100.f);
        hV[i] = __float2half((float)(rand()%100-50)/100.f);
    }
    cpu_attention(hQ, hK, hV, hO_cpu, BH, S, D);

    // Device data
    half *dQ,*dK,*dV,*dO;
    CUDA_CHECK(cudaMalloc(&dQ, bytes)); CUDA_CHECK(cudaMalloc(&dK, bytes));
    CUDA_CHECK(cudaMalloc(&dV, bytes)); CUDA_CHECK(cudaMalloc(&dO, bytes));
    CUDA_CHECK(cudaMemcpy(dQ, hQ, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK, hK, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, hV, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dO, 0, bytes));

    // Create tensor maps
    CUtensorMap map_q = make_tma_map(dQ, S, D, BH, BM);
    CUtensorMap map_k = make_tma_map(dK, S, D, BH, BN);
    CUtensorMap map_v = make_tma_map(dV, S, D, BH, BN);

    // Copy maps to device
    CUtensorMap *d_map_q, *d_map_k, *d_map_v;
    CUDA_CHECK(cudaMalloc(&d_map_q, sizeof(CUtensorMap)));
    CUDA_CHECK(cudaMalloc(&d_map_k, sizeof(CUtensorMap)));
    CUDA_CHECK(cudaMalloc(&d_map_v, sizeof(CUtensorMap)));
    CUDA_CHECK(cudaMemcpy(d_map_q, &map_q, sizeof(CUtensorMap), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_map_k, &map_k, sizeof(CUtensorMap), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_map_v, &map_v, sizeof(CUtensorMap), cudaMemcpyHostToDevice));

    // Launch
    float sc = 1.f / sqrtf((float)D);
    int nqt = S / BM;
    dim3 grid(nqt, BH);
    auto kern = flash_attn_tma_wgmma_kernel;
    if (SMEM_SZ > 48*1024)
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SZ);
    kern<<<grid, NTH, SMEM_SZ>>>(d_map_q, d_map_k, d_map_v, dO, S, sc);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Verify
    CUDA_CHECK(cudaMemcpy(hO_gpu, dO, bytes, cudaMemcpyDeviceToHost));
    float max_err = 0.f; int mis = 0;
    for (size_t i = 0; i < elems; i++) {
        float e = fabsf(__half2float(hO_gpu[i]) - __half2float(hO_cpu[i]));
        if (e > max_err) max_err = e;
        if (e > 1e-2f) mis++;
    }
    printf("max_abs_err: %.6f\n", max_err);
    printf("mismatches (>1e-2): %d / %zu\n", mis, elems);
    printf("Disposition: %s\n", max_err <= 1e-2f ? "Passed" : "Failed");

    // Timing
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < 5; i++) {
        CUDA_CHECK(cudaMemset(dO, 0, bytes));
        kern<<<grid, NTH, SMEM_SZ>>>(d_map_q, d_map_k, d_map_v, dO, S, sc);
    }
    CUDA_CHECK(cudaEventRecord(t1)); CUDA_CHECK(cudaEventSynchronize(t1));
    float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    printf("kernel_ms: %.4f (avg of 5)\n", ms/5.f);

    delete[] hQ; delete[] hK; delete[] hV; delete[] hO_gpu; delete[] hO_cpu;
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
    cudaFree(d_map_q); cudaFree(d_map_k); cudaFree(d_map_v);
    return max_err <= 1e-2f ? 0 : 1;
}
