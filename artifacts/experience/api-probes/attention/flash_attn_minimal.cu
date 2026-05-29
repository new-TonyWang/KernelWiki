// Minimal flash-attention kernel on Hopper (sm_90a).
//
// Computes scaled dot-product attention: O = softmax(Q @ K^T / sqrt(d)) @ V
// using the FlashAttention-2 online softmax algorithm.
//
// Templatized on BLOCK_M and BLOCK_N for kernel-parameter tuning sweeps.
// Thread-level matmul (not wgmma) for correctness clarity.
//
// Build:
//   nvcc -std=c++17 -O3 -gencode=arch=compute_90a,code=sm_90a \
//        flash_attn_minimal.cu -o flash_attn_minimal
//
// Run (correctness):   ./flash_attn_minimal
// Run (tuning sweep):  ./flash_attn_minimal --tune

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CUDA_CHECK(x) do { cudaError_t err = (x); if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
    std::exit(1); } } while (0)

static constexpr int HEAD_DIM = 64;
static constexpr int NTHREADS = 128;

// Compute total dynamic smem needed for a config
template<int BM, int BN>
constexpr size_t smem_total() {
    // Q[BM*HD] + K[BN*HD] + V[BN*HD] (half)
    size_t tiles = (BM + 2*BN) * HEAD_DIM * sizeof(half);
    // S[BM*BN] + P[BM*BN] (float)
    size_t scores = 2 * BM * BN * sizeof(float);
    // row_max[BM] + row_sum[BM] + row_rescale[BM] (float)
    size_t rows = 3 * BM * sizeof(float);
    return tiles + scores + rows;
}

// Flash-attention kernel templatized on tile sizes.
// All shared memory is dynamic to avoid ptxas static-smem limits.
template<int BM, int BN>
__global__ void flash_attn_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    int seq_len, int head_dim, float scale)
{
    int tid = threadIdx.x;
    int q_start = blockIdx.x * BM;
    int bh_idx = blockIdx.y;

    const half* Q_bh = Q + bh_idx * seq_len * head_dim;
    const half* K_bh = K + bh_idx * seq_len * head_dim;
    const half* V_bh = V + bh_idx * seq_len * head_dim;
    half* O_bh = O + bh_idx * seq_len * head_dim;

    // All-dynamic smem layout
    extern __shared__ uint8_t smem_raw[];
    half*  Q_s         = reinterpret_cast<half*>(smem_raw);
    half*  K_s         = Q_s + BM * HEAD_DIM;
    half*  V_s         = K_s + BN * HEAD_DIM;
    float* S_s         = reinterpret_cast<float*>(V_s + BN * HEAD_DIM);
    float* P_s         = S_s + BM * BN;
    float* row_max     = P_s + BM * BN;
    float* row_sum     = row_max + BM;
    float* row_rescale = row_sum + BM;

    // Load Q tile
    for (int i = tid; i < BM * HEAD_DIM; i += NTHREADS) {
        int r = q_start + i / HEAD_DIM;
        Q_s[i] = (r < seq_len) ? Q_bh[r * head_dim + (i % HEAD_DIM)] : __float2half(0.f);
    }
    for (int i = tid; i < BM; i += NTHREADS) {
        row_max[i] = -INFINITY;
        row_sum[i] = 0.f;
    }
    __syncthreads();

    constexpr int EPT = BM * HEAD_DIM / NTHREADS;
    float O_acc[EPT];
    for (int i = 0; i < EPT; i++) O_acc[i] = 0.f;

    int num_kv = (seq_len + BN - 1) / BN;
    for (int t = 0; t < num_kv; t++) {
        int kv_start = t * BN;

        // Phase 0: Load K and V
        for (int i = tid; i < BN * HEAD_DIM; i += NTHREADS) {
            int r = kv_start + i / HEAD_DIM;
            K_s[i] = (r < seq_len) ? K_bh[r * head_dim + (i % HEAD_DIM)] : __float2half(0.f);
            V_s[i] = (r < seq_len) ? V_bh[r * head_dim + (i % HEAD_DIM)] : __float2half(0.f);
        }
        __syncthreads();

        // Phase 1: S = Q @ K^T
        for (int idx = tid; idx < BM * BN; idx += NTHREADS) {
            int qi = idx / BN, ki = idx % BN;
            float dot = 0.f;
            for (int d = 0; d < HEAD_DIM; d++)
                dot += __half2float(Q_s[qi * HEAD_DIM + d]) *
                       __half2float(K_s[ki * HEAD_DIM + d]);
            S_s[idx] = dot * scale;
        }
        __syncthreads();

        // Phase 2: Per-row softmax state
        for (int qi = tid; qi < BM; qi += NTHREADS) {
            int kv_end = (kv_start + BN < seq_len) ? BN : (seq_len - kv_start);
            if (kv_end <= 0) kv_end = 0;
            float m_old = row_max[qi], m_new = m_old;
            for (int ki = 0; ki < kv_end; ki++)
                m_new = fmaxf(m_new, S_s[qi * BN + ki]);
            float rs = (m_old == -INFINITY) ? 0.f : expf(m_old - m_new);
            row_rescale[qi] = rs;
            row_sum[qi] *= rs;
            for (int ki = 0; ki < kv_end; ki++) {
                float p = expf(S_s[qi * BN + ki] - m_new);
                P_s[qi * BN + ki] = p;
                row_sum[qi] += p;
            }
            for (int ki = kv_end; ki < BN; ki++) P_s[qi * BN + ki] = 0.f;
            row_max[qi] = m_new;
        }
        __syncthreads();

        // Phase 3: Cooperative rescale + P@V
        for (int k = 0; k < EPT; k++) {
            int flat = tid + NTHREADS * k;
            int qi = flat / HEAD_DIM, d = flat % HEAD_DIM;
            if (q_start + qi >= seq_len) continue;
            O_acc[k] *= row_rescale[qi];
            float acc = 0.f;
            int kv_end = (kv_start + BN < seq_len) ? BN : (seq_len - kv_start);
            if (kv_end <= 0) kv_end = 0;
            for (int ki = 0; ki < kv_end; ki++)
                acc += P_s[qi * BN + ki] * __half2float(V_s[ki * HEAD_DIM + d]);
            O_acc[k] += acc;
        }
        __syncthreads();
    }

    // Writeback
    for (int k = 0; k < EPT; k++) {
        int flat = tid + NTHREADS * k;
        int qi = flat / HEAD_DIM, d = flat % HEAD_DIM;
        if (q_start + qi >= seq_len) continue;
        float l = row_sum[qi];
        if (l == 0.f) l = 1.f;
        O_bh[(q_start + qi) * head_dim + d] = __float2half(O_acc[k] / l);
    }
}

// CPU reference
void cpu_attention(const half* Q, const half* K, const half* V, half* O,
                   int BH, int S, int D) {
    float scale = 1.f / sqrtf((float)D);
    for (int bh = 0; bh < BH; bh++) {
        const half* Qp = Q + bh * S * D;
        const half* Kp = K + bh * S * D;
        const half* Vp = V + bh * S * D;
        half* Op = O + bh * S * D;
        for (int qi = 0; qi < S; qi++) {
            float m = -INFINITY;
            float* sc = new float[S];
            for (int ki = 0; ki < S; ki++) {
                float dot = 0.f;
                for (int d = 0; d < D; d++)
                    dot += __half2float(Qp[qi * D + d]) * __half2float(Kp[ki * D + d]);
                sc[ki] = dot * scale;
                m = fmaxf(m, sc[ki]);
            }
            float sum = 0.f;
            for (int ki = 0; ki < S; ki++) { sc[ki] = expf(sc[ki] - m); sum += sc[ki]; }
            for (int ki = 0; ki < S; ki++) sc[ki] /= sum;
            for (int d = 0; d < D; d++) {
                float v = 0.f;
                for (int ki = 0; ki < S; ki++) v += sc[ki] * __half2float(Vp[ki * D + d]);
                Op[qi * D + d] = __float2half(v);
            }
            delete[] sc;
        }
    }
}

struct RunResult { float max_err; int mismatches; float kernel_ms; bool passed; };

template<int BM, int BN>
RunResult run_config(int B, int H, int S, int D) {
    int BH = B * H;
    size_t elems = (size_t)BH * S * D, bytes = elems * sizeof(half);
    half *hQ = new half[elems], *hK = new half[elems], *hV = new half[elems];
    half *hO_gpu = new half[elems], *hO_cpu = new half[elems];
    srand(42);
    for (size_t i = 0; i < elems; i++) {
        hQ[i] = __float2half((float)(rand() % 100 - 50) / 100.f);
        hK[i] = __float2half((float)(rand() % 100 - 50) / 100.f);
        hV[i] = __float2half((float)(rand() % 100 - 50) / 100.f);
    }
    cpu_attention(hQ, hK, hV, hO_cpu, BH, S, D);

    half *dQ, *dK, *dV, *dO;
    CUDA_CHECK(cudaMalloc(&dQ, bytes)); CUDA_CHECK(cudaMalloc(&dK, bytes));
    CUDA_CHECK(cudaMalloc(&dV, bytes)); CUDA_CHECK(cudaMalloc(&dO, bytes));
    CUDA_CHECK(cudaMemcpy(dQ, hQ, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK, hK, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, hV, bytes, cudaMemcpyHostToDevice));

    float scale = 1.f / sqrtf((float)D);
    int nqt = (S + BM - 1) / BM;
    constexpr size_t smem = smem_total<BM, BN>();
    auto kernel = flash_attn_kernel<BM, BN>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    dim3 grid(nqt, BH);

    CUDA_CHECK(cudaMemset(dO, 0, bytes));
    kernel<<<grid, NTHREADS, smem>>>(dQ, dK, dV, dO, S, D, scale);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
    int niters = 5;
    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < niters; i++) {
        CUDA_CHECK(cudaMemset(dO, 0, bytes));
        kernel<<<grid, NTHREADS, smem>>>(dQ, dK, dV, dO, S, D, scale);
    }
    CUDA_CHECK(cudaEventRecord(t1)); CUDA_CHECK(cudaEventSynchronize(t1));
    float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    float kernel_ms = ms / niters;
    CUDA_CHECK(cudaEventDestroy(t0)); CUDA_CHECK(cudaEventDestroy(t1));

    CUDA_CHECK(cudaMemcpy(hO_gpu, dO, bytes, cudaMemcpyDeviceToHost));
    float max_err = 0.f; int mis = 0;
    for (size_t i = 0; i < elems; i++) {
        float e = fabsf(__half2float(hO_gpu[i]) - __half2float(hO_cpu[i]));
        if (e > max_err) max_err = e;
        if (e > 1e-2f) mis++;
    }
    delete[] hQ; delete[] hK; delete[] hV; delete[] hO_gpu; delete[] hO_cpu;
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
    return {max_err, mis, kernel_ms, max_err <= 1e-2f};
}

int main(int argc, char** argv) {
    bool tune = false;
    for (int i = 1; i < argc; i++)
        if (strcmp(argv[i], "--tune") == 0) tune = true;

    if (!tune) {
        int B=1, H=2, S=128, D=HEAD_DIM;
        printf("Flash-attention MVP correctness: B=%d H=%d S=%d D=%d seed=42 dtype=fp16 tol=1e-2\n", B,H,S,D);
        printf("Config: BLOCK_M=64 BLOCK_N=64 HEAD_DIM=%d NTHREADS=%d\n", HEAD_DIM, NTHREADS);
        auto r = run_config<64,64>(B,H,S,D);
        printf("max_abs_err: %.6f\n", r.max_err);
        printf("mismatches (>1e-2): %d / %d\n", r.mismatches, B*H*S*D);
        printf("Disposition: %s\n", r.passed ? "Passed" : "Failed");
        return r.passed ? 0 : 1;
    }

    // Fixed-workload kernel-parameter tuning sweep
    // Varies BLOCK_M x BLOCK_N at fixed workload S=256, B=1, H=2, D=64
    int B=1, H=2, S=256, D=HEAD_DIM;
    printf("# Kernel-parameter tuning sweep: B=%d H=%d S=%d D=%d\n", B,H,S,D);
    printf("block_m,block_n,head_dim,seq_len,kernel_ms,smem_kb,max_abs_err,disposition,q_tiles,kv_iters\n");

    auto emit = [&](int bm, int bn, size_t smem, RunResult r) {
        printf("%d,%d,%d,%d,%.4f,%.1f,%.6f,%s,%d,%d\n",
               bm, bn, D, S, r.kernel_ms, smem/1024.0f, r.max_err,
               r.passed?"Passed":"Failed", (S+bm-1)/bm, (S+bn-1)/bn);
    };

    emit(32,  32,  smem_total<32,32>(),   run_config<32,32>(B,H,S,D));
    emit(32,  64,  smem_total<32,64>(),   run_config<32,64>(B,H,S,D));
    emit(64,  32,  smem_total<64,32>(),   run_config<64,32>(B,H,S,D));
    emit(64,  64,  smem_total<64,64>(),   run_config<64,64>(B,H,S,D));
    emit(64,  128, smem_total<64,128>(),  run_config<64,128>(B,H,S,D));
    emit(128, 64,  smem_total<128,64>(),  run_config<128,64>(B,H,S,D));

    return 0;
}
