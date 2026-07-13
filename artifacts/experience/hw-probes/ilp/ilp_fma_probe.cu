// ILP microbenchmark: 1-accumulator vs 4-accumulator FMA throughput.
//
// Each kernel runs TOTAL_FMAS fused multiply-add instructions per thread.
// - 1-acc kernel: a single dependent FMA chain of length TOTAL_FMAS.
// - 4-acc kernel: 4 independent chains of length TOTAL_FMAS/4 each,
//   then reduced to a single value to defeat dead-code elimination.
//
// We measure cycles via clock64() on a single warp (block=32, grid=1).
// Throughput = TOTAL_FMAS / (elapsed_cycles / 32 threads) per SM.
// The interesting metric is the ratio: 4-acc cycles / 1-acc cycles.
// Expect 4-acc to be ~3-4x faster due to ILP filling the FMA pipeline.

#include <cstdint>
#include <cstdio>
#include <algorithm>
#include <vector>

static constexpr int TOTAL_FMAS = 1024;   // per thread, per trial
static constexpr int CHAIN_LEN_4 = TOTAL_FMAS / 4;  // 256 per chain
static constexpr int NUM_TRIALS  = 4096;

// ---------- 1-accumulator kernel ----------
__global__ void probe_1acc(uint64_t* __restrict__ out_cycles,
                           float*    __restrict__ out_val) {
    float acc = 1.0001f;
    float a   = 1.000001f;
    float b   = 0.000001f;

    for (int trial = 0; trial < NUM_TRIALS; ++trial) {
        uint64_t start = clock64();
        #pragma unroll
        for (int i = 0; i < TOTAL_FMAS; ++i) {
            acc = __fmaf_rn(acc, a, b);   // dependent chain
        }
        uint64_t end = clock64();
        if (threadIdx.x == 0) {
            out_cycles[trial] = end - start;
        }
    }
    // defeat DCE
    if (threadIdx.x == 0) out_val[0] = acc;
}

// ---------- 4-accumulator kernel ----------
__global__ void probe_4acc(uint64_t* __restrict__ out_cycles,
                           float*    __restrict__ out_val) {
    float acc0 = 1.0001f;
    float acc1 = 1.0002f;
    float acc2 = 1.0003f;
    float acc3 = 1.0004f;
    float a    = 1.000001f;
    float b    = 0.000001f;

    for (int trial = 0; trial < NUM_TRIALS; ++trial) {
        uint64_t start = clock64();
        #pragma unroll
        for (int i = 0; i < CHAIN_LEN_4; ++i) {
            acc0 = __fmaf_rn(acc0, a, b);  // chain 0
            acc1 = __fmaf_rn(acc1, a, b);  // chain 1 (independent)
            acc2 = __fmaf_rn(acc2, a, b);  // chain 2 (independent)
            acc3 = __fmaf_rn(acc3, a, b);  // chain 3 (independent)
        }
        uint64_t end = clock64();
        if (threadIdx.x == 0) {
            out_cycles[trial] = end - start;
        }
    }
    // reduce and defeat DCE
    float sum = acc0 + acc1 + acc2 + acc3;
    if (threadIdx.x == 0) out_val[0] = sum;
}

// ---------- 2-accumulator kernel (bonus data point) ----------
__global__ void probe_2acc(uint64_t* __restrict__ out_cycles,
                           float*    __restrict__ out_val) {
    float acc0 = 1.0001f;
    float acc1 = 1.0002f;
    float a    = 1.000001f;
    float b    = 0.000001f;
    static constexpr int CHAIN_LEN_2 = TOTAL_FMAS / 2;

    for (int trial = 0; trial < NUM_TRIALS; ++trial) {
        uint64_t start = clock64();
        #pragma unroll
        for (int i = 0; i < CHAIN_LEN_2; ++i) {
            acc0 = __fmaf_rn(acc0, a, b);
            acc1 = __fmaf_rn(acc1, a, b);
        }
        uint64_t end = clock64();
        if (threadIdx.x == 0) {
            out_cycles[trial] = end - start;
        }
    }
    float sum = acc0 + acc1;
    if (threadIdx.x == 0) out_val[0] = sum;
}

// ---------- 8-accumulator kernel (bonus data point) ----------
__global__ void probe_8acc(uint64_t* __restrict__ out_cycles,
                           float*    __restrict__ out_val) {
    float acc0 = 1.0001f, acc1 = 1.0002f, acc2 = 1.0003f, acc3 = 1.0004f;
    float acc4 = 1.0005f, acc5 = 1.0006f, acc6 = 1.0007f, acc7 = 1.0008f;
    float a    = 1.000001f;
    float b    = 0.000001f;
    static constexpr int CHAIN_LEN_8 = TOTAL_FMAS / 8;

    for (int trial = 0; trial < NUM_TRIALS; ++trial) {
        uint64_t start = clock64();
        #pragma unroll
        for (int i = 0; i < CHAIN_LEN_8; ++i) {
            acc0 = __fmaf_rn(acc0, a, b);
            acc1 = __fmaf_rn(acc1, a, b);
            acc2 = __fmaf_rn(acc2, a, b);
            acc3 = __fmaf_rn(acc3, a, b);
            acc4 = __fmaf_rn(acc4, a, b);
            acc5 = __fmaf_rn(acc5, a, b);
            acc6 = __fmaf_rn(acc6, a, b);
            acc7 = __fmaf_rn(acc7, a, b);
        }
        uint64_t end = clock64();
        if (threadIdx.x == 0) {
            out_cycles[trial] = end - start;
        }
    }
    float sum = acc0 + acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7;
    if (threadIdx.x == 0) out_val[0] = sum;
}

// ---------- host driver ----------
struct Stats {
    float median;
    float p10;
    float p90;
};

Stats compute_stats(std::vector<uint64_t>& v) {
    std::sort(v.begin(), v.end());
    int n = (int)v.size();
    Stats s;
    s.median = (float)v[n / 2];
    s.p10    = (float)v[n / 10];
    s.p90    = (float)v[n * 9 / 10];
    return s;
}

int main() {
    // device info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s  SM: %d.%d\n", prop.name, prop.major, prop.minor);

    uint64_t *d_cycles;
    float    *d_val;
    cudaMalloc(&d_cycles, NUM_TRIALS * sizeof(uint64_t));
    cudaMalloc(&d_val,    sizeof(float));

    std::vector<uint64_t> h_cycles(NUM_TRIALS);

    auto run_probe = [&](const char* label, auto kernel) {
        // warmup
        for (int i = 0; i < 5; ++i) {
            kernel<<<1, 32>>>(d_cycles, d_val);
        }
        cudaDeviceSynchronize();

        // measure
        kernel<<<1, 32>>>(d_cycles, d_val);
        cudaDeviceSynchronize();
        cudaMemcpy(h_cycles.data(), d_cycles,
                   NUM_TRIALS * sizeof(uint64_t), cudaMemcpyDeviceToHost);

        Stats st = compute_stats(h_cycles);
        float per_fma = st.median / TOTAL_FMAS;
        float per_fma_p10 = st.p10 / TOTAL_FMAS;
        float per_fma_p90 = st.p90 / TOTAL_FMAS;
        printf("%-12s  total_cycles median=%.0f  p10=%.0f  p90=%.0f  "
               "per_fma median=%.2f  p10=%.2f  p90=%.2f  cycles/FMA\n",
               label, st.median, st.p10, st.p90,
               per_fma, per_fma_p10, per_fma_p90);
    };

    printf("\nFMA ILP probe: %d FMAs per thread, %d trials\n", TOTAL_FMAS, NUM_TRIALS);
    printf("--------------------------------------------------------------\n");
    run_probe("1-acc", probe_1acc);
    run_probe("2-acc", probe_2acc);
    run_probe("4-acc", probe_4acc);
    run_probe("8-acc", probe_8acc);

    cudaFree(d_cycles);
    cudaFree(d_val);
    return 0;
}
