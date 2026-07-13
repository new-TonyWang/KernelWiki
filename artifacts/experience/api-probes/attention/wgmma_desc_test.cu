// wgmma m64n16k16 with canonical CUTLASS smem layout (Layout_K_SW128_Atom<half>).
// Key: atom is 8×64 (8 M-rows × 64 K-elements), row_stride=128 bytes, Swizzle<3,4,3>.
// For K=16, only first 16 elements per row are used; rest is zero padding.
// Build: nvcc -std=c++17 -O3 -gencode=arch=compute_90a,code=sm_90a wgmma_desc_test.cu -o wgmma_desc_test

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#define CUDA_CHECK(x) do{cudaError_t e=(x);if(e){fprintf(stderr,"CUDA %s at %s:%d\n",cudaGetErrorString(e),__FILE__,__LINE__);exit(1);}}while(0)

// Swizzle<3,4,3>: XOR bits[4:7] with bits[7:10] of byte address
__host__ __device__ int swz128(int byte_addr) {
    return byte_addr ^ (((byte_addr >> 7) & 7) << 4);
}

__device__ __forceinline__ uint64_t mk_desc_s128(const void* p, uint32_t ld, uint32_t sd) {
    uint32_t a=(uint32_t)__cvta_generic_to_shared(p);
    uint64_t d=0;
    d|=((uint64_t)(a>>4))&0x3FFF;
    d|=((uint64_t)((ld>>4)&0x3FFF))<<16;
    d|=((uint64_t)((sd>>4)&0x3FFF))<<32;
    d|=((uint64_t)1)<<62;
    return d;
}

// NN test: A(64×16) × B(16×16) = C(64×16)
// A: K-major smem with 128-byte rows (64 f16 per row, K=16 used + 48 padding)
// B: K-major col-major smem: B_cm[n][k] at byte n*128+k*2 (16 cols × 16 rows)
__global__ void test_canonical(const half* hA, const half* hB, float* Cg, int N_B) {
    // A smem: 64 rows × 64 cols × 2 bytes = 8192 bytes
    // B smem: N_B rows × 64 cols × 2 bytes
    extern __shared__ uint8_t smem[];
    int t=threadIdx.x;

    // Zero-fill A and B smem
    for(int i=t*4;i<8192+N_B*128;i+=128*4) {
        if(i+3<8192+N_B*128){*(int*)(smem+i)=0;}
        else for(int j=0;j<4&&i+j<8192+N_B*128;j++) smem[i+j]=0;
    }
    __syncthreads();

    // Load A[m][k] into canonical K-major SW128 layout
    for(int i=t;i<64*16;i+=128){
        int m=i/16,k=i%16;
        int byte_off=m*128+k*2; // 128-byte row stride
        int swz=swz128(byte_off);
        *(half*)(smem+swz)=hA[i];
    }
    // Load B col-major: B_cm[n][k] = hB[k*16+n] into canonical K-major layout
    // B is 16×16, stored as 16 "columns" of 16 elements each
    // In K-major smem: row=n (0..15), col=k (0..15), byte=n*128+k*2
    half* B_smem=(half*)(smem+8192);
    for(int i=t;i<16*16;i+=128){
        int k=i/16,n=i%16;
        int byte_off=n*128+k*2;
        int swz=swz128(byte_off);
        *(half*)((uint8_t*)B_smem+swz)=hB[k*16+n];
    }
    __syncthreads();

    float d[8]={};
    // SWIZZLE_128B: LBO=16, SBO=8*128=1024
    uint64_t dA=mk_desc_s128((half*)smem,16,1024);
    uint64_t dB=mk_desc_s128(B_smem,16,1024);

    asm volatile("wgmma.fence.sync.aligned;\n");
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %10, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, %8, %9, p, 1, 1, 0, 0;\n}\n"
        :"+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7])
        :"l"(dA),"l"(dB),"r"(1));
    asm volatile("wgmma.commit_group.sync.aligned;\n");
    asm volatile("wgmma.wait_group.sync.aligned 0;\n");

    int w=t/32,l=t%32,r0=w*16+l/4,c0=(l%4)*2;
    Cg[r0*16+c0]=d[0];Cg[r0*16+c0+1]=d[1];
    Cg[(r0+8)*16+c0]=d[2];Cg[(r0+8)*16+c0+1]=d[3];
    Cg[r0*16+c0+8]=d[4];Cg[r0*16+c0+9]=d[5];
    Cg[(r0+8)*16+c0+8]=d[6];Cg[(r0+8)*16+c0+9]=d[7];
}

int main(){
    int result=0;
    half hA[1024],hB[256];float hCc[1024],hCg[1024];

    // Test 1: all-ones
    for(int i=0;i<1024;i++)hA[i]=__float2half(1.f);
    for(int i=0;i<256;i++)hB[i]=__float2half(1.f);
    half*dA,*dB;float*dC;
    CUDA_CHECK(cudaMalloc(&dA,2048));CUDA_CHECK(cudaMalloc(&dB,512));CUDA_CHECK(cudaMalloc(&dC,4096));
    CUDA_CHECK(cudaMemcpy(dA,hA,2048,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB,hB,512,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dC,0,4096));
    size_t smem_bytes=8192+16*128; // A(64×64×2) + B(16×64×2)
    test_canonical<<<1,128,smem_bytes>>>(dA,dB,dC,16);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hCg,dC,4096,cudaMemcpyDeviceToHost));
    int m1=0;for(int i=0;i<1024;i++)if(fabsf(hCg[i]-16.f)>.01f)m1++;
    printf("NN canonical SW128 all-ones: mis=%d/1024 %s\n",m1,m1==0?"PASS":"FAIL");
    if(m1)result=1;

    // Test 2: random
    srand(42);
    for(int i=0;i<1024;i++)hA[i]=__float2half((float)(rand()%20-10)/10.f);
    for(int i=0;i<256;i++)hB[i]=__float2half((float)(rand()%20-10)/10.f);
    for(int m=0;m<64;m++)for(int n=0;n<16;n++){
        float a=0;for(int k=0;k<16;k++)a+=__half2float(hA[m*16+k])*__half2float(hB[k*16+n]);
        hCc[m*16+n]=a;}
    CUDA_CHECK(cudaMemcpy(dA,hA,2048,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB,hB,512,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dC,0,4096));
    test_canonical<<<1,128,smem_bytes>>>(dA,dB,dC,16);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hCg,dC,4096,cudaMemcpyDeviceToHost));
    float me=0;int mi=0;
    for(int i=0;i<1024;i++){float e=fabsf(hCg[i]-hCc[i]);if(e>me)me=e;if(e>.01f)mi++;}
    printf("NN canonical SW128 random:   max_err=%.6f mis=%d/1024 %s\n",me,mi,mi==0?"PASS":"FAIL");
    if(mi)result=1;

    cudaFree(dA);cudaFree(dB);cudaFree(dC);
    return result;
}
