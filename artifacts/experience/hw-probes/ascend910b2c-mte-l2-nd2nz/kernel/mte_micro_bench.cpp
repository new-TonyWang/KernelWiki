#include "kernel_operator.h"
#include "lib/matmul/matmul.h"
#include "mte_micro_bench_tiling.h"
using namespace AscendC;

namespace AscendC {
__aicore__ inline void clearWorkspace(GM_ADDR workspace) { (void)workspace; }
}

namespace {
constexpr uint32_t TILE_ELEMS = 16 * 1024;      // 32 KiB fp16 tile
constexpr uint32_t TILE_BYTES = TILE_ELEMS * sizeof(half);
constexpr uint8_t LOAD_REPEAT = 64;             // 64 * 512 B = 32 KiB for fp16 load2d

class MteMicroBenchKernel {
public:
    __aicore__ inline MteMicroBenchKernel() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const MteMicroBenchTilingData &t)
    {
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(x));
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(y));
        size = t.size;
        mode = t.mode;
        loops = t.loops;
        elemsPerLoop = t.elemsPerLoop;
        if (loops == 0) loops = 1;
        if (elemsPerLoop == 0 || elemsPerLoop > TILE_ELEMS) elemsPerLoop = TILE_ELEMS;
        // For MIX kernels only AIC executes the L1/L0 microbench. AIV returns in Run().
        if ASCEND_IS_AIC {
            if (mode == 0 || mode == 3 || mode == 7 || mode == 8 || mode == 20 || mode == 21) pipe.InitBuffer(qL0a, 1, TILE_BYTES);
            if (mode == 1 || mode == 2 || mode == 3 || mode == 4 || mode == 6 || mode == 7 || mode == 8) pipe.InitBuffer(qL1a, 1, TILE_BYTES);
            if (mode == 5 || mode == 6) pipe.InitBuffer(qL1b, 1, TILE_BYTES);
            if (mode == 4 || mode == 6 || mode == 8) pipe.InitBuffer(qL0a, 1, TILE_BYTES);
            if (mode == 5 || mode == 6) pipe.InitBuffer(qL0b, 1, TILE_BYTES);
        }
    }
    __aicore__ inline uint32_t Offset(uint32_t i)
    {
        uint32_t span = size > elemsPerLoop ? (size - elemsPerLoop) : 1;
        uint32_t core = static_cast<uint32_t>(GetBlockIdx());
        return ((i * elemsPerLoop) + core * 131072U) % span;
    }
    __aicore__ inline void Run()
    {
        if ASCEND_IS_AIV { return; }
        if (mode == 99) return;
        if (mode == 0) GmToL0A(false);          // MTE2: GM/L2 -> L0A, no transpose
        else if (mode == 1) GmToL1Raw();        // MTE2: GM/L2 -> L1 normal copy
        else if (mode == 2) GmToL1Nd2Nz();      // MTE2: GM ND -> L1 NZ
        else if (mode == 3) GmToL0AAndL1();     // same AIC issues GM->L0A and GM->L1, contention probe
        else if (mode == 4) L1ToL0A();          // MTE1: L1 -> L0A
        else if (mode == 5) L1ToL0B();          // MTE1: L1 -> L0B
        else if (mode == 6) L1ToL0ABoth();      // same AIC issues L1->L0A and L1->L0B
        else if (mode == 7) GmToL0A(true);      // LoadData transpose path into L0A
        else if (mode == 8) CombinedTranspose();// preload L1 ND2NZ + repeated L1->L0A transpose-ish feed
        else if (mode == 20) GmToL0ASameAddr();   // blockdim=2, both cubes read same addresses: cross-cube L2 reuse probe
        else if (mode == 21) GmToL0ADisjoint();   // blockdim=2, cubes read separate halves: contention baseline
    }
private:
    __aicore__ inline void FillLoadParams(LoadData2DParams &p, bool trans)
    {
        p.startIndex = 0;
        p.repeatTimes = LOAD_REPEAT;
        p.srcStride = 1;
        p.dstGap = 0;
        p.ifTranspose = trans;
        p.sid = 0;
        p.addrMode = 0;
    }
    __aicore__ inline void GmToL0ASameAddr()
    {
        LoadData2DParams p; FillLoadParams(p, false);
        for (uint32_t i = 0; i < loops; ++i) {
            uint32_t span = size > elemsPerLoop ? (size - elemsPerLoop) : 1;
            uint32_t off = (i * elemsPerLoop) % span;  // intentionally ignore GetBlockIdx(): all cubes touch same lines
            LocalTensor<half> a2 = qL0a.AllocTensor<half>();
            LoadData(a2, xGm[off], p);
            qL0a.EnQue(a2);
            a2 = qL0a.DeQue<half>();
            qL0a.FreeTensor(a2);
        }
    }
    __aicore__ inline void GmToL0ADisjoint()
    {
        LoadData2DParams p; FillLoadParams(p, false);
        uint32_t halfSpan = (size > 2 * elemsPerLoop) ? (size / 2) : elemsPerLoop;
        for (uint32_t i = 0; i < loops; ++i) {
            uint32_t base = (static_cast<uint32_t>(GetBlockIdx()) & 1U) * halfSpan;
            uint32_t off = base + ((i * elemsPerLoop) % (halfSpan > elemsPerLoop ? (halfSpan - elemsPerLoop) : 1));
            LocalTensor<half> a2 = qL0a.AllocTensor<half>();
            LoadData(a2, xGm[off], p);
            qL0a.EnQue(a2);
            a2 = qL0a.DeQue<half>();
            qL0a.FreeTensor(a2);
        }
    }
    __aicore__ inline void GmToL0A(bool trans)
    {
        LoadData2DParams p; FillLoadParams(p, trans);
        for (uint32_t i = 0; i < loops; ++i) {
            LocalTensor<half> a2 = qL0a.AllocTensor<half>();
            LoadData(a2, xGm[Offset(i)], p);
            qL0a.EnQue(a2);
            a2 = qL0a.DeQue<half>();
            qL0a.FreeTensor(a2);
        }
    }
    __aicore__ inline void GmToL1Raw()
    {
        DataCopyParams p;
        p.blockCount = 1;
        p.blockLen = static_cast<uint16_t>(elemsPerLoop * sizeof(half) / 32);
        p.srcStride = 0; p.dstStride = 0;
        for (uint32_t i = 0; i < loops; ++i) {
            LocalTensor<half> l1 = qL1a.AllocTensor<half>();
            DataCopy(l1, xGm[Offset(i)], p);
            qL1a.EnQue(l1);
            l1 = qL1a.DeQue<half>();
            qL1a.FreeTensor(l1);
        }
    }
    __aicore__ inline void GmToL1Nd2Nz()
    {
        Nd2NzParams p;
        p.ndNum = 1; p.nValue = 128; p.dValue = 128;
        p.srcNdMatrixStride = 0; p.srcDValue = 128;
        p.dstNzC0Stride = 128; p.dstNzNStride = 1; p.dstNzMatrixStride = 0;
        for (uint32_t i = 0; i < loops; ++i) {
            LocalTensor<half> l1 = qL1a.AllocTensor<half>();
            DataCopy(l1, xGm[Offset(i)], p);
            qL1a.EnQue(l1);
            l1 = qL1a.DeQue<half>();
            qL1a.FreeTensor(l1);
        }
    }
    __aicore__ inline void GmToL0AAndL1()
    {
        LoadData2DParams lp; FillLoadParams(lp, false);
        DataCopyParams cp; cp.blockCount = 1; cp.blockLen = TILE_BYTES / 32; cp.srcStride = 0; cp.dstStride = 0;
        for (uint32_t i = 0; i < loops; ++i) {
            uint32_t off = Offset(i);
            LocalTensor<half> a2 = qL0a.AllocTensor<half>();
            LocalTensor<half> l1 = qL1a.AllocTensor<half>();
            LoadData(a2, xGm[off], lp);
            DataCopy(l1, xGm[(off + TILE_ELEMS) % (size - TILE_ELEMS)], cp);
            qL0a.EnQue(a2); qL1a.EnQue(l1);
            a2 = qL0a.DeQue<half>(); l1 = qL1a.DeQue<half>();
            qL0a.FreeTensor(a2); qL1a.FreeTensor(l1);
        }
    }
    __aicore__ inline void PreloadL1A(LocalTensor<half> &l1)
    {
        DataCopyParams p; p.blockCount = 1; p.blockLen = TILE_BYTES / 32; p.srcStride = 0; p.dstStride = 0;
        l1 = qL1a.AllocTensor<half>();
        DataCopy(l1, xGm[0], p);
        qL1a.EnQue(l1); l1 = qL1a.DeQue<half>();
    }
    __aicore__ inline void PreloadL1B(LocalTensor<half> &l1)
    {
        DataCopyParams p; p.blockCount = 1; p.blockLen = TILE_BYTES / 32; p.srcStride = 0; p.dstStride = 0;
        l1 = qL1b.AllocTensor<half>();
        DataCopy(l1, xGm[TILE_ELEMS], p);
        qL1b.EnQue(l1); l1 = qL1b.DeQue<half>();
    }
    __aicore__ inline void L1ToL0A()
    {
        LocalTensor<half> l1; PreloadL1A(l1); LoadData2DParams p; FillLoadParams(p, false);
        for (uint32_t i = 0; i < loops; ++i) {
            LocalTensor<half> a2 = qL0a.AllocTensor<half>();
            LoadData(a2, l1, p);
            qL0a.EnQue(a2); a2 = qL0a.DeQue<half>(); qL0a.FreeTensor(a2);
        }
        qL1a.FreeTensor(l1);
    }
    __aicore__ inline void L1ToL0B()
    {
        LocalTensor<half> l1; PreloadL1B(l1); LoadData2DParams p; FillLoadParams(p, false);
        for (uint32_t i = 0; i < loops; ++i) {
            LocalTensor<half> b2 = qL0b.AllocTensor<half>();
            LoadData(b2, l1, p);
            qL0b.EnQue(b2); b2 = qL0b.DeQue<half>(); qL0b.FreeTensor(b2);
        }
        qL1b.FreeTensor(l1);
    }
    __aicore__ inline void L1ToL0ABoth()
    {
        LocalTensor<half> l1a; LocalTensor<half> l1b; PreloadL1A(l1a); PreloadL1B(l1b);
        LoadData2DParams p; FillLoadParams(p, false);
        for (uint32_t i = 0; i < loops; ++i) {
            LocalTensor<half> a2 = qL0a.AllocTensor<half>();
            LocalTensor<half> b2 = qL0b.AllocTensor<half>();
            LoadData(a2, l1a, p); LoadData(b2, l1b, p);
            qL0a.EnQue(a2); qL0b.EnQue(b2);
            a2 = qL0a.DeQue<half>(); b2 = qL0b.DeQue<half>();
            qL0a.FreeTensor(a2); qL0b.FreeTensor(b2);
        }
        qL1a.FreeTensor(l1a); qL1b.FreeTensor(l1b);
    }
    __aicore__ inline void CombinedTranspose()
    {
        // Prototype combined path: one stream preloads/transforms GM ND->L1 NZ, the other feeds L0A from L1.
        GmToL1Nd2Nz();
        L1ToL0A();
    }
private:
    TPipe pipe;
    TQue<TPosition::A1, 1> qL1a;
    TQue<TPosition::B1, 1> qL1b;
    TQue<TPosition::A2, 1> qL0a;
    TQue<TPosition::B2, 1> qL0b;
    GlobalTensor<half> xGm;
    GlobalTensor<half> yGm;
    uint32_t size, mode, loops, elemsPerLoop;
};
}
extern "C" __global__ __aicore__ void mte_micro_bench(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(MteMicroBenchTilingData);
    GET_TILING_DATA(tilingData, tiling);
    MteMicroBenchKernel op;
    op.Init(x, y, tilingData);
    op.Run();
}
