#pragma once

#include <cstddef>
#include <cstdint>

#include "bandwidth_tiling.h"
#include "cube_bandwidth_modes.h"
#include "kernel_operator.h"

template <typename T>
__aicore__ inline void CopyCubeTiling(T *tiling, GM_ADDR tilingGM)
{
    int32_t *dst = reinterpret_cast<int32_t *>(tiling);
    auto *src = reinterpret_cast<__gm__ int32_t *>(tilingGM);
    for (size_t i = 0; i < sizeof(T) / sizeof(int32_t); ++i) {
        dst[i] = src[i];
    }
}

class CubeBandwidthKernel {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR tilingGM, AscendC::TPipe *pipe)
    {
        (void)y;
        CopyCubeTiling(&tiling_, tilingGM);
        xGM_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(x), tiling_.totalBytes / sizeof(half));

        if ASCEND_IS_AIC {
            const uint32_t subBlockCount = AscendC::GetSubBlockNum();
            coreIdx_ = AscendC::GetBlockIdx() / subBlockCount;
            if (tiling_.mode == CUBE_GM_TO_A1 || tiling_.mode == CUBE_L1_TO_L0A) {
                pipe->InitBuffer(a1Queue_, 1, tiling_.chunkBytes);
            }
            if (tiling_.mode == CUBE_GM_TO_B1 || tiling_.mode == CUBE_L1_TO_L0B) {
                pipe->InitBuffer(b1Queue_, 1, tiling_.chunkBytes);
            }
            if (tiling_.mode == CUBE_L1_TO_L0A || tiling_.mode == CUBE_GM_TO_L0A) {
                pipe->InitBuffer(a2Queue_, 1, tiling_.chunkBytes);
            }
            if (tiling_.mode == CUBE_L1_TO_L0B || tiling_.mode == CUBE_GM_TO_L0B) {
                pipe->InitBuffer(b2Queue_, 1, tiling_.chunkBytes);
            }
        }
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIC {
            if (coreIdx_ >= tiling_.activeLanes || tiling_.chunkBytes == 0) {
                return;
            }
            switch (tiling_.mode) {
                case CUBE_GM_TO_A1:
                    RunGmToA1();
                    break;
                case CUBE_GM_TO_B1:
                    RunGmToB1();
                    break;
                case CUBE_L1_TO_L0A:
                    RunL1ToL0A();
                    break;
                case CUBE_L1_TO_L0B:
                    RunL1ToL0B();
                    break;
                case CUBE_GM_TO_L0A:
                    RunGmToL0A();
                    break;
                case CUBE_GM_TO_L0B:
                    RunGmToL0B();
                    break;
                default:
                    break;
            }
        }
    }

private:
    __aicore__ inline AscendC::DataCopyParams DataCopyParamsForChunk() const
    {
        return AscendC::DataCopyParams{1, static_cast<uint16_t>(tiling_.chunkBytes / 32U), 0, 0};
    }

    __aicore__ inline AscendC::LoadData2DParams LoadDataParamsForChunk() const
    {
        AscendC::LoadData2DParams params;
        params.repeatTimes = static_cast<uint16_t>(tiling_.chunkBytes / 512U);
        params.srcStride = 1;
        params.dstGap = 0;
        params.ifTranspose = false;
        return params;
    }

    __aicore__ inline void RunGmToA1()
    {
        const uint64_t chunks = tiling_.totalBytes / tiling_.chunkBytes;
        auto params = DataCopyParamsForChunk();
        auto local = a1Queue_.AllocTensor<half>();
        for (uint32_t iter = 0; iter < tiling_.iterations; ++iter) {
            for (uint64_t chunk = coreIdx_; chunk < chunks; chunk += tiling_.activeLanes) {
                AscendC::DataCopy(local, xGM_[(chunk * tiling_.chunkBytes) / sizeof(half)], params);
                AscendC::PipeBarrier<PIPE_MTE2>();
            }
        }
        a1Queue_.FreeTensor(local);
    }

    __aicore__ inline void RunGmToB1()
    {
        const uint64_t chunks = tiling_.totalBytes / tiling_.chunkBytes;
        auto params = DataCopyParamsForChunk();
        auto local = b1Queue_.AllocTensor<half>();
        for (uint32_t iter = 0; iter < tiling_.iterations; ++iter) {
            for (uint64_t chunk = coreIdx_; chunk < chunks; chunk += tiling_.activeLanes) {
                AscendC::DataCopy(local, xGM_[(chunk * tiling_.chunkBytes) / sizeof(half)], params);
                AscendC::PipeBarrier<PIPE_MTE2>();
            }
        }
        b1Queue_.FreeTensor(local);
    }

    __aicore__ inline void RunL1ToL0A()
    {
        auto copyParams = DataCopyParamsForChunk();
        auto loadParams = LoadDataParamsForChunk();
        auto a1 = a1Queue_.AllocTensor<half>();
        AscendC::DataCopy(a1, xGM_[(coreIdx_ * tiling_.chunkBytes) / sizeof(half)], copyParams);
        AscendC::PipeBarrier<PIPE_MTE2>();

        auto a2 = a2Queue_.AllocTensor<half>();
        for (uint32_t iter = 0; iter < tiling_.iterations; ++iter) {
            AscendC::LoadData(a2, a1, loadParams);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        a2Queue_.FreeTensor(a2);
        a1Queue_.FreeTensor(a1);
    }

    __aicore__ inline void RunL1ToL0B()
    {
        auto copyParams = DataCopyParamsForChunk();
        auto loadParams = LoadDataParamsForChunk();
        auto b1 = b1Queue_.AllocTensor<half>();
        AscendC::DataCopy(b1, xGM_[(coreIdx_ * tiling_.chunkBytes) / sizeof(half)], copyParams);
        AscendC::PipeBarrier<PIPE_MTE2>();

        auto b2 = b2Queue_.AllocTensor<half>();
        for (uint32_t iter = 0; iter < tiling_.iterations; ++iter) {
            AscendC::LoadData(b2, b1, loadParams);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        b2Queue_.FreeTensor(b2);
        b1Queue_.FreeTensor(b1);
    }

    __aicore__ inline void RunGmToL0A()
    {
        const uint64_t chunks = tiling_.totalBytes / tiling_.chunkBytes;
        auto params = DataCopyParamsForChunk();
        auto a2 = a2Queue_.AllocTensor<half>();
        for (uint32_t iter = 0; iter < tiling_.iterations; ++iter) {
            for (uint64_t chunk = coreIdx_; chunk < chunks; chunk += tiling_.activeLanes) {
                AscendC::DataCopy(a2, xGM_[(chunk * tiling_.chunkBytes) / sizeof(half)], params);
                AscendC::PipeBarrier<PIPE_ALL>();
            }
        }
        a2Queue_.FreeTensor(a2);
    }

    __aicore__ inline void RunGmToL0B()
    {
        const uint64_t chunks = tiling_.totalBytes / tiling_.chunkBytes;
        auto params = DataCopyParamsForChunk();
        auto b2 = b2Queue_.AllocTensor<half>();
        for (uint32_t iter = 0; iter < tiling_.iterations; ++iter) {
            for (uint64_t chunk = coreIdx_; chunk < chunks; chunk += tiling_.activeLanes) {
                AscendC::DataCopy(b2, xGM_[(chunk * tiling_.chunkBytes) / sizeof(half)], params);
                AscendC::PipeBarrier<PIPE_ALL>();
            }
        }
        b2Queue_.FreeTensor(b2);
    }

private:
    BandwidthTiling tiling_{};
    uint32_t coreIdx_{0};
    AscendC::GlobalTensor<half> xGM_;
    AscendC::TQue<AscendC::TPosition::A1, 1> a1Queue_;
    AscendC::TQue<AscendC::TPosition::B1, 1> b1Queue_;
    AscendC::TQue<AscendC::TPosition::A2, 1> a2Queue_;
    AscendC::TQue<AscendC::TPosition::B2, 1> b2Queue_;
};
