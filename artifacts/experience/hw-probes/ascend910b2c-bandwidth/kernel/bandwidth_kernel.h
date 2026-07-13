#pragma once

#include <cstddef>
#include <cstdint>

#include "bandwidth_tiling.h"
#include "kernel_operator.h"

template <typename T>
__aicore__ inline void CopyTiling(T *tiling, GM_ADDR tilingGM)
{
    int32_t *dst = reinterpret_cast<int32_t *>(tiling);
    auto *src = reinterpret_cast<__gm__ int32_t *>(tilingGM);
    for (size_t i = 0; i < sizeof(T) / sizeof(int32_t); ++i) {
        dst[i] = src[i];
    }
}

class BandwidthKernel {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR tilingGM, AscendC::TPipe *pipe)
    {
        CopyTiling(&tiling_, tilingGM);
        xGM_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(x), tiling_.totalBytes);
        yGM_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(y), tiling_.totalBytes);

        if ASCEND_IS_AIV {
            const uint32_t subBlockCount = AscendC::GetSubBlockNum() - 1U;
            const uint32_t coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
            const uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
            laneIdx_ = coreIdx * subBlockCount + subBlockIdx;
            pipe->InitBuffer(buffer_, tiling_.chunkBytes);
            local_ = buffer_.Get<uint8_t>();
        }
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            if (laneIdx_ >= tiling_.activeLanes || tiling_.chunkBytes == 0) {
                return;
            }
            const uint64_t chunks = tiling_.totalBytes / tiling_.chunkBytes;
            if (chunks == 0) {
                return;
            }

            AscendC::DataCopyParams params{1, static_cast<uint16_t>(tiling_.chunkBytes / 32U), 0, 0};
            AscendC::DataCopyParams tinyParams{1, 1, 0, 0};

            for (uint32_t iter = 0; iter < tiling_.iterations; ++iter) {
                for (uint64_t chunk = laneIdx_; chunk < chunks; chunk += tiling_.activeLanes) {
                    const uint64_t offset = chunk * static_cast<uint64_t>(tiling_.chunkBytes);
                    AscendC::DataCopy(local_, xGM_[offset], params);
                    AscendC::PipeBarrier<PIPE_MTE2>();
                    if (tiling_.mode != 0U) {
                        AscendC::DataCopy(yGM_[offset], local_, params);
                        AscendC::PipeBarrier<PIPE_MTE3>();
                    }
                }
            }

            if (tiling_.mode == 0U) {
                AscendC::DataCopy(yGM_[laneIdx_ * 32U], local_, tinyParams);
                AscendC::PipeBarrier<PIPE_MTE3>();
            }
        }
    }

private:
    BandwidthTiling tiling_{};
    uint32_t laneIdx_{0};
    AscendC::GlobalTensor<uint8_t> xGM_;
    AscendC::GlobalTensor<uint8_t> yGM_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> buffer_;
    AscendC::LocalTensor<uint8_t> local_;
};
