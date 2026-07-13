#pragma once

#include <cstdint>

struct BandwidthTiling {
    uint64_t totalBytes;
    uint32_t chunkBytes;
    uint32_t iterations;
    uint32_t activeLanes;
    uint32_t mode;
};
