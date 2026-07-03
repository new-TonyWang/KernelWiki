#pragma once

#include <cstdint>

enum CubeBandwidthMode : uint32_t {
    CUBE_GM_TO_A1 = 0,
    CUBE_GM_TO_B1 = 1,
    CUBE_L1_TO_L0A = 2,
    CUBE_L1_TO_L0B = 3,
    CUBE_GM_TO_L0A = 4,
    CUBE_GM_TO_L0B = 5,
};
