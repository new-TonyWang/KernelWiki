#ifndef MTE_MICRO_BENCH_TILING_H
#define MTE_MICRO_BENCH_TILING_H
#include <cstdint>
struct MteMicroBenchTilingData {
    uint32_t size;
    uint32_t mode;
    uint32_t loops;
    uint32_t elemsPerLoop;
};
#endif
