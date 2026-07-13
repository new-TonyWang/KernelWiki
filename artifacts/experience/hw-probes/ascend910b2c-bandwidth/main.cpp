#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "kernel/bandwidth_tiling.h"
#include "kernel/cube_bandwidth_modes.h"

extern "C" void bandwidth_kernel_do(
    uint32_t blockDim,
    void *stream,
    uint8_t *x,
    uint8_t *y,
    uint8_t *tiling);

extern "C" void cube_bandwidth_kernel_do(
    uint32_t blockDim,
    void *stream,
    uint8_t *x,
    uint8_t *y,
    uint8_t *tiling);

namespace {

constexpr uint32_t kAicBlocks = 24;
constexpr uint32_t kVecLanes = 48;
constexpr uint32_t kReadMode = 0;
constexpr uint32_t kCopyMode = 1;

void CheckAcl(aclError err, const char *expr)
{
    if (err != ACL_SUCCESS) {
        throw std::runtime_error(std::string(expr) + " failed, aclError=" + std::to_string(err));
    }
}

#define ACL_CHECK(expr) CheckAcl((expr), #expr)

uint64_t ParseSize(const char *text)
{
    if (text == nullptr || *text == '\0') {
        return 0;
    }
    char *end = nullptr;
    const double value = std::strtod(text, &end);
    double scale = 1.0;
    if (end != nullptr) {
        if (std::strcmp(end, "K") == 0 || std::strcmp(end, "KB") == 0) {
            scale = 1000.0;
        } else if (std::strcmp(end, "M") == 0 || std::strcmp(end, "MB") == 0) {
            scale = 1000.0 * 1000.0;
        } else if (std::strcmp(end, "G") == 0 || std::strcmp(end, "GB") == 0) {
            scale = 1000.0 * 1000.0 * 1000.0;
        } else if (std::strcmp(end, "KiB") == 0) {
            scale = 1024.0;
        } else if (std::strcmp(end, "MiB") == 0) {
            scale = 1024.0 * 1024.0;
        } else if (std::strcmp(end, "GiB") == 0) {
            scale = 1024.0 * 1024.0 * 1024.0;
        }
    }
    return static_cast<uint64_t>(value * scale);
}

struct DeviceBuffer {
    void *ptr = nullptr;

    explicit DeviceBuffer(size_t bytes)
    {
        ACL_CHECK(aclrtMalloc(&ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    }

    ~DeviceBuffer()
    {
        if (ptr != nullptr) {
            (void)aclrtFree(ptr);
        }
    }

    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;
};

struct BenchmarkCase {
    std::string name;
    uint32_t mode;
    uint32_t blockDim;
    uint32_t activeLanes;
    uint32_t chunkBytes;
    uint32_t iterations;
    uint64_t fixedWorkBytes;
    double bytesMultiplier;
};

double RunCase(
    const BenchmarkCase &bench,
    uint8_t *x,
    uint8_t *y,
    uint8_t *tilingDev,
    aclrtStream stream,
    uint64_t totalBytes,
    int repeats)
{
    BandwidthTiling tiling{};
    const uint64_t workBytes = bench.fixedWorkBytes == 0 ? totalBytes : bench.fixedWorkBytes;
    tiling.totalBytes = workBytes;
    tiling.chunkBytes = bench.chunkBytes;
    tiling.iterations = bench.iterations;
    tiling.activeLanes = bench.activeLanes;
    tiling.mode = bench.mode;

    ACL_CHECK(aclrtMemcpy(
        tilingDev,
        sizeof(BandwidthTiling),
        &tiling,
        sizeof(BandwidthTiling),
        ACL_MEMCPY_HOST_TO_DEVICE));

    for (int i = 0; i < 3; ++i) {
        cube_bandwidth_kernel_do(bench.blockDim, stream, x, y, tilingDev);
    }
    ACL_CHECK(aclrtSynchronizeStream(stream));

    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(repeats));
    for (int i = 0; i < repeats; ++i) {
        const auto start = std::chrono::steady_clock::now();
        cube_bandwidth_kernel_do(bench.blockDim, stream, x, y, tilingDev);
        ACL_CHECK(aclrtSynchronizeStream(stream));
        const auto stop = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>(stop - start).count();
        samples.push_back(seconds);
    }

    std::sort(samples.begin(), samples.end());
    const double medianSeconds = samples[samples.size() / 2];
    const double movedBytes =
        static_cast<double>(workBytes) * static_cast<double>(bench.iterations) * bench.bytesMultiplier;
    return movedBytes / medianSeconds;
}

}  // namespace

int main(int argc, char **argv)
{
    uint64_t totalBytes = 1ULL << 30;
    int repeats = 7;
    int device = 0;
    if (argc > 1) {
        totalBytes = ParseSize(argv[1]);
    }
    if (argc > 2) {
        repeats = std::max(1, std::atoi(argv[2]));
    }
    if (argc > 3) {
        device = std::atoi(argv[3]);
    }

    constexpr uint32_t readChunk = 128U * 1024U;
    constexpr uint32_t copyChunk = 128U * 1024U;
    totalBytes = (totalBytes / readChunk) * readChunk;
    if (totalBytes < readChunk) {
        std::cerr << "totalBytes must be at least " << readChunk << " bytes\n";
        return 2;
    }

    try {
        ACL_CHECK(aclInit(nullptr));
        ACL_CHECK(aclrtSetDevice(device));
        aclrtStream stream = nullptr;
        ACL_CHECK(aclrtCreateStream(&stream));

        DeviceBuffer x(totalBytes);
        DeviceBuffer y(totalBytes);
        DeviceBuffer tilingDev(sizeof(BandwidthTiling));

        std::vector<uint8_t> zeros(1024 * 1024, 0);
        for (uint64_t offset = 0; offset < totalBytes; offset += zeros.size()) {
            const size_t bytes = static_cast<size_t>(std::min<uint64_t>(zeros.size(), totalBytes - offset));
            ACL_CHECK(aclrtMemcpy(
                static_cast<uint8_t *>(x.ptr) + offset,
                bytes,
                zeros.data(),
                bytes,
                ACL_MEMCPY_HOST_TO_DEVICE));
        }
        ACL_CHECK(aclrtSynchronizeStream(stream));

        constexpr uint32_t l1Chunk = 512U * 1024U;
        constexpr uint32_t l0Chunk = 64U * 1024U;
        const std::vector<BenchmarkCase> cases = {
            {"cube_gm_to_a1_l1_mte2_24_aic", CUBE_GM_TO_A1, kAicBlocks, kAicBlocks, l1Chunk, 32, 0, 1.0},
            {"cube_gm_to_b1_l1_mte2_24_aic", CUBE_GM_TO_B1, kAicBlocks, kAicBlocks, l1Chunk, 32, 0, 1.0},
            {"cube_l1_to_l0a_mte1_24_aic", CUBE_L1_TO_L0A, kAicBlocks, kAicBlocks, l0Chunk, 65536, l0Chunk * static_cast<uint64_t>(kAicBlocks), 1.0},
            {"cube_l1_to_l0b_mte1_24_aic", CUBE_L1_TO_L0B, kAicBlocks, kAicBlocks, l0Chunk, 65536, l0Chunk * static_cast<uint64_t>(kAicBlocks), 1.0},
            {"cube_gm_to_a1_l1_mte2_1_aic", CUBE_GM_TO_A1, 1, 1, l1Chunk, 8, 0, 1.0},
            {"cube_gm_to_b1_l1_mte2_1_aic", CUBE_GM_TO_B1, 1, 1, l1Chunk, 8, 0, 1.0},
            {"cube_l1_to_l0a_mte1_1_aic", CUBE_L1_TO_L0A, 1, 1, l0Chunk, 65536, l0Chunk, 1.0},
            {"cube_l1_to_l0b_mte1_1_aic", CUBE_L1_TO_L0B, 1, 1, l0Chunk, 65536, l0Chunk, 1.0},
        };

        std::cout << "device=" << device << "\n";
        std::cout << "total_bytes=" << totalBytes << "\n";
        std::cout << "repeats=" << repeats << "\n";
        std::cout << "chunk_read_bytes=" << readChunk << "\n";
        std::cout << "chunk_copy_bytes=" << copyChunk << "\n";
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "case,block_dim,active_units,chunk_bytes,work_bytes,iterations,metric_GBps,metric_GiBps\n";

        for (const auto &bench : cases) {
            const double bytesPerSecond = RunCase(
                bench,
                static_cast<uint8_t *>(x.ptr),
                static_cast<uint8_t *>(y.ptr),
                static_cast<uint8_t *>(tilingDev.ptr),
                stream,
                totalBytes,
                repeats);
            std::cout << bench.name << ","
                      << bench.blockDim << ","
                      << bench.activeLanes << ","
                      << bench.chunkBytes << ","
                      << (bench.fixedWorkBytes == 0 ? totalBytes : bench.fixedWorkBytes) << ","
                      << bench.iterations << ","
                      << (bytesPerSecond / 1.0e9) << ","
                      << (bytesPerSecond / static_cast<double>(1ULL << 30)) << "\n";
        }

        ACL_CHECK(aclrtDestroyStream(stream));
        ACL_CHECK(aclrtResetDevice(device));
        ACL_CHECK(aclFinalize());
    } catch (const std::exception &ex) {
        std::cerr << "error: " << ex.what() << "\n";
        (void)aclFinalize();
        return 1;
    }

    return 0;
}
