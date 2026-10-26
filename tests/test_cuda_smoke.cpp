#include <gtest/gtest.h>

#include "cuda/device.h"

TEST(CudaSmoke, SeesTheDevice) {
    const cuda::DeviceInfo info = cuda::device_info();
    EXPECT_FALSE(info.name.empty());
    EXPECT_GT(info.total_memory, size_t{0});
}

TEST(CudaSmoke, KernelAddsVectors) {
    std::vector<float> a(1000);
    std::vector<float> b(1000);
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] = static_cast<float>(i);
        b[i] = 2.0f * static_cast<float>(i);
    }
    const std::vector<float> out = cuda::add(a, b);
    ASSERT_EQ(out.size(), a.size());
    for (size_t i = 0; i < out.size(); ++i) {
        EXPECT_FLOAT_EQ(out[i], 3.0f * static_cast<float>(i));
    }
}
