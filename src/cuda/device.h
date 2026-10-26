#pragma once

#include <string>
#include <vector>

namespace cuda {

struct DeviceInfo {
    std::string name;
    int major = 0;
    int minor = 0;
    size_t total_memory = 0;
};

// The first CUDA device. Throws std::runtime_error when there is none.
[[nodiscard]] DeviceInfo device_info();

// out[i] = a[i] + b[i] on the device; the check that kernels build and run.
[[nodiscard]] std::vector<float> add(const std::vector<float>& a, const std::vector<float>& b);

}
