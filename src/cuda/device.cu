#include "cuda/device.h"

#include <cuda_runtime.h>

#include <stdexcept>

namespace cuda {

namespace {

void check(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

__global__ void add_kernel(const float* a, const float* b, float* out, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        out[i] = a[i] + b[i];
    }
}

}

DeviceInfo device_info() {
    cudaDeviceProp prop{};
    check(cudaGetDeviceProperties(&prop, 0));
    return DeviceInfo{prop.name, prop.major, prop.minor, prop.totalGlobalMem};
}

std::vector<float> add(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("add: sizes differ");
    }
    const int count = static_cast<int>(a.size());
    const size_t bytes = a.size() * sizeof(float);
    float* da = nullptr;
    float* db = nullptr;
    float* dout = nullptr;
    check(cudaMalloc(&da, bytes));
    check(cudaMalloc(&db, bytes));
    check(cudaMalloc(&dout, bytes));
    check(cudaMemcpy(da, a.data(), bytes, cudaMemcpyHostToDevice));
    check(cudaMemcpy(db, b.data(), bytes, cudaMemcpyHostToDevice));
    add_kernel<<<(count + 255) / 256, 256>>>(da, db, dout, count);
    check(cudaGetLastError());
    std::vector<float> out(a.size());
    check(cudaMemcpy(out.data(), dout, bytes, cudaMemcpyDeviceToHost));
    cudaFree(da);
    cudaFree(db);
    cudaFree(dout);
    return out;
}

}
