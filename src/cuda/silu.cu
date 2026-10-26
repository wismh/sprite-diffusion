#include "cuda/silu.h"

#include "cuda/check.cuh"

namespace cuda {

namespace {

__global__ void forward_kernel(const float* x, float* y, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        y[i] = x[i] / (1.0f + expf(-x[i]));
    }
}

__global__ void backward_kernel(const float* x, const float* dy, float* dx, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        const float s = 1.0f / (1.0f + expf(-x[i]));
        dx[i] = dy[i] * s * (1.0f + x[i] * (1.0f - s));
    }
}

}

const Tensor& Silu::forward(const Tensor& x) {
    y_.resize(x.shape());
    forward_kernel<<<blocks_for(x.size()), kThreads>>>(x.data(), y_.data(), static_cast<int>(x.size()));
    check(cudaGetLastError());
    return y_;
}

const Tensor& Silu::backward(const Tensor& x, const Tensor& dy) {
    if (!(dy.shape() == x.shape())) {
        throw std::invalid_argument("Silu::backward: shapes differ");
    }
    dx_.resize(x.shape());
    backward_kernel<<<blocks_for(x.size()), kThreads>>>(x.data(), dy.data(), dx_.data(), static_cast<int>(x.size()));
    check(cudaGetLastError());
    return dx_;
}

}
