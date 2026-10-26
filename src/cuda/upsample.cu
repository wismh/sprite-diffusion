#include "cuda/upsample.h"

#include "cuda/check.cuh"

namespace cuda {

namespace {

__global__ void forward_kernel(const float* x, float* y, int count, int h, int w) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }
    const int ox = i % (2 * w);
    const int oy = (i / (2 * w)) % (2 * h);
    const int plane = i / (4 * h * w);
    y[i] = x[(plane * h + oy / 2) * w + ox / 2];
}

__global__ void backward_kernel(const float* dy, float* dx, int count, int h, int w) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }
    const int ix = i % w;
    const int iy = (i / w) % h;
    const int plane = i / (h * w);
    const float* block = dy + (static_cast<size_t>(plane) * 2 * h + 2 * iy) * 2 * w + 2 * ix;
    dx[i] = block[0] + block[1] + block[2 * w] + block[2 * w + 1];
}

}

const Tensor& Upsample2x::forward(const Tensor& x) {
    const Shape s = x.shape();
    y_.resize({s.n, s.c, 2 * s.h, 2 * s.w});
    forward_kernel<<<blocks_for(y_.size()), kThreads>>>(x.data(), y_.data(), static_cast<int>(y_.size()), s.h, s.w);
    check(cudaGetLastError());
    return y_;
}

const Tensor& Upsample2x::backward(const Tensor& dy) {
    const Shape s = dy.shape();
    dx_.resize({s.n, s.c, s.h / 2, s.w / 2});
    backward_kernel<<<blocks_for(dx_.size()), kThreads>>>(dy.data(), dx_.data(), static_cast<int>(dx_.size()),
                                                          s.h / 2, s.w / 2);
    check(cudaGetLastError());
    return dx_;
}

}
