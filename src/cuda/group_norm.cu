#include "cuda/group_norm.h"

#include "cuda/check.cuh"

namespace cuda {

namespace {

// One block per (sample, group). The channels of a group are contiguous in NCHW, so a group is one flat range.
__global__ void forward_kernel(const float* x, const float* gamma, const float* beta, float* y, float* mean_out,
                               float* rstd_out, int groups, int channels, int hw, float eps) {
    __shared__ float shared[32];
    const int cpg = channels / groups;
    const int count = cpg * hw;
    const int sample = blockIdx.x / groups;
    const int group = blockIdx.x % groups;
    const size_t base = (static_cast<size_t>(sample) * channels + group * cpg) * hw;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        sum += x[base + i];
    }
    const float mean = block_sum(sum, shared) / count;
    float sq = 0.0f;
    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        const float d = x[base + i] - mean;
        sq += d * d;
    }
    const float rstd = rsqrtf(block_sum(sq, shared) / count + eps);
    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        const int channel = group * cpg + i / hw;
        y[base + i] = (x[base + i] - mean) * rstd * gamma[channel] + beta[channel];
    }
    if (threadIdx.x == 0) {
        mean_out[blockIdx.x] = mean;
        rstd_out[blockIdx.x] = rstd;
    }
}

__global__ void backward_kernel(const float* x, const float* dy, const float* gamma, const float* mean_in,
                                const float* rstd_in, float* dx, int groups, int channels, int hw) {
    __shared__ float shared[32];
    const int cpg = channels / groups;
    const int count = cpg * hw;
    const int sample = blockIdx.x / groups;
    const int group = blockIdx.x % groups;
    const size_t base = (static_cast<size_t>(sample) * channels + group * cpg) * hw;
    const float mean = mean_in[blockIdx.x];
    const float rstd = rstd_in[blockIdx.x];
    float s1 = 0.0f;
    float s2 = 0.0f;
    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        const float dxhat = dy[base + i] * gamma[group * cpg + i / hw];
        const float xhat = (x[base + i] - mean) * rstd;
        s1 += dxhat;
        s2 += dxhat * xhat;
    }
    s1 = block_sum(s1, shared) / count;
    s2 = block_sum(s2, shared) / count;
    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        const float dxhat = dy[base + i] * gamma[group * cpg + i / hw];
        const float xhat = (x[base + i] - mean) * rstd;
        dx[base + i] = rstd * (dxhat - s1 - xhat * s2);
    }
}

// One block per channel.
__global__ void param_grad_kernel(const float* x, const float* dy, const float* mean_in, const float* rstd_in,
                                  float* dgamma, float* dbeta, int n, int groups, int channels, int hw) {
    __shared__ float shared[32];
    const int channel = blockIdx.x;
    const int group = channel / (channels / groups);
    float sum_g = 0.0f;
    float sum_b = 0.0f;
    for (int i = threadIdx.x; i < n * hw; i += blockDim.x) {
        const int sample = i / hw;
        const size_t at = (static_cast<size_t>(sample) * channels + channel) * hw + i % hw;
        const int stat = sample * groups + group;
        sum_g += dy[at] * (x[at] - mean_in[stat]) * rstd_in[stat];
        sum_b += dy[at];
    }
    sum_g = block_sum(sum_g, shared);
    sum_b = block_sum(sum_b, shared);
    if (threadIdx.x == 0) {
        dgamma[channel] += sum_g;
        dbeta[channel] += sum_b;
    }
}

__global__ void fill_kernel(float* data, float value, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        data[i] = value;
    }
}

}

GroupNorm::GroupNorm(int groups, int channels, float eps) : groups_(groups), channels_(channels), eps_(eps) {
    if (groups <= 0 || channels % groups != 0) {
        throw std::invalid_argument("GroupNorm: channels must divide into groups");
    }
    gamma_.init({1, channels, 1, 1});
    beta_.init({1, channels, 1, 1});
    fill_kernel<<<blocks_for(channels), kThreads>>>(gamma_.value.data(), 1.0f, channels);
    check(cudaGetLastError());
}

const Tensor& GroupNorm::forward(const Tensor& x) {
    const Shape s = x.shape();
    if (s.c != channels_) {
        throw std::invalid_argument("GroupNorm::forward: channel count differs");
    }
    y_.resize(s);
    mean_.resize({s.n, groups_, 1, 1});
    rstd_.resize({s.n, groups_, 1, 1});
    forward_kernel<<<s.n * groups_, kThreads>>>(x.data(), gamma_.value.data(), beta_.value.data(), y_.data(),
                                                mean_.data(), rstd_.data(), groups_, channels_, s.h * s.w, eps_);
    check(cudaGetLastError());
    return y_;
}

const Tensor& GroupNorm::backward(const Tensor& x, const Tensor& dy) {
    const Shape s = x.shape();
    if (!(dy.shape() == s) || !(y_.shape() == s)) {
        throw std::invalid_argument("GroupNorm::backward: shapes differ from the last forward");
    }
    dx_.resize(s);
    backward_kernel<<<s.n * groups_, kThreads>>>(x.data(), dy.data(), gamma_.value.data(), mean_.data(), rstd_.data(),
                                                 dx_.data(), groups_, channels_, s.h * s.w);
    param_grad_kernel<<<channels_, kThreads>>>(x.data(), dy.data(), mean_.data(), rstd_.data(), gamma_.grad.data(),
                                               beta_.grad.data(), s.n, groups_, channels_, s.h * s.w);
    check(cudaGetLastError());
    return dx_;
}

}
