#include "cuda/conv2d.h"

#include "cuda/check.cuh"
#include "cuda/gemm_core.cuh"

#include <algorithm>
#include <cmath>
#include <random>

namespace cuda {

namespace {

// The convolution of a batch is three products. With K = in * k * k and P = out_h * out_w, the columns of sample n are
// the K x P matrix of the input patches under its output pixels; none of them is ever stored, the operand readers below
// compute the address of an element from its row and column.
//
//   forward:     Y_n (out x P)  = W (out x K) * columns_n (K x P)           + bias
//   weights:     dW  (out x K) += sum over n of dY_n (out x P) * columns_n^T (P x K)
//   input:       dX_n (in x HW) = W^T (in x out*k*k, summed over the taps) * dY_n, which reads dY at the output pixels
//                                 that every tap of the input pixel fed

// Blocks wanted in flight for a product that has few tiles: a few per multiprocessor (14 on the GTX 1650).
constexpr int kTargetBlocks = 56;

struct Geometry {
    int in = 0;
    int out = 0;
    int k = 0;
    int stride = 0;
    int pad = 0;
    int h = 0;
    int w = 0;
    int ho = 0;
    int wo = 0;
    __host__ __device__ int p_count() const { return ho * wo; }
    __host__ __device__ int k_count() const { return in * k * k; }
};

struct ForwardOp {
    static constexpr bool kAKFast = true;
    static constexpr bool kBKFast = false;

    Geometry g;
    const float* x;
    const float* weight;
    const float* bias;
    float* y;
    FastDiv by_p, by_wo, by_taps, by_k;

    __device__ float load_a(int i, int kk) const { return weight[static_cast<size_t>(i) * g.k_count() + kk]; }

    __device__ float load_b(int kk, int j) const {
        int n, p, oy, ox, c, rem, ky, kx;
        by_p.divmod(j, n, p);
        by_wo.divmod(p, oy, ox);
        by_taps.divmod(kk, c, rem);
        by_k.divmod(rem, ky, kx);
        const int iy = oy * g.stride - g.pad + ky;
        const int ix = ox * g.stride - g.pad + kx;
        if (iy < 0 || iy >= g.h || ix < 0 || ix >= g.w) {
            return 0.0f;
        }
        return x[(static_cast<size_t>(n) * g.in + c) * g.h * g.w + iy * g.w + ix];
    }

    __device__ void store(int i, int j, float value, int) const {
        int n, p;
        by_p.divmod(j, n, p);
        y[(static_cast<size_t>(n) * g.out + i) * g.p_count() + p] = value + bias[i];
    }
};

struct WeightGradOp {
    static constexpr bool kAKFast = true;
    static constexpr bool kBKFast = true;

    Geometry g;
    const float* x;
    const float* dy;
    float* partial;  // {slices, out, K}: the sums of the slices of the pixel dimension
    FastDiv by_p, by_wo, by_taps, by_k;

    __device__ float load_a(int i, int kt) const {
        int n, q;
        by_p.divmod(kt, n, q);
        return dy[(static_cast<size_t>(n) * g.out + i) * g.p_count() + q];
    }

    __device__ float load_b(int kt, int j) const {
        int n, q, oy, ox, c, rem, ky, kx;
        by_p.divmod(kt, n, q);
        by_wo.divmod(q, oy, ox);
        by_taps.divmod(j, c, rem);
        by_k.divmod(rem, ky, kx);
        const int iy = oy * g.stride - g.pad + ky;
        const int ix = ox * g.stride - g.pad + kx;
        if (iy < 0 || iy >= g.h || ix < 0 || ix >= g.w) {
            return 0.0f;
        }
        return x[(static_cast<size_t>(n) * g.in + c) * g.h * g.w + iy * g.w + ix];
    }

    __device__ void store(int i, int j, float value, int slice) const {
        partial[(static_cast<size_t>(slice) * g.out + i) * g.k_count() + j] = value;
    }
};

struct InputGradOp {
    static constexpr bool kAKFast = true;
    static constexpr bool kBKFast = false;

    Geometry g;
    const float* weight;
    const float* dy;
    float* dx;
    FastDiv by_hw, by_w, by_taps, by_k;

    __device__ float load_a(int i, int kk) const {
        int co, rem;
        by_taps.divmod(kk, co, rem);
        return weight[(static_cast<size_t>(co) * g.in + i) * g.k * g.k + rem];
    }

    __device__ float load_b(int kk, int j) const {
        int n, pixel, iy, ix, co, rem, ky, kx;
        by_hw.divmod(j, n, pixel);
        by_w.divmod(pixel, iy, ix);
        by_taps.divmod(kk, co, rem);
        by_k.divmod(rem, ky, kx);
        const int ny = iy + g.pad - ky;
        const int nx = ix + g.pad - kx;
        if (ny < 0 || nx < 0) {
            return 0.0f;
        }
        int oy = ny;
        int ox = nx;
        if (g.stride != 1) {
            if (ny % g.stride != 0 || nx % g.stride != 0) {
                return 0.0f;
            }
            oy = ny / g.stride;
            ox = nx / g.stride;
        }
        if (oy >= g.ho || ox >= g.wo) {
            return 0.0f;
        }
        return dy[((static_cast<size_t>(n) * g.out + co) * g.ho + oy) * g.wo + ox];
    }

    __device__ void store(int i, int j, float value, int) const {
        int n, pixel;
        by_hw.divmod(j, n, pixel);
        dx[(static_cast<size_t>(n) * g.in + i) * g.h * g.w + pixel] = value;
    }
};

// dW += the partial sums of the slices, added in a fixed order so that a run is the same every time.
__global__ void add_slices_kernel(const float* partial, float* dweight, int count, int slices) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        float sum = 0.0f;
        for (int s = 0; s < slices; ++s) {
            sum += partial[static_cast<size_t>(s) * count + i];
        }
        dweight[i] += sum;
    }
}

// One block per output channel.
__global__ void bias_grad_kernel(const float* dy, float* db, int n, int c, int p_count) {
    __shared__ float shared[32];
    float sum = 0.0f;
    for (int i = threadIdx.x; i < n * p_count; i += blockDim.x) {
        sum += dy[((i / p_count) * c + blockIdx.x) * p_count + i % p_count];
    }
    sum = block_sum(sum, shared);
    if (threadIdx.x == 0) {
        db[blockIdx.x] += sum;
    }
}

}

Conv2d::Conv2d(int in_channels, int out_channels, int kernel, int stride, int padding, unsigned seed)
    : in_(in_channels), out_(out_channels), k_(kernel), stride_(stride), pad_(padding) {
    weight_.init({out_, in_, k_, k_});
    bias_.init({1, out_, 1, 1});
    // Kaiming uniform: the bound keeps the variance of the activations through a ReLU-like layer.
    const float bound = std::sqrt(6.0f / static_cast<float>(in_ * k_ * k_));
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-bound, bound);
    std::vector<float> values(weight_.value.size());
    for (float& v : values) {
        v = dist(rng);
    }
    weight_.value.upload(values);
}

const Tensor& Conv2d::forward(const Tensor& x) {
    const Shape s = x.shape();
    if (s.c != in_) {
        throw std::invalid_argument("Conv2d::forward: channel count differs");
    }
    Geometry g;
    g.in = in_;
    g.out = out_;
    g.k = k_;
    g.stride = stride_;
    g.pad = pad_;
    g.h = s.h;
    g.w = s.w;
    g.ho = (s.h + 2 * pad_ - k_) / stride_ + 1;
    g.wo = (s.w + 2 * pad_ - k_) / stride_ + 1;
    y_.resize({s.n, out_, g.ho, g.wo});
    launch_gemm(ForwardOp{g, x.data(), weight_.value.data(), bias_.value.data(), y_.data(), FastDiv(g.p_count()),
                          FastDiv(g.wo), FastDiv(k_ * k_), FastDiv(k_)},
                out_, s.n * g.p_count(), g.k_count());
    return y_;
}

const Tensor& Conv2d::backward(const Tensor& x, const Tensor& dy) {
    const Shape s = x.shape();
    const Shape ys = y_.shape();
    if (!(dy.shape() == ys) || s.n != ys.n) {
        throw std::invalid_argument("Conv2d::backward: shapes differ from the last forward");
    }
    Geometry g;
    g.in = in_;
    g.out = out_;
    g.k = k_;
    g.stride = stride_;
    g.pad = pad_;
    g.h = s.h;
    g.w = s.w;
    g.ho = ys.h;
    g.wo = ys.w;
    dx_.resize(s);
    // dW: the K dimension of the product runs over the pixels of every sample in turn.
    // The product is small (out x K) and its K dimension long (every pixel of the batch), so a block per tile would
    // leave most of the multiprocessors idle: cut K into slices, a few blocks per multiprocessor in all.
    const int tiles = ((out_ + 63) / 64) * ((g.k_count() + 63) / 64);
    const int pixels = s.n * g.p_count();
    const int slices = std::clamp(kTargetBlocks / tiles, 1, std::max(1, pixels / 256));
    partial_.resize({slices, out_, g.k_count(), 1});
    launch_gemm(WeightGradOp{g, x.data(), dy.data(), partial_.data(), FastDiv(g.p_count()), FastDiv(g.wo),
                             FastDiv(k_ * k_), FastDiv(k_)},
                out_, g.k_count(), pixels, slices);
    add_slices_kernel<<<blocks_for(weight_.grad.size()), kThreads>>>(partial_.data(), weight_.grad.data(),
                                                                       static_cast<int>(weight_.grad.size()), slices);
    bias_grad_kernel<<<out_, kThreads>>>(dy.data(), bias_.grad.data(), s.n, out_, g.p_count());
    launch_gemm(InputGradOp{g, weight_.value.data(), dy.data(), dx_.data(), FastDiv(s.h * s.w), FastDiv(s.w),
                            FastDiv(k_ * k_), FastDiv(k_)},
                in_, s.n * s.h * s.w, out_ * k_ * k_);
    check(cudaGetLastError());
    return dx_;
}

}
