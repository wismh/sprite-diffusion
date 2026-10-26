#include "cuda/ops.h"

#include "cuda/check.cuh"

namespace cuda {

namespace {

__global__ void add_kernel(const float* a, const float* b, float* out, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        out[i] = a[i] + b[i];
    }
}

__global__ void broadcast_add_kernel(const float* x, const float* v, float* out, int count, int hw) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        out[i] = x[i] + v[i / hw];
    }
}

// One block per (sample, channel).
__global__ void sum_hw_kernel(const float* x, float* out, int hw) {
    __shared__ float shared[32];
    float sum = 0.0f;
    for (int i = threadIdx.x; i < hw; i += blockDim.x) {
        sum += x[static_cast<size_t>(blockIdx.x) * hw + i];
    }
    sum = block_sum(sum, shared);
    if (threadIdx.x == 0) {
        out[blockIdx.x] = sum;
    }
}

__global__ void concat_kernel(const float* a, const float* b, float* out, int count, int ca, int cb, int hw) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }
    const int p = i % hw;
    const int c = (i / hw) % (ca + cb);
    const int n = i / (hw * (ca + cb));
    out[i] = c < ca ? a[(n * ca + c) * hw + p] : b[(n * cb + (c - ca)) * hw + p];
}

__global__ void split_kernel(const float* d, float* da, float* db, int count, int ca, int cb, int hw) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }
    const int p = i % hw;
    const int c = (i / hw) % (ca + cb);
    const int n = i / (hw * (ca + cb));
    if (c < ca) {
        da[(n * ca + c) * hw + p] = d[i];
    } else {
        db[(n * cb + (c - ca)) * hw + p] = d[i];
    }
}

__global__ void mse_kernel(const float* pred, const float* target, float* grad, float* total, int count) {
    __shared__ float shared[32];
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    float sq = 0.0f;
    if (i < count) {
        const float d = pred[i] - target[i];
        grad[i] = 2.0f * d / static_cast<float>(count);
        sq = d * d;
    }
    sq = block_sum(sq, shared);
    if (threadIdx.x == 0) {
        atomicAdd(total, sq);
    }
}

__global__ void ema_kernel(float* ema, const float* source, int count, float decay) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        ema[i] = decay * ema[i] + (1.0f - decay) * source[i];
    }
}

void require_same(const Tensor& a, const Tensor& b, const char* what) {
    if (!(a.shape() == b.shape())) {
        throw std::invalid_argument(what);
    }
}

}

void add(const Tensor& a, const Tensor& b, Tensor& out) {
    require_same(a, b, "add: shapes differ");
    out.resize(a.shape());
    add_kernel<<<blocks_for(a.size()), kThreads>>>(a.data(), b.data(), out.data(), static_cast<int>(a.size()));
    check(cudaGetLastError());
}

void add_inplace(Tensor& acc, const Tensor& x) {
    require_same(acc, x, "add_inplace: shapes differ");
    add_kernel<<<blocks_for(acc.size()), kThreads>>>(acc.data(), x.data(), acc.data(), static_cast<int>(acc.size()));
    check(cudaGetLastError());
}

void broadcast_add(const Tensor& x, const Tensor& v, Tensor& out) {
    const Shape s = x.shape();
    if (!(v.shape() == Shape{s.n, s.c, 1, 1})) {
        throw std::invalid_argument("broadcast_add: v must be {n, c, 1, 1}");
    }
    out.resize(s);
    broadcast_add_kernel<<<blocks_for(x.size()), kThreads>>>(x.data(), v.data(), out.data(),
                                                             static_cast<int>(x.size()), s.h * s.w);
    check(cudaGetLastError());
}

void sum_hw(const Tensor& x, Tensor& out) {
    const Shape s = x.shape();
    out.resize({s.n, s.c, 1, 1});
    sum_hw_kernel<<<s.n * s.c, kThreads>>>(x.data(), out.data(), s.h * s.w);
    check(cudaGetLastError());
}

void concat_channels(const Tensor& a, const Tensor& b, Tensor& out) {
    const Shape sa = a.shape();
    const Shape sb = b.shape();
    if (sa.n != sb.n || sa.h != sb.h || sa.w != sb.w) {
        throw std::invalid_argument("concat_channels: shapes differ");
    }
    out.resize({sa.n, sa.c + sb.c, sa.h, sa.w});
    concat_kernel<<<blocks_for(out.size()), kThreads>>>(a.data(), b.data(), out.data(), static_cast<int>(out.size()),
                                                        sa.c, sb.c, sa.h * sa.w);
    check(cudaGetLastError());
}

void split_channels(const Tensor& d, int channels_a, Tensor& da, Tensor& db) {
    const Shape s = d.shape();
    const int channels_b = s.c - channels_a;
    da.resize({s.n, channels_a, s.h, s.w});
    db.resize({s.n, channels_b, s.h, s.w});
    split_kernel<<<blocks_for(d.size()), kThreads>>>(d.data(), da.data(), db.data(), static_cast<int>(d.size()),
                                                     channels_a, channels_b, s.h * s.w);
    check(cudaGetLastError());
}

double mse_loss(const Tensor& pred, const Tensor& target, Tensor& grad) {
    require_same(pred, target, "mse_loss: shapes differ");
    grad.resize(pred.shape());
    float* total = nullptr;
    check(cudaMalloc(&total, sizeof(float)));
    check(cudaMemset(total, 0, sizeof(float)));
    mse_kernel<<<blocks_for(pred.size()), kThreads>>>(pred.data(), target.data(), grad.data(), total,
                                                      static_cast<int>(pred.size()));
    check(cudaGetLastError());
    float sum = 0.0f;
    check(cudaMemcpy(&sum, total, sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(total);
    return static_cast<double>(sum) / static_cast<double>(pred.size());
}

void ema_update(const std::vector<Param*>& ema, const std::vector<Param*>& source, float decay) {
    if (ema.size() != source.size()) {
        throw std::invalid_argument("ema_update: parameter lists differ");
    }
    for (size_t i = 0; i < ema.size(); ++i) {
        require_same(ema[i]->value, source[i]->value, "ema_update: shapes differ");
        ema_kernel<<<blocks_for(ema[i]->value.size()), kThreads>>>(ema[i]->value.data(), source[i]->value.data(),
                                                                   static_cast<int>(ema[i]->value.size()), decay);
    }
    check(cudaGetLastError());
}

}
