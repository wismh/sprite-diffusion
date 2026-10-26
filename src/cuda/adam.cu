#include "cuda/adam.h"

#include "cuda/check.cuh"

#include <cmath>

namespace cuda {

namespace {

__global__ void adam_kernel(float* value, const float* grad, float* m, float* v, int count, float lr, float beta1,
                            float beta2, float eps, float correction1, float correction2) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }
    const float g = grad[i];
    m[i] = beta1 * m[i] + (1.0f - beta1) * g;
    v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;
    value[i] -= lr * (m[i] / correction1) / (sqrtf(v[i] / correction2) + eps);
}

}

void Adam::step(const std::vector<Param*>& params) {
    ++step_;
    const float correction1 = 1.0f - std::pow(config_.beta1, static_cast<float>(step_));
    const float correction2 = 1.0f - std::pow(config_.beta2, static_cast<float>(step_));
    for (Param* param : params) {
        if (param->m.size() != param->value.size()) {
            param->m.resize(param->value.shape());
            param->m.zero();
            param->v.resize(param->value.shape());
            param->v.zero();
        }
        adam_kernel<<<blocks_for(param->value.size()), kThreads>>>(
            param->value.data(), param->grad.data(), param->m.data(), param->v.data(),
            static_cast<int>(param->value.size()), config_.learning_rate, config_.beta1, config_.beta2, config_.eps,
            correction1, correction2);
    }
    check(cudaGetLastError());
}

}
