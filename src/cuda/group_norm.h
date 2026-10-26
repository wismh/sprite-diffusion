#pragma once

#include "cuda/tensor.h"

#include <vector>

namespace cuda {

// Normalizes every group of channels of every sample to zero mean and unit variance, then scales and shifts each
// channel. backward() must follow the forward() of the same input.
class GroupNorm {
public:
    GroupNorm(int groups, int channels, float eps = 1e-5f);

    const Tensor& forward(const Tensor& x);
    // Adds to the gradients of gamma and beta (zero them first) and returns the gradient of the input.
    const Tensor& backward(const Tensor& x, const Tensor& dy);

    [[nodiscard]] Param& gamma() { return gamma_; }
    [[nodiscard]] Param& beta() { return beta_; }
    [[nodiscard]] std::vector<Param*> params() { return {&gamma_, &beta_}; }

private:
    int groups_;
    int channels_;
    float eps_;
    Param gamma_;
    Param beta_;
    Tensor mean_;
    Tensor rstd_;
    Tensor y_;
    Tensor dx_;
};

}
