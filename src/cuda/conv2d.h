#pragma once

#include "cuda/tensor.h"

#include <vector>

namespace cuda {

// A convolution computed as matrix products whose operands are read straight from the images (an implicit GEMM: no
// matrix of columns is built). The weights are {out, in, k, k}, the bias {1, out, 1, 1}.
class Conv2d {
public:
    Conv2d(int in_channels, int out_channels, int kernel, int stride, int padding, unsigned seed);

    const Tensor& forward(const Tensor& x);
    // Adds to the gradients of the weights and the bias (zero them first) and returns the gradient of the input.
    const Tensor& backward(const Tensor& x, const Tensor& dy);

    [[nodiscard]] Param& weight() { return weight_; }
    [[nodiscard]] Param& bias() { return bias_; }
    [[nodiscard]] std::vector<Param*> params() { return {&weight_, &bias_}; }

private:
    int in_;
    int out_;
    int k_;
    int stride_;
    int pad_;
    Param weight_;
    Param bias_;
    Tensor y_;
    Tensor dx_;
    Tensor partial_;  // the weight gradient in slices, before they are added up
};

}
