#pragma once

#include "cuda/tensor.h"

namespace cuda {

// y = x * sigmoid(x). No parameters.
class Silu {
public:
    const Tensor& forward(const Tensor& x);
    const Tensor& backward(const Tensor& x, const Tensor& dy);

private:
    Tensor y_;
    Tensor dx_;
};

}
