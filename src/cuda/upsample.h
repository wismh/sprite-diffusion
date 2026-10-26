#pragma once

#include "cuda/tensor.h"

namespace cuda {

// Nearest-neighbour 2x upsampling: every pixel becomes a 2x2 block.
class Upsample2x {
public:
    const Tensor& forward(const Tensor& x);
    // The gradient of each input pixel is the sum over its 2x2 block.
    const Tensor& backward(const Tensor& dy);

private:
    Tensor y_;
    Tensor dx_;
};

}
