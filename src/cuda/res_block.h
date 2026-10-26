#pragma once

#include "cuda/conv2d.h"
#include "cuda/group_norm.h"
#include "cuda/silu.h"
#include "cuda/tensor.h"

#include <memory>
#include <vector>

namespace cuda {

// GroupNorm - SiLU - conv, then the time embedding (SiLU - dense) is added to every pixel, then GroupNorm - SiLU - conv,
// and the input joins the result (through a 1x1 convolution when the channel count changes).
// backward() must follow the forward() of the same input and embedding.
class ResBlock {
public:
    ResBlock(int in_channels, int out_channels, int embedding, int groups, unsigned seed);

    const Tensor& forward(const Tensor& x, const Tensor& emb);
    // Adds to the gradients of the parameters (zero them first) and to demb, the gradient of the embedding, which
    // every block of the network contributes to. Returns the gradient of the input.
    const Tensor& backward(const Tensor& x, const Tensor& emb, const Tensor& dout, Tensor& demb);

    [[nodiscard]] std::vector<Param*> params();
    // The first convolution: the one that reads the input of the block.
    [[nodiscard]] Conv2d& first_conv() { return conv1_; }

private:
    GroupNorm gn1_;
    Silu silu1_;
    Conv2d conv1_;
    Silu emb_act_;
    Conv2d emb_proj_;
    GroupNorm gn2_;
    Silu silu2_;
    Conv2d conv2_;
    std::unique_ptr<Conv2d> skip_;

    // The activations of the last forward(), owned by the layers above.
    const Tensor* a_ = nullptr;
    const Tensor* b_ = nullptr;
    const Tensor* e_ = nullptr;
    const Tensor* f_ = nullptr;
    const Tensor* g_ = nullptr;
    Tensor h_;
    Tensor y_;
    Tensor dp_;
    Tensor dx_;
};

}
