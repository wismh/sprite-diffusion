#pragma once

#include "cuda/conv2d.h"
#include "cuda/group_norm.h"
#include "cuda/res_block.h"
#include "cuda/silu.h"
#include "cuda/tensor.h"
#include "cuda/upsample.h"

#include <array>
#include <vector>

namespace cuda {

struct UNetConfig {
    int base = 64;   // channels at 16x16; the levels below have 2x and 4x as many
    int groups = 8;  // GroupNorm groups; must divide base
};

// The places in the network where the activations can be looked at, in the order of the forward pass.
enum class Stage { ConvIn, Res1, Down1, Res2, Down2, Mid1, Mid2, Up1, Res3, Up2, Res4 };
constexpr int kStageCount = 11;

// What a stage holds: `channels` feature maps of size x size (also the number of filters of the convolution that makes it).
struct StageInfo {
    int channels = 0;
    int size = 0;
};

// The size of a stage for a network of this config.
[[nodiscard]] StageInfo stage_info(const UNetConfig& config, Stage stage);

// Predicts the noise in a 16x16 image from the image and its timestep.
//
//   16x16: conv_in - ResBlock ------------------------------------- concat - ResBlock - GN SiLU conv_out
//            | down (conv, stride 2)                                  | up (2x, conv)
//    8x8:  ResBlock ------------------------------- concat - ResBlock
//            | down                                     | up
//    4x4:  ResBlock - ResBlock -----------------------------+
//
// The timestep becomes a sinusoidal vector and goes through two dense layers; every ResBlock adds its own projection of
// the result. backward() must follow the forward() of the same input.
class UNet {
public:
    UNet(UNetConfig config, unsigned seed);

    // x is {n, 3, 16, 16}, t holds the timestep of every sample. The result is the predicted noise, same shape as x.
    const Tensor& forward(const Tensor& x, const std::vector<float>& t);
    // Adds to the gradients of all parameters (zero_grad() first). dout is the gradient of the loss with respect to the
    // result of forward().
    void backward(const Tensor& x, const Tensor& dout);

    // The gradient with respect to x of the activation of `stage` (dstage has its shape), by the same path backward() takes
    // but only through the stages before it. Adds to the gradients of the parameters on the way (the caller does not
    // use them). Follows the forward() of x.
    const Tensor& backward_from(const Tensor& x, Stage stage, const Tensor& dstage);
    // The activation of a stage in the last forward().
    [[nodiscard]] const Tensor& activation(Stage stage) const;
    // The weights ({out, in, 3, 3}) of the convolution that reads the input of the stage and makes its channels.
    [[nodiscard]] Param& stage_weight(Stage stage);
    [[nodiscard]] StageInfo stage_info(Stage stage) const { return cuda::stage_info(config_, stage); }

    void zero_grad();
    [[nodiscard]] std::vector<Param*> params();
    [[nodiscard]] const UNetConfig& config() const { return config_; }

private:
    UNetConfig config_;
    int emb_dim_;

    Conv2d time1_;
    Silu time_act_;
    Conv2d time2_;
    Conv2d conv_in_;
    ResBlock rb1_;
    Conv2d down1_;
    ResBlock rb2_;
    Conv2d down2_;
    ResBlock rbm1_;
    ResBlock rbm2_;
    Upsample2x up1_;
    Conv2d up1_conv_;
    ResBlock rb3_;
    Upsample2x up2_;
    Conv2d up2_conv_;
    ResBlock rb4_;
    GroupNorm gn_out_;
    Silu silu_out_;
    Conv2d conv_out_;

    // The activations of the last forward(), owned by the layers above (or by the tensors below).
    Tensor t_in_;
    const Tensor* l1_ = nullptr;
    const Tensor* act_ = nullptr;
    const Tensor* emb_ = nullptr;
    const Tensor* h0_ = nullptr;
    const Tensor* r1_ = nullptr;
    const Tensor* d1_ = nullptr;
    const Tensor* r2_ = nullptr;
    const Tensor* d2_ = nullptr;
    const Tensor* m1_ = nullptr;
    const Tensor* m2_ = nullptr;
    const Tensor* us1_ = nullptr;
    const Tensor* u1_ = nullptr;
    const Tensor* r3_ = nullptr;
    const Tensor* us2_ = nullptr;
    const Tensor* u2_ = nullptr;
    const Tensor* r4_ = nullptr;
    const Tensor* gno_ = nullptr;
    const Tensor* so_ = nullptr;
    Tensor c1_;
    Tensor c2_;

    // Gradients in flight during backward().
    Tensor demb_;
    Tensor du1_;
    Tensor du2_;
    Tensor dskip1_;
    Tensor dskip2_;
    Tensor dr1_;
    Tensor dr2_;

    // backward_from(): the gradient that has reached each stage, and whether anything has.
    void give(Stage stage, const Tensor& gradient);
    std::array<Tensor, kStageCount> reach_;
    std::array<bool, kStageCount> reached_{};
};

}
