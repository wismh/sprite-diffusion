#include "cuda/unet.h"

#include "cuda/ops.h"

#include <cmath>
#include <stdexcept>

namespace cuda {

UNet::UNet(UNetConfig config, unsigned seed)
    : config_(config),
      emb_dim_(4 * config.base),
      time1_(config.base, emb_dim_, 1, 1, 0, seed),
      time2_(emb_dim_, emb_dim_, 1, 1, 0, seed + 10),
      conv_in_(3, config.base, 3, 1, 1, seed + 20),
      rb1_(config.base, config.base, emb_dim_, config.groups, seed + 30),
      down1_(config.base, config.base, 3, 2, 1, seed + 40),
      rb2_(config.base, 2 * config.base, emb_dim_, config.groups, seed + 50),
      down2_(2 * config.base, 2 * config.base, 3, 2, 1, seed + 60),
      rbm1_(2 * config.base, 4 * config.base, emb_dim_, config.groups, seed + 70),
      rbm2_(4 * config.base, 4 * config.base, emb_dim_, config.groups, seed + 80),
      up1_conv_(4 * config.base, 2 * config.base, 3, 1, 1, seed + 90),
      rb3_(4 * config.base, 2 * config.base, emb_dim_, config.groups, seed + 100),
      up2_conv_(2 * config.base, config.base, 3, 1, 1, seed + 110),
      rb4_(2 * config.base, config.base, emb_dim_, config.groups, seed + 120),
      gn_out_(config.groups, config.base),
      conv_out_(config.base, 3, 3, 1, 1, seed + 130) {}

const Tensor& UNet::forward(const Tensor& x, const std::vector<float>& t) {
    const int n = x.shape().n;
    if (static_cast<int>(t.size()) != n) {
        throw std::invalid_argument("UNet::forward: one timestep per sample");
    }
    // Sinusoidal embedding of the timestep: sin and cos at base/2 geometric frequencies.
    const int half = config_.base / 2;
    std::vector<float> sinus(static_cast<size_t>(n) * config_.base);
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < half; ++k) {
            const double freq = std::exp(-std::log(10000.0) * k / half);
            sinus[static_cast<size_t>(i) * config_.base + k] = static_cast<float>(std::sin(t[i] * freq));
            sinus[static_cast<size_t>(i) * config_.base + half + k] = static_cast<float>(std::cos(t[i] * freq));
        }
    }
    t_in_.resize({n, config_.base, 1, 1});
    t_in_.upload(sinus);

    l1_ = &time1_.forward(t_in_);
    act_ = &time_act_.forward(*l1_);
    emb_ = &time2_.forward(*act_);

    h0_ = &conv_in_.forward(x);
    r1_ = &rb1_.forward(*h0_, *emb_);
    d1_ = &down1_.forward(*r1_);
    r2_ = &rb2_.forward(*d1_, *emb_);
    d2_ = &down2_.forward(*r2_);
    m1_ = &rbm1_.forward(*d2_, *emb_);
    m2_ = &rbm2_.forward(*m1_, *emb_);
    us1_ = &up1_.forward(*m2_);
    u1_ = &up1_conv_.forward(*us1_);
    concat_channels(*u1_, *r2_, c1_);
    r3_ = &rb3_.forward(c1_, *emb_);
    us2_ = &up2_.forward(*r3_);
    u2_ = &up2_conv_.forward(*us2_);
    concat_channels(*u2_, *r1_, c2_);
    r4_ = &rb4_.forward(c2_, *emb_);
    gno_ = &gn_out_.forward(*r4_);
    so_ = &silu_out_.forward(*gno_);
    return conv_out_.forward(*so_);
}

void UNet::backward(const Tensor& x, const Tensor& dout) {
    demb_.resize(emb_->shape());
    demb_.zero();

    const Tensor& dso = conv_out_.backward(*so_, dout);
    const Tensor& dg = silu_out_.backward(*gno_, dso);
    const Tensor& dr4 = gn_out_.backward(*r4_, dg);
    const Tensor& dc2 = rb4_.backward(c2_, *emb_, dr4, demb_);
    split_channels(dc2, config_.base, du2_, dskip1_);

    const Tensor& dus2 = up2_conv_.backward(*us2_, du2_);
    const Tensor& dr3 = up2_.backward(dus2);
    const Tensor& dc1 = rb3_.backward(c1_, *emb_, dr3, demb_);
    split_channels(dc1, 2 * config_.base, du1_, dskip2_);

    const Tensor& dus1 = up1_conv_.backward(*us1_, du1_);
    const Tensor& dm2 = up1_.backward(dus1);
    const Tensor& dm1 = rbm2_.backward(*m1_, *emb_, dm2, demb_);
    const Tensor& dd2 = rbm1_.backward(*d2_, *emb_, dm1, demb_);

    // r2 feeds both the decoder (skip) and the downsampling convolution.
    add(dskip2_, down2_.backward(*r2_, dd2), dr2_);
    const Tensor& dd1 = rb2_.backward(*d1_, *emb_, dr2_, demb_);
    add(dskip1_, down1_.backward(*r1_, dd1), dr1_);
    const Tensor& dh0 = rb1_.backward(*h0_, *emb_, dr1_, demb_);
    conv_in_.backward(x, dh0);

    const Tensor& da = time2_.backward(*act_, demb_);
    const Tensor& dl1 = time_act_.backward(*l1_, da);
    time1_.backward(t_in_, dl1);
}

void UNet::give(Stage stage, const Tensor& gradient) {
    const auto at = static_cast<size_t>(stage);
    if (reached_[at]) {
        add_inplace(reach_[at], gradient);
    } else {
        reach_[at].copy_from(gradient);
        reached_[at] = true;
    }
}

const Tensor& UNet::backward_from(const Tensor& x, Stage stage, const Tensor& dstage) {
    reached_.fill(false);
    demb_.resize(emb_->shape());
    demb_.zero();
    const int b = config_.base;
    give(stage, dstage);
    for (int k = static_cast<int>(stage); k >= 0; --k) {
        if (!reached_[k]) {
            continue;
        }
        const Tensor& g = reach_[k];
        switch (static_cast<Stage>(k)) {
        case Stage::Res4:
            split_channels(rb4_.backward(c2_, *emb_, g, demb_), b, du2_, dskip1_);
            give(Stage::Up2, du2_);
            give(Stage::Res1, dskip1_);
            break;
        case Stage::Up2:
            give(Stage::Res3, up2_.backward(up2_conv_.backward(*us2_, g)));
            break;
        case Stage::Res3:
            split_channels(rb3_.backward(c1_, *emb_, g, demb_), 2 * b, du1_, dskip2_);
            give(Stage::Up1, du1_);
            give(Stage::Res2, dskip2_);
            break;
        case Stage::Up1:
            give(Stage::Mid2, up1_.backward(up1_conv_.backward(*us1_, g)));
            break;
        case Stage::Mid2:
            give(Stage::Mid1, rbm2_.backward(*m1_, *emb_, g, demb_));
            break;
        case Stage::Mid1:
            give(Stage::Down2, rbm1_.backward(*d2_, *emb_, g, demb_));
            break;
        case Stage::Down2:
            give(Stage::Res2, down2_.backward(*r2_, g));
            break;
        case Stage::Res2:
            give(Stage::Down1, rb2_.backward(*d1_, *emb_, g, demb_));
            break;
        case Stage::Down1:
            give(Stage::Res1, down1_.backward(*r1_, g));
            break;
        case Stage::Res1:
            give(Stage::ConvIn, rb1_.backward(*h0_, *emb_, g, demb_));
            break;
        case Stage::ConvIn:
            return conv_in_.backward(x, g);
        }
    }
    throw std::logic_error("UNet::backward_from: the gradient did not reach the input");
}

const Tensor& UNet::activation(Stage stage) const {
    const Tensor* const all[kStageCount] = {h0_, r1_, d1_, r2_, d2_, m1_, m2_, u1_, r3_, u2_, r4_};
    const Tensor* found = all[static_cast<size_t>(stage)];
    if (found == nullptr) {
        throw std::logic_error("UNet::activation: no forward pass yet");
    }
    return *found;
}

Param& UNet::stage_weight(Stage stage) {
    switch (stage) {
    case Stage::ConvIn: return conv_in_.weight();
    case Stage::Res1: return rb1_.first_conv().weight();
    case Stage::Down1: return down1_.weight();
    case Stage::Res2: return rb2_.first_conv().weight();
    case Stage::Down2: return down2_.weight();
    case Stage::Mid1: return rbm1_.first_conv().weight();
    case Stage::Mid2: return rbm2_.first_conv().weight();
    case Stage::Up1: return up1_conv_.weight();
    case Stage::Res3: return rb3_.first_conv().weight();
    case Stage::Up2: return up2_conv_.weight();
    case Stage::Res4: return rb4_.first_conv().weight();
    }
    throw std::invalid_argument("UNet::stage_weight: no such stage");
}

StageInfo stage_info(const UNetConfig& config, Stage stage) {
    const int b = config.base;
    switch (stage) {
    case Stage::ConvIn: return {b, 16};
    case Stage::Res1: return {b, 16};
    case Stage::Down1: return {b, 8};
    case Stage::Res2: return {2 * b, 8};
    case Stage::Down2: return {2 * b, 4};
    case Stage::Mid1: return {4 * b, 4};
    case Stage::Mid2: return {4 * b, 4};
    case Stage::Up1: return {2 * b, 8};
    case Stage::Res3: return {2 * b, 8};
    case Stage::Up2: return {b, 16};
    case Stage::Res4: return {b, 16};
    }
    throw std::invalid_argument("UNet::stage_info: no such stage");
}

void UNet::zero_grad() {
    for (Param* param : params()) {
        param->zero_grad();
    }
}

std::vector<Param*> UNet::params() {
    std::vector<Param*> all;
    const auto append = [&all](const std::vector<Param*>& list) { all.insert(all.end(), list.begin(), list.end()); };
    append(time1_.params());
    append(time2_.params());
    append(conv_in_.params());
    append(rb1_.params());
    append(down1_.params());
    append(rb2_.params());
    append(down2_.params());
    append(rbm1_.params());
    append(rbm2_.params());
    append(up1_conv_.params());
    append(rb3_.params());
    append(up2_conv_.params());
    append(rb4_.params());
    append(gn_out_.params());
    append(conv_out_.params());
    return all;
}

}
