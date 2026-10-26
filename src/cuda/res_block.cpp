#include "cuda/res_block.h"

#include "cuda/ops.h"

namespace cuda {

ResBlock::ResBlock(int in_channels, int out_channels, int embedding, int groups, unsigned seed)
    : gn1_(groups, in_channels),
      conv1_(in_channels, out_channels, 3, 1, 1, seed),
      emb_proj_(embedding, out_channels, 1, 1, 0, seed + 1),
      gn2_(groups, out_channels),
      conv2_(out_channels, out_channels, 3, 1, 1, seed + 2) {
    if (in_channels != out_channels) {
        skip_ = std::make_unique<Conv2d>(in_channels, out_channels, 1, 1, 0, seed + 3);
    }
}

const Tensor& ResBlock::forward(const Tensor& x, const Tensor& emb) {
    a_ = &gn1_.forward(x);
    b_ = &silu1_.forward(*a_);
    const Tensor& c = conv1_.forward(*b_);
    e_ = &emb_act_.forward(emb);
    const Tensor& p = emb_proj_.forward(*e_);
    broadcast_add(c, p, h_);
    f_ = &gn2_.forward(h_);
    g_ = &silu2_.forward(*f_);
    const Tensor& i = conv2_.forward(*g_);
    add(i, skip_ ? skip_->forward(x) : x, y_);
    return y_;
}

const Tensor& ResBlock::backward(const Tensor& x, const Tensor& emb, const Tensor& dout, Tensor& demb) {
    const Tensor& dg = conv2_.backward(*g_, dout);
    const Tensor& df = silu2_.backward(*f_, dg);
    const Tensor& dh = gn2_.backward(h_, df);
    // h = conv1(b) + broadcast(proj(e)): dh flows to the convolution as it is and, summed over the pixels, to the
    // embedding.
    sum_hw(dh, dp_);
    const Tensor& de = emb_proj_.backward(*e_, dp_);
    add_inplace(demb, emb_act_.backward(emb, de));
    const Tensor& db = conv1_.backward(*b_, dh);
    const Tensor& da = silu1_.backward(*a_, db);
    const Tensor& dx_main = gn1_.backward(x, da);
    add(dx_main, skip_ ? skip_->backward(x, dout) : dout, dx_);
    return dx_;
}

std::vector<Param*> ResBlock::params() {
    std::vector<Param*> all;
    for (auto list : {gn1_.params(), conv1_.params(), emb_proj_.params(), gn2_.params(), conv2_.params()}) {
        all.insert(all.end(), list.begin(), list.end());
    }
    if (skip_) {
        const std::vector<Param*> list = skip_->params();
        all.insert(all.end(), list.begin(), list.end());
    }
    return all;
}

}
