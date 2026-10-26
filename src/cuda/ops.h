#pragma once

#include "cuda/tensor.h"

#include <vector>

namespace cuda {

// out = a + b (same shape). out is resized.
void add(const Tensor& a, const Tensor& b, Tensor& out);
// acc += x (same shape).
void add_inplace(Tensor& acc, const Tensor& x);
// out[n, c, h, w] = x[n, c, h, w] + v[n, c]; v is {n, c, 1, 1}. The time embedding joins a feature map this way.
void broadcast_add(const Tensor& x, const Tensor& v, Tensor& out);
// out[n, c] = sum over h, w of x[n, c, h, w]; out is {n, c, 1, 1}. The gradient of broadcast_add with respect to v.
void sum_hw(const Tensor& x, Tensor& out);
// out is {n, a.c + b.c, h, w}: the channels of a, then the channels of b.
void concat_channels(const Tensor& a, const Tensor& b, Tensor& out);
// The gradient of concat_channels: da gets the first channels of d, db the rest. Both are resized.
void split_channels(const Tensor& d, int channels_a, Tensor& da, Tensor& db);

// Mean squared error over all elements. Writes d(loss)/d(pred) to grad and returns the loss.
double mse_loss(const Tensor& pred, const Tensor& target, Tensor& grad);

// ema = decay * ema + (1 - decay) * source for every pair of parameters (same shapes). decay 0 copies.
void ema_update(const std::vector<Param*>& ema, const std::vector<Param*>& source, float decay);

}
