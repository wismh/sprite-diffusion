#include "cuda/diffusion.h"

#include "cuda/ops.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace cuda {

Diffusion::Diffusion(DiffusionConfig config, unsigned seed)
    : config_(config),
      schedule_(config.timesteps),
      model_(config.net, seed),
      ema_(config.net, seed),
      adam_(AdamConfig{.learning_rate = config.learning_rate}),
      rng_(seed) {
    ema_update(ema_.params(), model_.params(), 0.0f);
}

double Diffusion::train_step(const std::vector<float>& sprites) {
    if (sprites.empty() || sprites.size() % kSpriteValues != 0) {
        throw std::invalid_argument("Diffusion::train_step: a whole number of sprites");
    }
    const int count = static_cast<int>(sprites.size() / kSpriteValues);
    std::uniform_int_distribution<int> pick_t(0, schedule_.steps() - 1);
    std::normal_distribution<float> normal;

    // x_t = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * noise, at an independent random timestep per sprite.
    std::vector<float> timesteps(count);
    std::vector<float> noise(sprites.size());
    std::vector<float> noisy(sprites.size());
    for (int i = 0; i < count; ++i) {
        const int t = pick_t(rng_);
        timesteps[i] = static_cast<float>(t);
        const double signal = std::sqrt(schedule_.alpha_bar(t));
        const double spread = std::sqrt(1.0 - schedule_.alpha_bar(t));
        for (size_t k = 0; k < kSpriteValues; ++k) {
            const size_t at = static_cast<size_t>(i) * kSpriteValues + k;
            noise[at] = normal(rng_);
            noisy[at] = static_cast<float>(signal * sprites[at] + spread * noise[at]);
        }
    }
    const Shape shape{count, kSpriteChannels, kSpriteSize, kSpriteSize};
    x_t_.resize(shape);
    x_t_.upload(noisy);
    noise_.resize(shape);
    noise_.upload(noise);

    model_.zero_grad();
    const Tensor& predicted = model_.forward(x_t_, timesteps);
    const double loss = mse_loss(predicted, noise_, grad_);
    model_.backward(x_t_, grad_);
    adam_.step(model_.params());

    // The average starts close to the weights and slows down as training goes on.
    const int done = adam_.steps();
    const float decay = std::min(config_.ema_decay, static_cast<float>(1 + done) / static_cast<float>(10 + done));
    ema_update(ema_.params(), model_.params(), decay);
    return loss;
}

std::vector<float> Diffusion::sample(int count, int steps, unsigned seed, SampleTrace* trace) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal;
    std::vector<float> x(static_cast<size_t>(count) * kSpriteValues);
    for (float& v : x) {
        v = normal(rng);
    }

    // DDIM without randomness: from the noise, estimate the sprite, then step to the earlier timestep along the
    // estimated noise. The timesteps are spread evenly from the last one down to 0.
    std::vector<int> ts(steps);
    for (int i = 0; i < steps; ++i) {
        ts[i] = steps == 1 ? schedule_.steps() - 1
                           : static_cast<int>(std::lround((schedule_.steps() - 1.0) * (steps - 1 - i) / (steps - 1)));
    }
    const Shape shape{count, kSpriteChannels, kSpriteSize, kSpriteSize};
    Tensor input(shape);
    for (int i = 0; i < steps; ++i) {
        input.upload(x);
        const std::vector<float> t(count, static_cast<float>(ts[i]));
        const std::vector<float> predicted = ema_.forward(input, t).download();
        const double bar = schedule_.alpha_bar(ts[i]);
        const double bar_prev = i + 1 < steps ? schedule_.alpha_bar(ts[i + 1]) : 1.0;
        if (trace != nullptr) {
            trace->timesteps.push_back(ts[i]);
            trace->noisy.push_back(x);
            trace->estimate.emplace_back(x.size());
        }
        for (size_t k = 0; k < x.size(); ++k) {
            const double sprite =
                std::clamp((x[k] - std::sqrt(1.0 - bar) * predicted[k]) / std::sqrt(bar), -1.0, 1.0);
            if (trace != nullptr) {
                trace->estimate.back()[k] = static_cast<float>(sprite);
            }
            x[k] = static_cast<float>(std::sqrt(bar_prev) * sprite + std::sqrt(1.0 - bar_prev) * predicted[k]);
        }
    }
    for (float& v : x) {
        v = std::clamp(v, -1.0f, 1.0f);
    }
    return x;
}

}
