#include "cuda/probe.h"

#include <cmath>
#include <random>

namespace cuda {

Probe::Probe(const std::filesystem::path& checkpoint) : Probe(checkpoint, config_of(checkpoint)) {}

Probe::Probe(const std::filesystem::path& checkpoint, const DiffusionConfig& config)
    : timesteps_(config.timesteps), diffusion_(config, 1) {
    diffusion_.load(checkpoint);
}

Answer Probe::answer(const std::vector<float>& sprite, int t, unsigned seed) {
    Answer answer;
    answer.sprite = sprite;
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal;
    const double bar = diffusion_.schedule().alpha_bar(t);
    answer.noise.resize(kSpriteValues);
    answer.noisy.resize(kSpriteValues);
    for (size_t k = 0; k < kSpriteValues; ++k) {
        answer.noise[k] = normal(rng);
        answer.noisy[k] = static_cast<float>(std::sqrt(bar) * sprite[k] + std::sqrt(1.0 - bar) * answer.noise[k]);
    }
    Tensor input({1, kSpriteChannels, kSpriteSize, kSpriteSize});
    input.upload(answer.noisy);
    UNet& net = diffusion_.ema();
    answer.predicted_noise = net.forward(input, {static_cast<float>(t)}).download();
    answer.estimate.resize(kSpriteValues);
    for (size_t k = 0; k < kSpriteValues; ++k) {
        answer.estimate[k] = static_cast<float>(std::clamp(
                (answer.noisy[k] - std::sqrt(1.0 - bar) * answer.predicted_noise[k]) / std::sqrt(bar), -1.0, 1.0));
    }
    for (int i = 0; i < kStageCount; ++i) {
        answer.maps[static_cast<size_t>(i)] = net.activation(static_cast<Stage>(i)).download();
    }
    return answer;
}

std::vector<float> Probe::weights(Stage stage, int& in_channels) {
    Tensor& weight = diffusion_.ema().stage_weight(stage).value;
    in_channels = weight.shape().c;
    return weight.download();
}

}
