#pragma once

#include "cuda/diffusion.h"
#include "cuda/unet.h"

#include <array>
#include <filesystem>
#include <vector>

namespace cuda {

// What the network (the averaged weights) does with one sprite at one noise level.
struct Answer {
    std::vector<float> sprite;           // the clean sprite
    std::vector<float> noisy;            // the sprite with noise of the timestep added
    std::vector<float> noise;            // the noise that was added
    std::vector<float> predicted_noise;  // what the network says the noise is
    std::vector<float> estimate;         // the clean sprite as the network works it out from that
    // The feature maps of every stage: channels * size * size values each, a channel after a channel.
    std::array<std::vector<float>, kStageCount> maps;
};

// A saved model opened for looking into: its weights and what it does to a sprite. Used on one thread.
class Probe {
public:
    // Throws std::runtime_error when the checkpoint cannot be read.
    explicit Probe(const std::filesystem::path& checkpoint);

    [[nodiscard]] StageInfo info(Stage stage) { return diffusion_.ema().stage_info(stage); }
    [[nodiscard]] int timesteps() const { return timesteps_; }
    // `t` is a timestep 0..timesteps-1; the noise comes from `seed`, so the same arguments give the same answer.
    [[nodiscard]] Answer answer(const std::vector<float>& sprite, int t, unsigned seed);
    // The weights of the convolution that makes the channels of the stage: {channels, in, 3, 3}, `in` returned.
    [[nodiscard]] std::vector<float> weights(Stage stage, int& in_channels);
    // `count` sprites from the model and the steps they went through.
    [[nodiscard]] std::vector<float> sample(int count, int steps, unsigned seed, SampleTrace* trace = nullptr) {
        return diffusion_.sample(count, steps, seed, trace);
    }
    [[nodiscard]] Diffusion& diffusion() { return diffusion_; }

private:
    Probe(const std::filesystem::path& checkpoint, const DiffusionConfig& config);

    int timesteps_;
    Diffusion diffusion_;
};

}
