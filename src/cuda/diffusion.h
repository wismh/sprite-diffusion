#pragma once

#include "cuda/adam.h"
#include "cuda/noise_schedule.h"
#include "cuda/tensor.h"
#include "cuda/unet.h"

#include <filesystem>
#include <random>
#include <string>
#include <vector>

namespace cuda {

struct DiffusionConfig {
    int timesteps = 1000;
    UNetConfig net;
    float learning_rate = 3e-4f;
    float ema_decay = 0.999f;
};

// What a checkpoint says about itself, read without the weights.
struct CheckpointInfo {
    int timesteps = 0;
    int base = 0;
    int groups = 0;
    int steps = 0;  // training steps done
    std::string dataset;
};

// The header of a checkpoint file. Does not check the checksum (load() does); throws std::runtime_error when the header is
// not that of a checkpoint.
[[nodiscard]] CheckpointInfo peek_checkpoint(const std::filesystem::path& path);
// The config a Diffusion needs to load the checkpoint that `info` was read from. The other values keep their defaults.
[[nodiscard]] DiffusionConfig config_of(const CheckpointInfo& info);
// peek_checkpoint() and config_of() in one.
[[nodiscard]] DiffusionConfig config_of(const std::filesystem::path& checkpoint);

constexpr int kSpriteSize = 16;
constexpr int kSpriteChannels = 3;
constexpr size_t kSpriteValues = static_cast<size_t>(kSpriteChannels) * kSpriteSize * kSpriteSize;

// Denoising diffusion on 16x16 RGB sprites: a U-Net learns to predict the noise that was added to a sprite at a random
// timestep, and an exponential moving average of its weights generates new sprites from pure noise.
// Pixels are in [-1, 1], laid out as NCHW.
// What sampling went through: for each step the timestep, the sprites as they were fed to the network and the sprite the
// network estimated from them.
struct SampleTrace {
    std::vector<int> timesteps;
    std::vector<std::vector<float>> noisy;
    std::vector<std::vector<float>> estimate;
};

class Diffusion {
public:
    Diffusion(DiffusionConfig config, unsigned seed);

    // One step of Adam on a batch of sprites (count * kSpriteValues values). Returns the loss of the batch.
    double train_step(const std::vector<float>& sprites);

    // New sprites by DDIM from the averaged weights: `steps` network passes instead of one per timestep.
    // With a `trace` the steps are recorded in it.
    [[nodiscard]] std::vector<float> sample(int count, int steps, unsigned seed, SampleTrace* trace = nullptr);

    // The weights of the network and of its average, the moments and step count of Adam and the state of the random
    // generator, in one file: after load() training goes on exactly as if it had never stopped. The architecture and the
    // number of timesteps must match; load() throws std::runtime_error when they do not, or when the file is damaged
    // (it checks a checksum), and then leaves this object unchanged. save() writes a temporary file and renames it, so
    // a crash does not leave a half-written checkpoint.
    void save(const std::filesystem::path& path);
    void load(const std::filesystem::path& path);

    // The name of the dataset the model learns from, kept in the checkpoint so that a model opens with its data.
    [[nodiscard]] const std::string& dataset() const { return dataset_; }
    void set_dataset(std::string name) { dataset_ = std::move(name); }

    [[nodiscard]] UNet& model() { return model_; }
    [[nodiscard]] UNet& ema() { return ema_; }
    [[nodiscard]] const NoiseSchedule& schedule() const { return schedule_; }
    [[nodiscard]] int steps_done() const { return adam_.steps(); }

private:
    DiffusionConfig config_;
    NoiseSchedule schedule_;
    UNet model_;
    UNet ema_;
    Adam adam_;
    std::mt19937 rng_;
    std::string dataset_;
    Tensor x_t_;
    Tensor noise_;
    Tensor grad_;
};

}
