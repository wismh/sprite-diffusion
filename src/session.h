#pragma once

#include "cuda/diffusion.h"
#include "cuda/trainer.h"
#include "data/dataset_store.h"
#include "data/model_store.h"
#include "data/sprite_set.h"

#include <filesystem>
#include <memory>
#include <string>

namespace game {

// A model open for training: its Diffusion, the sprites of its dataset and the thread that trains it. The checkpoint of
// the model is the file the trainer saves to.
class Session {
public:
    // Opens a saved model: its checkpoint and the dataset the checkpoint names. Throws std::runtime_error with a message
    // fit for the status line (the checkpoint is damaged, the dataset is gone, the GPU is not there).
    // A non-empty `dataset` makes the model learn from that dataset from now on: the checkpoint is rewritten to name it.
    [[nodiscard]] static std::unique_ptr<Session> open(const std::string& name, const data::ModelStore& models,
            const data::DatasetStore& datasets, const std::string& dataset = {});
    // Makes a model of `base` channels that learns from `dataset`, writes its first checkpoint and opens it. Throws like
    // open(), and when a model of that name exists.
    [[nodiscard]] static std::unique_ptr<Session> create(const std::string& name, int base, const std::string& dataset,
            const data::ModelStore& models, const data::DatasetStore& datasets);

    [[nodiscard]] const std::string& name() const { return name_; }
    [[nodiscard]] const std::string& dataset_name() const { return dataset_name_; }
    [[nodiscard]] std::size_t sprite_count() const { return sprite_count_; }
    [[nodiscard]] int base() const { return base_; }
    [[nodiscard]] std::size_t parameter_count() const { return parameter_count_; }
    [[nodiscard]] const std::filesystem::path& checkpoint_path() const { return path_; }
    [[nodiscard]] cuda::Trainer& trainer() { return trainer_; }

    // True while no training thread is running, which is when the model may be used from the calling thread.
    [[nodiscard]] bool idle() const;
    // Sprites from the model as it is now, and the checkpoint written now, on the calling thread. Only when idle().
    [[nodiscard]] cuda::Preview preview();
    void save();

private:
    Session(std::string name, std::filesystem::path path, cuda::DiffusionConfig config, unsigned seed,
            const std::string& dataset_name, const data::Dataset& dataset, bool new_model);

    std::string name_;
    std::filesystem::path path_;
    std::string dataset_name_;
    std::size_t sprite_count_;
    int base_;
    // In the order they are built, the trainer last: it ends first, then what it reads from.
    cuda::Diffusion diffusion_;
    cuda::TrainerConfig training_;
    data::SpriteSet sprites_;
    std::size_t parameter_count_ = 0;
    cuda::Trainer trainer_;
};

}
