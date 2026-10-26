#include "session.h"

#include <random>
#include <stdexcept>

namespace game {

namespace {

constexpr int kGroups = 8;

cuda::TrainerConfig trainer_config(const std::filesystem::path& checkpoint) {
    cuda::TrainerConfig config;
    config.batch = 64;
    config.preview_every = 250;
    config.preview_count = 16;
    config.preview_steps = 30;
    config.checkpoint_every = 500;
    config.checkpoint_path = checkpoint;
    // A little room between steps for the game's own drawing on the same GPU.
    config.throttle_ms = 10;
    return config;
}

data::Dataset load_dataset_of(const std::string& name, const data::DatasetStore& datasets) {
    if (name.empty() || !datasets.exists(name)) {
        throw std::runtime_error("The dataset \"" + name + "\" of this model is not in the app any more");
    }
    data::Dataset dataset = datasets.load(name);
    if (dataset.count() == 0) {
        throw std::runtime_error("The dataset \"" + name + "\" has no sprites");
    }
    return dataset;
}

// The Diffusion of a model, with its checkpoint written (a new model) or loaded (a saved one) before the trainer sees it.
cuda::Diffusion make_diffusion(const cuda::DiffusionConfig& config, unsigned seed, const std::filesystem::path& path,
        const std::string& dataset_name, bool new_model) {
    cuda::Diffusion diffusion(config, seed);
    if (new_model) {
        diffusion.set_dataset(dataset_name);
        diffusion.save(path);
    } else {
        diffusion.load(path);
        if (diffusion.dataset() != dataset_name) {
            diffusion.set_dataset(dataset_name);
            diffusion.save(path);
        }
    }
    return diffusion;
}

}

Session::Session(std::string name, std::filesystem::path path, cuda::DiffusionConfig config, unsigned seed,
        const std::string& dataset_name, const data::Dataset& dataset, bool new_model)
    : name_(std::move(name))
    , path_(std::move(path))
    , dataset_name_(dataset_name)
    , sprite_count_(dataset.count())
    , base_(config.net.base)
    , diffusion_(make_diffusion(config, seed, path_, dataset_name, new_model))
    , training_(trainer_config(path_))
    , sprites_(data::to_floats(dataset, 0, dataset.count()), seed, true)
    , trainer_(diffusion_, training_, [this](int count) { return sprites_.next_batch(count); }) {
    for (const cuda::Param* param : diffusion_.model().params()) {
        parameter_count_ += param->value.size();
    }
}

bool Session::idle() const {
    const auto state = trainer_.status().state;
    return state != cuda::TrainerStatus::State::Running && state != cuda::TrainerStatus::State::Paused;
}

cuda::Preview Session::preview() {
    cuda::Preview preview;
    preview.step = diffusion_.steps_done();
    preview.sprites = diffusion_.sample(training_.preview_count, training_.preview_steps, std::random_device{}());
    return preview;
}

void Session::save() {
    diffusion_.save(path_);
}

std::unique_ptr<Session> Session::open(const std::string& name, const data::ModelStore& models,
        const data::DatasetStore& datasets, const std::string& dataset_name) {
    if (!models.exists(name)) {
        throw std::runtime_error("There is no model \"" + name + "\"");
    }
    const std::filesystem::path path = models.path_of(name);
    const cuda::CheckpointInfo info = cuda::peek_checkpoint(path);
    const cuda::DiffusionConfig config = cuda::config_of(info);
    const std::string& used = dataset_name.empty() ? info.dataset : dataset_name;
    const data::Dataset dataset = load_dataset_of(used, datasets);
    return std::unique_ptr<Session>(new Session(name, path, config, std::random_device{}(), used, dataset, false));
}

std::unique_ptr<Session> Session::create(const std::string& name, int base, const std::string& dataset_name,
        const data::ModelStore& models, const data::DatasetStore& datasets) {
    if (models.exists(name)) {
        throw std::runtime_error("A model named \"" + name + "\" exists already");
    }
    const data::Dataset dataset = load_dataset_of(dataset_name, datasets);
    cuda::DiffusionConfig config;
    config.net = {.base = base, .groups = kGroups};
    return std::unique_ptr<Session>(
            new Session(name, models.path_of(name), config, std::random_device{}(), dataset_name, dataset, true));
}

}
