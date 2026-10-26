#pragma once

#include "cuda/dream_worker.h"
#include "cuda/probe.h"
#include "data/dataset_store.h"
#include "data/model_store.h"

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace game {

// A saved model open for looking into (the weights as last saved): what it does to a sprite, what its filters look like,
// the sprites its channels answer to hardest, and the dataset it learned from for comparing.
class InspectSession {
public:
    // Throws std::runtime_error with a message fit for the status line (no such model, a damaged checkpoint, the dataset is
    // gone, no GPU).
    [[nodiscard]] static std::unique_ptr<InspectSession> open(const std::string& name, const data::ModelStore& models,
            const data::DatasetStore& datasets);

    [[nodiscard]] const std::string& name() const { return name_; }
    [[nodiscard]] const std::string& dataset_name() const { return dataset_name_; }
    [[nodiscard]] int steps() const { return steps_; }
    [[nodiscard]] const std::filesystem::path& checkpoint_path() const { return path_; }
    // The sprites of the dataset: NCHW, [-1, 1].
    [[nodiscard]] const std::vector<float>& dataset() const { return dataset_; }
    [[nodiscard]] size_t sprite_count() const { return dataset_.size() / cuda::kSpriteValues; }
    [[nodiscard]] cuda::Probe& probe() { return probe_; }
    [[nodiscard]] cuda::DreamWorker& dreams() { return dreams_; }

private:
    InspectSession(std::string name, std::filesystem::path path, std::string dataset_name, int steps,
            std::vector<float> dataset);

    std::string name_;
    std::filesystem::path path_;
    std::string dataset_name_;
    int steps_;
    std::vector<float> dataset_;
    cuda::Probe probe_;
    cuda::DreamWorker dreams_;
};

}
