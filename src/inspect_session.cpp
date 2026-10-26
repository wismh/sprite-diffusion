#include "inspect_session.h"

#include "data/dataset_file.h"

#include <stdexcept>

namespace game {

InspectSession::InspectSession(std::string name, std::filesystem::path path, std::string dataset_name, int steps,
        std::vector<float> dataset)
    : name_(std::move(name))
    , path_(std::move(path))
    , dataset_name_(std::move(dataset_name))
    , steps_(steps)
    , dataset_(std::move(dataset))
    , probe_(path_)
    , dreams_(path_) {}

std::unique_ptr<InspectSession> InspectSession::open(const std::string& name, const data::ModelStore& models,
        const data::DatasetStore& datasets) {
    if (!models.exists(name)) {
        throw std::runtime_error("There is no model \"" + name + "\"");
    }
    const std::filesystem::path path = models.path_of(name);
    const cuda::CheckpointInfo info = cuda::peek_checkpoint(path);
    if (info.dataset.empty() || !datasets.exists(info.dataset)) {
        throw std::runtime_error("The dataset \"" + info.dataset + "\" of this model is not in the app any more");
    }
    const data::Dataset dataset = datasets.load(info.dataset);
    if (dataset.count() == 0) {
        throw std::runtime_error("The dataset \"" + info.dataset + "\" has no sprites");
    }
    return std::unique_ptr<InspectSession>(new InspectSession(
            name, path, info.dataset, info.steps, data::to_floats(dataset, 0, dataset.count())));
}

}
