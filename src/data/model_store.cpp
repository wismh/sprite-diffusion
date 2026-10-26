#include "data/model_store.h"

#include "cuda/diffusion.h"
#include "data/file_names.h"

#include <algorithm>
#include <stdexcept>

namespace data {

namespace {

constexpr const char* kExtension = ".ckpt";

}

std::filesystem::path ModelStore::path_of(const std::string& name) const {
    if (!valid_file_name(name)) {
        throw std::invalid_argument("not a valid model name: " + name);
    }
    if (!available()) {
        throw std::runtime_error("there is no folder for models");
    }
    return directory_ / from_utf8(name + kExtension);
}

std::vector<ModelInfo> ModelStore::list() const {
    std::vector<ModelInfo> all;
    std::error_code error;
    if (!available() || !std::filesystem::is_directory(directory_, error)) {
        return all;
    }
    for (const auto& entry : std::filesystem::directory_iterator(directory_, error)) {
        if (!entry.is_regular_file() || entry.path().extension() != kExtension) {
            continue;
        }
        try {
            const cuda::CheckpointInfo header = cuda::peek_checkpoint(entry.path());
            ModelInfo info;
            info.name = to_utf8(entry.path().stem());
            info.path = entry.path();
            info.base = header.base;
            info.steps = header.steps;
            info.dataset = header.dataset;
            info.bytes = entry.file_size();
            all.push_back(std::move(info));
        } catch (const std::exception&) {
            // Not a checkpoint of this version: it has no place in the list.
        }
    }
    std::sort(all.begin(), all.end(), [](const ModelInfo& a, const ModelInfo& b) { return a.name < b.name; });
    return all;
}

bool ModelStore::exists(const std::string& name) const {
    return available() && valid_file_name(name) && std::filesystem::exists(path_of(name));
}

bool ModelStore::remove(const std::string& name) const {
    std::error_code error;
    return available() && valid_file_name(name) && std::filesystem::remove(path_of(name), error);
}

}
