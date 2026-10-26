#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace data {

// What the list of models shows, read from the header of a checkpoint.
struct ModelInfo {
    std::string name;
    std::filesystem::path path;
    int base = 0;  // channels of the first level of the network; the size of the model
    int steps = 0;
    std::string dataset;
    std::uintmax_t bytes = 0;
};

// The models of the app: one `.ckpt` checkpoint each, named after the model, in one folder. An empty folder path is a
// store that has nothing and cannot save.
class ModelStore {
public:
    explicit ModelStore(std::filesystem::path directory) : directory_(std::move(directory)) {}

    [[nodiscard]] bool available() const { return !directory_.empty(); }
    // By name. Files that are not checkpoints of this version are left out.
    [[nodiscard]] std::vector<ModelInfo> list() const;
    [[nodiscard]] bool exists(const std::string& name) const;
    // Where the checkpoint of `name` is (or would be). Throws std::invalid_argument for a name that is not
    // valid_file_name(), std::runtime_error when the store is not available.
    [[nodiscard]] std::filesystem::path path_of(const std::string& name) const;
    // Returns whether there was a model to remove.
    bool remove(const std::string& name) const;

private:
    std::filesystem::path directory_;
};

}
