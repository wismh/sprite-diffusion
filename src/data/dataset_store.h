#pragma once

#include "data/dataset_file.h"

#include <filesystem>
#include <string>
#include <vector>

namespace data {

// The datasets of the app: one `.spds` file each, named after the dataset, in one folder. An empty folder path is a store
// that has nothing and cannot save (no user data directory).
class DatasetStore {
public:
    explicit DatasetStore(std::filesystem::path directory) : directory_(std::move(directory)) {}

    [[nodiscard]] bool available() const { return !directory_.empty(); }
    // By name. Files that are not datasets are left out.
    [[nodiscard]] std::vector<DatasetInfo> list() const;
    [[nodiscard]] bool exists(const std::string& name) const;
    // Throws std::invalid_argument for a name that is not valid_file_name(), std::runtime_error when the store is not
    // available or the file cannot be written. Replaces a dataset of the same name.
    void save(const std::string& name, const Dataset& dataset) const;
    [[nodiscard]] Dataset load(const std::string& name) const;
    // Returns whether there was a dataset to remove.
    bool remove(const std::string& name) const;

private:
    [[nodiscard]] std::filesystem::path path_of(const std::string& name) const;

    std::filesystem::path directory_;
};

}
