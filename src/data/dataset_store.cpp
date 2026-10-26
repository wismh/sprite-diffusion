#include "data/dataset_store.h"

#include "data/file_names.h"

#include <algorithm>
#include <stdexcept>

namespace data {

namespace {

constexpr const char* kExtension = ".spds";

}

std::filesystem::path DatasetStore::path_of(const std::string& name) const {
    if (!valid_file_name(name)) {
        throw std::invalid_argument("not a valid dataset name: " + name);
    }
    return directory_ / from_utf8(name + kExtension);
}

std::vector<DatasetInfo> DatasetStore::list() const {
    std::vector<DatasetInfo> all;
    std::error_code error;
    if (!available() || !std::filesystem::is_directory(directory_, error)) {
        return all;
    }
    for (const auto& entry : std::filesystem::directory_iterator(directory_, error)) {
        if (!entry.is_regular_file() || entry.path().extension() != kExtension) {
            continue;
        }
        try {
            all.push_back(peek_dataset(entry.path()));
        } catch (const std::exception&) {
            // Not a dataset (or a damaged one): it has no place in the list.
        }
    }
    std::sort(all.begin(), all.end(), [](const DatasetInfo& a, const DatasetInfo& b) { return a.name < b.name; });
    return all;
}

bool DatasetStore::exists(const std::string& name) const {
    return available() && valid_file_name(name) && std::filesystem::exists(path_of(name));
}

void DatasetStore::save(const std::string& name, const Dataset& dataset) const {
    const std::filesystem::path path = path_of(name);
    if (!available()) {
        throw std::runtime_error("there is no folder for datasets");
    }
    save_dataset(path, dataset);
}

Dataset DatasetStore::load(const std::string& name) const {
    return load_dataset(path_of(name));
}

bool DatasetStore::remove(const std::string& name) const {
    std::error_code error;
    return available() && std::filesystem::remove(path_of(name), error);
}

}
