#pragma once

#include "data/dataset_store.h"
#include "data/model_store.h"
#include "view/name_list.h"

#include <string>

namespace game {

// What the tabs that use a dataset or a model of another tab need from it, and no more. (A panel hands out a small object
// of its own instead of being one: a bound member pointer to a class with two bases is more than the compiler manages.)
class SelectedDataset {
public:
    SelectedDataset(const NameList& names, const data::DatasetStore& store)
        : names_(names)
        , store_(store) {}

    // The marked dataset, or empty.
    [[nodiscard]] const std::string& selected() const { return names_.selected(); }
    [[nodiscard]] const data::DatasetStore& store() const { return store_; }

private:
    const NameList& names_;
    const data::DatasetStore& store_;
};

class SelectedModel {
public:
    SelectedModel(const NameList& names, const data::ModelStore& store)
        : names_(names)
        , store_(store) {}

    // The marked model, or empty.
    [[nodiscard]] const std::string& selected() const { return names_.selected(); }
    [[nodiscard]] const data::ModelStore& store() const { return store_; }

private:
    const NameList& names_;
    const data::ModelStore& store_;
};

}