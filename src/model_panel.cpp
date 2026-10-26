#include "model_panel.h"

#include "data/file_names.h"

#include <engine/log.h>

#include <algorithm>
#include <format>
#include <utility>

namespace game {

ModelPanel::ModelPanel(AppViewModel& view, data::ModelStore store, const SelectedDataset& datasets, TrainPanel& train,
        const Tr& tr, StatusLine& status)
    : view_(view)
    , store_(std::move(store))
    , datasets_(datasets)
    , train_(train)
    , tr_(tr)
    , status_(status)
    , names_(view.models)
    , choice_(names_, store_) {
    view_.pickSmall.bind_to<ModelPanel, &ModelPanel::pick_small>(*this);
    view_.pickMedium.bind_to<ModelPanel, &ModelPanel::pick_medium>(*this);
    view_.pickLarge.bind_to<ModelPanel, &ModelPanel::pick_large>(*this);
    view_.createModel.bind_to<ModelPanel, &ModelPanel::create, &ModelPanel::can_create>(*this);
    view_.openModel.bind_to<ModelPanel, &ModelPanel::open, &ModelPanel::can_use_selected>(*this);
    view_.openModelWithDataset.bind_to<ModelPanel, &ModelPanel::open_with_dataset, &ModelPanel::can_change_dataset>(*this);
    view_.deleteModel.bind_to<ModelPanel, &ModelPanel::remove, &ModelPanel::can_use_selected>(*this);
    pick_medium();
    refresh();
}

bool ModelPanel::can_create() const {
    return store_.available() && !datasets_.selected().empty() &&
           data::valid_file_name(data::trimmed(view_.newModelName.get()));
}

void ModelPanel::pick(int base) {
    base_ = base;
    view_.sizeSmall = base == kSmall;
    view_.sizeMedium = base == kMedium;
    view_.sizeLarge = base == kLarge;
    show_size_text();
}

void ModelPanel::show_size_text() {
    view_.newModelSizeText = tr_(base_ == kSmall ? "model.small" : base_ == kMedium ? "model.medium" : "model.large");
}

void ModelPanel::create() {
    const std::string name = data::trimmed(view_.newModelName.get());
    if (store_.exists(name)) {
        status_.say("model.exists", {{"name", name}});
        return;
    }
    try {
        train_.adopt(Session::create(name, base_, datasets_.selected(), store_, datasets_.store()));
    } catch (const std::exception& error) {
        engine::log::warn(std::format("model \"{}\" was not made: {}", name, error.what()));
        status_.say("model.cannot_make", {{"name", name}, {"detail", error.what()}});
        return;
    }
    names_.select(name);
    view_.newModelName = "";
    refresh();
    view_.show(AppViewModel::Tab::Train);
}

void ModelPanel::open() {
    if (train_.open(names_.selected())) {
        view_.show(AppViewModel::Tab::Train);
    }
}

void ModelPanel::open_with_dataset() {
    if (train_.open(names_.selected(), datasets_.selected())) {
        refresh();
        view_.show(AppViewModel::Tab::Train);
    }
}

void ModelPanel::remove() {
    const std::string name = names_.selected();
    train_.close_if_open(name);
    if (store_.remove(name)) {
        status_.say("model.deleted", {{"name", name}});
    }
    names_.clear_selection();
    refresh();
}

void ModelPanel::refresh() {
    const std::vector<data::ModelInfo> infos = store_.list();
    std::vector<NameList::Row> rows;
    for (const data::ModelInfo& info : infos) {
        std::string label = tr_("model.row",
                {{"name", info.name}, {"base", std::to_string(info.base)}, {"steps", std::to_string(info.steps)}});
        rows.push_back(NameList::Row{info.name, std::move(label)});
    }
    names_.show(rows);
    view_.modelInfoText = tr_(infos.empty() ? "model.none_yet" : "model.choose");
    for (const data::ModelInfo& info : infos) {
        if (info.name == names_.selected()) {
            view_.modelInfoText = tr_("model.info",
                    {{"name", info.name}, {"base", std::to_string(info.base)}, {"steps", std::to_string(info.steps)},
                            {"dataset", info.dataset},
                            {"size", std::format("{:.0f}", static_cast<double>(info.bytes) / (1024.0 * 1024.0))}});
        }
    }
}

void ModelPanel::apply_locale() {
    show_size_text();
    refresh();
}

void ModelPanel::update() {
    if (const std::optional<std::string> picked = names_.take_pick()) {
        names_.select(*picked);
        refresh();
    }
    const std::string& dataset = datasets_.selected();
    view_.newModelDatasetText = dataset.empty() ? tr_("model.pick_dataset") : tr_("model.dataset_is", {{"name", dataset}});
}

}
