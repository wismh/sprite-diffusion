#include "dataset_panel.h"

#include "data/file_names.h"

#include <algorithm>
#include <format>
#include <utility>

namespace game {

namespace {

// What stands in the place of transparent pixels: a color that the sprites hardly ever use, so that it can be keyed out of
// the generated ones.
constexpr std::array<std::uint8_t, 3> kBackground{255, 0, 255};

}

DatasetPanel::DatasetPanel(AppViewModel& view, data::DatasetStore store, const Tr& tr, StatusLine& status,
        engine::IWindowControl& windows)
    : view_(view), tr_(tr), status_(status), store_(std::move(store)), dialog_(windows), names_(view.datasets), choice_(names_, store_) {
    view_.browseFolder.bind_to<DatasetPanel, &DatasetPanel::browse, &DatasetPanel::can_browse>(*this);
    view_.importDataset.bind_to<DatasetPanel, &DatasetPanel::import_dataset, &DatasetPanel::can_import>(*this);
    view_.cancelImport.bind_to<DatasetPanel, &DatasetPanel::cancel_import, &DatasetPanel::can_cancel>(*this);
    view_.deleteDataset.bind_to<DatasetPanel, &DatasetPanel::remove, &DatasetPanel::can_remove>(*this);
    view_.pagePrev.bind_to<DatasetPanel, &DatasetPanel::page_prev, &DatasetPanel::can_page_prev>(*this);
    view_.pageNext.bind_to<DatasetPanel, &DatasetPanel::page_next, &DatasetPanel::can_page_next>(*this);
    if (!store_.available()) {
        import_message_ = Message("ds.no_store");
    }
    refresh();
    apply_locale();
}

bool DatasetPanel::can_import() const {
    return store_.available() && importer_ == nullptr && !folder_.empty() &&
           data::valid_file_name(data::trimmed(view_.datasetName.get()));
}

void DatasetPanel::say_import(Message message) {
    import_message_ = std::move(message);
    view_.importText = import_message_.text(tr_);
}

void DatasetPanel::browse() {
    dialog_.choose_folder(folder_);
}

void DatasetPanel::import_dataset() {
    const std::string name = data::trimmed(view_.datasetName.get());
    if (store_.exists(name)) {
        say_import(Message("ds.exists", {{"name", name}}));
        return;
    }
    importing_name_ = name;
    importer_ = std::make_unique<data::SpriteImporter>(folder_, kBackground);
    view_.importing = true;
    say_import(Message("ds.reading_folder"));
}

void DatasetPanel::cancel_import() {
    importer_->cancel();
}

void DatasetPanel::finish_import() {
    data::ImportResult result = importer_->take_result();
    importer_.reset();
    view_.importing = false;
    if (result.cancelled) {
        say_import(Message("ds.cancelled"));
        return;
    }
    if (!result.error.empty()) {
        say_import(Message("ds.failed", {{"detail", result.error}}));
        return;
    }
    try {
        store_.save(importing_name_, result.dataset);
    } catch (const std::exception& error) {
        say_import(Message("ds.save_failed", {{"detail", error.what()}}));
        return;
    }
    say_import(result.skipped > 0
                    ? Message("ds.imported_skipped",
                              {{"count", std::to_string(result.dataset.count())}, {"skipped", std::to_string(result.skipped)}})
                    : Message("ds.imported", {{"count", std::to_string(result.dataset.count())}}));
    status_.say("ds.added", {{"name", importing_name_}});
    names_.select(importing_name_);
    refresh();
}

void DatasetPanel::remove() {
    const std::string name = names_.selected();
    if (store_.remove(name)) {
        status_.say("ds.removed", {{"name", name}});
    }
    names_.clear_selection();
    refresh();
}

void DatasetPanel::page_prev() {
    --page_;
    show_page();
}

void DatasetPanel::page_next() {
    ++page_;
    show_page();
}

void DatasetPanel::refresh() {
    infos_ = store_.list();
    std::vector<NameList::Row> rows;
    for (const data::DatasetInfo& info : infos_) {
        std::string label = tr_("ds.row", {{"name", info.name}, {"count", std::to_string(info.count)}});
        rows.push_back(NameList::Row{info.name, std::move(label)});
    }
    names_.show(rows);

    preview_ = {};
    page_ = 0;
    if (const auto found = std::find_if(infos_.begin(), infos_.end(),
                [this](const data::DatasetInfo& info) { return info.name == names_.selected(); });
            found != infos_.end()) {
        try {
            preview_ = store_.load(names_.selected());
            info_message_ = Message("ds.info",
                    {{"name", found->name}, {"count", std::to_string(found->count)}, {"source", found->source}});
        } catch (const std::exception& error) {
            info_message_ = Message("ds.read_failed", {{"detail", error.what()}});
        }
    } else {
        info_message_ = Message(infos_.empty() ? "ds.none_yet" : "ds.choose");
    }
    view_.datasetInfoText = info_message_.text(tr_);
    show_page();
}

void DatasetPanel::show_page() {
    view_.datasetPaint.show(data::to_floats(preview_, page_ * kPageSize, kPageSize), kPageColumns);
    show_texts();
}

void DatasetPanel::show_texts() {
    const std::size_t pages = (preview_.count() + kPageSize - 1) / kPageSize;
    view_.pageText = pages == 0 ? ""
                                : tr_("ds.page", {{"page", std::to_string(page_ + 1)}, {"pages", std::to_string(pages)}});
}

void DatasetPanel::apply_locale() {
    refresh();
    view_.importFolderText = folder_.empty() ? tr_("ds.no_folder") : data::to_utf8(folder_);
    if (importer_ == nullptr) {
        view_.importText = import_message_.text(tr_);
    }
}

void DatasetPanel::update() {
    if (const std::optional<std::string> picked = names_.take_pick()) {
        names_.select(*picked);
        refresh();
    }
    if (const std::optional<std::filesystem::path> chosen = dialog_.take()) {
        folder_ = *chosen;
        view_.importFolderText = data::to_utf8(folder_);
        if (data::trimmed(view_.datasetName.get()).empty()) {
            view_.datasetName = data::to_utf8(folder_.filename());
        }
    }
    if (importer_ == nullptr) {
        return;
    }
    if (importer_->finished()) {
        finish_import();
    } else if (importer_->total() > 0) {
        view_.importText = tr_("ds.progress",
                {{"done", std::to_string(importer_->done())}, {"total", std::to_string(importer_->total())}});
    }
}

}
