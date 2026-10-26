#pragma once

#include "loc/localized.h"
#include "loc/message.h"
#include "path_dialog.h"
#include "selection.h"
#include "status_line.h"
#include "view/app_view_model.h"
#include "view/name_list.h"

#include "data/dataset_file.h"
#include "data/dataset_store.h"
#include "data/sprite_importer.h"

#include <engine/core/window_control.h>

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace game {

// The Datasets tab: turns a folder of pictures into a dataset of the app, lists the datasets, shows the sprites of the
// one that is marked and removes it.
class DatasetPanel final : public Localized {
public:
    DatasetPanel(AppViewModel& view, data::DatasetStore store, const Tr& tr, StatusLine& status,
            engine::IWindowControl& windows);

    // Once a frame, on the main thread: the folder dialog and the import in progress.
    void update();
    // Reads the store again; the marked dataset stays marked while it exists.
    void refresh();
    void apply_locale() override;

    // The marked dataset and the store it is in, for the tabs that use it.
    [[nodiscard]] const SelectedDataset& choice() const { return choice_; }

private:
    void browse();
    void import_dataset();
    void cancel_import();
    void remove();
    void page_prev();
    void page_next();
    [[nodiscard]] bool can_browse() const { return importer_ == nullptr; }
    [[nodiscard]] bool can_import() const;
    [[nodiscard]] bool can_cancel() const { return importer_ != nullptr; }
    [[nodiscard]] bool can_remove() const { return names_.has_selection() && importer_ == nullptr; }
    [[nodiscard]] bool can_page_prev() const { return page_ > 0; }
    [[nodiscard]] bool can_page_next() const { return (page_ + 1) * kPageSize < preview_.count(); }

    void show_page();
    void show_texts();
    void say_import(Message message);
    void finish_import();

    static constexpr std::size_t kPageSize = 32;
    static constexpr int kPageColumns = 8;

    AppViewModel& view_;
    const Tr& tr_;
    StatusLine& status_;
    data::DatasetStore store_;
    PathDialog dialog_;
    std::filesystem::path folder_;
    std::unique_ptr<data::SpriteImporter> importer_;
    std::string importing_name_;
    std::vector<data::DatasetInfo> infos_;
    NameList names_;
    SelectedDataset choice_;
    // The sprites of the marked dataset, for the pages.
    data::Dataset preview_;
    std::size_t page_ = 0;
    // What the import text and the line above the sprites say, as messages.
    Message import_message_;
    Message info_message_;
};

}
