#pragma once

#include "loc/localized.h"
#include "selection.h"
#include "status_line.h"
#include "train_panel.h"
#include "view/app_view_model.h"
#include "view/name_list.h"

#include "data/model_store.h"

#include <string>

namespace game {

// The Models tab: makes a new model (a name, a size and the dataset marked in the dataset list), lists the saved ones,
// opens one for training (on its own dataset or on the one marked in the list) and deletes it.
class ModelPanel final : public Localized {
public:
    ModelPanel(AppViewModel& view, data::ModelStore store, const SelectedDataset& datasets, TrainPanel& train, const Tr& tr,
            StatusLine& status);

    // Once a frame: what the form shows about the dataset marked in the list.
    void update();
    // Reads the store again; the marked model stays marked while it exists.
    void refresh();
    void apply_locale() override;

    // The marked model and the store it is in, for the tabs that use it.
    [[nodiscard]] const SelectedModel& choice() const { return choice_; }

private:
    // Channels of the first level of the network: the size of a model.
    static constexpr int kSmall = 32;
    static constexpr int kMedium = 64;
    static constexpr int kLarge = 96;

    void pick_small() { pick(kSmall); }
    void pick_medium() { pick(kMedium); }
    void pick_large() { pick(kLarge); }
    void create();
    void open();
    void open_with_dataset();
    void remove();
    [[nodiscard]] bool can_create() const;
    [[nodiscard]] bool can_use_selected() const { return names_.has_selection(); }
    [[nodiscard]] bool can_change_dataset() const { return names_.has_selection() && !datasets_.selected().empty(); }

    void pick(int base);
    void show_size_text();

    AppViewModel& view_;
    data::ModelStore store_;
    const SelectedDataset& datasets_;
    TrainPanel& train_;
    const Tr& tr_;
    StatusLine& status_;
    int base_ = 64;
    NameList names_;
    SelectedModel choice_;
};

}
