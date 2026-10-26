#pragma once

#include "inspect_session.h"
#include "loc/localized.h"
#include "selection.h"
#include "status_line.h"
#include "view/app_view_model.h"

#include "data/dataset_store.h"

#include <memory>

namespace game {

// The model that the Filters and Process tabs look into: the one marked in the Models tab, loaded on the Load button of
// either tab. Loading takes a moment (the weights and the dataset are read), and the tabs notice a new model by
// generation().
class Inspection final : public Localized {
public:
    Inspection(AppViewModel& view, const SelectedModel& models, data::DatasetStore datasets,
            const Tr& tr, StatusLine& status);

    // The loaded model, or nullptr.
    [[nodiscard]] InspectSession* session() { return session_.get(); }
    // 0 before the first load; one more each time a model is loaded.
    [[nodiscard]] unsigned generation() const { return generation_; }
    void apply_locale() override { show(); }

private:
    void load();
    [[nodiscard]] bool can_load() const { return !models_.selected().empty(); }
    void show();

    AppViewModel& view_;
    const SelectedModel& models_;
    data::DatasetStore datasets_;
    const Tr& tr_;
    StatusLine& status_;
    std::unique_ptr<InspectSession> session_;
    unsigned generation_ = 0;
};

}
