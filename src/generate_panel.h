#pragma once

#include "loc/localized.h"
#include "loc/message.h"
#include "path_dialog.h"
#include "selection.h"
#include "status_line.h"
#include "view/app_view_model.h"

#include "inspect/sprite_generator.h"

#include <engine/core/window_control.h>

#include <filesystem>
#include <memory>
#include <string>

namespace game {

// The Generate tab: draws a number of sprites with the model marked in the Models tab and writes them as PNG files, a new
// folder for each run.
class GeneratePanel final : public Localized {
public:
    // `directory` is the user data directory: pictures go to its "generated" folder unless another folder is chosen
    // (empty: only a chosen folder will do).
    GeneratePanel(AppViewModel& view, const SelectedModel& models, std::filesystem::path directory,
            const Tr& tr, StatusLine& status, engine::IWindowControl& windows);

    // Once a frame, on the main thread: the folder dialog, the model that is marked and the run in progress.
    void update();
    void apply_locale() override;

private:
    void pick_clear();
    void pick_keep();
    void choose_output();
    void generate();
    void cancel();
    void finish();
    void show_folder();
    void show_model();
    [[nodiscard]] bool can_generate() const;
    [[nodiscard]] bool can_cancel() const { return generator_ != nullptr; }
    [[nodiscard]] bool can_choose() const { return generator_ == nullptr; }
    // The number typed in the form, or 0 when it is not one.
    [[nodiscard]] int count() const;

    AppViewModel& view_;
    const SelectedModel& models_;
    const Tr& tr_;
    StatusLine& status_;
    PathDialog dialog_;
    std::filesystem::path base_;
    bool transparent_ = true;
    std::unique_ptr<inspect::SpriteGenerator> generator_;
    std::string generating_model_;
    // The outcome of the last run, as a message.
    Message result_;
};

}
