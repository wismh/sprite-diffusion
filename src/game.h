#pragma once

#include "dataset_panel.h"
#include "filters_panel.h"
#include "generate_panel.h"
#include "help/help.h"
#include "inspection.h"
#include "loc/tr.h"
#include "model_panel.h"
#include "process_panel.h"
#include "status_line.h"
#include "train_panel.h"
#include "view/app_view_model.h"

#include <engine/core/engine_services.h>
#include <engine/igame.h>
#include <engine/resources/assets_db.h>

#include <filesystem>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

namespace game {

// One window with the Train, Models, Generate, Filters, Process, Datasets and Help tabs. The behavior is in the panels;
// the game owns the lifecycle and the language.
class Game final : public engine::GameBase {
public:
    explicit Game(const engine::EngineServices& services);

    [[nodiscard]] engine::WindowDesc primary_window() const override;
    void on_start() override;

private:
    // Loads the English and Ukrainian messages and picks the language the player used last time.
    void load_locale();
    void set_locale(std::string_view tag);
    void use_english() { set_locale("en"); }
    void use_ukrainian() { set_locale("uk"); }

    engine::IWindowControl* windows_;
    engine::AssetsDb* assets_;
    std::filesystem::path preferences_directory_;
    std::optional<Tr> tr_;
    std::shared_ptr<AppViewModel> view_;
    std::optional<StatusLine> status_;
    std::optional<DatasetPanel> datasets_;
    std::optional<TrainPanel> train_;
    std::optional<ModelPanel> models_;
    std::optional<GeneratePanel> generate_;
    std::optional<Inspection> inspection_;
    std::optional<FiltersPanel> filters_;
    std::optional<ProcessPanel> process_;
    std::optional<Help> help_;
    std::vector<Localized*> localized_;  // every panel that has texts of its own
    engine::ecs::Entity screen_{};
};

}
