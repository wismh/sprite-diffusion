#include "game.h"

#include <asset_ids.h>

#include "loc/locale_prefs.h"

#include <engine/core/platform.h>
#include <engine/loc/catalog.h>
#include <engine/log.h>
#include <engine/ui/canvas.h>

namespace game {

namespace {

constexpr glm::vec2 kDesignSize{1040.0f, 640.0f};

}

Game::Game(const engine::EngineServices& services)
    : engine::GameBase(services.worlds)
    , windows_(&services.windows)
    , assets_(&services.assets) {}

engine::WindowDesc Game::primary_window() const {
    return engine::WindowDesc{.title = "Sprite diffusion",
            .size = {static_cast<int>(kDesignSize.x), static_cast<int>(kDesignSize.y)},
            .style = {.resizable = false}};
}

void Game::on_start() {
    // Datasets live in <user data>/datasets, models in <user data>/models. Without a user data directory nothing can be
    // imported or made, but the window still opens and says so.
    std::filesystem::path directory;
    if (const auto user_data = engine::user_data_directory("wismh", "diffusion")) {
        directory = *user_data;
        preferences_directory_ = directory;
    } else {
        engine::log::warn("no user data directory: datasets and models cannot be saved");
    }
    const std::filesystem::path dataset_directory = directory.empty() ? directory : directory / "datasets";
    const std::filesystem::path model_directory = directory.empty() ? directory : directory / "models";

    load_locale();
    view_ = std::make_shared<AppViewModel>();
    const auto& catalog = world().ctx<engine::loc::Catalog>();
    view_->localeIsEn = catalog.active() == "en";
    view_->localeIsUk = catalog.active() == "uk";
    view_->localeEn.bind_to<Game, &Game::use_english>(*this);
    view_->localeUk.bind_to<Game, &Game::use_ukrainian>(*this);

    status_.emplace(*view_, *tr_);
    datasets_.emplace(*view_, data::DatasetStore{dataset_directory}, *tr_, *status_, *windows_);
    train_.emplace(*view_, data::ModelStore{model_directory}, data::DatasetStore{dataset_directory}, directory, *tr_,
            *status_);
    models_.emplace(*view_, data::ModelStore{model_directory}, datasets_->choice(), *train_, *tr_, *status_);
    generate_.emplace(*view_, models_->choice(), directory, *tr_, *status_, *windows_);
    inspection_.emplace(*view_, models_->choice(), data::DatasetStore{dataset_directory}, *tr_, *status_);
    filters_.emplace(*view_, *inspection_, *tr_);
    process_.emplace(*view_, *inspection_, *tr_);
    help_.emplace(*view_, *tr_);
    localized_ = {&*datasets_, &*train_, &*models_, &*generate_, &*inspection_, &*filters_, &*process_, &*help_};
    for (Localized* panel : localized_) {
        panel->apply_locale();
    }
    status_->say("status.ready");
    train_->open_last();
    models_->refresh();

    engine::ui::UiCanvas canvas;
    canvas.document = assets::ui::app;
    canvas.stylesheet = assets::css::app;
    canvas.fit = engine::ui::UiFit::ScaleWithScreenSize;
    canvas.reference_size = kDesignSize;
    canvas.data_context = view_;
    screen_ = world().create();
    world().emplace<engine::ui::UiCanvas>(screen_, std::move(canvas));

    world().add_system(engine::ecs::Schedule::Frame, engine::ecs::Phase::Game, [this](engine::ecs::World&) {
        datasets_->update();
        models_->update();
        train_->update();
        generate_->update();
        filters_->update();
        process_->update();
    });
}

void Game::load_locale() {
    auto& catalog = world().ctx<engine::loc::Catalog>();
    catalog.add(*assets_->get<engine::loc::StringTable>(assets::locale::en), engine::loc::Role::Source);
    catalog.add(*assets_->get<engine::loc::StringTable>(assets::locale::uk));
    catalog.set_fallback("en");
    catalog.set_active(read_locale(preferences_directory_, "en"));
    tr_.emplace(catalog);
}

void Game::set_locale(std::string_view tag) {
    auto& catalog = world().ctx<engine::loc::Catalog>();
    if (catalog.active() == tag) {
        return;
    }
    catalog.set_active(std::string{tag});
    write_locale(preferences_directory_, tag);
    view_->localeIsEn = tag == "en";
    view_->localeIsUk = tag == "uk";
    for (Localized* panel : localized_) {
        panel->apply_locale();
    }
    status_->refresh();
}

}
