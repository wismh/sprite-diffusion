#pragma once

#include "loc/localized.h"
#include "session.h"
#include "status_line.h"
#include "view/app_view_model.h"

#include "data/dataset_store.h"
#include "data/model_store.h"

#include <filesystem>
#include <memory>
#include <optional>
#include <string>

namespace game {

// The Train tab: runs the open model through its Session and shows its training on the view model every frame. Which model
// is open is remembered between runs. Without an open model the buttons are disabled.
class TrainPanel final : public Localized {
public:
    // `directory` is where the name of the last open model is kept (empty: it is not).
    TrainPanel(AppViewModel& view, data::ModelStore models, data::DatasetStore datasets, std::filesystem::path directory,
            const Tr& tr, StatusLine& status);

    // Opens the model of the last run, if there was one and it opens; says why on the status line if it does not.
    void open_last();
    // Opens a saved model, closing the open one first. Returns whether it opened; the status line says why not.
    // A non-empty `dataset` is the one the model learns from from now on.
    bool open(const std::string& name, const std::string& dataset = {});
    // Opens a new model that has just been made.
    void adopt(std::unique_ptr<Session> session);
    // Closes the open model if it is `name`: the model is about to be deleted.
    void close_if_open(const std::string& name);
    // The name of the open model, or empty.
    [[nodiscard]] std::string open_name() const { return session_ ? session_->name() : std::string{}; }

    // Once a frame, on the main thread.
    void update();
    void apply_locale() override;

private:
    void toggle();
    void stop();
    void new_preview();
    void save_checkpoint();
    [[nodiscard]] bool has_session() const { return session_ != nullptr; }
    [[nodiscard]] bool can_stop() const;

    void show_session();
    void show_texts();
    void show_status_texts();
    void show_preview(cuda::Preview preview);
    void remember(const std::string& name) const;

    AppViewModel& view_;
    data::ModelStore models_;
    data::DatasetStore datasets_;
    const Tr& tr_;
    StatusLine& status_;
    std::filesystem::path last_model_file_;
    std::unique_ptr<Session> session_;
    // The step of the samples on show, for the line above them.
    std::optional<int> preview_step_;
};

}
