#include "train_panel.h"

#include "data/file_names.h"

#include <engine/log.h>

#include <format>
#include <fstream>
#include <utility>

namespace game {

namespace {

constexpr int kPreviewColumns = 8;

const char* state_key(cuda::TrainerStatus::State state) {
    switch (state) {
    case cuda::TrainerStatus::State::Running:
        return "train.state.training";
    case cuda::TrainerStatus::State::Paused:
        return "train.state.paused";
    case cuda::TrainerStatus::State::Finished:
        return "train.state.finished";
    case cuda::TrainerStatus::State::Failed:
        return "train.state.failed";
    case cuda::TrainerStatus::State::Idle:
        break;
    }
    return "train.state.stopped";
}

}

TrainPanel::TrainPanel(AppViewModel& view, data::ModelStore models, data::DatasetStore datasets,
        std::filesystem::path directory, const Tr& tr, StatusLine& status)
    : view_(view), models_(std::move(models)), datasets_(std::move(datasets)), tr_(tr), status_(status) {
    if (!directory.empty()) {
        last_model_file_ = directory / "last_model.txt";
    }
    view_.toggle.bind_to<TrainPanel, &TrainPanel::toggle, &TrainPanel::has_session>(*this);
    view_.stop.bind_to<TrainPanel, &TrainPanel::stop, &TrainPanel::can_stop>(*this);
    view_.newPreview.bind_to<TrainPanel, &TrainPanel::new_preview, &TrainPanel::has_session>(*this);
    view_.saveCheckpoint.bind_to<TrainPanel, &TrainPanel::save_checkpoint, &TrainPanel::has_session>(*this);
    show_session();
}

void TrainPanel::open_last() {
    if (last_model_file_.empty()) {
        return;
    }
    std::ifstream file(last_model_file_);
    std::string name;
    std::getline(file, name);
    name = data::trimmed(name);
    if (!name.empty() && models_.exists(name)) {
        open(name);
    }
}

void TrainPanel::remember(const std::string& name) const {
    if (!last_model_file_.empty()) {
        std::ofstream(last_model_file_, std::ios::trunc) << name << '\n';
    }
}

bool TrainPanel::open(const std::string& name, const std::string& dataset) {
    // The model that is open ends first: its trainer saves a last checkpoint and gives the GPU back.
    session_.reset();
    show_session();
    try {
        session_ = Session::open(name, models_, datasets_, dataset);
    } catch (const std::exception& error) {
        engine::log::warn(std::format("model \"{}\" did not open: {}", name, error.what()));
        status_.say("train.cannot_open", {{"name", name}, {"detail", error.what()}});
        return false;
    }
    remember(name);
    show_session();
    status_.say("train.opened", {{"name", name}});
    return true;
}

void TrainPanel::adopt(std::unique_ptr<Session> session) {
    session_.reset();
    session_ = std::move(session);
    remember(session_->name());
    show_session();
    status_.say("train.opened", {{"name", session_->name()}});
}

void TrainPanel::close_if_open(const std::string& name) {
    if (session_ && session_->name() == name) {
        session_.reset();
        show_session();
    }
}

void TrainPanel::show_session() {
    view_.lossPaint.reset();
    view_.previewPaint.show({}, kPreviewColumns);
    preview_step_.reset();
    if (session_) {
        // What the model draws as it is, so that the grid is not empty before the first step.
        show_preview(session_->preview());
    }
    show_texts();
}

void TrainPanel::show_texts() {
    view_.lossPaint.set_empty_text(tr_("train.chart_empty"));
    view_.previewText = preview_step_ ? tr_("train.samples_at", {{"step", std::to_string(*preview_step_)}}) : "";
    if (!session_) {
        view_.toggleText = tr_("train.start");
        view_.modelTitleText = tr_("train.none");
        view_.modelDetailText = tr_("train.none_hint");
        view_.stateText = "";
        view_.stepText = "";
        view_.lossText = "";
        view_.checkpointText = "";
        return;
    }
    view_.modelTitleText = session_->name();
    view_.modelDetailText = tr_("train.detail",
            {{"dataset", session_->dataset_name()}, {"count", std::to_string(session_->sprite_count())},
                    {"base", std::to_string(session_->base())},
                    {"params", std::format("{:.1f}", static_cast<double>(session_->parameter_count()) / 1e6)}});
    show_status_texts();
}

void TrainPanel::show_status_texts() {
    const cuda::TrainerStatus status = session_->trainer().status();
    view_.stateText = status.state == cuda::TrainerStatus::State::Failed
            ? tr_("train.failed", {{"detail", status.error}})
            : tr_(state_key(status.state));
    view_.stepText = tr_("train.step", {{"step", std::to_string(status.step)}});
    view_.lossText = status.loss > 0.0 ? tr_("train.loss", {{"loss", std::format("{:.4f}", status.loss)}}) : tr_("train.loss_none");
    view_.toggleText = tr_(status.state == cuda::TrainerStatus::State::Running  ? "train.pause"
                           : status.state == cuda::TrainerStatus::State::Paused ? "train.resume"
                                                                                : "train.start");
    view_.checkpointText = tr_("train.checkpoint",
            {{"path", data::to_utf8(session_->checkpoint_path())}, {"count", std::to_string(status.checkpoints)}});
}

bool TrainPanel::can_stop() const {
    if (!session_) {
        return false;
    }
    const auto state = session_->trainer().status().state;
    return state == cuda::TrainerStatus::State::Running || state == cuda::TrainerStatus::State::Paused;
}

void TrainPanel::toggle() {
    cuda::Trainer& trainer = session_->trainer();
    switch (trainer.status().state) {
    case cuda::TrainerStatus::State::Running:
        trainer.pause();
        break;
    case cuda::TrainerStatus::State::Paused:
        trainer.resume();
        break;
    default:
        trainer.start();
        // The first picture comes before the first step, so there is something to look at while the loss is still empty.
        trainer.request_preview();
        break;
    }
}

void TrainPanel::stop() {
    session_->trainer().stop();
}

void TrainPanel::new_preview() {
    // A running trainer makes the sprites between two steps; a stopped one leaves the model to this thread.
    if (session_->idle()) {
        show_preview(session_->preview());
    } else {
        session_->trainer().request_preview();
    }
}

void TrainPanel::save_checkpoint() {
    if (!session_->idle()) {
        session_->trainer().request_checkpoint();
        return;
    }
    try {
        session_->save();
        status_.say("train.saved", {{"name", session_->name()}});
    } catch (const std::exception& error) {
        status_.say("train.save_failed", {{"detail", error.what()}});
    }
}

void TrainPanel::show_preview(cuda::Preview preview) {
    preview_step_ = preview.step;
    view_.previewText = tr_("train.samples_at", {{"step", std::to_string(preview.step)}});
    view_.previewPaint.show(std::move(preview.sprites), kPreviewColumns);
}

void TrainPanel::apply_locale() {
    show_texts();
}

void TrainPanel::update() {
    if (!session_) {
        return;
    }
    cuda::Trainer& trainer = session_->trainer();
    show_status_texts();
    view_.lossPaint.show(trainer.loss_history());
    if (std::optional<cuda::Preview> preview = trainer.take_preview()) {
        show_preview(std::move(*preview));
    }
}

}
