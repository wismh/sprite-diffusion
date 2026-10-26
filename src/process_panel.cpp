#include "process_panel.h"

#include "inspect/process_grid.h"

#include <format>
#include <random>

namespace game {

namespace {

constexpr int kCheckSamples = 256;

}

ProcessPanel::ProcessPanel(AppViewModel& view, Inspection& inspection, const Tr& tr)
    : view_(view), inspection_(inspection), tr_(tr) {
    view_.newSamples.bind_to<ProcessPanel, &ProcessPanel::new_samples, &ProcessPanel::has_session>(*this);
    view_.pickEstimate.bind_to<ProcessPanel, &ProcessPanel::pick_estimate, &ProcessPanel::has_session>(*this);
    view_.pickNoisy.bind_to<ProcessPanel, &ProcessPanel::pick_noisy, &ProcessPanel::has_session>(*this);
    view_.checkCopies.bind_to<ProcessPanel, &ProcessPanel::check_copies, &ProcessPanel::can_check>(*this);
    view_.cancelCopies.bind_to<ProcessPanel, &ProcessPanel::cancel_check, &ProcessPanel::can_cancel>(*this);
}

void ProcessPanel::sync_session() {
    generation_ = inspection_.generation();
    check_.reset();
    view_.copyRunning = false;
    session_ = inspection_.session();
    trace_ = {};
    neighbors_.clear();
    copy_distance_ = 0.0;
    copy_message_ = Message();
    view_.copyText = "";
    view_.processText = "";
    view_.processPaint.show({}, kColumns);
    if (session_ != nullptr) {
        new_samples();
    }
}

void ProcessPanel::pick_estimate() {
    view_.viewEstimate = true;
    view_.viewNoisy = false;
    show_grid();
}

void ProcessPanel::pick_noisy() {
    view_.viewEstimate = false;
    view_.viewNoisy = true;
    show_grid();
}

void ProcessPanel::new_samples() {
    std::random_device device;
    trace_ = {};
    const std::vector<float> sprites = session_->probe().sample(kSamples, kSteps, device(), &trace_);
    neighbors_ = inspect::nearest_sprites(sprites, session_->dataset());
    show_grid();
    show_texts();
}

void ProcessPanel::show_grid() {
    const auto& steps = view_.viewEstimate.get() ? trace_.estimate : trace_.noisy;
    // The steps shown are spread evenly over the sampling; the last column is the closest sprite of the dataset.
    view_.processPaint.show(
            inspect::process_grid(steps, inspect::shown_steps(kSteps, kColumns - 1), session_->dataset(), neighbors_),
            kColumns);
}

void ProcessPanel::show_texts() {
    std::string text;
    for (int sample = 0; sample < kSamples; ++sample) {
        const inspect::Neighbor& neighbor = neighbors_[static_cast<size_t>(sample)];
        text += tr_(neighbor.mirrored ? "process.row_turned" : "process.row",
                {{"n", std::to_string(sample + 1)}, {"index", std::to_string(neighbor.index + 1)},
                        {"distance", std::format("{:.2f}", neighbor.distance)}});
        if (copy_distance_ > 0.0 && neighbor.distance <= copy_distance_) {
            text += tr_("process.near_data");
        }
        text += '\n';
    }
    view_.processText = text;
}

void ProcessPanel::check_copies() {
    check_ = std::make_unique<inspect::CopyCheck>(session_->checkpoint_path(), session_->dataset(), kCheckSamples);
    view_.copyRunning = true;
    copy_message_ = Message("process.measuring");
    view_.copyText = copy_message_.text(tr_);
}

void ProcessPanel::finish_check() {
    const inspect::CopyCheckResult result = check_->take_result();
    check_.reset();
    view_.copyRunning = false;
    if (!result.ok()) {
        copy_message_ = result.cancelled ? Message("process.cancelled")
                                         : Message("process.failed", {{"detail", result.error}});
        view_.copyText = copy_message_.text(tr_);
        return;
    }
    copy_distance_ = result.baseline.median;
    const auto two = [](double value) { return std::format("{:.2f}", value); };
    copy_message_ = Message("process.result",
            {{"share", std::format("{:.0f}", 100.0 * result.share_near)}, {"samples", std::to_string(result.samples)},
                    {"typical", two(result.baseline.median)}, {"new_low", two(result.samples_spread.low)},
                    {"new_median", two(result.samples_spread.median)}, {"new_high", two(result.samples_spread.high)},
                    {"data_low", two(result.baseline.low)}, {"data_median", two(result.baseline.median)},
                    {"data_high", two(result.baseline.high)}});
    view_.copyText = copy_message_.text(tr_);
    show_texts();
}

void ProcessPanel::apply_locale() {
    if (session_ == nullptr) {
        return;
    }
    show_texts();
    if (check_ == nullptr) {
        view_.copyText = copy_message_.text(tr_);
    }
}

void ProcessPanel::update() {
    if (generation_ != inspection_.generation()) {
        sync_session();
    }
    if (check_ == nullptr) {
        return;
    }
    if (check_->finished()) {
        finish_check();
    } else {
        view_.copyText = tr_("process.checking", {{"percent", std::format("{:.0f}", 100.0 * check_->progress())}});
    }
}

}
