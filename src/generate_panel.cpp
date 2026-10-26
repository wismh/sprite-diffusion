#include "generate_panel.h"

#include "data/file_names.h"

#include <charconv>
#include <chrono>
#include <format>
#include <utility>

namespace game {

namespace {

constexpr int kMaxCount = 100000;
constexpr int kPreviewColumns = 8;

// A path is one long word that a label cannot wrap: its end is the part that tells folders apart.
std::string shortened(const std::string& path) {
    constexpr std::size_t kShown = 36;
    return path.size() > kShown ? "..." + path.substr(path.size() - kShown) : path;
}

}

GeneratePanel::GeneratePanel(AppViewModel& view, const SelectedModel& models, std::filesystem::path directory,
        const Tr& tr, StatusLine& status, engine::IWindowControl& windows)
    : view_(view)
    , models_(models)
    , tr_(tr)
    , status_(status)
    , dialog_(windows)
    , base_(directory.empty() ? directory : directory / "generated") {
    view_.pickClear.bind_to<GeneratePanel, &GeneratePanel::pick_clear>(*this);
    view_.pickKeep.bind_to<GeneratePanel, &GeneratePanel::pick_keep>(*this);
    view_.chooseOutput.bind_to<GeneratePanel, &GeneratePanel::choose_output, &GeneratePanel::can_choose>(*this);
    view_.generate.bind_to<GeneratePanel, &GeneratePanel::generate, &GeneratePanel::can_generate>(*this);
    view_.cancelGenerate.bind_to<GeneratePanel, &GeneratePanel::cancel, &GeneratePanel::can_cancel>(*this);
    show_folder();
}

void GeneratePanel::pick_clear() {
    transparent_ = true;
    view_.backgroundClear = true;
    view_.backgroundKeep = false;
}

void GeneratePanel::pick_keep() {
    transparent_ = false;
    view_.backgroundClear = false;
    view_.backgroundKeep = true;
}

int GeneratePanel::count() const {
    const std::string text = data::trimmed(view_.generateCount.get());
    int value = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size() || value < 1 || value > kMaxCount) {
        return 0;
    }
    return value;
}

bool GeneratePanel::can_generate() const {
    return generator_ == nullptr && !base_.empty() && !models_.selected().empty() &&
           models_.store().exists(models_.selected()) && count() > 0;
}

void GeneratePanel::show_folder() {
    view_.generateFolderText = base_.empty() ? tr_("gen.no_folder") : shortened(data::to_utf8(base_));
}

void GeneratePanel::show_model() {
    const std::string& model = models_.selected();
    view_.generateModelText = generator_ ? tr_("gen.drawing", {{"name", generating_model_}})
            : model.empty()              ? tr_("gen.mark_model")
                                         : tr_("gen.model", {{"name", model}});
}

void GeneratePanel::choose_output() {
    dialog_.choose_folder(base_);
}

void GeneratePanel::generate() {
    const std::string model = models_.selected();
    inspect::GenerateRequest request;
    request.checkpoint = models_.store().path_of(model);
    request.prefix = model;
    request.directory = base_ / (model + "-" + data::timestamp_text(std::chrono::system_clock::now()));
    request.count = count();
    request.transparent = transparent_;
    generating_model_ = model;
    generator_ = std::make_unique<inspect::SpriteGenerator>(std::move(request));
    view_.generatePaint.show({}, kPreviewColumns);
    result_ = Message("gen.loading");
    view_.generateText = result_.text(tr_);
}

void GeneratePanel::cancel() {
    generator_->cancel();
}

void GeneratePanel::finish() {
    const inspect::GenerateResult result = generator_->take_result();
    const std::string folder = shortened(data::to_utf8(generator_->directory()));
    generator_.reset();
    const std::string written = std::to_string(result.written);
    if (result.ok()) {
        result_ = Message("gen.done", {{"count", written}, {"folder", folder}});
        status_.say("gen.saved", {{"count", written}, {"name", generating_model_}});
    } else if (result.cancelled) {
        result_ = Message("gen.cancelled", {{"count", written}, {"folder", folder}});
    } else {
        result_ = Message("gen.failed", {{"detail", result.error}});
    }
    view_.generateText = result_.text(tr_);
}

void GeneratePanel::apply_locale() {
    show_folder();
    show_model();
    if (generator_ == nullptr) {
        view_.generateText = result_.text(tr_);
    }
}

void GeneratePanel::update() {
    if (const std::optional<std::filesystem::path> chosen = dialog_.take()) {
        base_ = *chosen;
        show_folder();
    }
    show_model();
    if (generator_ == nullptr) {
        return;
    }
    if (generator_->finished()) {
        view_.generatePaint.show(generator_->latest(), kPreviewColumns);
        finish();
        return;
    }
    if (generator_->done() > 0) {
        view_.generateText = tr_("gen.progress",
                {{"done", std::to_string(generator_->done())}, {"total", std::to_string(generator_->total())}});
        view_.generatePaint.show(generator_->latest(), kPreviewColumns);
    }
}

}
