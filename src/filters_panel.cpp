#include "filters_panel.h"

#include "inspect/channel_stats.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <random>

namespace game {

namespace {

constexpr std::array<int, 12> kNoiseLevels{0, 25, 50, 100, 150, 200, 300, 400, 500, 650, 800, 999};
constexpr int kDefaultNoise = 6;

}

FiltersPanel::FiltersPanel(AppViewModel& view, Inspection& inspection, const Tr& tr)
    : view_(view), inspection_(inspection), tr_(tr) {
    view_.layerUp.bind_to<FiltersPanel, &FiltersPanel::layer_up, &FiltersPanel::can_layer_up>(*this);
    view_.layerDown.bind_to<FiltersPanel, &FiltersPanel::layer_down, &FiltersPanel::can_layer_down>(*this);
    view_.showAnswers.bind_to<FiltersPanel, &FiltersPanel::show_answers, &FiltersPanel::has_session>(*this);
    view_.showWeights.bind_to<FiltersPanel, &FiltersPanel::show_weights, &FiltersPanel::has_session>(*this);
    view_.showDream.bind_to<FiltersPanel, &FiltersPanel::show_dream, &FiltersPanel::has_session>(*this);
    view_.noiseLess.bind_to<FiltersPanel, &FiltersPanel::noise_less, &FiltersPanel::can_less_noise>(*this);
    view_.noiseMore.bind_to<FiltersPanel, &FiltersPanel::noise_more, &FiltersPanel::can_more_noise>(*this);
    view_.newPicture.bind_to<FiltersPanel, &FiltersPanel::new_picture, &FiltersPanel::has_session>(*this);
}

bool FiltersPanel::can_more_noise() const {
    return session_ != nullptr && noise_ + 1 < static_cast<int>(kNoiseLevels.size());
}

int FiltersPanel::timestep() const {
    return std::min(kNoiseLevels[static_cast<size_t>(noise_)], session_->probe().timesteps() - 1);
}

void FiltersPanel::sync_session() {
    generation_ = inspection_.generation();
    session_ = inspection_.session();
    rows_.clear();
    view_.filterRows.set({});
    answer_.reset();
    for (FilterPaint* paint : {&view_.inputSpritePaint, &view_.inputNoisyPaint, &view_.inputEstimatePaint,
                 &view_.inputNoisePaint, &view_.detailAnswerPaint, &view_.detailWeightsPaint, &view_.detailDreamPaint}) {
        paint->clear();
    }
    if (session_ == nullptr) {
        view_.filterLayerText = "";
        view_.filterNoteText = "";
        return;
    }
    noise_ = kDefaultNoise;
    mode_ = Mode::Answers;
    view_.modeAnswers = true;
    view_.modeWeights = false;
    view_.modeDream = false;
    std::random_device device;
    picture_ = device() % session_->sprite_count();
    seed_ = device();
    stage_ = 0;
    enter_stage(0);
}

void FiltersPanel::new_picture() {
    std::random_device device;
    picture_ = device() % session_->sprite_count();
    seed_ = device();
    answer();
    fill_cells();
    show_detail();
}

void FiltersPanel::set_mode(Mode mode) {
    mode_ = mode;
    view_.modeAnswers = mode == Mode::Answers;
    view_.modeWeights = mode == Mode::Weights;
    view_.modeDream = mode == Mode::Dream;
    if (mode == Mode::Dream) {
        session_->dreams().ask(stage(), timestep());
    }
    fill_cells();
    show_texts();
}

void FiltersPanel::set_noise(int noise) {
    noise_ = std::clamp(noise, 0, static_cast<int>(kNoiseLevels.size()) - 1);
    if (mode_ == Mode::Dream) {
        session_->dreams().ask(stage(), timestep());
    }
    answer();
    fill_cells();
    show_detail();
}

void FiltersPanel::enter_stage(int stage) {
    stage_ = std::clamp(stage, 0, cuda::kStageCount - 1);
    const cuda::StageInfo info = session_->probe().info(this->stage());
    units_ = info.channels;
    selected_ = 0;

    // The filters of the stage, each as a picture; a common scale keeps them comparable.
    int in = 0;
    const std::vector<float> weights = session_->probe().weights(this->stage(), in);
    const size_t per_unit = static_cast<size_t>(in) * 9;
    const float largest = inspect::largest_magnitude(weights.data(), weights.size());
    weight_images_.clear();
    for (int u = 0; u < units_; ++u) {
        weight_images_.push_back(inspect::filter_image(weights.data() + static_cast<size_t>(u) * per_unit, in, largest));
    }
    weight_strength_ = inspect::filter_lengths(weights.data(), units_, per_unit);
    inspect::scale_to_strongest(weight_strength_);

    constexpr int kCells = static_cast<int>(FilterRow::kCells);
    std::vector<std::shared_ptr<FilterRow>> rows;
    for (int first = 0; first < units_; first += kCells) {
        rows.push_back(std::make_shared<FilterRow>(static_cast<size_t>(std::min(units_ - first, kCells))));
    }
    rows_ = rows;
    for (int u = 0; u < units_; ++u) {
        rows_[static_cast<size_t>(u / kCells)]->pick(static_cast<size_t>(u % kCells)) = [this, u] { select(u); };
    }
    view_.filterRows.set(std::move(rows));

    if (mode_ == Mode::Dream) {
        session_->dreams().ask(this->stage(), timestep());
    }
    answer();
    fill_cells();
    select(0);
}

void FiltersPanel::answer() {
    cuda::Probe& probe = session_->probe();
    const std::vector<float>& dataset = session_->dataset();
    const std::vector<float> sprite(dataset.begin() + static_cast<std::ptrdiff_t>(picture_ * cuda::kSpriteValues),
            dataset.begin() + static_cast<std::ptrdiff_t>((picture_ + 1) * cuda::kSpriteValues));
    answer_ = probe.answer(sprite, timestep(), seed_);

    view_.inputSpritePaint.show(inspect::sprite_image(answer_->sprite.data()));
    view_.inputNoisyPaint.show(inspect::sprite_image(answer_->noisy.data()));
    view_.inputEstimatePaint.show(inspect::sprite_image(answer_->estimate.data()));
    view_.inputNoisePaint.show(inspect::sprite_image(answer_->predicted_noise.data(), 0.25f, 0.5f));

    // The strongest answer of the stage is white; each unit's meter is how much it answers on average.
    const cuda::StageInfo info = probe.info(stage());
    const std::vector<float>& maps = answer_->maps[static_cast<size_t>(stage_)];
    const size_t plane = static_cast<size_t>(info.size) * info.size;
    answer_scale_ = inspect::largest_magnitude(maps.data(), static_cast<size_t>(info.channels) * plane);
    answer_strength_ = inspect::mean_magnitudes(maps.data(), info.channels, plane);
    inspect::scale_to_strongest(answer_strength_);
    show_input_texts();
}

void FiltersPanel::show_input_texts() {
    view_.noiseText = tr_("filters.noise_level",
            {{"t", std::to_string(timestep())}, {"total", std::to_string(session_->probe().timesteps())}});
    view_.pictureText = tr_("filters.picture", {{"n", std::to_string(picture_ + 1)}});
}

inspect::CellImage FiltersPanel::answer_image(int unit) const {
    const cuda::StageInfo info = session_->probe().info(stage());
    const size_t plane = static_cast<size_t>(info.size) * info.size;
    return inspect::signed_image(answer_->maps[static_cast<size_t>(stage_)].data() + static_cast<size_t>(unit) * plane, info.size,
            answer_scale_);
}

inspect::CellImage FiltersPanel::dream_image(int unit) const {
    cuda::DreamWorker& dreams = session_->dreams();
    if (dreams.timestep() != timestep()) {
        return {};
    }
    const std::optional<std::vector<float>>& picture = dreams.picture(stage(), unit);
    return picture ? inspect::sprite_image(picture->data()) : inspect::CellImage{};
}

float FiltersPanel::meter(int unit) const {
    switch (mode_) {
    case Mode::Answers: return answer_strength_[static_cast<size_t>(unit)];
    case Mode::Weights: return weight_strength_[static_cast<size_t>(unit)];
    case Mode::Dream: break;
    }
    return -1.0f;
}

void FiltersPanel::fill_cell(int unit) {
    FilterPaint& paint = cell(unit);
    switch (mode_) {
    case Mode::Answers: paint.show(answer_image(unit)); break;
    case Mode::Weights: paint.show(weight_images_[static_cast<size_t>(unit)]); break;
    case Mode::Dream: paint.show(dream_image(unit)); break;
    }
    paint.set_meter(meter(unit));
    paint.set_selected(unit == selected_);
}

void FiltersPanel::fill_cells() {
    for (int u = 0; u < units_; ++u) {
        fill_cell(u);
    }
    show_texts();
}

void FiltersPanel::select(int unit) {
    cell(selected_).set_selected(false);
    selected_ = unit;
    cell(selected_).set_selected(true);
    show_detail();
}

void FiltersPanel::show_detail() {
    view_.detailAnswerPaint.show(answer_image(selected_));
    view_.detailWeightsPaint.show(weight_images_[static_cast<size_t>(selected_)]);
    view_.detailDreamPaint.show(dream_image(selected_));
    view_.filterTitleText = tr_("filters.channel", {{"n", std::to_string(selected_ + 1)}, {"total", std::to_string(units_)}});
    view_.filterDetailText = tr_("filters.channel_detail",
            {{"answer", std::format("{:.0f}", 100.0f * answer_strength_[static_cast<size_t>(selected_)])},
                    {"weights", std::format("{:.0f}", 100.0f * weight_strength_[static_cast<size_t>(selected_)])}});
}

void FiltersPanel::show_texts() {
    const cuda::StageInfo info = session_->probe().info(stage());
    view_.filterLayerText = tr_("filters.layer",
            {{"n", std::to_string(stage_ + 1)}, {"total", std::to_string(cuda::kStageCount)},
                    {"name", tr_("stage." + std::to_string(stage_))}, {"channels", std::to_string(info.channels)},
                    {"size", std::to_string(info.size)}});
    switch (mode_) {
    case Mode::Answers:
        view_.filterNoteText = tr_("filters.note_answers");
        view_.filterProgressText = "";
        break;
    case Mode::Weights:
        view_.filterNoteText = tr_("filters.note_weights");
        view_.filterProgressText = "";
        break;
    case Mode::Dream: {
        view_.filterNoteText = tr_("filters.note_dream");
        cuda::DreamWorker& dreams = session_->dreams();
        const std::string error = dreams.error();
        view_.filterProgressText = !error.empty() ? tr_("filters.dream_failed", {{"detail", error}})
                : dreams.timestep() != timestep()  ? ""
                                                    : tr_("filters.dream_progress", {{"done", std::to_string(dreams.done(stage()))},
                                                                  {"total", std::to_string(dreams.total(stage()))}});
        break;
    }
    }
}

void FiltersPanel::apply_locale() {
    if (session_ == nullptr || !answer_) {
        return;
    }
    show_texts();
    show_detail();
    show_input_texts();
}

void FiltersPanel::update() {
    if (generation_ != inspection_.generation()) {
        sync_session();
    }
    if (session_ == nullptr) {
        return;
    }
    if (session_->dreams().collect()) {
        if (mode_ == Mode::Dream) {
            fill_cells();
        }
        show_detail();
    } else if (mode_ == Mode::Dream) {
        show_texts();
    }
}

}
