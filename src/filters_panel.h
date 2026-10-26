#pragma once

#include "cuda/probe.h"
#include "inspection.h"
#include "loc/localized.h"
#include "view/app_view_model.h"
#include "inspect/cell_image.h"
#include "view/filter_row.h"

#include <memory>
#include <optional>
#include <vector>

namespace game {

// The Filters tab: what each channel of the network looks for and what it answers. One stage of the network at a time is a
// grid of pictures, one per channel, of one of three kinds: the channel's answer to a sprite at some noise level, its
// weights (the 3x3 filters it applies to the channels before it), or its dream, the sprite it answers to hardest (found in
// the background). Picking a channel shows all three of it, larger.
class FiltersPanel final : public Localized {
public:
    FiltersPanel(AppViewModel& view, Inspection& inspection, const Tr& tr);

    // Once a frame on the main thread: notices a newly loaded model and takes in finished dreams.
    void update();
    void apply_locale() override;

private:
    enum class Mode { Answers, Weights, Dream };

    [[nodiscard]] bool can_layer_up() const { return session_ != nullptr && stage_ + 1 < cuda::kStageCount; }
    [[nodiscard]] bool can_layer_down() const { return session_ != nullptr && stage_ > 0; }
    [[nodiscard]] bool has_session() const { return session_ != nullptr; }
    [[nodiscard]] bool can_less_noise() const { return session_ != nullptr && noise_ > 0; }
    [[nodiscard]] bool can_more_noise() const;

    void layer_up() { enter_stage(stage_ + 1); }
    void layer_down() { enter_stage(stage_ - 1); }
    void show_answers() { set_mode(Mode::Answers); }
    void show_weights() { set_mode(Mode::Weights); }
    void show_dream() { set_mode(Mode::Dream); }
    void noise_less() { set_noise(noise_ - 1); }
    void noise_more() { set_noise(noise_ + 1); }
    void new_picture();

    void sync_session();
    void set_mode(Mode mode);
    void set_noise(int noise);
    void enter_stage(int stage);
    void answer();
    void fill_cells();
    void fill_cell(int unit);
    void select(int unit);
    void show_detail();
    void show_texts();
    void show_input_texts();
    [[nodiscard]] int timestep() const;
    [[nodiscard]] cuda::Stage stage() const { return static_cast<cuda::Stage>(stage_); }
    [[nodiscard]] FilterPaint& cell(int unit) { return rows_[static_cast<size_t>(unit / 8)]->cell(static_cast<size_t>(unit % 8)); }
    [[nodiscard]] inspect::CellImage answer_image(int unit) const;
    [[nodiscard]] inspect::CellImage dream_image(int unit) const;
    [[nodiscard]] float meter(int unit) const;

    AppViewModel& view_;
    Inspection& inspection_;
    const Tr& tr_;
    InspectSession* session_ = nullptr;
    unsigned generation_ = 0;

    int stage_ = 0;
    int units_ = 0;
    int selected_ = 0;
    int noise_ = 0;  // an index into the noise levels
    Mode mode_ = Mode::Answers;

    size_t picture_ = 0;
    unsigned seed_ = 1;
    std::optional<cuda::Answer> answer_;
    float answer_scale_ = 0.0f;
    std::vector<float> answer_strength_;
    // The filters of the stage on show, as pictures, and how strong each is (relative to the strongest).
    std::vector<inspect::CellImage> weight_images_;
    std::vector<float> weight_strength_;
    std::vector<std::shared_ptr<FilterRow>> rows_;
};

}
