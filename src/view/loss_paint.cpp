#include "view/loss_paint.h"

#include "inspect/loss_chart.h"

#include <engine/builtin_ids.h>

#include <algorithm>
#include <cmath>
#include <format>

namespace game {

namespace {

constexpr float kMargin = 8.0f;
constexpr float kLabelWidth = 46.0f;

}

void LossPaint::paint(engine::ui::IDrawList& list, const engine::render::Rect& content) {
    list.fill_rect(content, {1.0f, 1.0f, 1.0f, 1.0f});
    list.stroke_rect(content, {0.89f, 0.91f, 0.94f, 1.0f}, 1.0f);
    list.set_font(engine::builtin::font_ui, 12.0f);
    if (history_.size() < 2) {
        list.text(empty_text_, {content.x + kMargin, content.y + kMargin}, {0.58f, 0.64f, 0.72f, 1.0f});
        return;
    }

    const engine::render::Rect plot{content.x + kLabelWidth, content.y + kMargin,
            content.w - kLabelWidth - kMargin, content.h - 2.0f * kMargin};
    // One point per two pixels at most: longer histories are averaged in buckets.
    const std::vector<float> values = inspect::bucket_means(history_, static_cast<std::size_t>(plot.w / 2.0f));
    const std::size_t points = values.size();

    const auto [lowest, highest] = std::minmax_element(values.begin(), values.end());
    const inspect::LogAxis axis(*lowest, *highest);
    const auto y_of = [&](float value) { return plot.y + plot.h * (1.0f - axis.position(value)); };

    for (int i = 0; i <= 4; ++i) {
        const float y = plot.y + plot.h * static_cast<float>(i) / 4.0f;
        list.line({plot.x, y}, {plot.x + plot.w, y}, {0.93f, 0.95f, 0.97f, 1.0f}, 1.0f);
        const float value = axis.value_at(1.0f - static_cast<float>(i) / 4.0f);
        list.text(std::format("{:.3f}", value), {content.x + 4.0f, y - 7.0f}, {0.39f, 0.45f, 0.55f, 1.0f});
    }
    for (std::size_t i = 1; i < points; ++i) {
        const float x0 = plot.x + plot.w * static_cast<float>(i - 1) / static_cast<float>(points - 1);
        const float x1 = plot.x + plot.w * static_cast<float>(i) / static_cast<float>(points - 1);
        list.line({x0, y_of(values[i - 1])}, {x1, y_of(values[i])}, {0.15f, 0.39f, 0.92f, 1.0f}, 2.0f);
    }
}

void LossPaint::show(std::vector<float> history) {
    if (history.size() != history_.size()) {
        history_ = std::move(history);
        ++version_;
    }
}

void LossPaint::reset() {
    history_.clear();
    ++version_;
}

}
