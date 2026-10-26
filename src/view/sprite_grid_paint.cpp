#include "view/sprite_grid_paint.h"

#include "data/sprite_set.h"

#include <algorithm>

namespace game {

namespace {

constexpr int kSide = 16;

float channel(float value) {
    return std::clamp((value + 1.0f) * 0.5f, 0.0f, 1.0f);
}

}

void SpriteGridPaint::paint(engine::ui::IDrawList& list, const engine::render::Rect& content) {
    list.fill_rect(content, {0.12f, 0.16f, 0.23f, 1.0f});
    const std::size_t count = sprites_.size() / data::kSpriteValues;
    const float cell = content.w / static_cast<float>(columns_);
    const float pixel = (cell - 4.0f) / kSide;
    for (std::size_t i = 0; i < count; ++i) {
        const float left = content.x + cell * static_cast<float>(i % columns_) + 2.0f;
        const float top = content.y + cell * static_cast<float>(i / columns_) + 2.0f;
        const float* sprite = sprites_.data() + i * data::kSpriteValues;
        for (int y = 0; y < kSide; ++y) {
            for (int x = 0; x < kSide; ++x) {
                const int at = y * kSide + x;
                list.fill_rect({left + static_cast<float>(x) * pixel, top + static_cast<float>(y) * pixel, pixel + 0.5f,
                                       pixel + 0.5f},
                        {channel(sprite[at]), channel(sprite[kSide * kSide + at]), channel(sprite[2 * kSide * kSide + at]),
                                1.0f});
            }
        }
    }
}

void SpriteGridPaint::show(std::vector<float> sprites, int columns) {
    sprites_ = std::move(sprites);
    columns_ = std::max(columns, 1);
    ++version_;
}

}
