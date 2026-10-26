#pragma once

#include <engine/ui/paint.h>

#include <cstdint>
#include <vector>

namespace game {

// 16x16 RGB sprites ([-1, 1], NCHW) in a grid of equal cells, drawn pixel by pixel. Without sprites it shows an empty
// frame. The paint keeps its own copy.
class SpriteGridPaint final : public engine::ui::IPaint {
public:
    void paint(engine::ui::IDrawList& list, const engine::render::Rect& content) override;
    [[nodiscard]] std::uint64_t version() const noexcept override { return version_; }

    void show(std::vector<float> sprites, int columns);

private:
    std::vector<float> sprites_;
    int columns_ = 1;
    std::uint64_t version_ = 0;
};

}
