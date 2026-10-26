#pragma once

#include "inspect/cell_image.h"

#include <engine/ui/paint.h>

#include <cstdint>

namespace game {

// One inspect::CellImage drawn into whatever element it is bound to, pixel by pixel: a cell of the filter grid, or one of the
// larger pictures. A cell can carry a bar at its foot (how strong the unit is) and a frame when it is picked. Without an
// image it is an empty frame.
class FilterPaint final : public engine::ui::IPaint {
public:
    void paint(engine::ui::IDrawList& list, const engine::render::Rect& content) override;
    [[nodiscard]] std::uint64_t version() const noexcept override { return version_; }

    void show(inspect::CellImage image);
    void clear();
    // Paints nothing at all: a cell at the end of a row that has no unit.
    void vacate();
    void set_selected(bool selected);
    // 0..1, or negative for no bar. Quantized, so that a small change does not redraw the picture.
    void set_meter(float meter);

private:
    inspect::CellImage image_;
    bool selected_ = false;
    bool vacant_ = false;
    int meter_steps_ = -1;
    std::uint64_t version_ = 0;
};

}
