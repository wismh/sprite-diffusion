#pragma once

#include "view/filter_paint.h"

#include <engine/ui/view_model.h>

#include <array>
#include <cstddef>

namespace game {

// One row of the filter grid, kCells units wide. The grid is an ItemsControl of these. The cells after the last unit of the
// stage stay empty and cannot be picked.
class FilterRow final : public engine::ui::ViewModel {
public:
    static constexpr std::size_t kCells = 8;

    // `used` cells from the left take part.
    explicit FilterRow(std::size_t used);

    [[nodiscard]] FilterPaint& cell(std::size_t column) { return cells_[column]; }
    [[nodiscard]] engine::ui::RelayCommand& pick(std::size_t column) { return picks_[column]; }
    [[nodiscard]] std::size_t used() const { return used_; }

private:
    std::size_t used_;
    std::array<FilterPaint, kCells> cells_;
    std::array<engine::ui::RelayCommand, kCells> picks_;
};

}
