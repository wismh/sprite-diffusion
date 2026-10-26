#include "view/filter_row.h"

#include <format>

namespace game {

namespace {

// "cell0".."cell7" and "pick0".."pick7", hashed once.
struct ColumnIds {
    std::array<engine::ui::BindingId, FilterRow::kCells> cells;
    std::array<engine::ui::BindingId, FilterRow::kCells> picks;

    ColumnIds() {
        for (std::size_t column = 0; column < FilterRow::kCells; ++column) {
            cells[column] = engine::ui::intern(std::format("cell{}", column));
            picks[column] = engine::ui::intern(std::format("pick{}", column));
        }
    }
};

const ColumnIds& column_ids() {
    static const ColumnIds ids;
    return ids;
}

}

FilterRow::FilterRow(std::size_t used) : used_(used) {
    for (std::size_t column = 0; column < kCells; ++column) {
        picks_[column].set_can_execute(column < used);
        if (column >= used) {
            cells_[column].vacate();
        }
        paint(column_ids().cells[column], cells_[column]);
        command(column_ids().picks[column], picks_[column]);
    }
}

}
