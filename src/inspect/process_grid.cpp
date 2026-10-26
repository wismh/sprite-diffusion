#include "inspect/process_grid.h"

#include "data/sprite_set.h"

#include <cmath>

namespace inspect {

std::vector<int> shown_steps(int steps, int count) {
    std::vector<int> shown;
    if (steps <= 0 || count <= 0) {
        return shown;
    }
    for (int i = 0; i < count; ++i) {
        shown.push_back(count == 1 ? 0 : static_cast<int>(std::lround(static_cast<double>(i) * (steps - 1) / (count - 1))));
    }
    return shown;
}

std::vector<float> process_grid(const std::vector<std::vector<float>>& steps, const std::vector<int>& shown,
        const std::vector<float>& dataset, const std::vector<Neighbor>& neighbors) {
    constexpr std::size_t kValues = data::kSpriteValues;
    std::vector<float> grid;
    grid.reserve(neighbors.size() * (shown.size() + 1) * kValues);
    for (std::size_t sample = 0; sample < neighbors.size(); ++sample) {
        for (const int step : shown) {
            const auto first = steps[static_cast<std::size_t>(step)].begin() + static_cast<std::ptrdiff_t>(sample * kValues);
            grid.insert(grid.end(), first, first + static_cast<std::ptrdiff_t>(kValues));
        }
        const Neighbor& neighbor = neighbors[sample];
        const std::vector<float> closest = dataset_sprite(dataset, neighbor.index, neighbor.mirrored);
        grid.insert(grid.end(), closest.begin(), closest.end());
    }
    return grid;
}

}
