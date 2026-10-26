#pragma once

#include "inspect/nearest_sprite.h"

#include <vector>

namespace inspect {

// `count` of `steps` sampling steps, spread evenly from the first to the last (the same step is never picked twice
// while count <= steps).
[[nodiscard]] std::vector<int> shown_steps(int steps, int count);

// The grid of the Process tab: a row per sample with the sprites of the `shown` steps, then the sprite of `dataset` that
// the sample is closest to. `steps[step]` holds all the samples of that step, one after another (NCHW, [-1, 1]).
// `neighbors[sample]` is the sprite to show at the end of the row.
[[nodiscard]] std::vector<float> process_grid(const std::vector<std::vector<float>>& steps, const std::vector<int>& shown,
        const std::vector<float>& dataset, const std::vector<Neighbor>& neighbors);

}
