#pragma once

#include <cstddef>
#include <vector>

namespace inspect {

// Divides every value by the largest, so the strongest is 1. All zeros stay zeros.
void scale_to_strongest(std::vector<float>& values);

// `maps` is `channels` maps of `plane` values, a channel after a channel. Returns the mean of |value| of each map.
[[nodiscard]] std::vector<float> mean_magnitudes(const float* maps, int channels, std::size_t plane);

// The largest |value| of `count` values (0 for none).
[[nodiscard]] float largest_magnitude(const float* values, std::size_t count);

// `weights` is `units` filters of `per_unit` weights each. Returns the length of each filter: the root of the sum of the
// squares of its weights.
[[nodiscard]] std::vector<float> filter_lengths(const float* weights, int units, std::size_t per_unit);

}
