#include "inspect/channel_stats.h"

#include <algorithm>
#include <cmath>

namespace inspect {

void scale_to_strongest(std::vector<float>& values) {
    const float strongest = values.empty() ? 0.0f : *std::max_element(values.begin(), values.end());
    for (float& value : values) {
        value = strongest > 0.0f ? value / strongest : 0.0f;
    }
}

std::vector<float> mean_magnitudes(const float* maps, int channels, std::size_t plane) {
    std::vector<float> means(static_cast<std::size_t>(channels), 0.0f);
    for (std::size_t unit = 0; unit < means.size(); ++unit) {
        double sum = 0.0;
        for (std::size_t k = 0; k < plane; ++k) {
            sum += std::abs(maps[unit * plane + k]);
        }
        means[unit] = static_cast<float>(sum / static_cast<double>(plane));
    }
    return means;
}

float largest_magnitude(const float* values, std::size_t count) {
    float largest = 0.0f;
    for (std::size_t i = 0; i < count; ++i) {
        largest = std::max(largest, std::abs(values[i]));
    }
    return largest;
}

std::vector<float> filter_lengths(const float* weights, int units, std::size_t per_unit) {
    std::vector<float> lengths(static_cast<std::size_t>(units), 0.0f);
    for (std::size_t unit = 0; unit < lengths.size(); ++unit) {
        double sum = 0.0;
        for (std::size_t k = 0; k < per_unit; ++k) {
            const double w = weights[unit * per_unit + k];
            sum += w * w;
        }
        lengths[unit] = static_cast<float>(std::sqrt(sum));
    }
    return lengths;
}

}
