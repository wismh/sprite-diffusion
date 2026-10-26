#include "inspect/loss_chart.h"

#include <algorithm>
#include <cmath>

namespace inspect {

namespace {

constexpr float kFloor = 1e-5f;

}

std::vector<float> bucket_means(const std::vector<float>& history, std::size_t points) {
    points = std::min(history.size(), points);
    std::vector<float> values(points);
    for (std::size_t i = 0; i < points; ++i) {
        const std::size_t begin = i * history.size() / points;
        const std::size_t end = std::max(begin + 1, (i + 1) * history.size() / points);
        float sum = 0.0f;
        for (std::size_t k = begin; k < end; ++k) {
            sum += history[k];
        }
        values[i] = sum / static_cast<float>(end - begin);
    }
    return values;
}

LogAxis::LogAxis(float lowest, float highest)
    : low_(std::log(std::max(lowest * 0.9f, kFloor)))
    , high_(std::log(std::max(highest * 1.1f, 2.0f * kFloor))) {}

float LogAxis::position(float value) const {
    const float t = (std::log(std::max(value, kFloor)) - low_) / std::max(high_ - low_, 1e-6f);
    return std::clamp(t, 0.0f, 1.0f);
}

float LogAxis::value_at(float position) const {
    return std::exp(low_ + (high_ - low_) * position);
}

}
