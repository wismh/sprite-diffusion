#pragma once

#include <cstddef>
#include <vector>

namespace inspect {

// `history` cut into `points` buckets of about equal length, each the mean of its values: a long run fits a chart that
// has a point per two pixels. A history shorter than `points` is returned as it is.
[[nodiscard]] std::vector<float> bucket_means(const std::vector<float>& history, std::size_t points);

// A logarithmic vertical axis over the values of a chart. The loss falls by orders of magnitude, so a linear axis would
// flatten everything after the first minute. The range is a little wider than the data and never reaches zero.
class LogAxis {
public:
    LogAxis(float lowest, float highest);

    // 0 at the bottom, 1 at the top; values outside the range are cut to it.
    [[nodiscard]] float position(float value) const;
    // The value at a position from 0 (bottom) to 1 (top).
    [[nodiscard]] float value_at(float position) const;

private:
    float low_;
    float high_;
};

}
