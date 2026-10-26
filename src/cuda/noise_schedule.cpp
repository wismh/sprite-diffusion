#include "cuda/noise_schedule.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace cuda {

NoiseSchedule::NoiseSchedule(int steps, double offset) {
    if (steps < 2) {
        throw std::invalid_argument("NoiseSchedule: at least two steps");
    }
    const auto f = [&](double progress) {
        const double angle = (progress + offset) / (1.0 + offset) * std::numbers::pi / 2.0;
        return std::cos(angle) * std::cos(angle);
    };
    const double f0 = f(0.0);
    alpha_bar_.resize(steps);
    beta_.resize(steps);
    double previous = 1.0;
    for (int t = 0; t < steps; ++t) {
        // beta is capped so that the last step does not destroy the signal completely (division by zero at sampling).
        beta_[t] = std::min(1.0 - f(static_cast<double>(t + 1) / steps) / f0 / previous, 0.999);
        alpha_bar_[t] = previous * (1.0 - beta_[t]);
        previous = alpha_bar_[t];
    }
}

}
