#pragma once

#include <vector>

namespace cuda {

// The cosine schedule of Nichol and Dhariwal: alpha_bar(t), the share of the signal left after t noising steps,
// follows cos^2 of the progress, so the image is destroyed evenly instead of in the first few steps.
class NoiseSchedule {
public:
    explicit NoiseSchedule(int steps, double offset = 0.008);

    [[nodiscard]] int steps() const { return static_cast<int>(alpha_bar_.size()); }
    // For t in [0, steps): the product of (1 - beta_s) for s <= t. Close to 1 at t = 0, close to 0 at the last step.
    [[nodiscard]] double alpha_bar(int t) const { return alpha_bar_[t]; }
    [[nodiscard]] double beta(int t) const { return beta_[t]; }

private:
    std::vector<double> alpha_bar_;
    std::vector<double> beta_;
};

}
