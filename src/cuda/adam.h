#pragma once

#include "cuda/tensor.h"

#include <vector>

namespace cuda {

struct AdamConfig {
    float learning_rate = 1e-3f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1e-8f;
};

class Adam {
public:
    explicit Adam(AdamConfig config = {}) : config_(config) {}

    void set_learning_rate(float rate) { config_.learning_rate = rate; }
    [[nodiscard]] int steps() const { return step_; }
    void set_steps(int steps) { step_ = steps; }

    // One update of every parameter from its gradient. The moments are created on the first step.
    void step(const std::vector<Param*>& params);

private:
    AdamConfig config_;
    int step_ = 0;
};

}
