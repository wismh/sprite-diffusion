#pragma once

#include "cuda/tensor.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <functional>
#include <random>
#include <thread>
#include <vector>

namespace test {

inline std::vector<float> random_values(size_t count, unsigned seed, float scale = 1.0f) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-scale, scale);
    std::vector<float> values(count);
    for (float& v : values) {
        v = dist(rng);
    }
    return values;
}

inline cuda::Tensor make_tensor(cuda::Shape shape, const std::vector<float>& values) {
    cuda::Tensor tensor(shape);
    tensor.upload(values);
    return tensor;
}

inline double dot(const std::vector<float>& a, const std::vector<float>& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        sum += static_cast<double>(a[i]) * b[i];
    }
    return sum;
}

// Central finite differences of loss() with respect to every element of values, against the analytic gradient.
// apply() writes values to wherever loss() reads them from (an input tensor or a parameter).
inline void expect_gradient(std::vector<float> values, const std::vector<float>& analytic,
                            const std::function<void(const std::vector<float>&)>& apply,
                            const std::function<double()>& loss, float eps = 1e-2f) {
    ASSERT_EQ(values.size(), analytic.size());
    for (size_t i = 0; i < values.size(); ++i) {
        const float original = values[i];
        values[i] = original + eps;
        apply(values);
        const double plus = loss();
        values[i] = original - eps;
        apply(values);
        const double minus = loss();
        values[i] = original;
        apply(values);
        const double numeric = (plus - minus) / (2.0 * eps);
        EXPECT_NEAR(numeric, analytic[i], 3e-3 + 1e-2 * std::abs(analytic[i])) << "element " << i;
    }
}

// expect_gradient for the elements of every parameter (max_per_param > 0: for that many elements per parameter, picked
// at random). Their gradients must already be computed.
inline void expect_param_gradients(const std::vector<cuda::Param*>& params, const std::function<double()>& loss,
                                   size_t max_per_param = 0, float eps = 1e-2f, float rel = 1e-2f) {
    std::mt19937 rng(99);
    for (size_t p = 0; p < params.size(); ++p) {
        cuda::Param* param = params[p];
        std::vector<float> values = param->value.download();
        const std::vector<float> analytic = param->grad.download();
        std::vector<size_t> picked(values.size());
        for (size_t i = 0; i < picked.size(); ++i) {
            picked[i] = i;
        }
        if (max_per_param > 0 && max_per_param < picked.size()) {
            std::shuffle(picked.begin(), picked.end(), rng);
            picked.resize(max_per_param);
        }
        for (const size_t i : picked) {
            const float original = values[i];
            values[i] = original + eps;
            param->value.upload(values);
            const double plus = loss();
            values[i] = original - eps;
            param->value.upload(values);
            const double minus = loss();
            values[i] = original;
            param->value.upload(values);
            const double numeric = (plus - minus) / (2.0 * eps);
            EXPECT_NEAR(numeric, analytic[i], 3e-3 + rel * std::abs(analytic[i])) << "param " << p << " element " << i;
        }
    }
}

// An empty directory of the tests, made new: <temp>/diffusion_tests/<name>.
inline std::filesystem::path fresh_directory(const char* name) {
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "diffusion_tests" / name;
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    return dir;
}

// Waits, up to `seconds`, until done() is true (for something that happens on another thread). Returns whether it did.
template <class Predicate>
bool wait_for(Predicate done, int seconds = 60) {
    const auto limit = std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
    while (std::chrono::steady_clock::now() < limit) {
        if (done()) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    return done();
}
}
