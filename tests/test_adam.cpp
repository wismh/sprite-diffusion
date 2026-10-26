#include "cuda/adam.h"

#include "test_support.h"

TEST(Adam, MatchesTheCpuUpdateOverSeveralSteps) {
    const cuda::AdamConfig config{.learning_rate = 0.01f};
    cuda::Adam adam(config);
    cuda::Param param;
    param.init({1, 5, 1, 1});
    std::vector<double> value(5, 0.0);
    std::vector<double> m(5, 0.0);
    std::vector<double> v(5, 0.0);
    const std::vector<float> start = test::random_values(5, 60);
    param.value.upload(start);
    for (size_t i = 0; i < 5; ++i) {
        value[i] = start[i];
    }

    for (int t = 1; t <= 4; ++t) {
        const std::vector<float> grad = test::random_values(5, 70 + t);
        param.grad.upload(grad);
        adam.step({&param});
        for (size_t i = 0; i < 5; ++i) {
            m[i] = config.beta1 * m[i] + (1.0 - config.beta1) * grad[i];
            v[i] = config.beta2 * v[i] + (1.0 - config.beta2) * grad[i] * grad[i];
            const double mh = m[i] / (1.0 - std::pow(config.beta1, t));
            const double vh = v[i] / (1.0 - std::pow(config.beta2, t));
            value[i] -= config.learning_rate * mh / (std::sqrt(vh) + config.eps);
        }
    }
    EXPECT_EQ(adam.steps(), 4);
    const std::vector<float> actual = param.value.download();
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_NEAR(actual[i], value[i], 1e-5);
    }
}
