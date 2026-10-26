#include "cuda/group_norm.h"

#include "test_support.h"

TEST(GroupNorm, ForwardMatchesTheCpuFormula) {
    const int groups = 2;
    const cuda::Shape s{2, 4, 3, 3};
    cuda::GroupNorm norm(groups, s.c);
    const std::vector<float> x = test::random_values(s.count(), 30, 2.0f);
    const std::vector<float> gamma = test::random_values(s.c, 31);
    const std::vector<float> beta = test::random_values(s.c, 32);
    norm.gamma().value.upload(gamma);
    norm.beta().value.upload(beta);
    const cuda::Tensor xt = test::make_tensor(s, x);
    const std::vector<float> y = norm.forward(xt).download();

    const int hw = s.h * s.w;
    const int cpg = s.c / groups;
    for (int n = 0; n < s.n; ++n) {
        for (int g = 0; g < groups; ++g) {
            const size_t base = static_cast<size_t>(n * s.c + g * cpg) * hw;
            const int count = cpg * hw;
            double mean = 0.0;
            for (int i = 0; i < count; ++i) {
                mean += x[base + i];
            }
            mean /= count;
            double var = 0.0;
            for (int i = 0; i < count; ++i) {
                var += (x[base + i] - mean) * (x[base + i] - mean);
            }
            var /= count;
            for (int i = 0; i < count; ++i) {
                const int c = g * cpg + i / hw;
                const double expected = (x[base + i] - mean) / std::sqrt(var + 1e-5) * gamma[c] + beta[c];
                EXPECT_NEAR(y[base + i], expected, 1e-4);
            }
        }
    }
}

TEST(GroupNorm, GradientsMatchFiniteDifferences) {
    const cuda::Shape s{2, 4, 3, 3};
    cuda::GroupNorm norm(2, s.c);
    std::vector<float> x = test::random_values(s.count(), 40, 2.0f);
    std::vector<float> gamma = test::random_values(s.c, 41);
    std::vector<float> beta = test::random_values(s.c, 42);
    norm.gamma().value.upload(gamma);
    norm.beta().value.upload(beta);
    cuda::Tensor xt = test::make_tensor(s, x);

    const std::vector<float> r = test::random_values(s.count(), 43);
    const cuda::Tensor dy = test::make_tensor(s, r);
    const auto loss = [&] { return test::dot(norm.forward(xt).download(), r); };

    norm.forward(xt);
    for (cuda::Param* param : norm.params()) {
        param->zero_grad();
    }
    const std::vector<float> dx = norm.backward(xt, dy).download();
    const std::vector<float> dgamma = norm.gamma().grad.download();
    const std::vector<float> dbeta = norm.beta().grad.download();

    test::expect_gradient(x, dx, [&](const std::vector<float>& v) { xt.upload(v); }, loss);
    test::expect_gradient(gamma, dgamma, [&](const std::vector<float>& v) { norm.gamma().value.upload(v); }, loss);
    test::expect_gradient(beta, dbeta, [&](const std::vector<float>& v) { norm.beta().value.upload(v); }, loss);
}
