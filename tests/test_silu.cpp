#include "cuda/silu.h"

#include "test_support.h"

TEST(Silu, ForwardMatchesTheFormula) {
    const cuda::Shape s{1, 2, 3, 3};
    cuda::Silu silu;
    const std::vector<float> x = test::random_values(s.count(), 50, 4.0f);
    const cuda::Tensor xt = test::make_tensor(s, x);
    const std::vector<float> y = silu.forward(xt).download();
    for (size_t i = 0; i < x.size(); ++i) {
        EXPECT_NEAR(y[i], x[i] / (1.0 + std::exp(-static_cast<double>(x[i]))), 1e-5);
    }
}

TEST(Silu, GradientMatchesFiniteDifferences) {
    const cuda::Shape s{1, 2, 3, 3};
    cuda::Silu silu;
    std::vector<float> x = test::random_values(s.count(), 51, 3.0f);
    cuda::Tensor xt = test::make_tensor(s, x);
    const std::vector<float> r = test::random_values(s.count(), 52);
    const cuda::Tensor dy = test::make_tensor(s, r);
    const auto loss = [&] { return test::dot(silu.forward(xt).download(), r); };

    silu.forward(xt);
    const std::vector<float> dx = silu.backward(xt, dy).download();
    test::expect_gradient(x, dx, [&](const std::vector<float>& v) { xt.upload(v); }, loss);
}
