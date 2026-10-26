#include "cuda/gemm.h"

#include "test_support.h"

namespace {

struct Size {
    int m;
    int n;
    int k;
};

class GemmTest : public testing::TestWithParam<Size> {};

TEST_P(GemmTest, MatchesTheCpuProductForEveryTransposition) {
    const auto [m, n, k] = GetParam();
    const std::vector<float> a = test::random_values(static_cast<size_t>(m) * k, 1);
    const std::vector<float> b = test::random_values(static_cast<size_t>(k) * n, 2);
    const std::vector<float> c0 = test::random_values(static_cast<size_t>(m) * n, 3);
    for (const bool ta : {false, true}) {
        for (const bool tb : {false, true}) {
            cuda::Tensor da = test::make_tensor({m * k, 1, 1, 1}, a);
            cuda::Tensor db = test::make_tensor({k * n, 1, 1, 1}, b);
            cuda::Tensor dc = test::make_tensor({m * n, 1, 1, 1}, c0);
            cuda::gemm(ta, tb, m, n, k, 2.0f, da.data(), db.data(), 0.5f, dc.data());
            const std::vector<float> c = dc.download();
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < n; ++j) {
                    double sum = 0.0;
                    for (int p = 0; p < k; ++p) {
                        const float av = ta ? a[p * m + i] : a[i * k + p];
                        const float bv = tb ? b[j * k + p] : b[p * n + j];
                        sum += static_cast<double>(av) * bv;
                    }
                    ASSERT_NEAR(c[i * n + j], 2.0 * sum + 0.5 * c0[i * n + j], 1e-3) << ta << tb << " " << i << "," << j;
                }
            }
        }
    }
}

// Tiles of 64 x 64 x 16: sizes below, on and across the edges, and the thin shapes of the dense layers.
INSTANTIATE_TEST_SUITE_P(Sizes, GemmTest,
                         testing::Values(Size{3, 4, 5}, Size{64, 64, 16}, Size{70, 33, 45}, Size{130, 17, 200},
                                         Size{1, 1, 1}, Size{256, 1, 256}, Size{5, 100, 3}));

}

TEST(Gemm, BetaZeroIgnoresWhatIsInC) {
    const std::vector<float> a = test::random_values(6, 1);
    const std::vector<float> b = test::random_values(6, 2);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    cuda::Tensor da = test::make_tensor({6, 1, 1, 1}, a);
    cuda::Tensor db = test::make_tensor({6, 1, 1, 1}, b);
    cuda::Tensor dc = test::make_tensor({4, 1, 1, 1}, std::vector<float>(4, nan));
    cuda::gemm(false, false, 2, 2, 3, 1.0f, da.data(), db.data(), 0.0f, dc.data());
    for (const float v : dc.download()) {
        EXPECT_TRUE(std::isfinite(v));
    }
}
