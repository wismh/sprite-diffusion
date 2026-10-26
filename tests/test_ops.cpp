#include "cuda/ops.h"
#include "cuda/upsample.h"

#include "test_support.h"

TEST(Ops, ConcatAndSplitAreInverse) {
    const cuda::Shape sa{2, 3, 2, 2};
    const cuda::Shape sb{2, 2, 2, 2};
    const std::vector<float> a = test::random_values(sa.count(), 1);
    const std::vector<float> b = test::random_values(sb.count(), 2);
    cuda::Tensor out;
    cuda::concat_channels(test::make_tensor(sa, a), test::make_tensor(sb, b), out);
    ASSERT_EQ(out.shape(), (cuda::Shape{2, 5, 2, 2}));
    const std::vector<float> joined = out.download();
    // Sample 1, channel 3 is channel 0 of b.
    EXPECT_FLOAT_EQ(joined[((1 * 5 + 3) * 2 + 1) * 2 + 0], b[((1 * 2 + 0) * 2 + 1) * 2 + 0]);

    cuda::Tensor da;
    cuda::Tensor db;
    cuda::split_channels(out, 3, da, db);
    EXPECT_EQ(da.download(), a);
    EXPECT_EQ(db.download(), b);
}

TEST(Ops, BroadcastAddAndSumHwAreTransposes) {
    const cuda::Shape s{2, 3, 2, 2};
    const std::vector<float> x = test::random_values(s.count(), 3);
    const std::vector<float> v = test::random_values(6, 4);
    cuda::Tensor out;
    cuda::broadcast_add(test::make_tensor(s, x), test::make_tensor({2, 3, 1, 1}, v), out);
    const std::vector<float> y = out.download();
    for (size_t i = 0; i < y.size(); ++i) {
        EXPECT_FLOAT_EQ(y[i], x[i] + v[i / 4]);
    }
    cuda::Tensor sums;
    cuda::sum_hw(test::make_tensor(s, x), sums);
    const std::vector<float> got = sums.download();
    for (int nc = 0; nc < 6; ++nc) {
        EXPECT_NEAR(got[nc], x[nc * 4] + x[nc * 4 + 1] + x[nc * 4 + 2] + x[nc * 4 + 3], 1e-5);
    }
}

TEST(Ops, MseLossAndGradient) {
    const cuda::Shape s{2, 1, 3, 3};
    const std::vector<float> pred = test::random_values(s.count(), 5);
    const std::vector<float> target = test::random_values(s.count(), 6);
    cuda::Tensor grad;
    const double loss = cuda::mse_loss(test::make_tensor(s, pred), test::make_tensor(s, target), grad);
    double expected = 0.0;
    const std::vector<float> g = grad.download();
    for (size_t i = 0; i < pred.size(); ++i) {
        expected += (pred[i] - target[i]) * (pred[i] - target[i]);
        EXPECT_NEAR(g[i], 2.0 * (pred[i] - target[i]) / pred.size(), 1e-6);
    }
    EXPECT_NEAR(loss, expected / pred.size(), 1e-6);
}

TEST(Ops, EmaMovesTowardTheSource) {
    cuda::Param ema;
    cuda::Param source;
    ema.init({1, 4, 1, 1});
    source.init({1, 4, 1, 1});
    ema.value.upload({0, 0, 0, 0});
    source.value.upload({1, 2, 3, 4});
    cuda::ema_update({&ema}, {&source}, 0.75f);
    EXPECT_EQ(ema.value.download(), (std::vector<float>{0.25f, 0.5f, 0.75f, 1.0f}));
    cuda::ema_update({&ema}, {&source}, 0.0f);
    EXPECT_EQ(ema.value.download(), source.value.download());
}

TEST(Upsample, ForwardRepeatsPixelsAndBackwardSumsBlocks) {
    const cuda::Shape s{1, 2, 2, 3};
    cuda::Upsample2x up;
    const std::vector<float> x = test::random_values(s.count(), 7);
    const cuda::Tensor xt = test::make_tensor(s, x);
    const std::vector<float> y = up.forward(xt).download();
    ASSERT_EQ(y.size(), s.count() * 4);
    for (int c = 0; c < 2; ++c) {
        for (int oy = 0; oy < 4; ++oy) {
            for (int ox = 0; ox < 6; ++ox) {
                EXPECT_FLOAT_EQ(y[(c * 4 + oy) * 6 + ox], x[(c * 2 + oy / 2) * 3 + ox / 2]);
            }
        }
    }
    const std::vector<float> dy = test::random_values(y.size(), 8);
    const std::vector<float> dx = up.backward(test::make_tensor({1, 2, 4, 6}, dy)).download();
    for (int c = 0; c < 2; ++c) {
        for (int iy = 0; iy < 2; ++iy) {
            for (int ix = 0; ix < 3; ++ix) {
                const float* p = &dy[(c * 4 + 2 * iy) * 6 + 2 * ix];
                EXPECT_NEAR(dx[(c * 2 + iy) * 3 + ix], p[0] + p[1] + p[6] + p[7], 1e-6);
            }
        }
    }
}
