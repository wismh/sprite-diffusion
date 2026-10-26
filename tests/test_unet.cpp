#include "cuda/ops.h"
#include "cuda/res_block.h"
#include "cuda/unet.h"

#include "test_support.h"

TEST(ResBlock, GradientsMatchFiniteDifferences) {
    for (const int out_channels : {4, 8}) {  // without and with the 1x1 skip convolution
        const cuda::Shape xs{2, 4, 4, 4};
        const cuda::Shape es{2, 6, 1, 1};
        cuda::ResBlock block(4, out_channels, 6, 4, 5);
        std::vector<float> x = test::random_values(xs.count(), 1);
        std::vector<float> emb = test::random_values(es.count(), 2);
        cuda::Tensor xt = test::make_tensor(xs, x);
        cuda::Tensor et = test::make_tensor(es, emb);

        const cuda::Shape ys{2, out_channels, 4, 4};
        const std::vector<float> r = test::random_values(ys.count(), 3);
        const cuda::Tensor dy = test::make_tensor(ys, r);
        const auto loss = [&] { return test::dot(block.forward(xt, et).download(), r); };

        block.forward(xt, et);
        for (cuda::Param* param : block.params()) {
            param->zero_grad();
        }
        cuda::Tensor demb(es);
        demb.zero();
        const std::vector<float> dx = block.backward(xt, et, dy, demb).download();
        const std::vector<float> de = demb.download();

        test::expect_gradient(x, dx, [&](const std::vector<float>& v) { xt.upload(v); }, loss);
        test::expect_gradient(emb, de, [&](const std::vector<float>& v) { et.upload(v); }, loss);
        test::expect_param_gradients(block.params(), loss);
    }
}

TEST(UNet, KeepsTheShapeAndGradientsMatchFiniteDifferences) {
    cuda::UNet net({.base = 8, .groups = 4}, 1);
    const cuda::Shape xs{2, 3, 16, 16};
    cuda::Tensor x = test::make_tensor(xs, test::random_values(xs.count(), 10));
    const std::vector<float> t = {10.0f, 700.0f};
    const std::vector<float> r = test::random_values(xs.count(), 11);
    const cuda::Tensor dy = test::make_tensor(xs, r);
    const auto loss = [&] { return test::dot(net.forward(x, t).download(), r); };

    ASSERT_EQ(net.forward(x, t).shape(), xs);
    net.zero_grad();
    net.backward(x, dy);
    test::expect_param_gradients(net.params(), loss, 3, 1e-2f, 3e-2f);
}
