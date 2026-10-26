#include "cuda/conv2d.h"

#include "test_support.h"

namespace {

// Direct convolution, the reference for the im2col path.
std::vector<float> reference_conv(const std::vector<float>& x, const cuda::Shape& xs, const std::vector<float>& w,
                                  const std::vector<float>& b, int out, int k, int stride, int pad, cuda::Shape& ys) {
    const int ho = (xs.h + 2 * pad - k) / stride + 1;
    const int wo = (xs.w + 2 * pad - k) / stride + 1;
    ys = {xs.n, out, ho, wo};
    std::vector<float> y(ys.count());
    for (int n = 0; n < xs.n; ++n) {
        for (int o = 0; o < out; ++o) {
            for (int oy = 0; oy < ho; ++oy) {
                for (int ox = 0; ox < wo; ++ox) {
                    double sum = b[o];
                    for (int c = 0; c < xs.c; ++c) {
                        for (int ky = 0; ky < k; ++ky) {
                            for (int kx = 0; kx < k; ++kx) {
                                const int iy = oy * stride - pad + ky;
                                const int ix = ox * stride - pad + kx;
                                if (iy < 0 || iy >= xs.h || ix < 0 || ix >= xs.w) {
                                    continue;
                                }
                                sum += static_cast<double>(x[((n * xs.c + c) * xs.h + iy) * xs.w + ix]) *
                                       w[((o * xs.c + c) * k + ky) * k + kx];
                            }
                        }
                    }
                    y[((n * out + o) * ho + oy) * wo + ox] = static_cast<float>(sum);
                }
            }
        }
    }
    return y;
}

struct ConvCase {
    int kernel;
    int stride;
    int pad;
};

class Conv2dTest : public testing::TestWithParam<ConvCase> {};

TEST_P(Conv2dTest, ForwardMatchesDirectConvolution) {
    const ConvCase p = GetParam();
    const cuda::Shape xs{2, 3, 6, 6};
    cuda::Conv2d conv(3, 4, p.kernel, p.stride, p.pad, 1);
    const std::vector<float> x = test::random_values(xs.count(), 10);
    const std::vector<float> w = test::random_values(4 * 3 * p.kernel * p.kernel, 11);
    const std::vector<float> b = test::random_values(4, 12);
    conv.weight().value.upload(w);
    conv.bias().value.upload(b);
    const cuda::Tensor xt = test::make_tensor(xs, x);

    cuda::Shape ys;
    const std::vector<float> expected = reference_conv(x, xs, w, b, 4, p.kernel, p.stride, p.pad, ys);
    const cuda::Tensor& y = conv.forward(xt);
    ASSERT_EQ(y.shape(), ys);
    const std::vector<float> actual = y.download();
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(actual[i], expected[i], 1e-4) << i;
    }
}

TEST_P(Conv2dTest, GradientsMatchFiniteDifferences) {
    const ConvCase p = GetParam();
    const cuda::Shape xs{2, 3, 6, 6};
    cuda::Conv2d conv(3, 4, p.kernel, p.stride, p.pad, 1);
    std::vector<float> x = test::random_values(xs.count(), 20);
    std::vector<float> w = test::random_values(4 * 3 * p.kernel * p.kernel, 21);
    std::vector<float> b = test::random_values(4, 22);
    conv.weight().value.upload(w);
    conv.bias().value.upload(b);
    cuda::Tensor xt = test::make_tensor(xs, x);

    const std::vector<float> r = test::random_values(conv.forward(xt).size(), 23);
    const cuda::Tensor dy = test::make_tensor(conv.forward(xt).shape(), r);
    const auto loss = [&] { return test::dot(conv.forward(xt).download(), r); };

    conv.forward(xt);
    for (cuda::Param* param : conv.params()) {
        param->zero_grad();
    }
    const std::vector<float> dx = conv.backward(xt, dy).download();
    const std::vector<float> dw = conv.weight().grad.download();
    const std::vector<float> db = conv.bias().grad.download();

    test::expect_gradient(x, dx, [&](const std::vector<float>& v) { xt.upload(v); }, loss);
    test::expect_gradient(w, dw, [&](const std::vector<float>& v) { conv.weight().value.upload(v); }, loss);
    test::expect_gradient(b, db, [&](const std::vector<float>& v) { conv.bias().value.upload(v); }, loss);
}

INSTANTIATE_TEST_SUITE_P(Kernels, Conv2dTest,
                         testing::Values(ConvCase{3, 1, 1}, ConvCase{3, 2, 1}, ConvCase{1, 1, 0}, ConvCase{3, 1, 0}));

}
