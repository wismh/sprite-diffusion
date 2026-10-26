#include "cuda/diffusion.h"
#include "cuda/dream_worker.h"
#include "cuda/probe.h"
#include "cuda/unet.h"
#include "inspect/copy_check.h"
#include "inspect/nearest_sprite.h"
#include "data/sprite_set.h"

#include "test_support.h"

#include <chrono>
#include <thread>

namespace {

cuda::DiffusionConfig small_config() {
    cuda::DiffusionConfig config;
    config.net = {.base = 8, .groups = 4};
    return config;
}

}

TEST(UNetStages, InfoMatchesTheActivations) {
    cuda::UNet net({.base = 8, .groups = 4}, 1);
    const cuda::Shape xs{1, 3, 16, 16};
    cuda::Tensor x = test::make_tensor(xs, test::random_values(xs.count(), 1));
    net.forward(x, {100.0f});
    for (int i = 0; i < cuda::kStageCount; ++i) {
        const auto stage = static_cast<cuda::Stage>(i);
        const cuda::StageInfo info = net.stage_info(stage);
        EXPECT_EQ(net.activation(stage).shape(), (cuda::Shape{1, info.channels, info.size, info.size})) << i;
        EXPECT_EQ(net.stage_weight(stage).value.shape().n, info.channels) << i;
    }
}

TEST(UNetStages, GradientFromAStageMatchesFiniteDifferences) {
    cuda::UNet net({.base = 8, .groups = 4}, 3);
    const cuda::Shape xs{2, 3, 16, 16};
    std::vector<float> x = test::random_values(xs.count(), 10);
    cuda::Tensor xt = test::make_tensor(xs, x);
    const std::vector<float> t = {10.0f, 700.0f};
    std::mt19937 rng(5);
    for (int i = 0; i < cuda::kStageCount; ++i) {
        const auto stage = static_cast<cuda::Stage>(i);
        net.forward(xt, t);
        const cuda::Shape stage_shape = net.activation(stage).shape();
        const std::vector<float> r = test::random_values(stage_shape.count(), 20 + i);
        const cuda::Tensor dr = test::make_tensor(stage_shape, r);
        const auto loss = [&] {
            net.forward(xt, t);
            return test::dot(net.activation(stage).download(), r);
        };
        net.forward(xt, t);
        net.zero_grad();
        const std::vector<float> dx = net.backward_from(xt, stage, dr).download();
        ASSERT_EQ(dx.size(), x.size());
        for (int pick = 0; pick < 6; ++pick) {
            const size_t at = rng() % x.size();
            const float original = x[at];
            x[at] = original + 1e-2f;
            xt.upload(x);
            const double plus = loss();
            x[at] = original - 1e-2f;
            xt.upload(x);
            const double minus = loss();
            x[at] = original;
            xt.upload(x);
            EXPECT_NEAR((plus - minus) / 2e-2, dx[at], 3e-3 + 3e-2 * std::abs(dx[at])) << "stage " << i << " element " << at;
        }
    }
}

TEST(Probe, AnswersWithTheMapsOfEveryStageAndTheSameWayTwice) {
    const std::filesystem::path path = test::fresh_directory("probe") / "m.ckpt";
    cuda::Diffusion diffusion(small_config(), 1);
    diffusion.train_step(std::vector<float>(2 * cuda::kSpriteValues, 0.25f));
    diffusion.save(path);

    cuda::Probe probe(path);
    const std::vector<float> sprite = test::random_values(cuda::kSpriteValues, 4);
    const cuda::Answer first = probe.answer(sprite, 300, 9);
    const cuda::Answer second = probe.answer(sprite, 300, 9);
    EXPECT_EQ(first.noisy, second.noisy);
    EXPECT_EQ(first.predicted_noise, second.predicted_noise);
    for (int i = 0; i < cuda::kStageCount; ++i) {
        const cuda::StageInfo info = probe.info(static_cast<cuda::Stage>(i));
        EXPECT_EQ(first.maps[static_cast<size_t>(i)].size(), static_cast<size_t>(info.channels) * info.size * info.size);
    }
    int in = 0;
    const std::vector<float> weights = probe.weights(cuda::Stage::ConvIn, in);
    EXPECT_EQ(in, 3);
    EXPECT_EQ(weights.size(), 8u * 3 * 3 * 3);
}

TEST(Sampling, TraceRecordsEveryStepAndEndsWhereSampleDoes) {
    cuda::Diffusion diffusion(small_config(), 1);
    cuda::SampleTrace trace;
    const std::vector<float> traced = diffusion.sample(2, 5, 7, &trace);
    EXPECT_EQ(trace.timesteps.size(), 5u);
    EXPECT_EQ(trace.noisy.size(), 5u);
    EXPECT_EQ(trace.estimate.size(), 5u);
    EXPECT_EQ(trace.timesteps.front(), 999);
    EXPECT_EQ(trace.timesteps.back(), 0);
    EXPECT_EQ(diffusion.sample(2, 5, 7), traced);
}

TEST(NearestSprite, FindsTheClosestAlsoTurnedAndLeavesOutItself) {
    std::vector<float> dataset(3 * data::kSpriteValues);
    for (size_t i = 0; i < dataset.size(); ++i) {
        dataset[i] = std::sin(static_cast<float>(i) * 0.37f) * (1.0f + static_cast<float>(i / data::kSpriteValues));
    }
    const std::vector<float> turned = inspect::dataset_sprite(dataset, 1, true);
    const std::vector<inspect::Neighbor> found = inspect::nearest_sprites(turned, dataset);
    ASSERT_EQ(found.size(), 1u);
    EXPECT_EQ(found[0].index, 1u);
    EXPECT_TRUE(found[0].mirrored);
    EXPECT_NEAR(found[0].distance, 0.0, 1e-6);

    const std::vector<float> own = inspect::dataset_sprite(dataset, 2, false);
    const std::vector<inspect::Neighbor> others = inspect::nearest_sprites(own, dataset, {2});
    EXPECT_NE(others[0].index, 2u);
    EXPECT_GT(others[0].distance, 0.0);
}

TEST(CopyCheck, MeasuresASmallModelAgainstItsDataset) {
    const std::filesystem::path path = test::fresh_directory("copy_check") / "m.ckpt";
    cuda::Diffusion diffusion(small_config(), 1);
    diffusion.save(path);
    inspect::CopyCheck check(path, test::random_values(40 * data::kSpriteValues, 3), 8);
    ASSERT_TRUE(test::wait_for([&] { return check.finished(); }));
    const inspect::CopyCheckResult result = check.take_result();
    EXPECT_EQ(result.error, "");
    EXPECT_EQ(result.samples, 8u);
    EXPECT_GT(result.baseline.median, 0.0);
    EXPECT_LE(result.baseline.low, result.baseline.median);
    EXPECT_LE(result.baseline.median, result.baseline.high);
    EXPECT_LE(result.samples_spread.low, result.samples_spread.high);
    EXPECT_GE(result.share_near, 0.0);
    EXPECT_LE(result.share_near, 1.0);
}

TEST(DreamWorker, MakesASpritePerChannelOfAStage) {
    const std::filesystem::path path = test::fresh_directory("dream") / "m.ckpt";
    cuda::Diffusion diffusion(small_config(), 1);
    diffusion.train_step(std::vector<float>(2 * cuda::kSpriteValues, 0.25f));
    diffusion.save(path);

    cuda::DreamWorker worker(path);
    worker.ask(cuda::Stage::Down2, 300);
    EXPECT_EQ(worker.total(cuda::Stage::Down2), 16u);
    test::wait_for([&] {
        worker.collect();
        return worker.done(cuda::Stage::Down2) >= 16;
    });
    EXPECT_EQ(worker.error(), "");
    ASSERT_EQ(worker.done(cuda::Stage::Down2), 16u);
    ASSERT_TRUE(worker.picture(cuda::Stage::Down2, 5).has_value());
    EXPECT_EQ(worker.picture(cuda::Stage::Down2, 5)->size(), cuda::kSpriteValues);
    // Another noise level throws the pictures away.
    worker.ask(cuda::Stage::Down2, 100);
    EXPECT_EQ(worker.done(cuda::Stage::Down2), 0u);
    EXPECT_FALSE(worker.picture(cuda::Stage::Down2, 5).has_value());
}
