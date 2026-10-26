#include "cuda/diffusion.h"
#include "cuda/noise_schedule.h"
#include "data/synthetic_sprites.h"

#include "test_support.h"

TEST(NoiseSchedule, CosineAlphaBarFallsFromOneToNearZero) {
    const cuda::NoiseSchedule schedule(1000);
    EXPECT_GT(schedule.alpha_bar(0), 0.999);
    EXPECT_LT(schedule.alpha_bar(999), 1e-3);
    EXPECT_GT(schedule.alpha_bar(999), 0.0);
    // Half way the signal is about half: cos^2 of a quarter turn.
    EXPECT_NEAR(schedule.alpha_bar(499), 0.5, 0.02);
    for (int t = 1; t < 1000; ++t) {
        EXPECT_LT(schedule.alpha_bar(t), schedule.alpha_bar(t - 1));
        EXPECT_LE(schedule.beta(t), 0.999);
        EXPECT_GT(schedule.beta(t), 0.0);
    }
}

TEST(NoiseSchedule, AlphaBarIsTheRunningProductOfOneMinusBeta) {
    const cuda::NoiseSchedule schedule(50);
    double product = 1.0;
    for (int t = 0; t < schedule.steps(); ++t) {
        product *= 1.0 - schedule.beta(t);
        EXPECT_NEAR(schedule.alpha_bar(t), product, 1e-12) << "t " << t;
    }
}

TEST(NoiseSchedule, NeedsAtLeastTwoSteps) {
    EXPECT_THROW((void)cuda::NoiseSchedule(1), std::invalid_argument);
    EXPECT_NO_THROW((void)cuda::NoiseSchedule(2));
}

TEST(Diffusion, LossFallsOnSyntheticSprites) {
    cuda::DiffusionConfig config;
    config.net = {.base = 16, .groups = 4};
    config.learning_rate = 1e-3f;
    cuda::Diffusion diffusion(config, 1);
    const int batch = 32;
    const std::vector<float> dataset = data::synthetic_sprites(batch * 8, 5);

    double first = 0.0;
    double last = 0.0;
    const int steps = 300;
    for (int step = 0; step < steps; ++step) {
        const size_t at = static_cast<size_t>(step % 8) * batch * cuda::kSpriteValues;
        const std::vector<float> chunk(dataset.begin() + at, dataset.begin() + at + batch * cuda::kSpriteValues);
        const double loss = diffusion.train_step(chunk);
        ASSERT_TRUE(std::isfinite(loss));
        if (step < 20) {
            first += loss / 20;
        }
        if (step >= steps - 20) {
            last += loss / 20;
        }
    }
    EXPECT_GT(first, 0.5);
    EXPECT_LT(last, 0.6 * first) << "first " << first << " last " << last;
}

TEST(Diffusion, SamplesAreFiniteAndInRange) {
    cuda::DiffusionConfig config;
    config.net = {.base = 8, .groups = 4};
    cuda::Diffusion diffusion(config, 2);
    const std::vector<float> sprites = diffusion.sample(4, 5, 9);
    ASSERT_EQ(sprites.size(), 4 * cuda::kSpriteValues);
    for (const float v : sprites) {
        EXPECT_TRUE(std::isfinite(v));
        EXPECT_GE(v, -1.0f);
        EXPECT_LE(v, 1.0f);
    }
    EXPECT_NE(sprites, diffusion.sample(4, 5, 10));
}
