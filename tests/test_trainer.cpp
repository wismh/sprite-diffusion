#include "cuda/trainer.h"
#include "data/sprite_set.h"
#include "data/synthetic_sprites.h"

#include "test_support.h"

#include <chrono>
#include <filesystem>
#include <thread>

namespace {

using State = cuda::TrainerStatus::State;

cuda::DiffusionConfig small_config() {
    cuda::DiffusionConfig config;
    config.net = {.base = 8, .groups = 4};
    config.learning_rate = 1e-3f;
    return config;
}

cuda::TrainerConfig small_trainer() {
    cuda::TrainerConfig config;
    config.batch = 8;
    config.preview_every = 0;
    config.preview_count = 2;
    config.preview_steps = 3;
    return config;
}

}

TEST(Trainer, RunsToTheStepLimitAndReports) {
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "diffusion_tests";
    const std::filesystem::path path = dir / "trainer.ckpt";
    std::filesystem::remove(path);

    cuda::Diffusion diffusion(small_config(), 1);
    data::SpriteSet set(data::synthetic_sprites(64, 1), 1);
    cuda::TrainerConfig config = small_trainer();
    config.max_steps = 30;
    config.preview_every = 10;
    config.checkpoint_every = 15;
    config.checkpoint_path = path;
    cuda::Trainer trainer(diffusion, config, [&](int n) { return set.next_batch(n); });

    trainer.start();
    ASSERT_TRUE(test::wait_for([&] { return trainer.status().state == State::Finished; }));
    const cuda::TrainerStatus status = trainer.status();
    EXPECT_EQ(status.step, 30);
    EXPECT_TRUE(std::isfinite(status.loss));
    EXPECT_GT(status.loss, 0.0);
    EXPECT_EQ(status.checkpoints, 3);  // after steps 15 and 30, and at the end
    EXPECT_EQ(trainer.loss_history().size(), 3u);

    const std::optional<cuda::Preview> preview = trainer.take_preview();
    ASSERT_TRUE(preview.has_value());
    EXPECT_EQ(preview->step, 30);
    EXPECT_EQ(preview->sprites.size(), 2 * cuda::kSpriteValues);
    EXPECT_FALSE(trainer.take_preview().has_value());

    cuda::Diffusion resumed(small_config(), 2);
    resumed.load(path);
    EXPECT_EQ(resumed.steps_done(), 30);
}

TEST(Trainer, PausesResumesAndStops) {
    cuda::Diffusion diffusion(small_config(), 1);
    data::SpriteSet set(data::synthetic_sprites(64, 1), 1);
    cuda::Trainer trainer(diffusion, small_trainer(), [&](int n) { return set.next_batch(n); });

    trainer.start();
    ASSERT_TRUE(test::wait_for([&] { return trainer.status().step >= 3; }));
    trainer.pause();
    EXPECT_EQ(trainer.status().state, State::Paused);
    std::this_thread::sleep_for(std::chrono::milliseconds(300));  // the step in progress ends
    const int paused_at = trainer.status().step;
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    EXPECT_EQ(trainer.status().step, paused_at);

    trainer.resume();
    EXPECT_EQ(trainer.status().state, State::Running);
    ASSERT_TRUE(test::wait_for([&] { return trainer.status().step >= paused_at + 3; }));
    trainer.stop();
    EXPECT_EQ(trainer.status().state, State::Idle);

    // A stopped trainer starts again where the Diffusion is.
    const int stopped_at = trainer.status().step;
    trainer.start();
    ASSERT_TRUE(test::wait_for([&] { return trainer.status().step >= stopped_at + 2; }));
    trainer.stop();
}

TEST(Trainer, ServesPreviewAndCheckpointRequestsWhilePaused) {
    const std::filesystem::path path = std::filesystem::temp_directory_path() / "diffusion_tests" / "requested.ckpt";
    std::filesystem::remove(path);
    cuda::Diffusion diffusion(small_config(), 1);
    data::SpriteSet set(data::synthetic_sprites(64, 1), 1);
    cuda::TrainerConfig config = small_trainer();
    config.checkpoint_path = path;
    cuda::Trainer trainer(diffusion, config, [&](int n) { return set.next_batch(n); });

    trainer.start();
    trainer.pause();
    trainer.request_preview();
    trainer.request_checkpoint();
    std::optional<cuda::Preview> preview;
    ASSERT_TRUE(test::wait_for([&] {
        if (!preview) {
            preview = trainer.take_preview();
        }
        return preview.has_value() && trainer.status().checkpoints == 1;
    }));
    EXPECT_TRUE(std::filesystem::exists(path));
    trainer.stop();
}

TEST(Trainer, ReportsAFailureOfTheBatchSource) {
    cuda::Diffusion diffusion(small_config(), 1);
    cuda::Trainer trainer(diffusion, small_trainer(), [](int) -> std::vector<float> { throw std::runtime_error("no data"); });
    trainer.start();
    ASSERT_TRUE(test::wait_for([&] { return trainer.status().state == State::Failed; }));
    EXPECT_EQ(trainer.status().error, "no data");
    trainer.stop();
    EXPECT_EQ(trainer.status().state, State::Failed);
}
