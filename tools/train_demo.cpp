// Trains the network on synthetic sprites through the background Trainer, resuming from <folder>/model.ckpt when it
// exists, and writes a grid of the training sprites and a grid of generated ones.
// Usage: train_demo <folder> [steps of this run] [batch] [base channels]
#include "cuda/trainer.h"
#include "data/png_writer.h"
#include "data/sprite_set.h"
#include "data/synthetic_sprites.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <thread>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: train_demo <folder> [steps] [batch] [base channels]\n");
        return 2;
    }
    const std::filesystem::path folder = argv[1];
    cuda::DiffusionConfig config;
    config.net.base = argc > 4 ? std::atoi(argv[4]) : 64;
    cuda::TrainerConfig trainer_config;
    trainer_config.max_steps = argc > 2 ? std::atoi(argv[2]) : 3000;
    trainer_config.batch = argc > 3 ? std::atoi(argv[3]) : 64;
    trainer_config.checkpoint_every = 1000;
    trainer_config.checkpoint_path = folder / "model.ckpt";
    trainer_config.preview_every = 0;
    std::filesystem::create_directories(folder);

    cuda::Diffusion diffusion(config, 1);
    if (std::filesystem::exists(trainer_config.checkpoint_path)) {
        diffusion.load(trainer_config.checkpoint_path);
        std::printf("resumed at step %d\n", diffusion.steps_done());
    }
    data::SpriteSet set(data::synthetic_sprites(2048, 7), 7);
    data::write_sprite_grid(folder / "training_data.png", set.next_batch(64), 8, 6);

    cuda::Trainer trainer(diffusion, trainer_config, [&set](int count) { return set.next_batch(count); });
    const auto start = std::chrono::steady_clock::now();
    trainer.start();
    for (;;) {
        std::this_thread::sleep_for(std::chrono::seconds(10));
        const cuda::TrainerStatus status = trainer.status();
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        std::printf("step %5d  loss %.4f  %.0f s\n", status.step, status.loss, seconds);
        std::fflush(stdout);
        if (status.state == cuda::TrainerStatus::State::Failed) {
            std::fprintf(stderr, "failed: %s\n", status.error.c_str());
            return 1;
        }
        if (status.state == cuda::TrainerStatus::State::Finished) {
            break;
        }
    }
    trainer.stop();
    data::write_sprite_grid(folder / "samples.png", diffusion.sample(64, 50, 3), 8, 6);
    std::printf("wrote %s\n", (folder / "samples.png").string().c_str());
    return 0;
}
