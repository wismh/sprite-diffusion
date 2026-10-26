// Time of one training step. Usage: bench [steps] [batch] [base channels]
#include "cuda/diffusion.h"
#include "data/synthetic_sprites.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
    const int steps = argc > 1 ? std::atoi(argv[1]) : 30;
    const int batch = argc > 2 ? std::atoi(argv[2]) : 64;
    cuda::DiffusionConfig config;
    config.net.base = argc > 3 ? std::atoi(argv[3]) : 64;
    cuda::Diffusion diffusion(config, 1);
    const std::vector<float> sprites = data::synthetic_sprites(batch, 1);

    for (int i = 0; i < 5; ++i) {
        diffusion.train_step(sprites);  // warm-up: allocations, caches
    }
    const auto start = std::chrono::steady_clock::now();
    double loss = 0.0;
    for (int i = 0; i < steps; ++i) {
        loss = diffusion.train_step(sprites);
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::printf("%.1f ms per step (batch %d, base %d), loss %.3f\n", 1000.0 * seconds / steps, batch,
            config.net.base, loss);
    return 0;
}
