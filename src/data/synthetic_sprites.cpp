#include "data/synthetic_sprites.h"

#include <cmath>
#include <random>

namespace data {

namespace {
constexpr int kSize = 16;
}

std::vector<float> synthetic_sprites(int count, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dark(-1.0f, -0.4f);
    std::uniform_real_distribution<float> bright(0.2f, 1.0f);
    std::uniform_real_distribution<float> center(5.0f, 10.0f);
    std::uniform_real_distribution<float> radius(3.0f, 5.0f);
    std::bernoulli_distribution square;

    std::vector<float> sprites(static_cast<size_t>(count) * 3 * kSize * kSize);
    for (int i = 0; i < count; ++i) {
        const float background[3] = {dark(rng), dark(rng), dark(rng)};
        const float color[3] = {bright(rng), bright(rng), bright(rng)};
        const float cx = center(rng);
        const float cy = center(rng);
        const float r = radius(rng);
        const bool is_square = square(rng);
        for (int y = 0; y < kSize; ++y) {
            for (int x = 0; x < kSize; ++x) {
                const float dx = static_cast<float>(x) + 0.5f - cx;
                const float dy = static_cast<float>(y) + 0.5f - cy;
                const bool inside = is_square ? std::abs(dx) <= r && std::abs(dy) <= r : dx * dx + dy * dy <= r * r;
                for (int c = 0; c < 3; ++c) {
                    sprites[((static_cast<size_t>(i) * 3 + c) * kSize + y) * kSize + x] =
                        inside ? color[c] : background[c];
                }
            }
        }
    }
    return sprites;
}

}
