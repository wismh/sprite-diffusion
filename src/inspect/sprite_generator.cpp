#include "inspect/sprite_generator.h"
#include "data/sprite_set.h"

#include "cuda/diffusion.h"
#include "data/png_writer.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <format>
#include <random>

namespace inspect {

namespace {

constexpr int kBatch = 64;
constexpr std::size_t kLatest = 32;
constexpr std::size_t kPlane = static_cast<std::size_t>(cuda::kSpriteSize) * cuda::kSpriteSize;

std::uint8_t to_byte(float value) {
    return static_cast<std::uint8_t>(std::clamp(std::lround((value + 1.0f) * 127.5f), 0L, 255L));
}

}

SpriteGenerator::SpriteGenerator(GenerateRequest request)
    : request_(std::move(request))
    , job_([this](const std::atomic<bool>& cancel) { return generate(cancel); }) {}
std::vector<float> SpriteGenerator::latest() const {
    std::lock_guard lock(mutex_);
    return latest_;
}

std::vector<std::uint8_t> SpriteGenerator::to_rgba(const float* sprite, bool transparent) {
    std::vector<std::uint8_t> rgba(kPlane * 4);
    for (std::size_t i = 0; i < kPlane; ++i) {
        const std::uint8_t r = to_byte(sprite[i]);
        const std::uint8_t g = to_byte(sprite[kPlane + i]);
        const std::uint8_t b = to_byte(sprite[2 * kPlane + i]);
        const bool background = transparent && r >= 200 && g <= 80 && b >= 200;
        rgba[i * 4 + 0] = r;
        rgba[i * 4 + 1] = g;
        rgba[i * 4 + 2] = b;
        rgba[i * 4 + 3] = background ? 0 : 255;
    }
    return rgba;
}

GenerateResult SpriteGenerator::generate(const std::atomic<bool>& cancel) {
    GenerateResult result;
    try {
        const cuda::DiffusionConfig config = cuda::config_of(request_.checkpoint);
        std::random_device device;
        cuda::Diffusion diffusion(config, device());
        diffusion.load(request_.checkpoint);
        std::filesystem::create_directories(request_.directory);

        std::size_t index = 0;
        while (index < total()) {
            if (cancel) {
                result.cancelled = true;
                break;
            }
            const int count = static_cast<int>(std::min<std::size_t>(kBatch, total() - index));
            const std::vector<float> sprites = diffusion.sample(count, request_.steps, device());
            for (int i = 0; i < count; ++i) {
                const std::vector<std::uint8_t> rgba =
                        to_rgba(sprites.data() + static_cast<std::size_t>(i) * cuda::kSpriteValues, request_.transparent);
                data::write_png(request_.directory / std::format("{}_{:05}.png", request_.prefix, index + 1),
                        cuda::kSpriteSize, cuda::kSpriteSize, rgba, 4);
                ++index;
                done_ = index;
            }
            std::lock_guard lock(mutex_);
            latest_.assign(sprites.begin(),
                    sprites.begin() + static_cast<std::ptrdiff_t>(std::min<std::size_t>(kLatest, count) * cuda::kSpriteValues));
        }
    } catch (const std::exception& error) {
        result.error = error.what();
    }
    result.written = done_;
    return result;
}

}