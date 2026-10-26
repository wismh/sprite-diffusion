#include "data/sprite_set.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace data {

SpriteSet::SpriteSet(std::vector<float> sprites, unsigned seed, bool mirror)
    : sprites_(std::move(sprites)), rng_(seed), mirror_(mirror) {
    if (sprites_.size() % kSpriteValues != 0) {
        throw std::invalid_argument("SpriteSet: a whole number of sprites");
    }
    order_.resize(sprites_.size() / kSpriteValues);
    std::iota(order_.begin(), order_.end(), size_t{0});
    std::shuffle(order_.begin(), order_.end(), rng_);
}

std::vector<float> SpriteSet::next_batch(int count) {
    if (order_.empty()) {
        throw std::runtime_error("SpriteSet: no sprites");
    }
    std::vector<float> batch;
    batch.reserve(static_cast<size_t>(count) * kSpriteValues);
    for (int i = 0; i < count; ++i) {
        if (next_ == order_.size()) {
            std::shuffle(order_.begin(), order_.end(), rng_);
            next_ = 0;
        }
        const auto first = sprites_.begin() + static_cast<std::ptrdiff_t>(order_[next_++] * kSpriteValues);
        const auto at = batch.insert(batch.end(), first, first + static_cast<std::ptrdiff_t>(kSpriteValues));
        if (mirror_ && (rng_() & 1u) != 0) {
            // Every row of 16 values, in each of the three channels.
            for (size_t row = 0; row < kSpriteValues / 16; ++row) {
                std::reverse(at + static_cast<std::ptrdiff_t>(row * 16), at + static_cast<std::ptrdiff_t>(row * 16 + 16));
            }
        }
    }
    return batch;
}

}
