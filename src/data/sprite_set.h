#pragma once

#include <cstddef>
#include <random>
#include <vector>

namespace data {

constexpr size_t kSpriteValues = 3 * 16 * 16;

// A set of 16x16 RGB sprites ([-1, 1], NCHW) that hands out shuffled batches: every sprite once per pass, a new order
// for the next pass. With `mirror`, each sprite handed out is turned left to right with a chance of one half.
class SpriteSet {
public:
    SpriteSet(std::vector<float> sprites, unsigned seed, bool mirror = false);

    [[nodiscard]] size_t size() const { return order_.size(); }
    // `count` sprites, concatenated. Throws when the set is empty.
    [[nodiscard]] std::vector<float> next_batch(int count);

private:
    std::vector<float> sprites_;
    std::vector<size_t> order_;
    size_t next_ = 0;
    bool mirror_;
    std::mt19937 rng_;
};

}
