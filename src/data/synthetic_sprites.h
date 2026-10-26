#pragma once

#include <vector>

namespace data {

// A stand-in for a real dataset while there is none: `count` RGB sprites of 16x16, each a disk or a square of a random
// color on a random dark background, at a random place and size. Values are in [-1, 1], laid out as NCHW.
[[nodiscard]] std::vector<float> synthetic_sprites(int count, unsigned seed);

}
