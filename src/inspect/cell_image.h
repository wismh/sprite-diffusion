#pragma once

#include <cstddef>
#include <vector>

namespace inspect {

// A small picture ready to be drawn pixel by pixel: a feature map, the weights of a filter, a sprite.
struct CellImage {
    enum class Kind { None, Signed, Color };

    Kind kind = Kind::None;
    int width = 0;
    int height = 0;
    // Signed: width * height values in -1..1 (positive reddish, negative bluish). Color: width * height * 3 values in 0..1,
    // pixel after pixel.
    std::vector<float> values;
};

// A map of side x side values, each divided by `scale` (0 gives all zeros) and cut to -1..1.
[[nodiscard]] CellImage signed_image(const float* map, int side, float scale);
// A sprite (NCHW, [-1, 1], 16x16) as a color picture; `gain` and `offset` map a value v to v * gain + offset before the
// cut to 0..1 (0.5 and 0.5 for a sprite, a lower gain for noise).
[[nodiscard]] CellImage sprite_image(const float* sprite, float gain = 0.5f, float offset = 0.5f);
// The weights of one filter ({in, 3, 3}). A filter that reads the three color channels is one 3x3 color picture; any
// other is a sheet with a 3x3 tile per input channel, as square as it can be, 1 pixel apart. Divided by `scale`.
[[nodiscard]] CellImage filter_image(const float* weights, int in_channels, float scale);

}
