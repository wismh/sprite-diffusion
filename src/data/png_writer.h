#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

namespace data {

// Writes an 8-bit image, row by row, as an uncompressed PNG: RGB (3 channels) or RGBA (4). Throws on I/O failure.
void write_png(const std::filesystem::path& path, int width, int height, const std::vector<uint8_t>& pixels,
        int channels = 3);

// Lays sprites (NCHW, [-1, 1], 16x16) out in a grid of `columns` columns, each pixel enlarged `scale` times, and
// writes the grid as a PNG.
void write_sprite_grid(const std::filesystem::path& path, const std::vector<float>& sprites, int columns, int scale);

}
