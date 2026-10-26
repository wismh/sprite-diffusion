#pragma once

#include "data/sprite_set.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace data {

// A set of 16x16 sprites as bytes. Transparent pixels have been replaced by a flat background color: the network learns
// opaque pictures, and the background doubles as the key to cut the generated sprites out again.
struct Dataset {
    std::array<std::uint8_t, 3> background{255, 0, 255};
    std::string source;  // where the sprites came from, for the person looking at the list
    // count * kSpriteValues bytes, one sprite after another, channel by channel (R plane, G plane, B plane), row by row.
    std::vector<std::uint8_t> pixels;

    [[nodiscard]] std::size_t count() const { return pixels.size() / kSpriteValues; }
};

// What the list of datasets shows, read without the pixels.
struct DatasetInfo {
    std::string name;
    std::filesystem::path path;
    std::size_t count = 0;
    std::string source;
    std::array<std::uint8_t, 3> background{255, 0, 255};
};

// The file: magic, version, count, background, source text, pixels, FNV-1a checksum of everything before it. Written to a
// temporary file and renamed. load_dataset() throws std::runtime_error on a damaged or foreign file.
void save_dataset(const std::filesystem::path& path, const Dataset& dataset);
[[nodiscard]] Dataset load_dataset(const std::filesystem::path& path);
// The header only: cheap for a list, no checksum (the length of the file is checked). Throws like load_dataset() when the
// header or the length is wrong.
[[nodiscard]] DatasetInfo peek_dataset(const std::filesystem::path& path);

// The sprites of `dataset` from `first` on, at most `count` of them, as floats in [-1, 1].
[[nodiscard]] std::vector<float> to_floats(const Dataset& dataset, std::size_t first, std::size_t count);

}
