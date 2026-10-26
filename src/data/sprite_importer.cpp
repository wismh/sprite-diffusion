#include "data/sprite_importer.h"

#include "data/file_names.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <stdexcept>
#include <utility>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_STDIO
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_ONLY_BMP
#define STBI_ONLY_TGA
#include "../../third_party/stb/stb_image.h"

namespace data {

namespace {

constexpr int kSide = 16;
constexpr int kAlphaThreshold = 128;

bool is_picture(const std::filesystem::path& path) {
    std::string extension = to_utf8(path.extension());
    std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) { return std::tolower(c); });
    return extension == ".png" || extension == ".bmp" || extension == ".tga" || extension == ".jpg" ||
           extension == ".jpeg";
}

// The 16x16 picture in `path` as RGBA bytes; empty when it is something else or cannot be read.
std::vector<std::uint8_t> read_sprite(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        return {};
    }
    const std::streamsize size = in.tellg();
    if (size <= 0) {
        return {};
    }
    std::vector<unsigned char> bytes(static_cast<std::size_t>(size));
    in.seekg(0);
    if (!in.read(reinterpret_cast<char*>(bytes.data()), size)) {
        return {};
    }
    int width = 0;
    int height = 0;
    int channels = 0;
    unsigned char* decoded = stbi_load_from_memory(bytes.data(), static_cast<int>(bytes.size()), &width, &height,
            &channels, 4);
    if (decoded == nullptr) {
        return {};
    }
    std::vector<std::uint8_t> rgba;
    if (width == kSide && height == kSide) {
        rgba.assign(decoded, decoded + kSide * kSide * 4);
    }
    stbi_image_free(decoded);
    return rgba;
}

}

SpriteImporter::SpriteImporter(std::filesystem::path folder, std::array<std::uint8_t, 3> background)
    : folder_(std::move(folder))
    , background_(background)
    , job_([this](const std::atomic<bool>& cancel) { return import(cancel); }) {}

ImportResult SpriteImporter::import(const std::atomic<bool>& cancel) {
    ImportResult result;
    result.dataset.background = background_;
    result.dataset.source = to_utf8(folder_);

    std::vector<std::filesystem::path> files;
    std::error_code error;
    if (!std::filesystem::is_directory(folder_, error)) {
        throw std::runtime_error("not a folder: " + to_utf8(folder_));
    }
    for (const auto& entry : std::filesystem::recursive_directory_iterator(folder_, error)) {
        if (cancel) {
            break;
        }
        if (entry.is_regular_file() && is_picture(entry.path())) {
            files.push_back(entry.path());
        }
    }
    // The same order on every run, whatever the file system hands out first.
    std::sort(files.begin(), files.end());
    total_ = files.size();

    result.dataset.pixels.reserve(files.size() * kSpriteValues);
    for (const std::filesystem::path& file : files) {
        if (cancel) {
            break;
        }
        const std::vector<std::uint8_t> rgba = read_sprite(file);
        if (rgba.empty()) {
            ++result.skipped;
        } else {
            // Planar: the R plane, the G plane, the B plane of the sprite.
            const std::size_t base = result.dataset.pixels.size();
            result.dataset.pixels.resize(base + kSpriteValues);
            for (int i = 0; i < kSide * kSide; ++i) {
                const std::uint8_t* pixel = rgba.data() + i * 4;
                const bool opaque = pixel[3] >= kAlphaThreshold;
                for (int c = 0; c < 3; ++c) {
                    result.dataset.pixels[base + static_cast<std::size_t>(c) * kSide * kSide + i] =
                            opaque ? pixel[c] : background_[c];
                }
            }
        }
        ++done_;
    }
    if (cancel) {
        result.cancelled = true;
    } else if (result.dataset.count() == 0) {
        result.error = "no 16x16 pictures in the folder";
    }
    return result;
}

}