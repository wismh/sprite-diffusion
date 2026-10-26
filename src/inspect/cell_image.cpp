#include "inspect/cell_image.h"

#include <algorithm>
#include <cmath>

namespace inspect {

namespace {

constexpr int kKernel = 3;

}

CellImage signed_image(const float* map, int side, float scale) {
    CellImage image{CellImage::Kind::Signed, side, side, std::vector<float>(static_cast<size_t>(side) * side, 0.0f)};
    if (scale > 0.0f) {
        for (size_t i = 0; i < image.values.size(); ++i) {
            image.values[i] = std::clamp(map[i] / scale, -1.0f, 1.0f);
        }
    }
    return image;
}

CellImage sprite_image(const float* sprite, float gain, float offset) {
    constexpr int kSide = 16;
    constexpr size_t kPlane = kSide * kSide;
    CellImage image{CellImage::Kind::Color, kSide, kSide, std::vector<float>(kPlane * 3)};
    for (size_t i = 0; i < kPlane; ++i) {
        for (size_t c = 0; c < 3; ++c) {
            image.values[i * 3 + c] = std::clamp(sprite[c * kPlane + i] * gain + offset, 0.0f, 1.0f);
        }
    }
    return image;
}

CellImage filter_image(const float* weights, int in_channels, float scale) {
    const float inverse = scale > 0.0f ? 1.0f / scale : 0.0f;
    if (in_channels == 3) {
        CellImage image{CellImage::Kind::Color, kKernel, kKernel, std::vector<float>(kKernel * kKernel * 3)};
        for (int i = 0; i < kKernel * kKernel; ++i) {
            for (int c = 0; c < 3; ++c) {
                image.values[static_cast<size_t>(i) * 3 + c] =
                        std::clamp(weights[c * kKernel * kKernel + i] * inverse * 0.5f + 0.5f, 0.0f, 1.0f);
            }
        }
        return image;
    }
    const int columns = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(in_channels))));
    const int rows = (in_channels + columns - 1) / columns;
    const int pitch = kKernel + 1;
    CellImage image{CellImage::Kind::Signed, columns * pitch - 1, rows * pitch - 1, {}};
    image.values.assign(static_cast<size_t>(image.width) * image.height, 0.0f);
    for (int c = 0; c < in_channels; ++c) {
        const int left = (c % columns) * pitch;
        const int top = (c / columns) * pitch;
        for (int y = 0; y < kKernel; ++y) {
            for (int x = 0; x < kKernel; ++x) {
                image.values[static_cast<size_t>(top + y) * image.width + left + x] =
                        std::clamp(weights[c * kKernel * kKernel + y * kKernel + x] * inverse, -1.0f, 1.0f);
            }
        }
    }
    return image;
}

}
