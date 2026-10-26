#include "data/png_writer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <stdexcept>

namespace data {

namespace {

constexpr int kSpriteSize = 16;

uint32_t crc32(const uint8_t* data, size_t size, uint32_t crc = 0) {
    static const std::array<uint32_t, 256> table = [] {
        std::array<uint32_t, 256> t{};
        for (uint32_t n = 0; n < 256; ++n) {
            uint32_t c = n;
            for (int k = 0; k < 8; ++k) {
                c = (c & 1) ? 0xedb88320u ^ (c >> 1) : c >> 1;
            }
            t[n] = c;
        }
        return t;
    }();
    crc = ~crc;
    for (size_t i = 0; i < size; ++i) {
        crc = table[(crc ^ data[i]) & 0xff] ^ (crc >> 8);
    }
    return ~crc;
}

void put_u32(std::vector<uint8_t>& out, uint32_t v) {
    for (int shift = 24; shift >= 0; shift -= 8) {
        out.push_back(static_cast<uint8_t>(v >> shift));
    }
}

void write_chunk(std::ofstream& file, const char type[4], const std::vector<uint8_t>& body) {
    std::vector<uint8_t> chunk;
    put_u32(chunk, static_cast<uint32_t>(body.size()));
    chunk.insert(chunk.end(), type, type + 4);
    chunk.insert(chunk.end(), body.begin(), body.end());
    // The CRC covers the type and the body.
    put_u32(chunk, crc32(chunk.data() + 4, chunk.size() - 4));
    file.write(reinterpret_cast<const char*>(chunk.data()), static_cast<std::streamsize>(chunk.size()));
}

}

void write_png(const std::filesystem::path& path, int width, int height, const std::vector<uint8_t>& rgb,
        int channels) {
    if ((channels != 3 && channels != 4) || rgb.size() != static_cast<size_t>(width) * height * channels) {
        throw std::invalid_argument("write_png: size differs from width * height * channels");
    }
    // Every row starts with the filter byte 0 (none).
    std::vector<uint8_t> raw;
    raw.reserve(static_cast<size_t>(height) * (width * channels + 1));
    for (int y = 0; y < height; ++y) {
        raw.push_back(0);
        raw.insert(raw.end(), rgb.begin() + static_cast<size_t>(y) * width * channels,
                   rgb.begin() + static_cast<size_t>(y + 1) * width * channels);
    }
    // zlib stream of stored (uncompressed) deflate blocks.
    std::vector<uint8_t> z = {0x78, 0x01};
    for (size_t at = 0; at < raw.size() || at == 0;) {
        const size_t len = std::min<size_t>(65535, raw.size() - at);
        const bool last = at + len >= raw.size();
        z.push_back(last ? 1 : 0);
        z.push_back(static_cast<uint8_t>(len & 0xff));
        z.push_back(static_cast<uint8_t>(len >> 8));
        z.push_back(static_cast<uint8_t>(~len & 0xff));
        z.push_back(static_cast<uint8_t>((~len >> 8) & 0xff));
        z.insert(z.end(), raw.begin() + static_cast<std::ptrdiff_t>(at),
                 raw.begin() + static_cast<std::ptrdiff_t>(at + len));
        at += len;
        if (last) {
            break;
        }
    }
    uint32_t a = 1;
    uint32_t b = 0;
    for (const uint8_t byte : raw) {
        a = (a + byte) % 65521;
        b = (b + a) % 65521;
    }
    put_u32(z, (b << 16) | a);

    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("write_png: cannot open " + path.string());
    }
    const uint8_t signature[8] = {0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a};
    file.write(reinterpret_cast<const char*>(signature), 8);
    std::vector<uint8_t> header;
    put_u32(header, static_cast<uint32_t>(width));
    put_u32(header, static_cast<uint32_t>(height));
    header.insert(header.end(), {8, static_cast<uint8_t>(channels == 4 ? 6 : 2), 0, 0, 0});  // 8 bits, RGB(A), deflate, ...
    write_chunk(file, "IHDR", header);
    write_chunk(file, "IDAT", z);
    write_chunk(file, "IEND", {});
    if (!file) {
        throw std::runtime_error("write_png: cannot write " + path.string());
    }
}

void write_sprite_grid(const std::filesystem::path& path, const std::vector<float>& sprites, int columns, int scale) {
    const size_t per_sprite = static_cast<size_t>(3) * kSpriteSize * kSpriteSize;
    const int count = static_cast<int>(sprites.size() / per_sprite);
    const int rows = (count + columns - 1) / columns;
    const int cell = kSpriteSize * scale + 2;
    const int width = columns * cell;
    const int height = rows * cell;
    std::vector<uint8_t> rgb(static_cast<size_t>(width) * height * 3, 40);
    for (int i = 0; i < count; ++i) {
        const int left = (i % columns) * cell + 1;
        const int top = (i / columns) * cell + 1;
        for (int y = 0; y < kSpriteSize * scale; ++y) {
            for (int x = 0; x < kSpriteSize * scale; ++x) {
                for (int c = 0; c < 3; ++c) {
                    const float v = sprites[(static_cast<size_t>(i) * 3 + c) * kSpriteSize * kSpriteSize +
                                            (y / scale) * kSpriteSize + x / scale];
                    rgb[(static_cast<size_t>(top + y) * width + left + x) * 3 + c] =
                        static_cast<uint8_t>(std::lround(std::clamp((v + 1.0f) * 0.5f, 0.0f, 1.0f) * 255.0f));
                }
            }
        }
    }
    write_png(path, width, height, rgb);
}

}
