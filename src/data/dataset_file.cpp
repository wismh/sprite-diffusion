#include "data/dataset_file.h"

#include "data/file_names.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace data {

namespace {

constexpr std::uint32_t kMagic = 0x53445053;  // "SPDS"
constexpr std::uint32_t kVersion = 1;
constexpr std::size_t kMaxSource = 4096;

std::uint64_t checksum(const char* data, std::size_t size) {
    std::uint64_t hash = 1469598103934665603ull;
    for (std::size_t i = 0; i < size; ++i) {
        hash = (hash ^ static_cast<unsigned char>(data[i])) * 1099511628211ull;
    }
    return hash;
}

template <class T>
void put(std::vector<char>& out, T value) {
    const char* p = reinterpret_cast<const char*>(&value);
    out.insert(out.end(), p, p + sizeof(T));
}

struct Header {
    std::uint32_t count = 0;
    std::array<std::uint8_t, 3> background{};
    std::string source;
    std::size_t pixels_at = 0;  // offset of the pixels in the file
};

// Reads the header out of `in`, which holds at least `size` bytes of the file's start.
Header read_header(std::istream& in, const std::filesystem::path& path) {
    const auto fail = [&path](const char* what) {
        return std::runtime_error("dataset " + path.string() + ": " + what);
    };
    const auto get = [&](auto& value) {
        in.read(reinterpret_cast<char*>(&value), sizeof(value));
        if (!in) {
            throw fail("the file ends early");
        }
    };
    std::uint32_t magic = 0;
    std::uint32_t version = 0;
    get(magic);
    get(version);
    if (magic != kMagic) {
        throw fail("not a sprite dataset");
    }
    if (version != kVersion) {
        throw fail("unsupported version");
    }
    Header header;
    get(header.count);
    for (std::uint8_t& channel : header.background) {
        get(channel);
    }
    std::uint32_t source_size = 0;
    get(source_size);
    if (source_size > kMaxSource) {
        throw fail("the source text is too long");
    }
    header.source.resize(source_size);
    in.read(header.source.data(), source_size);
    if (!in) {
        throw fail("the file ends early");
    }
    header.pixels_at = static_cast<std::size_t>(in.tellg());
    return header;
}

}

void save_dataset(const std::filesystem::path& path, const Dataset& dataset) {
    if (dataset.pixels.size() % kSpriteValues != 0) {
        throw std::invalid_argument("save_dataset: a whole number of sprites");
    }
    std::vector<char> bytes;
    bytes.reserve(dataset.pixels.size() + 4096);
    put(bytes, kMagic);
    put(bytes, kVersion);
    put(bytes, static_cast<std::uint32_t>(dataset.count()));
    for (const std::uint8_t channel : dataset.background) {
        put(bytes, channel);
    }
    const std::string source = dataset.source.substr(0, kMaxSource);
    put(bytes, static_cast<std::uint32_t>(source.size()));
    bytes.insert(bytes.end(), source.begin(), source.end());
    bytes.insert(bytes.end(), reinterpret_cast<const char*>(dataset.pixels.data()),
            reinterpret_cast<const char*>(dataset.pixels.data()) + dataset.pixels.size());
    put(bytes, checksum(bytes.data(), bytes.size()));

    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::filesystem::path temporary = path;
    temporary += ".tmp";
    {
        std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
        file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        if (!file) {
            throw std::runtime_error("dataset: cannot write " + temporary.string());
        }
    }
    std::filesystem::rename(temporary, path);
}

Dataset load_dataset(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("dataset: cannot open " + path.string());
    }
    const std::size_t size = static_cast<std::size_t>(file.tellg());
    if (size < sizeof(std::uint64_t)) {
        throw std::runtime_error("dataset " + path.string() + ": the file is too short");
    }
    std::vector<char> bytes(size);
    file.seekg(0);
    file.read(bytes.data(), static_cast<std::streamsize>(size));
    if (!file) {
        throw std::runtime_error("dataset: cannot read " + path.string());
    }
    std::uint64_t stored = 0;
    std::memcpy(&stored, bytes.data() + size - sizeof(stored), sizeof(stored));
    if (stored != checksum(bytes.data(), size - sizeof(stored))) {
        throw std::runtime_error("dataset " + path.string() + ": the checksum differs, the file is damaged");
    }

    file.clear();
    file.seekg(0);
    const Header header = read_header(file, path);
    const std::size_t pixel_bytes = static_cast<std::size_t>(header.count) * kSpriteValues;
    if (header.pixels_at + pixel_bytes + sizeof(stored) != size) {
        throw std::runtime_error("dataset " + path.string() + ": the size does not fit the count");
    }
    Dataset dataset;
    dataset.background = header.background;
    dataset.source = header.source;
    dataset.pixels.assign(reinterpret_cast<const std::uint8_t*>(bytes.data()) + header.pixels_at,
            reinterpret_cast<const std::uint8_t*>(bytes.data()) + header.pixels_at + pixel_bytes);
    return dataset;
}

DatasetInfo peek_dataset(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("dataset: cannot open " + path.string());
    }
    const Header header = read_header(file, path);
    // The pixels are not read, but a file of the wrong length is not a dataset.
    std::error_code error;
    if (std::filesystem::file_size(path, error) !=
        header.pixels_at + static_cast<std::size_t>(header.count) * kSpriteValues + sizeof(std::uint64_t)) {
        throw std::runtime_error("dataset " + path.string() + ": the size does not fit the count");
    }
    DatasetInfo info;
    info.name = to_utf8(path.stem());
    info.path = path;
    info.count = header.count;
    info.source = header.source;
    info.background = header.background;
    return info;
}

std::vector<float> to_floats(const Dataset& dataset, std::size_t first, std::size_t count) {
    const std::size_t available = dataset.count();
    first = std::min(first, available);
    count = std::min(count, available - first);
    std::vector<float> values(count * kSpriteValues);
    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<float>(dataset.pixels[first * kSpriteValues + i]) / 127.5f - 1.0f;
    }
    return values;
}

}
