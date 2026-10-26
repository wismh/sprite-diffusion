#include "cuda/diffusion.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

// The file, all integers little-endian (the host's own order on every supported platform):
//   u32 magic, u32 version, i32 timesteps, i32 base channels, i32 groups, i32 Adam steps,
//   u32 size and bytes of the dataset name, u32 size and bytes of the random generator state (text),
//   u32 parameter count, then per parameter:
//     u64 element count, floats of the weights, floats of the average, u8 has-moments, [floats of m, floats of v],
//   u64 FNV-1a checksum of everything above.

namespace cuda {

namespace {

constexpr uint32_t kMagic = 0x4e534644;  // "DFSN"
constexpr uint32_t kVersion = 2;
constexpr uint32_t kMaxName = 4096;

uint64_t checksum(const char* data, size_t size) {
    uint64_t hash = 1469598103934665603ull;
    for (size_t i = 0; i < size; ++i) {
        hash = (hash ^ static_cast<unsigned char>(data[i])) * 1099511628211ull;
    }
    return hash;
}

class Writer {
public:
    template <class T>
    void put(T value) {
        append(&value, sizeof(T));
    }
    void put_floats(const std::vector<float>& values) { append(values.data(), values.size() * sizeof(float)); }
    void put_text(const std::string& text) {
        put(static_cast<uint32_t>(text.size()));
        append(text.data(), text.size());
    }
    [[nodiscard]] std::vector<char>& bytes() { return bytes_; }

private:
    void append(const void* data, size_t size) {
        const char* p = static_cast<const char*>(data);
        bytes_.insert(bytes_.end(), p, p + size);
    }

    std::vector<char> bytes_;
};

class Reader {
public:
    explicit Reader(const std::vector<char>& bytes, size_t end) : bytes_(bytes), end_(end) {}

    template <class T>
    T get() {
        T value;
        take(&value, sizeof(T));
        return value;
    }
    std::vector<float> get_floats(size_t count) {
        std::vector<float> values(count);
        take(values.data(), count * sizeof(float));
        return values;
    }
    std::string get_text() {
        const uint32_t size = get<uint32_t>();
        std::string text(size, '\0');
        take(text.data(), size);
        return text;
    }

private:
    void take(void* out, size_t size) {
        if (size > end_ - pos_) {
            throw std::runtime_error("checkpoint: the file ends early");
        }
        std::memcpy(out, bytes_.data() + pos_, size);
        pos_ += size;
    }

    const std::vector<char>& bytes_;
    size_t end_;
    size_t pos_ = 0;
};

}

void Diffusion::save(const std::filesystem::path& path) {
    Writer w;
    w.put(kMagic);
    w.put(kVersion);
    w.put<int32_t>(config_.timesteps);
    w.put<int32_t>(config_.net.base);
    w.put<int32_t>(config_.net.groups);
    w.put<int32_t>(adam_.steps());
    w.put_text(dataset_);
    std::ostringstream rng;
    rng << rng_;
    w.put_text(rng.str());

    const std::vector<Param*> model = model_.params();
    const std::vector<Param*> ema = ema_.params();
    w.put(static_cast<uint32_t>(model.size()));
    for (size_t i = 0; i < model.size(); ++i) {
        w.put(static_cast<uint64_t>(model[i]->value.size()));
        w.put_floats(model[i]->value.download());
        w.put_floats(ema[i]->value.download());
        const bool has_moments = model[i]->m.size() == model[i]->value.size();
        w.put<uint8_t>(has_moments ? 1 : 0);
        if (has_moments) {
            w.put_floats(model[i]->m.download());
            w.put_floats(model[i]->v.download());
        }
    }
    w.put(checksum(w.bytes().data(), w.bytes().size()));

    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::filesystem::path temporary = path;
    temporary += ".tmp";
    {
        std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
        file.write(w.bytes().data(), static_cast<std::streamsize>(w.bytes().size()));
        if (!file) {
            throw std::runtime_error("checkpoint: cannot write " + temporary.string());
        }
    }
    std::filesystem::rename(temporary, path);
}

void Diffusion::load(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("checkpoint: cannot open " + path.string());
    }
    std::vector<char> bytes(static_cast<size_t>(file.tellg()));
    file.seekg(0);
    file.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!file || bytes.size() < sizeof(uint64_t)) {
        throw std::runtime_error("checkpoint: cannot read " + path.string());
    }
    const size_t body = bytes.size() - sizeof(uint64_t);
    uint64_t stored = 0;
    std::memcpy(&stored, bytes.data() + body, sizeof(stored));
    if (stored != checksum(bytes.data(), body)) {
        throw std::runtime_error("checkpoint: the checksum differs, the file is damaged");
    }

    Reader r(bytes, body);
    if (r.get<uint32_t>() != kMagic) {
        throw std::runtime_error("checkpoint: not a diffusion checkpoint");
    }
    if (r.get<uint32_t>() != kVersion) {
        throw std::runtime_error("checkpoint: unsupported version");
    }
    if (r.get<int32_t>() != config_.timesteps || r.get<int32_t>() != config_.net.base ||
        r.get<int32_t>() != config_.net.groups) {
        throw std::runtime_error("checkpoint: made for a different architecture");
    }
    const int32_t adam_steps = r.get<int32_t>();
    const std::string dataset = r.get_text();
    const std::string rng_text = r.get_text();

    const std::vector<Param*> model = model_.params();
    const std::vector<Param*> ema = ema_.params();
    if (r.get<uint32_t>() != model.size()) {
        throw std::runtime_error("checkpoint: made for a different architecture");
    }

    // Everything is read into memory before anything is applied, so a failure leaves this object as it was.
    struct Saved {
        std::vector<float> value;
        std::vector<float> ema;
        std::vector<float> m;
        std::vector<float> v;
    };
    std::vector<Saved> saved(model.size());
    for (size_t i = 0; i < model.size(); ++i) {
        if (r.get<uint64_t>() != model[i]->value.size()) {
            throw std::runtime_error("checkpoint: made for a different architecture");
        }
        const size_t count = model[i]->value.size();
        saved[i].value = r.get_floats(count);
        saved[i].ema = r.get_floats(count);
        if (r.get<uint8_t>() != 0) {
            saved[i].m = r.get_floats(count);
            saved[i].v = r.get_floats(count);
        }
    }
    std::mt19937 rng;
    std::istringstream rng_stream(rng_text);
    if (!(rng_stream >> rng)) {
        throw std::runtime_error("checkpoint: the random generator state is damaged");
    }

    for (size_t i = 0; i < model.size(); ++i) {
        model[i]->value.upload(saved[i].value);
        ema[i]->value.upload(saved[i].ema);
        if (saved[i].m.empty()) {
            // No moments yet: Adam creates them at its next step.
            model[i]->m.resize({0, 0, 0, 0});
            model[i]->v.resize({0, 0, 0, 0});
        } else {
            model[i]->m.resize(model[i]->value.shape());
            model[i]->m.upload(saved[i].m);
            model[i]->v.resize(model[i]->value.shape());
            model[i]->v.upload(saved[i].v);
        }
    }
    adam_.set_steps(adam_steps);
    rng_ = rng;
    dataset_ = dataset;
}

}

namespace cuda {

CheckpointInfo peek_checkpoint(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("checkpoint: cannot open " + path.string());
    }
    const auto get = [&](auto& value) {
        file.read(reinterpret_cast<char*>(&value), sizeof(value));
        if (!file) {
            throw std::runtime_error("checkpoint " + path.string() + ": the file ends early");
        }
    };
    uint32_t magic = 0;
    uint32_t version = 0;
    get(magic);
    get(version);
    if (magic != kMagic) {
        throw std::runtime_error("checkpoint " + path.string() + ": not a diffusion checkpoint");
    }
    if (version != kVersion) {
        throw std::runtime_error("checkpoint " + path.string() + ": unsupported version");
    }
    CheckpointInfo info;
    int32_t value = 0;
    get(value);
    info.timesteps = value;
    get(value);
    info.base = value;
    get(value);
    info.groups = value;
    get(value);
    info.steps = value;
    uint32_t size = 0;
    get(size);
    if (size > kMaxName) {
        throw std::runtime_error("checkpoint " + path.string() + ": the dataset name is too long");
    }
    info.dataset.resize(size);
    file.read(info.dataset.data(), size);
    if (!file) {
        throw std::runtime_error("checkpoint " + path.string() + ": the file ends early");
    }
    return info;
}

DiffusionConfig config_of(const CheckpointInfo& info) {
    DiffusionConfig config;
    config.timesteps = info.timesteps;
    config.net = {.base = info.base, .groups = info.groups};
    return config;
}

DiffusionConfig config_of(const std::filesystem::path& checkpoint) {
    return config_of(peek_checkpoint(checkpoint));
}

}
