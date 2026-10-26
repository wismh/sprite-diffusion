#pragma once

#include "data/background_job.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <string>
#include <vector>

namespace inspect {

struct GenerateRequest {
    // The saved model to draw with: its checkpoint, read on the generating thread (the weights as last saved).
    std::filesystem::path checkpoint;
    // Where the pictures go (made if it is not there) and the start of their names: <prefix>_00001.png, ...
    std::filesystem::path directory;
    std::string prefix;
    int count = 0;
    // DDIM steps per sprite.
    int steps = 50;
    // A pixel that is close to magenta (the background of imported datasets) is written transparent.
    bool transparent = true;
};

struct GenerateResult : data::JobOutcome {
    std::size_t written = 0;
};

// Draws `count` sprites with a saved model on a thread of its own and writes each as a 16x16 PNG. Cancelled or destroyed
// before it ends, it stops after the batch in hand; pictures already written stay.
class SpriteGenerator {
public:
    explicit SpriteGenerator(GenerateRequest request);

    void cancel() { job_.cancel(); }
    [[nodiscard]] bool finished() const { return job_.finished(); }
    [[nodiscard]] std::size_t done() const { return done_; }
    [[nodiscard]] std::size_t total() const { return static_cast<std::size_t>(request_.count); }
    [[nodiscard]] const std::filesystem::path& directory() const { return request_.directory; }
    // Once finished().
    [[nodiscard]] GenerateResult take_result() { return job_.take_result(); }
    // The sprites of the newest batch (up to 32, NCHW, [-1, 1]), for a look at what is being made.
    [[nodiscard]] std::vector<float> latest() const;

    // A sprite (3 * 16 * 16 values in [-1, 1]) as RGBA bytes; `transparent` clears pixels near magenta.
    [[nodiscard]] static std::vector<std::uint8_t> to_rgba(const float* sprite, bool transparent);

private:
    [[nodiscard]] GenerateResult generate(const std::atomic<bool>& cancel);

    GenerateRequest request_;
    std::atomic<std::size_t> done_{0};
    mutable std::mutex mutex_;
    std::vector<float> latest_;
    data::BackgroundJob<GenerateResult> job_;  // last: it ends first, then what it reads
};

}
