#pragma once

#include "data/background_job.h"
#include "data/dataset_file.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <filesystem>

namespace data {

struct ImportResult : JobOutcome {
    Dataset dataset;
    std::size_t skipped = 0;  // files that are not 16x16 pictures
};

// Reads every picture of a folder (and its subfolders) into a Dataset on a thread of its own. Only 16x16 pictures are
// taken; PNG, BMP, TGA and JPEG are read. A pixel with alpha of 128 or more keeps its color, any other becomes the
// background color: the sprites of the set are opaque and crisp, and one background marks what was transparent.
class SpriteImporter {
public:
    // Starts at once. The folder is not scanned on the calling thread.
    SpriteImporter(std::filesystem::path folder, std::array<std::uint8_t, 3> background);

    void cancel() { job_.cancel(); }
    [[nodiscard]] std::size_t done() const { return done_; }
    // Known once the folder is scanned; 0 before that.
    [[nodiscard]] std::size_t total() const { return total_; }
    [[nodiscard]] bool finished() const { return job_.finished(); }
    // After finished(): the result, once.
    [[nodiscard]] ImportResult take_result() { return job_.take_result(); }

private:
    [[nodiscard]] ImportResult import(const std::atomic<bool>& cancel);

    std::filesystem::path folder_;
    std::array<std::uint8_t, 3> background_;
    std::atomic<std::size_t> done_{0};
    std::atomic<std::size_t> total_{0};
    BackgroundJob<ImportResult> job_;  // last: it ends first, then what it reads
};

}
