#pragma once

#include "data/background_job.h"

#include <atomic>
#include <cstddef>
#include <filesystem>
#include <vector>

namespace inspect {

struct CopyCheckResult : data::JobOutcome {
    size_t samples = 0;
    // How far a sprite of the dataset is from its nearest other sprite, and how far a generated one is from its nearest
    // sprite of the dataset: the 10th, 50th and 90th percentiles. Root mean square per value, [-1, 1] units. A dataset with
    // many near-duplicates (a sprite in several colors) has small baseline distances, so the baseline is a yardstick of
    // "as close as the data is to itself", not of an exact copy.
    struct Spread {
        double low = 0.0;
        double median = 0.0;
        double high = 0.0;
    };
    Spread baseline;
    Spread samples_spread;
    // The part (0..1) of the generated sprites that are as close to the dataset as the typical (median) dataset sprite is to
    // its nearest neighbour. About one half for a model whose sprites sit among the dataset's as the dataset's own do;
    // more for one that stays close to what it learned.
    double share_near = 0.0;
};

// Tells whether a model copies its training sprites: draws sprites with it and measures how close each is to the
// dataset, against how close the dataset's own sprites are to each other. Runs on a thread of its own.
class CopyCheck {
public:
    // `dataset` is the sprites the model learned from (NCHW, [-1, 1]).
    CopyCheck(std::filesystem::path checkpoint, std::vector<float> dataset, int samples);

    void cancel() { job_.cancel(); }
    [[nodiscard]] bool finished() const { return job_.finished(); }
    // 0..1.
    [[nodiscard]] double progress() const { return progress_; }
    // Once finished().
    [[nodiscard]] CopyCheckResult take_result() { return job_.take_result(); }

private:
    [[nodiscard]] CopyCheckResult check(const std::atomic<bool>& cancel);

    std::filesystem::path checkpoint_;
    std::vector<float> dataset_;
    int samples_;
    std::atomic<double> progress_{0.0};
    data::BackgroundJob<CopyCheckResult> job_;  // last: it ends first, then what it reads
};

}
