#include "inspect/copy_check.h"

#include "cuda/probe.h"
#include "inspect/nearest_sprite.h"
#include "data/sprite_set.h"

#include <algorithm>
#include <numeric>
#include <random>

namespace inspect {

namespace {

constexpr int kBaselineQueries = 300;
constexpr int kBatch = 64;
constexpr int kSteps = 30;

double percentile(std::vector<double> values, double share) {
    std::sort(values.begin(), values.end());
    const size_t at = std::min(values.size() - 1, static_cast<size_t>(share * static_cast<double>(values.size())));
    return values[at];
}

}

CopyCheck::CopyCheck(std::filesystem::path checkpoint, std::vector<float> dataset, int samples)
    : checkpoint_(std::move(checkpoint))
    , dataset_(std::move(dataset))
    , samples_(samples)
    , job_([this](const std::atomic<bool>& cancel) { return check(cancel); }) {}

CopyCheckResult CopyCheck::check(const std::atomic<bool>& cancel) {
    CopyCheckResult result;
    const size_t count = dataset_.size() / data::kSpriteValues;
    if (count < 2) {
        throw std::runtime_error("The dataset has too few sprites to compare with");
    }
    std::random_device device;
    std::mt19937 rng(device());

    // How close the dataset's own sprites are to their neighbours: the yardstick.
    std::vector<size_t> own(count);
    std::iota(own.begin(), own.end(), size_t{0});
    std::shuffle(own.begin(), own.end(), rng);
    own.resize(std::min<size_t>(kBaselineQueries, count));
    std::vector<float> queries;
    for (const size_t index : own) {
        const std::vector<float> sprite = dataset_sprite(dataset_, index, false);
        queries.insert(queries.end(), sprite.begin(), sprite.end());
    }
    std::vector<double> baseline;
    for (const Neighbor& neighbor : nearest_sprites(queries, dataset_, own)) {
        baseline.push_back(neighbor.distance);
    }
    result.baseline = {percentile(baseline, 0.10), percentile(baseline, 0.50), percentile(baseline, 0.90)};
    progress_ = 0.3;
    if (cancel) {
        result.cancelled = true;
        progress_ = 1.0;
        return result;
    }

    std::vector<double> distances;
    cuda::Probe probe(checkpoint_);
    for (int done = 0; done < samples_ && !cancel; done += kBatch) {
        const int batch = std::min(kBatch, samples_ - done);
        const std::vector<float> sprites = probe.sample(batch, kSteps, device());
        for (const Neighbor& neighbor : nearest_sprites(sprites, dataset_)) {
            distances.push_back(neighbor.distance);
        }
        progress_ = 0.3 + 0.7 * static_cast<double>(done + batch) / samples_;
    }
    progress_ = 1.0;
    if (cancel) {
        result.cancelled = true;
        return result;
    }
    result.samples = distances.size();
    result.samples_spread = {percentile(distances, 0.10), percentile(distances, 0.50), percentile(distances, 0.90)};
    result.share_near = static_cast<double>(std::count_if(distances.begin(), distances.end(),
                                [&](double d) { return d <= result.baseline.median; })) /
                        static_cast<double>(distances.size());
    return result;
}

}