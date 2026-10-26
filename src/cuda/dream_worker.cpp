#include "cuda/dream_worker.h"

#include <algorithm>
#include <cmath>
#include <random>

namespace cuda {

namespace {

constexpr float kStep = 0.05f;   // length of one step, in units of the root mean square of the gradient
constexpr float kDecay = 0.01f;  // how much of the sprite is pulled back toward gray each step
constexpr float kLimit = 1.5f;   // the sprite stays within this
constexpr float kStart = 0.5f;   // the standard deviation of the noise it starts from
constexpr float kBlur = 0.4f;    // how much of each pixel is replaced by the mean of its 3x3 neighbourhood each step
constexpr int kBlurUntil = 80;   // the steps that blur; the last ones sharpen what is left

// Pulls every pixel of a sprite (3 planes of 16x16) toward the mean of its neighbourhood, which keeps the high frequencies
// that gradient steps pile up from taking over the picture.
void blur(float* sprite) {
    constexpr int kSide = kSpriteSize;
    float copy[kSide * kSide];
    for (int c = 0; c < kSpriteChannels; ++c) {
        float* plane = sprite + c * kSide * kSide;
        std::copy(plane, plane + kSide * kSide, copy);
        for (int y = 0; y < kSide; ++y) {
            for (int x = 0; x < kSide; ++x) {
                float sum = 0.0f;
                int count = 0;
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        const int yy = y + dy;
                        const int xx = x + dx;
                        if (yy >= 0 && yy < kSide && xx >= 0 && xx < kSide) {
                            sum += copy[yy * kSide + xx];
                            ++count;
                        }
                    }
                }
                plane[y * kSide + x] += kBlur * (sum / static_cast<float>(count) - plane[y * kSide + x]);
            }
        }
    }
}

}

DreamWorker::DreamWorker(const std::filesystem::path& checkpoint)
    : checkpoint_(checkpoint)
    , config_(config_of(checkpoint)) {
    for (int i = 0; i < kStageCount; ++i) {
        pictures_[static_cast<size_t>(i)].resize(static_cast<size_t>(stage_info(config_.net, static_cast<Stage>(i)).channels));
    }
    thread_ = std::thread([this] { run(); });
}

DreamWorker::~DreamWorker() {
    {
        std::lock_guard lock(mutex_);
        quit_ = true;
    }
    wake_.notify_all();
    thread_.join();
}

size_t DreamWorker::total(Stage stage) const {
    return static_cast<size_t>(stage_info(config_.net, stage).channels);
}

std::string DreamWorker::error() const {
    std::lock_guard lock(mutex_);
    return error_;
}

void DreamWorker::ask(Stage stage, int t) {
    t = std::clamp(t, 0, config_.timesteps - 1);
    if (t != pictures_t_) {
        for (auto& stage_pictures : pictures_) {
            std::fill(stage_pictures.begin(), stage_pictures.end(), std::nullopt);
        }
        done_.fill(0);
        asked_.fill(false);
        pictures_t_ = t;
        ++epoch_;
    }
    asked_[static_cast<size_t>(stage)] = true;
    {
        std::lock_guard lock(mutex_);
        request_ = {.stage = stage, .t = t, .epoch = epoch_, .wanted = true};
    }
    wake_.notify_all();
}

bool DreamWorker::collect() {
    std::vector<Finished> taken;
    {
        std::lock_guard lock(mutex_);
        taken.swap(finished_);
    }
    bool any = false;
    for (Finished& item : taken) {
        if (item.epoch != epoch_) {
            continue;
        }
        auto& slot = pictures_[static_cast<size_t>(item.stage)][static_cast<size_t>(item.unit)];
        if (!slot) {
            ++done_[static_cast<size_t>(item.stage)];
        }
        slot = std::move(item.sprite);
        any = true;
    }
    return any;
}

void DreamWorker::run() {
    std::optional<Diffusion> diffusion;
    try {
        diffusion.emplace(config_, 1);
        diffusion->load(checkpoint_);
    } catch (const std::exception& error) {
        std::lock_guard lock(mutex_);
        error_ = error.what();
        return;
    }
    // Which units of each stage have been worked out at the epoch the worker is on.
    std::array<std::vector<bool>, kStageCount> made;
    unsigned made_epoch = ~0u;
    for (;;) {
        Request request;
        {
            std::unique_lock lock(mutex_);
            wake_.wait(lock, [this] { return quit_ || request_.wanted; });
            if (quit_) {
                return;
            }
            request = request_;
        }
        if (request.epoch != made_epoch) {
            made_epoch = request.epoch;
            for (auto& flags : made) {
                flags.clear();
            }
        }
        std::vector<bool>& flags = made[static_cast<size_t>(request.stage)];
        flags.resize(total(request.stage), false);
        try {
            work(*diffusion, request, flags);
        } catch (const std::exception& error) {
            std::lock_guard lock(mutex_);
            error_ = error.what();
            request_.wanted = false;
            continue;
        }
        std::lock_guard lock(mutex_);
        // Done with this request, unless a newer one came while it ran.
        if (request_.stage == request.stage && request_.epoch == request.epoch) {
            request_.wanted = false;
        }
    }
}

void DreamWorker::work(Diffusion& diffusion, const Request& request, std::vector<bool>& made) {
    UNet& net = diffusion.ema();
    const StageInfo info = stage_info(config_.net, request.stage);
    const size_t plane = static_cast<size_t>(info.size) * info.size;
    std::mt19937 rng(static_cast<unsigned>(request.t) * 7919u + static_cast<unsigned>(request.stage));
    std::normal_distribution<float> normal(0.0f, kStart);

    const auto superseded = [&] {
        std::lock_guard lock(mutex_);
        return quit_ || request_.stage != request.stage || request_.epoch != request.epoch;
    };

    for (;;) {
        std::vector<int> units;
        for (int unit = 0; unit < info.channels && static_cast<int>(units.size()) < kBatch; ++unit) {
            if (!made[static_cast<size_t>(unit)]) {
                units.push_back(unit);
            }
        }
        if (units.empty() || superseded()) {
            return;
        }
        const int n = static_cast<int>(units.size());
        std::vector<float> x(static_cast<size_t>(n) * kSpriteValues);
        for (float& v : x) {
            v = normal(rng);
        }
        const std::vector<float> t(static_cast<size_t>(n), static_cast<float>(request.t));
        Tensor input({n, kSpriteChannels, kSpriteSize, kSpriteSize});
        Tensor target;
        for (int iteration = 0; iteration < kIterations; ++iteration) {
            input.upload(x);
            net.zero_grad();
            net.forward(input, t);
            if (iteration == 0) {
                // The answer of a unit is the mean of its map; every sprite of the batch pushes its own unit.
                const Shape shape = net.activation(request.stage).shape();
                std::vector<float> wanted(shape.count(), 0.0f);
                for (int i = 0; i < n; ++i) {
                    const size_t first = (static_cast<size_t>(i) * info.channels + static_cast<size_t>(units[i])) * plane;
                    std::fill_n(wanted.begin() + static_cast<std::ptrdiff_t>(first), plane,
                            1.0f / static_cast<float>(plane));
                }
                target.resize(shape);
                target.upload(wanted);
            }
            const std::vector<float> gradient = net.backward_from(input, request.stage, target).download();
            for (int i = 0; i < n; ++i) {
                const size_t first = static_cast<size_t>(i) * kSpriteValues;
                double sum = 0.0;
                for (size_t k = 0; k < kSpriteValues; ++k) {
                    sum += static_cast<double>(gradient[first + k]) * gradient[first + k];
                }
                const float rms = static_cast<float>(std::sqrt(sum / static_cast<double>(kSpriteValues))) + 1e-12f;
                for (size_t k = 0; k < kSpriteValues; ++k) {
                    float& v = x[first + k];
                    v = std::clamp((v + kStep * gradient[first + k] / rms) * (1.0f - kDecay), -kLimit, kLimit);
                }
                if (iteration < kBlurUntil) {
                    blur(x.data() + first);
                }
            }
            if ((iteration & 15) == 15 && superseded()) {
                return;
            }
        }
        {
            std::lock_guard lock(mutex_);
            for (int i = 0; i < n; ++i) {
                const auto first = x.begin() + static_cast<std::ptrdiff_t>(static_cast<size_t>(i) * kSpriteValues);
                finished_.push_back({request.stage, units[i], request.epoch,
                        std::vector<float>(first, first + static_cast<std::ptrdiff_t>(kSpriteValues))});
            }
        }
        for (const int unit : units) {
            made[static_cast<size_t>(unit)] = true;
        }
    }
}

}
