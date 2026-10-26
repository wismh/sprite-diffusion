#pragma once

#include "cuda/diffusion.h"
#include "cuda/unet.h"

#include <array>
#include <condition_variable>
#include <cstddef>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace cuda {

// Works out, in the background, the sprite each channel of a stage answers to hardest: starting from noise, the sprite is
// pushed, step after step, toward a stronger answer of that channel (gradient ascent through the network at a fixed
// noise level). The units of a stage are worked on together, a batch at a time, once somebody asks for that stage; asking
// for another stage (or another noise level) replaces the work in progress, and what was finished stays. Uses its own copy
// of the weights, so it does not meet the thread that looks at the same model. Everything but the thread is for the main
// thread.
class DreamWorker {
public:
    // Throws std::runtime_error when the checkpoint has no readable header.
    explicit DreamWorker(const std::filesystem::path& checkpoint);
    ~DreamWorker();
    DreamWorker(const DreamWorker&) = delete;
    DreamWorker& operator=(const DreamWorker&) = delete;

    // Works on the units of `stage` that have no picture, at timestep `t`. A different `t` throws away every picture.
    void ask(Stage stage, int t);
    // Takes in the pictures that have finished since the last call. Returns whether there were any.
    bool collect();

    // A sprite (NCHW, [-1, 1], kSpriteValues values) once it is ready.
    [[nodiscard]] const std::optional<std::vector<float>>& picture(Stage stage, int unit) const {
        return pictures_[static_cast<size_t>(stage)][static_cast<size_t>(unit)];
    }
    [[nodiscard]] size_t done(Stage stage) const { return done_[static_cast<size_t>(stage)]; }
    [[nodiscard]] size_t total(Stage stage) const;
    // The noise level of the pictures that are there (-1 before the first ask()).
    [[nodiscard]] int timestep() const { return pictures_t_; }
    [[nodiscard]] bool asked(Stage stage) const { return asked_[static_cast<size_t>(stage)]; }
    // Empty, or why the weights could not be read.
    [[nodiscard]] std::string error() const;

    // The number of gradient steps per sprite and the batch of sprites worked on together.
    static constexpr int kIterations = 100;
    static constexpr int kBatch = 32;

private:
    struct Request {
        Stage stage = Stage::ConvIn;
        int t = 0;
        unsigned epoch = 0;  // changes with t, which makes finished pictures of the other t useless
        bool wanted = false;
    };
    struct Finished {
        Stage stage;
        int unit;
        unsigned epoch;
        std::vector<float> sprite;
    };

    void run();
    void work(Diffusion& diffusion, const Request& request, std::vector<bool>& made);

    std::filesystem::path checkpoint_;
    DiffusionConfig config_;

    std::array<std::vector<std::optional<std::vector<float>>>, kStageCount> pictures_;
    std::array<size_t, kStageCount> done_{};
    std::array<bool, kStageCount> asked_{};
    int pictures_t_ = -1;
    unsigned epoch_ = 0;

    mutable std::mutex mutex_;
    std::condition_variable wake_;
    Request request_;
    std::vector<Finished> finished_;
    std::string error_;
    bool quit_ = false;
    std::thread thread_;
};

}
