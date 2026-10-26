#pragma once

#include "cuda/diffusion.h"

#include <condition_variable>
#include <filesystem>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace cuda {

struct TrainerConfig {
    int batch = 64;
    int max_steps = 0;         // steps of this run (not counting earlier ones); 0 trains until stop()
    int preview_every = 500;   // a preview after this many steps; 0 only on request
    int preview_count = 16;
    int preview_steps = 30;    // DDIM steps of a preview
    int checkpoint_every = 0;  // 0: only on request and at the end
    std::filesystem::path checkpoint_path;  // empty: no checkpoints
    int throttle_ms = 0;       // a pause after every step, to leave the GPU to the game's own drawing
    int history_block = 10;    // steps averaged into one point of loss_history()
};

struct Preview {
    int step = 0;
    std::vector<float> sprites;  // preview_count sprites, [-1, 1], NCHW
};

struct TrainerStatus {
    enum class State { Idle, Running, Paused, Finished, Failed };
    State state = State::Idle;
    int step = 0;         // total steps of the Diffusion, including earlier runs
    double loss = 0.0;    // smoothed over about the last 50 steps
    int checkpoints = 0;  // saved by this trainer
    std::string error;    // what stopped a Failed run
};

// Trains a Diffusion on a thread of its own, so that the game stays responsive. While the trainer runs it owns the
// Diffusion and the GPU context: do not call the Diffusion, or any CUDA code, from other threads until stop() or
// until the state is Finished or Failed. start() and stop() are for the one thread that controls the trainer (the main
// one); the rest of the methods below are safe to call from any thread.
class Trainer {
public:
    using BatchSource = std::function<std::vector<float>(int count)>;

    // `batches` is called on the training thread.
    Trainer(Diffusion& diffusion, TrainerConfig config, BatchSource batches);
    ~Trainer();
    Trainer(const Trainer&) = delete;
    Trainer& operator=(const Trainer&) = delete;

    void start();    // does nothing while running; starts a new run after Finished, Failed or stop()
    void pause();    // after the current step
    void resume();
    void stop();     // after the current step; saves a checkpoint if a path is set; returns when the thread has ended

    void request_preview();     // between two steps
    void request_checkpoint();  // between two steps

    [[nodiscard]] TrainerStatus status() const;
    [[nodiscard]] std::vector<float> loss_history() const;
    // The newest preview that was not taken yet.
    [[nodiscard]] std::optional<Preview> take_preview();

private:
    void run();
    void make_preview();
    void save_checkpoint();

    Diffusion& diffusion_;
    TrainerConfig config_;
    BatchSource batches_;

    mutable std::mutex mutex_;
    std::condition_variable wake_;
    std::thread thread_;
    bool stop_ = false;
    bool paused_ = false;
    bool preview_requested_ = false;
    bool checkpoint_requested_ = false;
    TrainerStatus status_;
    std::vector<float> history_;
    std::optional<Preview> preview_;
};

}
