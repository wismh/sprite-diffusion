#include "cuda/trainer.h"

#include <chrono>
#include <utility>

namespace cuda {

Trainer::Trainer(Diffusion& diffusion, TrainerConfig config, BatchSource batches)
    : diffusion_(diffusion), config_(std::move(config)), batches_(std::move(batches)) {
    status_.step = diffusion_.steps_done();
}

Trainer::~Trainer() {
    stop();
}

void Trainer::start() {
    {
        std::lock_guard lock(mutex_);
        if (status_.state == TrainerStatus::State::Running || status_.state == TrainerStatus::State::Paused) {
            return;
        }
    }
    if (thread_.joinable()) {
        thread_.join();
    }
    {
        std::lock_guard lock(mutex_);
        stop_ = false;
        paused_ = false;
        status_.state = TrainerStatus::State::Running;
        status_.error.clear();
    }
    thread_ = std::thread([this] { run(); });
}

void Trainer::pause() {
    std::lock_guard lock(mutex_);
    if (status_.state == TrainerStatus::State::Running) {
        paused_ = true;
        status_.state = TrainerStatus::State::Paused;
    }
}

void Trainer::resume() {
    {
        std::lock_guard lock(mutex_);
        if (status_.state != TrainerStatus::State::Paused) {
            return;
        }
        paused_ = false;
        status_.state = TrainerStatus::State::Running;
    }
    wake_.notify_all();
}

void Trainer::stop() {
    {
        std::lock_guard lock(mutex_);
        stop_ = true;
    }
    wake_.notify_all();
    if (thread_.joinable()) {
        thread_.join();
    }
}

void Trainer::request_preview() {
    {
        std::lock_guard lock(mutex_);
        preview_requested_ = true;
    }
    wake_.notify_all();
}

void Trainer::request_checkpoint() {
    {
        std::lock_guard lock(mutex_);
        checkpoint_requested_ = true;
    }
    wake_.notify_all();
}

TrainerStatus Trainer::status() const {
    std::lock_guard lock(mutex_);
    return status_;
}

std::vector<float> Trainer::loss_history() const {
    std::lock_guard lock(mutex_);
    return history_;
}

std::optional<Preview> Trainer::take_preview() {
    std::lock_guard lock(mutex_);
    return std::exchange(preview_, std::nullopt);
}

void Trainer::make_preview() {
    Preview preview;
    preview.step = diffusion_.steps_done();
    preview.sprites = diffusion_.sample(config_.preview_count, config_.preview_steps, static_cast<unsigned>(preview.step));
    std::lock_guard lock(mutex_);
    preview_ = std::move(preview);
}

void Trainer::save_checkpoint() {
    if (config_.checkpoint_path.empty()) {
        return;
    }
    diffusion_.save(config_.checkpoint_path);
    std::lock_guard lock(mutex_);
    ++status_.checkpoints;
}

void Trainer::run() {
    try {
        int ran = 0;
        double block_sum = 0.0;
        int block_count = 0;
        bool first_loss = true;
        for (;;) {
            bool paused = false;
            bool want_preview = false;
            bool want_checkpoint = false;
            {
                std::unique_lock lock(mutex_);
                wake_.wait(lock, [this] { return stop_ || !paused_ || preview_requested_ || checkpoint_requested_; });
                if (stop_) {
                    break;
                }
                paused = paused_;
                want_preview = std::exchange(preview_requested_, false);
                want_checkpoint = std::exchange(checkpoint_requested_, false);
            }
            if (want_preview) {
                make_preview();
            }
            if (want_checkpoint) {
                save_checkpoint();
            }
            if (paused) {
                continue;
            }
            if (config_.max_steps > 0 && ran >= config_.max_steps) {
                break;
            }

            const double loss = diffusion_.train_step(batches_(config_.batch));
            ++ran;
            const int step = diffusion_.steps_done();
            block_sum += loss;
            if (++block_count >= config_.history_block) {
                std::lock_guard lock(mutex_);
                history_.push_back(static_cast<float>(block_sum / block_count));
                block_sum = 0.0;
                block_count = 0;
            }
            {
                std::lock_guard lock(mutex_);
                status_.step = step;
                status_.loss = first_loss ? loss : 0.98 * status_.loss + 0.02 * loss;
                first_loss = false;
            }
            if (config_.preview_every > 0 && ran % config_.preview_every == 0) {
                make_preview();
            }
            if (config_.checkpoint_every > 0 && ran % config_.checkpoint_every == 0) {
                save_checkpoint();
            }
            if (config_.throttle_ms > 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(config_.throttle_ms));
            }
        }
        if (ran > 0) {
            save_checkpoint();
        }
        std::lock_guard lock(mutex_);
        status_.state = stop_ ? TrainerStatus::State::Idle : TrainerStatus::State::Finished;
    } catch (const std::exception& error) {
        std::lock_guard lock(mutex_);
        status_.error = error.what();
        status_.state = TrainerStatus::State::Failed;
    }
}

}
