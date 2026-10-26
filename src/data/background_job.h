#pragma once

#include <atomic>
#include <exception>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <utility>

namespace data {

// What every background job hands back: `cancelled` when it was stopped (whatever it had made stays), `error` when it
// failed. Both are clear when it ran to its end. A job's result type derives from this.
struct JobOutcome {
    bool cancelled = false;
    std::string error;

    [[nodiscard]] bool ok() const { return !cancelled && error.empty(); }
};

// Runs `work` on a thread of its own, starting at once. `work` gets the cancel flag and returns the result; an exception
// it lets out becomes the result's `error`. Destroying the job cancels it and waits, so declare it after everything the
// work reads. Everything but the thread is for one other thread (the main one).
template <class Result>
class BackgroundJob {
public:
    using Work = std::function<Result(const std::atomic<bool>& cancel)>;

    explicit BackgroundJob(Work work)
        : work_(std::move(work))
        , thread_([this] { run(); }) {}

    ~BackgroundJob() {
        cancel();
        thread_.join();
    }

    BackgroundJob(const BackgroundJob&) = delete;
    BackgroundJob& operator=(const BackgroundJob&) = delete;

    void cancel() { cancel_ = true; }
    [[nodiscard]] bool finished() const { return finished_; }
    // Once finished(); the result is handed over, a second call gets an empty one.
    [[nodiscard]] Result take_result() {
        std::lock_guard lock(mutex_);
        return std::exchange(result_, Result{});
    }

private:
    void run() {
        Result result;
        try {
            result = work_(cancel_);
        } catch (const std::exception& error) {
            result.error = error.what();
        }
        {
            std::lock_guard lock(mutex_);
            result_ = std::move(result);
        }
        finished_ = true;
    }

    Work work_;
    std::atomic<bool> cancel_{false};
    std::atomic<bool> finished_{false};
    std::mutex mutex_;
    Result result_;
    std::thread thread_;
};

}
