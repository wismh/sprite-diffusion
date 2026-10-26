#include "data/background_job.h"

#include "test_support.h"

#include <atomic>
#include <stdexcept>

namespace {

struct Counted : data::JobOutcome {
    int value = 0;
};

}

TEST(BackgroundJob, HandsOverTheResultOnceAndLeavesAnEmptyOneAfter) {
    data::BackgroundJob<Counted> job([](const std::atomic<bool>&) {
        Counted result;
        result.value = 7;
        return result;
    });
    ASSERT_TRUE(test::wait_for([&] { return job.finished(); }));
    const Counted first = job.take_result();
    EXPECT_TRUE(first.ok());
    EXPECT_EQ(first.value, 7);
    EXPECT_EQ(job.take_result().value, 0);
}

TEST(BackgroundJob, TurnsAnExceptionIntoTheError) {
    data::BackgroundJob<Counted> job([](const std::atomic<bool>&) -> Counted { throw std::runtime_error("no luck"); });
    ASSERT_TRUE(test::wait_for([&] { return job.finished(); }));
    const Counted result = job.take_result();
    EXPECT_FALSE(result.ok());
    EXPECT_FALSE(result.cancelled);
    EXPECT_EQ(result.error, "no luck");
}

TEST(BackgroundJob, CancelIsSeenByTheWork) {
    std::atomic<bool> started{false};
    data::BackgroundJob<Counted> job([&](const std::atomic<bool>& cancel) {
        started = true;
        while (!cancel) {
            std::this_thread::yield();
        }
        Counted result;
        result.cancelled = true;
        return result;
    });
    ASSERT_TRUE(test::wait_for([&] { return started.load(); }));
    EXPECT_FALSE(job.finished());
    job.cancel();
    ASSERT_TRUE(test::wait_for([&] { return job.finished(); }));
    const Counted result = job.take_result();
    EXPECT_TRUE(result.cancelled);
    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.error, "");
}

TEST(BackgroundJob, DestroyingItCancelsAndWaits) {
    std::atomic<bool> ended{false};
    {
        data::BackgroundJob<Counted> job([&](const std::atomic<bool>& cancel) {
            while (!cancel) {
                std::this_thread::yield();
            }
            ended = true;
            return Counted{};
        });
    }
    EXPECT_TRUE(ended);
}