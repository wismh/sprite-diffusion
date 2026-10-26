#include "inspect/loss_chart.h"

#include <gtest/gtest.h>

#include <vector>

TEST(LossChart, BucketMeansAveragesNeighboursAndKeepsShortHistories) {
    const std::vector<float> history{1.0f, 3.0f, 5.0f, 7.0f};
    EXPECT_EQ(inspect::bucket_means(history, 2), (std::vector<float>{2.0f, 6.0f}));
    EXPECT_EQ(inspect::bucket_means(history, 10), history);
    EXPECT_TRUE(inspect::bucket_means({}, 5).empty());
}

TEST(LossChart, BucketMeansCoversEveryValueOnce) {
    std::vector<float> history(10, 1.0f);
    history.back() = 11.0f;
    const std::vector<float> buckets = inspect::bucket_means(history, 3);
    ASSERT_EQ(buckets.size(), 3u);
    float total = 0.0f;
    for (const float value : buckets) {
        total += value;
    }
    EXPECT_GT(buckets.back(), buckets.front());
    EXPECT_GT(total, 3.0f);
}

TEST(LossChart, LogAxisRunsFromBottomToTopAndBackToValues) {
    const inspect::LogAxis axis(0.01f, 1.0f);
    EXPECT_LT(axis.position(0.01f), 0.2f);
    EXPECT_GT(axis.position(1.0f), 0.8f);
    EXPECT_LT(axis.position(0.01f), axis.position(0.1f));
    EXPECT_LT(axis.position(0.1f), axis.position(1.0f));
    // Equal ratios are equal steps: that is what makes the axis logarithmic.
    EXPECT_NEAR(axis.position(0.1f) - axis.position(0.01f), axis.position(1.0f) - axis.position(0.1f), 1e-4f);
    EXPECT_NEAR(axis.position(axis.value_at(0.3f)), 0.3f, 1e-4f);
}

TEST(LossChart, LogAxisCutsValuesOutsideTheRangeAndSurvivesZero) {
    const inspect::LogAxis axis(0.5f, 2.0f);
    EXPECT_FLOAT_EQ(axis.position(1e-9f), 0.0f);
    EXPECT_FLOAT_EQ(axis.position(1e9f), 1.0f);
    const inspect::LogAxis flat(0.0f, 0.0f);
    EXPECT_GE(flat.position(0.0f), 0.0f);
    EXPECT_LE(flat.position(0.0f), 1.0f);
}