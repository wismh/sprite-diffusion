#include "inspect/channel_stats.h"

#include <gtest/gtest.h>

#include <vector>

TEST(ChannelStats, ScaleToStrongestMakesTheLargestOne) {
    std::vector<float> values{1.0f, 4.0f, 2.0f};
    inspect::scale_to_strongest(values);
    EXPECT_FLOAT_EQ(values[0], 0.25f);
    EXPECT_FLOAT_EQ(values[1], 1.0f);
    EXPECT_FLOAT_EQ(values[2], 0.5f);
}

TEST(ChannelStats, ScaleToStrongestKeepsZerosAndEmptyLists) {
    std::vector<float> zeros{0.0f, 0.0f};
    inspect::scale_to_strongest(zeros);
    EXPECT_EQ(zeros, (std::vector<float>{0.0f, 0.0f}));
    std::vector<float> none;
    inspect::scale_to_strongest(none);
    EXPECT_TRUE(none.empty());
}

TEST(ChannelStats, MeanMagnitudesAveragesTheAbsoluteValueOfEachMap) {
    // Two maps of three values.
    const std::vector<float> maps{1.0f, -3.0f, 2.0f, -1.0f, -1.0f, -1.0f};
    const std::vector<float> means = inspect::mean_magnitudes(maps.data(), 2, 3);
    ASSERT_EQ(means.size(), 2u);
    EXPECT_FLOAT_EQ(means[0], 2.0f);
    EXPECT_FLOAT_EQ(means[1], 1.0f);
}

TEST(ChannelStats, LargestMagnitudeIgnoresTheSign) {
    const std::vector<float> values{0.5f, -2.5f, 1.0f};
    EXPECT_FLOAT_EQ(inspect::largest_magnitude(values.data(), values.size()), 2.5f);
    EXPECT_FLOAT_EQ(inspect::largest_magnitude(values.data(), 0), 0.0f);
}

TEST(ChannelStats, FilterLengthsAreTheRootOfTheSumOfSquares) {
    // Two filters of four weights: (1, 2, 2, 0) is 3 long, (0, 0, 0, 4) is 4 long.
    const std::vector<float> weights{1.0f, 2.0f, 2.0f, 0.0f, 0.0f, 0.0f, 0.0f, 4.0f};
    const std::vector<float> lengths = inspect::filter_lengths(weights.data(), 2, 4);
    ASSERT_EQ(lengths.size(), 2u);
    EXPECT_FLOAT_EQ(lengths[0], 3.0f);
    EXPECT_FLOAT_EQ(lengths[1], 4.0f);
}