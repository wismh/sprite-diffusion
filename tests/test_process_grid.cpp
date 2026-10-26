#include "data/sprite_set.h"
#include "inspect/process_grid.h"

#include <gtest/gtest.h>

#include <algorithm>

TEST(ProcessGrid, ShownStepsAreSpreadFromTheFirstToTheLast) {
    const std::vector<int> shown = inspect::shown_steps(30, 8);
    ASSERT_EQ(shown.size(), 8u);
    EXPECT_EQ(shown.front(), 0);
    EXPECT_EQ(shown.back(), 29);
    EXPECT_TRUE(std::is_sorted(shown.begin(), shown.end()));
    EXPECT_EQ(std::adjacent_find(shown.begin(), shown.end()), shown.end());
}

TEST(ProcessGrid, ShownStepsHandleTheEdges) {
    EXPECT_TRUE(inspect::shown_steps(0, 8).empty());
    EXPECT_TRUE(inspect::shown_steps(30, 0).empty());
    EXPECT_EQ(inspect::shown_steps(30, 1), (std::vector<int>{0}));
    EXPECT_EQ(inspect::shown_steps(5, 5), (std::vector<int>{0, 1, 2, 3, 4}));
}

TEST(ProcessGrid, RowsHoldTheShownStepsOfASampleThenItsClosestDatasetSprite) {
    constexpr std::size_t kValues = data::kSpriteValues;
    // Two samples, three steps: the sprite of sample s at step t is filled with 10 * t + s.
    std::vector<std::vector<float>> steps(3);
    for (int t = 0; t < 3; ++t) {
        for (int s = 0; s < 2; ++s) {
            steps[static_cast<std::size_t>(t)].insert(
                    steps[static_cast<std::size_t>(t)].end(), kValues, static_cast<float>(10 * t + s));
        }
    }
    // A dataset of two sprites, filled with 100 and 200; sample 0 is closest to sprite 1, sample 1 to sprite 0.
    std::vector<float> dataset(2 * kValues, 100.0f);
    std::fill(dataset.begin() + static_cast<std::ptrdiff_t>(kValues), dataset.end(), 200.0f);
    const std::vector<inspect::Neighbor> neighbors{{1, false, 0.0}, {0, false, 0.0}};

    const std::vector<float> grid = inspect::process_grid(steps, {0, 2}, dataset, neighbors);
    ASSERT_EQ(grid.size(), 2u * 3u * kValues);
    const auto at = [&](std::size_t row, std::size_t column) { return grid[(row * 3 + column) * kValues]; };
    EXPECT_FLOAT_EQ(at(0, 0), 0.0f);    // step 0, sample 0
    EXPECT_FLOAT_EQ(at(0, 1), 20.0f);   // step 2, sample 0
    EXPECT_FLOAT_EQ(at(0, 2), 200.0f);  // closest: sprite 1
    EXPECT_FLOAT_EQ(at(1, 0), 1.0f);
    EXPECT_FLOAT_EQ(at(1, 1), 21.0f);
    EXPECT_FLOAT_EQ(at(1, 2), 100.0f);
}