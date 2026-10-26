#include "inspect/cell_image.h"

#include <gtest/gtest.h>

#include <vector>

TEST(CellImage, SignedImageDividesByTheScaleAndCutsToOne) {
    const std::vector<float> map{0.5f, -0.5f, 2.0f, -4.0f};
    const inspect::CellImage image = inspect::signed_image(map.data(), 2, 1.0f);
    EXPECT_EQ(image.kind, inspect::CellImage::Kind::Signed);
    EXPECT_EQ(image.width, 2);
    EXPECT_EQ(image.height, 2);
    EXPECT_EQ(image.values, (std::vector<float>{0.5f, -0.5f, 1.0f, -1.0f}));
}

TEST(CellImage, SignedImageWithoutAScaleIsAllZeros) {
    const std::vector<float> map{3.0f, -3.0f, 1.0f, 2.0f};
    EXPECT_EQ(inspect::signed_image(map.data(), 2, 0.0f).values, (std::vector<float>(4, 0.0f)));
}

TEST(CellImage, SpriteImageMapsMinusOneToOneToZeroToOneAndKeepsTheChannelsApart) {
    std::vector<float> sprite(3 * 16 * 16, 0.0f);
    sprite[0] = 1.0f;            // red of the first pixel
    sprite[256] = -1.0f;         // green of the first pixel
    sprite[512] = 0.0f;          // blue of the first pixel
    const inspect::CellImage image = inspect::sprite_image(sprite.data());
    EXPECT_EQ(image.kind, inspect::CellImage::Kind::Color);
    ASSERT_EQ(image.values.size(), 16u * 16u * 3u);
    EXPECT_FLOAT_EQ(image.values[0], 1.0f);
    EXPECT_FLOAT_EQ(image.values[1], 0.0f);
    EXPECT_FLOAT_EQ(image.values[2], 0.5f);
    EXPECT_FLOAT_EQ(image.values[3], 0.5f);  // the next pixel is gray
}

TEST(CellImage, SpriteImageGainAndOffsetAreApplied) {
    std::vector<float> sprite(3 * 16 * 16, 1.0f);
    const inspect::CellImage image = inspect::sprite_image(sprite.data(), 0.25f, 0.5f);
    EXPECT_FLOAT_EQ(image.values[0], 0.75f);
}

TEST(CellImage, FilterOfThreeInputsIsOneColorTile) {
    std::vector<float> weights(3 * 9, 0.0f);
    weights[0] = 1.0f;  // red of the top left tap
    const inspect::CellImage image = inspect::filter_image(weights.data(), 3, 1.0f);
    EXPECT_EQ(image.kind, inspect::CellImage::Kind::Color);
    EXPECT_EQ(image.width, 3);
    EXPECT_EQ(image.height, 3);
    EXPECT_FLOAT_EQ(image.values[0], 1.0f);
    EXPECT_FLOAT_EQ(image.values[1], 0.5f);
}

TEST(CellImage, FilterOfManyInputsIsASheetWithATilePerInputChannel) {
    const std::vector<float> weights(4 * 9, 0.5f);
    const inspect::CellImage image = inspect::filter_image(weights.data(), 4, 1.0f);
    EXPECT_EQ(image.kind, inspect::CellImage::Kind::Signed);
    // 2 x 2 tiles of 3 x 3, one pixel apart.
    EXPECT_EQ(image.width, 7);
    EXPECT_EQ(image.height, 7);
    EXPECT_FLOAT_EQ(image.values[0], 0.5f);
    EXPECT_FLOAT_EQ(image.values[3], 0.0f);  // the gap between two tiles
}

TEST(CellImage, FilterWithoutAScaleIsBlank) {
    const std::vector<float> weights(4 * 9, 0.5f);
    const inspect::CellImage image = inspect::filter_image(weights.data(), 4, 0.0f);
    for (const float value : image.values) {
        EXPECT_FLOAT_EQ(value, 0.0f);
    }
}