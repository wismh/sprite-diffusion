#include "data/dataset_file.h"
#include "data/dataset_store.h"
#include "data/file_names.h"
#include "data/png_writer.h"
#include "data/sprite_importer.h"
#include "data/sprite_set.h"

#include "test_support.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <thread>

namespace {

// A 16x16 RGBA picture: `color` where x >= edge, `other` (with `other_alpha`) where x < edge.
void write_sprite(const std::filesystem::path& path, std::array<std::uint8_t, 3> color, int edge = 0,
                  std::uint8_t other_alpha = 0, std::array<std::uint8_t, 3> other = {200, 200, 200}) {
    std::vector<std::uint8_t> rgba;
    for (int y = 0; y < 16; ++y) {
        for (int x = 0; x < 16; ++x) {
            const bool left = x < edge;
            const auto& rgb = left ? other : color;
            rgba.insert(rgba.end(), {rgb[0], rgb[1], rgb[2], static_cast<std::uint8_t>(left ? other_alpha : 255)});
        }
    }
    data::write_png(path, 16, 16, rgba, 4);
}

data::Dataset small_dataset() {
    data::Dataset dataset;
    dataset.source = "somewhere";
    dataset.background = {1, 2, 3};
    dataset.pixels.resize(3 * data::kSpriteValues);
    for (std::size_t i = 0; i < dataset.pixels.size(); ++i) {
        dataset.pixels[i] = static_cast<std::uint8_t>(i * 7);
    }
    return dataset;
}

}

TEST(FileNames, AcceptsPlainNamesAndRejectsWhatWindowsCannotStore) {
    EXPECT_TRUE(data::valid_file_name("icons 16"));
    EXPECT_TRUE(data::valid_file_name("Іконки-1_a"));
    EXPECT_FALSE(data::valid_file_name(""));
    EXPECT_FALSE(data::valid_file_name("a/b"));
    EXPECT_FALSE(data::valid_file_name("a:b"));
    EXPECT_FALSE(data::valid_file_name("what?"));
    EXPECT_FALSE(data::valid_file_name("trailing "));
    EXPECT_FALSE(data::valid_file_name("dot."));
    EXPECT_FALSE(data::valid_file_name("con"));
    EXPECT_FALSE(data::valid_file_name("LPT1.txt"));
    EXPECT_FALSE(data::valid_file_name(std::string(65, 'a')));
    EXPECT_EQ(data::trimmed("  name \t"), "name");
}

TEST(DatasetFile, RoundTripKeepsEverything) {
    const std::filesystem::path path = test::fresh_directory("dataset_file") / "x.spds";
    const data::Dataset dataset = small_dataset();
    data::save_dataset(path, dataset);
    EXPECT_FALSE(std::filesystem::exists(path.string() + ".tmp"));

    const data::Dataset loaded = data::load_dataset(path);
    EXPECT_EQ(loaded.pixels, dataset.pixels);
    EXPECT_EQ(loaded.background, dataset.background);
    EXPECT_EQ(loaded.source, "somewhere");
    EXPECT_EQ(loaded.count(), 3u);

    const data::DatasetInfo info = data::peek_dataset(path);
    EXPECT_EQ(info.name, "x");
    EXPECT_EQ(info.count, 3u);
    EXPECT_EQ(info.source, "somewhere");
    EXPECT_EQ(info.background, dataset.background);
}

TEST(DatasetFile, RejectsDamagedAndForeignFiles) {
    const std::filesystem::path dir = test::fresh_directory("dataset_damaged");
    const std::filesystem::path path = dir / "x.spds";
    data::save_dataset(path, small_dataset());
    std::vector<char> bytes;
    {
        std::ifstream in(path, std::ios::binary);
        bytes.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    }
    const auto write = [&](const std::vector<char>& content) {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        out.write(content.data(), static_cast<std::streamsize>(content.size()));
    };

    std::vector<char> flipped = bytes;
    flipped[flipped.size() / 2] ^= 0x10;
    write(flipped);
    EXPECT_THROW(data::load_dataset(path), std::runtime_error);

    write(std::vector<char>(bytes.begin(), bytes.begin() + 30));
    EXPECT_THROW(data::load_dataset(path), std::runtime_error);
    EXPECT_THROW(data::peek_dataset(path), std::runtime_error);

    write(std::vector<char>(200, 'x'));
    EXPECT_THROW(data::load_dataset(path), std::runtime_error);
    EXPECT_THROW(data::load_dataset(dir / "missing.spds"), std::runtime_error);
}

TEST(DatasetFile, ToFloatsMapsBytesToMinusOneAndOne) {
    data::Dataset dataset;
    dataset.pixels.assign(2 * data::kSpriteValues, 0);
    std::fill(dataset.pixels.begin() + data::kSpriteValues, dataset.pixels.end(), 255);
    const std::vector<float> all = data::to_floats(dataset, 0, 10);
    ASSERT_EQ(all.size(), 2 * data::kSpriteValues);
    EXPECT_FLOAT_EQ(all.front(), -1.0f);
    EXPECT_FLOAT_EQ(all.back(), 1.0f);
    EXPECT_EQ(data::to_floats(dataset, 1, 1).size(), data::kSpriteValues);
    EXPECT_TRUE(data::to_floats(dataset, 5, 1).empty());
}

TEST(DatasetStore, SavesListsLoadsAndRemoves) {
    const std::filesystem::path dir = test::fresh_directory("dataset_store");
    const data::DatasetStore store(dir);
    EXPECT_TRUE(store.list().empty());
    store.save("zeta", small_dataset());
    store.save("Іконки", small_dataset());
    store.save("alpha", small_dataset());
    std::ofstream(dir / "notes.txt") << "not a dataset";

    const std::vector<data::DatasetInfo> list = store.list();
    ASSERT_EQ(list.size(), 3u);
    EXPECT_EQ(list[0].name, "alpha");
    EXPECT_EQ(list[1].name, "zeta");
    EXPECT_EQ(list[2].name, "Іконки");
    EXPECT_TRUE(store.exists("Іконки"));
    EXPECT_EQ(store.load("alpha").pixels, small_dataset().pixels);

    EXPECT_TRUE(store.remove("zeta"));
    EXPECT_FALSE(store.remove("zeta"));
    EXPECT_FALSE(store.exists("zeta"));
    EXPECT_THROW(store.save("bad/name", small_dataset()), std::invalid_argument);
    EXPECT_THROW(store.load("missing"), std::runtime_error);
}

TEST(DatasetStore, WithoutAFolderItHasNothingAndCannotSave) {
    const data::DatasetStore store{std::filesystem::path{}};
    EXPECT_FALSE(store.available());
    EXPECT_TRUE(store.list().empty());
    EXPECT_FALSE(store.exists("x"));
    EXPECT_THROW(store.save("x", small_dataset()), std::runtime_error);
}

TEST(SpriteImporter, ReadsPicturesKeysTransparencyAndSkipsTheRest) {
    const std::filesystem::path dir = test::fresh_directory("importer");
    write_sprite(dir / "a.png", {10, 20, 30});
    write_sprite(dir / "b.png", {50, 60, 70}, 8);              // left half fully transparent
    write_sprite(dir / "c.png", {1, 2, 3}, 8, 127, {9, 9, 9});  // left half: alpha 127, below the threshold
    std::filesystem::create_directories(dir / "sub");
    write_sprite(dir / "sub" / "d.png", {4, 5, 6}, 8, 128, {7, 8, 9});  // left half: alpha 128, kept
    data::write_png(dir / "small.png", 8, 8, std::vector<std::uint8_t>(8 * 8 * 3, 0));
    std::ofstream(dir / "readme.txt") << "not a picture";

    data::SpriteImporter importer(dir, {255, 0, 255});
    ASSERT_TRUE(test::wait_for([&] { return importer.finished(); }));
    EXPECT_EQ(importer.total(), 5u);  // the pictures, not the text file
    data::ImportResult result = importer.take_result();
    ASSERT_EQ(result.error, "");
    EXPECT_EQ(result.skipped, 1u);
    ASSERT_EQ(result.dataset.count(), 4u);
    EXPECT_EQ(result.dataset.background, (std::array<std::uint8_t, 3>{255, 0, 255}));

    const auto at = [&](std::size_t sprite, int channel, int x, int y = 0) {
        return result.dataset.pixels[sprite * data::kSpriteValues + channel * 256 + y * 16 + x];
    };
    // a: opaque everywhere.
    EXPECT_EQ(at(0, 0, 3), 10);
    EXPECT_EQ(at(0, 2, 15, 15), 30);
    // b: transparent left half is the background, the right half keeps its color.
    EXPECT_EQ(at(1, 0, 0), 255);
    EXPECT_EQ(at(1, 1, 0), 0);
    EXPECT_EQ(at(1, 2, 7), 255);
    EXPECT_EQ(at(1, 0, 8), 50);
    // c: alpha 127 is transparent.
    EXPECT_EQ(at(2, 0, 0), 255);
    EXPECT_EQ(at(2, 0, 9), 1);
    // d (in the subfolder, sorted after the files of the folder): alpha 128 is opaque.
    EXPECT_EQ(at(3, 0, 0), 7);
    EXPECT_EQ(at(3, 2, 12), 6);
}

TEST(SpriteImporter, ReportsAFolderWithoutSpritesAndAMissingOne) {
    const std::filesystem::path dir = test::fresh_directory("importer_empty");
    {
        data::SpriteImporter importer(dir, {0, 0, 0});
        ASSERT_TRUE(test::wait_for([&] { return importer.finished(); }));
        EXPECT_NE(importer.take_result().error, "");
    }
    {
        data::SpriteImporter importer(dir / "missing", {0, 0, 0});
        ASSERT_TRUE(test::wait_for([&] { return importer.finished(); }));
        EXPECT_NE(importer.take_result().error, "");
    }
}

TEST(SpriteImporter, CanBeCancelled) {
    const std::filesystem::path dir = test::fresh_directory("importer_cancel");
    for (int i = 0; i < 200; ++i) {
        write_sprite(dir / ("s" + std::to_string(1000 + i) + ".png"), {static_cast<std::uint8_t>(i), 1, 2});
    }
    data::SpriteImporter importer(dir, {0, 0, 0});
    importer.cancel();
    ASSERT_TRUE(test::wait_for([&] { return importer.finished(); }));
    EXPECT_LE(importer.done(), importer.total());
    const data::ImportResult result = importer.take_result();
    // Either it stopped or it had already read everything.
    EXPECT_TRUE(result.cancelled || (result.ok() && result.dataset.count() == 200));
    EXPECT_EQ(result.error, "");
}

TEST(SpriteSet, MirrorTurnsSpritesLeftToRightAndOnlySometimes) {
    std::vector<float> sprite(data::kSpriteValues);
    for (size_t i = 0; i < sprite.size(); ++i) {
        sprite[i] = static_cast<float>(i);
    }
    data::SpriteSet plain(sprite, 3);
    EXPECT_EQ(plain.next_batch(1), sprite);

    data::SpriteSet mirrored(sprite, 3, true);
    std::vector<float> flipped = sprite;
    for (size_t row = 0; row < data::kSpriteValues / 16; ++row) {
        std::reverse(flipped.begin() + static_cast<std::ptrdiff_t>(row * 16),
                     flipped.begin() + static_cast<std::ptrdiff_t>(row * 16 + 16));
    }
    int same = 0;
    int turned = 0;
    for (int i = 0; i < 200; ++i) {
        const std::vector<float> got = mirrored.next_batch(1);
        if (got == sprite) {
            ++same;
        } else {
            EXPECT_EQ(got, flipped);
            ++turned;
        }
    }
    EXPECT_GT(same, 60);
    EXPECT_GT(turned, 60);
}

TEST(FileNames, TimestampTextIsTheTimeInTheGivenZoneAndSortsLikeTime) {
    const std::chrono::time_zone* utc = std::chrono::locate_zone("UTC");
    const auto time = std::chrono::sys_days{std::chrono::year{2026} / 10 / 26} + std::chrono::hours{9} +
                      std::chrono::minutes{30} + std::chrono::seconds{15};
    EXPECT_EQ(data::timestamp_text(time, utc), "20261026-093015");
    EXPECT_LT(data::timestamp_text(time, utc), data::timestamp_text(time + std::chrono::seconds{1}, utc));
    // Fractions of a second are dropped.
    EXPECT_EQ(data::timestamp_text(time + std::chrono::milliseconds{999}, utc), "20261026-093015");
}