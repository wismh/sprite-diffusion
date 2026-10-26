#include "cuda/diffusion.h"
#include "data/model_store.h"
#include "inspect/sprite_generator.h"

#include "test_support.h"

#include <chrono>
#include <fstream>
#include <thread>

namespace {

cuda::DiffusionConfig small_config(int base) {
    cuda::DiffusionConfig config;
    config.net = {.base = base, .groups = 4};
    return config;
}

}

TEST(CheckpointHeader, PeekReadsTheConfigStepsAndDataset) {
    const std::filesystem::path path = test::fresh_directory("peek") / "m.ckpt";
    cuda::Diffusion diffusion(small_config(8), 1);
    diffusion.set_dataset("icons");
    diffusion.train_step(std::vector<float>(2 * cuda::kSpriteValues, 0.25f));
    diffusion.save(path);

    const cuda::CheckpointInfo info = cuda::peek_checkpoint(path);
    EXPECT_EQ(info.base, 8);
    EXPECT_EQ(info.groups, 4);
    EXPECT_EQ(info.timesteps, 1000);
    EXPECT_EQ(info.steps, 1);
    EXPECT_EQ(info.dataset, "icons");

    cuda::Diffusion other(small_config(8), 2);
    other.load(path);
    EXPECT_EQ(other.dataset(), "icons");
}

TEST(CheckpointHeader, PeekRejectsForeignAndShortFiles) {
    const std::filesystem::path dir = test::fresh_directory("peek_bad");
    std::ofstream(dir / "x.ckpt") << "this is not a checkpoint at all";
    EXPECT_THROW(cuda::peek_checkpoint(dir / "x.ckpt"), std::runtime_error);
    std::ofstream(dir / "y.ckpt") << "ab";
    EXPECT_THROW(cuda::peek_checkpoint(dir / "y.ckpt"), std::runtime_error);
    EXPECT_THROW(cuda::peek_checkpoint(dir / "missing.ckpt"), std::runtime_error);
}

TEST(ModelStore, ListsCheckpointsByNameAndRemovesThem) {
    const std::filesystem::path dir = test::fresh_directory("model_store");
    const data::ModelStore store(dir);
    EXPECT_TRUE(store.list().empty());

    cuda::Diffusion small(small_config(8), 1);
    small.set_dataset("a");
    small.save(store.path_of("small"));
    cuda::Diffusion wide(small_config(16), 1);
    wide.set_dataset("b");
    wide.save(store.path_of("Модель"));
    std::ofstream(dir / "junk.ckpt") << "junk";
    std::ofstream(dir / "notes.txt") << "junk";

    const std::vector<data::ModelInfo> list = store.list();
    ASSERT_EQ(list.size(), 2u);
    EXPECT_EQ(list[0].name, "small");
    EXPECT_EQ(list[0].base, 8);
    EXPECT_EQ(list[0].dataset, "a");
    EXPECT_EQ(list[0].steps, 0);
    EXPECT_GT(list[0].bytes, 0u);
    EXPECT_EQ(list[1].name, "Модель");
    EXPECT_EQ(list[1].base, 16);
    EXPECT_TRUE(store.exists("Модель"));

    EXPECT_TRUE(store.remove("small"));
    EXPECT_FALSE(store.remove("small"));
    EXPECT_EQ(store.list().size(), 1u);
    EXPECT_THROW(store.path_of("a/b"), std::invalid_argument);
}

TEST(ModelStore, WithoutAFolderItHasNothing) {
    const data::ModelStore store{std::filesystem::path{}};
    EXPECT_FALSE(store.available());
    EXPECT_TRUE(store.list().empty());
    EXPECT_FALSE(store.exists("x"));
    EXPECT_FALSE(store.remove("x"));
    EXPECT_THROW(store.path_of("x"), std::runtime_error);
}

TEST(SpriteGenerator, WritesTheRequestedNumberOfPictures) {
    const std::filesystem::path dir = test::fresh_directory("generate");
    cuda::Diffusion diffusion(small_config(8), 1);
    diffusion.save(dir / "m.ckpt");

    inspect::GenerateRequest request;
    request.checkpoint = dir / "m.ckpt";
    request.directory = dir / "out";
    request.prefix = "m";
    request.count = 70;  // a full batch and a part of one
    request.steps = 3;
    inspect::SpriteGenerator generator(request);
    ASSERT_TRUE(test::wait_for([&] { return generator.finished(); }));
    const inspect::GenerateResult result = generator.take_result();
    EXPECT_EQ(result.error, "");
    EXPECT_EQ(result.written, 70u);
    EXPECT_EQ(generator.done(), 70u);
    EXPECT_EQ(generator.latest().size(), 6 * cuda::kSpriteValues);  // the newest batch is the part of one
    EXPECT_TRUE(std::filesystem::exists(dir / "out" / "m_00001.png"));
    EXPECT_TRUE(std::filesystem::exists(dir / "out" / "m_00070.png"));
    EXPECT_FALSE(std::filesystem::exists(dir / "out" / "m_00071.png"));
}

TEST(SpriteGenerator, ClearsMagentaOnlyWhenAsked) {
    std::vector<float> sprite(cuda::kSpriteValues, 0.0f);
    for (int i = 0; i < 256; ++i) {
        sprite[i] = 1.0f;        // red
        sprite[256 + i] = -1.0f; // green
        sprite[512 + i] = 1.0f;  // blue
    }
    EXPECT_EQ(inspect::SpriteGenerator::to_rgba(sprite.data(), true)[3], 0);
    EXPECT_EQ(inspect::SpriteGenerator::to_rgba(sprite.data(), false)[3], 255);
    std::fill(sprite.begin(), sprite.end(), 0.0f);
    EXPECT_EQ(inspect::SpriteGenerator::to_rgba(sprite.data(), true)[3], 255);
}

TEST(SpriteGenerator, ReportsAMissingCheckpoint) {
    const std::filesystem::path dir = test::fresh_directory("generate_missing");
    inspect::GenerateRequest request;
    request.checkpoint = dir / "nope.ckpt";
    request.directory = dir / "out";
    request.prefix = "x";
    request.count = 1;
    inspect::SpriteGenerator generator(request);
    ASSERT_TRUE(test::wait_for([&] { return generator.finished(); }));
    EXPECT_NE(generator.take_result().error, "");
}

TEST(CheckpointHeader, ConfigOfRebuildsTheConfigTheCheckpointWasMadeWith) {
    const std::filesystem::path path = test::fresh_directory("config_of") / "m.ckpt";
    cuda::DiffusionConfig config = small_config(8);
    config.timesteps = 200;
    cuda::Diffusion(config, 1).save(path);

    for (const cuda::DiffusionConfig& rebuilt : {cuda::config_of(path), cuda::config_of(cuda::peek_checkpoint(path))}) {
        EXPECT_EQ(rebuilt.timesteps, 200);
        EXPECT_EQ(rebuilt.net.base, 8);
        EXPECT_EQ(rebuilt.net.groups, 4);
    }
    // It is enough to load the checkpoint back.
    cuda::Diffusion again(cuda::config_of(path), 2);
    EXPECT_NO_THROW(again.load(path));
    EXPECT_THROW((void)cuda::config_of(path.parent_path() / "missing.ckpt"), std::runtime_error);
}