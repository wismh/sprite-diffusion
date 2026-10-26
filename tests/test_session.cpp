#include "inspect_session.h"
#include "session.h"

#include "test_support.h"

#include <stdexcept>

namespace {

constexpr int kSmall = 8;

data::Dataset sprites(int count) {
    data::Dataset dataset;
    dataset.source = "tests";
    dataset.pixels.resize(static_cast<std::size_t>(count) * data::kSpriteValues);
    for (std::size_t i = 0; i < dataset.pixels.size(); ++i) {
        dataset.pixels[i] = static_cast<std::uint8_t>((i * 31) % 251);
    }
    return dataset;
}

// Stores in a fresh directory, with datasets "a" (20 sprites) and "b" (5 sprites).
struct Stores {
    explicit Stores(const char* name)
        : directory(test::fresh_directory(name))
        , models(directory / "models")
        , datasets(directory / "datasets") {
        datasets.save("a", sprites(20));
        datasets.save("b", sprites(5));
    }

    std::filesystem::path directory;
    data::ModelStore models;
    data::DatasetStore datasets;
};

}

TEST(Session, CreateWritesTheFirstCheckpointNamingItsDataset) {
    Stores stores("session_create");
    const auto session = game::Session::create("m", kSmall, "a", stores.models, stores.datasets);
    EXPECT_EQ(session->name(), "m");
    EXPECT_EQ(session->dataset_name(), "a");
    EXPECT_EQ(session->sprite_count(), 20u);
    EXPECT_EQ(session->base(), kSmall);
    EXPECT_GT(session->parameter_count(), 0u);
    EXPECT_TRUE(session->idle());
    ASSERT_TRUE(stores.models.exists("m"));
    const cuda::CheckpointInfo info = cuda::peek_checkpoint(stores.models.path_of("m"));
    EXPECT_EQ(info.dataset, "a");
    EXPECT_EQ(info.base, kSmall);
    EXPECT_EQ(info.steps, 0);
}

TEST(Session, CreateRefusesATakenNameAndAMissingDataset) {
    Stores stores("session_create_refused");
    const auto first = game::Session::create("m", kSmall, "a", stores.models, stores.datasets);
    EXPECT_THROW(game::Session::create("m", kSmall, "a", stores.models, stores.datasets), std::runtime_error);
    EXPECT_THROW(game::Session::create("n", kSmall, "nope", stores.models, stores.datasets), std::runtime_error);
    EXPECT_THROW(game::Session::create("n", kSmall, "", stores.models, stores.datasets), std::runtime_error);
    EXPECT_FALSE(stores.models.exists("n"));
}

TEST(Session, OpenFindsTheModelAndItsDatasetAgain) {
    Stores stores("session_open");
    game::Session::create("m", kSmall, "a", stores.models, stores.datasets).reset();
    const auto session = game::Session::open("m", stores.models, stores.datasets);
    EXPECT_EQ(session->name(), "m");
    EXPECT_EQ(session->dataset_name(), "a");
    EXPECT_EQ(session->sprite_count(), 20u);
    EXPECT_EQ(session->base(), kSmall);
}

TEST(Session, OpenWithAnotherDatasetRewritesTheCheckpoint) {
    Stores stores("session_open_other");
    game::Session::create("m", kSmall, "a", stores.models, stores.datasets).reset();
    const auto session = game::Session::open("m", stores.models, stores.datasets, "b");
    EXPECT_EQ(session->dataset_name(), "b");
    EXPECT_EQ(session->sprite_count(), 5u);
    EXPECT_EQ(cuda::peek_checkpoint(stores.models.path_of("m")).dataset, "b");
}

TEST(Session, CreateRefusesADatasetWithoutSprites) {
    Stores stores("session_empty_dataset");
    stores.datasets.save("empty", sprites(0));
    EXPECT_THROW(game::Session::create("m", kSmall, "empty", stores.models, stores.datasets), std::runtime_error);
    EXPECT_FALSE(stores.models.exists("m"));
}

TEST(Session, OpenReportsAMissingModelAndAMissingDataset) {
    Stores stores("session_open_missing");
    EXPECT_THROW(game::Session::open("nope", stores.models, stores.datasets), std::runtime_error);
    game::Session::create("m", kSmall, "a", stores.models, stores.datasets).reset();
    ASSERT_TRUE(stores.datasets.remove("a"));
    EXPECT_THROW(game::Session::open("m", stores.models, stores.datasets), std::runtime_error);
}

TEST(Session, PreviewDrawsTheConfiguredNumberOfSprites) {
    Stores stores("session_preview");
    const auto session = game::Session::create("m", kSmall, "a", stores.models, stores.datasets);
    const cuda::Preview preview = session->preview();
    EXPECT_EQ(preview.step, 0);
    EXPECT_EQ(preview.sprites.size(), 16u * cuda::kSpriteValues);
}

TEST(Session, SaveWritesTheCheckpointAgain) {
    Stores stores("session_save");
    const auto session = game::Session::create("m", kSmall, "a", stores.models, stores.datasets);
    std::filesystem::remove(session->checkpoint_path());
    session->save();
    EXPECT_TRUE(stores.models.exists("m"));
}

TEST(InspectSession, OpensTheSavedModelWithItsDatasetAsFloats) {
    Stores stores("inspect_open");
    game::Session::create("m", kSmall, "a", stores.models, stores.datasets).reset();
    const auto session = game::InspectSession::open("m", stores.models, stores.datasets);
    EXPECT_EQ(session->name(), "m");
    EXPECT_EQ(session->dataset_name(), "a");
    EXPECT_EQ(session->steps(), 0);
    EXPECT_EQ(session->sprite_count(), 20u);
    EXPECT_EQ(session->dataset().size(), 20u * data::kSpriteValues);
    for (const float value : session->dataset()) {
        ASSERT_GE(value, -1.0f);
        ASSERT_LE(value, 1.0f);
    }
}

TEST(InspectSession, RefusesADatasetWithoutSprites) {
    Stores stores("inspect_empty_dataset");
    game::Session::create("m", kSmall, "a", stores.models, stores.datasets).reset();
    stores.datasets.save("a", sprites(0));
    EXPECT_THROW(game::InspectSession::open("m", stores.models, stores.datasets), std::runtime_error);
}

TEST(InspectSession, ReportsAMissingModelAndAMissingDataset) {
    Stores stores("inspect_missing");
    EXPECT_THROW(game::InspectSession::open("nope", stores.models, stores.datasets), std::runtime_error);
    game::Session::create("m", kSmall, "a", stores.models, stores.datasets).reset();
    ASSERT_TRUE(stores.datasets.remove("a"));
    EXPECT_THROW(game::InspectSession::open("m", stores.models, stores.datasets), std::runtime_error);
}