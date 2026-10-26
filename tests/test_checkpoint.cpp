#include "cuda/diffusion.h"
#include "data/synthetic_sprites.h"

#include "test_support.h"

#include <filesystem>
#include <fstream>

namespace {

cuda::DiffusionConfig small_config(int base = 8) {
    cuda::DiffusionConfig config;
    config.net = {.base = base, .groups = 4};
    config.learning_rate = 1e-3f;
    return config;
}

std::filesystem::path temp_file(const char* name) {
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "diffusion_tests";
    std::filesystem::create_directories(dir);
    return dir / name;
}

void expect_same_params(const std::vector<cuda::Param*>& a, const std::vector<cuda::Param*>& b, bool moments) {
    ASSERT_EQ(a.size(), b.size());
    for (size_t i = 0; i < a.size(); ++i) {
        EXPECT_EQ(a[i]->value.download(), b[i]->value.download()) << "param " << i;
        if (moments) {
            ASSERT_EQ(a[i]->m.size(), b[i]->m.size());
            EXPECT_EQ(a[i]->m.download(), b[i]->m.download()) << "m " << i;
            EXPECT_EQ(a[i]->v.download(), b[i]->v.download()) << "v " << i;
        }
    }
}

std::vector<char> read_all(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

void write_all(const std::filesystem::path& path, const std::vector<char>& bytes) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

}

TEST(Checkpoint, RoundTripRestoresWeightsAverageMomentsAndTheRun) {
    const std::filesystem::path path = temp_file("roundtrip.ckpt");
    const std::vector<float> batch = data::synthetic_sprites(8, 3);

    cuda::Diffusion a(small_config(), 1);
    a.set_dataset("icons");
    for (int i = 0; i < 5; ++i) {
        a.train_step(batch);
    }
    a.save(path);

    cuda::Diffusion b(small_config(), 2);
    b.load(path);
    EXPECT_EQ(b.steps_done(), 5);
    EXPECT_EQ(b.dataset(), "icons");
    expect_same_params(a.model().params(), b.model().params(), true);
    expect_same_params(a.ema().params(), b.ema().params(), false);
    EXPECT_EQ(a.sample(2, 5, 3), b.sample(2, 5, 3));

    // The next step is the same too: the random generator came along.
    EXPECT_NEAR(a.train_step(batch), b.train_step(batch), 1e-6);
    expect_same_params(a.model().params(), b.model().params(), true);
    EXPECT_EQ(b.steps_done(), 6);
}

TEST(Checkpoint, SavedBeforeTheFirstStepTrainsAfterLoad) {
    const std::filesystem::path path = temp_file("fresh.ckpt");
    cuda::Diffusion a(small_config(), 1);
    a.save(path);
    cuda::Diffusion b(small_config(), 2);
    b.load(path);
    expect_same_params(a.model().params(), b.model().params(), false);
    const std::vector<float> batch = data::synthetic_sprites(8, 3);
    EXPECT_NEAR(a.train_step(batch), b.train_step(batch), 1e-6);
}

TEST(Checkpoint, SaveReplacesTheFileAndLeavesNoTemporary) {
    const std::filesystem::path path = temp_file("replace.ckpt");
    cuda::Diffusion a(small_config(), 1);
    a.save(path);
    const auto first = std::filesystem::file_size(path);
    a.train_step(data::synthetic_sprites(8, 3));
    a.save(path);
    EXPECT_GT(std::filesystem::file_size(path), first);  // now with the moments
    EXPECT_FALSE(std::filesystem::exists(path.string() + ".tmp"));
}

TEST(Checkpoint, RejectsADifferentArchitecture) {
    const std::filesystem::path path = temp_file("arch.ckpt");
    cuda::Diffusion a(small_config(8), 1);
    a.save(path);
    cuda::Diffusion wider(small_config(16), 1);
    EXPECT_THROW(wider.load(path), std::runtime_error);
}

TEST(Checkpoint, RejectsDamagedFilesAndLeavesTheObjectAsItWas) {
    const std::filesystem::path path = temp_file("damaged.ckpt");
    cuda::Diffusion a(small_config(), 1);
    a.train_step(data::synthetic_sprites(8, 3));
    a.save(path);
    std::vector<char> bytes = read_all(path);

    cuda::Diffusion b(small_config(), 2);
    const std::vector<float> before = b.sample(2, 3, 1);

    std::vector<char> flipped = bytes;
    flipped[flipped.size() / 2] ^= 0x40;
    write_all(path, flipped);
    EXPECT_THROW(b.load(path), std::runtime_error);

    write_all(path, std::vector<char>(bytes.begin(), bytes.begin() + static_cast<std::ptrdiff_t>(bytes.size() / 2)));
    EXPECT_THROW(b.load(path), std::runtime_error);

    write_all(path, std::vector<char>(100, 'x'));
    EXPECT_THROW(b.load(path), std::runtime_error);

    EXPECT_THROW(b.load(temp_file("missing.ckpt")), std::runtime_error);
    EXPECT_EQ(b.sample(2, 3, 1), before);
}
