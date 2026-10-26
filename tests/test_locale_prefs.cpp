#include "loc/locale_prefs.h"

#include "test_support.h"

#include <fstream>

TEST(LocalePrefs, ReadsWhatWasWritten) {
    const std::filesystem::path dir = test::fresh_directory("locale");
    game::write_locale(dir, "uk");
    EXPECT_EQ(game::read_locale(dir, "en"), "uk");
    game::write_locale(dir, "en");
    EXPECT_EQ(game::read_locale(dir, "uk"), "en");
}

TEST(LocalePrefs, FallsBackWithoutAFileOrForAnUnknownLanguage) {
    const std::filesystem::path dir = test::fresh_directory("locale_fallback");
    EXPECT_EQ(game::read_locale(dir, "en"), "en");
    std::ofstream(dir / "locale.txt") << "klingon\n";
    EXPECT_EQ(game::read_locale(dir, "en"), "en");
}

TEST(LocalePrefs, WithoutADirectoryNothingIsReadOrWritten) {
    EXPECT_EQ(game::read_locale({}, "uk"), "uk");
    game::write_locale({}, "en");
}

TEST(LocalePrefs, WriteMakesTheDirectory) {
    const std::filesystem::path dir = test::fresh_directory("locale_new") / "deeper";
    game::write_locale(dir, "uk");
    EXPECT_EQ(game::read_locale(dir, "en"), "uk");
}