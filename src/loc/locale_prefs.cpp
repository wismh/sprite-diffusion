#include "loc/locale_prefs.h"

#include <fstream>

namespace game {

std::string read_locale(const std::filesystem::path& directory, std::string_view fallback) {
    if (directory.empty()) {
        return std::string{fallback};
    }
    std::ifstream in(directory / "locale.txt");
    std::string tag;
    in >> tag;
    return tag == "en" || tag == "uk" ? tag : std::string{fallback};
}

void write_locale(const std::filesystem::path& directory, std::string_view tag) {
    if (directory.empty()) {
        return;
    }
    std::error_code error;
    std::filesystem::create_directories(directory, error);
    std::ofstream out(directory / "locale.txt", std::ios::trunc);
    out << tag << '\n';
}

}
