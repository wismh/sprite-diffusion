#include "data/file_names.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <format>

namespace data {

std::string trimmed(std::string_view name) {
    const auto space = [](unsigned char c) { return std::isspace(c) != 0; };
    while (!name.empty() && space(static_cast<unsigned char>(name.front()))) {
        name.remove_prefix(1);
    }
    while (!name.empty() && space(static_cast<unsigned char>(name.back()))) {
        name.remove_suffix(1);
    }
    return std::string(name);
}

std::string timestamp_text(std::chrono::system_clock::time_point time, const std::chrono::time_zone* zone) {
    return std::format("{:%Y%m%d-%H%M%S}",
            std::chrono::zoned_time{zone, std::chrono::floor<std::chrono::seconds>(time)});
}

bool valid_file_name(std::string_view name) {
    if (name.empty() || name.size() > 64) {
        return false;
    }
    if (std::any_of(name.begin(), name.end(), [](char c) {
            return static_cast<unsigned char>(c) < 32 || std::string_view("<>:\"/\\|?*").find(c) != std::string_view::npos;
        })) {
        return false;
    }
    if (name.back() == ' ' || name.back() == '.') {
        return false;
    }
    // CON, PRN, AUX, NUL, COM1..9, LPT1..9, with or without an extension.
    std::string stem(name.substr(0, name.find('.')));
    std::transform(stem.begin(), stem.end(), stem.begin(), [](unsigned char c) { return std::toupper(c); });
    static const std::array<const char*, 22> reserved = {"CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4",
            "COM5", "COM6", "COM7", "COM8", "COM9", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8",
            "LPT9"};
    return std::none_of(reserved.begin(), reserved.end(), [&stem](const char* word) { return stem == word; });
}

}
