#pragma once

#include <chrono>
#include <filesystem>
#include <string>
#include <string_view>

namespace data {

// Whether `name` can be the name of a dataset or a model, which is also the name of its file: not empty, at most 64
// bytes, no characters a Windows file name cannot have, no control characters, not ending in a space or a dot, and not a
// reserved device name.
[[nodiscard]] bool valid_file_name(std::string_view name);

// `name` without the spaces around it, ready for valid_file_name().
[[nodiscard]] std::string trimmed(std::string_view name);

// `time` as local time for a folder name: "20261026-093015". Sorts the same way as the time does.
[[nodiscard]] std::string timestamp_text(std::chrono::system_clock::time_point time,
        const std::chrono::time_zone* zone = std::chrono::current_zone());

// Names are UTF-8 strings in the app; a path made from a std::string would read them in the system code page.
[[nodiscard]] inline std::filesystem::path from_utf8(std::string_view text) {
    return std::filesystem::path(std::u8string(text.begin(), text.end()));
}
[[nodiscard]] inline std::string to_utf8(const std::filesystem::path& path) {
    const std::u8string text = path.u8string();
    return std::string(text.begin(), text.end());
}

}
