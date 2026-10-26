#pragma once

#include <filesystem>
#include <string>
#include <string_view>

namespace game {

// The language the player chose last, kept in `locale.txt` in the user data directory. `read` returns `fallback` when
// there is no file or it holds something that is not one of the languages the program has.
[[nodiscard]] std::string read_locale(const std::filesystem::path& directory, std::string_view fallback);
void write_locale(const std::filesystem::path& directory, std::string_view tag);

}
