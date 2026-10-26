#pragma once

#include <engine/loc/catalog.h>

#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace game {

// Looks a message up in the catalog in the language the player chose. `{name}` places in the message are filled from
// `args`. The catalog changes language under it, so one Tr serves for the whole run.
class Tr {
public:
    using Args = std::vector<std::pair<std::string, std::string>>;

    explicit Tr(const engine::loc::Catalog& catalog)
        : catalog_(&catalog) {}

    [[nodiscard]] std::string operator()(std::string_view key) const {
        return catalog_->text(key).text;
    }

    [[nodiscard]] std::string operator()(std::string_view key, const Args& args) const {
        std::vector<engine::loc::Arg> values;
        values.reserve(args.size());
        for (const auto& [name, value] : args) {
            values.push_back(engine::loc::Arg{name, std::string_view{value}});
        }
        return catalog_->text(key, values).text;
    }

private:
    const engine::loc::Catalog* catalog_;
};

}
