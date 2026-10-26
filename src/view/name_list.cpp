#include "view/name_list.h"

#include <algorithm>

namespace game {

void NameList::show(const std::vector<Row>& rows) {
    const bool present = std::ranges::any_of(rows, [this](const Row& row) { return row.name == selected_; });
    if (!present) {
        selected_.clear();
    }
    std::vector<std::shared_ptr<ListItem>> items;
    items.reserve(rows.size());
    for (const Row& row : rows) {
        auto item = std::make_shared<ListItem>();
        item->label = row.label;
        item->selected = row.name == selected_;
        item->select = [this, name = row.name] { pending_ = name; };
        items.push_back(std::move(item));
    }
    list_.set(std::move(items));
}

void NameList::select(const std::string& name) {
    selected_ = name;
}

}