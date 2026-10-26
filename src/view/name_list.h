#pragma once

#include "view/list_item.h"

#include <engine/ui/view_model.h>

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace game {

// A list of named things on screen with one of them marked: the datasets, the models. The marked row is the one the buttons
// beside the list act on. A click on a row is not applied at once: the row (and the command that is running) goes with the
// list, so the click waits in take_pick() for the next frame.
class NameList {
public:
    using List = engine::ui::BindableList<std::shared_ptr<ListItem>>;

    struct Row {
        std::string name;
        std::string label;  // what the row says, in the language in use
    };

    explicit NameList(List& list)
        : list_(list) {}

    NameList(const NameList&) = delete;
    NameList& operator=(const NameList&) = delete;

    // Shows `rows`. The marked name stays marked while it is among them, otherwise nothing is.
    void show(const std::vector<Row>& rows);
    void select(const std::string& name);
    void clear_selection() { selected_.clear(); }

    // The marked name, or empty.
    [[nodiscard]] const std::string& selected() const { return selected_; }
    [[nodiscard]] bool has_selection() const { return !selected_.empty(); }
    // The name of the row clicked since the last call.
    [[nodiscard]] std::optional<std::string> take_pick() { return std::exchange(pending_, std::nullopt); }

private:
    List& list_;
    std::string selected_;
    std::optional<std::string> pending_;
};

}