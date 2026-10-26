#pragma once

#include <engine/ui/view_model.h>

#include <string>

namespace game {

// One row of a list of names: the datasets, the models. The marked row is the one the buttons beside the list act on.
class ListItem final : public engine::ui::ViewModel {
public:
    ListItem() {
        property(engine::ui::intern("label"), label);
        property(engine::ui::intern("selected"), selected);
        command(engine::ui::intern("select"), select);
    }

    engine::ui::Bindable<std::string> label;
    engine::ui::Bindable<bool> selected{false};
    engine::ui::RelayCommand select;
};

}
