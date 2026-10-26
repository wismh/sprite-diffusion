#pragma once

#include "help/help_figures.h"

#include <engine/ui/view_model.h>

#include <array>
#include <string>

namespace game {

// One row of the topic list on the left of the Help tab: a section heading, a topic, or the topic being read.
class HelpNavItem final : public engine::ui::ViewModel {
public:
    HelpNavItem() {
        property(engine::ui::intern("title"), title);
        property(engine::ui::intern("isHeader"), is_header);
        property(engine::ui::intern("isTopic"), is_topic);
        property(engine::ui::intern("isActive"), is_active);
        command(engine::ui::intern("select"), select);
    }

    engine::ui::Bindable<std::string> title;
    engine::ui::Bindable<bool> is_header{false};
    engine::ui::Bindable<bool> is_topic{false};   // a topic that is not the one shown
    engine::ui::Bindable<bool> is_active{false};  // the topic that is shown
    engine::ui::RelayCommand select;
};

// What a topic document binds: its pictures and the button that opens the tab it talks about.
class HelpTopicViewModel final : public engine::ui::ViewModel {
public:
    HelpTopicViewModel() {
        command(engine::ui::intern("tryIt"), tryIt);
        for (std::size_t i = 0; i < kFiguresPerTopic; ++i) {
            paint(engine::ui::intern("diagram" + std::to_string(i + 1)), diagram[i]);
        }
    }

    std::array<engine::ui::RelayPaint, kFiguresPerTopic> diagram;
    engine::ui::RelayCommand tryIt;
};

}
