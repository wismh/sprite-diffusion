#include "help/help.h"

namespace game {

Help::Help(AppViewModel& view, const Tr& tr) : view_(view), tr_(tr) {
    const char* section = nullptr;
    for (std::size_t i = 0; i < kHelpTopicCount; ++i) {
        const HelpTopicInfo& info = kHelpTopics[i];
        if (section != info.section) {
            section = info.section;
            auto header = std::make_shared<HelpNavItem>();
            header->is_header = true;
            nav_.push_back(std::move(header));
        }
        nav_row_[i] = nav_.size();
        auto item = std::make_shared<HelpNavItem>();
        item->select = [this, i] { select(i); };
        nav_.push_back(std::move(item));

        auto topic = std::make_shared<HelpTopicViewModel>();
        for (std::size_t figure = 0; figure < info.figure_count; ++figure) {
            const FigureId id = info.figures[figure];
            topic->diagram[figure] = [this, id](engine::ui::IDrawList& list, const engine::render::Rect& area) {
                draw_figure(id, list, area, tr_);
            };
        }
        if (info.open != HelpOpen::None) {
            topic->tryIt = [this, open = info.open] { this->open(open); };
        }
        view_.helpTopic[i].set({std::move(topic)});
    }
    view_.helpNav.set(nav_);
    apply_locale();
    select(0);
}

void Help::apply_locale() {
    const char* section = nullptr;
    std::size_t row = 0;
    for (std::size_t i = 0; i < kHelpTopicCount; ++i) {
        if (section != kHelpTopics[i].section) {
            section = kHelpTopics[i].section;
            nav_[row]->title = tr_(section);
            ++row;
        }
        nav_[row]->title = tr_(kHelpTopics[i].title);
        ++row;
    }
}

void Help::select(std::size_t topic) {
    if (topic != selected_) {
        view_.helpScroll = 0.0f;
    }
    selected_ = topic;
    for (std::size_t i = 0; i < kHelpTopicCount; ++i) {
        const bool active = i == selected_;
        view_.helpShown[i] = active;
        HelpNavItem& row = *nav_[nav_row_[i]];
        row.is_topic = !active;
        row.is_active = active;
    }
}

void Help::open(HelpOpen tab) {
    switch (tab) {
    case HelpOpen::None: break;
    case HelpOpen::Datasets: view_.showDatasets.execute(); break;
    case HelpOpen::Filters: view_.showFilters.execute(); break;
    case HelpOpen::Generate: view_.showGenerate.execute(); break;
    case HelpOpen::Models: view_.showModels.execute(); break;
    case HelpOpen::Process: view_.showProcess.execute(); break;
    case HelpOpen::Train: view_.showTrain.execute(); break;
    }
}

}
