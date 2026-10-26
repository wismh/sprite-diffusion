#pragma once

#include "help/help_topic_table.h"
#include "help/help_view_model.h"
#include "loc/localized.h"
#include "loc/tr.h"
#include "view/app_view_model.h"

#include <array>
#include <memory>
#include <vector>

namespace game {

// The Help tab: the list of topics, the topic shown, and the pictures in the topics. The topics themselves are documents
// under assets/ui/help_topics; this class only feeds them what they bind.
class Help final : public Localized {
public:
    Help(AppViewModel& view, const Tr& tr);

    Help(const Help&) = delete;
    Help& operator=(const Help&) = delete;

    // Writes the titles in the list again after a change of language. The text inside the topics follows by itself.
    void apply_locale() override;

private:
    void select(std::size_t topic);
    void open(HelpOpen tab);

    AppViewModel& view_;
    const Tr& tr_;

    std::vector<std::shared_ptr<HelpNavItem>> nav_;
    std::array<std::size_t, kHelpTopicCount> nav_row_{};  // the row of each topic in nav_
    std::size_t selected_ = 0;
};

}
