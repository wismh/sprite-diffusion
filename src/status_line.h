#pragma once

#include "loc/message.h"
#include "view/app_view_model.h"

#include <string>
#include <utility>

namespace game {

// The line at the bottom of the window. It remembers which message it shows, not only the text, so a change of language can
// say the same thing again in the new language.
class StatusLine {
public:
    StatusLine(AppViewModel& view, const Tr& tr)
        : view_(view)
        , tr_(tr) {}

    // A catalog message; `args` fill its `{name}` places.
    void say(std::string key, Tr::Args args = {}) {
        message_ = Message(std::move(key), std::move(args));
        refresh();
    }

    // Text with no translation, such as the reason a file could not be read.
    void say_raw(std::string text) {
        message_ = Message::raw(std::move(text));
        refresh();
    }

    void refresh() { view_.statusText = message_.text(tr_); }

private:
    AppViewModel& view_;
    const Tr& tr_;
    Message message_;
};

}
