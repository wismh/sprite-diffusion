#pragma once

#include "loc/tr.h"

#include <string>
#include <utility>

namespace game {

// A message of the catalog with its arguments, or plain text that has no translation (the reason a file could not be
// read). Kept instead of the finished text so that a change of language can say the same thing again.
class Message {
public:
    Message() = default;
    explicit Message(std::string key, Tr::Args args = {})
        : key_(std::move(key))
        , args_(std::move(args)) {}

    [[nodiscard]] static Message raw(std::string text) {
        Message message;
        message.raw_ = std::move(text);
        return message;
    }

    [[nodiscard]] bool empty() const { return key_.empty() && raw_.empty(); }
    [[nodiscard]] std::string text(const Tr& tr) const { return key_.empty() ? raw_ : tr(key_, args_); }

private:
    std::string key_;
    Tr::Args args_;
    std::string raw_;
};

}
