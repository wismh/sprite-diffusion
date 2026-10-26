#pragma once

namespace game {

// Something on screen that writes its texts again after a change of language.
class Localized {
public:
    virtual ~Localized() = default;

    virtual void apply_locale() = 0;
};

}
