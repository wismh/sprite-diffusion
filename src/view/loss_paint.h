#pragma once

#include <engine/ui/paint.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace game {

// The loss of the training run as a line, one point per block of steps, on a logarithmic scale: the loss falls by
// orders of magnitude, and a linear axis would flatten everything after the first minute.
class LossPaint final : public engine::ui::IPaint {
public:
    void paint(engine::ui::IDrawList& list, const engine::render::Rect& content) override;
    [[nodiscard]] std::uint64_t version() const noexcept override { return version_; }

    // The history only grows, so a longer one is a change and one of the same length is not.
    void show(std::vector<float> history);
    // Back to an empty chart, for another run.
    void reset();
    // What the chart says while it has nothing to draw.
    void set_empty_text(std::string text) {
        empty_text_ = std::move(text);
        ++version_;
    }

private:
    std::vector<float> history_;
    std::uint64_t version_ = 0;
    std::string empty_text_;
};

}
