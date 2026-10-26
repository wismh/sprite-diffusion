#include "help/help_figures.h"

#include <engine/builtin_ids.h>

#include <algorithm>
#include <cmath>
#include <format>
#include <string>
#include <vector>

namespace game {

namespace {

using engine::render::Rect;
using engine::ui::IDrawList;

constexpr glm::vec4 kInk{0.20f, 0.25f, 0.33f, 1.0f};
constexpr glm::vec4 kMuted{0.58f, 0.64f, 0.72f, 1.0f};
constexpr glm::vec4 kGrid{0.89f, 0.91f, 0.94f, 1.0f};
constexpr glm::vec4 kBlue{0.15f, 0.39f, 0.92f, 1.0f};
constexpr glm::vec4 kRed{0.86f, 0.15f, 0.15f, 1.0f};
constexpr glm::vec4 kPaper{1.0f, 1.0f, 1.0f, 1.0f};
constexpr glm::vec4 kPale{0.93f, 0.96f, 1.0f, 1.0f};

// A small drawing board: text, boxes and lines in the area of one figure.
class Board {
public:
    Board(IDrawList& list, const Rect& area, const Tr& tr)
        : list_(list)
        , area_(area)
        , tr_(tr) {}

    [[nodiscard]] const Rect& area() const { return area_; }
    [[nodiscard]] std::string tr(const char* key) const { return tr_(key); }

    void line(glm::vec2 a, glm::vec2 b, glm::vec4 color, float width = 1.5f) {
        list_.line(abs(a), abs(b), color, width);
    }

    void dashed(glm::vec2 a, glm::vec2 b, glm::vec4 color, float width = 1.5f) {
        const glm::vec2 from = abs(a);
        const glm::vec2 to = abs(b);
        const float length = glm::length(to - from);
        const int pieces = std::max(1, static_cast<int>(length / 8.0f));
        for (int i = 0; i < pieces; ++i) {
            const float t0 = static_cast<float>(i) / static_cast<float>(pieces);
            const float t1 = (static_cast<float>(i) + 0.55f) / static_cast<float>(pieces);
            list_.line(from + (to - from) * t0, from + (to - from) * t1, color, width);
        }
    }

    void polyline(const std::vector<glm::vec2>& points, glm::vec4 color, float width = 2.5f) {
        for (std::size_t i = 1; i < points.size(); ++i) {
            list_.line(abs(points[i - 1]), abs(points[i]), color, width);
        }
    }

    void box(const Rect& rect, glm::vec4 fill, glm::vec4 border, float width = 1.0f) {
        const Rect r{area_.x + rect.x, area_.y + rect.y, rect.w, rect.h};
        list_.fill_rect(r, fill);
        list_.stroke_rect(r, border, width);
    }

    // `at` is the top-left of the text; `align` 0 left, 0.5 centred, 1 right of that point (width is estimated).
    void text(const std::string& value, glm::vec2 at, glm::vec4 color, float size = 13.0f, float align = 0.0f) {
        list_.set_font(engine::builtin::font_ui, size);
        // Letters, not bytes: Ukrainian takes two bytes a letter.
        const auto letters = std::count_if(value.begin(), value.end(), [](char c) { return (static_cast<unsigned char>(c) & 0xC0) != 0x80; });
        const float width = static_cast<float>(letters) * size * 0.52f;
        list_.text(value, abs({at.x - width * align, at.y}), color);
    }

    void arrow(glm::vec2 from, glm::vec2 to) {
        line(from, to, kMuted, 2.5f);
        const glm::vec2 direction = glm::normalize(to - from);
        const glm::vec2 side{-direction.y, direction.x};
        line(to - direction * 10.0f + side * 6.0f, to, kMuted, 2.5f);
        line(to - direction * 10.0f - side * 6.0f, to, kMuted, 2.5f);
    }

private:
    [[nodiscard]] glm::vec2 abs(glm::vec2 p) const { return {area_.x + p.x, area_.y + p.y}; }

    IDrawList& list_;
    Rect area_;
    const Tr& tr_;
};

// The cosine schedule: how much of the sprite is left at noise level t (of 1000).
double signal_left(double t) {
    constexpr double kOffset = 0.008;
    const auto f = [](double x) {
        const double angle = (x + kOffset) / (1.0 + kOffset) * 1.5707963267948966;
        return std::cos(angle) * std::cos(angle);
    };
    return f(t / 1000.0) / f(0.0);
}

void schedule_figure(Board& board) {
    const float w = board.area().w;
    const float h = board.area().h;
    const Rect frame{56.0f, 26.0f, w - 90.0f, h - 78.0f};
    const auto at = [&](double t, double y) {
        return glm::vec2{frame.x + static_cast<float>(t / 1000.0) * frame.w,
                frame.y + frame.h - static_cast<float>(y) * frame.h};
    };
    for (int i = 0; i <= 4; ++i) {
        const double y = i / 4.0;
        board.line(at(0.0, y), at(1000.0, y), kGrid, 1.0f);
        board.text(std::format("{:g}", y), {frame.x - 8.0f, at(0.0, y).y - 8.0f}, kMuted, 11.0f, 1.0f);
    }
    board.line(at(0.0, 0.0), at(1000.0, 0.0), kMuted, 1.5f);
    board.line(at(0.0, 0.0), at(0.0, 1.0), kMuted, 1.5f);
    std::vector<glm::vec2> signal;
    std::vector<glm::vec2> noise;
    for (int t = 0; t <= 1000; t += 10) {
        const double bar = std::clamp(signal_left(t), 0.0, 1.0);
        signal.push_back(at(t, std::sqrt(bar)));
        noise.push_back(at(t, std::sqrt(1.0 - bar)));
    }
    board.polyline(signal, kBlue);
    board.polyline(noise, kRed);
    board.text(board.tr("fig.signal"), {frame.x + 40.0f, at(0.0, 0.5).y}, kBlue, 13.0f);
    board.text(board.tr("fig.noise"), {frame.x + frame.w - 40.0f, at(0.0, 0.5).y}, kRed, 13.0f, 1.0f);
    board.text(board.tr("fig.clean"), {frame.x, frame.y + frame.h + 8.0f}, kInk, 12.0f);
    board.text(board.tr("fig.noisy"), {frame.x + frame.w, frame.y + frame.h + 8.0f}, kInk, 12.0f, 1.0f);
    board.text(board.tr("fig.level"), {frame.x + frame.w * 0.5f, frame.y + frame.h + 28.0f}, kMuted, 12.0f, 0.5f);
}

void unet_figure(Board& board) {
    const float w = board.area().w;
    const float box_w = w * 0.24f;
    const float box_h = 44.0f;
    const float left = w * 0.06f;
    const float right = w - left - box_w;
    const float middle = (w - box_w * 1.5f) * 0.5f;
    const float top = 40.0f;
    const float gap = 36.0f;
    const auto y = [&](int level) { return top + static_cast<float>(level) * (box_h + gap); };

    const char* sizes[3] = {"16x16", "8x8", "4x4"};
    for (int level = 0; level < 2; ++level) {
        board.box({left, y(level), box_w, box_h}, kPale, kBlue, 1.5f);
        board.box({right, y(level), box_w, box_h}, kPale, kBlue, 1.5f);
        board.text(std::string(sizes[level]) + "  " + board.tr("fig.unet_block"), {left + box_w * 0.5f, y(level) + 14.0f},
                kInk, 13.0f, 0.5f);
        board.text(std::string(sizes[level]) + "  " + board.tr("fig.unet_block"), {right + box_w * 0.5f, y(level) + 14.0f},
                kInk, 13.0f, 0.5f);
        board.dashed({left + box_w, y(level) + box_h * 0.5f}, {right, y(level) + box_h * 0.5f}, kMuted);
        board.text(board.tr("fig.unet_skip"), {w * 0.5f, y(level) + box_h * 0.5f - 18.0f}, kMuted, 11.0f, 0.5f);
    }
    board.box({middle, y(2), box_w * 1.5f, box_h}, kPale, kBlue, 1.5f);
    board.text(std::string(sizes[2]) + "  " + board.tr("fig.unet_blocks"), {middle + box_w * 0.75f, y(2) + 14.0f}, kInk,
            13.0f, 0.5f);

    // The way down on the left, the way up on the right.
    for (int level = 0; level < 2; ++level) {
        const float x = left + box_w * 0.5f;
        board.arrow({x, y(level) + box_h + 4.0f}, {x, y(level + 1) - 4.0f});
        const float x2 = right + box_w * 0.5f;
        board.arrow({x2, y(level + 1) - 4.0f}, {x2, y(level) + box_h + 4.0f});
    }
    board.arrow({left + box_w * 0.5f, y(1) + box_h + 4.0f}, {middle + box_w * 0.4f, y(2) + 2.0f});
    board.arrow({middle + box_w * 1.1f, y(2) + 2.0f}, {right + box_w * 0.5f, y(1) + box_h + 4.0f});
    board.text(board.tr("fig.unet_down"), {left + box_w * 0.5f + 8.0f, y(0) + box_h + 22.0f}, kMuted, 11.0f);
    board.text(board.tr("fig.unet_up"), {right + box_w * 0.5f + 8.0f, y(0) + box_h + 22.0f}, kMuted, 11.0f);

    board.text(board.tr("fig.unet_in"), {left, 8.0f}, kInk, 12.0f);
    board.arrow({left + box_w * 0.5f, 26.0f}, {left + box_w * 0.5f, y(0) - 4.0f});
    board.text(board.tr("fig.unet_out"), {right + box_w, 8.0f}, kInk, 12.0f, 1.0f);
    board.arrow({right + box_w * 0.5f, y(0) - 4.0f}, {right + box_w * 0.5f, 26.0f});
}

}

void draw_figure(FigureId figure, IDrawList& list, const Rect& area, const Tr& tr) {
    Board board(list, area, tr);
    board.box({0.0f, 0.0f, area.w, area.h}, kPaper, kGrid);
    switch (figure) {
    case FigureId::None: break;
    case FigureId::Schedule: schedule_figure(board); break;
    case FigureId::UNet: unet_figure(board); break;
    }
}

}
