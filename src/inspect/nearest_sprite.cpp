#include "inspect/nearest_sprite.h"

#include "data/sprite_set.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <stdexcept>
#include <thread>

namespace inspect {

namespace {

constexpr size_t kRowLength = 16;

void mirror_rows(float* sprite) {
    for (size_t row = 0; row < data::kSpriteValues / kRowLength; ++row) {
        std::reverse(sprite + row * kRowLength, sprite + (row + 1) * kRowLength);
    }
}

Neighbor closest(const float* query, const float* flipped, const std::vector<float>& dataset, size_t skip) {
    const size_t count = dataset.size() / data::kSpriteValues;
    double best = std::numeric_limits<double>::max();
    Neighbor found;
    for (size_t i = 0; i < count; ++i) {
        if (i == skip) {
            continue;
        }
        const float* candidate = dataset.data() + i * data::kSpriteValues;
        float plain = 0.0f;
        float turned = 0.0f;
        for (size_t k = 0; k < data::kSpriteValues; ++k) {
            const float a = query[k] - candidate[k];
            const float b = flipped[k] - candidate[k];
            plain += a * a;
            turned += b * b;
        }
        if (plain < best) {
            best = plain;
            found = {i, false, 0.0};
        }
        if (turned < best) {
            best = turned;
            found = {i, true, 0.0};
        }
    }
    if (best == std::numeric_limits<double>::max()) {
        throw std::invalid_argument("nearest_sprites: no sprite to compare with");
    }
    found.distance = std::sqrt(best / static_cast<double>(data::kSpriteValues));
    return found;
}

}

std::vector<Neighbor> nearest_sprites(const std::vector<float>& queries, const std::vector<float>& dataset,
        const std::vector<size_t>& own) {
    if (queries.size() % data::kSpriteValues != 0 || dataset.size() % data::kSpriteValues != 0) {
        throw std::invalid_argument("nearest_sprites: a whole number of sprites");
    }
    const size_t count = queries.size() / data::kSpriteValues;
    if (!own.empty() && own.size() != count) {
        throw std::invalid_argument("nearest_sprites: one own index per query");
    }
    std::vector<Neighbor> result(count);
    const size_t workers = std::max<size_t>(1, std::min<size_t>(count, std::thread::hardware_concurrency()));
    std::vector<std::thread> threads;
    std::vector<std::exception_ptr> failures(workers);
    for (size_t w = 0; w < workers; ++w) {
        threads.emplace_back([&, w] {
            try {
                std::vector<float> flipped(data::kSpriteValues);
                for (size_t q = w; q < count; q += workers) {
                    const float* query = queries.data() + q * data::kSpriteValues;
                    std::copy(query, query + data::kSpriteValues, flipped.begin());
                    mirror_rows(flipped.data());
                    result[q] = closest(query, flipped.data(), dataset, own.empty() ? static_cast<size_t>(-1) : own[q]);
                }
            } catch (...) {
                failures[w] = std::current_exception();
            }
        });
    }
    for (std::thread& thread : threads) {
        thread.join();
    }
    for (const std::exception_ptr& error : failures) {
        if (error) {
            std::rethrow_exception(error);
        }
    }
    return result;
}

std::vector<float> dataset_sprite(const std::vector<float>& dataset, size_t index, bool mirrored) {
    const auto first = dataset.begin() + static_cast<std::ptrdiff_t>(index * data::kSpriteValues);
    std::vector<float> sprite(first, first + static_cast<std::ptrdiff_t>(data::kSpriteValues));
    if (mirrored) {
        mirror_rows(sprite.data());
    }
    return sprite;
}

}
