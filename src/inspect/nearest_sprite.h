#pragma once

#include <cstddef>
#include <vector>

namespace inspect {

// The sprite of a dataset that is closest to another one.
struct Neighbor {
    size_t index = 0;
    // The dataset sprite turned left to right is the closer one (training shows the model both ways).
    bool mirrored = false;
    // Root mean square of the difference per value, in the [-1, 1] units of the sprites.
    double distance = 0.0;
};

// For each of the `queries` (sprites, NCHW, 768 values each), the closest sprite of `dataset` (the same layout), looking
// at every sprite both as it is and turned left to right. `own`, when it is not empty, holds for each query the index in
// the dataset of the very same sprite, which is left out (turned or not). Works on several threads. The dataset must have a sprite besides the ones left out.
[[nodiscard]] std::vector<Neighbor> nearest_sprites(const std::vector<float>& queries, const std::vector<float>& dataset,
        const std::vector<size_t>& own = {});

// The sprite at `index` of the dataset, turned left to right when `mirrored`.
[[nodiscard]] std::vector<float> dataset_sprite(const std::vector<float>& dataset, size_t index, bool mirrored);

}
