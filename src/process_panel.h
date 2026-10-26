#pragma once

#include "cuda/diffusion.h"
#include "inspect/copy_check.h"
#include "inspect/nearest_sprite.h"
#include "inspection.h"
#include "loc/localized.h"
#include "loc/message.h"
#include "view/app_view_model.h"

#include <memory>
#include <vector>

namespace game {

// The Process tab: how sprites come out of noise (the steps of sampling, one row per sprite, as the network's estimate of
// the sprite or as the noisy picture it was fed), the sprite of the dataset each result is closest to, and a check that
// counts how many of a few hundred sprites look like copies of the dataset.
class ProcessPanel final : public Localized {
public:
    ProcessPanel(AppViewModel& view, Inspection& inspection, const Tr& tr);

    // Once a frame on the main thread: notices a newly loaded model and follows the copy check.
    void update();
    void apply_locale() override;

private:
    [[nodiscard]] bool has_session() const { return session_ != nullptr; }
    [[nodiscard]] bool can_check() const { return session_ != nullptr && check_ == nullptr; }
    [[nodiscard]] bool can_cancel() const { return check_ != nullptr; }

    void new_samples();
    void pick_estimate();
    void pick_noisy();
    void check_copies();
    void cancel_check() { check_->cancel(); }

    void sync_session();
    void show_grid();
    void show_texts();
    void finish_check();

    static constexpr int kSamples = 4;
    static constexpr int kSteps = 30;
    static constexpr int kColumns = 9;  // the steps shown, then the closest sprite of the dataset

    AppViewModel& view_;
    Inspection& inspection_;
    const Tr& tr_;
    InspectSession* session_ = nullptr;
    unsigned generation_ = 0;

    cuda::SampleTrace trace_;
    std::vector<inspect::Neighbor> neighbors_;
    // How far a dataset sprite typically is from its closest neighbour (a sprite this close is as near the data as the data is
    // to itself); 0 until a check has told.
    double copy_distance_ = 0.0;
    std::unique_ptr<inspect::CopyCheck> check_;
    // The text of the check: the result of the last one, as a message.
    Message copy_message_;
};

}
