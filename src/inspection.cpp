#include "inspection.h"

#include <engine/log.h>

#include <format>
#include <utility>

namespace game {

Inspection::Inspection(AppViewModel& view, const SelectedModel& models, data::DatasetStore datasets,
        const Tr& tr, StatusLine& status)
    : view_(view), models_(models), datasets_(std::move(datasets)), tr_(tr), status_(status) {
    view_.loadInspect.bind_to<Inspection, &Inspection::load, &Inspection::can_load>(*this);
    show();
}

void Inspection::load() {
    const std::string name = models_.selected();
    // The model that was loaded ends first: its worker gives the GPU back.
    session_.reset();
    try {
        session_ = InspectSession::open(name, models_.store(), datasets_);
    } catch (const std::exception& error) {
        engine::log::warn(std::format("model \"{}\" cannot be inspected: {}", name, error.what()));
        status_.say("insp.cannot_load", {{"name", name}, {"detail", error.what()}});
        ++generation_;
        show();
        return;
    }
    ++generation_;
    show();
    status_.say("insp.loaded_status", {{"name", name}});
}

void Inspection::show() {
    view_.inspectLoaded = session_ != nullptr;
    view_.inspectEmpty = session_ == nullptr;
    view_.inspectModelText = session_ == nullptr
            ? tr_("insp.hint")
            : tr_("insp.loaded", {{"name", session_->name()}, {"steps", std::to_string(session_->steps())},
                          {"dataset", session_->dataset_name()}, {"count", std::to_string(session_->sprite_count())}});
}

}
