#include "path_dialog.h"

namespace game {

void PathDialog::choose_folder(const std::filesystem::path& start) {
    if (!call_.pending()) {
        call_ = windows_.request_open_folder(engine::kPrimaryWindow, start);
    }
}

void PathDialog::choose_file(std::vector<engine::FileFilter> filters) {
    if (!call_.pending()) {
        call_ = windows_.request_open_file(engine::kPrimaryWindow, std::move(filters));
    }
}

std::optional<std::filesystem::path> PathDialog::take() {
    const std::optional<engine::FileDialogResult> answer = call_.take();
    if (!answer) {
        return std::nullopt;
    }
    return answer->path;
}

}
