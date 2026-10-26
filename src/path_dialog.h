#pragma once

#include <engine/core/file_dialog.h>
#include <engine/core/window_control.h>

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace game {

// One native choose-file or choose-folder dialog at a time, for whoever owns it. choose_*() opens the dialog and
// returns at once; take() (once per frame) gives the chosen path in the frame the answer arrives, and nothing when the
// user cancelled.
class PathDialog {
public:
    explicit PathDialog(engine::IWindowControl& windows)
        : windows_(windows) {}

    // `start` is the folder the dialog opens in; empty leaves that to the platform.
    void choose_folder(const std::filesystem::path& start = {});
    // `filters` are the file types offered; empty offers every file.
    void choose_file(std::vector<engine::FileFilter> filters = {});

    [[nodiscard]] bool pending() const { return call_.pending(); }
    // The chosen path, once. Nothing while the dialog is open, after a cancel, and when no dialog was asked for.
    [[nodiscard]] std::optional<std::filesystem::path> take();

private:
    engine::IWindowControl& windows_;
    engine::FileDialogCall call_;
};

}
