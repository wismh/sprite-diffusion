cmake_minimum_required(VERSION 3.20)
project(diffusion LANGUAGES CXX CUDA)

# The Wind SDK. The editor passes CMAKE_PREFIX_PATH when it builds the game; CMakeUserPresets.json does for an IDE.
find_package(Wind REQUIRED)

find_package(CUDAToolkit REQUIRED)
add_subdirectory(src/cuda)
add_subdirectory(src/data)
add_subdirectory(src/inspect)

engine_add_game(diffusion
    src/main.cpp
    src/game.h
    src/game.cpp
    src/path_dialog.h
    src/path_dialog.cpp
    src/session.h
    src/session.cpp
    src/dataset_panel.h
    src/dataset_panel.cpp
    src/generate_panel.h
    src/generate_panel.cpp
    src/inspect_session.h
    src/inspect_session.cpp
    src/inspection.h
    src/inspection.cpp
    src/filters_panel.h
    src/filters_panel.cpp
    src/process_panel.h
    src/process_panel.cpp
    src/view/filter_paint.h
    src/view/filter_paint.cpp
    src/view/filter_row.h
    src/view/filter_row.cpp
    src/selection.h
    src/status_line.h
    src/loc/tr.h
    src/loc/localized.h
    src/loc/message.h
    src/loc/locale_prefs.h
    src/loc/locale_prefs.cpp
    src/help/help.h
    src/help/help.cpp
    src/help/help_figures.h
    src/help/help_figures.cpp
    src/help/help_topic_table.h
    src/help/help_view_model.h
    src/model_panel.h
    src/model_panel.cpp
    src/train_panel.h
    src/train_panel.cpp
    src/view/app_view_model.h
    src/view/app_view_model.cpp
    src/view/list_item.h
    src/view/name_list.h
    src/view/name_list.cpp
    src/view/loss_paint.h
    src/view/loss_paint.cpp
    src/view/sprite_grid_paint.h
    src/view/sprite_grid_paint.cpp)
target_include_directories(diffusion PRIVATE src)
target_link_libraries(diffusion PRIVATE diffusion_cuda diffusion_data diffusion_inspect)

# Unit tests of the domain. Off by default so that the editor build does not compile them.
option(DIFFUSION_TESTS "Build the unit tests" OFF)
if(DIFFUSION_TESTS)
    enable_testing()
    add_subdirectory(tests)
    add_subdirectory(tools)
endif()
