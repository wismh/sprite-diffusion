# What is done with a trained model and its dataset: drawing sprites to files, comparing them with the dataset, the
# numbers and pictures that the Filters, Process and Train tabs show. No engine types, so the game module and the tests
# link the same code.
# Numeric code: it is compiled with optimization in every configuration (the game's DebugGame builds with /Od, which
# makes it many times slower). The flags are set for this directory only.
if(MSVC)
    set(CMAKE_CXX_FLAGS_DEBUGGAME "/MD /O2 /Ob2 /Zi")
endif()

add_library(diffusion_inspect STATIC
    cell_image.h
    cell_image.cpp
    channel_stats.h
    channel_stats.cpp
    copy_check.h
    copy_check.cpp
    loss_chart.h
    loss_chart.cpp
    nearest_sprite.h
    nearest_sprite.cpp
    process_grid.h
    process_grid.cpp
    sprite_generator.h
    sprite_generator.cpp)
target_include_directories(diffusion_inspect PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(diffusion_inspect PUBLIC diffusion_data diffusion_cuda)
target_compile_features(diffusion_inspect PUBLIC cxx_std_23)
set_target_properties(diffusion_inspect PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(MSVC)
    # The CRT of the game module (see engine_add_game), so the library links into it and into the tests.
    if(WIND_SDK_CONFIG STREQUAL "Debug")
        set_target_properties(diffusion_inspect PROPERTIES MSVC_RUNTIME_LIBRARY MultiThreadedDebugDLL)
    else()
        set_target_properties(diffusion_inspect PROPERTIES MSVC_RUNTIME_LIBRARY MultiThreadedDLL)
    endif()
endif()
