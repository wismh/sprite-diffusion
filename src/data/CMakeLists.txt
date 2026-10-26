# Sprites in and out, and the stores of datasets and models: no engine types.
# Numeric code: it is compiled with optimization in every configuration (the game's DebugGame builds with /Od, which
# makes training many times slower). The flags are set for this directory only.
if(MSVC)
    set(CMAKE_CXX_FLAGS_DEBUGGAME "/MD /O2 /Ob2 /Zi")
endif()

add_library(diffusion_data STATIC
    background_job.h
    dataset_file.h
    dataset_file.cpp
    dataset_store.h
    dataset_store.cpp
    file_names.h
    file_names.cpp
    model_store.h
    model_store.cpp
    png_writer.h
    png_writer.cpp
    sprite_importer.h
    sprite_importer.cpp
    sprite_set.h
    sprite_set.cpp
    synthetic_sprites.h
    synthetic_sprites.cpp)
target_include_directories(diffusion_data PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
# The model store reads the headers of checkpoints.
target_link_libraries(diffusion_data PUBLIC diffusion_cuda)
target_compile_features(diffusion_data PUBLIC cxx_std_23)
set_target_properties(diffusion_data PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(MSVC)
    # The CRT of the game module (see engine_add_game), so the library links into it and into the tests.
    if(WIND_SDK_CONFIG STREQUAL "Debug")
        set_target_properties(diffusion_data PROPERTIES MSVC_RUNTIME_LIBRARY MultiThreadedDebugDLL)
    else()
        set_target_properties(diffusion_data PROPERTIES MSVC_RUNTIME_LIBRARY MultiThreadedDLL)
    endif()
endif()
