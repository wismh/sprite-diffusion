# Unit tests of the domain. They link the same libraries as the game module and GTest from the SDK; no engine code is
# compiled or linked.
# The parts of the app that have no engine types in them are compiled into the tests from their place in src/.
set(APP_CORE ../src/session.cpp ../src/inspect_session.cpp ../src/loc/locale_prefs.cpp)

add_executable(diffusion_tests
    test_support.h
    test_cuda_smoke.cpp
    test_gemm.cpp
    test_conv2d.cpp
    test_group_norm.cpp
    test_silu.cpp
    test_adam.cpp
    test_ops.cpp
    test_unet.cpp
    test_diffusion.cpp
    test_checkpoint.cpp
    test_trainer.cpp
    test_datasets.cpp
    test_models.cpp
    test_inspect.cpp
    test_background_job.cpp
    test_channel_stats.cpp
    test_loss_chart.cpp
    test_process_grid.cpp
    test_cell_image.cpp
    test_locale_prefs.cpp
    test_session.cpp
    ${APP_CORE})
target_include_directories(diffusion_tests PRIVATE ../src)
target_link_libraries(diffusion_tests PRIVATE diffusion_cuda diffusion_data diffusion_inspect GTest::gtest_main)
if(MSVC)
    if(WIND_SDK_CONFIG STREQUAL "Debug")
        set_target_properties(diffusion_tests PROPERTIES MSVC_RUNTIME_LIBRARY MultiThreadedDebugDLL)
    else()
        set_target_properties(diffusion_tests PROPERTIES MSVC_RUNTIME_LIBRARY MultiThreadedDLL)
    endif()
endif()
add_test(NAME diffusion_tests COMMAND diffusion_tests)
