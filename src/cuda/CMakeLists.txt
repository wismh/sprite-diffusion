# The CUDA backend: no engine types, so the game module and the tests link the same code. Turing (GTX 1650) is sm_75.
# The headers are plain C++; only the .cu files see CUDA.
# Numeric code: it is compiled with optimization in every configuration (the game's DebugGame builds with /Od, which
# makes training many times slower). The flags are set for this directory only.
if(MSVC)
    set(CMAKE_CXX_FLAGS_DEBUGGAME "/MD /O2 /Ob2 /Zi")
endif()

add_library(diffusion_cuda STATIC
    device.h
    device.cu
    tensor.h
    tensor.cu
    check.cuh
    gemm_core.cuh
    gemm.h
    gemm.cu
    conv2d.h
    conv2d.cu
    group_norm.h
    group_norm.cu
    silu.h
    silu.cu
    adam.h
    adam.cu
    ops.h
    ops.cu
    upsample.h
    upsample.cu
    res_block.h
    res_block.cpp
    unet.h
    unet.cpp
    noise_schedule.h
    noise_schedule.cpp
    diffusion.h
    diffusion.cpp
    diffusion_checkpoint.cpp
    probe.h
    probe.cpp
    dream_worker.h
    dream_worker.cpp
    trainer.h
    trainer.cpp)
target_include_directories(diffusion_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(diffusion_cuda PUBLIC CUDA::cudart_static)
target_compile_features(diffusion_cuda PUBLIC cxx_std_23)
set_target_properties(diffusion_cuda PROPERTIES
    CUDA_ARCHITECTURES 75
    CUDA_STANDARD 20
    POSITION_INDEPENDENT_CODE ON)
if(MSVC)
    # The CRT of the game module (see engine_add_game), so the library links into it and into the tests.
    if(WIND_SDK_CONFIG STREQUAL "Debug")
        set_target_properties(diffusion_cuda PROPERTIES MSVC_RUNTIME_LIBRARY MultiThreadedDebugDLL)
    else()
        set_target_properties(diffusion_cuda PROPERTIES MSVC_RUNTIME_LIBRARY MultiThreadedDLL)
    endif()
endif()
