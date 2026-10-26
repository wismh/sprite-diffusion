#pragma once

// Only for .cu files: the headers of the library stay plain C++ so that the game and the tests need no CUDA includes.
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace cuda {

inline void check(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

constexpr int kThreads = 256;

[[nodiscard]] inline int blocks_for(size_t count) {
    return static_cast<int>((count + kThreads - 1) / kThreads);
}

// Sum of v over the block (blockDim.x a multiple of 32), returned to every thread. shared holds 32 floats.
__device__ inline float block_sum(float v, float* shared) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) {
        shared[warp] = v;
    }
    __syncthreads();
    v = threadIdx.x < (blockDim.x >> 5) ? shared[threadIdx.x] : 0.0f;
    if (warp == 0) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            v += __shfl_down_sync(0xffffffffu, v, offset);
        }
        if (lane == 0) {
            shared[0] = v;
        }
    }
    __syncthreads();
    const float total = shared[0];
    __syncthreads();
    return total;
}

}
