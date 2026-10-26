#pragma once

// Only for .cu files. One tiled matrix-product kernel that the product-shaped operations of the library share: the
// plain GEMM and the three products of a convolution, which read their operands straight from the images instead of
// from a matrix of columns (an "implicit" GEMM: no im2col buffer is written or read back).

#include "cuda/check.cuh"


namespace cuda {

constexpr int kGemmThreads = 256;
constexpr int kGemmDepth = 16;  // the K values of one tile

// Integer division by a divisor that is the same for the whole launch, without an integer divide: the quotient is
// estimated with a float reciprocal and corrected by one. Exact for dividends below 2^22.
struct FastDiv {
    int divisor = 1;
    float inverse = 1.0f;

    FastDiv() = default;
    explicit FastDiv(int d) : divisor(d), inverse(1.0f / static_cast<float>(d)) {}

    __device__ void divmod(int a, int& quotient, int& remainder) const {
        int q = __float2int_rz(static_cast<float>(a) * inverse);
        int r = a - q * divisor;
        if (r < 0) {
            --q;
            r += divisor;
        } else if (r >= divisor) {
            ++q;
            r -= divisor;
        }
        quotient = q;
        remainder = r;
    }
};

// C (m x n) = A (m x k) * B (k x n) with the operands and the result described by `op`:
//   op.load_a(i, k), op.load_b(k, j): the elements, only called inside the matrices;
//   op.store(i, j, value, slice): receives the finished element of C (of one slice of K, see below);
//   Op::kAKFast / Op::kBKFast: true when neighbouring k are neighbouring in memory for A (for B), false when neighbouring
//   i (j) are; the tile is then loaded so that neighbouring threads read neighbouring addresses.
// A block computes BM x BN of C with 256 threads, each a (BM/16) x (BN/16) patch, over tiles of 16 values of K. When K is
// long and C small (a weight gradient) the blocks of z = 0, 1, ... each take their own slice of K of k_per_slice values
// and hand their partial sums to op.store, which must keep them apart.
template <int BM, int BN, class Op>
__global__ void __launch_bounds__(kGemmThreads, 2) gemm_core(Op op, int m, int n, int k, int k_per_slice) {
    constexpr int TM = BM / 16;
    constexpr int TN = BN / 16;
    constexpr int kLoadsA = BM * kGemmDepth / kGemmThreads;
    constexpr int kLoadsB = BN * kGemmDepth / kGemmThreads;
    // A row of BM + 4 floats keeps the rows 16-byte aligned for the vector reads.
    __shared__ __align__(16) float tile_a[kGemmDepth][BM + 4];
    __shared__ __align__(16) float tile_b[kGemmDepth][BN + 4];

    const int tid = threadIdx.x;
    const int tx = tid % 16;
    const int ty = tid / 16;
    const int row0 = blockIdx.y * BM;
    const int col0 = blockIdx.x * BN;
    float acc[TM][TN] = {};

    const int slice = blockIdx.z;
    const int k_begin = slice * k_per_slice;
    const int k_end = min(k, k_begin + k_per_slice);

    // The values of the next tile are read from global memory into registers while the current one is multiplied: the
    // reads (and the address arithmetic of an implicit operand) overlap with the multiplications instead of stalling them.
    float next_a[kLoadsA];
    float next_b[kLoadsB];
    const auto fetch = [&](int k0) {
#pragma unroll
        for (int r = 0; r < kLoadsA; ++r) {
            const int i = Op::kAKFast ? tid / 16 + r * 16 : tid % BM;
            const int kk = Op::kAKFast ? tid % 16 : tid / BM + r * (kGemmThreads / BM);
            const int gi = row0 + i;
            const int gk = k0 + kk;
            next_a[r] = gi < m && gk < k_end ? op.load_a(gi, gk) : 0.0f;
        }
#pragma unroll
        for (int r = 0; r < kLoadsB; ++r) {
            const int j = Op::kBKFast ? tid / 16 + r * 16 : tid % BN;
            const int kk = Op::kBKFast ? tid % 16 : tid / BN + r * (kGemmThreads / BN);
            const int gj = col0 + j;
            const int gk = k0 + kk;
            next_b[r] = gj < n && gk < k_end ? op.load_b(gk, gj) : 0.0f;
        }
    };
    fetch(k_begin);

    for (int k0 = k_begin; k0 < k_end; k0 += kGemmDepth) {
#pragma unroll
        for (int r = 0; r < kLoadsA; ++r) {
            const int i = Op::kAKFast ? tid / 16 + r * 16 : tid % BM;
            const int kk = Op::kAKFast ? tid % 16 : tid / BM + r * (kGemmThreads / BM);
            tile_a[kk][i] = next_a[r];
        }
#pragma unroll
        for (int r = 0; r < kLoadsB; ++r) {
            const int j = Op::kBKFast ? tid / 16 + r * 16 : tid % BN;
            const int kk = Op::kBKFast ? tid % 16 : tid / BN + r * (kGemmThreads / BN);
            tile_b[kk][j] = next_b[r];
        }
        __syncthreads();
        if (k0 + kGemmDepth < k_end) {
            fetch(k0 + kGemmDepth);
        }
#pragma unroll
        for (int kk = 0; kk < kGemmDepth; ++kk) {
            float ra[TM];
            float rb[TN];
#pragma unroll
            for (int v = 0; v < TM / 4; ++v) {
                const float4 value = *reinterpret_cast<const float4*>(&tile_a[kk][ty * TM + 4 * v]);
                ra[4 * v] = value.x;
                ra[4 * v + 1] = value.y;
                ra[4 * v + 2] = value.z;
                ra[4 * v + 3] = value.w;
            }
#pragma unroll
            for (int v = 0; v < TN / 4; ++v) {
                const float4 value = *reinterpret_cast<const float4*>(&tile_b[kk][tx * TN + 4 * v]);
                rb[4 * v] = value.x;
                rb[4 * v + 1] = value.y;
                rb[4 * v + 2] = value.z;
                rb[4 * v + 3] = value.w;
            }
#pragma unroll
            for (int i = 0; i < TM; ++i) {
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] += ra[i] * rb[j];
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int gi = row0 + ty * TM + i;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int gj = col0 + tx * TN + j;
            if (gi < m && gj < n) {
                op.store(gi, gj, acc[i][j], slice);
            }
        }
    }
}

// Runs `op` as an m x n x k product, with K cut into `slices` parts computed by separate blocks. The big tile has twice
// the arithmetic per value read from shared memory but needs a product large enough to keep the multiprocessors busy
// with so few blocks.
// Runs `op` as an m x n x k product, with K cut into `slices` parts computed by separate blocks. A product with 128 or
// more rows takes tiles of 128 x 64, 8 x 4 values per thread, which need half the shared-memory reads per product of the
// 4 x 4 of a 64 x 64 tile; a wider tile or 8 x 8 values per thread were measured slower on the GTX 1650 (more registers,
// fewer blocks per multiprocessor).
template <class Op>
void launch_gemm(const Op& op, int m, int n, int k, int slices = 1) {
    const int k_per_slice = (((k + slices - 1) / slices + kGemmDepth - 1) / kGemmDepth) * kGemmDepth;
    if (m >= 128) {
        const dim3 grid((n + 63) / 64, (m + 127) / 128, slices);
        gemm_core<128, 64, Op><<<grid, kGemmThreads>>>(op, m, n, k, k_per_slice);
    } else {
        const dim3 grid((n + 63) / 64, (m + 63) / 64, slices);
        gemm_core<64, 64, Op><<<grid, kGemmThreads>>>(op, m, n, k, k_per_slice);
    }
    check(cudaGetLastError());
}

}
