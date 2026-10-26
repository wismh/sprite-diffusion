#include "cuda/gemm.h"

#include "cuda/gemm_core.cuh"

namespace cuda {

namespace {

template <bool TransposeA, bool TransposeB>
struct PlainOp {
    // Neighbouring k are neighbouring in memory when A is stored m x k and when B is stored n x k.
    static constexpr bool kAKFast = !TransposeA;
    static constexpr bool kBKFast = TransposeB;

    int m;
    int n;
    int k;
    float alpha;
    float beta;
    const float* a;
    const float* b;
    float* c;

    __device__ float load_a(int i, int p) const {
        return TransposeA ? a[static_cast<size_t>(p) * m + i] : a[static_cast<size_t>(i) * k + p];
    }
    __device__ float load_b(int p, int j) const {
        return TransposeB ? b[static_cast<size_t>(j) * k + p] : b[static_cast<size_t>(p) * n + j];
    }
    __device__ void store(int i, int j, float value, int) const {
        float& out = c[static_cast<size_t>(i) * n + j];
        // beta 0 must not read C: it may hold anything, NaN included.
        out = alpha * value + (beta == 0.0f ? 0.0f : beta * out);
    }
};

template <bool TransposeA, bool TransposeB>
void run(int m, int n, int k, float alpha, const float* a, const float* b, float beta, float* c) {
    launch_gemm(PlainOp<TransposeA, TransposeB>{m, n, k, alpha, beta, a, b, c}, m, n, k);
}

}

void gemm(bool ta, bool tb, int m, int n, int k, float alpha, const float* a, const float* b, float beta, float* c) {
    if (ta) {
        tb ? run<true, true>(m, n, k, alpha, a, b, beta, c) : run<true, false>(m, n, k, alpha, a, b, beta, c);
    } else {
        tb ? run<false, true>(m, n, k, alpha, a, b, beta, c) : run<false, false>(m, n, k, alpha, a, b, beta, c);
    }
}

}
