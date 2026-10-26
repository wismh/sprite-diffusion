#pragma once

namespace cuda {

// Row-major C (m x n) = alpha * op(A) (m x k) * op(B) (k x n) + beta * C, where op transposes when its flag is set.
// A is stored m x k (k x m when ta), B is stored k x n (n x k when tb). The pointers are on the device.
// The convolutions do not call this: they read their operands straight from the images (see conv2d.cu).
void gemm(bool ta, bool tb, int m, int n, int k, float alpha, const float* a, const float* b, float beta, float* c);

}
