#pragma once

#include <cstddef>
#include <vector>

namespace cuda {

// NCHW. A dense layer is a tensor of shape {n, features, 1, 1}.
struct Shape {
    int n = 0;
    int c = 0;
    int h = 1;
    int w = 1;

    [[nodiscard]] size_t count() const { return static_cast<size_t>(n) * c * h * w; }
    bool operator==(const Shape&) const = default;
};

// A float array on the device. Moves, does not copy.
class Tensor {
public:
    Tensor() = default;
    explicit Tensor(Shape shape);
    ~Tensor();
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    // Sets the shape and reallocates only when the new one does not fit. The contents are unspecified afterwards.
    void resize(Shape shape);
    void zero();
    void upload(const std::vector<float>& values);
    // Becomes a copy of `other` (device to device).
    void copy_from(const Tensor& other);
    [[nodiscard]] std::vector<float> download() const;

    [[nodiscard]] Shape shape() const { return shape_; }
    [[nodiscard]] size_t size() const { return shape_.count(); }
    [[nodiscard]] float* data() { return data_; }
    [[nodiscard]] const float* data() const { return data_; }

private:
    void release();

    Shape shape_;
    float* data_ = nullptr;
    size_t capacity_ = 0;
};

// A learnable tensor with its gradient and the Adam moments (allocated by the optimizer on the first step).
struct Param {
    Tensor value;
    Tensor grad;
    Tensor m;
    Tensor v;

    void init(Shape shape) {
        value.resize(shape);
        value.zero();
        grad.resize(shape);
        grad.zero();
    }
    void zero_grad() { grad.zero(); }
};

}
