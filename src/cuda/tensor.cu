#include "cuda/tensor.h"

#include "cuda/check.cuh"

#include <utility>

namespace cuda {

Tensor::Tensor(Shape shape) {
    resize(shape);
}

Tensor::~Tensor() {
    release();
}

Tensor::Tensor(Tensor&& other) noexcept
    : shape_(other.shape_), data_(std::exchange(other.data_, nullptr)), capacity_(std::exchange(other.capacity_, 0)) {
    other.shape_ = Shape{};
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        release();
        shape_ = std::exchange(other.shape_, Shape{});
        data_ = std::exchange(other.data_, nullptr);
        capacity_ = std::exchange(other.capacity_, 0);
    }
    return *this;
}

void Tensor::release() {
    if (data_ != nullptr) {
        cudaFree(data_);
        data_ = nullptr;
    }
    capacity_ = 0;
}

void Tensor::resize(Shape shape) {
    const size_t count = shape.count();
    if (count > capacity_) {
        release();
        check(cudaMalloc(&data_, count * sizeof(float)));
        capacity_ = count;
    }
    shape_ = shape;
}

void Tensor::zero() {
    if (data_ != nullptr) {
        check(cudaMemset(data_, 0, size() * sizeof(float)));
    }
}

void Tensor::upload(const std::vector<float>& values) {
    if (values.size() != size()) {
        throw std::invalid_argument("Tensor::upload: size differs from the shape");
    }
    if (!values.empty()) {
        check(cudaMemcpy(data_, values.data(), values.size() * sizeof(float), cudaMemcpyHostToDevice));
    }
}

void Tensor::copy_from(const Tensor& other) {
    resize(other.shape());
    if (other.size() > 0) {
        check(cudaMemcpy(data_, other.data(), other.size() * sizeof(float), cudaMemcpyDeviceToDevice));
    }
}

std::vector<float> Tensor::download() const {
    std::vector<float> values(size());
    if (!values.empty()) {
        check(cudaMemcpy(values.data(), data_, values.size() * sizeof(float), cudaMemcpyDeviceToHost));
    }
    return values;
}

}
