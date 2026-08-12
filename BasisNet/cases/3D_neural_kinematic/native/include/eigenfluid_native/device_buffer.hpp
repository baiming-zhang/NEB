#pragma once

#include "eigenfluid_native/cuda_error.hpp"

#include <cstddef>
#include <utility>

namespace eigenfluid::native {

template <typename T>
class DeviceBuffer final {
 public:
  DeviceBuffer() = default;

  explicit DeviceBuffer(const std::size_t count) { allocate(count); }

  ~DeviceBuffer() noexcept { release_noexcept(); }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept { swap(other); }

  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      release_noexcept();
      swap(other);
    }
    return *this;
  }

  void allocate(const std::size_t count) {
    if (count == count_ && data_ != nullptr) {
      return;
    }
    reset();
    if (count == 0) {
      return;
    }
    EF_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_),
                             count * sizeof(T)));
    count_ = count;
  }

  void reset() {
    if (data_ != nullptr) {
      T* const old_data = data_;
      data_ = nullptr;
      count_ = 0;
      EF_CUDA_CHECK(cudaFree(old_data));
    }
  }

  void copy_from_host(const T* source, const std::size_t count) {
    require_capacity(count);
    EF_CUDA_CHECK(cudaMemcpy(data_, source, count * sizeof(T),
                             cudaMemcpyHostToDevice));
  }

  void copy_to_host(T* destination, const std::size_t count) const {
    require_capacity(count);
    EF_CUDA_CHECK(cudaMemcpy(destination, data_, count * sizeof(T),
                             cudaMemcpyDeviceToHost));
  }

  [[nodiscard]] T* data() noexcept { return data_; }
  [[nodiscard]] const T* data() const noexcept { return data_; }
  [[nodiscard]] std::size_t size() const noexcept { return count_; }

 private:
  void require_capacity(const std::size_t count) const {
    if (count > count_) {
      throw BackendError("DeviceBuffer copy exceeds allocated capacity");
    }
  }

  void release_noexcept() noexcept {
    if (data_ != nullptr) {
      static_cast<void>(cudaFree(data_));
      data_ = nullptr;
      count_ = 0;
    }
  }

  void swap(DeviceBuffer& other) noexcept {
    std::swap(data_, other.data_);
    std::swap(count_, other.count_);
  }

  T* data_ = nullptr;
  std::size_t count_ = 0;
};

class CublasHandle final {
 public:
  CublasHandle() { EF_CUBLAS_CHECK(cublasCreate(&handle_)); }

  ~CublasHandle() noexcept {
    if (handle_ != nullptr) {
      static_cast<void>(cublasDestroy(handle_));
    }
  }

  CublasHandle(const CublasHandle&) = delete;
  CublasHandle& operator=(const CublasHandle&) = delete;
  CublasHandle(CublasHandle&&) = delete;
  CublasHandle& operator=(CublasHandle&&) = delete;

  [[nodiscard]] cublasHandle_t get() const noexcept { return handle_; }

 private:
  cublasHandle_t handle_ = nullptr;
};

}  // namespace eigenfluid::native
