#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>
#include <string>

namespace eigenfluid::native {

class BackendError final : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

inline const char* cublas_status_name(const cublasStatus_t status) noexcept {
  switch (status) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR:
      return "CUBLAS_STATUS_LICENSE_ERROR";
    default:
      return "CUBLAS_STATUS_UNKNOWN";
  }
}

inline void check_cuda(const cudaError_t status, const char* expression,
                       const char* file, const int line) {
  if (status == cudaSuccess) {
    return;
  }
  std::ostringstream message;
  message << "CUDA failure at " << file << ':' << line << " while evaluating "
          << expression << ": " << cudaGetErrorName(status) << " ("
          << cudaGetErrorString(status) << ')';
  throw BackendError(message.str());
}

inline void check_cublas(const cublasStatus_t status, const char* expression,
                         const char* file, const int line) {
  if (status == CUBLAS_STATUS_SUCCESS) {
    return;
  }
  std::ostringstream message;
  message << "cuBLAS failure at " << file << ':' << line
          << " while evaluating " << expression << ": "
          << cublas_status_name(status) << " (status="
          << static_cast<int>(status) << ')';
  throw BackendError(message.str());
}

}  // namespace eigenfluid::native

#define EF_CUDA_CHECK(expression)                                             \
  ::eigenfluid::native::check_cuda((expression), #expression, __FILE__,       \
                                   __LINE__)

#define EF_CUDA_CHECK_LAST() EF_CUDA_CHECK(cudaGetLastError())

#define EF_CUBLAS_CHECK(expression)                                           \
  ::eigenfluid::native::check_cublas((expression), #expression, __FILE__,     \
                                     __LINE__)
