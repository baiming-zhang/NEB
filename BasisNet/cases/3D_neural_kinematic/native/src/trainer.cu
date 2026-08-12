#include "eigenfluid_native/trainer.hpp"

#include "eigenfluid_native/cuda_error.hpp"
#include "eigenfluid_native/device_buffer.hpp"
#include "eigenfluid_native/jet3.cuh"
#include "eigenfluid_native/objective_contract.hpp"
#include "eigenfluid_native/training_contract.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace eigenfluid::native {
namespace {

constexpr int kThreads = 256;
constexpr int kModes = static_cast<int>(TrainingContract::modes);
constexpr int kPoints = static_cast<int>(TrainingContract::points_per_step);
constexpr int kHidden = static_cast<int>(TrainingContract::hidden);
constexpr int kInput = static_cast<int>(TrainingContract::input_features);
constexpr int kPotential =
    static_cast<int>(TrainingContract::potential_components);
constexpr int kOmegaDimension = 3 * kPoints;
constexpr int kGradOmegaDimension = 9 * kPoints;
constexpr int kParameterCount =
    kHidden * kInput + kHidden + 2 * (kHidden * kHidden + kHidden) +
    kPotential * kHidden + kPotential;
constexpr float kBoundaryEpsilon = 0.04F;
constexpr float kBoxLow = -kBoundaryEpsilon;
constexpr float kBoxWidth = 1.0F + 2.0F * kBoundaryEpsilon;
constexpr float kEnvelopeScale = 0.05F;
constexpr float kRoundedCubeRadius = 0.2F;
constexpr float kBlobK = 30.0F;

static_assert(kParameterCount == 4771);

std::size_t checked_product(std::initializer_list<std::size_t> values) {
  std::size_t result = 1;
  for (const std::size_t value : values) {
    if (value != 0 && result > std::numeric_limits<std::size_t>::max() / value) {
      throw BackendError("Native buffer size overflow");
    }
    result *= value;
  }
  return result;
}

__host__ __device__ std::uint64_t mix64(std::uint64_t value) {
  value += 0x9E3779B97F4A7C15ULL;
  value = (value ^ (value >> 30U)) * 0xBF58476D1CE4E5B9ULL;
  value = (value ^ (value >> 27U)) * 0x94D049BB133111EBULL;
  return value ^ (value >> 31U);
}

__device__ float uniform01(const std::uint64_t seed,
                           const std::uint64_t index) {
  const std::uint32_t bits = static_cast<std::uint32_t>(
      mix64(seed + index * 0xD1B54A32D192ED03ULL) >> 32U);
  return static_cast<float>(bits >> 8U) * (1.0F / 16777216.0F);
}

__host__ __device__ Jet3 jet_abs_branch(const Jet3& value) {
  return value.c[0] >= 0.0F ? value : jet_scale(value, -1.0F);
}

__host__ __device__ Jet3 jet_max_branch(const Jet3& a, const Jet3& b) {
  return a.c[0] >= b.c[0] ? a : b;
}

__host__ __device__ Jet3 jet_min_branch(const Jet3& a, const Jet3& b) {
  return a.c[0] <= b.c[0] ? a : b;
}

__device__ Jet3 rounded_cube_level(const Jet3& x, const Jet3& y,
                                   const Jet3& z) {
  const Jet3 zero = jet_constant(0.0F);
  const Jet3 qx = jet_add(jet_abs_branch(jet_add(x, jet_constant(-0.5F))),
                          jet_constant(-(0.5F - kRoundedCubeRadius)));
  const Jet3 qy = jet_add(jet_abs_branch(jet_add(y, jet_constant(-0.5F))),
                          jet_constant(-(0.5F - kRoundedCubeRadius)));
  const Jet3 qz = jet_add(jet_abs_branch(jet_add(z, jet_constant(-0.5F))),
                          jet_constant(-(0.5F - kRoundedCubeRadius)));
  const Jet3 ox = jet_max_branch(qx, zero);
  const Jet3 oy = jet_max_branch(qy, zero);
  const Jet3 oz = jet_max_branch(qz, zero);
  const Jet3 norm2 = jet_add(jet_add(jet_mul(ox, ox), jet_mul(oy, oy)),
                             jet_mul(oz, oz));
  Jet3 outside_norm = jet_zero();
  if (norm2.c[0] > 1.0e-20F) {
    outside_norm = jet_sqrt(norm2);
  }
  const Jet3 maxq = jet_max_branch(qx, jet_max_branch(qy, qz));
  const Jet3 sdf = jet_add(
      jet_add(outside_norm, jet_min_branch(maxq, zero)),
      jet_constant(-kRoundedCubeRadius));
  return jet_scale(sdf, -1.0F);
}

__device__ Jet3 obstacle_level(const Jet3& x, const Jet3& y, const Jet3& z,
                               const float* geometry) {
  Jet3 logits[10];
  int maximum = 0;
  for (int sphere = 0; sphere < 10; ++sphere) {
    const float* row = geometry + 4 * sphere;
    const Jet3 dx = jet_add(x, jet_constant(-row[0]));
    const Jet3 dy = jet_add(y, jet_constant(-row[1]));
    const Jet3 dz = jet_add(z, jet_constant(-row[2]));
    Jet3 level = jet_add(jet_add(jet_mul(dx, dx), jet_mul(dy, dy)),
                         jet_mul(dz, dz));
    level = jet_add(level, jet_constant(-row[3] * row[3]));
    logits[sphere] = jet_scale(level, -kBlobK);
    if (logits[sphere].c[0] > logits[maximum].c[0]) {
      maximum = sphere;
    }
  }
  Jet3 sum = jet_zero();
  for (int sphere = 0; sphere < 10; ++sphere) {
    sum = jet_add(sum, jet_exp(jet_sub(logits[sphere], logits[maximum])));
  }
  const Jet3 inside = jet_scale(jet_add(jet_log(sum), logits[maximum]),
                                1.0F / kBlobK);
  return jet_scale(inside, -1.0F);
}

__global__ void prepare_step_kernel(const std::uint64_t seed,
                                    const std::uint32_t step,
                                    const float* geometry, __half* input_value,
                                    float* input_derivative, Jet3* envelope,
                                    float* sqrt_weight) {
  const int point = blockIdx.x * blockDim.x + threadIdx.x;
  if (point >= kPoints) {
    return;
  }
  const std::uint64_t base =
      static_cast<std::uint64_t>(step) * 3ULL * kPoints + 3ULL * point;
  const float px = kBoxLow + kBoxWidth * uniform01(seed, base + 0);
  const float py = kBoxLow + kBoxWidth * uniform01(seed, base + 1);
  const float pz = kBoxLow + kBoxWidth * uniform01(seed, base + 2);
  const Jet3 coordinates[3] = {jet_variable(px, 0), jet_variable(py, 1),
                                jet_variable(pz, 2)};

  int feature = 0;
  for (int trig = 0; trig < 2; ++trig) {
    for (int axis = 0; axis < 3; ++axis) {
      const Jet3 normal = jet_scale(
          jet_add(coordinates[axis], jet_constant(-kBoxLow)),
          1.0F / kBoxWidth);
      for (int frequency = 0; frequency < 6; ++frequency) {
        const float angular =
            6.2831853071795864769F * static_cast<float>(1 << frequency);
        const Jet3 phase = jet_scale(normal, angular);
        const Jet3 value = trig == 0
                               ? jet_sin(phase)
                               : jet_sin(jet_add(phase, jet_constant(
                                                           1.5707963267948966F)));
        input_value[feature + kInput * point] = __float2half_rn(value.c[0]);
        for (int derivative = 0; derivative < kJetDerivativeSize;
             ++derivative) {
          input_derivative[derivative * kInput * kPoints + feature +
                           kInput * point] = value.c[1 + derivative];
        }
        ++feature;
      }
    }
  }
  for (int axis = 0; axis < 3; ++axis) {
    const Jet3 normal = jet_add(
        jet_scale(jet_add(coordinates[axis], jet_constant(-kBoxLow)),
                  2.0F / kBoxWidth),
        jet_constant(-1.0F));
    input_value[feature + kInput * point] = __float2half_rn(normal.c[0]);
    for (int derivative = 0; derivative < kJetDerivativeSize; ++derivative) {
      input_derivative[derivative * kInput * kPoints + feature +
                       kInput * point] = normal.c[1 + derivative];
    }
    ++feature;
  }
  for (int parameter = 0; parameter < 40; ++parameter) {
    input_value[feature + kInput * point] = __float2half_rn(geometry[parameter]);
    for (int derivative = 0; derivative < kJetDerivativeSize; ++derivative) {
      input_derivative[derivative * kInput * kPoints + feature +
                       kInput * point] = 0.0F;
    }
    ++feature;
  }

  const Jet3 outer =
      rounded_cube_level(coordinates[0], coordinates[1], coordinates[2]);
  const Jet3 obstacle = obstacle_level(coordinates[0], coordinates[1],
                                      coordinates[2], geometry);
  const Jet3 signed_product = jet_mul(
      jet_tanh(jet_scale(outer, 1.0F / kEnvelopeScale)),
      jet_tanh(jet_scale(obstacle, 1.0F / kEnvelopeScale)));
  envelope[point] = jet_mul(signed_product, signed_product);

  const float level = fminf(outer.c[0], obstacle.c[0]);
  float weight = 1.0F;
  if (level < 0.0F) {
    const float ratio = fminf(fabsf(level) / kBoundaryEpsilon, 1.0F);
    const float d = 2.0F * ratio * ratio * ratio -
                    3.0F * ratio * ratio + 1.0F;
    weight = fabsf(level) <= kBoundaryEpsilon ? d * d * d * d : 0.0F;
  }
  sqrt_weight[point] = sqrtf(weight * (kBoxWidth * kBoxWidth * kBoxWidth /
                                       static_cast<float>(kPoints)));
}

struct Layer final {
  int input = 0;
  int output = 0;
  DeviceBuffer<float> master_weight;
  DeviceBuffer<float> master_bias;
  DeviceBuffer<__half> weight_half;
  DeviceBuffer<__half> bias_half;
  DeviceBuffer<float> weight_quantized;
  DeviceBuffer<float> gradient_weight;
  DeviceBuffer<float> gradient_bias;
  DeviceBuffer<float> first_weight;
  DeviceBuffer<float> second_weight;
  DeviceBuffer<float> first_bias;
  DeviceBuffer<float> second_bias;

  Layer() = default;
  Layer(const int input_size, const int output_size)
      : input(input_size),
        output(output_size),
        master_weight(checked_product(
            {static_cast<std::size_t>(kModes), static_cast<std::size_t>(input),
             static_cast<std::size_t>(output)})),
        master_bias(checked_product({static_cast<std::size_t>(kModes),
                                    static_cast<std::size_t>(output)})),
        weight_half(master_weight.size()),
        bias_half(master_bias.size()),
        weight_quantized(master_weight.size()),
        gradient_weight(master_weight.size()),
        gradient_bias(master_bias.size()),
        first_weight(master_weight.size()),
        second_weight(master_weight.size()),
        first_bias(master_bias.size()),
        second_bias(master_bias.size()) {}
};

struct Activation final {
  int width = 0;
  DeviceBuffer<__half> pre_value;
  DeviceBuffer<__half> post_value;
  DeviceBuffer<float> pre_derivative;
  DeviceBuffer<float> post_derivative;

  explicit Activation(const int layer_width)
      : width(layer_width),
        pre_value(checked_product({static_cast<std::size_t>(kModes),
                                   static_cast<std::size_t>(width),
                                   static_cast<std::size_t>(kPoints)})),
        post_value(pre_value.size()),
        pre_derivative(checked_product(
            {static_cast<std::size_t>(kJetDerivativeSize),
             static_cast<std::size_t>(kModes), static_cast<std::size_t>(width),
             static_cast<std::size_t>(kPoints)})),
        post_derivative(pre_derivative.size()) {}
};

__global__ void quantize_parameters_kernel(
    const float* master, __half* half, float* quantized,
    const std::size_t count) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    const __half value = __float2half_rn(master[index]);
    half[index] = value;
    quantized[index] = __half2float(value);
  }
}

__global__ void quantize_bias_kernel(const float* master, __half* half,
                                     const std::size_t count) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    half[index] = __float2half_rn(master[index]);
  }
}

void quantize_layer(Layer& layer) {
  const int weight_blocks = static_cast<int>(
      (layer.master_weight.size() + kThreads - 1) / kThreads);
  quantize_parameters_kernel<<<weight_blocks, kThreads>>>(
      layer.master_weight.data(), layer.weight_half.data(),
      layer.weight_quantized.data(), layer.master_weight.size());
  const int bias_blocks = static_cast<int>(
      (layer.master_bias.size() + kThreads - 1) / kThreads);
  quantize_bias_kernel<<<bias_blocks, kThreads>>>(
      layer.master_bias.data(), layer.bias_half.data(),
      layer.master_bias.size());
  EF_CUDA_CHECK_LAST();
}

__global__ void sine_activation_kernel(
    __half* pre_value, const float* pre_derivative, __half* post_value,
    float* post_derivative, const __half* bias, const int width) {
  const std::size_t item =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t items =
      static_cast<std::size_t>(kModes) * width * kPoints;
  if (item >= items) {
    return;
  }
  const int mode = static_cast<int>(item / (width * kPoints));
  const int local = static_cast<int>(item % (width * kPoints));
  const int neuron = local % width;
  Jet3 input = jet_zero();
  input.c[0] = __half2float(pre_value[item]) +
               __half2float(bias[mode * width + neuron]);
  pre_value[item] = __float2half_rn(input.c[0]);
  for (int derivative = 0; derivative < kJetDerivativeSize; ++derivative) {
    input.c[1 + derivative] = pre_derivative[derivative * items + item];
  }
  const Jet3 output = jet_sin(input);
  post_value[item] = __float2half_rn(output.c[0]);
  for (int derivative = 0; derivative < kJetDerivativeSize; ++derivative) {
    post_derivative[derivative * items + item] = output.c[1 + derivative];
  }
}

__global__ void output_bias_kernel(__half* value, const __half* bias) {
  const std::size_t item =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t items =
      static_cast<std::size_t>(kModes) * kPotential * kPoints;
  if (item < items) {
    const int mode = static_cast<int>(item / (kPotential * kPoints));
    const int local = static_cast<int>(item % (kPotential * kPoints));
    const int component = local % kPotential;
    value[item] = __float2half_rn(
        __half2float(value[item]) +
        __half2float(bias[mode * kPotential + component]));
  }
}

void gemm_half_forward(CublasHandle& handle, const Layer& layer,
                       const __half* input, const long long input_stride,
                       __half* output) {
  const float alpha = 1.0F;
  const float beta = 0.0F;
  EF_CUBLAS_CHECK(cublasGemmStridedBatchedEx(
      handle.get(), CUBLAS_OP_N, CUBLAS_OP_N, layer.output, kPoints,
      layer.input, &alpha, layer.weight_half.data(), CUDA_R_16F, layer.output,
      static_cast<long long>(layer.output) * layer.input, input, CUDA_R_16F,
      layer.input, input_stride, &beta, output, CUDA_R_16F, layer.output,
      static_cast<long long>(layer.output) * kPoints, kModes, CUDA_R_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

void gemm_derivative_forward(CublasHandle& handle, const Layer& layer,
                             const float* input,
                             const long long input_mode_stride,
                             float* output) {
  const float alpha = 1.0F;
  const float beta = 0.0F;
  const std::size_t input_component_stride =
      input_mode_stride == 0
          ? static_cast<std::size_t>(layer.input) * kPoints
          : static_cast<std::size_t>(kModes) * input_mode_stride;
  const std::size_t output_component_stride =
      static_cast<std::size_t>(kModes) * layer.output * kPoints;
  for (int derivative = 0; derivative < kJetDerivativeSize; ++derivative) {
    EF_CUBLAS_CHECK(cublasSgemmStridedBatched(
        handle.get(), CUBLAS_OP_N, CUBLAS_OP_N, layer.output, kPoints,
        layer.input, &alpha, layer.weight_quantized.data(), layer.output,
        static_cast<long long>(layer.output) * layer.input,
        input + derivative * input_component_stride, layer.input,
        input_mode_stride, &beta, output + derivative * output_component_stride,
        layer.output, static_cast<long long>(layer.output) * kPoints, kModes));
  }
}

void forward_hidden(CublasHandle& handle, const Layer& layer,
                    const __half* input_value, const float* input_derivative,
                    const long long input_mode_value_stride,
                    const long long input_mode_derivative_stride,
                    Activation& output) {
  gemm_half_forward(handle, layer, input_value, input_mode_value_stride,
                    output.pre_value.data());
  gemm_derivative_forward(handle, layer, input_derivative,
                          input_mode_derivative_stride,
                          output.pre_derivative.data());
  const std::size_t items = output.pre_value.size();
  sine_activation_kernel<<<static_cast<int>((items + kThreads - 1) / kThreads),
                           kThreads>>>(
      output.pre_value.data(), output.pre_derivative.data(),
      output.post_value.data(), output.post_derivative.data(),
      layer.bias_half.data(), layer.output);
  EF_CUDA_CHECK_LAST();
}

__global__ void physical_forward_kernel(
    const __half* raw_value, const float* raw_derivative,
    const Jet3* envelope, const float* sqrt_weight, float* raw_omega,
    float* raw_grad_omega) {
  const std::size_t item =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t items = static_cast<std::size_t>(kModes) * kPoints;
  if (item >= items) {
    return;
  }
  const int mode = static_cast<int>(item / kPoints);
  const int point = static_cast<int>(item % kPoints);
  Jet3 potential[3];
  const std::size_t field_items =
      static_cast<std::size_t>(kModes) * kPotential * kPoints;
  for (int component = 0; component < 3; ++component) {
    const std::size_t offset =
        static_cast<std::size_t>(mode) * kPotential * kPoints +
        component + kPotential * point;
    Jet3 raw = jet_zero();
    raw.c[0] = __half2float(raw_value[offset]);
    for (int derivative = 0; derivative < kJetDerivativeSize; ++derivative) {
      raw.c[1 + derivative] =
          raw_derivative[derivative * field_items + offset];
    }
    potential[component] = jet_mul(envelope[point], raw);
  }
  const Jet3& a = potential[0];
  const Jet3& b = potential[1];
  const Jet3& c = potential[2];
  float omega[3];
  omega[0] = jet_actual_derivative(a, 6) -
             jet_actual_derivative(b, 5) -
             jet_actual_derivative(c, 7) -
             jet_actual_derivative(c, 9);
  omega[1] = jet_actual_derivative(a, 8) +
             jet_actual_derivative(b, 4) +
             jet_actual_derivative(b, 9) +
             jet_actual_derivative(c, 5);
  omega[2] = -jet_actual_derivative(a, 4) -
             jet_actual_derivative(a, 7) -
             jet_actual_derivative(b, 8) +
             jet_actual_derivative(c, 6);
  constexpr int third[3][6] = {{10, 11, 12, 13, 14, 15},
                               {11, 13, 14, 16, 17, 18},
                               {12, 14, 15, 17, 18, 19}};
  const float scale = sqrt_weight[point];
  for (int component = 0; component < 3; ++component) {
    raw_omega[mode * kOmegaDimension + component * kPoints + point] =
        scale * omega[component];
  }
  for (int axis = 0; axis < 3; ++axis) {
    const int* d = third[axis];
    const float grad0 = jet_actual_derivative(a, d[2]) -
                        jet_actual_derivative(b, d[1]) -
                        jet_actual_derivative(c, d[3]) -
                        jet_actual_derivative(c, d[5]);
    const float grad1 = jet_actual_derivative(a, d[4]) +
                        jet_actual_derivative(b, d[0]) +
                        jet_actual_derivative(b, d[5]) +
                        jet_actual_derivative(c, d[1]);
    const float grad2 = -jet_actual_derivative(a, d[0]) -
                        jet_actual_derivative(a, d[3]) -
                        jet_actual_derivative(b, d[4]) +
                        jet_actual_derivative(c, d[2]);
    const float values[3] = {grad0, grad1, grad2};
    for (int component = 0; component < 3; ++component) {
      raw_grad_omega[mode * kGradOmegaDimension +
                     (component * 3 + axis) * kPoints + point] =
          scale * values[component];
    }
  }
}

void check_finite_positive(const float value, const char* name,
                           const int mode) {
  if (!(value > 0.0F) || !std::isfinite(value)) {
    throw BackendError(std::string(name) + " is invalid for mode " +
                       std::to_string(mode) + ": " +
                       std::to_string(value));
  }
}

void check_cusolver(const cusolverStatus_t status, const char* expression) {
  if (status != CUSOLVER_STATUS_SUCCESS) {
    throw BackendError(std::string("cuSOLVER failure while evaluating ") +
                       expression + ": status=" +
                       std::to_string(static_cast<int>(status)));
  }
}

__global__ void peeled_column_statistics_kernel(const float* r, const float* g,
                                                float* roots, float* losses) {
  const int mode = blockIdx.x;
  __shared__ float reduction[256];
  float sum = 0.0F;
  for (int index = threadIdx.x; index < kGradOmegaDimension;
       index += blockDim.x) {
    const float value = g[mode * kGradOmegaDimension + index];
    sum += value * value;
  }
  reduction[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      reduction[threadIdx.x] += reduction[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    roots[mode] = r[mode + kModes * mode];
    losses[mode] = reduction[0];
  }
}

float peel_forward(CublasHandle& handle, cusolverDnHandle_t solver,
                   const float* raw_omega, const float* raw_grad_omega,
                   float* q, float* g, float* r, float* solver_workspace,
                   const int solver_workspace_count, int* solver_info,
                   DeviceBuffer<float>& device_roots,
                   DeviceBuffer<float>& device_losses,
                   std::vector<float>& roots, std::vector<float>& losses) {
  EF_CUDA_CHECK(cudaMemcpy(q, raw_omega,
                           sizeof(float) * static_cast<std::size_t>(kModes) *
                               kOmegaDimension,
                           cudaMemcpyDeviceToDevice));
  const float one = 1.0F;
  const float zero = 0.0F;
  EF_CUBLAS_CHECK(cublasSsyrk(handle.get(), CUBLAS_FILL_MODE_UPPER,
                              CUBLAS_OP_T, kModes, kOmegaDimension, &one,
                              raw_omega, kOmegaDimension, &zero, r, kModes));
  check_cusolver(cusolverDnSpotrf(solver, CUBLAS_FILL_MODE_UPPER, kModes, r,
                                  kModes, solver_workspace,
                                  solver_workspace_count, solver_info),
                 "cusolverDnSpotrf");
  int info = 0;
  EF_CUDA_CHECK(cudaMemcpy(&info, solver_info, sizeof(info),
                           cudaMemcpyDeviceToHost));
  if (info != 0) {
    throw BackendError("Full causal peeling Cholesky failed at leading minor " +
                       std::to_string(info));
  }
  EF_CUBLAS_CHECK(cublasStrsm(handle.get(), CUBLAS_SIDE_RIGHT,
                              CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                              CUBLAS_DIAG_NON_UNIT, kOmegaDimension, kModes,
                              &one, r, kModes, q, kOmegaDimension));
  EF_CUDA_CHECK(cudaMemcpy(g, raw_grad_omega,
                           sizeof(float) * static_cast<std::size_t>(kModes) *
                               kGradOmegaDimension,
                           cudaMemcpyDeviceToDevice));
  EF_CUBLAS_CHECK(cublasStrsm(handle.get(), CUBLAS_SIDE_RIGHT,
                              CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                              CUBLAS_DIAG_NON_UNIT, kGradOmegaDimension,
                              kModes, &one, r, kModes, g,
                              kGradOmegaDimension));
  peeled_column_statistics_kernel<<<kModes, 256>>>(
      r, g, device_roots.data(), device_losses.data());
  EF_CUDA_CHECK_LAST();
  device_roots.copy_to_host(roots.data(), roots.size());
  device_losses.copy_to_host(losses.data(), losses.size());
  float total = 0.0F;
  for (int mode = 0; mode < kModes; ++mode) {
    check_finite_positive(roots[mode], "peeled vorticity norm", mode);
    if (!(losses[mode] >= 0.0F) || !std::isfinite(losses[mode])) {
      throw BackendError("peeled vorticity Rayleigh quotient is invalid for mode " +
                         std::to_string(mode));
    }
    total += losses[mode];
  }
  return total;
}

__global__ void initialize_raw_adjoint_kernel(
    const float* q, const float* g, const float* root, const float* loss,
    float* omega_adjoint, float* grad_adjoint) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t omega_items =
      static_cast<std::size_t>(kModes) * kOmegaDimension;
  const std::size_t grad_items =
      static_cast<std::size_t>(kModes) * kGradOmegaDimension;
  if (index < omega_items) {
    const int mode = static_cast<int>(index / kOmegaDimension);
    omega_adjoint[index] =
        -2.0F * loss[mode] / root[mode] * q[index];
  }
  if (index < grad_items) {
    const int mode = static_cast<int>(index / kGradOmegaDimension);
    grad_adjoint[index] =
        2.0F / root[mode] * g[index];
  }
}

__global__ void strict_upper_kernel(float* matrix) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < static_cast<std::size_t>(kModes) * kModes) {
    const int row = static_cast<int>(index % kModes);
    const int column = static_cast<int>(index / kModes);
    if (row >= column) {
      matrix[index] = 0.0F;
    }
  }
}

void peel_reverse(CublasHandle& handle, const float* q, const float* g,
                  const std::vector<float>& roots,
                  const std::vector<float>& losses, float* omega_adjoint,
                  float* grad_adjoint, float* coefficient_adjoint,
                  DeviceBuffer<float>& device_roots,
                  DeviceBuffer<float>& device_loss) {
  device_roots.copy_from_host(roots.data(), roots.size());
  device_loss.copy_from_host(losses.data(), losses.size());
  const std::size_t grad_items =
      static_cast<std::size_t>(kModes) * kGradOmegaDimension;
  initialize_raw_adjoint_kernel<<<
      static_cast<int>((grad_items + kThreads - 1) / kThreads), kThreads>>>(
      q, g, device_roots.data(), device_loss.data(), omega_adjoint,
      grad_adjoint);
  EF_CUDA_CHECK_LAST();
  const float one = 1.0F;
  const float minus_one = -1.0F;
  const float zero = 0.0F;
  EF_CUBLAS_CHECK(cublasSgemm(handle.get(), CUBLAS_OP_T, CUBLAS_OP_N,
                              kModes, kModes, kOmegaDimension, &minus_one, q,
                              kOmegaDimension, omega_adjoint, kOmegaDimension,
                              &zero, coefficient_adjoint, kModes));
  EF_CUBLAS_CHECK(cublasSgemm(handle.get(), CUBLAS_OP_T, CUBLAS_OP_N,
                              kModes, kModes, kGradOmegaDimension, &minus_one,
                              g, kGradOmegaDimension, grad_adjoint,
                              kGradOmegaDimension, &one, coefficient_adjoint,
                              kModes));
  const std::size_t matrix_items =
      static_cast<std::size_t>(kModes) * kModes;
  strict_upper_kernel<<<
      static_cast<int>((matrix_items + kThreads - 1) / kThreads), kThreads>>>(
      coefficient_adjoint);
  EF_CUDA_CHECK_LAST();
  EF_CUBLAS_CHECK(cublasSgemm(handle.get(), CUBLAS_OP_N, CUBLAS_OP_N,
                              kOmegaDimension, kModes, kModes, &one, q,
                              kOmegaDimension, coefficient_adjoint, kModes,
                              &one, omega_adjoint, kOmegaDimension));
}

__device__ void add_actual_derivative_adjoint(Jet3& jet, const int index,
                                               const float value) {
  jet.c[index] += jet_factorial(index) * value;
}

__global__ void physical_reverse_kernel(
    const __half* raw_value, const float* raw_derivative,
    const Jet3* envelope, const float* sqrt_weight,
    const float* omega_adjoint, const float* grad_adjoint,
    float* potential_adjoint) {
  const std::size_t item =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t items = static_cast<std::size_t>(kModes) * kPoints;
  if (item >= items) {
    return;
  }
  const int mode = static_cast<int>(item / kPoints);
  const int point = static_cast<int>(item % kPoints);
  Jet3 a_adj = jet_zero();
  Jet3 b_adj = jet_zero();
  Jet3 c_adj = jet_zero();
  const float scale = sqrt_weight[point];
  const float w0 = scale *
                   omega_adjoint[mode * kOmegaDimension + point];
  const float w1 = scale *
                   omega_adjoint[mode * kOmegaDimension + kPoints + point];
  const float w2 = scale *
                   omega_adjoint[mode * kOmegaDimension + 2 * kPoints + point];
  add_actual_derivative_adjoint(a_adj, 6, w0);
  add_actual_derivative_adjoint(b_adj, 5, -w0);
  add_actual_derivative_adjoint(c_adj, 7, -w0);
  add_actual_derivative_adjoint(c_adj, 9, -w0);
  add_actual_derivative_adjoint(a_adj, 8, w1);
  add_actual_derivative_adjoint(b_adj, 4, w1);
  add_actual_derivative_adjoint(b_adj, 9, w1);
  add_actual_derivative_adjoint(c_adj, 5, w1);
  add_actual_derivative_adjoint(a_adj, 4, -w2);
  add_actual_derivative_adjoint(a_adj, 7, -w2);
  add_actual_derivative_adjoint(b_adj, 8, -w2);
  add_actual_derivative_adjoint(c_adj, 6, w2);
  constexpr int third[3][6] = {{10, 11, 12, 13, 14, 15},
                               {11, 13, 14, 16, 17, 18},
                               {12, 14, 15, 17, 18, 19}};
  for (int axis = 0; axis < 3; ++axis) {
    const int* d = third[axis];
    const float v0 = scale *
        grad_adjoint[mode * kGradOmegaDimension + axis * kPoints + point];
    const float v1 = scale * grad_adjoint[
        mode * kGradOmegaDimension + (3 + axis) * kPoints + point];
    const float v2 = scale * grad_adjoint[
        mode * kGradOmegaDimension + (6 + axis) * kPoints + point];
    add_actual_derivative_adjoint(a_adj, d[2], v0);
    add_actual_derivative_adjoint(b_adj, d[1], -v0);
    add_actual_derivative_adjoint(c_adj, d[3], -v0);
    add_actual_derivative_adjoint(c_adj, d[5], -v0);
    add_actual_derivative_adjoint(a_adj, d[4], v1);
    add_actual_derivative_adjoint(b_adj, d[0], v1);
    add_actual_derivative_adjoint(b_adj, d[5], v1);
    add_actual_derivative_adjoint(c_adj, d[1], v1);
    add_actual_derivative_adjoint(a_adj, d[0], -v2);
    add_actual_derivative_adjoint(a_adj, d[3], -v2);
    add_actual_derivative_adjoint(b_adj, d[4], -v2);
    add_actual_derivative_adjoint(c_adj, d[2], v2);
  }
  const Jet3 adjoints[3] = {a_adj, b_adj, c_adj};
  const std::size_t field_items =
      static_cast<std::size_t>(kModes) * kPotential * kPoints;
  for (int component = 0; component < 3; ++component) {
    const std::size_t offset =
        static_cast<std::size_t>(mode) * kPotential * kPoints + component +
        kPotential * point;
    Jet3 raw = jet_zero();
    raw.c[0] = __half2float(raw_value[offset]);
    for (int derivative = 0; derivative < kJetDerivativeSize; ++derivative) {
      raw.c[1 + derivative] =
          raw_derivative[derivative * field_items + offset];
    }
    Jet3 envelope_adjoint = jet_zero();
    Jet3 raw_adjoint = jet_zero();
    jet_mul_reverse(envelope[point], raw, adjoints[component],
                    envelope_adjoint, raw_adjoint);
    for (int jet = 0; jet < kJetSize; ++jet) {
      potential_adjoint[jet * field_items + offset] = raw_adjoint.c[jet];
    }
  }
}

__global__ void sine_reverse_kernel(
    const __half* pre_value, const float* pre_derivative,
    const float* post_adjoint, float* pre_adjoint, const int width) {
  const std::size_t item =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t items =
      static_cast<std::size_t>(kModes) * width * kPoints;
  if (item >= items) {
    return;
  }
  Jet3 input = jet_zero();
  Jet3 output_adjoint = jet_zero();
  input.c[0] = __half2float(pre_value[item]);
  for (int jet = 0; jet < kJetSize; ++jet) {
    output_adjoint.c[jet] = post_adjoint[jet * items + item];
    if (jet > 0) {
      input.c[jet] = pre_derivative[(jet - 1) * items + item];
    }
  }
  const Jet3 input_adjoint = jet_sin_reverse(input, output_adjoint);
  for (int jet = 0; jet < kJetSize; ++jet) {
    pre_adjoint[jet * items + item] = input_adjoint.c[jet];
  }
}

__global__ void bias_gradient_kernel(const float* adjoint,
                                     float* gradient, const int width) {
  const int item = blockIdx.x * blockDim.x + threadIdx.x;
  const int count = kModes * width;
  if (item >= count) {
    return;
  }
  const int mode = item / width;
  const int neuron = item % width;
  float sum = 0.0F;
  const std::size_t base =
      static_cast<std::size_t>(mode) * width * kPoints + neuron;
  for (int point = 0; point < kPoints; ++point) {
    sum += adjoint[base + static_cast<std::size_t>(width) * point];
  }
  gradient[item] = sum;
}

__global__ void half_to_float_kernel(const __half* input, float* output,
                                     const std::size_t count) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    output[index] = __half2float(input[index]);
  }
}

void weight_gradient(CublasHandle& handle, Layer& layer,
                     const float* output_adjoint, const float* input_value,
                     const float* input_derivative,
                     const long long input_mode_stride) {
  const float alpha = 1.0F;
  float beta = 0.0F;
  const long long output_mode_stride =
      static_cast<long long>(layer.output) * kPoints;
  const std::size_t output_component_stride =
      static_cast<std::size_t>(kModes) * output_mode_stride;
  const std::size_t input_component_stride =
      input_mode_stride == 0
          ? static_cast<std::size_t>(layer.input) * kPoints
          : static_cast<std::size_t>(kModes) * input_mode_stride;
  EF_CUBLAS_CHECK(cublasSgemmStridedBatched(
      handle.get(), CUBLAS_OP_N, CUBLAS_OP_T, layer.output, layer.input,
      kPoints, &alpha, output_adjoint, layer.output, output_mode_stride,
      input_value, layer.input, input_mode_stride, &beta,
      layer.gradient_weight.data(), layer.output,
      static_cast<long long>(layer.output) * layer.input, kModes));
  beta = 1.0F;
  for (int derivative = 0; derivative < kJetDerivativeSize; ++derivative) {
    EF_CUBLAS_CHECK(cublasSgemmStridedBatched(
        handle.get(), CUBLAS_OP_N, CUBLAS_OP_T, layer.output, layer.input,
        kPoints, &alpha,
        output_adjoint + (1 + derivative) * output_component_stride,
        layer.output, output_mode_stride,
        input_derivative + derivative * input_component_stride, layer.input,
        input_mode_stride, &beta, layer.gradient_weight.data(), layer.output,
        static_cast<long long>(layer.output) * layer.input, kModes));
  }
  bias_gradient_kernel<<<
      (kModes * layer.output + kThreads - 1) / kThreads, kThreads>>>(
      output_adjoint, layer.gradient_bias.data(), layer.output);
  EF_CUDA_CHECK_LAST();
}

void linear_reverse(CublasHandle& handle, const Layer& layer,
                    const float* output_adjoint, float* input_adjoint) {
  const float alpha = 1.0F;
  const float beta = 0.0F;
  const long long output_mode_stride =
      static_cast<long long>(layer.output) * kPoints;
  const long long input_mode_stride =
      static_cast<long long>(layer.input) * kPoints;
  const std::size_t output_component_stride =
      static_cast<std::size_t>(kModes) * output_mode_stride;
  const std::size_t input_component_stride =
      static_cast<std::size_t>(kModes) * input_mode_stride;
  for (int jet = 0; jet < kJetSize; ++jet) {
    EF_CUBLAS_CHECK(cublasSgemmStridedBatched(
        handle.get(), CUBLAS_OP_T, CUBLAS_OP_N, layer.input, kPoints,
        layer.output, &alpha, layer.weight_quantized.data(), layer.output,
        static_cast<long long>(layer.output) * layer.input,
        output_adjoint + jet * output_component_stride, layer.output,
        output_mode_stride, &beta,
        input_adjoint + jet * input_component_stride, layer.input,
        input_mode_stride, kModes));
  }
}

__global__ void adam_kernel(float* parameter, const float* gradient,
                            float* first, float* second,
                            const std::size_t count, const float correction1,
                            const float correction2) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  const float grad = gradient[index];
  const float m = TrainingContract::adam_beta1 * first[index] +
                  (1.0F - TrainingContract::adam_beta1) * grad;
  const float v = TrainingContract::adam_beta2 * second[index] +
                  (1.0F - TrainingContract::adam_beta2) * grad * grad;
  first[index] = m;
  second[index] = v;
  parameter[index] -= TrainingContract::learning_rate * (m / correction1) /
                      (sqrtf(v / correction2) +
                       TrainingContract::adam_epsilon);
}

void adam_layer(Layer& layer, const std::uint32_t step) {
  const float correction1 =
      1.0F - std::pow(TrainingContract::adam_beta1, static_cast<float>(step));
  const float correction2 =
      1.0F - std::pow(TrainingContract::adam_beta2, static_cast<float>(step));
  const int weight_blocks = static_cast<int>(
      (layer.master_weight.size() + kThreads - 1) / kThreads);
  adam_kernel<<<weight_blocks, kThreads>>>(
      layer.master_weight.data(), layer.gradient_weight.data(),
      layer.first_weight.data(), layer.second_weight.data(),
      layer.master_weight.size(), correction1, correction2);
  const int bias_blocks = static_cast<int>(
      (layer.master_bias.size() + kThreads - 1) / kThreads);
  adam_kernel<<<bias_blocks, kThreads>>>(
      layer.master_bias.data(), layer.gradient_bias.data(),
      layer.first_bias.data(), layer.second_bias.data(),
      layer.master_bias.size(), correction1, correction2);
  EF_CUDA_CHECK_LAST();
  quantize_layer(layer);
}

struct Trainer final {
  explicit Trainer(const RuntimeConfig& runtime)
      : config(runtime),
        layer0(kInput, kHidden),
        layer1(kHidden, kHidden),
        layer2(kHidden, kHidden),
        output(kHidden, kPotential),
        activation0(kHidden),
        activation1(kHidden),
        activation2(kHidden),
        input_value(static_cast<std::size_t>(kInput) * kPoints),
        input_derivative(static_cast<std::size_t>(kJetDerivativeSize) *
                         kInput * kPoints),
        geometry(40),
        envelope(kPoints),
        sqrt_weight(kPoints),
        potential_value(static_cast<std::size_t>(kModes) * kPotential *
                        kPoints),
        potential_derivative(
            static_cast<std::size_t>(kJetDerivativeSize) * kModes *
            kPotential * kPoints),
        raw_omega(static_cast<std::size_t>(kModes) * kOmegaDimension),
        raw_grad_omega(static_cast<std::size_t>(kModes) *
                       kGradOmegaDimension),
        q(raw_omega.size()),
        g(raw_grad_omega.size()),
        coefficients(static_cast<std::size_t>(kModes) * kModes),
        omega_adjoint(raw_omega.size()),
        grad_adjoint(raw_grad_omega.size()),
        coefficient_adjoint(coefficients.size()),
        potential_adjoint(static_cast<std::size_t>(kJetSize) * kModes *
                          kPotential * kPoints),
        hidden_adjoint_a(static_cast<std::size_t>(kJetSize) * kModes *
                         kHidden * kPoints),
        hidden_adjoint_b(hidden_adjoint_a.size()),
        backward_input_value(checked_product(
            {static_cast<std::size_t>(kModes),
             static_cast<std::size_t>(kHidden),
             static_cast<std::size_t>(kPoints)})),
        device_roots(kModes),
        device_loss(kModes),
        solver_info(1),
        roots(kModes),
        losses(kModes) {
    check_cusolver(cusolverDnCreate(&solver), "cusolverDnCreate");
  }

  ~Trainer() {
    if (solver != nullptr) {
      static_cast<void>(cusolverDnDestroy(solver));
    }
  }

  RuntimeConfig config;
  CublasHandle handle;
  Layer layer0;
  Layer layer1;
  Layer layer2;
  Layer output;
  Activation activation0;
  Activation activation1;
  Activation activation2;
  DeviceBuffer<__half> input_value;
  DeviceBuffer<float> input_derivative;
  DeviceBuffer<float> geometry;
  DeviceBuffer<Jet3> envelope;
  DeviceBuffer<float> sqrt_weight;
  DeviceBuffer<__half> potential_value;
  DeviceBuffer<float> potential_derivative;
  DeviceBuffer<float> raw_omega;
  DeviceBuffer<float> raw_grad_omega;
  DeviceBuffer<float> q;
  DeviceBuffer<float> g;
  DeviceBuffer<float> coefficients;
  DeviceBuffer<float> omega_adjoint;
  DeviceBuffer<float> grad_adjoint;
  DeviceBuffer<float> coefficient_adjoint;
  DeviceBuffer<float> potential_adjoint;
  DeviceBuffer<float> hidden_adjoint_a;
  DeviceBuffer<float> hidden_adjoint_b;
  DeviceBuffer<float> backward_input_value;
  DeviceBuffer<float> device_roots;
  DeviceBuffer<float> device_loss;
  DeviceBuffer<int> solver_info;
  DeviceBuffer<float> solver_workspace;
  cusolverDnHandle_t solver = nullptr;
  int solver_workspace_count = 0;
  std::vector<float> roots;
  std::vector<float> losses;
  std::vector<std::array<float, 40>> geometries;
  std::filesystem::path run_dir;
  std::uint32_t start_step = 0;

  void initialize() {
    run_dir = std::filesystem::absolute(config.run_dir);
    std::filesystem::create_directories(run_dir);
    EF_CUBLAS_CHECK(cublasSetMathMode(handle.get(), CUBLAS_DEFAULT_MATH));
    EF_CUBLAS_CHECK(
        cublasSetAtomicsMode(handle.get(), CUBLAS_ATOMICS_NOT_ALLOWED));
    check_cusolver(cusolverDnSpotrf_bufferSize(
                       solver, CUBLAS_FILL_MODE_UPPER, kModes,
                       coefficients.data(), kModes, &solver_workspace_count),
                   "cusolverDnSpotrf_bufferSize");
    solver_workspace.allocate(static_cast<std::size_t>(solver_workspace_count));
    initialize_geometries();
    initialize_layer(layer0, config.seed ^ 0x101ULL);
    initialize_layer(layer1, config.seed ^ 0x202ULL);
    initialize_layer(layer2, config.seed ^ 0x303ULL);
    initialize_layer(output, config.seed ^ 0x404ULL);
    load_checkpoint_if_present();
    write_manifest();
  }

  void initialize_geometries() {
    std::mt19937_64 generator(config.seed ^ 0xBADC0FFEEULL);
    std::uniform_real_distribution<float> center(0.225F, 0.675F);
    std::uniform_real_distribution<float> radius(0.03F, 0.09F);
    geometries.resize(1000);
    for (auto& item : geometries) {
      for (int sphere = 0; sphere < 10; ++sphere) {
        item[4 * sphere + 0] = center(generator);
        item[4 * sphere + 1] = center(generator);
        item[4 * sphere + 2] = center(generator);
        item[4 * sphere + 3] = radius(generator);
      }
    }
  }

  void initialize_layer(Layer& layer, const std::uint64_t seed) {
    std::mt19937_64 generator(seed);
    const float limit = std::sqrt(6.0F / (layer.input + layer.output));
    std::uniform_real_distribution<float> distribution(-limit, limit);
    std::vector<float> weights(layer.master_weight.size());
    std::vector<float> biases(layer.master_bias.size(), 0.0F);
    for (float& value : weights) {
      value = distribution(generator);
    }
    layer.master_weight.copy_from_host(weights.data(), weights.size());
    layer.master_bias.copy_from_host(biases.data(), biases.size());
    EF_CUDA_CHECK(cudaMemset(layer.first_weight.data(), 0,
                             layer.first_weight.size() * sizeof(float)));
    EF_CUDA_CHECK(cudaMemset(layer.second_weight.data(), 0,
                             layer.second_weight.size() * sizeof(float)));
    EF_CUDA_CHECK(cudaMemset(layer.first_bias.data(), 0,
                             layer.first_bias.size() * sizeof(float)));
    EF_CUDA_CHECK(cudaMemset(layer.second_bias.data(), 0,
                             layer.second_bias.size() * sizeof(float)));
    quantize_layer(layer);
  }

  void write_manifest() const {
    std::ofstream file(run_dir / "protocol.json", std::ios::trunc);
    file << std::setprecision(9)
         << "{\n"
         << "  \"backend\": \"native_cuda_no_pytorch\",\n"
         << "  \"modes\": 1024,\n"
         << "  \"optimizer_steps\": 10000,\n"
         << "  \"points_per_step\": 1024,\n"
         << "  \"objective\": \"integral_abs_grad_omega_sq_over_integral_omega_sq\",\n"
         << "  \"objective_scale\": 1.0,\n"
         << "  \"mode_aggregation\": \"independent_unscaled_sum\",\n"
         << "  \"orthogonality\": \"full_causal_least_squares_peeling\",\n"
         << "  \"auxiliary_losses\": [],\n"
         << "  \"gradient_clipping\": false,\n"
         << "  \"weight_decay\": false,\n"
         << "  \"spatial_derivatives\": \"hand_written_exact_third_order\",\n"
         << "  \"parameter_backward\": \"hand_written_reverse\",\n"
         << "  \"optimizer\": \"hand_written_adam\",\n"
         << "  \"network_compute\": \"fp16\",\n"
         << "  \"all_other_compute\": \"fp32\",\n"
         << "  \"seed\": " << config.seed << "\n"
         << "}\n";
  }

  void prepare_step(const std::uint32_t step) {
    const auto& current = geometries[step % geometries.size()];
    geometry.copy_from_host(current.data(), current.size());
    prepare_step_kernel<<<(kPoints + kThreads - 1) / kThreads, kThreads>>>(
        config.seed, step, geometry.data(), input_value.data(),
        input_derivative.data(), envelope.data(), sqrt_weight.data());
    EF_CUDA_CHECK_LAST();
  }

  void forward() {
    forward_hidden(handle, layer0, input_value.data(), input_derivative.data(),
                   0, 0, activation0);
    forward_hidden(handle, layer1, activation0.post_value.data(),
                   activation0.post_derivative.data(),
                   static_cast<long long>(kHidden) * kPoints,
                   static_cast<long long>(kHidden) * kPoints, activation1);
    forward_hidden(handle, layer2, activation1.post_value.data(),
                   activation1.post_derivative.data(),
                   static_cast<long long>(kHidden) * kPoints,
                   static_cast<long long>(kHidden) * kPoints, activation2);
    gemm_half_forward(handle, output, activation2.post_value.data(),
                      static_cast<long long>(kHidden) * kPoints,
                      potential_value.data());
    gemm_derivative_forward(handle, output,
                            activation2.post_derivative.data(),
                            static_cast<long long>(kHidden) * kPoints,
                            potential_derivative.data());
    const std::size_t potential_items = potential_value.size();
    output_bias_kernel<<<
        static_cast<int>((potential_items + kThreads - 1) / kThreads),
        kThreads>>>(potential_value.data(), output.bias_half.data());
    physical_forward_kernel<<<
        static_cast<int>((static_cast<std::size_t>(kModes) * kPoints +
                          kThreads - 1) /
                         kThreads),
        kThreads>>>(potential_value.data(), potential_derivative.data(),
                    envelope.data(), sqrt_weight.data(), raw_omega.data(),
                    raw_grad_omega.data());
    EF_CUDA_CHECK_LAST();
  }

  float objective_forward() {
    return peel_forward(handle, solver, raw_omega.data(),
                        raw_grad_omega.data(), q.data(), g.data(),
                        coefficients.data(), solver_workspace.data(),
                        solver_workspace_count, solver_info.data(),
                        device_roots, device_loss, roots, losses);
  }

  void backward() {
    EF_CUDA_CHECK(cudaMemset(coefficient_adjoint.data(), 0,
                             coefficient_adjoint.size() * sizeof(float)));
    peel_reverse(handle, q.data(), g.data(), roots, losses,
                 omega_adjoint.data(), grad_adjoint.data(),
                 coefficient_adjoint.data(), device_roots, device_loss);
    const std::size_t mode_points = static_cast<std::size_t>(kModes) * kPoints;
    physical_reverse_kernel<<<
        static_cast<int>((mode_points + kThreads - 1) / kThreads), kThreads>>>(
        potential_value.data(), potential_derivative.data(), envelope.data(),
        sqrt_weight.data(), omega_adjoint.data(), grad_adjoint.data(),
        potential_adjoint.data());
    EF_CUDA_CHECK_LAST();

    convert_backward_value(activation2.post_value.data(),
                           activation2.post_value.size());
    weight_gradient(handle, output, potential_adjoint.data(),
                    backward_input_value.data(),
                    activation2.post_derivative.data(),
                    static_cast<long long>(kHidden) * kPoints);
    linear_reverse(handle, output, potential_adjoint.data(),
                   hidden_adjoint_a.data());

    reverse_hidden(layer2, activation2, hidden_adjoint_a,
                   hidden_adjoint_b, activation1.post_value.data(),
                   activation1.post_derivative.data());
    // linear_reverse writes the previous layer's post-adjoint back into the
    // first buffer. Reuse the same A/B pair at every hidden layer; swapping
    // them here would feed layer-2 pre-adjoints into layer 1.
    reverse_hidden(layer1, activation1, hidden_adjoint_a,
                   hidden_adjoint_b, activation0.post_value.data(),
                   activation0.post_derivative.data());
    reverse_hidden(layer0, activation0, hidden_adjoint_a,
                   hidden_adjoint_b, input_value.data(),
                   input_derivative.data(), true);
  }

  void reverse_hidden(Layer& layer, const Activation& activation,
                      DeviceBuffer<float>& post_adjoint,
                      DeviceBuffer<float>& pre_adjoint,
                      const __half* previous_value,
                      const float* previous_derivative,
                      const bool input_shared = false) {
    const std::size_t items = activation.pre_value.size();
    sine_reverse_kernel<<<
        static_cast<int>((items + kThreads - 1) / kThreads), kThreads>>>(
        activation.pre_value.data(), activation.pre_derivative.data(),
        post_adjoint.data(), pre_adjoint.data(), layer.output);
    EF_CUDA_CHECK_LAST();
    const std::size_t previous_count =
        input_shared ? static_cast<std::size_t>(layer.input) * kPoints
                     : static_cast<std::size_t>(kModes) * layer.input *
                           kPoints;
    convert_backward_value(previous_value, previous_count);
    weight_gradient(handle, layer, pre_adjoint.data(),
                    backward_input_value.data(),
                    previous_derivative,
                    input_shared ? 0
                                 : static_cast<long long>(layer.input) *
                                       kPoints);
    if (!input_shared) {
      linear_reverse(handle, layer, pre_adjoint.data(), post_adjoint.data());
    }
  }

  void convert_backward_value(const __half* input, const std::size_t count) {
    half_to_float_kernel<<<
        static_cast<int>((count + kThreads - 1) / kThreads), kThreads>>>(
        input, backward_input_value.data(), count);
    EF_CUDA_CHECK_LAST();
  }

  void update(const std::uint32_t step) {
    adam_layer(layer0, step);
    adam_layer(layer1, step);
    adam_layer(layer2, step);
    adam_layer(output, step);
  }

  void checkpoint(const std::uint32_t step, const float total_loss) {
    const std::filesystem::path temporary = run_dir / "latest.bin.tmp";
    const std::filesystem::path final = run_dir / "latest.bin";
    std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
    if (!file) {
      throw BackendError("Could not create native checkpoint");
    }
    const std::array<char, 16> magic = {'E', 'F', 'S', 'S', 'S', 'N', 'A', 'T',
                                        'I', 'V', 'E', 'V', '1', '\0', '\0', '\0'};
    file.write(magic.data(), magic.size());
    file.write(reinterpret_cast<const char*>(&step), sizeof(step));
    file.write(reinterpret_cast<const char*>(&total_loss), sizeof(total_loss));
    write_layer(file, layer0);
    write_layer(file, layer1);
    write_layer(file, layer2);
    write_layer(file, output);
    file.flush();
    file.close();
    std::error_code ignored;
    std::filesystem::remove(final, ignored);
    std::filesystem::rename(temporary, final);
  }

  static void write_device(std::ofstream& file,
                           const DeviceBuffer<float>& source) {
    std::vector<float> host(source.size());
    source.copy_to_host(host.data(), host.size());
    const std::uint64_t count = host.size();
    file.write(reinterpret_cast<const char*>(&count), sizeof(count));
    file.write(reinterpret_cast<const char*>(host.data()),
               static_cast<std::streamsize>(host.size() * sizeof(float)));
  }

  static void write_layer(std::ofstream& file, const Layer& layer) {
    write_device(file, layer.master_weight);
    write_device(file, layer.master_bias);
    write_device(file, layer.first_weight);
    write_device(file, layer.second_weight);
    write_device(file, layer.first_bias);
    write_device(file, layer.second_bias);
  }

  static void read_device(std::ifstream& file, DeviceBuffer<float>& target) {
    std::uint64_t count = 0;
    file.read(reinterpret_cast<char*>(&count), sizeof(count));
    if (!file || count != target.size()) {
      throw BackendError("Native checkpoint tensor size mismatch");
    }
    std::vector<float> host(static_cast<std::size_t>(count));
    file.read(reinterpret_cast<char*>(host.data()),
              static_cast<std::streamsize>(host.size() * sizeof(float)));
    if (!file) {
      throw BackendError("Native checkpoint ended inside a tensor");
    }
    target.copy_from_host(host.data(), host.size());
  }

  static void read_layer(std::ifstream& file, Layer& layer) {
    read_device(file, layer.master_weight);
    read_device(file, layer.master_bias);
    read_device(file, layer.first_weight);
    read_device(file, layer.second_weight);
    read_device(file, layer.first_bias);
    read_device(file, layer.second_bias);
  }

  void load_checkpoint_if_present() {
    const std::filesystem::path path = run_dir / "latest.bin";
    if (!std::filesystem::exists(path)) {
      return;
    }
    std::ifstream file(path, std::ios::binary);
    std::array<char, 16> magic{};
    file.read(magic.data(), magic.size());
    const std::array<char, 16> expected = {
        'E', 'F', 'S', 'S', 'S', 'N', 'A', 'T',
        'I', 'V', 'E', 'V', '1', '\0', '\0', '\0'};
    if (!file || magic != expected) {
      throw BackendError("Native checkpoint has an invalid header");
    }
    float saved_loss = 0.0F;
    file.read(reinterpret_cast<char*>(&start_step), sizeof(start_step));
    file.read(reinterpret_cast<char*>(&saved_loss), sizeof(saved_loss));
    if (!file || start_step > TrainingContract::optimizer_steps) {
      throw BackendError("Native checkpoint has an invalid step");
    }
    read_layer(file, layer0);
    read_layer(file, layer1);
    read_layer(file, layer2);
    read_layer(file, output);
    quantize_layer(layer0);
    quantize_layer(layer1);
    quantize_layer(layer2);
    quantize_layer(output);
    std::cout << "checkpoint_resume=PASS step=" << start_step
              << " saved_loss=" << saved_loss << '\n';
  }

  void train() {
    initialize();
    std::ofstream history(run_dir / "history.jsonl", std::ios::app);
    const auto start = std::chrono::steady_clock::now();
    for (std::uint32_t step = start_step + 1;
         step <= TrainingContract::optimizer_steps; ++step) {
      prepare_step(step);
      forward();
      const float total_loss = objective_forward();
      backward();
      update(step);
      if (step == 1 || step % TrainingContract::log_every_steps == 0) {
        EF_CUDA_CHECK(cudaDeviceSynchronize());
        const double elapsed = std::chrono::duration<double>(
                                   std::chrono::steady_clock::now() - start)
                                   .count();
        const auto [minimum, maximum] =
            std::minmax_element(losses.begin(), losses.end());
        history << std::setprecision(9)
                << "{\"step\":" << step << ",\"loss_sum\":"
                << total_loss << ",\"rayleigh_min\":" << *minimum
                << ",\"rayleigh_max\":" << *maximum
                << ",\"elapsed_seconds\":" << elapsed << "}\n";
        history.flush();
        std::cout << "step=" << step << " loss_sum=" << total_loss
                  << " rayleigh_min=" << *minimum
                  << " rayleigh_max=" << *maximum
                  << " elapsed_seconds=" << elapsed << '\n';
      }
      if (step == 1 ||
          step % TrainingContract::checkpoint_every_steps == 0 ||
          step == TrainingContract::optimizer_steps) {
        EF_CUDA_CHECK(cudaDeviceSynchronize());
        checkpoint(step, total_loss);
      }
    }
    std::ofstream summary(run_dir / "summary.json", std::ios::trunc);
    summary << "{\n  \"status\": \"complete\",\n"
            << "  \"steps\": 10000,\n"
            << "  \"modes\": 1024,\n"
            << "  \"objective_scale\": 1.0,\n"
            << "  \"auxiliary_losses\": []\n}\n";
  }
};

}  // namespace

void run_training(const RuntimeConfig& config) {
  try {
    EF_CUDA_CHECK(cudaSetDevice(config.device));
    EF_CUDA_CHECK(cudaFree(nullptr));
    Trainer trainer(config);
    trainer.train();
  } catch (const std::exception& error) {
    std::error_code ignored;
    std::filesystem::create_directories(config.run_dir, ignored);
    std::ofstream failure(std::filesystem::path(config.run_dir) / "failure.txt",
                          std::ios::trunc);
    failure << error.what() << '\n';
    throw;
  }
}

}  // namespace eigenfluid::native
