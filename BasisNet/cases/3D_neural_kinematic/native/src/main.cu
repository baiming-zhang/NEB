#include "eigenfluid_native/cuda_error.hpp"
#include "eigenfluid_native/device_buffer.hpp"
#include "eigenfluid_native/jet3.cuh"
#include "eigenfluid_native/memory_plan.hpp"
#include "eigenfluid_native/objective_contract.hpp"
#include "eigenfluid_native/runtime_config.hpp"
#include "eigenfluid_native/training_contract.hpp"
#include "eigenfluid_native/trainer.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace ef = eigenfluid::native;

namespace {

constexpr int kBlockSize = 256;

__host__ __device__ std::uint32_t deterministic_bits(std::uint64_t state) {
  state += 0x9E3779B97F4A7C15ULL;
  state = (state ^ (state >> 30U)) * 0xBF58476D1CE4E5B9ULL;
  state = (state ^ (state >> 27U)) * 0x94D049BB133111EBULL;
  state ^= state >> 31U;
  return static_cast<std::uint32_t>(state >> 32U);
}

__host__ __device__ float deterministic_value(const std::uint64_t seed,
                                               const std::size_t index) {
  const std::uint32_t bits = deterministic_bits(
      seed + static_cast<std::uint64_t>(index) * 0xD1B54A32D192ED03ULL);
  constexpr float kScale = 1.0F / 16777216.0F;
  const float unit = static_cast<float>(bits >> 8U) * kScale;
  return 1.25F * (2.0F * unit - 1.0F) - 0.5F;
}

__global__ void deterministic_fill(float* output, const std::size_t count,
                                   const std::uint64_t seed) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    output[index] = deterministic_value(seed, index);
  }
}

__global__ void jet3_validation_kernel(float* output) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  const ef::Jet3 x = ef::jet_variable(0.31F, 0);
  const ef::Jet3 y = ef::jet_variable(-0.27F, 1);
  const ef::Jet3 z = ef::jet_variable(0.19F, 2);
  const ef::Jet3 polynomial = ef::jet_add(
      ef::jet_mul(ef::jet_mul(x, x), y),
      ef::jet_scale(ef::jet_mul(x, ef::jet_mul(y, z)), 3.0F));
  const ef::Jet3 value = ef::jet_sin(polynomial);
  for (int index = 0; index < ef::kJetSize; ++index) {
    output[index] = value.c[index];
  }
}

void print_usage(const char* executable) {
  std::cout << "Usage: " << executable
            << " [--self-test] [--memory-dry-run] [--train]"
            << " [--run-dir PATH] [--device N] [--seed N]\n\n"
            << "  --self-test   Run CUDA runtime, custom-kernel, and cuBLAS checks.\n"
            << "  --memory-dry-run  Allocate and touch the complete planned training\n"
            << "                    buffers, selecting the largest block under the\n"
            << "                    locked 23 GiB total-device cap.\n"
            << "  --device N    Select a zero-based CUDA device (default: 0).\n"
            << "  --seed N      Set a deterministic 64-bit seed; decimal or 0x-prefixed.\n"
            << "  --train       Run the immutable K=1024, 10000-step native trainer.\n"
            << "  --run-dir P   Output directory for manifest, history, and checkpoints.\n";
}

void print_device(const int device) {
  int device_count = 0;
  EF_CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (device < 0 || device >= device_count) {
    throw ef::BackendError("Requested CUDA device " + std::to_string(device) +
                           ", but only " + std::to_string(device_count) +
                           " device(s) are visible");
  }
  EF_CUDA_CHECK(cudaSetDevice(device));

  cudaDeviceProp properties{};
  EF_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  EF_CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));

  std::cout << "device=" << device << " name=\"" << properties.name << "\""
            << " compute=" << properties.major << '.' << properties.minor
            << " multiprocessors=" << properties.multiProcessorCount
            << " memory_free_mib=" << (free_bytes / (1024ULL * 1024ULL))
            << " memory_total_mib=" << (total_bytes / (1024ULL * 1024ULL))
            << '\n';
}

void test_custom_kernel(const std::uint64_t seed) {
  constexpr std::size_t kCount = 4096;
  ef::DeviceBuffer<float> device_values(kCount);
  std::vector<float> observed(kCount);

  const int grid_size = static_cast<int>((kCount + kBlockSize - 1) / kBlockSize);
  deterministic_fill<<<grid_size, kBlockSize>>>(device_values.data(), kCount,
                                                seed);
  EF_CUDA_CHECK_LAST();
  EF_CUDA_CHECK(cudaDeviceSynchronize());
  device_values.copy_to_host(observed.data(), observed.size());

  float max_absolute_error = 0.0F;
  for (std::size_t index = 0; index < observed.size(); ++index) {
    max_absolute_error =
        std::max(max_absolute_error,
                 std::abs(observed[index] - deterministic_value(seed, index)));
  }
  if (max_absolute_error > 2.0e-6F) {
    throw ef::BackendError("Custom kernel validation failed: max error=" +
                           std::to_string(max_absolute_error));
  }
  std::cout << "custom_kernel=PASS count=" << kCount
            << " max_abs_error=" << std::scientific << max_absolute_error
            << std::defaultfloat << '\n';
}

void test_jet3() {
  ef::DeviceBuffer<float> device_values(ef::kJetSize);
  std::vector<float> observed(ef::kJetSize, 0.0F);
  jet3_validation_kernel<<<1, 1>>>(device_values.data());
  EF_CUDA_CHECK_LAST();
  EF_CUDA_CHECK(cudaDeviceSynchronize());
  device_values.copy_to_host(observed.data(), observed.size());

  const double x = 0.31;
  const double y = -0.27;
  const double z = 0.19;
  const double p = x * x * y + 3.0 * x * y * z;
  const double px = 2.0 * x * y + 3.0 * y * z;
  const double py = x * x + 3.0 * x * z;
  const double pz = 3.0 * x * y;
  const double pxy = 2.0 * x + 3.0 * z;
  const double pxz = 3.0 * y;
  const double pyz = 3.0 * x;
  const double pxx = 2.0 * y;
  const double pxyz = 3.0;
  const double pz2 = 0.0;
  const double expected_xyz =
      -std::cos(p) * px * py * pz - std::sin(p) *
      (pxy * pz + pxz * py + pyz * px) + std::cos(p) * pxyz;
  const double expected_xx = -std::sin(p) * px * px + std::cos(p) * pxx;
  const double expected_zz = -std::sin(p) * pz * pz + std::cos(p) * pz2;
  const float actual_xyz = observed[14];
  const float actual_xx = observed[4] * 2.0F;
  const float actual_zz = observed[9] * 2.0F;
  const double error = std::max(
      {std::abs(static_cast<double>(actual_xyz) - expected_xyz),
       std::abs(static_cast<double>(actual_xx) - expected_xx),
       std::abs(static_cast<double>(actual_zz) - expected_zz)});
  if (error > 2.0e-5) {
    throw ef::BackendError("Jet3 analytic derivative validation failed: max error=" +
                           std::to_string(error));
  }
  std::cout << "jet3_exact_derivatives=PASS max_abs_error=" << std::scientific
            << error << std::defaultfloat << '\n';
}

void test_cublas() {
  // Column-major matrices:
  // A = [[1, 2, 3], [4, 5, 6]], B = [[7, 8], [9, 10], [11, 12]].
  const std::vector<float> host_a{1.0F, 4.0F, 2.0F, 5.0F, 3.0F, 6.0F};
  const std::vector<float> host_b{7.0F, 9.0F, 11.0F, 8.0F, 10.0F, 12.0F};
  const std::vector<float> expected{58.0F, 139.0F, 64.0F, 154.0F};
  std::vector<float> observed(expected.size(), 0.0F);

  ef::DeviceBuffer<float> device_a(host_a.size());
  ef::DeviceBuffer<float> device_b(host_b.size());
  ef::DeviceBuffer<float> device_c(expected.size());
  device_a.copy_from_host(host_a.data(), host_a.size());
  device_b.copy_from_host(host_b.data(), host_b.size());

  ef::CublasHandle handle;
  EF_CUBLAS_CHECK(
      cublasSetAtomicsMode(handle.get(), CUBLAS_ATOMICS_NOT_ALLOWED));
  EF_CUBLAS_CHECK(cublasSetMathMode(handle.get(), CUBLAS_DEFAULT_MATH));

  constexpr int m = 2;
  constexpr int n = 2;
  constexpr int k = 3;
  constexpr float alpha = 1.0F;
  constexpr float beta = 0.0F;
  EF_CUBLAS_CHECK(cublasSgemm(handle.get(), CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                              &alpha, device_a.data(), m, device_b.data(), k,
                              &beta, device_c.data(), m));
  EF_CUDA_CHECK(cudaDeviceSynchronize());
  device_c.copy_to_host(observed.data(), observed.size());

  float max_absolute_error = 0.0F;
  for (std::size_t index = 0; index < observed.size(); ++index) {
    max_absolute_error = std::max(
        max_absolute_error, std::abs(observed[index] - expected[index]));
  }
  if (max_absolute_error != 0.0F) {
    throw ef::BackendError("cuBLAS SGEMM validation failed: max error=" +
                           std::to_string(max_absolute_error));
  }
  std::cout << "cublas_sgemm=PASS shape=2x3x2 max_abs_error="
            << std::scientific << max_absolute_error << std::defaultfloat
            << '\n';
}

void test_objective_contract() {
  constexpr float kStiffness = 9.0F;
  constexpr float kMass = 4.0F;
  constexpr float kExpected = 2.25F;
  const float observed =
      ef::vorticity_rayleigh_loss(kStiffness, kMass, 1.0e-12F);
  if (observed != kExpected) {
    throw ef::BackendError("Vorticity-Rayleigh-only objective check failed");
  }
  std::cout << "objective_contract=PASS name=" << ef::ObjectiveContract::name
            << " loss=" << observed << " scale="
            << ef::ObjectiveContract::multiplicative_scale
            << " auxiliary_terms=0\n";
}

void test_training_contract() {
  std::cout << "training_contract=PASS modes=" << ef::TrainingContract::modes
            << " block_policy=largest_divisor_dry_run"
            << " minimum_block="
            << ef::TrainingContract::minimum_active_mode_block
            << " target_total_device_gib="
            << (ef::TrainingContract::target_total_device_bytes /
                (1024ULL * 1024ULL * 1024ULL))
            << " optimizer_steps=" << ef::TrainingContract::optimizer_steps
            << " network_compute=" << ef::TrainingContract::network_compute
            << " physical_compute=" << ef::TrainingContract::physical_fields
            << '\n';
}

void run_self_test(const ef::RuntimeConfig& config) {
  print_device(config.device);
  std::cout << "seed=" << config.seed << '\n';
  EF_CUDA_CHECK(cudaFree(nullptr));
  std::cout << "cuda_runtime=PASS runtime_version=";
  int runtime_version = 0;
  EF_CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
  std::cout << runtime_version / 1000 << '.' << (runtime_version % 1000) / 10
            << '\n';
  test_custom_kernel(config.seed);
  test_jet3();
  test_cublas();
  test_objective_contract();
  test_training_contract();
  std::cout << "SELF_TEST_PASS\n";
}

double gib(const std::uint64_t bytes) {
  return static_cast<double>(bytes) /
         static_cast<double>(1024ULL * 1024ULL * 1024ULL);
}

void run_memory_dry_run(const ef::RuntimeConfig& config) {
  print_device(config.device);
  EF_CUDA_CHECK(cudaFree(nullptr));
  std::size_t initial_free = 0;
  std::size_t total = 0;
  EF_CUDA_CHECK(cudaMemGetInfo(&initial_free, &total));
  const std::uint64_t initially_used = total - initial_free;
  const std::uint64_t cap_for_new_allocations =
      ef::TrainingContract::target_total_device_bytes > initially_used
          ? ef::TrainingContract::target_total_device_bytes - initially_used
          : 0ULL;
  const std::uint64_t reserve_limited =
      initial_free > ef::TrainingContract::minimum_free_reserve_bytes
          ? initial_free - ef::TrainingContract::minimum_free_reserve_bytes
          : 0ULL;
  const std::uint64_t allocation_budget =
      std::min(cap_for_new_allocations, reserve_limited);

  std::cout << std::fixed << std::setprecision(3)
            << "memory_dry_run initial_used_gib=" << gib(initially_used)
            << " initial_free_gib=" << gib(initial_free)
            << " allocation_budget_gib=" << gib(allocation_budget) << '\n';

  std::uint32_t selected = 0;
  for (const std::uint32_t candidate : ef::kBlockCandidates) {
    const ef::MemoryPlan plan = ef::make_training_memory_plan(candidate);
    const std::uint64_t planned = plan.total_bytes();
    std::cout << "memory_candidate block=" << candidate
              << " planned_process_gib=" << gib(planned);
    if (planned > allocation_budget) {
      std::cout << " status=SKIP_CAP\n";
      continue;
    }

    bool allocated = false;
    std::size_t free_after = initial_free;
    try {
      std::vector<ef::DeviceBuffer<std::uint8_t>> buffers;
      buffers.reserve(plan.regions.size());
      for (const ef::MemoryRegion& region : plan.regions) {
        buffers.emplace_back(static_cast<std::size_t>(region.bytes));
        EF_CUDA_CHECK(cudaMemset(buffers.back().data(), 0,
                                 static_cast<std::size_t>(region.bytes)));
      }
      EF_CUDA_CHECK(cudaDeviceSynchronize());
      std::size_t ignored_total = 0;
      EF_CUDA_CHECK(cudaMemGetInfo(&free_after, &ignored_total));
      allocated = free_after >=
                  ef::TrainingContract::minimum_free_reserve_bytes;
      std::cout << " actual_total_used_gib=" << gib(total - free_after)
                << " free_after_gib=" << gib(free_after)
                << " status=" << (allocated ? "PASS" : "FAIL_RESERVE")
                << '\n';
    } catch (const ef::BackendError& error) {
      static_cast<void>(cudaGetLastError());
      std::cout << " status=FAIL_ALLOC reason=\"" << error.what() << "\"\n";
    }
    EF_CUDA_CHECK(cudaDeviceSynchronize());
    if (allocated) {
      selected = candidate;
      break;
    }
  }
  if (selected == 0) {
    throw ef::BackendError("No active-mode block passed the memory dry run");
  }
  std::cout << "MEMORY_DRY_RUN_PASS selected_block=" << selected
            << " blocks_per_sweep="
            << (ef::TrainingContract::modes / selected) << '\n';
}

}  // namespace

int main(const int argc, char** argv) {
  try {
    const ef::RuntimeConfig config = ef::parse_runtime_config(argc, argv);
    if (config.show_help ||
        (!config.self_test && !config.memory_dry_run && !config.train)) {
      print_usage(argv[0]);
      return config.show_help ? 0 : 2;
    }
    if (config.self_test) {
      run_self_test(config);
    }
    if (config.memory_dry_run) {
      run_memory_dry_run(config);
    }
    if (config.train) {
      run_training(config);
    }
    EF_CUDA_CHECK(cudaDeviceReset());
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "eigenfluid_native: " << error.what() << '\n';
    return 1;
  }
}
