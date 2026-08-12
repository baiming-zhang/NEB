#pragma once

#include "eigenfluid_native/training_contract.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <string_view>
#include <vector>

namespace eigenfluid::native {

struct MemoryRegion final {
  std::string_view name;
  std::uint64_t bytes;
};

struct MemoryPlan final {
  std::uint32_t active_modes = 0;
  std::vector<MemoryRegion> regions;

  [[nodiscard]] std::uint64_t total_bytes() const {
    std::uint64_t total = 0;
    for (const MemoryRegion& region : regions) {
      total += region.bytes;
    }
    return total;
  }
};

// The planner is shared by the startup dry run and the production allocator.
// Changing a live buffer layout therefore changes the dry-run footprint too.
inline MemoryPlan make_training_memory_plan(const std::uint32_t active_modes) {
  constexpr std::uint64_t kPoints = TrainingContract::points_per_step;
  constexpr std::uint64_t kHidden = TrainingContract::hidden;
  constexpr std::uint64_t kHiddenLayers = TrainingContract::hidden_layers;
  constexpr std::uint64_t kInputFeatures = TrainingContract::input_features;
  constexpr std::uint64_t kOutputComponents =
      TrainingContract::potential_components;
  constexpr std::uint64_t kJetDerivatives = 19;
  constexpr std::uint64_t kJetComponents = 20;
  constexpr std::uint64_t kParametersPerMode = 4771;
  constexpr std::uint64_t kFloatBytes = 4;
  constexpr std::uint64_t kHalfBytes = 2;
  constexpr std::uint64_t kMiB = 1024ULL * 1024ULL;

  if (active_modes < TrainingContract::minimum_active_mode_block ||
      active_modes > TrainingContract::maximum_active_mode_block ||
      TrainingContract::modes % active_modes != 0) {
    throw BackendError("Active-mode block is not an allowed divisor of K");
  }

  const std::uint64_t active = active_modes;
  const std::uint64_t prefix = TrainingContract::modes - active_modes;
  const std::uint64_t hidden_values = kPoints * active * kHidden;
  const std::uint64_t mode_points = kPoints * active;

  MemoryPlan plan;
  plan.active_modes = active_modes;
  plan.regions = {
      // FP32 master + FP16 compute copy + FP32 gradient + two FP32 Adam
      // moments. These cover all K modes, not just the active block.
      {"network_master_compute_grad_adam",
       static_cast<std::uint64_t>(TrainingContract::modes) *
           kParametersPerMode * (4ULL + 2ULL + 4ULL + 8ULL)},
      // Save both pre- and post-sine states for fastest hand-written reverse:
      // two FP16 values and two sets of 19 FP32 spatial derivatives.
      {"saved_hidden_pre_post_jet3",
       hidden_values * kHiddenLayers *
           (2ULL * kHalfBytes + 2ULL * kJetDerivatives * kFloatBytes)},
      // FP32 reverse accumulation ping-pong buffers for all 20 jet slots.
      {"hidden_adjoint_ping_pong",
       hidden_values * 2ULL * kJetComponents * kFloatBytes},
      {"enveloped_potential_jet3",
       mode_points * kOutputComponents * kJetComponents * kFloatBytes},
      // A, u, grad(u), omega, grad(omega): 3 + 3 + 9 + 3 + 9 floats.
      {"active_physical_fields", mode_points * 27ULL * kFloatBytes},
      // Normalized omega and grad(omega) retained for the active block.
      {"peeled_active_vorticity", mode_points * 12ULL * kFloatBytes},
      // Complete frozen causal prefix, streamed in vorticity state space.
      {"causal_prefix_vorticity",
       kPoints * prefix * 12ULL * kFloatBytes},
      {"peeling_coefficients", active * prefix * kFloatBytes},
      // Fourier encoding: FP16 value plus 19 FP32 spatial derivatives.
      {"fourier_input_jet3",
       kPoints * kInputFeatures *
           (kHalfBytes + kJetDerivatives * kFloatBytes)},
      // cuBLAS workspace, reductions, RNG, checkpoint staging, and allocator
      // scratch. This is deliberately concrete rather than hidden padding.
      {"runtime_and_linear_algebra_workspace", 768ULL * kMiB},
  };
  return plan;
}

inline constexpr std::array<std::uint32_t, 5> kBlockCandidates = {
    1024, 512, 256, 128, 64};

}  // namespace eigenfluid::native
