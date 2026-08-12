#pragma once

#include "eigenfluid_native/objective_contract.hpp"

#include <cstdint>
#include <string_view>

namespace eigenfluid::native {

// Fixed before the formal run. The executable will not expose runtime flags
// that silently change these values.
struct TrainingContract final {
  static constexpr std::uint32_t modes = 1024;
  static constexpr std::uint32_t optimizer_steps = 10000;
  static constexpr std::uint32_t points_per_step = 1024;
  static constexpr std::uint32_t hidden = 32;
  static constexpr std::uint32_t hidden_layers = 3;
  static constexpr std::uint32_t frequencies = 6;
  static constexpr std::uint32_t geometry_parameters = 40;
  static constexpr std::uint32_t input_features = 79;
  static constexpr std::uint32_t potential_components = 3;
  static constexpr float learning_rate = 2.5e-5F;
  static constexpr float adam_beta1 = 0.9F;
  static constexpr float adam_beta2 = 0.999F;
  static constexpr float adam_epsilon = 1.0e-8F;
  static constexpr std::uint32_t checkpoint_every_steps = 256;
  static constexpr std::uint32_t log_every_steps = 8;

  // A deterministic pre-training dry run chooses the largest divisor of K
  // that fits the memory cap. The selected value is then immutable and is
  // stored in every manifest/checkpoint. No loss or evaluation metric enters
  // this hardware-only choice.
  static constexpr std::uint32_t minimum_active_mode_block = 64;
  static constexpr std::uint32_t maximum_active_mode_block = modes;
  static constexpr std::uint64_t target_total_device_bytes =
      23ULL * 1024ULL * 1024ULL * 1024ULL;
  static constexpr std::uint64_t minimum_free_reserve_bytes =
      768ULL * 1024ULL * 1024ULL;

  static constexpr std::string_view spatial_derivatives =
      "hand_written_exact_jet3";
  static constexpr std::string_view parameter_backward =
      "hand_written_reverse";
  static constexpr std::string_view optimizer = "hand_written_adam";
  static constexpr std::string_view mode_loss_aggregation =
      "independent_unscaled_sum";
  static constexpr bool gradient_clipping = false;
  static constexpr bool weight_decay = false;

  // Only the neural-network compute path is FP16. Every physical and
  // optimization quantity remains FP32.
  static constexpr std::string_view network_compute = "fp16";
  static constexpr std::string_view network_gemm_accumulator = "fp32";
  static constexpr std::string_view master_weights = "fp32";
  static constexpr std::string_view parameter_gradients = "fp32";
  static constexpr std::string_view optimizer_state = "fp32";
  static constexpr std::string_view coordinate_jets = "fp32";
  static constexpr std::string_view geometry_and_envelope = "fp32";
  static constexpr std::string_view physical_fields = "fp32";
  static constexpr std::string_view sss_and_loss = "fp32";
};

static_assert(TrainingContract::modes == 1024);
static_assert(TrainingContract::modes %
                  TrainingContract::minimum_active_mode_block ==
              0);
static_assert(TrainingContract::optimizer_steps == 10000);
static_assert(TrainingContract::points_per_step * 3 >= TrainingContract::modes);
static_assert(!TrainingContract::gradient_clipping);
static_assert(!TrainingContract::weight_decay);

}  // namespace eigenfluid::native
