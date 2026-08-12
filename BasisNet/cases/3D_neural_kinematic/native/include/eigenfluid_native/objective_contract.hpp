#pragma once

#include "eigenfluid_native/cuda_error.hpp"

#include <cmath>
#include <string_view>

namespace eigenfluid::native {

// This is intentionally a closed contract, not a user-selectable loss menu.
// The production trainer has exactly one scalar objective. Orthogonality,
// divergence, and boundary conditions are imposed constructively and must
// never reappear here as weighted penalties.
struct ObjectiveContract final {
  static constexpr std::string_view name =
      "peeled_vorticity_rayleigh_only";
  static constexpr bool uses_velocity_loss = false;
  static constexpr bool uses_task_loss = false;
  static constexpr bool uses_orthogonality_penalty = false;
  static constexpr bool uses_divergence_penalty = false;
  static constexpr bool uses_boundary_penalty = false;
  static constexpr bool uses_regularization_penalty = false;
  static constexpr float multiplicative_scale = 1.0F;
};

inline float vorticity_rayleigh_loss(const float peeled_grad_omega_sq_integral,
                                     const float peeled_omega_sq_integral,
                                     const float /*unused_mass_floor*/) {
  if (!(peeled_grad_omega_sq_integral >= 0.0F) ||
      !(peeled_omega_sq_integral > 0.0F) ||
      !std::isfinite(peeled_grad_omega_sq_integral) ||
      !std::isfinite(peeled_omega_sq_integral)) {
    throw BackendError("Invalid vorticity-Rayleigh integral");
  }
  return peeled_grad_omega_sq_integral / peeled_omega_sq_integral;
}

static_assert(!ObjectiveContract::uses_velocity_loss);
static_assert(!ObjectiveContract::uses_task_loss);
static_assert(!ObjectiveContract::uses_orthogonality_penalty);
static_assert(!ObjectiveContract::uses_divergence_penalty);
static_assert(!ObjectiveContract::uses_boundary_penalty);
static_assert(!ObjectiveContract::uses_regularization_penalty);
static_assert(ObjectiveContract::multiplicative_scale == 1.0F);

}  // namespace eigenfluid::native
