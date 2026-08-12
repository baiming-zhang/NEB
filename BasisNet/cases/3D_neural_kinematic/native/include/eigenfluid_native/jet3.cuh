#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>

namespace eigenfluid::native {

constexpr int kJetSize = 20;
constexpr int kJetDerivativeSize = 19;

// Symmetric normalized Taylor coefficients: coefficient alpha stores
// d^alpha f / alpha!. This makes multiplication an ordinary truncated
// polynomial convolution.
struct Jet3 final {
  float c[kJetSize];
};

struct MultiIndex final {
  int x;
  int y;
  int z;
};

__host__ __device__ constexpr MultiIndex jet_multi_index(const int index) {
  constexpr MultiIndex indices[kJetSize] = {
      {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {2, 0, 0},
      {1, 1, 0}, {1, 0, 1}, {0, 2, 0}, {0, 1, 1}, {0, 0, 2},
      {3, 0, 0}, {2, 1, 0}, {2, 0, 1}, {1, 2, 0}, {1, 1, 1},
      {1, 0, 2}, {0, 3, 0}, {0, 2, 1}, {0, 1, 2}, {0, 0, 3}};
  return indices[index];
}

__host__ __device__ constexpr int jet_index(const int x, const int y,
                                             const int z) {
  const int degree = x + y + z;
  if (degree == 0) return 0;
  if (degree == 1) {
    if (x == 1) return 1;
    if (y == 1) return 2;
    return 3;
  }
  if (degree == 2) {
    if (x == 2) return 4;
    if (x == 1 && y == 1) return 5;
    if (x == 1 && z == 1) return 6;
    if (y == 2) return 7;
    if (y == 1 && z == 1) return 8;
    return 9;
  }
  if (degree == 3) {
    if (x == 3) return 10;
    if (x == 2 && y == 1) return 11;
    if (x == 2 && z == 1) return 12;
    if (x == 1 && y == 2) return 13;
    if (x == 1 && y == 1 && z == 1) return 14;
    if (x == 1 && z == 2) return 15;
    if (y == 3) return 16;
    if (y == 2 && z == 1) return 17;
    if (y == 1 && z == 2) return 18;
    return 19;
  }
  return -1;
}

__host__ __device__ constexpr int jet_degree(const int index) {
  const MultiIndex item = jet_multi_index(index);
  return item.x + item.y + item.z;
}

__host__ __device__ constexpr float jet_factorial(const int index) {
  const MultiIndex item = jet_multi_index(index);
  constexpr int factorial[4] = {1, 1, 2, 6};
  return static_cast<float>(factorial[item.x] * factorial[item.y] *
                            factorial[item.z]);
}

__host__ __device__ inline Jet3 jet_zero() {
  Jet3 result{};
  for (float& value : result.c) {
    value = 0.0F;
  }
  return result;
}

__host__ __device__ inline Jet3 jet_constant(const float value) {
  Jet3 result = jet_zero();
  result.c[0] = value;
  return result;
}

__host__ __device__ inline Jet3 jet_variable(const float value,
                                              const int axis) {
  Jet3 result = jet_constant(value);
  result.c[1 + axis] = 1.0F;
  return result;
}

__host__ __device__ inline Jet3 jet_add(const Jet3& a, const Jet3& b) {
  Jet3 result;
  for (int index = 0; index < kJetSize; ++index) {
    result.c[index] = a.c[index] + b.c[index];
  }
  return result;
}

__host__ __device__ inline Jet3 jet_sub(const Jet3& a, const Jet3& b) {
  Jet3 result;
  for (int index = 0; index < kJetSize; ++index) {
    result.c[index] = a.c[index] - b.c[index];
  }
  return result;
}

__host__ __device__ inline Jet3 jet_scale(const Jet3& a, const float scale) {
  Jet3 result;
  for (int index = 0; index < kJetSize; ++index) {
    result.c[index] = a.c[index] * scale;
  }
  return result;
}

__host__ __device__ inline Jet3 jet_mul(const Jet3& a, const Jet3& b) {
  Jet3 result = jet_zero();
  for (int left = 0; left < kJetSize; ++left) {
    const MultiIndex li = jet_multi_index(left);
    const int remaining_degree = 3 - (li.x + li.y + li.z);
    const int right_count = remaining_degree == 3 ? 20
                            : remaining_degree == 2 ? 10
                            : remaining_degree == 1 ? 4
                                                    : 1;
    for (int right = 0; right < right_count; ++right) {
      const MultiIndex ri = jet_multi_index(right);
      const int output = jet_index(li.x + ri.x, li.y + ri.y, li.z + ri.z);
      result.c[output] += a.c[left] * b.c[right];
    }
  }
  return result;
}

// Reverse one truncated polynomial product output=a*b.
__device__ inline void jet_mul_reverse(const Jet3& a, const Jet3& b,
                                       const Jet3& output_adjoint,
                                       Jet3& a_adjoint,
                                       Jet3& b_adjoint) {
  for (int left = 0; left < kJetSize; ++left) {
    const MultiIndex li = jet_multi_index(left);
    const int remaining_degree = 3 - (li.x + li.y + li.z);
    const int right_count = remaining_degree == 3 ? 20
                            : remaining_degree == 2 ? 10
                            : remaining_degree == 1 ? 4
                                                    : 1;
    for (int right = 0; right < right_count; ++right) {
      const MultiIndex ri = jet_multi_index(right);
      const int output = jet_index(li.x + ri.x, li.y + ri.y, li.z + ri.z);
      a_adjoint.c[left] += output_adjoint.c[output] * b.c[right];
      b_adjoint.c[right] += output_adjoint.c[output] * a.c[left];
    }
  }
}

// Compose an analytic scalar function using its first three derivatives.
__host__ __device__ inline Jet3 jet_compose(const Jet3& x, const float f0,
                                            const float f1, const float f2,
                                            const float f3) {
  Jet3 delta = x;
  delta.c[0] = 0.0F;
  const Jet3 delta2 = jet_mul(delta, delta);
  const Jet3 delta3 = jet_mul(delta2, delta);
  Jet3 result = jet_constant(f0);
  result = jet_add(result, jet_scale(delta, f1));
  result = jet_add(result, jet_scale(delta2, 0.5F * f2));
  result = jet_add(result, jet_scale(delta3, f3 / 6.0F));
  return result;
}

__host__ __device__ inline Jet3 jet_sin(const Jet3& x) {
  const float sine = sinf(x.c[0]);
  const float cosine = cosf(x.c[0]);
  return jet_compose(x, sine, cosine, -sine, -cosine);
}

__host__ __device__ inline Jet3 jet_exp(const Jet3& x) {
  const float value = expf(x.c[0]);
  return jet_compose(x, value, value, value, value);
}

__host__ __device__ inline Jet3 jet_log(const Jet3& x) {
  const float inverse = 1.0F / x.c[0];
  return jet_compose(x, logf(x.c[0]), inverse, -inverse * inverse,
                     2.0F * inverse * inverse * inverse);
}

__host__ __device__ inline Jet3 jet_sqrt(const Jet3& x) {
  const float root = sqrtf(x.c[0]);
  const float inverse = 1.0F / root;
  return jet_compose(x, root, 0.5F * inverse,
                     -0.25F * inverse * inverse * inverse,
                     0.375F * inverse * inverse * inverse * inverse * inverse);
}

__host__ __device__ inline Jet3 jet_tanh(const Jet3& x) {
  const float value = tanhf(x.c[0]);
  const float first = 1.0F - value * value;
  const float second = -2.0F * value * first;
  const float third = (-2.0F + 6.0F * value * value) * first;
  return jet_compose(x, value, first, second, third);
}

// Reverse y=sin(x) in normalized Taylor-coefficient space.
__device__ inline Jet3 jet_sin_reverse(const Jet3& x,
                                       const Jet3& y_adjoint) {
  Jet3 delta = x;
  delta.c[0] = 0.0F;
  const Jet3 delta2 = jet_mul(delta, delta);
  const Jet3 delta3 = jet_mul(delta2, delta);
  Jet3 a = jet_constant(1.0F);
  a = jet_add(a, jet_scale(delta2, -0.5F));
  Jet3 b = jet_add(delta, jet_scale(delta3, -1.0F / 6.0F));
  const float sine = sinf(x.c[0]);
  const float cosine = cosf(x.c[0]);

  Jet3 a_adjoint = jet_zero();
  Jet3 b_adjoint = jet_zero();
  float sine_adjoint = 0.0F;
  float cosine_adjoint = 0.0F;
  for (int index = 0; index < kJetSize; ++index) {
    sine_adjoint += y_adjoint.c[index] * a.c[index];
    cosine_adjoint += y_adjoint.c[index] * b.c[index];
    a_adjoint.c[index] += y_adjoint.c[index] * sine;
    b_adjoint.c[index] += y_adjoint.c[index] * cosine;
  }

  Jet3 delta_adjoint = b_adjoint;
  Jet3 delta2_adjoint = jet_scale(a_adjoint, -0.5F);
  Jet3 delta3_adjoint = jet_scale(b_adjoint, -1.0F / 6.0F);
  Jet3 extra_delta2 = jet_zero();
  Jet3 extra_delta = jet_zero();
  jet_mul_reverse(delta2, delta, delta3_adjoint, extra_delta2,
                  extra_delta);
  delta2_adjoint = jet_add(delta2_adjoint, extra_delta2);
  delta_adjoint = jet_add(delta_adjoint, extra_delta);
  Jet3 first_delta = jet_zero();
  Jet3 second_delta = jet_zero();
  jet_mul_reverse(delta, delta, delta2_adjoint, first_delta, second_delta);
  delta_adjoint = jet_add(delta_adjoint, first_delta);
  delta_adjoint = jet_add(delta_adjoint, second_delta);

  Jet3 x_adjoint = delta_adjoint;
  x_adjoint.c[0] = sine_adjoint * cosine - cosine_adjoint * sine;
  return x_adjoint;
}

__host__ __device__ inline float jet_actual_derivative(const Jet3& value,
                                                       const int index) {
  return value.c[index] * jet_factorial(index);
}

}  // namespace eigenfluid::native
