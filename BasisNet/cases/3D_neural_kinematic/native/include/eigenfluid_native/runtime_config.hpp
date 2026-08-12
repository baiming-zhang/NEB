#pragma once

#include "eigenfluid_native/cuda_error.hpp"

#include <cstdint>
#include <limits>
#include <string>

namespace eigenfluid::native {

struct RuntimeConfig final {
  bool self_test = false;
  bool memory_dry_run = false;
  bool train = false;
  bool show_help = false;
  int device = 0;
  std::uint64_t seed = 0x6A09E667F3BCC909ULL;
  std::string run_dir =
      "BasisNet/cases/3D_neural_kinematic/results/"
      "eigenbasis_k1024_native_exact_vorticity_rayleigh";
};

inline std::string option_value(const int argc, char** argv, int& index,
                                const std::string& option) {
  const std::string argument(argv[index]);
  const std::string prefix = option + '=';
  if (argument.compare(0, prefix.size(), prefix) == 0) {
    const std::string value = argument.substr(prefix.size());
    if (value.empty()) {
      throw BackendError("Missing value for " + option);
    }
    return value;
  }
  if (argument == option) {
    if (++index >= argc) {
      throw BackendError("Missing value for " + option);
    }
    return std::string(argv[index]);
  }
  return {};
}

inline std::uint64_t parse_unsigned(const std::string& value,
                                    const std::string& option) {
  std::size_t parsed = 0;
  unsigned long long result = 0;
  try {
    result = std::stoull(value, &parsed, 0);
  } catch (const std::exception&) {
    throw BackendError("Invalid unsigned integer for " + option + ": " +
                       value);
  }
  if (parsed != value.size()) {
    throw BackendError("Invalid trailing characters for " + option + ": " +
                       value);
  }
  return static_cast<std::uint64_t>(result);
}

inline int parse_device(const std::string& value) {
  const std::uint64_t parsed = parse_unsigned(value, "--device");
  if (parsed > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    throw BackendError("--device is outside the supported integer range");
  }
  return static_cast<int>(parsed);
}

inline RuntimeConfig parse_runtime_config(const int argc, char** argv) {
  RuntimeConfig config;
  for (int index = 1; index < argc; ++index) {
    const std::string argument(argv[index]);
    if (argument == "--self-test") {
      config.self_test = true;
    } else if (argument == "--memory-dry-run") {
      config.memory_dry_run = true;
    } else if (argument == "--train") {
      config.train = true;
    } else if (argument == "--help" || argument == "-h") {
      config.show_help = true;
    } else if (argument == "--device" || argument.rfind("--device=", 0) == 0) {
      config.device =
          parse_device(option_value(argc, argv, index, "--device"));
    } else if (argument == "--seed" || argument.rfind("--seed=", 0) == 0) {
      config.seed =
          parse_unsigned(option_value(argc, argv, index, "--seed"), "--seed");
    } else if (argument == "--run-dir" ||
               argument.rfind("--run-dir=", 0) == 0) {
      config.run_dir = option_value(argc, argv, index, "--run-dir");
    } else {
      throw BackendError("Unknown command-line option: " + argument);
    }
  }
  return config;
}

}  // namespace eigenfluid::native
