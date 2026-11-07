// A description of the machine and build, recorded in every benchmark row.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ssolve {

struct Manifest {
  std::string git_commit;
  std::string cpu_model;
  std::string compiler;
  std::string flags;
  std::string eigen_commit;
  std::string affinity = "unavailable";
  std::string hostname;
  int logical_cpus = 0;
  int physical_cores = 0;  // 0 if unknown, as under WSL2
  std::int64_t l1d_bytes = 0;
  std::int64_t l2_bytes = 0;
  std::int64_t l3_bytes = 0;
  std::int64_t mem_bytes = 0;
};

// Anything that cannot be read stays at its default, because a run should never stop for it.
Manifest collect_manifest();

std::vector<std::string> manifest_columns();
std::vector<std::string> manifest_values(const Manifest& manifest);

}  // namespace ssolve
