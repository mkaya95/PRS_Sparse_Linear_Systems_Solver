#include "registry.hpp"

#include <stdexcept>

namespace ssolve_bench {

std::vector<NamedSolver> all_solvers() { return eigen_solvers(); }

NamedSolver find_solver(const std::string& name) {
  std::string names;
  for (NamedSolver& solver : all_solvers()) {
    if (solver.name == name) return std::move(solver);
    names += (names.empty() ? "" : ", ") + solver.name;
  }
  throw std::runtime_error("unknown solver '" + name + "'. Valid names: " + names);
}

}  // namespace ssolve_bench
