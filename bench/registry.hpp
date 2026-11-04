// Every solver the benchmarks can run, addressed by name.
#pragma once

#include <functional>
#include <string>
#include <vector>

#include "ssolve/matrix.hpp"
#include "ssolve/solver.hpp"

namespace ssolve_bench {

using SolverFunction = std::function<ssolve::SolveResult(const ssolve::CsrMatrix& a, const double* b, double* x,
                                                         const ssolve::SolveOptions& options)>;

struct NamedSolver {
  std::string name;
  ssolve::Precond precond;
  SolverFunction run;
};

std::vector<NamedSolver> eigen_solvers();   // eigen-E0 to eigen-E3
std::vector<NamedSolver> all_solvers();

// Throws, listing the valid names, if `name` is unknown.
NamedSolver find_solver(const std::string& name);

}  // namespace ssolve_bench
