// Types shared by every iterative solver.
#pragma once

#include <cstdint>

namespace ssolve {

enum class SolveStatus {
  Converged,      // recursive ‖r‖ < tol · ‖b‖
  MaxIterations,  // ran out of iterations
  Breakdown       // pᵀAp is not positive or is NaN: the matrix is not SPD or holds invalid values
};

inline const char* to_string(SolveStatus status) {
  switch (status) {
    case SolveStatus::Converged:
      return "Converged";
    case SolveStatus::MaxIterations:
      return "MaxIterations";
    case SolveStatus::Breakdown:
      return "Breakdown";
  }
  return "Unknown";
}

enum class Precond { None, Jacobi };

inline const char* to_string(Precond precond) { return precond == Precond::None ? "none" : "jacobi"; }

struct SolveOptions {
  double tol = 1e-8;
  std::int64_t max_iterations = 0;  // 0 means 2n, Eigen's default
  bool use_initial_guess = false;   // false solves from x = 0, whatever x holds
};

struct SolveResult {
  SolveStatus status = SolveStatus::MaxIterations;
  std::int64_t spmv_count = 0;      // SpMVs in the iteration loop, the count to compare between solvers
  double recursive_residual = 0.0;  // the solver's own ‖r‖ / ‖b‖ estimate
  double setup_seconds = 0.0;
  double solve_seconds = 0.0;
};

}  // namespace ssolve
