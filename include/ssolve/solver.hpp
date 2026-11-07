// Types shared by every iterative solver.
#pragma once

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace ssolve {

enum class SolveStatus {
  Converged,      // recursive residual met tol, and the true residual too once verified
  MaxIterations,  // ran out of iterations
  Breakdown,      // invalid curvature or non-finite arithmetic
  ResidualFailure // recursive convergence failed independent verification
};

inline const char* to_string(SolveStatus status) {
  switch (status) {
    case SolveStatus::Converged:
      return "Converged";
    case SolveStatus::MaxIterations:
      return "MaxIterations";
    case SolveStatus::Breakdown:
      return "Breakdown";
    case SolveStatus::ResidualFailure:
      return "ResidualFailure";
  }
  return "Unknown";
}

enum class Precond { None, Jacobi, IncompleteCholesky };

inline const char* to_string(Precond precond) {
  switch (precond) {
    case Precond::None:
      return "none";
    case Precond::Jacobi:
      return "jacobi";
    case Precond::IncompleteCholesky:
      return "incomplete-cholesky";
  }
  return "unknown";
}

struct SolveOptions {
  double tol = 1e-8;
  std::int64_t max_iterations = 0;  // 0 means 2n, Eigen's default
  bool use_initial_guess = false;   // false solves from x = 0, whatever x holds
};

inline void validate_options(const SolveOptions& options) {
  if (!std::isfinite(options.tol) || options.tol <= 0.0) {
    throw std::runtime_error("tolerance must be finite and positive");
  }
  if (options.max_iterations < 0) throw std::runtime_error("maximum iterations must be nonnegative");
}

struct SolveResult {
  SolveStatus status = SolveStatus::MaxIterations;
  std::int64_t spmv_count = 0;      // SpMVs in the iteration loop, the count to compare between solvers
  double recursive_residual = 0.0;  // the solver's own ‖r‖ / ‖b‖ estimate
  double setup_seconds = 0.0;
  double solve_seconds = 0.0;
  double true_residual = 0.0;
  double verification_seconds = 0.0;
  const char* detail = "";
};

inline void accept_verified_residual(SolveResult& result, double residual, const SolveOptions& options) {
  result.true_residual = residual;
  if (!std::isfinite(residual)) {
    result.status = SolveStatus::Breakdown;
    result.detail = "non-finite solution or true residual";
  } else if (result.status == SolveStatus::Converged && residual > options.tol) {
    result.status = SolveStatus::ResidualFailure;
    result.detail = "true residual exceeds tolerance";
  }
}

}  // namespace ssolve
