// Right-hand side handling shared by every CG variant.
#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

#include "ssolve/solver.hpp"

namespace ssolve {

// Handles the cases that must not reach the iteration loop as they are:
//   b = 0            the solution is 0, with no SpMV at all
//   extreme |b|      ‖b‖² would underflow or overflow, so solve with b / max|bᵢ|
// Calls solve(rhs, x) for everything else. Normal inputs pass through untouched,
// so iteration counts stay directly comparable with Eigen.
// A NaN or infinity in b also passes through, and the iteration reports Breakdown.
template <class Solve>
SolveResult with_rhs_scaling(std::size_t n, const double* b, double* x, const SolveOptions& options, Solve solve) {
  double largest = 0.0;
  double probe = 0.0;  // becomes NaN if any entry is NaN or infinite, because inf * 0 is NaN
  for (std::size_t i = 0; i < n; ++i) {
    largest = std::fmax(largest, std::fabs(b[i]));
    probe += b[i] * 0.0;
  }
  if (std::isnan(probe)) return solve(b, x);

  if (largest == 0.0) {
    for (std::size_t i = 0; i < n; ++i) x[i] = 0.0;
    SolveResult result;
    result.status = SolveStatus::Converged;
    return result;
  }

  constexpr double kSmallest = 1e-100;
  constexpr double kLargest = 1e+100;
  if (largest >= kSmallest && largest <= kLargest) return solve(b, x);

  std::vector<double> scaled_rhs(n);
  for (std::size_t i = 0; i < n; ++i) scaled_rhs[i] = b[i] / largest;
  if (options.use_initial_guess) {
    for (std::size_t i = 0; i < n; ++i) x[i] /= largest;
  }
  const SolveResult result = solve(scaled_rhs.data(), x);
  for (std::size_t i = 0; i < n; ++i) x[i] *= largest;
  return result;
}

}  // namespace ssolve
