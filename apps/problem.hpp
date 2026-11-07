// The --problem option shared by bench_cg and solve: which right-hand side b to build for A.
#pragma once

#include <stdexcept>
#include <string>

#include "ssolve/aligned_array.hpp"
#include "ssolve/generators.hpp"
#include "ssolve/matrix.hpp"

namespace ssolve_cli {

// random: b = A·x for a fixed random x. a1: b = A·1. unit: b = 1.
inline ssolve::AlignedArray<double> rhs_for_problem(const std::string& problem, const ssolve::CsrMatrix& a) {
  if (problem == "random") return ssolve::rhs_for_random_solution(a);
  if (problem == "a1") return ssolve::rhs_for_unit_solution(a);
  if (problem == "unit") {
    ssolve::AlignedArray<double> b(static_cast<std::size_t>(a.n_rows));
    b.fill(1.0);
    return b;
  }
  throw std::runtime_error("--problem must be random, a1 or unit");
}

}  // namespace ssolve_cli
