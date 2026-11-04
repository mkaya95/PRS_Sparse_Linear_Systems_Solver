// The optimization ladder: each rung is CG with one more change than the one before.
//
// Old rungs stay here on purpose. They are measurement fixtures, so the results table
// can be regenerated at any time instead of being reconstructed from git history. The
// shipping solver lives in src/ and holds only what the measurements justified.
//
// Kernels are noinline so that each stays a separate unit, as with Eigen's solver boundary.
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "registry.hpp"
#include "ssolve/cg_common.hpp"
#include "ssolve/timer.hpp"

using namespace ssolve;

#define SSOLVE_NOINLINE __attribute__((noinline))

namespace ssolve_bench {

namespace {

// L0: what anyone writes first. No restrict, no pragmas, one accumulator.
struct KNaive {
  SSOLVE_NOINLINE static void spmv(const CsrMatrix& a, const double* x, double* y) {
    for (Index i = 0; i < a.n_rows; ++i) {
      double sum = 0.0;
      for (std::int64_t k = a.row_ptr[i]; k < a.row_ptr[i + 1]; ++k) sum += a.val[k] * x[a.col[k]];
      y[i] = sum;
    }
  }
  SSOLVE_NOINLINE static double dot(const double* x, const double* y, std::size_t n) {
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) sum += x[i] * y[i];
    return sum;
  }
  SSOLVE_NOINLINE static void axpy(double alpha, const double* x, double* y, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) y[i] += alpha * x[i];
  }
  SSOLVE_NOINLINE static void xpby(const double* x, double beta, double* y, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) y[i] = x[i] + beta * y[i];
  }
};

// Textbook CG: one separate pass per operation, as in Eigen's loop.
template <class K>
SolveResult cg_passes(const CsrMatrix& a, const double* b, double* x, const SolveOptions& options) {
  if (a.n_rows != a.n_cols) throw std::runtime_error("CG needs a square matrix");
  const std::size_t n = static_cast<std::size_t>(a.n_rows);
  const double start = wall_seconds();

  SolveResult result = with_rhs_scaling(n, b, x, options, [&](const double* rhs, double* solution) {
    const std::int64_t limit =
        options.max_iterations > 0 ? options.max_iterations : 2 * static_cast<std::int64_t>(n);
    AlignedArray<double> r_array(n), p_array(n), q_array(n);
    double* const r = r_array.data();
    double* const p = p_array.data();
    double* const q = q_array.data();

    if (!options.use_initial_guess) {
      for (std::size_t i = 0; i < n; ++i) solution[i] = 0.0;
    }
    K::spmv(a, solution, q);
    for (std::size_t i = 0; i < n; ++i) r[i] = rhs[i] - q[i];
    for (std::size_t i = 0; i < n; ++i) p[i] = r[i];

    const double rhs_squared = K::dot(rhs, rhs, n);
    const double threshold =
        std::max(options.tol * options.tol * rhs_squared, std::numeric_limits<double>::min());
    double rr = K::dot(r, r, n);

    SolveResult out;
    const auto finish = [&](SolveStatus status) {
      out.status = status;
      out.recursive_residual = std::sqrt(rr / rhs_squared);
      return out;
    };
    if (rr < threshold) return finish(SolveStatus::Converged);

    while (out.spmv_count < limit) {
      K::spmv(a, p, q);
      ++out.spmv_count;
      const double pq = K::dot(p, q, n);
      if (!(pq > 0.0)) return finish(SolveStatus::Breakdown);  // also catches NaN
      const double alpha = rr / pq;
      K::axpy(alpha, p, solution, n);
      K::axpy(-alpha, q, r, n);
      const double rr_new = K::dot(r, r, n);
      if (rr_new < threshold) {
        rr = rr_new;
        return finish(SolveStatus::Converged);
      }
      K::xpby(r, rr_new / rr, p, n);
      rr = rr_new;
    }
    return finish(SolveStatus::MaxIterations);
  });
  result.solve_seconds = wall_seconds() - start;
  return result;
}

}  // namespace

std::vector<NamedSolver> ladder_solvers() {
  return {
      {"L0-naive", Precond::None, cg_passes<KNaive>},
  };
}

}  // namespace ssolve_bench
