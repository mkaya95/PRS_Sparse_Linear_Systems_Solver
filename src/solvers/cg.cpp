#include "ssolve/cg.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "ssolve/cg_common.hpp"
#include "ssolve/timer.hpp"

namespace ssolve {

namespace {

// One iteration moves each vector through memory as few times as possible:
//   spmv_dot       q = A p and p·q in one sweep
//   update         x, r and the dot products in one sweep
//   direction      p, with no stored z for Jacobi
SolveResult cg_loop(const Backend& backend, const double* inv_diag, const double* rhs, double* x,
                    const SolveOptions& options) {
  const std::size_t n = static_cast<std::size_t>(backend.n());
  const std::int64_t limit = options.max_iterations > 0 ? options.max_iterations : 2 * static_cast<std::int64_t>(n);
  AlignedArray<double> r_array(n), p_array(n), q_array(n);
  double* const r = r_array.data();
  double* const p = p_array.data();
  double* const q = q_array.data();

  if (options.use_initial_guess) {
    backend.spmv(x, q);
    backend.copy(q, r);
    backend.xpby(rhs, -1.0, r);  // r = b - A x
  } else {
    backend.set_zero(x);
    backend.copy(rhs, r);  // with x = 0 the residual is b, and no SpMV is needed
  }

  const double rhs_squared = backend.dot(rhs, rhs);
  const double threshold = std::max(options.tol * options.tol * rhs_squared, std::numeric_limits<double>::min());
  double rr = backend.dot(r, r);

  SolveResult result;
  const auto finish = [&](SolveStatus status) {
    result.status = status;
    result.recursive_residual = std::sqrt(rr / rhs_squared);
    return result;
  };
  if (rr < threshold) return finish(SolveStatus::Converged);

  double rz = rr;
  if (inv_diag == nullptr) {
    backend.copy(r, p);
  } else {
    backend.set_zero(p);
    backend.jacobi_direction(inv_diag, r, 0.0, p);
    rz = backend.dot(r, p);
  }

  while (result.spmv_count < limit) {
    const double pq = backend.spmv_dot(p, q);
    ++result.spmv_count;
    if (!(pq > 0.0)) return finish(SolveStatus::Breakdown);  // also catches NaN
    const double alpha = rz / pq;

    double rz_new = 0.0;
    if (inv_diag == nullptr) {
      rr = backend.cg_update(alpha, p, q, x, r);
      rz_new = rr;
    } else {
      const RrRz sums = backend.jacobi_update(alpha, p, q, x, r, inv_diag);
      rr = sums.rr;
      rz_new = sums.rz;
    }
    if (rr < threshold) return finish(SolveStatus::Converged);

    const double beta = rz_new / rz;
    if (inv_diag == nullptr) {
      backend.xpby(r, beta, p);
    } else {
      backend.jacobi_direction(inv_diag, r, beta, p);
    }
    rz = rz_new;
  }
  return finish(SolveStatus::MaxIterations);
}

}  // namespace

SolveResult cg(const Backend& backend, const double* inv_diag, const double* b, double* x, const SolveOptions& options) {
  const double start = wall_seconds();
  SolveResult result = with_rhs_scaling(static_cast<std::size_t>(backend.n()), b, x, options,
                                        [&](const double* rhs, double* solution) {
                                          return cg_loop(backend, inv_diag, rhs, solution, options);
                                        });
  result.solve_seconds = wall_seconds() - start;
  return result;
}

AlignedArray<double> jacobi_inverse_diagonal(const CsrMatrix& a) {
  AlignedArray<double> inv_diag(static_cast<std::size_t>(a.n_rows));
  for (Index i = 0; i < a.n_rows; ++i) {
    double value = 1.0;
    for (std::int64_t k = a.row_ptr[i]; k < a.row_ptr[i + 1]; ++k) {
      if (a.col[k] == i) {
        if (a.val[k] != 0.0) value = 1.0 / a.val[k];
        break;
      }
    }
    inv_diag[i] = value;
  }
  return inv_diag;
}

}  // namespace ssolve
