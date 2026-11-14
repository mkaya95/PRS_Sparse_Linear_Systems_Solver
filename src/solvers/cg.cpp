#include "ssolve/cg.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>

#include "ssolve/cg_common.hpp"
#include "ssolve/timer.hpp"
#include "ssolve/verify.hpp"

namespace ssolve {

namespace {

// One iteration moves each vector through memory as few times as possible:
//   spmv_dot       q = A p and p·q in one sweep
//   update         x, r and the dot products in one sweep
//   direction      p, the preconditioner's job; none and Jacobi never store z
SolveResult cg_loop(const Backend& backend, Preconditioner& precond, const double* rhs, double* x,
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
  const auto finish = [&](SolveStatus status, const char* detail = "") {
    result.status = status;
    result.detail = detail;
    result.recursive_residual = std::sqrt(rr / rhs_squared);
    return result;
  };
  if (!std::isfinite(rhs_squared) || !std::isfinite(rr) || !std::isfinite(threshold)) {
    return finish(SolveStatus::Breakdown, "non-finite initial norm or stopping threshold");
  }
  if (rr < threshold) return finish(SolveStatus::Converged);

  double rz = precond.start(backend, r, rr, p);

  while (result.spmv_count < limit) {
    const double pq = backend.spmv_dot(p, q);
    ++result.spmv_count;
    if (!(pq > 0.0) || !std::isfinite(pq) || !(rz > 0.0) || !std::isfinite(rz)) {
      return finish(SolveStatus::Breakdown, "invalid curvature or preconditioned residual product");
    }
    const double alpha = rz / pq;
    if (!std::isfinite(alpha)) return finish(SolveStatus::Breakdown, "non-finite CG step");

    const RrRz sums = precond.update(backend, alpha, p, q, x, r);
    rr = sums.rr;
    const double rz_new = sums.rz;
    if (!std::isfinite(rr) || !std::isfinite(rz_new)) {
      return finish(SolveStatus::Breakdown, "non-finite residual update");
    }
    if (rr < threshold) return finish(SolveStatus::Converged);
    if (!(rz_new > 0.0)) return finish(SolveStatus::Breakdown, "nonpositive preconditioned residual product");

    const double beta = rz_new / rz;
    if (!std::isfinite(beta)) return finish(SolveStatus::Breakdown, "non-finite direction coefficient");
    precond.direction(backend, r, beta, p);
    rz = rz_new;
  }
  return finish(SolveStatus::MaxIterations);
}

}  // namespace

SolveResult cg(const Backend& backend, Preconditioner& precond, const double* b, double* x,
               const SolveOptions& options) {
  validate_options(options);
  if (options.history != nullptr) {
    SolveOptions plain = options;
    plain.history = nullptr;
    options.history->clear();
    if (options.use_initial_guess) return cg(backend, precond, b, x, plain);
    return cg_with_history(backend, precond, b, x, plain, *options.history);
  }
  const double start = wall_seconds();
  SolveResult result = with_rhs_scaling(static_cast<std::size_t>(backend.n()), b, x, options,
                                        [&](const double* rhs, double* solution) {
                                          return cg_loop(backend, precond, rhs, solution, options);
                                        });
  result.solve_seconds = wall_seconds() - start;
  return result;
}

std::unique_ptr<Preconditioner> preconditioner_for(const Backend& backend, const double* inv_diag) {
  if (inv_diag == nullptr) return make_identity_preconditioner();
  for (Index i = 0; i < backend.n(); ++i) {
    if (!std::isfinite(inv_diag[i]) || inv_diag[i] <= 0.0) {
      throw std::runtime_error("Jacobi inverse diagonal must be finite and positive");
    }
  }
  return make_jacobi_preconditioner(inv_diag);
}

SolveResult cg(const Backend& backend, const double* inv_diag, const double* b, double* x, const SolveOptions& options) {
  validate_options(options);
  const std::unique_ptr<Preconditioner> precond = preconditioner_for(backend, inv_diag);
  return cg(backend, *precond, b, x, options);
}

SolveResult verified_cg(const CsrMatrix& a, const Backend& backend, Preconditioner& precond, const double* b,
                        double* x, const SolveOptions& options) {
  SolveResult result = cg(backend, precond, b, x, options);
  verify_result(a, b, x, options, result);
  return result;
}

SolveResult verified_cg(const CsrMatrix& a, const Backend& backend, const double* inv_diag, const double* b, double* x,
                        const SolveOptions& options) {
  SolveResult result = cg(backend, inv_diag, b, x, options);
  verify_result(a, b, x, options, result);
  return result;
}

AlignedArray<double> jacobi_inverse_diagonal(const CsrMatrix& a) {
  AlignedArray<double> inv_diag(static_cast<std::size_t>(a.n_rows));
  for (Index i = 0; i < a.n_rows; ++i) {
    double value = 0.0;
    for (std::int64_t k = a.row_ptr[i]; k < a.row_ptr[i + 1]; ++k) {
      if (a.col[k] == i) {
        if (!std::isfinite(a.val[k]) || a.val[k] <= 0.0) {
          throw std::runtime_error("Jacobi requires a finite positive diagonal in every row");
        }
        value = 1.0 / a.val[k];
        break;
      }
    }
    if (!std::isfinite(value) || value <= 0.0) {
      throw std::runtime_error("Jacobi requires a finite positive inverse diagonal in every row");
    }
    inv_diag[i] = value;
  }
  return inv_diag;
}

}  // namespace ssolve
