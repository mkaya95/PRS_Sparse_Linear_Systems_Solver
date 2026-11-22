#include "ssolve/jacobi_scaled.hpp"

#include <cmath>
#include <stdexcept>

#include "ssolve/cg.hpp"
#include "ssolve/kernels_common.hpp"
#include "ssolve/timer.hpp"

namespace ssolve {

namespace {

class ScaledJacobiPreconditioner final : public Preconditioner {
 public:
  explicit ScaledJacobiPreconditioner(const double* diagonal) : diagonal_(diagonal) {}

  double start(const Backend& backend, const double* r, double, double* p) override {
    backend.copy(r, p);  // z = r̂
    return backend.dot(r, r);
  }

  // The backend's Jacobi update with D in place of D⁻¹ returns {r̂·r̂, Σ dᵢ r̂ᵢ²} = {r·z, r·r} here.
  RrRz update(const Backend& backend, double a, const double* p, const double* q, double* x, double* r) override {
    const RrRz sums = backend.jacobi_update(a, p, q, x, r, diagonal_);
    return {sums.rz, sums.rr};
  }

  void direction(const Backend& backend, const double* r, double beta, double* p) override {
    backend.xpby(r, beta, p);
  }

  double norm_squared(const Backend& backend, const double* v) override {
    const std::size_t n = static_cast<std::size_t>(backend.n());
    const double* SSOLVE_RESTRICT const d = diagonal_;
    double sum = 0.0;
#pragma omp simd reduction(+ : sum)
    for (std::size_t i = 0; i < n; ++i) sum += d[i] * v[i] * v[i];
    return sum;
  }

 private:
  const double* diagonal_;
};

}  // namespace

JacobiScaling jacobi_scaling(const CsrMatrix& a) {
  if (a.n_rows != a.n_cols) throw std::runtime_error("the solver backend needs a square matrix");
  const Index n = a.n_rows;
  const std::size_t un = static_cast<std::size_t>(n);
  JacobiScaling s;
  s.diagonal = AlignedArray<double>(un);
  s.scale = AlignedArray<double>(un);
  const AlignedArray<double> inv_diag = jacobi_inverse_diagonal(a);  // throws on a bad diagonal
  for (std::size_t i = 0; i < un; ++i) {
    s.diagonal[i] = 1.0 / inv_diag[i];
    s.scale[i] = std::sqrt(inv_diag[i]);
  }
  CsrMatrix& off = s.off_diagonal;
  off.n_rows = off.n_cols = n;
  off.row_ptr = AlignedArray<std::int64_t>(un + 1);
  off.row_ptr[0] = 0;
  for (Index i = 0; i < n; ++i) {
    std::int64_t count = 0;
    for (std::int64_t k = a.row_ptr[i]; k < a.row_ptr[i + 1]; ++k) count += a.col[k] != i;
    off.row_ptr[i + 1] = off.row_ptr[i] + count;
  }
  off.col = AlignedArray<Index>(static_cast<std::size_t>(off.nnz()));
  off.val = AlignedArray<double>(static_cast<std::size_t>(off.nnz()));
  for (Index i = 0; i < n; ++i) {
    std::int64_t at = off.row_ptr[i];
    for (std::int64_t k = a.row_ptr[i]; k < a.row_ptr[i + 1]; ++k) {
      const Index j = a.col[k];
      if (j == i) continue;
      off.col[at] = j;
      off.val[at++] = s.scale[i] * a.val[k] * s.scale[j];
    }
  }
  return s;
}

std::unique_ptr<Preconditioner> make_scaled_jacobi_preconditioner(const double* diagonal) {
  return std::make_unique<ScaledJacobiPreconditioner>(diagonal);
}

SolveResult cg_scaled_jacobi(const Backend& backend, const JacobiScaling& scaling, const double* b, double* x,
                             const SolveOptions& options) {
  const std::size_t n = static_cast<std::size_t>(backend.n());
  if (scaling.scale.size() != n) throw std::runtime_error("the scaling does not match the backend");
  const double start = wall_seconds();
  const double* SSOLVE_RESTRICT const s = scaling.scale.data();
  AlignedArray<double> scaled_b(n), scaled_x(n);
  for (std::size_t i = 0; i < n; ++i) scaled_b[i] = s[i] * b[i];
  if (options.use_initial_guess) {
    for (std::size_t i = 0; i < n; ++i) scaled_x[i] = x[i] / s[i];
  }
  ScaledJacobiPreconditioner precond(scaling.diagonal.data());
  SolveResult result = cg(backend, precond, scaled_b.data(), scaled_x.data(), options);
  for (std::size_t i = 0; i < n; ++i) x[i] = s[i] * scaled_x[i];
  result.solve_seconds = wall_seconds() - start;
  return result;
}

}  // namespace ssolve
