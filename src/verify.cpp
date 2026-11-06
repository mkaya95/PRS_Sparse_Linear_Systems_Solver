#include "ssolve/verify.hpp"

#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "ssolve/timer.hpp"

namespace ssolve {

namespace {

// ‖v‖₂, scaled by the largest entry so squares neither overflow nor underflow.
double scaled_norm(const std::vector<double>& v) {
  double largest = 0.0;
  for (const double value : v) {
    if (!std::isfinite(value)) return std::numeric_limits<double>::quiet_NaN();
    largest = std::fmax(largest, std::fabs(value));
  }
  if (largest == 0.0) return 0.0;
  double sum = 0.0;
  for (const double value : v) {
    const double ratio = value / largest;
    sum += ratio * ratio;
  }
  return largest * std::sqrt(sum);
}

}  // namespace

void validate_cg_matrix(const CsrMatrix& a) {
  if (a.n_rows <= 0 || a.n_rows != a.n_cols) throw std::runtime_error("CG needs a nonempty square matrix");
  for (Index i = 0; i < a.n_rows; ++i) {
    bool positive_diagonal = false;
    for (std::int64_t k = a.row_ptr[i]; k < a.row_ptr[i + 1]; ++k) {
      if (!std::isfinite(a.val[k])) throw std::runtime_error("CG matrix values must be finite");
      if (a.col[k] == i) positive_diagonal = a.val[k] > 0.0;
    }
    if (!positive_diagonal) throw std::runtime_error("CG requires a positive diagonal in every row");
  }
  const double defect = symmetry_defect(a);
  if (!std::isfinite(defect) || defect > 1e-8) throw std::runtime_error("matrix is not symmetric (CG symmetry check)");
}

std::string matrix_fingerprint(const CsrMatrix& a) {
  // Stable FNV-1a over dimensions, structure and FP64 bit patterns (not a security hash).
  std::uint64_t hash = 14695981039346656037ULL;
  const auto mix = [&](std::uint64_t word) {
    for (int byte = 0; byte < 8; ++byte) {
      hash = (hash ^ (word & 255)) * 1099511628211ULL;
      word >>= 8;
    }
  };
  mix(a.n_rows);
  mix(a.n_cols);
  for (const auto ptr : a.row_ptr) mix(ptr);
  for (std::int64_t k = 0; k < a.nnz(); ++k) {
    mix(a.col[k]);
    std::uint64_t bits;
    std::memcpy(&bits, &a.val[k], sizeof(bits));
    mix(bits);
  }
  std::ostringstream out;
  out << std::hex << hash;
  return out.str();
}

void spmv_reference(const CsrMatrix& a, const double* x, double* y) {
  for (Index i = 0; i < a.n_rows; ++i) {
    double sum = 0.0;
    for (std::int64_t k = a.row_ptr[i]; k < a.row_ptr[i + 1]; ++k) sum += a.val[k] * x[a.col[k]];
    y[i] = sum;
  }
}

template <class Ptr>
double residual_impl(const CsrMatrixT<Ptr>& a, const double* b, const double* x) {
  for (Index j = 0; j < a.n_cols; ++j) {
    if (!std::isfinite(x[j])) return std::numeric_limits<double>::quiet_NaN();
  }
  const std::size_t n = static_cast<std::size_t>(a.n_rows);
  std::vector<double> residual(n);
  for (Index i = 0; i < a.n_rows; ++i) {
    double sum = 0.0;
    for (std::int64_t k = a.row_ptr[i]; k < a.row_ptr[i + 1]; ++k) sum += a.val[k] * x[a.col[k]];
    residual[i] = sum;
  }
  for (std::size_t i = 0; i < n; ++i) residual[i] = b[i] - residual[i];
  const double residual_norm = scaled_norm(residual);
  const double rhs_norm = scaled_norm(std::vector<double>(b, b + n));
  if (rhs_norm == 0.0) return residual_norm;
  return residual_norm / rhs_norm;
}

double true_relative_residual(const CsrMatrix& a, const double* b, const double* x) {
  return residual_impl(a, b, x);
}

double true_relative_residual(const CsrMatrix32& a, const double* b, const double* x) {
  return residual_impl(a, b, x);
}

void verify_result(const CsrMatrix& a, const double* b, const double* x, const SolveOptions& options,
                   SolveResult& result) {
  const double start = wall_seconds();
  accept_verified_residual(result, true_relative_residual(a, b, x), options);
  result.verification_seconds = wall_seconds() - start;
}

double symmetry_defect(const CsrMatrix& a) {
  if (a.n_rows != a.n_cols) throw std::runtime_error("symmetry check needs a square matrix");
  const std::size_t n = static_cast<std::size_t>(a.n_rows);
  std::mt19937_64 rng(20261104);
  std::uniform_real_distribution<double> value(-1.0, 1.0);
  std::vector<double> x(n), y(n), ax(n), ay(n);
  for (std::size_t i = 0; i < n; ++i) {
    x[i] = value(rng);
    y[i] = value(rng);
  }
  spmv_reference(a, x.data(), ax.data());
  spmv_reference(a, y.data(), ay.data());

  double x_ay = 0.0, y_ax = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    x_ay += x[i] * ay[i];
    y_ax += y[i] * ax[i];
  }
  const std::vector<double> values(a.val.begin(), a.val.end());
  const double scale = scaled_norm(x) * scaled_norm(y) * scaled_norm(values);
  return scale == 0.0 ? 0.0 : std::fabs(x_ay - y_ax) / scale;
}

}  // namespace ssolve
