#include "ssolve/verify.hpp"

#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

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

void spmv_reference(const CsrMatrix& a, const double* x, double* y) {
  for (Index i = 0; i < a.n_rows; ++i) {
    double sum = 0.0;
    for (std::int64_t k = a.row_ptr[i]; k < a.row_ptr[i + 1]; ++k) sum += a.val[k] * x[a.col[k]];
    y[i] = sum;
  }
}

double true_relative_residual(const CsrMatrix& a, const double* b, const double* x) {
  const std::size_t n = static_cast<std::size_t>(a.n_rows);
  std::vector<double> residual(n);
  spmv_reference(a, x, residual.data());
  for (std::size_t i = 0; i < n; ++i) residual[i] = b[i] - residual[i];
  const double residual_norm = scaled_norm(residual);
  const double rhs_norm = scaled_norm(std::vector<double>(b, b + n));
  if (rhs_norm == 0.0) return residual_norm;
  return residual_norm / rhs_norm;
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
