// Sequential kernels, as chosen by the measurements in results/.
//
// The CSR row loop uses two accumulators, the best of the variants tried on the optimization
// ladder. Vector loops need `omp simd`, because GCC does not otherwise vectorize floating-point
// sums, and fused kernels save a separate pass over each vector.
#include <cstdint>
#include <stdexcept>
#include <utility>

#include "ssolve/backend.hpp"
#include "ssolve/kernels_common.hpp"
#include "ssolve/verify.hpp"

namespace ssolve {

namespace {

template <class Ptr>
class SeqBackend final : public Backend {
 public:
  explicit SeqBackend(CsrMatrixT<Ptr> a) : a_(std::move(a)), size_(static_cast<std::size_t>(a_.n_rows)) {}

  Index n() const override { return a_.n_rows; }
  double true_residual(const double* b, const double* x) const override {
    return true_relative_residual(a_, b, x);
  }

  void set_zero(double* SSOLVE_RESTRICT x) const override {
#pragma omp simd
    for (std::size_t i = 0; i < size_; ++i) x[i] = 0.0;
  }

  void copy(const double* SSOLVE_RESTRICT source, double* SSOLVE_RESTRICT destination) const override {
#pragma omp simd
    for (std::size_t i = 0; i < size_; ++i) destination[i] = source[i];
  }

  double dot(const double* SSOLVE_RESTRICT x, const double* SSOLVE_RESTRICT y) const override {
    double sum = 0.0;
#pragma omp simd reduction(+ : sum)
    for (std::size_t i = 0; i < size_; ++i) sum += x[i] * y[i];
    return sum;
  }

  void spmv(const double* SSOLVE_RESTRICT x, double* SSOLVE_RESTRICT y) const override {
    const Ptr* SSOLVE_RESTRICT const row_ptr = a_.row_ptr.data();
    const Index* SSOLVE_RESTRICT const col = a_.col.data();
    const double* SSOLVE_RESTRICT const val = a_.val.data();
    for (std::size_t i = 0; i < size_; ++i) y[i] = row_times_vector(row_ptr, col, val, x, i);
  }

  void xpby(const double* SSOLVE_RESTRICT x, double beta, double* SSOLVE_RESTRICT y) const override {
#pragma omp simd
    for (std::size_t i = 0; i < size_; ++i) y[i] = x[i] + beta * y[i];
  }

  double spmv_dot(const double* SSOLVE_RESTRICT x, double* SSOLVE_RESTRICT y) const override {
    const Ptr* SSOLVE_RESTRICT const row_ptr = a_.row_ptr.data();
    const Index* SSOLVE_RESTRICT const col = a_.col.data();
    const double* SSOLVE_RESTRICT const val = a_.val.data();
    double sum = 0.0;
    for (std::size_t i = 0; i < size_; ++i) {
      const double value = row_times_vector(row_ptr, col, val, x, i);
      y[i] = value;
      sum += x[i] * value;
    }
    return sum;
  }

  double cg_update(double a, const double* SSOLVE_RESTRICT p, const double* SSOLVE_RESTRICT q, double* SSOLVE_RESTRICT x,
                   double* SSOLVE_RESTRICT r) const override {
    double rr = 0.0;
#pragma omp simd reduction(+ : rr)
    for (std::size_t i = 0; i < size_; ++i) {
      x[i] += a * p[i];
      r[i] -= a * q[i];
      rr += r[i] * r[i];
    }
    return rr;
  }

  RrRz jacobi_update(double a, const double* SSOLVE_RESTRICT p, const double* SSOLVE_RESTRICT q,
                     double* SSOLVE_RESTRICT x, double* SSOLVE_RESTRICT r,
                     const double* SSOLVE_RESTRICT inv_diag) const override {
    double rr = 0.0, rz = 0.0;
#pragma omp simd reduction(+ : rr, rz)
    for (std::size_t i = 0; i < size_; ++i) {
      x[i] += a * p[i];
      r[i] -= a * q[i];
      const double r_squared = r[i] * r[i];
      rr += r_squared;
      rz += r_squared * inv_diag[i];
    }
    return {rr, rz};
  }

  void jacobi_direction(const double* SSOLVE_RESTRICT inv_diag, const double* SSOLVE_RESTRICT r, double beta,
                        double* SSOLVE_RESTRICT p) const override {
#pragma omp simd
    for (std::size_t i = 0; i < size_; ++i) p[i] = inv_diag[i] * r[i] + beta * p[i];
  }

 private:
  // Two accumulators break the dependency between consecutive additions, so the CPU can run two at once.
  static inline double row_times_vector(const Ptr* SSOLVE_RESTRICT row_ptr, const Index* SSOLVE_RESTRICT col,
                                        const double* SSOLVE_RESTRICT val, const double* SSOLVE_RESTRICT x,
                                        std::size_t i) {
    double sum0 = 0.0, sum1 = 0.0;
    // Widen once per row, so a 32-bit row pointer adds no sign extension to the inner loop.
    std::int64_t k = row_ptr[i];
    const std::int64_t end = row_ptr[i + 1];
    for (; k + 1 < end; k += 2) {
      sum0 += val[k] * x[col[k]];
      sum1 += val[k + 1] * x[col[k + 1]];
    }
    if (k < end) sum0 += val[k] * x[col[k]];
    return sum0 + sum1;
  }

  CsrMatrixT<Ptr> a_;
  std::size_t size_;
};

}  // namespace

std::unique_ptr<Backend> make_seq_backend(CsrMatrix a, bool allow_narrow_row_ptr) {
  if (a.n_rows != a.n_cols) throw std::runtime_error("the solver backend needs a square matrix");
  if (allow_narrow_row_ptr && fits_row_ptr32(a)) {
    return std::make_unique<SeqBackend<std::int32_t>>(narrow_row_ptr(std::move(a)));
  }
  return std::make_unique<SeqBackend<std::int64_t>>(std::move(a));
}

}  // namespace ssolve
