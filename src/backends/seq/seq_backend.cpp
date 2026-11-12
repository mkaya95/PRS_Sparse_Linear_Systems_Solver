// The sequential CSR backend, as chosen by the measurements in results/.
//
// The CSR row loop uses two accumulators, the best of the variants tried on the optimization
// ladder. The vector kernels are shared with the other formats in seq_vectors.hpp.
#include <cstdint>
#include <stdexcept>
#include <utility>

#include "seq_vectors.hpp"
#include "ssolve/backend.hpp"
#include "ssolve/kernels_common.hpp"

namespace ssolve {

namespace {

template <class Ptr>
class SeqBackend final : public SeqVectorBackend {
 public:
  explicit SeqBackend(CsrMatrixT<Ptr> a)
      : SeqVectorBackend(static_cast<std::size_t>(a.n_rows)), a_(std::move(a)) {}

  Index n() const override { return a_.n_rows; }

  void spmv(const double* SSOLVE_RESTRICT x, double* SSOLVE_RESTRICT y) const override {
    const Ptr* SSOLVE_RESTRICT const row_ptr = a_.row_ptr.data();
    const Index* SSOLVE_RESTRICT const col = a_.col.data();
    const double* SSOLVE_RESTRICT const val = a_.val.data();
    for (std::size_t i = 0; i < size_; ++i) y[i] = row_times_vector(row_ptr, col, val, x, i);
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
