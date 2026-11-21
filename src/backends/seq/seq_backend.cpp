// The sequential CSR backend.
//
// The matrix is plain CSR, read front to back. Each row goes through masked_row (masked_row.hpp): it is
// read as W consecutive entries with those past its end switched off, so the loop has no branch on the
// row's length. With the row-by-row loop it replaced, a fifth of the processor's work was thrown away
// on wrong guesses of where a row ends (VTune, docs/results/vtune-i5-1335U-linux.md): thermal2's rows have
// 5 to 9 entries, in no pattern. W is 4, 8 or 16, chosen once from the row lengths.
//
// The vector kernels are shared with the other formats in seq_vectors.hpp.
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>

#include "seq_vectors.hpp"
#include "ssolve/backend.hpp"
#include "ssolve/kernels_common.hpp"
#include "ssolve/masked_row.hpp"

namespace ssolve {

namespace {

template <class Ptr, bool UnitDiagonal>
class SeqBackend final : public SeqVectorBackend {
 public:
  explicit SeqBackend(CsrMatrixT<Ptr> a)
      : SeqVectorBackend(static_cast<std::size_t>(a.n_rows)), a_(std::move(a)),
        width_(masked_width(a_.row_ptr.data(), a_.n_rows)) {}

  Index n() const override { return a_.n_rows; }

  void spmv(const double* SSOLVE_RESTRICT x, double* SSOLVE_RESTRICT y) const override {
    with_width([&](auto w) { sweep<decltype(w)::value, false>(x, y); });
  }

  double spmv_dot(const double* SSOLVE_RESTRICT x, double* SSOLVE_RESTRICT y) const override {
    double dot = 0.0;
    with_width([&](auto w) { dot = sweep<decltype(w)::value, true>(x, y); });
    return dot;
  }

 private:
  template <class Body>
  void with_width(Body&& body) const {
    if (width_ == 4) {
      body(std::integral_constant<int, 4>{});
    } else if (width_ == 8) {
      body(std::integral_constant<int, 8>{});
    } else {
      body(std::integral_constant<int, 16>{});
    }
  }

  // y = A x, and x·y when Dot.
  template <int W, bool Dot>
  double sweep(const double* SSOLVE_RESTRICT x, double* SSOLVE_RESTRICT y) const {
    const Ptr* SSOLVE_RESTRICT const row_ptr = a_.row_ptr.data();
    const Index* SSOLVE_RESTRICT const col = a_.col.data();
    const double* SSOLVE_RESTRICT const val = a_.val.data();
    double dot = 0.0;
    for (std::size_t i = 0; i < size_; ++i) {
      // Widen once per row, so a 32-bit row pointer adds no sign extension to the inner loop.
      const std::int64_t start = row_ptr[i];
      double value = masked_row<W>(col + start, val + start, x, row_ptr[i + 1] - start);
      if (UnitDiagonal) value += x[i];
      y[i] = value;
      if (Dot) dot += x[i] * value;
    }
    return dot;
  }

  CsrMatrixT<Ptr> a_;
  int width_;
};

template <bool UnitDiagonal>
std::unique_ptr<Backend> make(CsrMatrix a, bool allow_narrow_row_ptr) {
  if (a.n_rows != a.n_cols) throw std::runtime_error("the solver backend needs a square matrix");
  if (allow_narrow_row_ptr && fits_row_ptr32(a)) {
    return std::make_unique<SeqBackend<std::int32_t, UnitDiagonal>>(narrow_row_ptr(std::move(a)));
  }
  return std::make_unique<SeqBackend<std::int64_t, UnitDiagonal>>(std::move(a));
}

}  // namespace

void require_no_diagonal(const CsrMatrix& a) {
  for (Index i = 0; i < a.n_rows; ++i) {
    for (std::int64_t k = a.row_ptr[i]; k < a.row_ptr[i + 1]; ++k) {
      if (a.col[k] == i) throw std::runtime_error("a unit-diagonal backend needs the matrix without its diagonal");
    }
  }
}

std::unique_ptr<Backend> make_seq_backend(CsrMatrix a, bool allow_narrow_row_ptr) {
  return make<false>(std::move(a), allow_narrow_row_ptr);
}

std::unique_ptr<Backend> make_seq_backend_unit_diagonal(CsrMatrix off_diagonal) {
  require_no_diagonal(off_diagonal);
  return make<true>(std::move(off_diagonal), true);
}

}  // namespace ssolve
