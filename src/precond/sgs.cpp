// Symmetric Gauss-Seidel, from HPCG's reference kernel ComputeSYMGS_ref (src/ComputeSYMGS_ref.cpp,
// github.com/hpcg-benchmark/hpcg). The two sweeps are HPCG's loops; only the storage differs: CSR arrays
// instead of HPCG's per-row pointers, and the diagonal values found once instead of a pointer per row.
#include "ssolve/sgs.hpp"

#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>

#include "ssolve/backend.hpp"

namespace ssolve {

SymGaussSeidel::SymGaussSeidel(CsrMatrix a) : a_(std::move(a)), diag_(static_cast<std::size_t>(a_.n_rows)) {
  if (a_.n_rows != a_.n_cols) throw std::runtime_error("symmetric Gauss-Seidel needs a square matrix");
  for (Index i = 0; i < a_.n_rows; ++i) {
    double value = 0.0;
    for (std::int64_t k = a_.row_ptr[i]; k < a_.row_ptr[i + 1]; ++k) {
      if (a_.col[k] == i) value = a_.val[k];
    }
    if (!std::isfinite(value) || value <= 0.0) {
      throw std::runtime_error("symmetric Gauss-Seidel requires a finite positive diagonal in every row");
    }
    diag_[i] = value;
  }
}

void SymGaussSeidel::apply(const double* r, double* z) const {
  // HPCG: "x should be initially zero on the first GS sweep, but we do not attempt to exploit this fact."
  const std::int64_t nrow = a_.n_rows;
  const double* const rv = r;
  double* const xv = z;
  for (std::int64_t i = 0; i < nrow; ++i) xv[i] = 0.0;

  for (std::int64_t i = 0; i < nrow; i++) {
    const double* const currentValues = a_.val.data() + a_.row_ptr[i];
    const Index* const currentColIndices = a_.col.data() + a_.row_ptr[i];
    const std::int64_t currentNumberOfNonzeros = a_.row_ptr[i + 1] - a_.row_ptr[i];
    const double currentDiagonal = diag_[i];  // Current diagonal value
    double sum = rv[i];                        // RHS value

    for (std::int64_t j = 0; j < currentNumberOfNonzeros; j++) {
      const Index curCol = currentColIndices[j];
      sum -= currentValues[j] * xv[curCol];
    }
    sum += xv[i] * currentDiagonal;  // Remove diagonal contribution from previous loop

    xv[i] = sum / currentDiagonal;
  }

  // Now the back sweep.

  for (std::int64_t i = nrow - 1; i >= 0; i--) {
    const double* const currentValues = a_.val.data() + a_.row_ptr[i];
    const Index* const currentColIndices = a_.col.data() + a_.row_ptr[i];
    const std::int64_t currentNumberOfNonzeros = a_.row_ptr[i + 1] - a_.row_ptr[i];
    const double currentDiagonal = diag_[i];  // Current diagonal value
    double sum = rv[i];                        // RHS value

    for (std::int64_t j = 0; j < currentNumberOfNonzeros; j++) {
      const Index curCol = currentColIndices[j];
      sum -= currentValues[j] * xv[curCol];
    }
    sum += xv[i] * currentDiagonal;  // Remove diagonal contribution from previous loop

    xv[i] = sum / currentDiagonal;
  }
}

namespace {

// z is stored: the sweeps need all of r before any of z is known, so nothing merges into the updates.
class SgsPreconditioner final : public Preconditioner {
 public:
  explicit SgsPreconditioner(const CsrMatrix& a) : m_(clone(a)), z_(static_cast<std::size_t>(a.n_rows)) {}

  double start(const Backend& backend, const double* r, double, double* p) override {
    m_.apply(r, z_.data());
    backend.copy(z_.data(), p);
    return backend.dot(r, z_.data());
  }

  RrRz update(const Backend& backend, double a, const double* p, const double* q, double* x, double* r) override {
    const double rr = backend.cg_update(a, p, q, x, r);
    m_.apply(r, z_.data());
    return {rr, backend.dot(r, z_.data())};
  }

  void direction(const Backend& backend, const double*, double beta, double* p) override {
    backend.xpby(z_.data(), beta, p);
  }

 private:
  SymGaussSeidel m_;
  AlignedArray<double> z_;
};

}  // namespace

std::unique_ptr<Preconditioner> make_sgs_preconditioner(const CsrMatrix& a) {
  return std::make_unique<SgsPreconditioner>(a);
}

}  // namespace ssolve
