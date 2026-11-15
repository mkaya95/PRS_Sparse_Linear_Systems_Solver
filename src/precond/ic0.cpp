#include "ssolve/ic0.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "ssolve/backend.hpp"

namespace ssolve {

namespace {

constexpr int kMaxAttempts = 40;

// Fills f.val and f.inv_diag from the scaled lower triangle `lower` and the diagonal 1 + shift.
// Returns false at the first pivot that is not finite and positive.
bool factorize(const std::vector<double>& lower, double shift, std::vector<double>& row_values, Ic0Factor& f) {
  const std::int64_t* const row_ptr = f.row_ptr.data();
  const Index* const col = f.col.data();
  double* const val = f.val.data();
  for (Index i = 0; i < f.n; ++i) {
    // row_values[c] holds l(i, c) for the columns of row i already computed, and 0 elsewhere.
    double diagonal = 1.0 + shift;
    bool ok = true;
    for (std::int64_t k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
      const Index j = col[k];
      double sum = lower[static_cast<std::size_t>(k)];
      for (std::int64_t m = row_ptr[j]; m < row_ptr[j + 1]; ++m) sum -= row_values[col[m]] * val[m];
      const double entry = sum * f.inv_diag[j];
      val[k] = entry;
      row_values[j] = entry;
      diagonal -= entry * entry;
    }
    if (!(diagonal > 0.0) || !std::isfinite(diagonal)) ok = false;
    for (std::int64_t k = row_ptr[i]; k < row_ptr[i + 1]; ++k) row_values[col[k]] = 0.0;
    if (!ok) return false;
    f.inv_diag[i] = 1.0 / std::sqrt(diagonal);
  }
  return true;
}

}  // namespace

Ic0Factor ic0(const CsrMatrix& a, double initial_shift) {
  if (a.n_rows != a.n_cols) throw std::runtime_error("IC(0) needs a square matrix");
  const Index n = a.n_rows;
  const std::size_t un = static_cast<std::size_t>(n);

  Ic0Factor f;
  f.n = n;
  f.row_ptr = AlignedArray<std::int64_t>(un + 1);
  f.scale = AlignedArray<double>(un);
  f.inv_diag = AlignedArray<double>(un);
  f.row_ptr[0] = 0;
  for (Index i = 0; i < n; ++i) {
    std::int64_t count = 0;
    double diagonal = 0.0;
    for (std::int64_t k = a.row_ptr[i]; k < a.row_ptr[i + 1]; ++k) {
      if (k > a.row_ptr[i] && a.col[k] <= a.col[k - 1]) {
        throw std::runtime_error("IC(0) needs the columns of every row in ascending order");
      }
      if (a.col[k] < i) ++count;
      if (a.col[k] == i) diagonal = a.val[k];
    }
    if (!std::isfinite(diagonal) || diagonal <= 0.0) {
      throw std::runtime_error("IC(0) requires a finite positive diagonal in every row");
    }
    f.scale[i] = 1.0 / std::sqrt(diagonal);
    f.row_ptr[i + 1] = f.row_ptr[i] + count;
  }
  const std::size_t nnz = static_cast<std::size_t>(f.row_ptr[un]);
  f.col = AlignedArray<Index>(nnz);
  f.val = AlignedArray<double>(nnz);
  std::vector<double> lower(nnz);  // the scaled entries of A, kept for a restart
  for (Index i = 0; i < n; ++i) {
    std::int64_t at = f.row_ptr[i];
    for (std::int64_t k = a.row_ptr[i]; k < a.row_ptr[i + 1] && a.col[k] < i; ++k) {
      f.col[at] = a.col[k];
      lower[static_cast<std::size_t>(at++)] = a.val[k] * f.scale[i] * f.scale[a.col[k]];
    }
  }

  std::vector<double> row_values(un, 0.0);
  double shift = 0.0;
  for (f.attempts = 1; !factorize(lower, shift, row_values, f); ++f.attempts) {
    if (f.attempts >= kMaxAttempts) throw std::runtime_error("IC(0) failed after 40 shifts");
    shift = std::max(initial_shift, 2.0 * shift);
  }
  f.shift = shift;
  return f;
}

void ic0_apply(const Ic0Factor& f, const double* r, double* z) {
  const Index n = f.n;
  const std::int64_t* const row_ptr = f.row_ptr.data();
  const Index* const col = f.col.data();
  const double* const val = f.val.data();
  const double* const inv_diag = f.inv_diag.data();
  const double* const scale = f.scale.data();

  // y = L⁻¹ S r, forward along the rows.
  for (Index i = 0; i < n; ++i) {
    double sum = scale[i] * r[i];
    for (std::int64_t k = row_ptr[i]; k < row_ptr[i + 1]; ++k) sum -= val[k] * z[col[k]];
    z[i] = sum * inv_diag[i];
  }
  // z = S L⁻ᵀ y. Row i of L is column i of Lᵀ, so once zᵢ is known it is taken out of the rows above.
  for (Index i = n - 1; i >= 0; --i) {
    const double zi = z[i] * inv_diag[i];
    for (std::int64_t k = row_ptr[i]; k < row_ptr[i + 1]; ++k) z[col[k]] -= val[k] * zi;
    z[i] = scale[i] * zi;
  }
}

namespace {

// z is stored: the triangular solves need all of r before any of z is known.
class Ic0Preconditioner final : public Preconditioner {
 public:
  explicit Ic0Preconditioner(const CsrMatrix& a) : f_(ic0(a)), z_(static_cast<std::size_t>(a.n_rows)) {}

  double start(const Backend& backend, const double* r, double, double* p) override {
    ic0_apply(f_, r, z_.data());
    backend.copy(z_.data(), p);
    return backend.dot(r, z_.data());
  }

  RrRz update(const Backend& backend, double a, const double* p, const double* q, double* x, double* r) override {
    const double rr = backend.cg_update(a, p, q, x, r);
    ic0_apply(f_, r, z_.data());
    return {rr, backend.dot(r, z_.data())};
  }

  void direction(const Backend& backend, const double*, double beta, double* p) override {
    backend.xpby(z_.data(), beta, p);
  }

 private:
  Ic0Factor f_;
  AlignedArray<double> z_;
};

}  // namespace

std::unique_ptr<Preconditioner> make_ic0_preconditioner(const CsrMatrix& a) {
  return std::make_unique<Ic0Preconditioner>(a);
}

}  // namespace ssolve
