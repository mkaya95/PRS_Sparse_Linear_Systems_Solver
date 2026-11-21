#include "ssolve/triangular.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "ssolve/kernels_common.hpp"

namespace ssolve {

namespace {

// Pads the rows given by (ptr, col, val) into blocks; ptr has n + 1 entries.
PaddedTriangle pad_rows(Index n, const std::vector<std::int64_t>& ptr, const std::vector<Index>& col,
                        const std::vector<double>& val) {
  const std::int64_t blocks = (static_cast<std::int64_t>(n) + kTriangularBlock - 1) / kTriangularBlock;
  PaddedTriangle t;
  t.block_start = AlignedArray<std::int64_t>(static_cast<std::size_t>(blocks + 1));
  t.width = AlignedArray<std::uint8_t>(static_cast<std::size_t>(blocks));
  t.block_start[0] = 0;
  for (std::int64_t b = 0; b < blocks; ++b) {
    const Index begin = static_cast<Index>(b * kTriangularBlock);
    const Index end = std::min<Index>(n, begin + kTriangularBlock);
    std::int64_t width = 0;
    for (Index i = begin; i < end; ++i) width = std::max(width, ptr[i + 1] - ptr[i]);
    if (width > 255) throw std::runtime_error("triangular factor: a row has more than 255 entries");
    t.width[b] = static_cast<std::uint8_t>(width);
    t.block_start[b + 1] = t.block_start[b] + width * (end - begin);
  }
  const std::size_t stored = static_cast<std::size_t>(t.block_start[blocks]);
  t.col = AlignedArray<Index>(stored);
  t.val = AlignedArray<FactorValue>(stored);
  for (std::int64_t b = 0; b < blocks; ++b) {
    const Index begin = static_cast<Index>(b * kTriangularBlock);
    const Index end = std::min<Index>(n, begin + kTriangularBlock);
    const int width = t.width[b];
    for (Index i = begin; i < end; ++i) {
      const std::int64_t at = t.block_start[b] + static_cast<std::int64_t>(i - begin) * width;
      const std::int64_t count = ptr[i + 1] - ptr[i];
      for (int k = 0; k < width; ++k) {
        t.col[at + k] = k < count ? col[ptr[i] + k] : n;
        t.val[at + k] = k < count ? static_cast<FactorValue>(val[ptr[i] + k]) : FactorValue{0};
      }
    }
  }
  return t;
}

// Σ v[k] w[c[k]] over one padded row. W >= 0 is the width known at compile time, which unrolls the loop
// and leaves no branch; W = -1 reads it from `width`.
template <int W>
inline double row_sum(const Index* SSOLVE_RESTRICT c, const FactorValue* SSOLVE_RESTRICT v, const double* w, int width) {
  const int count = W >= 0 ? W : width;
  double s0 = 0.0, s1 = 0.0;
  int k = 0;
#pragma GCC unroll 8
  for (; k + 1 < count; k += 2) {
    s0 += v[k] * w[c[k]];
    s1 += v[k + 1] * w[c[k + 1]];
  }
  if (k < count) s0 += v[k] * w[c[k]];
  return s0 + s1;
}

// Runs body(std::integral_constant<int, W>) with the block's width as a compile-time constant.
template <class Body>
inline void with_width(int width, Body&& body) {
  switch (width) {
    case 0: body(std::integral_constant<int, 0>{}); break;
    case 1: body(std::integral_constant<int, 1>{}); break;
    case 2: body(std::integral_constant<int, 2>{}); break;
    case 3: body(std::integral_constant<int, 3>{}); break;
    case 4: body(std::integral_constant<int, 4>{}); break;
    case 5: body(std::integral_constant<int, 5>{}); break;
    case 6: body(std::integral_constant<int, 6>{}); break;
    case 7: body(std::integral_constant<int, 7>{}); break;
    case 8: body(std::integral_constant<int, 8>{}); break;
    default: body(std::integral_constant<int, -1>{}); break;
  }
}
static_assert(kTriangularMaxWidth == 8, "with_width lists the widths 0 to 8");

// Forward sweep, K w = r, optionally with the CG update of r and x first in each row.
template <bool Update>
double forward(const TriangularFactor& f, double a, const double* SSOLVE_RESTRICT p, const double* SSOLVE_RESTRICT q,
               double* SSOLVE_RESTRICT x, double* SSOLVE_RESTRICT r, double* w) {
  const PaddedTriangle& t = f.lower;
  const FactorValue* SSOLVE_RESTRICT const near = f.near.data();
  const FactorValue* SSOLVE_RESTRICT const inv_diag = f.inv_diag.data();
  const Index n = f.n;
  double prev = 0.0;  // w[i - 1], kept in a register
  double rr = 0.0;
  for (Index begin = 0, b = 0; begin < n; begin += kTriangularBlock, ++b) {
    const Index end = std::min<Index>(n, begin + kTriangularBlock);
    const int width = t.width[b];
    const Index* c = t.col.data() + t.block_start[b];
    const FactorValue* v = t.val.data() + t.block_start[b];
    with_width(width, [&](auto tag) {
      constexpr int W = decltype(tag)::value;
      for (Index i = begin; i < end; ++i, c += width, v += width) {
        double ri = r[i];
        if (Update) {
          ri -= a * q[i];
          r[i] = ri;
          x[i] += a * p[i];
          rr += ri * ri;
        }
        const double known = ri - row_sum<W>(c, v, w, width);
        prev = (known - near[i] * prev) * inv_diag[i];
        w[i] = prev;
      }
    });
  }
  return rr;
}

// Backward sweep, Kᵀ w = w in place, returning r·w.
double backward(const TriangularFactor& f, const double* SSOLVE_RESTRICT r, double* w) {
  const PaddedTriangle& t = f.upper;
  const FactorValue* SSOLVE_RESTRICT const near = f.near.data();
  const FactorValue* SSOLVE_RESTRICT const inv_diag = f.inv_diag.data();
  const Index n = f.n;
  const Index blocks = static_cast<Index>(t.width.size());
  double next = 0.0;  // w[i + 1], kept in a register
  double rz = 0.0;
  for (Index b = blocks - 1; b >= 0; --b) {
    const Index begin = b * kTriangularBlock;
    const Index end = std::min<Index>(n, begin + kTriangularBlock);
    const int width = t.width[b];
    const Index* const c0 = t.col.data() + t.block_start[b];
    const FactorValue* const v0 = t.val.data() + t.block_start[b];
    with_width(width, [&](auto tag) {
      constexpr int W = decltype(tag)::value;
      for (Index i = end - 1; i >= begin; --i) {
        const std::int64_t at = static_cast<std::int64_t>(i - begin) * width;
        const double known = w[i] - row_sum<W>(c0 + at, v0 + at, w, width);
        next = (known - near[i + 1] * next) * inv_diag[i];
        w[i] = next;
        rz += r[i] * next;
      }
    });
  }
  return rz;
}

void check_diagonal(const AlignedArray<double>& diagonal, const char* what) {
  for (std::size_t i = 0; i < diagonal.size(); ++i) {
    if (!std::isfinite(diagonal[i]) || diagonal[i] <= 0.0) {
      throw std::runtime_error(std::string(what) + " requires a finite positive diagonal in every row");
    }
  }
}

// z is stored in w_, which has the padding slot w_[n] = 0.
class TriangularPreconditioner final : public Preconditioner {
 public:
  explicit TriangularPreconditioner(TriangularFactor f)
      : f_(std::move(f)), w_(static_cast<std::size_t>(f_.n) + 1) {
    w_.fill(0.0);
  }

  double start(const Backend& backend, const double* r, double, double* p) override {
    const double rz = triangular_apply(f_, r, w_.data());
    backend.copy(w_.data(), p);
    return rz;
  }

  RrRz update(const Backend&, double a, const double* p, const double* q, double* x, double* r) override {
    return triangular_update_apply(f_, a, p, q, x, r, w_.data());
  }

  void direction(const Backend& backend, const double*, double beta, double* p) override {
    backend.xpby(w_.data(), beta, p);
  }

 private:
  TriangularFactor f_;
  AlignedArray<double> w_;
};

}  // namespace

TriangularFactor make_triangular_factor(const CsrMatrix& strict_lower, const AlignedArray<double>& diagonal) {
  const Index n = strict_lower.n_rows;
  if (strict_lower.n_cols != n || diagonal.size() != static_cast<std::size_t>(n)) {
    throw std::runtime_error("triangular factor: sizes do not match");
  }
  const std::size_t un = static_cast<std::size_t>(n);

  TriangularFactor f;
  f.n = n;
  f.near = AlignedArray<FactorValue>(un + 1);
  f.near.fill(0.0);
  f.inv_diag = AlignedArray<FactorValue>(un);
  for (Index i = 0; i < n; ++i) f.inv_diag[i] = static_cast<FactorValue>(1.0 / diagonal[i]);

  // The lower rows without the neighbour, and their transpose, as plain row lists.
  std::vector<std::int64_t> lower_ptr(un + 1, 0), upper_ptr(un + 1, 0);
  for (Index i = 0; i < n; ++i) {
    for (std::int64_t k = strict_lower.row_ptr[i]; k < strict_lower.row_ptr[i + 1]; ++k) {
      const Index j = strict_lower.col[k];
      if (j < 0 || j >= i) throw std::runtime_error("triangular factor: an entry is not below the diagonal");
      if (j == i - 1) {
        f.near[i] = static_cast<FactorValue>(f.near[i] + strict_lower.val[k]);
      } else {
        ++lower_ptr[i + 1];
        ++upper_ptr[j + 1];
      }
    }
  }
  for (std::size_t i = 0; i < un; ++i) {
    lower_ptr[i + 1] += lower_ptr[i];
    upper_ptr[i + 1] += upper_ptr[i];
  }
  std::vector<Index> lower_col(static_cast<std::size_t>(lower_ptr[un])), upper_col(lower_col.size());
  std::vector<double> lower_val(lower_col.size()), upper_val(lower_col.size());
  std::vector<std::int64_t> upper_at(upper_ptr.begin(), upper_ptr.end() - 1);
  for (Index i = 0; i < n; ++i) {
    std::int64_t at = lower_ptr[i];
    for (std::int64_t k = strict_lower.row_ptr[i]; k < strict_lower.row_ptr[i + 1]; ++k) {
      const Index j = strict_lower.col[k];
      if (j == i - 1) continue;
      lower_col[at] = j;
      lower_val[at++] = strict_lower.val[k];
      upper_col[upper_at[j]] = i;
      upper_val[upper_at[j]++] = strict_lower.val[k];
    }
  }
  f.lower = pad_rows(n, lower_ptr, lower_col, lower_val);
  f.upper = pad_rows(n, upper_ptr, upper_col, upper_val);
  return f;
}

// K = (D + L) D^(-1/2): K(i, j) = a(i, j) / sqrt(a(j, j)) below the diagonal, K(i, i) = sqrt(a(i, i)).
TriangularFactor sgs_triangular_factor(const CsrMatrix& a) {
  if (a.n_rows != a.n_cols) throw std::runtime_error("symmetric Gauss-Seidel needs a square matrix");
  const Index n = a.n_rows;
  AlignedArray<double> diagonal(static_cast<std::size_t>(n));
  diagonal.fill(0.0);
  CsrMatrix lower;
  lower.n_rows = lower.n_cols = n;
  lower.row_ptr = AlignedArray<std::int64_t>(static_cast<std::size_t>(n) + 1);
  lower.row_ptr[0] = 0;
  for (Index i = 0; i < n; ++i) {
    std::int64_t count = 0;
    for (std::int64_t k = a.row_ptr[i]; k < a.row_ptr[i + 1]; ++k) {
      if (a.col[k] < i) ++count;
      if (a.col[k] == i) diagonal[i] = a.val[k];
    }
    lower.row_ptr[i + 1] = lower.row_ptr[i] + count;
  }
  check_diagonal(diagonal, "symmetric Gauss-Seidel");
  lower.col = AlignedArray<Index>(static_cast<std::size_t>(lower.nnz()));
  lower.val = AlignedArray<double>(static_cast<std::size_t>(lower.nnz()));
  for (Index i = 0; i < n; ++i) {
    std::int64_t at = lower.row_ptr[i];
    for (std::int64_t k = a.row_ptr[i]; k < a.row_ptr[i + 1]; ++k) {
      const Index j = a.col[k];
      if (j >= i) continue;
      lower.col[at] = j;
      lower.val[at++] = a.val[k] / std::sqrt(diagonal[j]);
    }
  }
  for (Index i = 0; i < n; ++i) diagonal[i] = std::sqrt(diagonal[i]);
  return make_triangular_factor(lower, diagonal);
}

// M⁻¹ = S L⁻ᵀ L⁻¹ S, so K = S⁻¹ L: row i of L divided by S(i).
TriangularFactor ic0_triangular_factor(const Ic0Factor& f) {
  const Index n = f.n;
  CsrMatrix lower;
  lower.n_rows = lower.n_cols = n;
  lower.row_ptr = AlignedArray<std::int64_t>(f.row_ptr.size());
  for (std::size_t i = 0; i < f.row_ptr.size(); ++i) lower.row_ptr[i] = f.row_ptr[i];
  lower.col = AlignedArray<Index>(f.col.size());
  lower.val = AlignedArray<double>(f.val.size());
  AlignedArray<double> diagonal(static_cast<std::size_t>(n));
  for (Index i = 0; i < n; ++i) {
    for (std::int64_t k = f.row_ptr[i]; k < f.row_ptr[i + 1]; ++k) {
      lower.col[k] = f.col[k];
      lower.val[k] = f.val[k] / f.scale[i];
    }
    diagonal[i] = 1.0 / (f.inv_diag[i] * f.scale[i]);
  }
  return make_triangular_factor(lower, diagonal);
}

// The same for Eigen's factor, stored by columns: entry (i, j) of column j goes to row i of K.
TriangularFactor ic_triangular_factor(const CholeskyFactor& f) {
  const Index n = f.n;
  const std::size_t un = static_cast<std::size_t>(n);
  CsrMatrix lower;
  lower.n_rows = lower.n_cols = n;
  lower.row_ptr = AlignedArray<std::int64_t>(un + 1);
  std::vector<std::int64_t> count(un + 1, 0);
  AlignedArray<double> diagonal(un);
  diagonal.fill(0.0);
  for (Index j = 0; j < n; ++j) {
    for (std::int64_t k = f.col_ptr[j]; k < f.col_ptr[j + 1]; ++k) {
      const Index i = f.row[k];
      if (i == j) {
        diagonal[j] = f.val[k] / f.scale[j];
      } else if (i > j) {
        ++count[static_cast<std::size_t>(i) + 1];
      } else {
        throw std::runtime_error("incomplete Cholesky factor has an entry above the diagonal");
      }
    }
  }
  for (std::size_t i = 0; i < un; ++i) count[i + 1] += count[i];
  for (std::size_t i = 0; i <= un; ++i) lower.row_ptr[i] = count[i];
  lower.col = AlignedArray<Index>(static_cast<std::size_t>(count[un]));
  lower.val = AlignedArray<double>(static_cast<std::size_t>(count[un]));
  for (Index j = 0; j < n; ++j) {
    for (std::int64_t k = f.col_ptr[j]; k < f.col_ptr[j + 1]; ++k) {
      const Index i = f.row[k];
      if (i <= j) continue;
      const std::int64_t at = count[static_cast<std::size_t>(i)]++;
      lower.col[at] = j;
      lower.val[at] = f.val[k] / f.scale[i];
    }
  }
  return make_triangular_factor(lower, diagonal);
}

double triangular_apply(const TriangularFactor& f, const double* r, double* w) {
  w[f.n] = 0.0;
  forward<false>(f, 0.0, nullptr, nullptr, nullptr, const_cast<double*>(r), w);
  return backward(f, r, w);
}

RrRz triangular_update_apply(const TriangularFactor& f, double a, const double* p, const double* q, double* x,
                             double* r, double* w) {
  const double rr = forward<true>(f, a, p, q, x, r, w);
  return {rr, backward(f, r, w)};
}

std::unique_ptr<Preconditioner> make_triangular_preconditioner(TriangularFactor f) {
  return std::make_unique<TriangularPreconditioner>(std::move(f));
}

}  // namespace ssolve
