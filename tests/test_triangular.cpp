// The branch-free sweeps of triangular.hpp against the reference apply of each preconditioner.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

#include "ssolve/generators.hpp"
#include "ssolve/ic0.hpp"
#include "ssolve/incomplete_cholesky.hpp"
#include "ssolve/reorder.hpp"
#include "ssolve/sgs.hpp"
#include "ssolve/triangular.hpp"
#include "test_util.hpp"

using namespace ssolve;

namespace {

std::vector<double> wave(std::size_t n, double phase) {
  std::vector<double> v(n);
  for (std::size_t i = 0; i < n; ++i) v[i] = std::sin(0.7 * static_cast<double>(i) + phase) + 0.25;
  return v;
}

double max_abs(const std::vector<double>& v) {
  double m = 0.0;
  for (const double x : v) m = std::max(m, std::fabs(x));
  return m;
}

// Matrices that reach every path: narrow rows, rows wider than kTriangularMaxWidth (point27 has 13 entries
// left of the diagonal), rows with and without the neighbour entry (a random order), and a partial last block.
std::vector<CsrMatrix> matrices() {
  std::vector<CsrMatrix> all;
  all.push_back(make_stencil(Stencil::Poisson2D, 13));
  all.push_back(make_stencil(Stencil::Poisson3D, 7));
  all.push_back(make_stencil(Stencil::Point27, 6));
  const CsrMatrix p = make_stencil(Stencil::Poisson3D, 6);
  all.push_back(permute_symmetric(p, reverse_cuthill_mckee(p)));
  std::vector<Index> shuffled(static_cast<std::size_t>(p.n_rows));
  std::iota(shuffled.begin(), shuffled.end(), Index{0});
  std::shuffle(shuffled.begin(), shuffled.end(), std::mt19937(7));
  all.push_back(permute_symmetric(p, shuffled));
  return all;
}

// Checks triangular_apply against `reference` (z = M⁻¹ r), and r·z.
template <class Reference>
void check_apply(const TriangularFactor& f, Reference reference) {
  const std::size_t n = static_cast<std::size_t>(f.n);
  const std::vector<double> r = wave(n, 0.3);
  std::vector<double> expected(n), w(n + 1, 0.0);
  reference(r.data(), expected.data());
  const double rz = triangular_apply(f, r.data(), w.data());
  const double scale = max_abs(expected);
  double expected_rz = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    CHECK_NEAR(w[i], expected[i], 1e-12 * scale);
    expected_rz += r[i] * expected[i];
  }
  CHECK_NEAR(rz, expected_rz, 1e-12 * std::fabs(expected_rz));
  CHECK_EQ(w[n], 0.0);
}

}  // namespace

TEST(sgs_matches_hpcg_sweeps) {
  for (const CsrMatrix& a : matrices()) {
    const SymGaussSeidel m(clone(a));
    check_apply(sgs_triangular_factor(a), [&](const double* r, double* z) { m.apply(r, z); });
  }
}

TEST(ic0_matches_ic0_apply) {
  for (const CsrMatrix& a : matrices()) {
    const Ic0Factor f = ic0(a);
    check_apply(ic0_triangular_factor(f), [&](const double* r, double* z) { ic0_apply(f, r, z); });
  }
}

TEST(ic_matches_eigen_style_apply) {
  for (const CsrMatrix& a : matrices()) {
    for (const EigenBugs bugs : {EigenBugs::Copied, EigenBugs::Fixed}) {
      const CholeskyFactor f = incomplete_cholesky(a, 1e-3, bugs);
      check_apply(ic_triangular_factor(f), [&](const double* r, double* z) { incomplete_cholesky_apply(f, r, z); });
    }
  }
}

TEST(update_apply_is_the_update_then_the_apply) {
  for (const CsrMatrix& a : matrices()) {
    const TriangularFactor f = ic0_triangular_factor(ic0(a));
    const std::size_t n = static_cast<std::size_t>(a.n_rows);
    const double step = 0.37;
    const std::vector<double> p = wave(n, 1.1), q = wave(n, 2.3);
    std::vector<double> x = wave(n, 0.5), r = wave(n, 0.9), w(n + 1, 0.0);
    std::vector<double> x2 = x, r2 = r, w2(n + 1, 0.0);

    const RrRz sums = triangular_update_apply(f, step, p.data(), q.data(), x.data(), r.data(), w.data());

    double rr = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      x2[i] += step * p[i];
      r2[i] -= step * q[i];
      rr += r2[i] * r2[i];
    }
    const double rz = triangular_apply(f, r2.data(), w2.data());
    for (std::size_t i = 0; i < n; ++i) {
      CHECK_EQ(x[i], x2[i]);
      CHECK_EQ(r[i], r2[i]);
      CHECK_NEAR(w[i], w2[i], 1e-14 * max_abs(w2));
    }
    CHECK_NEAR(sums.rr, rr, 1e-13 * rr);
    CHECK_NEAR(sums.rz, rz, 1e-13 * std::fabs(rz));
  }
}

TEST(padding_points_at_the_zero_slot) {
  const CsrMatrix a = make_stencil(Stencil::Poisson2D, 9);
  const TriangularFactor f = sgs_triangular_factor(a);
  for (const PaddedTriangle* t : {&f.lower, &f.upper}) {
    for (std::size_t k = 0; k < t->col.size(); ++k) {
      if (t->val[k] == 0.0) CHECK_EQ(t->col[k], a.n_rows);
      CHECK(t->col[k] >= 0 && t->col[k] <= a.n_rows);
    }
  }
}

TEST(factor_rejects_bad_input) {
  CsrMatrix lower = make_stencil(Stencil::Poisson2D, 3);  // has entries above the diagonal
  AlignedArray<double> diagonal(static_cast<std::size_t>(lower.n_rows));
  diagonal.fill(1.0);
  CHECK_THROWS_WITH(make_triangular_factor(lower, diagonal), "not below the diagonal");
  AlignedArray<double> short_diagonal(2);
  short_diagonal.fill(1.0);
  CHECK_THROWS_WITH(make_triangular_factor(lower, short_diagonal), "sizes");
  CsrMatrix zero_diagonal = make_stencil(Stencil::Poisson2D, 2);
  for (std::int64_t k = zero_diagonal.row_ptr[1]; k < zero_diagonal.row_ptr[2]; ++k) {
    if (zero_diagonal.col[k] == 1) zero_diagonal.val[k] = 0.0;
  }
  CHECK_THROWS_WITH(sgs_triangular_factor(zero_diagonal), "positive diagonal");
}

TEST_MAIN()
