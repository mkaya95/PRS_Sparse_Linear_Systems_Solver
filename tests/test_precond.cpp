// Preconditioners copied from established code: symmetric Gauss-Seidel (HPCG) and incomplete Cholesky (Eigen).
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "ssolve/backend.hpp"
#include "ssolve/cg.hpp"
#include "ssolve/generators.hpp"
#include "ssolve/ic0.hpp"
#include "ssolve/incomplete_cholesky.hpp"
#include "ssolve/preconditioner.hpp"
#include "ssolve/sgs.hpp"
#include "ssolve/verify.hpp"
#include "test_util.hpp"

using namespace ssolve;

namespace {

// A copy of path_graph_matrix from test_backend.cpp.
CsrMatrix path(Index n) {
  Coo coo;
  coo.n_rows = n;
  coo.n_cols = n;
  for (Index i = 0; i < n; ++i) {
    if (i > 0) coo.add(i, i - 1, -1.0);
    coo.add(i, i, 2.5);
    if (i + 1 < n) coo.add(i, i + 1, -1.0);
  }
  return csr_from_coo(coo);
}

std::vector<double> dense(const CsrMatrix& a) {
  const std::size_t n = static_cast<std::size_t>(a.n_rows);
  std::vector<double> d(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::int64_t k = a.row_ptr[i]; k < a.row_ptr[i + 1]; ++k) d[i * n + static_cast<std::size_t>(a.col[k])] = a.val[k];
  }
  return d;
}

std::vector<double> wave(std::size_t n, double phase) {
  std::vector<double> v(n);
  for (std::size_t i = 0; i < n; ++i) v[i] = std::sin(0.7 * static_cast<double>(i) + phase) + 0.25;
  return v;
}

double dot(const std::vector<double>& x, const std::vector<double>& y) {
  double sum = 0.0;
  for (std::size_t i = 0; i < x.size(); ++i) sum += x[i] * y[i];
  return sum;
}

// z = (D + U)⁻¹ D (D + L)⁻¹ r with dense triangular solves.
std::vector<double> dense_sgs(const CsrMatrix& a, const std::vector<double>& r) {
  const std::size_t n = r.size();
  const std::vector<double> m = dense(a);
  std::vector<double> y(n), z(n);
  for (std::size_t i = 0; i < n; ++i) {
    double sum = r[i];
    for (std::size_t j = 0; j < i; ++j) sum -= m[i * n + j] * y[j];
    y[i] = sum / m[i * n + i];
  }
  for (std::size_t i = n; i-- > 0;) {
    double sum = m[i * n + i] * y[i];
    for (std::size_t j = i + 1; j < n; ++j) sum -= m[i * n + j] * z[j];
    z[i] = sum / m[i * n + i];
  }
  return z;
}

CsrMatrix diagonal(const std::vector<double>& values) {
  Coo coo;
  coo.n_rows = static_cast<Index>(values.size());
  coo.n_cols = coo.n_rows;
  for (Index i = 0; i < coo.n_rows; ++i) coo.add(i, i, values[static_cast<std::size_t>(i)]);
  return csr_from_coo(coo);
}

CsrMatrix two_by_two(double diagonal_value, double off) {
  Coo coo;
  coo.n_rows = 2;
  coo.n_cols = 2;
  coo.add(0, 0, diagonal_value);
  coo.add(0, 1, off);
  coo.add(1, 0, off);
  coo.add(1, 1, diagonal_value);
  return csr_from_coo(coo);
}

}  // namespace

TEST(ic_is_exact_without_fill) {
  // A tridiagonal matrix has no fill, so the incomplete factor is the complete one and M = A.
  const CsrMatrix a = path(50);
  const std::size_t n = 50;
  const CholeskyFactor f = incomplete_cholesky(a);
  CHECK_EQ(f.shift, 0.0);
  const std::vector<double> r = wave(n, 0.5);
  std::vector<double> z(n), az(n);
  incomplete_cholesky_apply(f, r.data(), z.data());
  spmv_reference(a, z.data(), az.data());
  for (std::size_t i = 0; i < n; ++i) CHECK_NEAR(az[i], r[i], 1e-12);
}

TEST(ic_apply_is_symmetric) {
  const CsrMatrix a = make_stencil(Stencil::Point27, 4);
  const std::size_t n = static_cast<std::size_t>(a.n_rows);
  const CholeskyFactor f = incomplete_cholesky(a);
  const std::vector<double> u = wave(n, 0.1), v = wave(n, 1.9);
  std::vector<double> mu(n), mv(n);
  incomplete_cholesky_apply(f, u.data(), mu.data());
  incomplete_cholesky_apply(f, v.data(), mv.data());
  CHECK_NEAR(dot(u, mv), dot(v, mu), 1e-12 * std::fabs(dot(u, mv)));
}

TEST(ic_starts_with_a_shift_for_a_nonpositive_scaled_diagonal) {
  const CholeskyFactor f = incomplete_cholesky(diagonal({1, -1}));
  CHECK_NEAR(f.shift, 1e-3 + 1.0, 1e-15);
}

TEST(ic_doubles_the_shift_until_the_factorization_works) {
  // Scaled, the pivots fail until the shift exceeds (c - 1) / sqrt(1 + c^2) = 0.007: 1e-3, 2e-3, 4e-3 fail, 8e-3 works.
  const CholeskyFactor f = incomplete_cholesky(two_by_two(1.0, 1.01));
  CHECK_EQ(f.shift, 1e-3 * 8);
}

TEST(ic_gives_up_after_ten_shifts) {
  CHECK_THROWS_WITH(incomplete_cholesky(two_by_two(1.0, 2.0)), "failed after 10 shifts");
}

TEST(ic_keeps_the_original_count_per_column) {
  const CsrMatrix a = make_stencil(Stencil::Poisson2D, 6);
  const CholeskyFactor f = incomplete_cholesky(a);
  CHECK_EQ(f.n, a.n_rows);
  for (Index j = 0; j < a.n_rows; ++j) {
    std::int64_t lower = 0;
    for (std::int64_t k = a.row_ptr[j]; k < a.row_ptr[j + 1]; ++k) lower += a.col[k] >= j ? 1 : 0;
    CHECK_EQ(f.col_ptr[j + 1] - f.col_ptr[j], lower);
    CHECK_EQ(f.row[f.col_ptr[j]], j);  // the diagonal comes first
  }
}

TEST(ic_cg_needs_fewer_iterations_than_jacobi) {
  const CsrMatrix a = make_stencil(Stencil::Poisson3D, 12);
  const std::size_t n = static_cast<std::size_t>(a.n_rows);
  const AlignedArray<double> b = rhs_for_unit_solution(a);
  const std::unique_ptr<Backend> backend = make_seq_backend(clone(a));
  AlignedArray<double> x_ic(n), x_jacobi(n);
  const std::unique_ptr<Preconditioner> ic = make_preconditioner(Precond::IncompleteCholesky, a);
  const std::unique_ptr<Preconditioner> jacobi = make_jacobi_preconditioner(a);
  const SolveResult with_ic = verified_cg(a, *backend, *ic, b.data(), x_ic.data(), {});
  const SolveResult with_jacobi = verified_cg(a, *backend, *jacobi, b.data(), x_jacobi.data(), {});
  CHECK(with_ic.status == SolveStatus::Converged);
  CHECK(with_ic.spmv_count < with_jacobi.spmv_count);
  for (std::size_t i = 0; i < n; ++i) CHECK_NEAR(x_ic[i], 1.0, 1e-6);
}


TEST(sgs_matches_a_dense_computation) {
  for (const CsrMatrix& a : {make_stencil(Stencil::Poisson2D, 4), path(7)}) {
    const std::size_t n = static_cast<std::size_t>(a.n_rows);
    const std::vector<double> r = wave(n, 0.3);
    const std::vector<double> expected = dense_sgs(a, r);
    std::vector<double> z(n, 99.0);
    SymGaussSeidel(clone(a)).apply(r.data(), z.data());
    for (std::size_t i = 0; i < n; ++i) CHECK_NEAR(z[i], expected[i], 1e-13);
  }
}

TEST(sgs_is_symmetric) {
  const CsrMatrix a = make_stencil(Stencil::Point27, 4);
  const std::size_t n = static_cast<std::size_t>(a.n_rows);
  const SymGaussSeidel m(clone(a));
  const std::vector<double> u = wave(n, 0.1), v = wave(n, 1.9);
  std::vector<double> mu(n), mv(n);
  m.apply(u.data(), mu.data());
  m.apply(v.data(), mv.data());
  CHECK_NEAR(dot(u, mv), dot(v, mu), 1e-12 * std::fabs(dot(u, mv)));
}

TEST(sgs_is_exact_on_a_diagonal_matrix) {
  const std::vector<double> d = {2, 4, 8};
  const std::vector<double> r = {1, 3, 5};
  std::vector<double> z(3);
  SymGaussSeidel(diagonal(d)).apply(r.data(), z.data());
  for (std::size_t i = 0; i < 3; ++i) CHECK_EQ(z[i], r[i] / d[i]);
}

TEST(sgs_needs_a_positive_diagonal) {
  CHECK_THROWS_WITH(SymGaussSeidel(diagonal({1, 0, 2})), "positive diagonal");
  Coo coo;
  coo.n_rows = 2;
  coo.n_cols = 2;
  coo.add(0, 0, 1.0);
  coo.add(1, 0, 1.0);  // row 1 has no diagonal entry
  CHECK_THROWS_WITH(SymGaussSeidel(csr_from_coo(coo)), "positive diagonal");
}

TEST(sgs_cg_needs_fewer_iterations_than_jacobi) {
  const CsrMatrix a = make_stencil(Stencil::Poisson3D, 12);
  const std::size_t n = static_cast<std::size_t>(a.n_rows);
  const AlignedArray<double> b = rhs_for_unit_solution(a);
  const std::unique_ptr<Backend> backend = make_seq_backend(clone(a));
  AlignedArray<double> x_sgs(n), x_jacobi(n);
  const std::unique_ptr<Preconditioner> sgs = make_sgs_preconditioner(a);
  const std::unique_ptr<Preconditioner> jacobi = make_jacobi_preconditioner(a);
  const SolveResult with_sgs = verified_cg(a, *backend, *sgs, b.data(), x_sgs.data(), {});
  const SolveResult with_jacobi = verified_cg(a, *backend, *jacobi, b.data(), x_jacobi.data(), {});
  CHECK(with_sgs.status == SolveStatus::Converged);
  CHECK(with_sgs.spmv_count < with_jacobi.spmv_count);
  for (std::size_t i = 0; i < n; ++i) CHECK_NEAR(x_sgs[i], 1.0, 1e-6);
}

TEST(the_factory_builds_sgs) {
  CHECK(make_preconditioner(Precond::SymGaussSeidel, path(5)) != nullptr);
}

TEST_MAIN()

namespace {

// The matrix of Kershaw (1978): SPD, with eigenvalues of about 0.17 and 5.83, yet IC(0) without a
// shift reaches the pivot -5.
CsrMatrix kershaw() {
  const double rows[4][4] = {{3, -2, 0, 2}, {-2, 3, -2, 0}, {0, -2, 3, -2}, {2, 0, -2, 3}};
  Coo coo;
  coo.n_rows = 4;
  coo.n_cols = 4;
  for (Index i = 0; i < 4; ++i) {
    for (Index j = 0; j < 4; ++j) {
      if (rows[i][j] != 0.0) coo.add(i, j, rows[i][j]);
    }
  }
  return csr_from_coo(coo);
}

}  // namespace

TEST(ic_fixed_equals_the_copy_where_nothing_is_dropped) {
  // Both bugs act only on entries outside the pattern, and a tridiagonal matrix has none.
  const CsrMatrix a = path(40);
  const CholeskyFactor copied = incomplete_cholesky(a);
  const CholeskyFactor fixed = incomplete_cholesky(a, 1e-3, EigenBugs::Fixed);
  for (std::size_t k = 0; k < copied.val.size(); ++k) CHECK_EQ(fixed.val[k], copied.val[k]);
}

TEST(ic_fixed_needs_fewer_iterations_than_the_copy) {
  const CsrMatrix a = make_stencil(Stencil::Point27, 12);
  const std::size_t n = static_cast<std::size_t>(a.n_rows);
  const AlignedArray<double> b = rhs_for_unit_solution(a);
  const std::unique_ptr<Backend> backend = make_seq_backend(clone(a));
  AlignedArray<double> x(n);
  const std::unique_ptr<Preconditioner> copied = make_preconditioner(Precond::IncompleteCholesky, a);
  const std::unique_ptr<Preconditioner> fixed = make_preconditioner(Precond::IncompleteCholeskyFixed, a);
  const SolveResult with_copied = verified_cg(a, *backend, *copied, b.data(), x.data(), {});
  const SolveResult with_fixed = verified_cg(a, *backend, *fixed, b.data(), x.data(), {});
  CHECK(with_fixed.status == SolveStatus::Converged);
  CHECK(with_fixed.spmv_count < with_copied.spmv_count);
  for (std::size_t i = 0; i < n; ++i) CHECK_NEAR(x[i], 1.0, 1e-6);
}

TEST(ic0_is_exact_without_fill) {
  // A tridiagonal matrix has no fill to drop, so M = A and CG is done after one SpMV.
  const CsrMatrix a = path(50);
  const std::size_t n = 50;
  const Ic0Factor f = ic0(a);
  CHECK_EQ(f.shift, 0.0);
  CHECK_EQ(f.attempts, 1);
  const std::vector<double> r = wave(n, 0.5);
  std::vector<double> z(n), az(n);
  ic0_apply(f, r.data(), z.data());
  spmv_reference(a, z.data(), az.data());
  for (std::size_t i = 0; i < n; ++i) CHECK_NEAR(az[i], r[i], 1e-12);

  const AlignedArray<double> b = rhs_for_unit_solution(a);
  const std::unique_ptr<Backend> backend = make_seq_backend(clone(a));
  const std::unique_ptr<Preconditioner> m = make_preconditioner(Precond::IncompleteCholeskyZero, a);
  AlignedArray<double> x(n);
  const SolveResult result = verified_cg(a, *backend, *m, b.data(), x.data(), {});
  CHECK(result.status == SolveStatus::Converged);
  // The solver's preconditioner stores the factor in single precision (triangular.hpp), so there M = A
  // only to about 1e-7, and the second SpMV removes what is left. ic0_apply above is exact.
  CHECK(result.spmv_count <= 2);
}

TEST(ic0_keeps_the_pattern_of_the_lower_triangle) {
  const CsrMatrix a = make_stencil(Stencil::Poisson2D, 6);
  const Ic0Factor f = ic0(a);
  CHECK_EQ(f.n, a.n_rows);
  for (Index i = 0; i < a.n_rows; ++i) {
    std::int64_t at = f.row_ptr[i];
    for (std::int64_t k = a.row_ptr[i]; k < a.row_ptr[i + 1]; ++k) {
      if (a.col[k] < i) CHECK_EQ(f.col[at++], a.col[k]);
    }
    CHECK_EQ(at, f.row_ptr[i + 1]);
  }
}

TEST(ic0_reproduces_the_scaled_matrix_on_its_pattern) {
  // The defining property of IC(0): L L^T equals S A S wherever A has an entry.
  const CsrMatrix a = make_stencil(Stencil::Poisson2D, 5);
  const std::size_t n = static_cast<std::size_t>(a.n_rows);
  const Ic0Factor f = ic0(a);
  std::vector<double> l(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::int64_t k = f.row_ptr[i]; k < f.row_ptr[i + 1]; ++k) l[i * n + static_cast<std::size_t>(f.col[k])] = f.val[k];
    l[i * n + i] = 1.0 / f.inv_diag[i];
  }
  for (std::size_t i = 0; i < n; ++i) {
    for (std::int64_t k = a.row_ptr[i]; k < a.row_ptr[i + 1]; ++k) {
      const std::size_t j = static_cast<std::size_t>(a.col[k]);
      double product = 0.0;
      for (std::size_t m = 0; m < n; ++m) product += l[i * n + m] * l[j * n + m];
      CHECK_NEAR(product, a.val[k] * f.scale[i] * f.scale[j], 1e-13);
    }
  }
}

TEST(ic0_apply_is_symmetric) {
  const CsrMatrix a = make_stencil(Stencil::Point27, 4);
  const std::size_t n = static_cast<std::size_t>(a.n_rows);
  const Ic0Factor f = ic0(a);
  const std::vector<double> u = wave(n, 0.1), v = wave(n, 1.9);
  std::vector<double> mu(n), mv(n);
  ic0_apply(f, u.data(), mu.data());
  ic0_apply(f, v.data(), mv.data());
  CHECK_NEAR(dot(u, mv), dot(v, mu), 1e-12 * std::fabs(dot(u, mv)));
}

TEST(ic0_shifts_on_kershaws_matrix_and_still_solves) {
  const CsrMatrix a = kershaw();
  const Ic0Factor f = ic0(a);
  CHECK(f.shift > 0.0);
  CHECK(f.attempts > 1);
  const AlignedArray<double> b = rhs_for_unit_solution(a);
  const std::unique_ptr<Backend> backend = make_seq_backend(clone(a));
  const std::unique_ptr<Preconditioner> m = make_preconditioner(Precond::IncompleteCholeskyZero, a);
  AlignedArray<double> x(4);
  const SolveResult result = verified_cg(a, *backend, *m, b.data(), x.data(), {});
  CHECK(result.status == SolveStatus::Converged);
  for (std::size_t i = 0; i < 4; ++i) CHECK_NEAR(x[i], 1.0, 1e-7);
}

TEST(ic0_needs_a_positive_diagonal) {
  CHECK_THROWS_WITH(ic0(diagonal({1, -1})), "positive diagonal");
  CHECK_THROWS_WITH(ic0(two_by_two(0.0, 1.0)), "positive diagonal");
}

TEST(ic0_cg_needs_fewer_iterations_than_jacobi) {
  const CsrMatrix a = make_stencil(Stencil::Poisson3D, 12);
  const std::size_t n = static_cast<std::size_t>(a.n_rows);
  const AlignedArray<double> b = rhs_for_unit_solution(a);
  const std::unique_ptr<Backend> backend = make_seq_backend(clone(a));
  AlignedArray<double> x_ic(n), x_jacobi(n);
  const std::unique_ptr<Preconditioner> ic = make_preconditioner(Precond::IncompleteCholeskyZero, a);
  const std::unique_ptr<Preconditioner> jacobi = make_jacobi_preconditioner(a);
  const SolveResult with_ic = verified_cg(a, *backend, *ic, b.data(), x_ic.data(), {});
  const SolveResult with_jacobi = verified_cg(a, *backend, *jacobi, b.data(), x_jacobi.data(), {});
  CHECK(with_ic.status == SolveStatus::Converged);
  CHECK(with_ic.spmv_count < with_jacobi.spmv_count);
  for (std::size_t i = 0; i < n; ++i) CHECK_NEAR(x_ic[i], 1.0, 1e-6);
}
