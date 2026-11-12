// Preconditioners copied from established code: symmetric Gauss-Seidel (HPCG) and incomplete Cholesky (Eigen).
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "ssolve/backend.hpp"
#include "ssolve/cg.hpp"
#include "ssolve/generators.hpp"
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

}  // namespace

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
