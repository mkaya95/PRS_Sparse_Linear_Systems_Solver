#include <cmath>
#include <limits>
#include <vector>

#include "ssolve/backend.hpp"
#include "ssolve/cg.hpp"
#include "ssolve/generators.hpp"
#include "ssolve/verify.hpp"
#include "test_util.hpp"

using namespace ssolve;

namespace {

// Sizes around common SIMD widths, so vector loop remainders get exercised.
const std::size_t kSizes[] = {1, 3, 4, 5, 7, 8, 9, 17, 33};

std::vector<double> ramp(std::size_t n, double start, double step) {
  std::vector<double> v(n);
  for (std::size_t i = 0; i < n; ++i) v[i] = start + step * static_cast<double>(i);
  return v;
}

CsrMatrix path_graph_matrix(Index n) {
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

}  // namespace

TEST(spmv_matches_the_reference_for_narrow_and_wide_row_pointers) {
  const CsrMatrix a = make_stencil(Stencil::Point27, 5);
  const std::size_t n = static_cast<std::size_t>(a.n_rows);
  const std::vector<double> x = ramp(n, 1.0, 0.25);
  std::vector<double> expected(n), y(n);
  spmv_reference(a, x.data(), expected.data());
  for (const bool narrow : {true, false}) {
    make_seq_backend(clone(a), narrow)->spmv(x.data(), y.data());
    for (std::size_t i = 0; i < n; ++i) CHECK_NEAR(y[i], expected[i], 1e-12);
  }
}

TEST(spmv_dot_equals_spmv_followed_by_dot) {
  for (const std::size_t n : kSizes) {
    const CsrMatrix a = path_graph_matrix(static_cast<Index>(n));
    const std::vector<double> x = ramp(n, 1.0, 0.5);
    std::vector<double> y_fused(n), y_separate(n);
    const std::unique_ptr<Backend> backend = make_seq_backend(clone(a));
    const double fused = backend->spmv_dot(x.data(), y_fused.data());
    backend->spmv(x.data(), y_separate.data());
    CHECK_NEAR(fused, backend->dot(x.data(), y_separate.data()), 1e-11);
    for (std::size_t i = 0; i < n; ++i) CHECK_NEAR(y_fused[i], y_separate[i], 1e-13);
  }
}

TEST(cg_update_equals_two_axpys_and_a_dot) {
  for (const std::size_t n : kSizes) {
    const std::unique_ptr<Backend> backend = make_seq_backend(path_graph_matrix(static_cast<Index>(n)));
    const double a = 0.375;
    const std::vector<double> p = ramp(n, 1.0, 1.0), q = ramp(n, -2.0, 0.5);
    std::vector<double> x = ramp(n, 3.0, -1.0), r = ramp(n, 5.0, 0.25);
    std::vector<double> expected_x = x, expected_r = r;
    double expected_rr = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      expected_x[i] += a * p[i];
      expected_r[i] -= a * q[i];
      expected_rr += expected_r[i] * expected_r[i];
    }
    const double rr = backend->cg_update(a, p.data(), q.data(), x.data(), r.data());
    CHECK_NEAR(rr, expected_rr, 1e-11);
    for (std::size_t i = 0; i < n; ++i) {
      CHECK_EQ(x[i], expected_x[i]);
      CHECK_EQ(r[i], expected_r[i]);
    }
  }
}

TEST(xpby_matches_hand_computation) {
  const std::unique_ptr<Backend> backend = make_seq_backend(path_graph_matrix(3));
  const double x[] = {1, 2, 3};
  double y[] = {10, 20, 30};
  backend->xpby(x, 0.5, y);
  CHECK_EQ(y[0], 6.0);
  CHECK_EQ(y[1], 12.0);
  CHECK_EQ(y[2], 18.0);
}

TEST(copy_and_set_zero_touch_every_element) {
  for (const std::size_t n : kSizes) {
    const std::unique_ptr<Backend> backend = make_seq_backend(path_graph_matrix(static_cast<Index>(n)));
    const std::vector<double> source = ramp(n, 1.0, 1.0);
    std::vector<double> destination(n, 0.0);
    backend->copy(source.data(), destination.data());
    CHECK(destination == source);
    backend->set_zero(destination.data());
    for (const double value : destination) CHECK_EQ(value, 0.0);
  }
}

TEST(jacobi_update_equals_cg_update_plus_r_dot_z) {
  for (const std::size_t n : kSizes) {
    const std::unique_ptr<Backend> backend = make_seq_backend(path_graph_matrix(static_cast<Index>(n)));
    const double a = -0.625;
    const std::vector<double> p = ramp(n, 1.0, 1.0), q = ramp(n, -2.0, 0.5), inv_diag = ramp(n, 0.5, 0.125);
    std::vector<double> x = ramp(n, 3.0, -1.0), r = ramp(n, 5.0, 0.25);
    std::vector<double> plain_x = x, plain_r = r;
    const double plain_rr = backend->cg_update(a, p.data(), q.data(), plain_x.data(), plain_r.data());
    const RrRz sums = backend->jacobi_update(a, p.data(), q.data(), x.data(), r.data(), inv_diag.data());
    double expected_rz = 0.0;
    for (std::size_t i = 0; i < n; ++i) expected_rz += plain_r[i] * inv_diag[i] * plain_r[i];
    CHECK_NEAR(sums.rr, plain_rr, 1e-11);
    CHECK_NEAR(sums.rz, expected_rz, 1e-11);
    for (std::size_t i = 0; i < n; ++i) {
      CHECK_EQ(x[i], plain_x[i]);
      CHECK_EQ(r[i], plain_r[i]);
    }
  }
}

TEST(jacobi_direction_matches_hand_computation) {
  const std::unique_ptr<Backend> backend = make_seq_backend(path_graph_matrix(3));
  const double inv_diag[] = {0.5, 0.25, 2.0};
  const double r[] = {2, 4, 1};
  double p[] = {10, 20, 30};
  backend->jacobi_direction(inv_diag, r, 0.5, p);
  CHECK_EQ(p[0], 6.0);
  CHECK_EQ(p[1], 11.0);
  CHECK_EQ(p[2], 17.0);
}

TEST(jacobi_rejects_invalid_diagonals) {
  Coo coo;
  coo.n_rows = 4;
  coo.n_cols = 4;
  coo.add(0, 0, 4.0);
  coo.add(1, 1, 0.0);   // stored zero
  coo.add(2, 0, -1.0);  // no diagonal entry in this row
  coo.add(3, 3, -8.0);  // a negative diagonal keeps its sign
  CHECK_THROWS_WITH(jacobi_inverse_diagonal(csr_from_coo(coo)), "positive");
  for (const double value : {0.0, -1.0, std::numeric_limits<double>::infinity()}) {
    Coo diagonal;
    diagonal.n_rows = diagonal.n_cols = 1;
    diagonal.add(0, 0, value);
    CHECK_THROWS_WITH(jacobi_inverse_diagonal(csr_from_coo(diagonal)), "positive");
  }
}

TEST(jacobi_rejects_a_system_with_a_zero_diagonal_entry) {
  Coo coo;
  coo.n_rows = 2;
  coo.n_cols = 2;
  coo.add(0, 0, 2.0);
  coo.add(0, 1, 1.0);
  coo.add(1, 0, 1.0);
  coo.add(1, 1, 0.0);
  const CsrMatrix a = csr_from_coo(coo);
  CHECK_THROWS_WITH(jacobi_inverse_diagonal(a), "positive");
}

TEST(empty_matrix_backend_does_nothing) {
  Coo coo;
  const std::unique_ptr<Backend> backend = make_seq_backend(csr_from_coo(coo));
  CHECK_EQ(backend->n(), 0);
  CHECK_EQ(backend->dot(nullptr, nullptr), 0.0);
  CHECK_EQ(backend->spmv_dot(nullptr, nullptr), 0.0);
}

TEST(backend_rejects_a_non_square_matrix) {
  Coo coo;
  coo.n_rows = 1;
  coo.n_cols = 2;
  CHECK_THROWS_WITH(make_seq_backend(csr_from_coo(coo)), "square");
}

TEST(cg_on_the_backend_solves_a_poisson_system) {
  const CsrMatrix a = make_stencil(Stencil::Poisson3D, 12);
  const AlignedArray<double> b = rhs_for_unit_solution(a);
  AlignedArray<double> x(static_cast<std::size_t>(a.n_rows));
  const std::unique_ptr<Backend> backend = make_seq_backend(clone(a));
  const SolveResult result = cg(*backend, nullptr, b.data(), x.data(), SolveOptions{});
  CHECK(result.status == SolveStatus::Converged);
  CHECK(true_relative_residual(a, b.data(), x.data()) < 1e-7);
}

TEST_MAIN()
