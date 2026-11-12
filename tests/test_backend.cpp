#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

#include "ssolve/backend.hpp"
#include "ssolve/cg.hpp"
#include "ssolve/generators.hpp"
#include "ssolve/preconditioner.hpp"
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

// Jacobi through the general path: z is stored and read back, as any preconditioner that cannot
// merge its work into the vector updates does it.
class StoredZJacobi final : public Preconditioner {
 public:
  explicit StoredZJacobi(const CsrMatrix& a) : d_(jacobi_inverse_diagonal(a)), z_(d_.size()) {}

  double start(const Backend& backend, const double* r, double, double* p) override {
    apply(r);
    backend.copy(z_.data(), p);
    return backend.dot(r, z_.data());
  }

  RrRz update(const Backend& backend, double a, const double* p, const double* q, double* x, double* r) override {
    const double rr = backend.cg_update(a, p, q, x, r);
    apply(r);
    return {rr, backend.dot(r, z_.data())};
  }

  void direction(const Backend& backend, const double*, double beta, double* p) override {
    backend.xpby(z_.data(), beta, p);
  }

 private:
  void apply(const double* r) {
    for (std::size_t i = 0; i < d_.size(); ++i) z_[i] = d_[i] * r[i];
  }

  AlignedArray<double> d_, z_;
};

// A path graph whose diagonal varies from row to row, so Jacobi really changes the iteration.
CsrMatrix scaled_path_matrix(Index n) {
  Coo coo;
  coo.n_rows = n;
  coo.n_cols = n;
  for (Index i = 0; i < n; ++i) {
    if (i > 0) coo.add(i, i - 1, -1.0);
    coo.add(i, i, 2.5 * (1 + i % 7));
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

TEST(cg_with_history_records_one_residual_per_iteration_and_solves_identically) {
  const CsrMatrix a = make_stencil(Stencil::Poisson2D, 20);
  const AlignedArray<double> b = rhs_for_random_solution(a);
  const AlignedArray<double> inv_diag = jacobi_inverse_diagonal(a);
  const std::unique_ptr<Backend> backend = make_seq_backend(clone(a));
  const std::size_t n = static_cast<std::size_t>(a.n_rows);
  for (const double* diag : {static_cast<const double*>(nullptr), inv_diag.data()}) {
    std::vector<double> plain_x(n), recorded_x(n), history = {42.0};
    const SolveResult plain = cg(*backend, diag, b.data(), plain_x.data(), SolveOptions{});
    const SolveResult recorded = cg_with_history(*backend, diag, b.data(), recorded_x.data(), SolveOptions{}, history);
    CHECK(recorded.status == SolveStatus::Converged);
    CHECK_EQ(recorded.spmv_count, plain.spmv_count);
    CHECK_EQ(static_cast<std::int64_t>(history.size()), recorded.spmv_count + 1);
    CHECK_EQ(history.front(), 1.0);  // x starts at 0, so r = b
    CHECK_NEAR(history.back() / recorded.recursive_residual, 1.0, 1e-12);
    CHECK(history.back() < 1e-8);
    CHECK(history[1] < 1.0);
    CHECK(plain_x == recorded_x);
  }
}

TEST(cg_with_history_normalizes_a_rescaled_rhs) {
  // A tiny b makes cg() solve with b / max|bᵢ|, which must not shift the recorded residuals.
  const CsrMatrix a = make_stencil(Stencil::Poisson2D, 10);
  AlignedArray<double> b = rhs_for_random_solution(a);
  for (double& value : b) value *= 1e-150;
  const std::unique_ptr<Backend> backend = make_seq_backend(clone(a));
  std::vector<double> x(b.size()), history;
  const SolveResult result = cg_with_history(*backend, nullptr, b.data(), x.data(), SolveOptions{}, history);
  CHECK(result.status == SolveStatus::Converged);
  CHECK_EQ(history.front(), 1.0);
  CHECK_NEAR(history.back() / result.recursive_residual, 1.0, 1e-12);
}

TEST(cg_with_history_refuses_an_initial_guess) {
  const CsrMatrix a = make_stencil(Stencil::Poisson2D, 4);
  const std::unique_ptr<Backend> backend = make_seq_backend(clone(a));
  const AlignedArray<double> b = rhs_for_unit_solution(a);
  std::vector<double> x(b.size()), history;
  SolveOptions options;
  options.use_initial_guess = true;
  CHECK_THROWS_WITH(cg_with_history(*backend, nullptr, b.data(), x.data(), options, history), "x = 0");
}

TEST(cg_with_history_is_empty_for_a_zero_rhs) {
  const CsrMatrix a = make_stencil(Stencil::Poisson2D, 4);
  const std::unique_ptr<Backend> backend = make_seq_backend(clone(a));
  const std::vector<double> b(static_cast<std::size_t>(a.n_rows), 0.0);
  std::vector<double> x(b.size()), history = {1.0};
  const SolveResult result = cg_with_history(*backend, nullptr, b.data(), x.data(), SolveOptions{}, history);
  CHECK(result.status == SolveStatus::Converged);
  CHECK(history.empty());
}

TEST(verification_uses_the_given_matrix_so_a_wrong_kernel_cannot_hide) {
  // The backend holds a different matrix, which is how a buggy kernel would behave.
  Coo truth, wrong;
  truth.n_rows = truth.n_cols = wrong.n_rows = wrong.n_cols = 2;
  truth.add(0, 0, 1.0);
  truth.add(1, 1, 2.0);
  wrong.add(0, 0, 1.0);
  wrong.add(1, 1, 2.5);
  const CsrMatrix a = csr_from_coo(truth);
  const std::unique_ptr<Backend> backend = make_seq_backend(csr_from_coo(wrong));
  const double b[] = {1.0, 1.0};
  double x[] = {0.0, 0.0};
  const SolveResult result = verified_cg(a, *backend, nullptr, b, x, SolveOptions{});
  CHECK(result.status == SolveStatus::ResidualFailure);
  // x = (1, 0.4), so b - A x = (0, 0.2), and the relative residual is 0.2 / sqrt(2).
  CHECK_NEAR(result.true_residual, 0.2 / std::sqrt(2.0), 1e-6);
}

TEST(verified_cg_accepts_a_correct_answer) {
  const CsrMatrix a = make_stencil(Stencil::Poisson3D, 8);
  const AlignedArray<double> b = rhs_for_unit_solution(a);
  AlignedArray<double> x(static_cast<std::size_t>(a.n_rows));
  const std::unique_ptr<Backend> backend = make_seq_backend(clone(a));
  const SolveResult result = verified_cg(a, *backend, nullptr, b.data(), x.data(), SolveOptions{});
  CHECK(result.status == SolveStatus::Converged);
  CHECK(result.true_residual <= 1e-8);
}

TEST(sell_spmv_matches_the_reference_for_every_chunk_and_sigma) {
  std::vector<CsrMatrix> matrices;
  matrices.push_back(make_stencil(Stencil::Point27, 5));
  for (const std::size_t n : kSizes) matrices.push_back(path_graph_matrix(static_cast<Index>(n)));
  for (const CsrMatrix& a : matrices) {
    const std::size_t n = static_cast<std::size_t>(a.n_rows);
    const std::vector<double> x = ramp(n, 1.0, -0.375);
    std::vector<double> expected(n);
    spmv_reference(a, x.data(), expected.data());
    for (const int chunk : {4, 8, 16, 32}) {
      for (const Index sigma : {Index{1}, Index{chunk}, Index{256}}) {
        std::vector<double> y(n, -9.0);
        make_sell_backend(a, chunk, sigma)->spmv(x.data(), y.data());
        for (std::size_t i = 0; i < n; ++i) {
          if (y[i] != expected[i]) {
            test_util::fail(__FILE__, __LINE__, "n=" + std::to_string(n) + " C=" + std::to_string(chunk) +
                                                    " sigma=" + std::to_string(sigma) + " row " + std::to_string(i));
          }
        }
      }
    }
  }
}

TEST(sell_spmv_dot_returns_x_dot_y_in_the_original_order) {
  const CsrMatrix a = make_stencil(Stencil::Point27, 5);
  const std::size_t n = static_cast<std::size_t>(a.n_rows);
  const std::vector<double> x = ramp(n, -2.0, 0.03125);
  std::vector<double> expected(n), y(n);
  spmv_reference(a, x.data(), expected.data());
  double dot = 0.0;
  for (std::size_t i = 0; i < n; ++i) dot += x[i] * expected[i];
  for (const int chunk : {4, 8, 16, 32}) {
    const double fused = make_sell_backend(a, chunk, 256)->spmv_dot(x.data(), y.data());
    CHECK_NEAR(fused, dot, 1e-12 * std::fabs(dot));
    for (std::size_t i = 0; i < n; ++i) CHECK_EQ(y[i], expected[i]);
  }
}

TEST(cg_with_the_sell_backend_matches_csr) {
  const CsrMatrix a = make_stencil(Stencil::Poisson3D, 12);
  const AlignedArray<double> b = rhs_for_unit_solution(a);
  const std::size_t n = static_cast<std::size_t>(a.n_rows);
  AlignedArray<double> x_csr(n), x_sell(n);
  const SolveResult csr = cg(*make_seq_backend(clone(a)), nullptr, b.data(), x_csr.data(), {});
  const SolveResult sell = cg(*make_sell_backend(a), nullptr, b.data(), x_sell.data(), {});
  CHECK(csr.status == SolveStatus::Converged);
  CHECK(sell.status == SolveStatus::Converged);
  CHECK(std::llabs(csr.spmv_count - sell.spmv_count) <= std::max<long long>(1, csr.spmv_count / 100));
  for (std::size_t i = 0; i < n; ++i) CHECK_NEAR(x_sell[i], 1.0, 1e-6);
}

TEST(sell_backend_needs_a_square_matrix) {
  Coo coo;
  coo.n_rows = 2;
  coo.n_cols = 3;
  coo.add(0, 0, 1.0);
  CHECK_THROWS_WITH(make_sell_backend(csr_from_coo(coo)), "square");
}

TEST(the_general_path_needs_the_same_iterations_as_the_merged_one) {
  std::vector<CsrMatrix> matrices;
  matrices.push_back(make_stencil(Stencil::Poisson3D, 12));
  matrices.push_back(scaled_path_matrix(200));
  for (const CsrMatrix& a : matrices) {
    const std::size_t n = static_cast<std::size_t>(a.n_rows);
    const AlignedArray<double> b = rhs_for_random_solution(a);
    const std::unique_ptr<Backend> backend = make_seq_backend(clone(a));
    AlignedArray<double> x_general(n), x_merged(n);
    StoredZJacobi general(a);
    const std::unique_ptr<Preconditioner> merged = make_jacobi_preconditioner(a);
    const SolveResult g = cg(*backend, general, b.data(), x_general.data(), {});
    const SolveResult m = cg(*backend, *merged, b.data(), x_merged.data(), {});
    CHECK(g.status == SolveStatus::Converged);
    CHECK_EQ(g.spmv_count, m.spmv_count);
    for (std::size_t i = 0; i < n; ++i) CHECK_NEAR(x_general[i], x_merged[i], 1e-9);
  }
}

TEST(history_is_recorded_through_any_preconditioner) {
  const CsrMatrix a = scaled_path_matrix(200);
  const AlignedArray<double> b = rhs_for_random_solution(a);
  const std::unique_ptr<Backend> backend = make_seq_backend(clone(a));
  AlignedArray<double> x(static_cast<std::size_t>(a.n_rows));
  StoredZJacobi general(a);
  std::vector<double> history;
  const SolveResult result = cg_with_history(*backend, general, b.data(), x.data(), {}, history);
  CHECK(result.status == SolveStatus::Converged);
  CHECK_EQ(history.size(), static_cast<std::size_t>(result.spmv_count + 1));
  CHECK_NEAR(history.back(), result.recursive_residual, 1e-15);
}

TEST(the_old_entry_point_still_works) {
  const CsrMatrix a = make_stencil(Stencil::Poisson3D, 10);
  const std::size_t n = static_cast<std::size_t>(a.n_rows);
  const AlignedArray<double> b = rhs_for_random_solution(a);
  const AlignedArray<double> inv_diag = jacobi_inverse_diagonal(a);
  const std::unique_ptr<Backend> backend = make_seq_backend(clone(a));
  for (const bool jacobi : {false, true}) {
    AlignedArray<double> x_old(n), x_new(n);
    const SolveResult old_way = cg(*backend, jacobi ? inv_diag.data() : nullptr, b.data(), x_old.data(), {});
    const std::unique_ptr<Preconditioner> m =
        jacobi ? make_jacobi_preconditioner(a) : make_identity_preconditioner();
    const SolveResult new_way = cg(*backend, *m, b.data(), x_new.data(), {});
    CHECK_EQ(old_way.spmv_count, new_way.spmv_count);
    for (std::size_t i = 0; i < n; ++i) CHECK_EQ(x_old[i], x_new[i]);
  }
}

TEST(the_factory_builds_each_kind_it_knows) {
  const CsrMatrix a = make_stencil(Stencil::Poisson2D, 5);
  CHECK(make_preconditioner(Precond::None, a) != nullptr);
  CHECK(make_preconditioner(Precond::Jacobi, a) != nullptr);
  CHECK_EQ(std::string(to_string(Precond::SymGaussSeidel)), std::string("sgs"));
}

TEST_MAIN()
