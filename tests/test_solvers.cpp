// Behaviour of the sequential CG solver, with and without Jacobi.
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "ssolve/cg.hpp"
#include "ssolve/matrix_spec.hpp"
#include "ssolve/generators.hpp"
#include "ssolve/verify.hpp"
#include "test_util.hpp"

using namespace ssolve;

namespace {

struct NamedSolver {
  std::string name;
  Precond precond;

  SolveResult run(const CsrMatrix& a, const double* b, double* x, const SolveOptions& options) const {
    AlignedArray<double> inv_diag;
    if (precond == Precond::Jacobi) inv_diag = jacobi_inverse_diagonal(a);
    const auto backend = make_seq_backend(clone(a));
    return verified_cg(a, *backend, inv_diag.data(), b, x, options);
  }
};

const NamedSolver plain_solver{"cg", Precond::None};
const NamedSolver jacobi_solver{"cg-jacobi", Precond::Jacobi};
const std::vector<NamedSolver> solvers{plain_solver, jacobi_solver};

CsrMatrix diagonal_matrix(const std::vector<double>& values) {
  Coo coo;
  coo.n_rows = static_cast<Index>(values.size());
  coo.n_cols = coo.n_rows;
  for (Index i = 0; i < coo.n_rows; ++i) coo.add(i, i, values[i]);
  return csr_from_coo(coo);
}

SolveResult solve(const NamedSolver& solver, const CsrMatrix& a, const std::vector<double>& b, std::vector<double>& x,
                  SolveOptions options = {}) {
  x.assign(b.size(), 7.0);  // garbage: a solver must not read x unless asked to
  return solver.run(a, b.data(), x.data(), options);
}

std::vector<double> unit_rhs(const CsrMatrix& a) {
  const AlignedArray<double> b = rhs_for_unit_solution(a);
  return std::vector<double>(b.begin(), b.end());
}

void check_near_ones(const std::string& name, const std::vector<double>& x, double tol) {
  double worst = 0.0;
  for (const double value : x) worst = std::fmax(worst, std::fabs(value - 1.0));
  if (!(worst <= tol)) test_util::fail(name.c_str(), __LINE__, "x is not close to ones: " + test_util::str(worst));
}

}  // namespace

TEST(every_solver_solves_a_3d_poisson_system) {
  const CsrMatrix a = make_stencil(Stencil::Poisson3D, 10);
  const std::vector<double> b = unit_rhs(a);
  for (const NamedSolver& solver : solvers) {
    std::vector<double> x;
    const SolveResult result = solve(solver, a, b, x);
    CHECK_EQ(std::string(to_string(result.status)) + " " + solver.name, std::string("Converged ") + solver.name);
    CHECK(true_relative_residual(a, b.data(), x.data()) < 1e-7);
    check_near_ones(solver.name, x, 1e-5);
    CHECK(result.spmv_count > 1);
    CHECK(result.recursive_residual < 1e-8);
    CHECK(result.solve_seconds >= 0.0);
  }
}

TEST(every_solver_stops_at_the_iteration_limit) {
  const CsrMatrix a = make_stencil(Stencil::Poisson3D, 10);
  const std::vector<double> b = unit_rhs(a);
  SolveOptions options;
  options.max_iterations = 3;
  for (const NamedSolver& solver : solvers) {
    std::vector<double> x;
    const SolveResult result = solve(solver, a, b, x, options);
    CHECK_EQ(std::string(to_string(result.status)) + " " + solver.name, std::string("MaxIterations ") + solver.name);
    CHECK_EQ(result.spmv_count, std::int64_t{3});
  }
}

TEST(every_solver_counts_one_spmv_on_the_identity) {
  const CsrMatrix a = diagonal_matrix({1, 1, 1});
  const std::vector<double> b = {1, 2, 3};
  for (const NamedSolver& solver : solvers) {
    std::vector<double> x;
    const SolveResult result = solve(solver, a, b, x);
    CHECK_EQ(result.spmv_count, std::int64_t{1});
    CHECK_NEAR(x[2], 3.0, 1e-12);
  }
}

TEST(plain_solvers_count_two_spmvs_with_two_distinct_eigenvalues) {
  const CsrMatrix a = diagonal_matrix({1, 2});
  const std::vector<double> b = {1, 1};
  for (const NamedSolver& solver : solvers) {
    if (solver.precond != Precond::None) continue;
    std::vector<double> x;
    const SolveResult result = solve(solver, a, b, x);
    CHECK_EQ(result.spmv_count, std::int64_t{2});
    CHECK_NEAR(x[1], 0.5, 1e-12);
  }
}

TEST(every_solver_returns_zero_for_a_zero_rhs_without_any_spmv) {
  const CsrMatrix a = make_stencil(Stencil::Poisson2D, 5);
  const std::vector<double> b(25, 0.0);
  for (const NamedSolver& solver : solvers) {
    std::vector<double> x;
    const SolveResult result = solve(solver, a, b, x);
    CHECK(result.status == SolveStatus::Converged);
    CHECK_EQ(result.spmv_count, std::int64_t{0});
    for (const double value : x) CHECK_EQ(value, 0.0);
  }
}

TEST(a_zero_iteration_limit_means_twice_n) {
  // Eigen's default. On this system CG needs more iterations than a tiny fixed limit would allow.
  const CsrMatrix a = make_stencil(Stencil::Poisson2D, 12);
  const std::vector<double> b = unit_rhs(a);
  for (const NamedSolver& solver : solvers) {
    std::vector<double> x;
    SolveOptions options;
    options.max_iterations = 0;
    CHECK(solve(solver, a, b, x, options).status == SolveStatus::Converged);
  }
}

TEST(our_solvers_solve_a_large_2d_poisson_system) {
  const CsrMatrix a = make_stencil(Stencil::Poisson2D, 100);
  const std::vector<double> b = unit_rhs(a);
  for (const NamedSolver& solver : solvers) {
    std::vector<double> x;
    SolveOptions options;
    options.max_iterations = 2000;
    const SolveResult result = solve(solver, a, b, x, options);
    CHECK(result.status == SolveStatus::Converged);
    CHECK(true_relative_residual(a, b.data(), x.data()) < 1e-7);
    check_near_ones(solver.name, x, 1e-5);
  }
}

TEST(our_solvers_solve_a_1x1_system) {
  const CsrMatrix a = diagonal_matrix({4});
  const std::vector<double> b = {8};
  for (const NamedSolver& solver : solvers) {
    std::vector<double> x;
    const SolveResult result = solve(solver, a, b, x);
    CHECK_EQ(result.spmv_count, std::int64_t{1});
    CHECK_NEAR(x[0], 2.0, 1e-14);
  }
}

TEST(our_solvers_report_breakdown_on_an_indefinite_matrix) {
  const CsrMatrix a = diagonal_matrix({1, -1});
  const std::vector<double> b = {1, 1};
  for (const NamedSolver& solver : solvers) {
    std::vector<double> x;
    if (solver.precond == Precond::Jacobi) {
      CHECK_THROWS_WITH(solve(solver, a, b, x), "positive");
      continue;
    }
    CHECK_EQ(std::string(to_string(solve(solver, a, b, x).status)), std::string("Breakdown"));
  }
}

TEST(our_solvers_report_breakdown_on_nan_values_without_spinning) {
  const CsrMatrix a = diagonal_matrix({1, std::numeric_limits<double>::quiet_NaN()});
  const std::vector<double> b = {1, 1};
  for (const NamedSolver& solver : solvers) {
    std::vector<double> x;
    if (solver.precond == Precond::Jacobi) {
      CHECK_THROWS_WITH(solve(solver, a, b, x), "positive");
      continue;
    }
    const SolveResult result = solve(solver, a, b, x);
    CHECK_EQ(std::string(to_string(result.status)), std::string("Breakdown"));
    CHECK(result.spmv_count <= 1);
  }
}

TEST(our_solvers_report_breakdown_on_a_nan_rhs) {
  const CsrMatrix a = diagonal_matrix({1, 2});
  const std::vector<double> b = {1, std::numeric_limits<double>::quiet_NaN()};
  for (const NamedSolver& solver : solvers) {
    std::vector<double> x;
    CHECK_EQ(std::string(to_string(solve(solver, a, b, x).status)), std::string("Breakdown"));
  }
}

TEST(our_solvers_do_not_mistake_an_all_nan_rhs_for_zero) {
  const CsrMatrix a = diagonal_matrix({1, 2});
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const std::vector<double> b = {nan, nan};
  for (const NamedSolver& solver : solvers) {
    std::vector<double> x;
    CHECK_EQ(std::string(to_string(solve(solver, a, b, x).status)), std::string("Breakdown"));
  }
}

TEST(our_solvers_report_breakdown_on_an_infinite_rhs) {
  const CsrMatrix a = diagonal_matrix({1, 2});
  const std::vector<double> b = {1, std::numeric_limits<double>::infinity()};
  for (const NamedSolver& solver : solvers) {
    std::vector<double> x;
    CHECK_EQ(std::string(to_string(solve(solver, a, b, x).status)), std::string("Breakdown"));
  }
}

TEST(our_solvers_reject_a_non_square_matrix) {
  Coo coo;
  coo.n_rows = 1;
  coo.n_cols = 2;
  coo.add(0, 0, 1);
  const CsrMatrix a = csr_from_coo(coo);
  const std::vector<double> b = {1};
  for (const NamedSolver& solver : solvers) {
    std::vector<double> x;
    CHECK_THROWS_WITH(solve(solver, a, b, x), "square");
  }
}

TEST(our_solvers_handle_a_very_small_rhs) {
  // Squaring 1e-200 underflows to 0, which must not look like b = 0.
  const CsrMatrix a = make_stencil(Stencil::Poisson2D, 20);
  std::vector<double> b = unit_rhs(a);
  for (double& value : b) value = value * 1e-200 + 1e-200;
  for (const NamedSolver& solver : solvers) {
    std::vector<double> x;
    const SolveResult result = solve(solver, a, b, x);
    CHECK(result.status == SolveStatus::Converged);
    CHECK(result.spmv_count > 1);
    CHECK(true_relative_residual(a, b.data(), x.data()) < 1e-7);
  }
}

TEST(our_solvers_handle_a_very_large_rhs) {
  // Squaring 1e+200 overflows to infinity, which must not look like a failure.
  const CsrMatrix a = make_stencil(Stencil::Poisson2D, 20);
  std::vector<double> b = unit_rhs(a);
  for (double& value : b) value = value * 1e+200 + 1e+200;
  for (const NamedSolver& solver : solvers) {
    std::vector<double> x;
    const SolveResult result = solve(solver, a, b, x);
    CHECK(result.status == SolveStatus::Converged);
    CHECK(result.spmv_count > 1);
    CHECK(true_relative_residual(a, b.data(), x.data()) < 1e-7);
  }
}

TEST(our_solvers_start_from_a_given_guess) {
  const CsrMatrix a = make_stencil(Stencil::Poisson2D, 10);
  const std::vector<double> b = unit_rhs(a);
  SolveOptions options;
  options.use_initial_guess = true;
  for (const NamedSolver& solver : solvers) {
    std::vector<double> exact(b.size(), 1.0);
    const SolveResult at_solution = solver.run(a, b.data(), exact.data(), options);
    CHECK(at_solution.status == SolveStatus::Converged);
    CHECK_EQ(at_solution.spmv_count, std::int64_t{0});

    std::vector<double> nearby(b.size(), 1.001);
    const SolveResult from_nearby = solver.run(a, b.data(), nearby.data(), options);
    CHECK(from_nearby.status == SolveStatus::Converged);
    CHECK(true_relative_residual(a, b.data(), nearby.data()) < 1e-7);
  }
}

TEST(jacobi_solvers_converge_at_once_on_a_badly_scaled_diagonal_system) {
  const CsrMatrix a = diagonal_matrix({1, 1e6, 1e-6});
  const std::vector<double> b = {1, 2, 3};
  int jacobi_solvers = 0;
  for (const NamedSolver& solver : solvers) {
    if (solver.precond != Precond::Jacobi) continue;
    ++jacobi_solvers;
    std::vector<double> x;
    const SolveResult result = solve(solver, a, b, x);
    CHECK(result.status == SolveStatus::Converged);
    CHECK(result.spmv_count <= 2);
    CHECK_NEAR(x[2] / 3e6, 1.0, 1e-9);
  }
  CHECK_EQ(jacobi_solvers, 1);
}

TEST(a_plain_solver_needs_more_iterations_on_that_system_than_jacobi) {
  // Proves the Jacobi solvers are actually preconditioning, not just running plain CG.
  const CsrMatrix a = diagonal_matrix({1, 1e3, 1e-3, 5, 7});
  const std::vector<double> b = {1, 2, 3, 4, 5};
  std::vector<double> x;
  const SolveResult plain = solve(plain_solver, a, b, x);
  const SolveResult jacobi = solve(jacobi_solver, a, b, x);
  CHECK(jacobi.spmv_count < plain.spmv_count);
  CHECK_EQ(jacobi.spmv_count, std::int64_t{1});
}

TEST(matrix_specs_build_generated_matrices) {
  CHECK_EQ(load_matrix_spec("poisson2d:4").n_rows, 16);
  CHECK_EQ(load_matrix_spec("poisson3d:3").n_rows, 27);
  CHECK_EQ(load_matrix_spec("point27:3").nnz(), std::int64_t{343});
}

TEST(bad_matrix_specs_are_rejected) {
  CHECK_THROWS_WITH(load_matrix_spec("poisson3d:abc"), "poisson3d:abc");
  CHECK_THROWS_WITH(load_matrix_spec("poisson3d:llc=-1"), "llc");
  CHECK_THROWS_WITH(load_matrix_spec("cubic:5.mtx"), "cubic:5.mtx");
}

TEST_MAIN()
