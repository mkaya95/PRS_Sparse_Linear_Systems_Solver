// Behaviour every solver must show. Each solver name registered in bench/ runs through these.
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

#include "kernels.hpp"
#include "matrix_spec.hpp"
#include "registry.hpp"
#include "ssolve/generators.hpp"
#include "ssolve/verify.hpp"
#include "test_util.hpp"

using namespace ssolve;
using ssolve_bench::NamedSolver;

namespace {

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
  for (const NamedSolver& solver : ssolve_bench::all_solvers()) {
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
  for (const NamedSolver& solver : ssolve_bench::all_solvers()) {
    std::vector<double> x;
    const SolveResult result = solve(solver, a, b, x, options);
    CHECK_EQ(std::string(to_string(result.status)) + " " + solver.name, std::string("MaxIterations ") + solver.name);
    CHECK_EQ(result.spmv_count, std::int64_t{3});
  }
}

TEST(every_solver_counts_one_spmv_on_the_identity) {
  const CsrMatrix a = diagonal_matrix({1, 1, 1});
  const std::vector<double> b = {1, 2, 3};
  for (const NamedSolver& solver : ssolve_bench::all_solvers()) {
    std::vector<double> x;
    const SolveResult result = solve(solver, a, b, x);
    CHECK_EQ(result.spmv_count, std::int64_t{1});
    CHECK_NEAR(x[2], 3.0, 1e-12);
  }
}

TEST(plain_solvers_count_two_spmvs_with_two_distinct_eigenvalues) {
  const CsrMatrix a = diagonal_matrix({1, 2});
  const std::vector<double> b = {1, 1};
  for (const NamedSolver& solver : ssolve_bench::all_solvers()) {
    if (solver.precond != Precond::None || solver.name == "eigen-E3") continue;
    std::vector<double> x;
    const SolveResult result = solve(solver, a, b, x);
    CHECK_EQ(result.spmv_count, std::int64_t{2});
    CHECK_NEAR(x[1], 0.5, 1e-12);
  }
}

TEST(every_solver_returns_zero_for_a_zero_rhs_without_any_spmv) {
  const CsrMatrix a = make_stencil(Stencil::Poisson2D, 5);
  const std::vector<double> b(25, 0.0);
  for (const NamedSolver& solver : ssolve_bench::all_solvers()) {
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
  for (const NamedSolver& solver : ssolve_bench::all_solvers()) {
    std::vector<double> x;
    SolveOptions options;
    options.max_iterations = 0;
    CHECK(solve(solver, a, b, x, options).status == SolveStatus::Converged);
  }
}

TEST(our_solvers_solve_a_large_2d_poisson_system) {
  const CsrMatrix a = make_stencil(Stencil::Poisson2D, 100);
  const std::vector<double> b = unit_rhs(a);
  for (const NamedSolver& solver : ssolve_bench::ladder_solvers()) {
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
  for (const NamedSolver& solver : ssolve_bench::ladder_solvers()) {
    std::vector<double> x;
    const SolveResult result = solve(solver, a, b, x);
    CHECK_EQ(result.spmv_count, std::int64_t{1});
    CHECK_NEAR(x[0], 2.0, 1e-14);
  }
}

TEST(our_solvers_report_breakdown_on_an_indefinite_matrix) {
  const CsrMatrix a = diagonal_matrix({1, -1});
  const std::vector<double> b = {1, 1};
  for (const NamedSolver& solver : ssolve_bench::ladder_solvers()) {
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
  for (const NamedSolver& solver : ssolve_bench::ladder_solvers()) {
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
  for (const NamedSolver& solver : ssolve_bench::ladder_solvers()) {
    std::vector<double> x;
    CHECK_EQ(std::string(to_string(solve(solver, a, b, x).status)), std::string("Breakdown"));
  }
}

TEST(our_solvers_do_not_mistake_an_all_nan_rhs_for_zero) {
  const CsrMatrix a = diagonal_matrix({1, 2});
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const std::vector<double> b = {nan, nan};
  for (const NamedSolver& solver : ssolve_bench::ladder_solvers()) {
    std::vector<double> x;
    CHECK_EQ(std::string(to_string(solve(solver, a, b, x).status)), std::string("Breakdown"));
  }
}

TEST(our_solvers_report_breakdown_on_an_infinite_rhs) {
  const CsrMatrix a = diagonal_matrix({1, 2});
  const std::vector<double> b = {1, std::numeric_limits<double>::infinity()};
  for (const NamedSolver& solver : ssolve_bench::ladder_solvers()) {
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
  for (const NamedSolver& solver : ssolve_bench::ladder_solvers()) {
    std::vector<double> x;
    CHECK_THROWS_WITH(solve(solver, a, b, x), "square");
  }
}

TEST(our_solvers_handle_a_very_small_rhs) {
  // Squaring 1e-200 underflows to 0, which must not look like b = 0.
  const CsrMatrix a = make_stencil(Stencil::Poisson2D, 20);
  std::vector<double> b = unit_rhs(a);
  for (double& value : b) value = value * 1e-200 + 1e-200;
  for (const NamedSolver& solver : ssolve_bench::ladder_solvers()) {
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
  for (const NamedSolver& solver : ssolve_bench::ladder_solvers()) {
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
  for (const NamedSolver& solver : ssolve_bench::ladder_solvers()) {
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

TEST(our_solvers_match_eigen_spmv_counts) {
  const CsrMatrix poisson = make_stencil(Stencil::Poisson3D, 15);
  const std::vector<double> b = unit_rhs(poisson);
  std::vector<double> x;
  const SolveResult plain = solve(ssolve_bench::find_solver("eigen-E0"), poisson, b, x);
  const SolveResult jacobi = solve(ssolve_bench::find_solver("eigen-E1"), poisson, b, x);
  for (const NamedSolver& solver : ssolve_bench::ladder_solvers()) {
    const SolveResult ours = solve(solver, poisson, b, x);
    const SolveResult& theirs = solver.precond == Precond::None ? plain : jacobi;
    CHECK(std::llabs(ours.spmv_count - theirs.spmv_count) <=
          std::max<long long>(1, theirs.spmv_count / 100));
  }
}

TEST(jacobi_solvers_converge_at_once_on_a_badly_scaled_diagonal_system) {
  const CsrMatrix a = diagonal_matrix({1, 1e6, 1e-6});
  const std::vector<double> b = {1, 2, 3};
  int jacobi_solvers = 0;
  for (const NamedSolver& solver : ssolve_bench::ladder_solvers()) {
    if (solver.precond != Precond::Jacobi) continue;
    ++jacobi_solvers;
    std::vector<double> x;
    const SolveResult result = solve(solver, a, b, x);
    CHECK(result.status == SolveStatus::Converged);
    CHECK(result.spmv_count <= 2);
    CHECK_NEAR(x[2] / 3e6, 1.0, 1e-9);
  }
  CHECK(jacobi_solvers >= 3);
}

TEST(a_plain_solver_needs_more_iterations_on_that_system_than_jacobi) {
  // Proves the Jacobi solvers are actually preconditioning, not just running plain CG.
  const CsrMatrix a = diagonal_matrix({1, 1e3, 1e-3, 5, 7});
  const std::vector<double> b = {1, 2, 3, 4, 5};
  std::vector<double> x;
  const SolveResult plain = solve(ssolve_bench::find_solver("ours"), a, b, x);
  const SolveResult jacobi = solve(ssolve_bench::find_solver("ours-jacobi"), a, b, x);
  CHECK(jacobi.spmv_count < plain.spmv_count);
  CHECK_EQ(jacobi.spmv_count, std::int64_t{1});
}

TEST(every_ladder_spmv_matches_the_reference_on_all_short_row_lengths) {
  // Row lengths 0 to 9 exercise each unrolled loop's remainder path.
  const Index n = 16;
  Coo coo;
  coo.n_rows = n;
  coo.n_cols = n;
  for (Index i = 0; i < n; ++i) {
    const Index length = i % 10;
    for (Index offset = 0; offset < length; ++offset) {
      coo.add(i, (i + 3 * offset) % n, 1.0 + 0.25 * i - 0.125 * offset);
    }
  }
  const CsrMatrix a = csr_from_coo(coo);

  std::vector<double> x(n), expected(n), y(n);
  for (Index i = 0; i < n; ++i) x[i] = 1.0 / (1.0 + i);
  spmv_reference(a, x.data(), expected.data());

  for (const std::string& name : ssolve_bench::ladder_kernel_names()) {
    y.assign(n, 99.0);
    ssolve_bench::ladder_kernels(name, a).spmv(x.data(), y.data());
    for (Index i = 0; i < n; ++i) {
      if (!(std::fabs(y[i] - expected[i]) <= 1e-14 * (1.0 + std::fabs(expected[i])))) {
        test_util::fail(name.c_str(), i, "SpMV differs from the reference");
      }
    }
  }
}

TEST(the_ladder_has_every_rung) {
  for (const char* name : {"L0-naive", "L1-restrict", "L2-simd", "L3-acc2", "L3-unroll4", "L3-simdrow", "L4-fused",
                           "L5-nopass", "L6-rowptr32", "J0-passes", "J1-fused", "ours", "ours-jacobi"}) {
    CHECK_EQ(ssolve_bench::find_solver(name).name, std::string(name));
  }
}

TEST(unknown_solver_name_lists_the_valid_ones) {
  CHECK_THROWS_WITH(ssolve_bench::find_solver("no-such-solver"), "eigen-E0");
}

TEST(matrix_specs_build_generated_matrices) {
  CHECK_EQ(ssolve_bench::load_matrix_spec("poisson2d:4").n_rows, 16);
  CHECK_EQ(ssolve_bench::load_matrix_spec("poisson3d:3").n_rows, 27);
  CHECK_EQ(ssolve_bench::load_matrix_spec("point27:3").nnz(), std::int64_t{343});
}

TEST(matrix_specs_size_to_the_last_level_cache) {
  const CsrMatrix small = ssolve_bench::load_matrix_spec("poisson3d:llc=0.5");
  const CsrMatrix large = ssolve_bench::load_matrix_spec("poisson3d:llc=4");
  CHECK(large.nnz() > small.nnz());
  // A factor of 8 in bytes, within the rounding of N to a whole number.
  const double ratio = static_cast<double>(large.nnz()) / static_cast<double>(small.nnz());
  CHECK(ratio > 7.0 && ratio < 9.0);
}

TEST(bad_matrix_specs_are_rejected) {
  CHECK_THROWS_WITH(ssolve_bench::load_matrix_spec("poisson3d:abc"), "poisson3d:abc");
  CHECK_THROWS_WITH(ssolve_bench::load_matrix_spec("poisson3d:llc=-1"), "llc");
  CHECK_THROWS_WITH(ssolve_bench::load_matrix_spec("cubic:5.mtx"), "cubic:5.mtx");
}

TEST(display_names_are_short) {
  CHECK_EQ(ssolve_bench::matrix_display_name("poisson3d:llc=4"), std::string("poisson3d:llc=4"));
  CHECK_EQ(ssolve_bench::matrix_display_name("/data/x/ldoor.mtx"), std::string("ldoor"));
}

TEST_MAIN()
