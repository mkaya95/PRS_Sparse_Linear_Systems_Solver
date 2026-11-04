// Behaviour every solver must show. Each solver name registered in bench/ runs through these.
#include <cmath>
#include <limits>
#include <string>
#include <vector>

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
