// Eigen's conjugate gradient in the four configurations we compare against.
//
//   E0  row-major     Lower|Upper  Identity            plain CG baseline
//   E1  row-major     Lower|Upper  Diagonal            Jacobi baseline
//   E2  column-major  Lower        Diagonal            what a default Eigen user gets
//   E3  row-major     Lower|Upper  IncompleteCholesky  preconditioner baseline
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Sparse>
#include <limits>
#include <stdexcept>

#include "registry.hpp"
#include "ssolve/aligned_array.hpp"
#include "ssolve/timer.hpp"
#include "ssolve/verify.hpp"

using namespace ssolve;

namespace ssolve_bench {

namespace {

using RowMajorMatrix = Eigen::SparseMatrix<double, Eigen::RowMajor, int>;
using ColMajorMatrix = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;

template <class Sparse, int UpLo, class Preconditioner>
SolveResult run_eigen(const CsrMatrix& a, const double* b, double* x, const SolveOptions& options) {
  validate_options(options);
  if (a.n_rows != a.n_cols) throw std::runtime_error("Eigen conjugate gradient needs a square matrix");
  if (a.nnz() > std::numeric_limits<int>::max()) {
    throw std::runtime_error("Eigen driver supports at most 2^31 - 1 non-zeros");
  }

  const double setup_start = wall_seconds();
  // Storage construction is part of setup on both sides.
  AlignedArray<int> row_ptr(a.row_ptr.size());
  for (std::size_t i = 0; i < a.row_ptr.size(); ++i) row_ptr[i] = static_cast<int>(a.row_ptr[i]);
  const Eigen::Map<const RowMajorMatrix> mapped(a.n_rows, a.n_cols, a.nnz(), row_ptr.data(), a.col.data(),
                                                a.val.data());
  const Sparse matrix(mapped);

  const Eigen::Index n = a.n_rows;
  const Eigen::Map<const Eigen::VectorXd> rhs(b, n);
  Eigen::Map<Eigen::VectorXd> solution(x, n);
  const bool rhs_is_zero = !(rhs.array() != 0.0).any();

  Eigen::ConjugateGradient<Sparse, UpLo, Preconditioner> cg;
  cg.setTolerance(options.tol);
  cg.setMaxIterations(options.max_iterations > 0 ? static_cast<Eigen::Index>(options.max_iterations) : 2 * n);

  SolveResult result;
  cg.compute(matrix);
  result.setup_seconds = wall_seconds() - setup_start;
  if (cg.info() != Eigen::Success) {
    solution.setZero();
    result.status = SolveStatus::Breakdown;
    result.detail = "Eigen preconditioner setup failed";
    verify_result(a, b, x, options, result);
    return result;
  }

  const double solve_start = wall_seconds();
  if (options.use_initial_guess) {
    const Eigen::VectorXd guess = solution;
    solution = cg.solveWithGuess(rhs, guess);
  } else {
    solution = cg.solve(rhs);
  }
  result.solve_seconds = wall_seconds() - solve_start;

  result.status = cg.info() == Eigen::Success ? SolveStatus::Converged : SolveStatus::MaxIterations;
  result.recursive_residual = cg.error();
  // Convergence breaks out of Eigen's loop before its counter advances, so a converged
  // run did one more SpMV than iterations() reports. A zero rhs returns before the loop.
  if (rhs_is_zero) {
    result.spmv_count = 0;
  } else {
    result.spmv_count = cg.iterations() + (result.status == SolveStatus::Converged ? 1 : 0);
  }
  verify_result(a, b, x, options, result);
  return result;
}

}  // namespace

std::string eigen_version() {
#ifdef EIGEN_PATCH_VERSION
  return std::to_string(EIGEN_MAJOR_VERSION) + "." + std::to_string(EIGEN_MINOR_VERSION) + "." +
         std::to_string(EIGEN_PATCH_VERSION);
#else
  return std::to_string(EIGEN_WORLD_VERSION) + "." + std::to_string(EIGEN_MAJOR_VERSION) + "." +
         std::to_string(EIGEN_MINOR_VERSION);
#endif
}

std::vector<NamedSolver> eigen_solvers() {
  using Eigen::DiagonalPreconditioner;
  using Eigen::IdentityPreconditioner;
  using Eigen::IncompleteCholesky;
  using Eigen::Lower;
  using Eigen::Upper;
  return {
      {"eigen-E0", Precond::None, run_eigen<RowMajorMatrix, Lower | Upper, IdentityPreconditioner>},
      {"eigen-E1", Precond::Jacobi, run_eigen<RowMajorMatrix, Lower | Upper, DiagonalPreconditioner<double>>},
      {"eigen-E2", Precond::Jacobi, run_eigen<ColMajorMatrix, Lower, DiagonalPreconditioner<double>>},
      {"eigen-E3", Precond::None, run_eigen<RowMajorMatrix, Lower | Upper, IncompleteCholesky<double>>},
  };
}

}  // namespace ssolve_bench
