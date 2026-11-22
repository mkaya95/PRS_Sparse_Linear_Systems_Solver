// Eigen's conjugate gradient for the two Eigen methods, as bench/eigen_solvers.cpp runs it:
// a row-major matrix, both triangles used (Lower | Upper), and the given preconditioner.
#pragma once

#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Sparse>
#include <limits>
#include <stdexcept>

#include "method_main.hpp"

namespace ssolve_methods {

template <class Preconditioner>
ssolve::SolveResult eigen_cg(const ssolve::CsrMatrix& a, const double* b, double* x,
                             const ssolve::SolveOptions& options) {
  using namespace ssolve;
  using RowMajorMatrix = Eigen::SparseMatrix<double, Eigen::RowMajor, int>;
  if (a.nnz() > std::numeric_limits<int>::max()) throw std::runtime_error("Eigen supports at most 2^31 - 1 non-zeros");

  const double setup_start = wall_seconds();
  AlignedArray<int> row_ptr(a.row_ptr.size());
  for (std::size_t i = 0; i < a.row_ptr.size(); ++i) row_ptr[i] = static_cast<int>(a.row_ptr[i]);
  const Eigen::Map<const RowMajorMatrix> mapped(a.n_rows, a.n_cols, a.nnz(), row_ptr.data(), a.col.data(),
                                                a.val.data());
  const RowMajorMatrix matrix(mapped);
  Eigen::ConjugateGradient<RowMajorMatrix, Eigen::Lower | Eigen::Upper, Preconditioner> cg;
  cg.setTolerance(options.tol);
  cg.setMaxIterations(options.max_iterations > 0 ? static_cast<Eigen::Index>(options.max_iterations) : 2 * a.n_rows);
  cg.compute(matrix);
  SolveResult result;
  result.setup_seconds = wall_seconds() - setup_start;
  if (cg.info() != Eigen::Success) {
    result.status = SolveStatus::Breakdown;
    result.detail = "Eigen preconditioner setup failed";
    return result;
  }

  const Eigen::Map<const Eigen::VectorXd> rhs(b, a.n_rows);
  Eigen::Map<Eigen::VectorXd> solution(x, a.n_rows);
  {
    const ProfiledRegion region;
    const double solve_start = wall_seconds();
    solution = cg.solve(rhs);
    result.solve_seconds = wall_seconds() - solve_start;
  }
  // NoConvergence is running out of iterations; NumericalIssue is a breakdown.
  result.status = cg.info() == Eigen::Success          ? SolveStatus::Converged
                  : cg.info() == Eigen::NumericalIssue ? SolveStatus::Breakdown
                                                       : SolveStatus::MaxIterations;
  result.recursive_residual = cg.error();
  // A converged run did one more SpMV than iterations() reports (see bench/eigen_solvers.cpp).
  result.spmv_count = cg.iterations() + (result.status == SolveStatus::Converged ? 1 : 0);
  verify_result(a, b, x, options, result);
  return result;
}

}  // namespace ssolve_methods
