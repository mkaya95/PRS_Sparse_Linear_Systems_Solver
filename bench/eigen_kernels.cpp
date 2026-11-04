// Eigen's kernels, written exactly as its conjugate gradient calls them.
#include <Eigen/Sparse>
#include <memory>

#include "kernels.hpp"
#include "ssolve/aligned_array.hpp"

using namespace ssolve;

namespace ssolve_bench {

KernelSet eigen_kernels(const CsrMatrix& a) {
  using Matrix = Eigen::SparseMatrix<double, Eigen::RowMajor, int>;
  using ConstVector = Eigen::Map<const Eigen::VectorXd>;
  using Vector = Eigen::Map<Eigen::VectorXd>;

  AlignedArray<int> row_ptr(a.row_ptr.size());
  for (std::size_t i = 0; i < a.row_ptr.size(); ++i) row_ptr[i] = static_cast<int>(a.row_ptr[i]);
  const Eigen::Map<const Matrix> mapped(a.n_rows, a.n_cols, a.nnz(), row_ptr.data(), a.col.data(), a.val.data());
  const auto matrix = std::make_shared<const Matrix>(mapped);
  const Eigen::Index n = a.n_rows;

  KernelSet set;
  set.spmv = [matrix, n](const double* x, double* y) { Vector(y, n).noalias() = *matrix * ConstVector(x, n); };
  set.dot = [n](const double* x, const double* y) { return ConstVector(x, n).dot(ConstVector(y, n)); };
  set.axpy = [n](double alpha, const double* x, double* y) { Vector(y, n) += alpha * ConstVector(x, n); };
  set.norm2 = [n](const double* x) { return ConstVector(x, n).squaredNorm(); };
  set.xpby = [n](const double* x, double beta, double* y) { Vector(y, n) = ConstVector(x, n) + beta * Vector(y, n); };
  return set;
}

}  // namespace ssolve_bench
