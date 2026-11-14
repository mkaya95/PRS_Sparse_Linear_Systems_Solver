// Incomplete Cholesky, from Eigen 5.0.1 (third_party/eigen):
//   factorize and updateList   Eigen/src/IterativeLinearSolvers/IncompleteCholesky.h
//   QuickSplit                 Eigen/src/IterativeLinearSolvers/IncompleteLUT.h
//   apply                      IncompleteCholesky::_solve_impl with the sparse_solve_triangular_selector
//                              <Lower, ColMajor> and <Upper, RowMajor> loops, Eigen/src/SparseCore/TriangularSolver.h
// Eigen's loops and their order of operations are kept. Eigen's containers become plain arrays, and
// only natural ordering is supported, so there is no permutation.
//
// Eigen's factorization has two bugs, marked "Eigen bug" below. They are copied on purpose, so that
// ours-ic gives the numbers of eigen-E4 and has it to check against. Do not fix them in place:
// EigenBugs::Fixed corrects both, as the separate solver ours-ic-fixed.
#include "ssolve/incomplete_cholesky.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <list>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "ssolve/backend.hpp"

namespace ssolve {

namespace {

using VectorList = std::vector<std::list<Index>>;

// Eigen's internal::QuickSplit: moves the ncut entries of largest magnitude to the front.
void QuickSplit(double* row, Index* ind, std::int64_t n, std::int64_t ncut) {
  using std::swap;
  std::int64_t mid;
  std::int64_t first, last;

  ncut--; /* to fit the zero-based indices */
  first = 0;
  last = n - 1;
  if (ncut < first || ncut > last) return;

  do {
    mid = first;
    double abskey = std::abs(row[mid]);
    for (std::int64_t j = first + 1; j <= last; j++) {
      if (std::abs(row[j]) > abskey) {
        ++mid;
        swap(row[mid], row[j]);
        swap(ind[mid], ind[j]);
      }
    }
    /* Interchange for the pivot element */
    swap(row[mid], row[first]);
    swap(ind[mid], ind[first]);

    if (mid > ncut)
      last = mid - 1;
    else if (mid < ncut)
      first = mid + 1;
  } while (mid != ncut);
}

// Eigen's IncompleteCholesky::updateList.
void updateList(const std::int64_t* colPtr, Index* rowIdx, double* vals, Index col, std::int64_t jk,
                std::vector<std::int64_t>& firstElt, VectorList& listCol) {
  if (jk < colPtr[col + 1]) {
    const std::int64_t p = colPtr[col + 1] - jk;
    std::int64_t minpos = 0;
    for (std::int64_t k = 1; k < p; ++k) {  // rowIdx.segment(jk, p).minCoeff(&minpos)
      if (rowIdx[jk + k] < rowIdx[jk + minpos]) minpos = k;
    }
    minpos += jk;
    if (rowIdx[minpos] != rowIdx[jk]) {
      // Swap
      std::swap(rowIdx[jk], rowIdx[minpos]);
      std::swap(vals[jk], vals[minpos]);
    }
    firstElt[static_cast<std::size_t>(col)] = jk;
    listCol[static_cast<std::size_t>(rowIdx[jk])].push_back(col);
  }
}

}  // namespace

CholeskyFactor incomplete_cholesky(const CsrMatrix& a, double initial_shift, EigenBugs bugs) {
  if (a.n_rows != a.n_cols) throw std::runtime_error("incomplete Cholesky needs a square matrix");
  const Index n = a.n_rows;
  const std::size_t un = static_cast<std::size_t>(n);

  // m_L = the lower triangle by columns. a is symmetric, so column j is row j's entries from the diagonal
  // on, already sorted. As Eigen's findOrInsertCoeff does, a missing diagonal gets a stored zero.
  CholeskyFactor f;
  f.n = n;
  f.col_ptr = AlignedArray<std::int64_t>(un + 1);
  f.col_ptr[0] = 0;
  for (Index j = 0; j < n; ++j) {
    std::int64_t count = 0;
    bool diagonal = false;
    for (std::int64_t k = a.row_ptr[j]; k < a.row_ptr[j + 1]; ++k) {
      if (a.col[k] >= j) ++count;
      if (a.col[k] == j) diagonal = true;
    }
    f.col_ptr[j + 1] = f.col_ptr[j] + count + (diagonal ? 0 : 1);
  }
  const std::int64_t nnz = f.col_ptr[un];
  f.row = AlignedArray<Index>(static_cast<std::size_t>(nnz));
  f.val = AlignedArray<double>(static_cast<std::size_t>(nnz));
  for (Index j = 0; j < n; ++j) {
    std::int64_t at = f.col_ptr[j];
    bool diagonal = false;
    for (std::int64_t k = a.row_ptr[j]; k < a.row_ptr[j + 1]; ++k) diagonal = diagonal || a.col[k] == j;
    if (!diagonal) {
      f.row[at] = j;
      f.val[at++] = 0.0;
    }
    for (std::int64_t k = a.row_ptr[j]; k < a.row_ptr[j + 1]; ++k) {
      if (a.col[k] >= j) {
        f.row[at] = a.col[k];
        f.val[at++] = a.val[k];
      }
    }
  }

  double* const vals = f.val.data();      // values
  Index* const rowIdx = f.row.data();     // Row indices
  const std::int64_t* const colPtr = f.col_ptr.data();  // Pointer to the beginning of each column
  std::vector<std::int64_t> firstElt(un);  // for each j, points to the next entry in vals that will be used in the factorization
  VectorList listCol(un);                  // listCol(j) is a linked list of columns to update column j
  std::vector<double> col_vals(un);        // Store a  nonzero values in each column
  std::vector<Index> col_irow(un);         // Row indices of nonzero elements in each column
  std::vector<std::int64_t> col_pattern(un, -1);
  std::int64_t col_nnz;

  // Computes the scaling factors
  f.scale = AlignedArray<double>(un);
  f.scale.fill(0.0);
  double* const scale = f.scale.data();
  for (Index j = 0; j < n; j++)
    for (std::int64_t k = colPtr[j]; k < colPtr[j + 1]; k++) {
      scale[j] += vals[k] * vals[k];
      if (rowIdx[k] != j) scale[rowIdx[k]] += vals[k] * vals[k];
    }

  for (Index j = 0; j < n; j++) scale[j] = std::sqrt(std::sqrt(scale[j]));

  for (Index j = 0; j < n; ++j)
    if (scale[j] > (std::numeric_limits<double>::min)())
      scale[j] = 1.0 / scale[j];
    else
      scale[j] = 1;

  // Scale and compute the shift for the matrix
  double mindiag = std::numeric_limits<double>::max();
  for (Index j = 0; j < n; j++) {
    for (std::int64_t k = colPtr[j]; k < colPtr[j + 1]; k++) vals[k] *= (scale[j] * scale[rowIdx[k]]);
    mindiag = std::min(vals[colPtr[j]], mindiag);
  }

  const std::vector<double> vals_save(vals, vals + nnz);
  const std::vector<Index> rowIdx_save(rowIdx, rowIdx + nnz);

  double shift = 0.0;
  if (mindiag <= 0.0) shift = initial_shift - mindiag;

  bool success = false;

  // Try to perform the incomplete factorization using the current shift
  int iter = 0;
  do {
    // Apply the shift to the diagonal elements of the matrix
    for (Index j = 0; j < n; j++) vals[colPtr[j]] += shift;

    // jki version of the Cholesky factorization
    Index j = 0;
    for (; j < n; ++j) {
      // Left-looking factorization of the j-th column
      // First, load the j-th column into col_vals
      double diag = vals[colPtr[j]];  // It is assumed that only the lower part is stored
      col_nnz = 0;
      for (std::int64_t i = colPtr[j] + 1; i < colPtr[j + 1]; i++) {
        const Index l = rowIdx[i];
        col_vals[col_nnz] = vals[i];
        col_irow[col_nnz] = l;
        col_pattern[l] = col_nnz;
        col_nnz++;
      }
      {
        // Browse all previous columns that will update column j
        for (auto k = listCol[j].begin(); k != listCol[j].end(); k++) {
          std::int64_t jk = firstElt[*k];  // First element to use in the column
          const double v_j_jk = vals[jk];

          jk += 1;
          for (std::int64_t i = jk; i < colPtr[*k + 1]; i++) {
            const Index l = rowIdx[i];
            if (col_pattern[l] < 0) {
              // Eigen bug 1: an entry that column j does not have yet starts as 0 - vals[i] * v_j_jk,
              // like the update in the other branch, but Eigen stores it with a plus sign.
              col_vals[col_nnz] = bugs == EigenBugs::Fixed ? -vals[i] * v_j_jk : vals[i] * v_j_jk;
              col_irow[col_nnz] = l;
              col_pattern[l] = col_nnz;
              col_nnz++;
            } else
              col_vals[col_pattern[l]] -= vals[i] * v_j_jk;
          }
          updateList(colPtr, rowIdx, vals, *k, jk, firstElt, listCol);
        }
      }

      // Scale the current column
      if (diag <= 0) {
        if (++iter >= 10) throw std::runtime_error("incomplete Cholesky failed after 10 shifts");

        // increase shift
        shift = std::max(initial_shift, 2.0 * shift);
        // restore m_L, col_pattern, and listCol
        std::copy(vals_save.begin(), vals_save.end(), vals);
        std::copy(rowIdx_save.begin(), rowIdx_save.end(), rowIdx);
        std::fill(col_pattern.begin(), col_pattern.end(), -1);
        for (std::size_t i = 0; i < un; ++i) listCol[i].clear();

        break;
      }

      const double rdiag = std::sqrt(diag);
      vals[colPtr[j]] = rdiag;
      for (std::int64_t k = 0; k < col_nnz; ++k) {
        const Index i = col_irow[k];
        // Scale
        col_vals[k] /= rdiag;
        // Update the remaining diagonals with col_vals
        vals[colPtr[i]] -= col_vals[k] * col_vals[k];
      }
      // Select the largest p elements
      // p is the original number of elements in the column (without the diagonal)
      const std::int64_t p = colPtr[j + 1] - colPtr[j] - 1;
      QuickSplit(col_vals.data(), col_irow.data(), col_nnz, p);
      // Insert the largest p elements in the matrix
      std::int64_t cpt = 0;
      for (std::int64_t i = colPtr[j] + 1; i < colPtr[j + 1]; i++) {
        vals[i] = col_vals[cpt];
        rowIdx[i] = col_irow[cpt];
        // restore col_pattern:
        col_pattern[col_irow[cpt]] = -1;
        cpt++;
      }
      // Eigen bug 2: only the p entries kept are reset above. The entries QuickSplit dropped keep their
      // position in col_pattern, so a later column takes them for its own and updates the wrong slot.
      if (bugs == EigenBugs::Fixed) {
        for (std::int64_t k = cpt; k < col_nnz; ++k) col_pattern[col_irow[k]] = -1;
      }
      // Get the first smallest row index and put it after the diagonal element
      const std::int64_t jk = colPtr[j] + 1;
      updateList(colPtr, rowIdx, vals, j, jk, firstElt, listCol);
    }

    if (j == n) success = true;
  } while (!success);

  f.shift = shift;
  return f;
}

void incomplete_cholesky_apply(const CholeskyFactor& f, const double* r, double* z) {
  const Index n = f.n;
  const std::int64_t* const col_ptr = f.col_ptr.data();
  const Index* const row = f.row.data();
  const double* const val = f.val.data();

  // x = m_scale.asDiagonal() * x
  for (Index i = 0; i < n; ++i) z[i] = f.scale[i] * r[i];

  // x = m_L.triangularView<Lower>().solve(x): forward substitution, col-major
  for (Index i = 0; i < n; ++i) {
    double& tmp = z[i];
    if (tmp != 0.0)  // optimization when other is actually sparse
    {
      std::int64_t it = col_ptr[i];
      while (it < col_ptr[i + 1] && row[it] < i) ++it;
      tmp /= val[it];
      if (it < col_ptr[i + 1] && row[it] == i) ++it;
      for (; it < col_ptr[i + 1]; ++it) {
        z[row[it]] = -tmp * val[it] + z[row[it]];
      }
    }
  }

  // x = m_L.adjoint().triangularView<Upper>().solve(x): backward substitution, row-major
  for (Index i = n - 1; i >= 0; --i) {
    double tmp = z[i];
    double l_ii = 0.0;
    std::int64_t it = col_ptr[i];
    while (it < col_ptr[i + 1] && row[it] < i) ++it;
    l_ii = val[it];
    ++it;
    for (; it < col_ptr[i + 1]; ++it) {
      tmp = -val[it] * z[row[it]] + tmp;
    }
    z[i] = tmp / l_ii;
  }

  // x = m_scale.asDiagonal() * x
  for (Index i = 0; i < n; ++i) z[i] = f.scale[i] * z[i];
}

namespace {

// z is stored: the triangular solves need all of r before any of z is known.
class IcPreconditioner final : public Preconditioner {
 public:
  IcPreconditioner(const CsrMatrix& a, EigenBugs bugs)
      : f_(incomplete_cholesky(a, 1e-3, bugs)), z_(static_cast<std::size_t>(a.n_rows)) {}

  double start(const Backend& backend, const double* r, double, double* p) override {
    incomplete_cholesky_apply(f_, r, z_.data());
    backend.copy(z_.data(), p);
    return backend.dot(r, z_.data());
  }

  RrRz update(const Backend& backend, double a, const double* p, const double* q, double* x, double* r) override {
    const double rr = backend.cg_update(a, p, q, x, r);
    incomplete_cholesky_apply(f_, r, z_.data());
    return {rr, backend.dot(r, z_.data())};
  }

  void direction(const Backend& backend, const double*, double beta, double* p) override {
    backend.xpby(z_.data(), beta, p);
  }

 private:
  CholeskyFactor f_;
  AlignedArray<double> z_;
};

}  // namespace

std::unique_ptr<Preconditioner> make_ic_preconditioner(const CsrMatrix& a, EigenBugs bugs) {
  return std::make_unique<IcPreconditioner>(a, bugs);
}

}  // namespace ssolve
