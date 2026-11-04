// Sparse matrix storage: COO for input, CSR for computation.
#pragma once

#include <cstdint>
#include <vector>

#include "ssolve/aligned_array.hpp"

namespace ssolve {

using Index = std::int32_t;  // rows and columns

// Triplets in any order. Used only while loading a matrix.
struct Coo {
  Index n_rows = 0;
  Index n_cols = 0;
  std::vector<Index> row;
  std::vector<Index> col;
  std::vector<double> val;

  void add(Index r, Index c, double v) {
    row.push_back(r);
    col.push_back(c);
    val.push_back(v);
  }
};

// Compressed Sparse Row with sorted columns inside each row. Ptr is the row pointer type.
template <class Ptr>
struct CsrMatrixT {
  Index n_rows = 0;
  Index n_cols = 0;
  AlignedArray<Ptr> row_ptr;  // n_rows + 1 entries
  AlignedArray<Index> col;    // nnz entries
  AlignedArray<double> val;   // nnz entries

  std::int64_t nnz() const {
    return row_ptr.size() == 0 ? 0 : static_cast<std::int64_t>(row_ptr[row_ptr.size() - 1]);
  }
};

using CsrMatrix = CsrMatrixT<std::int64_t>;
using CsrMatrix32 = CsrMatrixT<std::int32_t>;

// Sorts each row, sums duplicate entries and keeps explicitly stored zeros.
// Throws std::runtime_error on an out-of-range index or inconsistent arrays.
CsrMatrix csr_from_coo(const Coo& coo);

CsrMatrix clone(const CsrMatrix& a);

bool fits_row_ptr32(const CsrMatrix& a);

// Moves col and val without copying. Throws if nnz does not fit in 32 bits.
CsrMatrix32 narrow_row_ptr(CsrMatrix&& a);

}  // namespace ssolve
