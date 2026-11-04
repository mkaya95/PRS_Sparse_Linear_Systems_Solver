#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include "ssolve/matrix.hpp"

namespace ssolve {

namespace {

void validate(const Coo& coo) {
  if (coo.n_rows < 0 || coo.n_cols < 0) {
    throw std::runtime_error("matrix dimensions must not be negative");
  }
  if (coo.row.size() != coo.val.size() || coo.col.size() != coo.val.size()) {
    throw std::runtime_error("COO row, col and val arrays must have the same length");
  }
  for (std::size_t k = 0; k < coo.val.size(); ++k) {
    if (coo.row[k] < 0 || coo.row[k] >= coo.n_rows) {
      throw std::runtime_error("row " + std::to_string(coo.row[k]) + " is out of range for " +
                               std::to_string(coo.n_rows) + " rows");
    }
    if (coo.col[k] < 0 || coo.col[k] >= coo.n_cols) {
      throw std::runtime_error("column " + std::to_string(coo.col[k]) + " is out of range for " +
                               std::to_string(coo.n_cols) + " columns");
    }
  }
}

// Sums neighbouring equal columns in each row and shrinks the arrays to fit.
void merge_duplicates(CsrMatrix& a) {
  std::int64_t write = 0;
  std::int64_t row_begin = 0;
  for (Index i = 0; i < a.n_rows; ++i) {
    const std::int64_t row_end = a.row_ptr[i + 1];
    const std::int64_t first_written = write;
    for (std::int64_t k = row_begin; k < row_end; ++k) {
      if (write > first_written && a.col[write - 1] == a.col[k]) {
        a.val[write - 1] += a.val[k];
      } else {
        a.col[write] = a.col[k];
        a.val[write] = a.val[k];
        ++write;
      }
    }
    row_begin = row_end;
    a.row_ptr[i + 1] = write;
  }
  if (static_cast<std::size_t>(write) == a.val.size()) return;

  AlignedArray<Index> col(write);
  AlignedArray<double> val(write);
  std::copy_n(a.col.data(), write, col.data());
  std::copy_n(a.val.data(), write, val.data());
  a.col = std::move(col);
  a.val = std::move(val);
}

}  // namespace

CsrMatrix csr_from_coo(const Coo& coo) {
  validate(coo);
  const std::size_t nnz = coo.val.size();

  // Two stable counting sorts, by column then by row, give sorted rows in linear
  // time. Files with tens of millions of entries make this worth it over std::sort.
  std::vector<std::int64_t> column_start(static_cast<std::size_t>(coo.n_cols) + 1, 0);
  for (std::size_t k = 0; k < nnz; ++k) ++column_start[coo.col[k] + 1];
  for (Index j = 0; j < coo.n_cols; ++j) column_start[j + 1] += column_start[j];
  std::vector<std::int64_t> by_column(nnz);
  for (std::size_t k = 0; k < nnz; ++k) by_column[column_start[coo.col[k]]++] = static_cast<std::int64_t>(k);

  CsrMatrix a;
  a.n_rows = coo.n_rows;
  a.n_cols = coo.n_cols;
  a.row_ptr = AlignedArray<std::int64_t>(static_cast<std::size_t>(coo.n_rows) + 1);
  a.col = AlignedArray<Index>(nnz);
  a.val = AlignedArray<double>(nnz);

  a.row_ptr.fill(0);
  for (std::size_t k = 0; k < nnz; ++k) ++a.row_ptr[coo.row[k] + 1];
  for (Index i = 0; i < coo.n_rows; ++i) a.row_ptr[i + 1] += a.row_ptr[i];

  std::vector<std::int64_t> next(a.row_ptr.begin(), a.row_ptr.end() - 1);
  for (const std::int64_t k : by_column) {
    const std::int64_t position = next[coo.row[k]]++;
    a.col[position] = coo.col[k];
    a.val[position] = coo.val[k];
  }

  merge_duplicates(a);
  return a;
}

CsrMatrix clone(const CsrMatrix& a) {
  CsrMatrix copy;
  copy.n_rows = a.n_rows;
  copy.n_cols = a.n_cols;
  copy.row_ptr = AlignedArray<std::int64_t>(a.row_ptr.size());
  copy.col = AlignedArray<Index>(a.col.size());
  copy.val = AlignedArray<double>(a.val.size());
  std::copy(a.row_ptr.begin(), a.row_ptr.end(), copy.row_ptr.begin());
  std::copy(a.col.begin(), a.col.end(), copy.col.begin());
  std::copy(a.val.begin(), a.val.end(), copy.val.begin());
  return copy;
}

bool fits_row_ptr32(const CsrMatrix& a) { return a.nnz() <= std::numeric_limits<std::int32_t>::max(); }

CsrMatrix32 narrow_row_ptr(CsrMatrix&& a) {
  if (!fits_row_ptr32(a)) {
    throw std::runtime_error("cannot use 32-bit row pointers: nnz is at least 2^31");
  }
  CsrMatrix32 narrow;
  narrow.n_rows = a.n_rows;
  narrow.n_cols = a.n_cols;
  narrow.row_ptr = AlignedArray<std::int32_t>(a.row_ptr.size());
  for (std::size_t i = 0; i < a.row_ptr.size(); ++i) {
    narrow.row_ptr[i] = static_cast<std::int32_t>(a.row_ptr[i]);
  }
  narrow.col = std::move(a.col);
  narrow.val = std::move(a.val);
  a.row_ptr = AlignedArray<std::int64_t>();
  a.n_rows = 0;
  a.n_cols = 0;
  return narrow;
}

}  // namespace ssolve
