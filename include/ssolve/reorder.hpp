// Reordering the rows and columns of a symmetric matrix, to bring its non-zeros close to the diagonal.
//
// An order lists the old index of each new row: row k of the reordered matrix is row order[k] of A.
#pragma once

#include <vector>

#include "ssolve/matrix.hpp"

namespace ssolve {

// Reverse Cuthill-McKee (Cuthill and McKee 1969; George and Liu 1981), from SciPy's
// reverse_cuthill_mckee (scipy/sparse/csgraph/_reordering.pyx). a must be square with a symmetric
// pattern. Rows of equal degree keep their original order.
std::vector<Index> reverse_cuthill_mckee(const CsrMatrix& a);

// P A Pᵀ: entry (order[k], order[l]) of a becomes entry (k, l). Throws unless a is square and order
// is a permutation of its rows.
CsrMatrix permute_symmetric(const CsrMatrix& a, const std::vector<Index>& order);

// to[k] = from[order[k]]: a vector of the original system, in the reordered system's order.
void to_new_order(const std::vector<Index>& order, const double* from, double* to);

// to[order[k]] = from[k]: the reverse.
void to_original_order(const std::vector<Index>& order, const double* from, double* to);

// The largest |i - j| over the stored entries.
Index bandwidth(const CsrMatrix& a);

}  // namespace ssolve
