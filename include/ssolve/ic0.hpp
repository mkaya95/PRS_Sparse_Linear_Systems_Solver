// IC(0): incomplete Cholesky with no fill (Saad, Iterative Methods for Sparse Linear Systems, 2nd ed.,
// section 10.3). S A S + shift·I ≈ L Lᵀ, where L keeps exactly the pattern of the lower triangle of A
// and S = diag(1 / sqrt(aᵢᵢ)) scales the diagonal to 1. A pivot that is not positive restarts the
// factorization with a larger shift (Manteuffel's remedy): 1e-3 first, then doubled each time.
//
// Unlike ssolve/incomplete_cholesky.hpp, the factor is stored by rows with its columns in order, so
// both triangular solves walk memory front to back.
#pragma once

#include <cstdint>
#include <memory>

#include "ssolve/aligned_array.hpp"
#include "ssolve/matrix.hpp"
#include "ssolve/preconditioner.hpp"

namespace ssolve {

// L by rows: the entries left of the diagonal, columns ascending, and 1 / lᵢᵢ apart.
struct Ic0Factor {
  Index n = 0;
  AlignedArray<std::int64_t> row_ptr;  // n + 1
  AlignedArray<Index> col;
  AlignedArray<double> val;
  AlignedArray<double> inv_diag;  // 1 / lᵢᵢ
  AlignedArray<double> scale;     // S
  double shift = 0.0;
  int attempts = 1;  // factorizations run, counting the one that worked
};

// Reads the lower triangle of a, whose rows must have their columns in ascending order. Throws if a is
// not square, if a diagonal entry is missing or not finite and positive, or after 40 attempts.
Ic0Factor ic0(const CsrMatrix& a, double initial_shift = 1e-3);

// z = S L⁻ᵀ L⁻¹ S r.
void ic0_apply(const Ic0Factor& f, const double* r, double* z);

std::unique_ptr<Preconditioner> make_ic0_preconditioner(const CsrMatrix& a);

}  // namespace ssolve
