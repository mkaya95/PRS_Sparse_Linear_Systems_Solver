// Incomplete Cholesky with limited memory (Lin and Moré 1999), as in Eigen's IncompleteCholesky with
// natural ordering: S A S + shift·I ≈ L Lᵀ, keeping in each column of L only as many entries as the
// lower triangle of A has there, the largest ones.
#pragma once

#include <cstdint>
#include <memory>

#include "ssolve/aligned_array.hpp"
#include "ssolve/matrix.hpp"
#include "ssolve/preconditioner.hpp"

namespace ssolve {

// L by compressed columns, the diagonal first in each column; the other rows in any order.
struct CholeskyFactor {
  Index n = 0;
  AlignedArray<std::int64_t> col_ptr;  // n + 1
  AlignedArray<Index> row;
  AlignedArray<double> val;
  AlignedArray<double> scale;  // S
  double shift = 0.0;
};

// Eigen's factorization has two bugs (see incomplete_cholesky.cpp). Copied keeps them, so the factor is
// Eigen's, operation for operation (the two then differ only by rounding); Fixed corrects both, which
// gives a better preconditioner.
enum class EigenBugs { Copied, Fixed };

// Reads the upper triangle of a's rows as the lower triangle's columns, since a is symmetric.
// Throws "incomplete Cholesky failed after 10 shifts" where Eigen reports NumericalIssue.
CholeskyFactor incomplete_cholesky(const CsrMatrix& a, double initial_shift = 1e-3,
                                   EigenBugs bugs = EigenBugs::Copied);

// z = S L⁻ᵀ L⁻¹ S r.
void incomplete_cholesky_apply(const CholeskyFactor& f, const double* r, double* z);

std::unique_ptr<Preconditioner> make_ic_preconditioner(const CsrMatrix& a, EigenBugs bugs = EigenBugs::Copied);

}  // namespace ssolve
