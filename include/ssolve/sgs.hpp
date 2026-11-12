// Symmetric Gauss-Seidel: z = M⁻¹ r for M = (D + L) D⁻¹ (D + U), one forward and one backward
// sweep from z = 0. M is symmetric positive definite when A is, so CG can use it.
#pragma once

#include <memory>

#include "ssolve/aligned_array.hpp"
#include "ssolve/matrix.hpp"
#include "ssolve/preconditioner.hpp"

namespace ssolve {

class SymGaussSeidel {
 public:
  // Rows must be sorted (ours are), each with a finite positive diagonal; throws otherwise.
  explicit SymGaussSeidel(CsrMatrix a);

  void apply(const double* r, double* z) const;

 private:
  CsrMatrix a_;
  AlignedArray<double> diag_;
};

// Copies a.
std::unique_ptr<Preconditioner> make_sgs_preconditioner(const CsrMatrix& a);

}  // namespace ssolve
