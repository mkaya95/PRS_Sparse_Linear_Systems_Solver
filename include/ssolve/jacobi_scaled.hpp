// Conjugate gradient with the Jacobi preconditioner, run as plain CG on the symmetrically scaled system
//
//   Â x̂ = b̂,   Â = S A S,   S = D^(-1/2),   b̂ = S b,   x = S x̂,
//
// which gives the same iterates as PCG with M = D (Saad, Iterative Methods for Sparse Linear Systems,
// 2nd ed., section 9.2.1, split preconditioning). What it saves per iteration, against cg() with
// make_jacobi_preconditioner:
//   - Â has 1 on its diagonal, which is not stored (make_*_backend_unit_diagonal): n entries fewer to
//     read in every product, a seventh of thermal2.
//   - z = r̂, so the new direction p = r̂ + β p no longer reads the inverse diagonal.
// The stopping test stays the caller's: ‖r‖² = Σ dᵢ r̂ᵢ² against ‖b‖², by Preconditioner::norm_squared.
#pragma once

#include <memory>

#include "ssolve/aligned_array.hpp"
#include "ssolve/backend.hpp"
#include "ssolve/matrix.hpp"
#include "ssolve/preconditioner.hpp"
#include "ssolve/solver.hpp"

namespace ssolve {

struct JacobiScaling {
  CsrMatrix off_diagonal;         // Â without its diagonal, which is 1
  AlignedArray<double> diagonal;  // D: aᵢᵢ
  AlignedArray<double> scale;     // S: 1 / sqrt(aᵢᵢ)
};

// Throws, as jacobi_inverse_diagonal does, unless every diagonal entry is finite and positive.
JacobiScaling jacobi_scaling(const CsrMatrix& a);

// CG's preconditioner for Â: the identity, but with r·r and the stopping norm weighted by D, so that
// they are those of the unscaled residual. Uses `diagonal` without copying it.
std::unique_ptr<Preconditioner> make_scaled_jacobi_preconditioner(const double* diagonal);

// Solves A x = b given a backend of Â (with a unit diagonal) and the scaling: b̂ = S b, CG, x = S x̂.
SolveResult cg_scaled_jacobi(const Backend& backend, const JacobiScaling& scaling, const double* b, double* x,
                             const SolveOptions& options);

}  // namespace ssolve
