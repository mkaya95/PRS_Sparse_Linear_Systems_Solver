// Conjugate gradient, plain or with a Jacobi preconditioner.
#pragma once

#include "ssolve/aligned_array.hpp"
#include "ssolve/backend.hpp"
#include "ssolve/matrix.hpp"
#include "ssolve/solver.hpp"

namespace ssolve {

// Solves A x = b for an SPD backend matrix. Recursive convergence is independently
// verified before returning Converged. Pass inv_diag = nullptr for plain CG. x holds the solution on return, and the
// initial guess on entry if options.use_initial_guess is set.
SolveResult cg(const Backend& backend, const double* inv_diag, const double* b, double* x, const SolveOptions& options);

// 1 / aᵢᵢ per row. Requires finite positive diagonal entries and inverses.
AlignedArray<double> jacobi_inverse_diagonal(const CsrMatrix& a);

}  // namespace ssolve
