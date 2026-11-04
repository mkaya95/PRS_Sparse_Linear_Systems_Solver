// Conjugate gradient, plain or with a Jacobi preconditioner.
#pragma once

#include "ssolve/aligned_array.hpp"
#include "ssolve/backend.hpp"
#include "ssolve/matrix.hpp"
#include "ssolve/solver.hpp"

namespace ssolve {

// Solves A x = b for the backend's matrix, stopping when ‖r‖ < tol · ‖b‖ on the recursive
// residual. Pass inv_diag = nullptr for plain CG. x holds the solution on return, and the
// initial guess on entry if options.use_initial_guess is set.
SolveResult cg(const Backend& backend, const double* inv_diag, const double* b, double* x, const SolveOptions& options);

// 1 / aᵢᵢ per row. A zero or missing diagonal entry gives 1, as Eigen does, so that
// iteration counts stay comparable and no infinity reaches the solver.
AlignedArray<double> jacobi_inverse_diagonal(const CsrMatrix& a);

}  // namespace ssolve
