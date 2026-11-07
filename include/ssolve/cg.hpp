// Conjugate gradient, plain or with a Jacobi preconditioner.
#pragma once

#include <vector>

#include "ssolve/aligned_array.hpp"
#include "ssolve/backend.hpp"
#include "ssolve/matrix.hpp"
#include "ssolve/solver.hpp"

namespace ssolve {

// Solves A x = b for an SPD backend matrix, stopping when ‖r‖ < tol · ‖b‖ on the recursive residual,
// so Converged here says nothing about the true residual. Pass inv_diag = nullptr for plain CG.
// x holds the solution on return, and the initial guess on entry if options.use_initial_guess is set.
SolveResult cg(const Backend& backend, const double* inv_diag, const double* b, double* x, const SolveOptions& options);

// cg() from x = 0, also recording the recursive residual ‖r‖ / ‖b‖ before the first iteration and
// after each one, for plotting convergence. history is replaced, and stays empty for b = 0.
// It runs cg() itself through a backend wrapper, so cg() and the benchmarks are not changed by it.
// Throws if options.use_initial_guess is set.
SolveResult cg_with_history(const Backend& backend, const double* inv_diag, const double* b, double* x,
                            const SolveOptions& options, std::vector<double>& history);

// cg(), then an independent check of x against `a`, outside the solve timer. Converged then means the
// true residual met the tolerance too. `a` is the matrix the caller wants solved, not whatever the
// backend holds, so a wrong kernel or format cannot hide itself. A backend need not keep a CSR copy.
SolveResult verified_cg(const CsrMatrix& a, const Backend& backend, const double* inv_diag, const double* b, double* x,
                        const SolveOptions& options);

// 1 / aᵢᵢ per row. Requires finite positive diagonal entries and inverses.
AlignedArray<double> jacobi_inverse_diagonal(const CsrMatrix& a);

}  // namespace ssolve
