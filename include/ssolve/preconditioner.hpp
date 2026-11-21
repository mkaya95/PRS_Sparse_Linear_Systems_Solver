// A preconditioner as the CG loop sees it: three jobs, each free to merge its work into the vector updates.
//
// Identity and Jacobi use the backend's fused kernels, so z = M⁻¹ r is never stored. A preconditioner
// that cannot merge (Gauss-Seidel, incomplete Cholesky) keeps z itself and uses the plain kernels.
#pragma once

#include <memory>

#include "ssolve/backend.hpp"
#include "ssolve/matrix.hpp"
#include "ssolve/solver.hpp"

namespace ssolve {

class Preconditioner {
 public:
  virtual ~Preconditioner() = default;

  // p = z for z = M⁻¹ r. rr is r·r, already computed by the caller. Returns r·z.
  virtual double start(const Backend& backend, const double* r, double rr, double* p) = 0;

  // x += a p and r −= a q, then z = M⁻¹ r for the new r. Returns {r·r, r·z}.
  virtual RrRz update(const Backend& backend, double a, const double* p, const double* q, double* x, double* r) = 0;

  // p = z + beta p, with z from the last start() or update().
  virtual void direction(const Backend& backend, const double* r, double beta, double* p) = 0;

  // v·v in the norm CG stops on, ‖r‖ < tol·‖b‖. A preconditioner that solves a rescaled system
  // (jacobi_scaled.hpp) overrides it, so the test stays on the caller's own r and b.
  virtual double norm_squared(const Backend& backend, const double* v) { return backend.dot(v, v); }
};

std::unique_ptr<Preconditioner> make_identity_preconditioner();

// Keeps its own copy of 1 / aᵢᵢ. Throws as jacobi_inverse_diagonal does.
std::unique_ptr<Preconditioner> make_jacobi_preconditioner(const CsrMatrix& a);

// Uses inv_diag without copying it; inv_diag must outlive the preconditioner.
std::unique_ptr<Preconditioner> make_jacobi_preconditioner(const double* inv_diag);

// The preconditioner of the given kind for a. Throws for a kind it does not know.
std::unique_ptr<Preconditioner> make_preconditioner(Precond kind, const CsrMatrix& a);

}  // namespace ssolve
