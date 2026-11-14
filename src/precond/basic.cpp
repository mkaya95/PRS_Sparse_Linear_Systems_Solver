// No preconditioner and Jacobi, with the backend's fused kernels: the calls cg() made before the
// preconditioner interface existed, in the same order, so iteration counts and speed are unchanged.
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "ssolve/cg.hpp"
#include "ssolve/ic0.hpp"
#include "ssolve/incomplete_cholesky.hpp"
#include "ssolve/preconditioner.hpp"
#include "ssolve/sgs.hpp"

namespace ssolve {

namespace {

class IdentityPreconditioner final : public Preconditioner {
 public:
  double start(const Backend& backend, const double* r, double rr, double* p) override {
    backend.copy(r, p);
    return rr;
  }

  RrRz update(const Backend& backend, double a, const double* p, const double* q, double* x, double* r) override {
    const double rr = backend.cg_update(a, p, q, x, r);
    return {rr, rr};
  }

  void direction(const Backend& backend, const double* r, double beta, double* p) override {
    backend.xpby(r, beta, p);
  }
};

// z = inv_diag · r is folded into the residual update and the direction, never stored.
class JacobiPreconditioner final : public Preconditioner {
 public:
  explicit JacobiPreconditioner(AlignedArray<double> owned) : owned_(std::move(owned)), inv_diag_(owned_.data()) {}
  explicit JacobiPreconditioner(const double* inv_diag) : inv_diag_(inv_diag) {}

  double start(const Backend& backend, const double* r, double, double* p) override {
    backend.set_zero(p);
    backend.jacobi_direction(inv_diag_, r, 0.0, p);
    return backend.dot(r, p);
  }

  RrRz update(const Backend& backend, double a, const double* p, const double* q, double* x, double* r) override {
    return backend.jacobi_update(a, p, q, x, r, inv_diag_);
  }

  void direction(const Backend& backend, const double* r, double beta, double* p) override {
    backend.jacobi_direction(inv_diag_, r, beta, p);
  }

 private:
  AlignedArray<double> owned_;
  const double* inv_diag_;
};

}  // namespace

std::unique_ptr<Preconditioner> make_identity_preconditioner() { return std::make_unique<IdentityPreconditioner>(); }

std::unique_ptr<Preconditioner> make_jacobi_preconditioner(const CsrMatrix& a) {
  return std::make_unique<JacobiPreconditioner>(jacobi_inverse_diagonal(a));
}

std::unique_ptr<Preconditioner> make_jacobi_preconditioner(const double* inv_diag) {
  return std::make_unique<JacobiPreconditioner>(inv_diag);
}

std::unique_ptr<Preconditioner> make_preconditioner(Precond kind, const CsrMatrix& a) {
  switch (kind) {
    case Precond::None:
      return make_identity_preconditioner();
    case Precond::Jacobi:
      return make_jacobi_preconditioner(a);
    case Precond::SymGaussSeidel:
      return make_sgs_preconditioner(a);
    case Precond::IncompleteCholesky:
      return make_ic_preconditioner(a);
    case Precond::IncompleteCholeskyFixed:
      return make_ic_preconditioner(a, EigenBugs::Fixed);
    case Precond::IncompleteCholeskyZero:
      return make_ic0_preconditioner(a);
    default:
      throw std::runtime_error(std::string("no preconditioner of kind ") + to_string(kind));
  }
}

}  // namespace ssolve
