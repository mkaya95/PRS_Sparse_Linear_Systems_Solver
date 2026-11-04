// The narrow interface between solvers and hardware.
//
// CG is written once against this class, so an OpenMP or CUDA backend only reimplements
// these kernels. Virtual calls cost nothing measurable here, since each one runs a few
// times per iteration and processes the whole matrix or vector.
#pragma once

#include <memory>

#include "ssolve/matrix.hpp"

namespace ssolve {

struct RrRz {
  double rr;  // r·r
  double rz;  // r·z, where z = D⁻¹r
};

class Backend {
 public:
  virtual ~Backend() = default;

  virtual Index n() const = 0;

  // Basic operations
  virtual void set_zero(double* x) const = 0;
  virtual void copy(const double* source, double* destination) const = 0;
  virtual double dot(const double* x, const double* y) const = 0;
  virtual void spmv(const double* x, double* y) const = 0;         // y = A x
  virtual void xpby(const double* x, double beta, double* y) const = 0;  // y = x + beta y

  // Fused operations: one sweep instead of several, so each vector crosses the memory bus once.
  virtual double spmv_dot(const double* x, double* y) const = 0;  // y = A x, returns x·y

  // x += a p, r −= a q, returns r·r
  virtual double cg_update(double a, const double* p, const double* q, double* x, double* r) const = 0;

  // The same, also returning r·z for the Jacobi preconditioner z = inv_diag · r
  virtual RrRz jacobi_update(double a, const double* p, const double* q, double* x, double* r,
                             const double* inv_diag) const = 0;

  // p = inv_diag · r + beta p, without ever storing z
  virtual void jacobi_direction(const double* inv_diag, const double* r, double beta, double* p) const = 0;
};

// The sequential backend takes ownership of the matrix. It uses 32-bit row pointers when
// they fit, which allow_narrow_row_ptr = false turns off for the optimization ladder.
// Throws if the matrix is not square.
std::unique_ptr<Backend> make_seq_backend(CsrMatrix a, bool allow_narrow_row_ptr = true);

}  // namespace ssolve
