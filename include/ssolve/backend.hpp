// The narrow interface between solvers and hardware.
//
// CG is written once against this class, so an OpenMP or CUDA backend only reimplements
// these kernels. Virtual calls cost nothing measurable here, since each one runs a few
// times per iteration and processes the whole matrix or vector.
#pragma once

#include <memory>

#include "ssolve/matrix.hpp"
#include "ssolve/sell.hpp"

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

// The sequential SELL-C-σ backend, built from a copy of a; it keeps no CSR. Throws if the matrix is
// not square, if C is not 4, 8, 16 or 32, or if σ is not 1 or a multiple of C.
std::unique_ptr<Backend> make_sell_backend(const CsrMatrix& a, int chunk = kSellChunk, Index sigma = kSellSigma);

// The same from a SELL matrix already built, for a caller that also wants its layout. Throws if not square.
std::unique_ptr<Backend> make_sell_backend(SellMatrix s);

// Backends of I + off_diagonal: the diagonal is 1 and not stored, so each product reads n entries fewer.
// For the Jacobi-scaled system of jacobi_scaled.hpp. off_diagonal must have no diagonal entries.
// Throws if off_diagonal stores a diagonal entry, which would be counted twice.
std::unique_ptr<Backend> make_seq_backend_unit_diagonal(CsrMatrix off_diagonal);
std::unique_ptr<Backend> make_sell_backend_unit_diagonal(const CsrMatrix& off_diagonal, int chunk = kSellChunk,
                                                         Index sigma = kSellSigma);

// Throws if a stores any entry on its diagonal.
void require_no_diagonal(const CsrMatrix& a);

}  // namespace ssolve
