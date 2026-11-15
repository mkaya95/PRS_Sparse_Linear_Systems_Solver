// Sequential vector kernels shared by every sequential storage format.
//
// Vector loops need `omp simd`, because GCC does not otherwise vectorize floating-point sums,
// and fused kernels save a separate pass over each vector. A format's backend derives from this
// class and adds only n(), spmv and spmv_dot.
#pragma once

#include <cstddef>

#include "ssolve/backend.hpp"
#include "ssolve/kernels_common.hpp"

namespace ssolve {

class SeqVectorBackend : public Backend {
 public:
  explicit SeqVectorBackend(std::size_t n) : size_(n) {}

  void set_zero(double* SSOLVE_RESTRICT x) const override {
#pragma omp simd
    for (std::size_t i = 0; i < size_; ++i) x[i] = 0.0;
  }

  void copy(const double* SSOLVE_RESTRICT source, double* SSOLVE_RESTRICT destination) const override {
#pragma omp simd
    for (std::size_t i = 0; i < size_; ++i) destination[i] = source[i];
  }

  double dot(const double* SSOLVE_RESTRICT x, const double* SSOLVE_RESTRICT y) const override {
    double sum = 0.0;
#pragma omp simd reduction(+ : sum)
    for (std::size_t i = 0; i < size_; ++i) sum += x[i] * y[i];
    return sum;
  }

  void xpby(const double* SSOLVE_RESTRICT x, double beta, double* SSOLVE_RESTRICT y) const override {
#pragma omp simd
    for (std::size_t i = 0; i < size_; ++i) y[i] = x[i] + beta * y[i];
  }

  double cg_update(double a, const double* SSOLVE_RESTRICT p, const double* SSOLVE_RESTRICT q, double* SSOLVE_RESTRICT x,
                   double* SSOLVE_RESTRICT r) const override {
    double rr = 0.0;
#pragma omp simd reduction(+ : rr)
    for (std::size_t i = 0; i < size_; ++i) {
      x[i] += a * p[i];
      r[i] -= a * q[i];
      rr += r[i] * r[i];
    }
    return rr;
  }

  RrRz jacobi_update(double a, const double* SSOLVE_RESTRICT p, const double* SSOLVE_RESTRICT q,
                     double* SSOLVE_RESTRICT x, double* SSOLVE_RESTRICT r,
                     const double* SSOLVE_RESTRICT inv_diag) const override {
    double rr = 0.0, rz = 0.0;
#pragma omp simd reduction(+ : rr, rz)
    for (std::size_t i = 0; i < size_; ++i) {
      x[i] += a * p[i];
      r[i] -= a * q[i];
      const double r_squared = r[i] * r[i];
      rr += r_squared;
      rz += r_squared * inv_diag[i];
    }
    return {rr, rz};
  }

  void jacobi_direction(const double* SSOLVE_RESTRICT inv_diag, const double* SSOLVE_RESTRICT r, double beta,
                        double* SSOLVE_RESTRICT p) const override {
#pragma omp simd
    for (std::size_t i = 0; i < size_; ++i) p[i] = inv_diag[i] * r[i] + beta * p[i];
  }

 protected:
  std::size_t size_;
};

}  // namespace ssolve
