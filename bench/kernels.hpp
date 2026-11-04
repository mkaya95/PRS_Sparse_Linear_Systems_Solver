// Individual kernels bound to a matrix, for per-kernel timing against Eigen.
#pragma once

#include <functional>
#include <string>
#include <vector>

#include "ssolve/matrix.hpp"

namespace ssolve_bench {

struct KernelSet {
  std::function<void(const double* x, double* y)> spmv;                   // y = A x
  std::function<double(const double* x, const double* y)> dot;            // x·y
  std::function<void(double alpha, const double* x, double* y)> axpy;     // y += alpha x
  std::function<double(const double* x)> norm2;                           // x·x
  std::function<void(const double* x, double beta, double* y)> xpby;      // y = x + beta y
};

// The matrix must outlive the returned set, unless the set copies it, as Eigen's does.
KernelSet eigen_kernels(const ssolve::CsrMatrix& a);

std::vector<std::string> ladder_kernel_names();  // L0, L1, ...

// Throws if the name is unknown.
KernelSet ladder_kernels(const std::string& name, const ssolve::CsrMatrix& a);

}  // namespace ssolve_bench
