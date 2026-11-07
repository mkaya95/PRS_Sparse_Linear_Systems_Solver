// Symmetric positive definite test matrices from finite-difference stencils.
#pragma once

#include <cstdint>

#include "ssolve/matrix.hpp"

namespace ssolve {

enum class Stencil {
  Poisson2D,  // 5-point, diagonal 4, n = N²
  Poisson3D,  // 7-point, diagonal 6, n = N³
  Point27     // 27-point as in HPCG, diagonal 26, n = N³
};

// N is the number of grid points per side. Throws if N is not positive or n exceeds Index.
CsrMatrix make_stencil(Stencil stencil, Index N);

std::int64_t stencil_nnz(Stencil stencil, std::int64_t N);

// b = A·1, so the exact solution is the vector of ones.
AlignedArray<double> rhs_for_unit_solution(const CsrMatrix& a);

}  // namespace ssolve
