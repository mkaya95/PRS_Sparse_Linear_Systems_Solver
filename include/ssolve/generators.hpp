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

// Smallest N whose matrix data, at 12 bytes per non-zero, reaches `bytes`.
Index n_for_target_bytes(Stencil stencil, std::int64_t bytes);

// b = A·1, so the exact solution is the vector of ones.
AlignedArray<double> rhs_for_unit_solution(const CsrMatrix& a);

// A fixed pseudo-random vector in [-1, 1). The generator's output is defined by the C++ standard
// and the conversion is ours, so every machine and standard library gets the same vector.
AlignedArray<double> random_solution(Index n);

// b = A·x for x = random_solution. The rows of A·1 can nearly cancel, leaving a b close to rounding
// noise that no solver can reach tol against, while this b stays well above it.
AlignedArray<double> rhs_for_random_solution(const CsrMatrix& a);

}  // namespace ssolve
