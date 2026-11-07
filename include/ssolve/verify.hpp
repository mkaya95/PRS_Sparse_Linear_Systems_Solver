// Independent checks on matrices and solutions.
//
// These are plain loops that share no code with any solver kernel, so a bug in
// a tuned kernel cannot hide itself here. Never call a backend from verify.cpp.
#pragma once

#include <string>

#include "ssolve/matrix.hpp"
#include "ssolve/solver.hpp"

namespace ssolve {

// Necessary CG checks, not a proof of positive definiteness. SPD provenance is required.
void validate_cg_matrix(const CsrMatrix& a);
std::string matrix_fingerprint(const CsrMatrix& a);

// y = A x
void spmv_reference(const CsrMatrix& a, const double* x, double* y);

// ‖b − Ax‖₂ / ‖b‖₂, computed without overflow or underflow in the squares.
// For b = 0 it returns the absolute residual ‖Ax‖₂. Returns NaN if any value is not finite.
double true_relative_residual(const CsrMatrix& a, const double* b, const double* x);

// Independent validation, outside the solve-only timer.
void verify_result(const CsrMatrix& a, const double* b, const double* x, const SolveOptions& options,
                   SolveResult& result);

// |xᵀAy − yᵀAx| / (‖x‖₂ ‖y‖₂ ‖A‖_F) for fixed pseudo-random x and y.
// About machine epsilon for a symmetric matrix. Throws for a non-square matrix.
double symmetry_defect(const CsrMatrix& a);

}  // namespace ssolve
