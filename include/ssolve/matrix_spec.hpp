// Turns a command-line matrix description into a matrix.
#pragma once

#include <string>

#include "ssolve/matrix.hpp"

namespace ssolve {

// A spec is one of:
//   poisson2d:N, poisson3d:N, point27:N   N grid points per side
//   path/to/file.mtx                      a Matrix Market file
CsrMatrix load_matrix_spec(const std::string& spec);

// Plain whitespace-separated FP64 values, exactly one value per matrix row.
AlignedArray<double> read_rhs(const std::string& path, Index n);

}  // namespace ssolve
