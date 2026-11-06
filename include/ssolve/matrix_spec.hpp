// Turns a command-line matrix description into a matrix.
#pragma once

#include <string>

#include "ssolve/matrix.hpp"

namespace ssolve {

// A spec is one of:
//   poisson2d:N, poisson3d:N, point27:N   N grid points per side
//   poisson3d:llc=4                       matrix data of 4x the last-level cache
//   path/to/file.mtx                      a Matrix Market file
CsrMatrix load_matrix_spec(const std::string& spec);

// Short name for results: generated specs as written, files by base name without extension.
std::string matrix_display_name(const std::string& spec);

// Plain whitespace-separated FP64 values, exactly one value per matrix row.
AlignedArray<double> read_rhs(const std::string& path, Index n);

}  // namespace ssolve
