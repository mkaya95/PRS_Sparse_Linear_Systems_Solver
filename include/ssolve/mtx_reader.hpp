// Matrix Market coordinate files, as distributed by the SuiteSparse collection.
#pragma once

#include <string>
#include <string_view>

#include "ssolve/matrix.hpp"

namespace ssolve {

// Supports real, integer and pattern fields with general, symmetric and
// skew-symmetric structure. Symmetric entries are mirrored so the result holds
// the full matrix. Errors are std::runtime_error, formatted as name:line: message.
CsrMatrix read_matrix_market(const std::string& path);
CsrMatrix parse_matrix_market(std::string_view text, const std::string& name_for_errors);

}  // namespace ssolve
