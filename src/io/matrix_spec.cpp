#include "ssolve/matrix_spec.hpp"

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <stdexcept>

#include "ssolve/generators.hpp"
#include "ssolve/mtx_reader.hpp"

namespace ssolve {

namespace {

bool parse_stencil(const std::string& name, Stencil& stencil) {
  if (name == "poisson2d") {
    stencil = Stencil::Poisson2D;
  } else if (name == "poisson3d") {
    stencil = Stencil::Poisson3D;
  } else if (name == "point27") {
    stencil = Stencil::Point27;
  } else {
    return false;
  }
  return true;
}

}  // namespace

CsrMatrix load_matrix_spec(const std::string& spec) {
  const std::size_t colon = spec.find(':');
  Stencil stencil{};
  if (colon == std::string::npos || !parse_stencil(spec.substr(0, colon), stencil)) {
    return read_matrix_market(spec);
  }

  const std::string size = spec.substr(colon + 1);
  char* end = nullptr;
  errno = 0;
  const long long N = std::strtoll(size.c_str(), &end, 10);
  if (size.empty() || *end != '\0' || errno == ERANGE) {
    throw std::runtime_error("bad grid size in matrix spec '" + spec + "'");
  }
  if (N <= 0 || N > 2147483647LL) throw std::runtime_error("grid side out of range in matrix spec '" + spec + "'");
  return make_stencil(stencil, static_cast<Index>(N));
}

AlignedArray<double> read_rhs(const std::string& path, Index n) {
  if (n < 0) throw std::runtime_error("RHS dimension must be nonnegative");
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open RHS file: " + path);
  AlignedArray<double> rhs(static_cast<std::size_t>(n));
  for (Index i = 0; i < n; ++i) {
    if (!(input >> rhs[i]) || !std::isfinite(rhs[i])) {
      throw std::runtime_error("RHS must contain exactly n finite values: " + path);
    }
  }
  std::string extra;
  if (input >> extra || !input.eof()) throw std::runtime_error("RHS has extra data or a read error: " + path);
  return rhs;
}

}  // namespace ssolve
