#include "ssolve/matrix_spec.hpp"

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <stdexcept>

#include "ssolve/generators.hpp"
#include "ssolve/manifest.hpp"
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

std::int64_t last_level_cache_bytes() {
  const std::int64_t bytes = collect_manifest().l3_bytes;
  return bytes > 0 ? bytes : std::int64_t{8} * 1024 * 1024;  // when the size cannot be read
}

}  // namespace

CsrMatrix load_matrix_spec(const std::string& spec) {
  const std::size_t colon = spec.find(':');
  Stencil stencil{};
  if (colon == std::string::npos || !parse_stencil(spec.substr(0, colon), stencil)) {
    return read_matrix_market(spec);
  }

  const std::string size = spec.substr(colon + 1);
  if (size.rfind("llc=", 0) == 0) {
    const std::string factor_text = size.substr(4);
    char* end = nullptr;
    const double factor = std::strtod(factor_text.c_str(), &end);
    if (factor_text.empty() || *end != '\0' || !(factor > 0.0) || !std::isfinite(factor)) {
      throw std::runtime_error("bad llc factor in matrix spec '" + spec + "': it must be a positive number");
    }
    const double requested = factor * static_cast<double>(last_level_cache_bytes());
    if (!std::isfinite(requested) || requested >= static_cast<double>(std::numeric_limits<std::int64_t>::max())) {
      throw std::runtime_error("llc size is out of range");
    }
    const auto bytes = static_cast<std::int64_t>(requested);
    return make_stencil(stencil, n_for_target_bytes(stencil, bytes));
  }

  char* end = nullptr;
  errno = 0;
  const long long N = std::strtoll(size.c_str(), &end, 10);
  if (size.empty() || *end != '\0' || errno == ERANGE) {
    throw std::runtime_error("bad grid size in matrix spec '" + spec + "'");
  }
  if (N <= 0 || N > 2147483647LL) throw std::runtime_error("grid side out of range in matrix spec '" + spec + "'");
  return make_stencil(stencil, static_cast<Index>(N));
}

std::string matrix_display_name(const std::string& spec) {
  const std::size_t colon = spec.find(':');
  Stencil stencil{};
  if (colon != std::string::npos && parse_stencil(spec.substr(0, colon), stencil)) return spec;
  const std::size_t slash = spec.find_last_of("/\\");
  std::string name = slash == std::string::npos ? spec : spec.substr(slash + 1);
  const std::size_t dot = name.rfind(".mtx");
  if (dot != std::string::npos && dot + 4 == name.size()) name.resize(dot);
  return name;
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
