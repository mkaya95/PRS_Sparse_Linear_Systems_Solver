#include "matrix_spec.hpp"

#include <cstdlib>
#include <stdexcept>

#include "ssolve/generators.hpp"
#include "ssolve/manifest.hpp"
#include "ssolve/mtx_reader.hpp"

using namespace ssolve;

namespace ssolve_bench {

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
    if (factor_text.empty() || *end != '\0' || !(factor > 0.0)) {
      throw std::runtime_error("bad llc factor in matrix spec '" + spec + "': it must be a positive number");
    }
    const auto bytes = static_cast<std::int64_t>(factor * static_cast<double>(last_level_cache_bytes()));
    return make_stencil(stencil, n_for_target_bytes(stencil, bytes));
  }

  char* end = nullptr;
  const long long N = std::strtoll(size.c_str(), &end, 10);
  if (size.empty() || *end != '\0') throw std::runtime_error("bad grid size in matrix spec '" + spec + "'");
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

}  // namespace ssolve_bench
