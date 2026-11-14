#include "ssolve/compare.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace ssolve {

namespace {

constexpr int kMaxGrid = 64;

// ‖v‖₂ without overflow or underflow in the squares, as verify.cpp computes it. NaN if any entry is not finite.
template <class Entry>
double scaled_norm(std::size_t n, Entry entry) {
  double largest = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double value = entry(i);
    if (!std::isfinite(value)) return std::numeric_limits<double>::quiet_NaN();
    largest = std::fmax(largest, std::fabs(value));
  }
  if (largest == 0.0) return 0.0;
  double sum = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double ratio = entry(i) / largest;
    sum += ratio * ratio;
  }
  return largest * std::sqrt(sum);
}

}  // namespace

int difference_grid(std::int64_t n) {
  int g = 1;
  while (g < kMaxGrid && static_cast<std::int64_t>(g) * g < n) ++g;
  return g;
}

VectorDifference compare_vectors(const std::string& a_name, const double* a, const std::string& b_name,
                                 const double* b, std::size_t n) {
  VectorDifference d;
  d.a = a_name;
  d.b = b_name;
  const std::int64_t count = static_cast<std::int64_t>(n);
  const std::int64_t g = difference_grid(count);
  const std::int64_t cells = g * g;
  d.cells.assign(static_cast<std::size_t>(cells), std::numeric_limits<double>::quiet_NaN());

  bool finite = true;
  for (std::int64_t i = 0; i < count; ++i) {
    const double diff = std::fabs(a[i] - b[i]);
    const std::size_t cell = static_cast<std::size_t>(count >= cells ? i * cells / count : i);
    double& slot = d.cells[cell];
    if (a[i] == b[i]) ++d.equal;
    if (!std::isfinite(diff)) {
      finite = false;
      slot = -1.0;
    } else if (slot != -1.0) {
      slot = std::isnan(slot) ? diff : std::fmax(slot, diff);
      d.max_abs = std::fmax(d.max_abs, diff);
    }
  }
  if (!finite) {
    d.max_abs = std::numeric_limits<double>::quiet_NaN();
    d.rel_norm = std::numeric_limits<double>::quiet_NaN();
    return d;
  }
  const double diff_norm = scaled_norm(n, [&](std::size_t i) { return a[i] - b[i]; });
  const double b_norm = scaled_norm(n, [&](std::size_t i) { return b[i]; });
  d.rel_norm = b_norm == 0.0 ? diff_norm : diff_norm / b_norm;
  return d;
}

std::string json_string(const std::string& text) {
  std::string out = "\"";
  for (const char c : text) {
    if (c == '"' || c == '\\') {
      out += '\\';
      out += c;
    } else if (static_cast<unsigned char>(c) < 0x20) {
      out += ' ';
    } else {
      out += c;
    }
  }
  return out + "\"";
}

std::string json_number(double value, int digits) {
  if (!std::isfinite(value)) return "null";
  char text[40];
  std::snprintf(text, sizeof text, "%.*g", digits, value);
  return text;
}

std::string solutions_json(const std::string& matrix, std::int64_t n, const std::string& problem, double x_scale,
                           const std::vector<VectorDifference>& diffs,
                           const std::vector<ResidualHistory>& histories) {
  std::string out = "{\"matrix\":" + json_string(matrix) + ",\"n\":" + std::to_string(n) +
                    ",\"grid\":" + std::to_string(difference_grid(n)) + ",\"problem\":" + json_string(problem) +
                    ",\"x_scale\":" + json_number(x_scale) + ",\"comparisons\":[";
  for (std::size_t k = 0; k < diffs.size(); ++k) {
    const VectorDifference& d = diffs[k];
    out += (k ? ",{" : "{");
    out += "\"a\":" + json_string(d.a) + ",\"b\":" + json_string(d.b) + ",\"max_abs\":" + json_number(d.max_abs, 3) +
           ",\"rel_norm\":" + json_number(d.rel_norm, 3) + ",\"equal\":" + std::to_string(d.equal) + ",\"cells\":[";
    for (std::size_t c = 0; c < d.cells.size(); ++c) {
      if (c) out += ',';
      out += json_number(d.cells[c], 3);
    }
    out += "]}";
  }
  out += "],\"histories\":[";
  for (std::size_t k = 0; k < histories.size(); ++k) {
    out += (k ? ",{" : "{");
    out += "\"solver\":" + json_string(histories[k].solver) + ",\"residuals\":[";
    for (std::size_t i = 0; i < histories[k].residuals.size(); ++i) {
      if (i) out += ',';
      out += json_number(histories[k].residuals[i], 4);
    }
    out += "]}";
  }
  return out + "]}\n";
}

void write_file_atomically(const std::string& path, const std::string& text) {
  const std::string part = path + ".part";
  {
    std::ofstream out(part);
    if (!out) throw std::runtime_error("cannot write " + part);
    out << text;
    if (!out) throw std::runtime_error("failed while writing " + part);
  }
  if (std::rename(part.c_str(), path.c_str()) != 0) {
    std::remove(part.c_str());
    throw std::runtime_error("cannot move the file into place at " + path);
  }
}

}  // namespace ssolve
