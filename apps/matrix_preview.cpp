// Draws a matrix's non-zero pattern for the browser page, as JSON.
//
// Usage: matrix_preview --matrix SPEC --out FILE [--grid 160] [--zoom 40]
//
// The real matrices are too big to draw entry by entry (up to 78 million non-zeros), so the
// pattern is binned: the matrix is cut into grid x grid blocks and each block counts its
// non-zeros. The zoom lists the top-left zoom x zoom corner entry by entry, with values.
// The file is written whole and then renamed into place, so a stopped run leaves no half file.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cli.hpp"
#include "ssolve/matrix_spec.hpp"

using namespace ssolve;

namespace {

std::string json_number(double value) {
  if (!std::isfinite(value)) return "null";
  char text[32];
  std::snprintf(text, sizeof text, "%.17g", value);
  return text;
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

}  // namespace

int main(int argc, char** argv) {
  try {
    const ssolve_cli::Args args(argc, argv, {"matrix", "out", "grid", "zoom"});
    const std::string spec = args.required("matrix");
    const std::string out_path = args.required("out");
    const int grid_wanted = args.get_count("grid", 160, 1);
    const int zoom_wanted = args.get_count("zoom", 40, 1);
    if (grid_wanted > 1024 || zoom_wanted > 200) throw std::runtime_error("--grid is at most 1024 and --zoom at most 200");

    const CsrMatrix a = load_matrix_spec(spec);
    const std::int64_t n = a.n_rows;
    if (n <= 0) throw std::runtime_error("the matrix is empty");
    const std::int64_t grid = std::min<std::int64_t>(grid_wanted, n);
    const std::int64_t zoom = std::min<std::int64_t>(zoom_wanted, n);

    std::vector<std::int64_t> cells(static_cast<std::size_t>(grid * grid), 0);
    std::int64_t row_min = a.nnz(), row_max = 0, bandwidth = 0;
    std::string zoom_entries;
    for (std::int64_t i = 0; i < n; ++i) {
      const std::int64_t begin = a.row_ptr[i], end = a.row_ptr[i + 1];
      row_min = std::min(row_min, end - begin);
      row_max = std::max(row_max, end - begin);
      const std::int64_t cell_row = i * grid / n;
      for (std::int64_t k = begin; k < end; ++k) {
        const std::int64_t j = a.col[k];
        bandwidth = std::max(bandwidth, std::abs(i - j));
        ++cells[static_cast<std::size_t>(cell_row * grid + j * grid / n)];
        if (i < zoom && j < zoom) {
          if (!zoom_entries.empty()) zoom_entries += ',';
          zoom_entries += "[" + std::to_string(i) + "," + std::to_string(j) + "," + json_number(a.val[k]) + "]";
        }
      }
    }

    const std::string part = out_path + ".part";
    {
      std::ofstream out(part);
      if (!out) throw std::runtime_error("cannot write " + part);
      out << "{\"matrix\":" << json_string(matrix_display_name(spec)) << ",\"n\":" << n << ",\"nnz\":" << a.nnz()
          << ",\"row_nnz\":{\"min\":" << row_min << ",\"mean\":" << json_number(static_cast<double>(a.nnz()) / n)
          << ",\"max\":" << row_max << "},\"bandwidth\":" << bandwidth << ",\"grid\":" << grid << ",\"cells\":[";
      for (std::size_t c = 0; c < cells.size(); ++c) out << (c ? "," : "") << cells[c];
      out << "],\"zoom\":{\"size\":" << zoom << ",\"entries\":[" << zoom_entries << "]}}\n";
      if (!out) throw std::runtime_error("failed while writing " + part);
    }
    if (std::rename(part.c_str(), out_path.c_str()) != 0) {
      std::remove(part.c_str());
      throw std::runtime_error("cannot move the drawing into place at " + out_path);
    }
    std::printf("drew %s: n = %lld, nnz = %lld, %lld x %lld blocks\n", matrix_display_name(spec).c_str(),
                static_cast<long long>(n), static_cast<long long>(a.nnz()), static_cast<long long>(grid),
                static_cast<long long>(grid));
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "matrix_preview: %s\n", e.what());
    return 1;
  }
}
