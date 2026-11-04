#include "ssolve/csv.hpp"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace ssolve {

CsvWriter::CsvWriter(const std::string& path, std::vector<std::string> columns) : column_count_(columns.size()) {
  bool needs_header = true;
  {
    std::ifstream existing(path, std::ios::binary | std::ios::ate);
    if (existing && existing.tellg() > 0) needs_header = false;
  }
  out_.open(path, std::ios::app);
  if (!out_) throw std::runtime_error("cannot open CSV file: " + path);
  if (needs_header) write_row(columns);
}

void CsvWriter::write_row(const std::vector<std::string>& values) {
  if (values.size() != column_count_) {
    throw std::runtime_error("CSV row has " + std::to_string(values.size()) + " values for " +
                             std::to_string(column_count_) + " columns");
  }
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) out_ << ',';
    out_ << csv_escape(values[i]);
  }
  out_ << '\n';
  out_.flush();  // a later crash or out-of-memory kill must not lose finished rows
}

std::string csv_escape(const std::string& text) {
  if (text.find_first_of(",\"\n\r") == std::string::npos) return text;
  std::string out = "\"";
  for (const char c : text) {
    if (c == '"') out += '"';
    out += c;
  }
  out += '"';
  return out;
}

std::string fmt_double(double value) {
  char buffer[64];
  for (int precision = 15; precision <= 17; ++precision) {
    std::snprintf(buffer, sizeof(buffer), "%.*g", precision, value);
    if (std::strtod(buffer, nullptr) == value) break;
  }
  return buffer;
}

}  // namespace ssolve
