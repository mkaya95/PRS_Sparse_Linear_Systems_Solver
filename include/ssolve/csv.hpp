// Append-only CSV output, so benchmark runs can add to one results file.
#pragma once

#include <fstream>
#include <string>
#include <vector>

namespace ssolve {

class CsvWriter {
 public:
  // Writes the header only if the file is new or empty. Throws if it cannot be opened.
  CsvWriter(const std::string& path, std::vector<std::string> columns);

  // Throws if the number of values differs from the number of columns.
  void write_row(const std::vector<std::string>& values);

 private:
  std::ofstream out_;
  std::size_t column_count_;
};

std::string csv_escape(const std::string& text);

// Shortest decimal form that parses back to exactly the same double.
std::string fmt_double(double value);

}  // namespace ssolve
