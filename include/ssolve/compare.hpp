// How far apart two solution vectors are, as numbers and as a block picture for the browser page.
//
// The n entries are laid row by row into a g x g grid, g = min(64, ceil(sqrt(n))). Entry i goes to
// cell floor(i * g² / n) when n >= g², else to cell i, and each cell keeps the largest |a_i - b_i| in it.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ssolve {

struct VectorDifference {
  std::string a, b;            // solver names; b is "true" for the exact solution
  double max_abs = 0.0;        // max |a_i - b_i|, NaN if any difference is not finite
  double rel_norm = 0.0;       // ‖a - b‖₂ / ‖b‖₂, or ‖a - b‖₂ when b = 0; NaN if not finite
  std::int64_t equal = 0;      // entries with a_i == b_i exactly
  std::vector<double> cells;   // grid * grid; NaN = empty cell, -1 = non-finite difference
};

// min(64, ceil(sqrt(n))), at least 1.
int difference_grid(std::int64_t n);

VectorDifference compare_vectors(const std::string& a_name, const double* a, const std::string& b_name,
                                 const double* b, std::size_t n);

// One solver's recursive ‖r‖ / ‖b‖ before the first iteration and after each one.
struct ResidualHistory {
  std::string solver;
  std::vector<double> residuals;
};

// {"matrix","n","grid","problem","x_scale","comparisons":[{"a","b","max_abs","rel_norm","equal","cells"}],
//  "histories":[{"solver","residuals"}]}, with three significant digits for the differences and four for
// the residuals. A NaN becomes null.
std::string solutions_json(const std::string& matrix, std::int64_t n, const std::string& problem, double x_scale,
                           const std::vector<VectorDifference>& diffs,
                           const std::vector<ResidualHistory>& histories = {});

// A JSON string literal; control characters become spaces.
std::string json_string(const std::string& text);

// A JSON number with the given significant digits, or null when not finite.
std::string json_number(double value, int digits = 17);

// Writes PATH.part, then renames it over PATH, so a stopped run never leaves half a file.
void write_file_atomically(const std::string& path, const std::string& text);

}  // namespace ssolve
