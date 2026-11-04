#include "ssolve/mtx_reader.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace ssolve {

namespace {

enum class Field { Real, Pattern };
enum class Structure { General, Symmetric, SkewSymmetric };

bool is_blank(char c) { return c == ' ' || c == '\t' || c == '\r'; }

std::string lowercase(std::string_view text) {
  std::string out(text);
  std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) { return std::tolower(c); });
  return out;
}

// Walks a text buffer one line at a time and formats errors with the line number.
class LineReader {
 public:
  LineReader(std::string_view text, const std::string& name) : text_(text), name_(name) {}

  // Returns false at end of input.
  bool next(std::string_view& line) {
    if (position_ >= text_.size()) return false;
    const std::size_t newline = text_.find('\n', position_);
    const std::size_t end = newline == std::string_view::npos ? text_.size() : newline;
    line = text_.substr(position_, end - position_);
    position_ = end + 1;
    ++line_number_;
    return true;
  }

  // Skips blank lines and % comments.
  bool next_data(std::string_view& line) {
    while (next(line)) {
      std::size_t i = 0;
      while (i < line.size() && is_blank(line[i])) ++i;
      if (i < line.size() && line[i] != '%') return true;
    }
    return false;
  }

  [[noreturn]] void fail(const std::string& message) const {
    throw std::runtime_error(name_ + ":" + std::to_string(line_number_) + ": " + message);
  }

 private:
  std::string_view text_;
  const std::string& name_;
  std::size_t position_ = 0;
  std::size_t line_number_ = 0;
};

// Splits one line into whitespace-separated tokens, without allocating.
class Tokens {
 public:
  explicit Tokens(std::string_view line) : line_(line) {}

  bool next(std::string_view& token) {
    while (position_ < line_.size() && is_blank(line_[position_])) ++position_;
    if (position_ >= line_.size()) return false;
    const std::size_t begin = position_;
    while (position_ < line_.size() && !is_blank(line_[position_])) ++position_;
    token = line_.substr(begin, position_ - begin);
    return true;
  }

 private:
  std::string_view line_;
  std::size_t position_ = 0;
};

bool parse_integer(std::string_view token, std::int64_t& value) {
  std::size_t i = 0;
  const bool negative = !token.empty() && token[0] == '-';
  if (negative) i = 1;
  if (i == token.size() || token.size() - i > 18) return false;
  std::int64_t result = 0;
  for (; i < token.size(); ++i) {
    if (token[i] < '0' || token[i] > '9') return false;
    result = result * 10 + (token[i] - '0');
  }
  value = negative ? -result : result;
  return true;
}

bool parse_real(std::string_view token, double& value) {
  char buffer[128];  // strtod needs a terminated string, and the buffer may not be
  if (token.size() >= sizeof(buffer)) return false;
  std::memcpy(buffer, token.data(), token.size());
  buffer[token.size()] = '\0';
  char* end = nullptr;
  value = std::strtod(buffer, &end);
  return end == buffer + token.size();
}

void parse_banner(LineReader& lines, Field& field, Structure& structure) {
  std::string_view line;
  if (!lines.next(line)) lines.fail("missing %%MatrixMarket banner");
  Tokens tokens(line);
  std::string_view word;
  if (!tokens.next(word) || lowercase(word) != "%%matrixmarket") lines.fail("missing %%MatrixMarket banner");

  std::string_view object_word, format_word, field_word, structure_word;
  if (!tokens.next(object_word) || !tokens.next(format_word) || !tokens.next(field_word) ||
      !tokens.next(structure_word)) {
    lines.fail("incomplete %%MatrixMarket banner");
  }
  if (lowercase(object_word) != "matrix") lines.fail("only matrix objects are supported");
  if (lowercase(format_word) != "coordinate") lines.fail("only coordinate format is supported");

  const std::string field_name = lowercase(field_word);
  if (field_name == "real" || field_name == "double" || field_name == "integer") {
    field = Field::Real;
  } else if (field_name == "pattern") {
    field = Field::Pattern;
  } else {
    lines.fail("unsupported field '" + field_name + "': complex matrices are not supported");
  }

  const std::string structure_name = lowercase(structure_word);
  if (structure_name == "general") {
    structure = Structure::General;
  } else if (structure_name == "symmetric") {
    structure = Structure::Symmetric;
  } else if (structure_name == "skew-symmetric") {
    structure = Structure::SkewSymmetric;
  } else {
    lines.fail("unsupported structure '" + structure_name + "'");
  }
}

void read_entry(LineReader& lines, std::string_view line, Field field, const Coo& shape, std::int64_t& row,
                std::int64_t& col, double& value) {
  Tokens tokens(line);
  std::string_view word;
  if (!tokens.next(word) || !parse_integer(word, row)) lines.fail("expected an integer row index");
  if (!tokens.next(word) || !parse_integer(word, col)) lines.fail("expected an integer column index");
  if (row < 1 || row > shape.n_rows) {
    lines.fail("row " + std::to_string(row) + " is out of range 1.." + std::to_string(shape.n_rows));
  }
  if (col < 1 || col > shape.n_cols) {
    lines.fail("column " + std::to_string(col) + " is out of range 1.." + std::to_string(shape.n_cols));
  }
  value = 1.0;
  if (field == Field::Real) {
    if (!tokens.next(word) || !parse_real(word, value)) lines.fail("expected a numeric value");
  }
  if (tokens.next(word)) lines.fail("unexpected text after the entry");
}

}  // namespace

CsrMatrix parse_matrix_market(std::string_view text, const std::string& name_for_errors) {
  LineReader lines(text, name_for_errors);
  Field field = Field::Real;
  Structure structure = Structure::General;
  parse_banner(lines, field, structure);

  std::string_view line;
  if (!lines.next_data(line)) lines.fail("missing size line");
  std::int64_t rows = 0, cols = 0, declared = 0;
  {
    Tokens tokens(line);
    std::string_view word;
    if (!tokens.next(word) || !parse_integer(word, rows) || !tokens.next(word) || !parse_integer(word, cols) ||
        !tokens.next(word) || !parse_integer(word, declared)) {
      lines.fail("malformed size line");
    }
  }
  if (rows < 0 || cols < 0 || declared < 0) lines.fail("negative value on the size line");
  const std::int64_t index_limit = std::numeric_limits<Index>::max();
  if (rows > index_limit || cols > index_limit) lines.fail("matrix is too large for 32-bit indices");
  if (structure != Structure::General && rows != cols) lines.fail("symmetric matrix must be square");

  Coo coo;
  coo.n_rows = static_cast<Index>(rows);
  coo.n_cols = static_cast<Index>(cols);
  // A bogus header must not cause a huge allocation: an entry takes at least 4 bytes of text.
  const std::size_t room = std::min<std::size_t>(static_cast<std::size_t>(declared), text.size() / 4 + 1);
  const std::size_t reserve = structure == Structure::General ? room : 2 * room;
  coo.row.reserve(reserve);
  coo.col.reserve(reserve);
  coo.val.reserve(reserve);

  std::int64_t found = 0;
  while (lines.next_data(line)) {
    if (found == declared) {
      lines.fail("too many entries: expected " + std::to_string(declared));
    }
    std::int64_t row = 0, col = 0;
    double value = 0.0;
    read_entry(lines, line, field, coo, row, col, value);
    coo.add(static_cast<Index>(row - 1), static_cast<Index>(col - 1), value);
    if (structure != Structure::General && row != col) {
      coo.add(static_cast<Index>(col - 1), static_cast<Index>(row - 1),
              structure == Structure::SkewSymmetric ? -value : value);
    }
    ++found;
  }
  if (found != declared) {
    lines.fail("too few entries: expected " + std::to_string(declared) + ", found " + std::to_string(found));
  }
  return csr_from_coo(coo);
}

CsrMatrix read_matrix_market(const std::string& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) throw std::runtime_error("cannot open matrix file: " + path);
  const std::streamoff size = file.tellg();
  if (size < 0) throw std::runtime_error("cannot read matrix file: " + path);
  std::string text(static_cast<std::size_t>(size), '\0');
  file.seekg(0);
  file.read(text.data(), size);
  if (!file) throw std::runtime_error("cannot read matrix file: " + path);
  return parse_matrix_market(text, path);
}

}  // namespace ssolve
