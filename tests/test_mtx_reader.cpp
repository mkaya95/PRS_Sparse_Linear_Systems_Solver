#include <string>
#include <vector>

#include "ssolve/mtx_reader.hpp"
#include "ssolve/verify.hpp"
#include "test_util.hpp"

using namespace ssolve;

namespace {

CsrMatrix parse(const std::string& text) { return parse_matrix_market(text, "test.mtx"); }

double entry(const CsrMatrix& a, Index i, Index j) {
  for (std::int64_t k = a.row_ptr[i]; k < a.row_ptr[i + 1]; ++k) {
    if (a.col[k] == j) return a.val[k];
  }
  return 0.0;
}

}  // namespace

TEST(real_general_converts_to_zero_based_indices) {
  const CsrMatrix a = parse(
      "%%MatrixMarket matrix coordinate real general\n"
      "2 3 3\n"
      "1 1 1.5\n"
      "2 3 -2.5\n"
      "1 2 4\n");
  CHECK_EQ(a.n_rows, 2);
  CHECK_EQ(a.n_cols, 3);
  CHECK_EQ(a.nnz(), std::int64_t{3});
  CHECK_EQ(entry(a, 0, 0), 1.5);
  CHECK_EQ(entry(a, 0, 1), 4.0);
  CHECK_EQ(entry(a, 1, 2), -2.5);
}

TEST(symmetric_file_is_mirrored) {
  const CsrMatrix a = parse(
      "%%MatrixMarket matrix coordinate real symmetric\n"
      "3 3 5\n"
      "1 1 4\n"
      "2 2 5\n"
      "2 1 -1\n"
      "3 1 -2\n"
      "3 2 -3\n");
  CHECK_EQ(a.nnz(), std::int64_t{8});  // 2 diagonal + 2 * 3 off-diagonal
  CHECK_EQ(entry(a, 1, 0), -1.0);
  CHECK_EQ(entry(a, 0, 1), -1.0);
  CHECK_EQ(entry(a, 2, 1), -3.0);
  CHECK_EQ(entry(a, 1, 2), -3.0);
  CHECK(symmetry_defect(a) < 1e-14);
}

TEST(skew_symmetric_mirror_is_negated) {
  const CsrMatrix a = parse(
      "%%MatrixMarket matrix coordinate real skew-symmetric\n"
      "2 2 1\n"
      "2 1 3\n");
  CHECK_EQ(entry(a, 1, 0), 3.0);
  CHECK_EQ(entry(a, 0, 1), -3.0);
}

TEST(pattern_entries_become_one) {
  const CsrMatrix a = parse(
      "%%MatrixMarket matrix coordinate pattern symmetric\n"
      "2 2 2\n"
      "1 1\n"
      "2 1\n");
  CHECK_EQ(entry(a, 0, 0), 1.0);
  CHECK_EQ(entry(a, 0, 1), 1.0);
  CHECK_EQ(entry(a, 1, 0), 1.0);
}

TEST(integer_field_is_read_as_double) {
  const CsrMatrix a = parse(
      "%%MatrixMarket matrix coordinate integer general\n"
      "1 1 1\n"
      "1 1 7\n");
  CHECK_EQ(a.val[0], 7.0);
}

TEST(crlf_line_endings_parse_like_lf) {
  const CsrMatrix a = parse(
      "%%MatrixMarket matrix coordinate real general\r\n"
      "% written on Windows\r\n"
      "2 2 2\r\n"
      "1 1 1.5\r\n"
      "2 2 2.5\r\n");
  CHECK_EQ(a.n_rows, 2);
  CHECK_EQ(entry(a, 1, 1), 2.5);
}

TEST(banner_is_case_insensitive) {
  const CsrMatrix a = parse(
      "%%MATRIXMARKET MATRIX COORDINATE REAL GENERAL\n"
      "1 1 1\n"
      "1 1 2\n");
  CHECK_EQ(a.val[0], 2.0);
}

TEST(comments_and_blank_lines_are_ignored_everywhere) {
  const CsrMatrix a = parse(
      "%%MatrixMarket matrix coordinate real general\n"
      "% first comment\n"
      "\n"
      "%second comment\n"
      "   \n"
      "2 2 2\n"
      "% between entries\n"
      "1 1 1\n"
      "\n"
      "2 2 2\n"
      "\n");
  CHECK_EQ(a.nnz(), std::int64_t{2});
  CHECK_EQ(entry(a, 1, 1), 2.0);
}

TEST(scientific_notation_is_parsed) {
  const CsrMatrix a = parse(
      "%%MatrixMarket matrix coordinate real general\n"
      "1 1 1\n"
      "1 1 1.5e+03\n");
  CHECK_EQ(a.val[0], 1500.0);
}

TEST(duplicate_entries_are_summed) {
  const CsrMatrix a = parse(
      "%%MatrixMarket matrix coordinate real general\n"
      "1 1 2\n"
      "1 1 1\n"
      "1 1 2\n");
  CHECK_EQ(a.nnz(), std::int64_t{1});
  CHECK_EQ(a.val[0], 3.0);
}

TEST(missing_final_newline_is_fine) {
  const CsrMatrix a = parse(
      "%%MatrixMarket matrix coordinate real general\n"
      "1 1 1\n"
      "1 1 9");
  CHECK_EQ(a.val[0], 9.0);
}

TEST(rejects_complex_field) {
  CHECK_THROWS_WITH(parse("%%MatrixMarket matrix coordinate complex general\n1 1 1\n1 1 1 0\n"), "complex");
}

TEST(rejects_array_format) {
  CHECK_THROWS_WITH(parse("%%MatrixMarket matrix array real general\n1 1\n1\n"), "coordinate");
}

TEST(rejects_a_missing_banner) {
  CHECK_THROWS_WITH(parse("1 1 1\n1 1 1\n"), "banner");
  CHECK_THROWS_WITH(parse(""), "banner");
}

TEST(rejects_a_missing_size_line) {
  CHECK_THROWS_WITH(parse("%%MatrixMarket matrix coordinate real general\n% only a comment\n"), "size line");
}

TEST(rejects_row_zero_and_names_the_file_line) {
  CHECK_THROWS_WITH(parse("%%MatrixMarket matrix coordinate real general\n2 2 1\n0 1 1\n"), "test.mtx:3");
}

TEST(rejects_a_row_above_n_and_names_the_file_line) {
  CHECK_THROWS_WITH(parse("%%MatrixMarket matrix coordinate real general\n% c\n2 2 2\n1 1 1\n3 1 1\n"),
                    "test.mtx:5");
}

TEST(rejects_too_few_entries) {
  CHECK_THROWS_WITH(parse("%%MatrixMarket matrix coordinate real general\n2 2 3\n1 1 1\n"), "expected 3");
}

TEST(rejects_too_many_entries) {
  CHECK_THROWS_WITH(parse("%%MatrixMarket matrix coordinate real general\n2 2 1\n1 1 1\n2 2 1\n"), "expected 1");
}

TEST(rejects_a_non_numeric_value) {
  CHECK_THROWS_WITH(parse("%%MatrixMarket matrix coordinate real general\n1 1 1\n1 1 abc\n"), "test.mtx:3");
}

TEST(rejects_a_missing_value) {
  CHECK_THROWS_WITH(parse("%%MatrixMarket matrix coordinate real general\n1 1 1\n1 1\n"), "test.mtx:3");
}

TEST(rejects_symmetric_header_on_a_non_square_matrix) {
  CHECK_THROWS_WITH(parse("%%MatrixMarket matrix coordinate real symmetric\n2 3 1\n1 1 1\n"), "square");
}

TEST(rejects_dimensions_too_large_for_32_bit_indices) {
  CHECK_THROWS_WITH(parse("%%MatrixMarket matrix coordinate real general\n3000000000 1 0\n"), "too large");
}

TEST(reading_a_missing_file_names_the_path) {
  CHECK_THROWS_WITH(read_matrix_market("no/such/file.mtx"), "no/such/file.mtx");
}

TEST_MAIN()
