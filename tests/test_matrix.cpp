#include <cmath>
#include <random>
#include <vector>

#include "ssolve/matrix.hpp"
#include "ssolve/verify.hpp"
#include "test_util.hpp"

using namespace ssolve;

namespace {

// The worked example from the team's guide: rows 10 0 2 0 / 3 9 0 0 / 0 7 8 7 / 0 0 0 5.
Coo guide_coo() {
  Coo coo;
  coo.n_rows = 4;
  coo.n_cols = 4;
  coo.add(0, 0, 10);
  coo.add(0, 2, 2);
  coo.add(1, 0, 3);
  coo.add(1, 1, 9);
  coo.add(2, 1, 7);
  coo.add(2, 2, 8);
  coo.add(2, 3, 7);
  coo.add(3, 3, 5);
  return coo;
}

template <class Array, class Expected>
void check_array(const Array& actual, const Expected& expected) {
  CHECK_EQ(actual.size(), expected.size());
  for (std::size_t i = 0; i < expected.size() && i < actual.size(); ++i) {
    CHECK_EQ(static_cast<double>(actual[i]), static_cast<double>(expected[i]));
  }
}

}  // namespace

TEST(guide_matrix_has_the_expected_csr_arrays) {
  const CsrMatrix a = csr_from_coo(guide_coo());
  CHECK_EQ(a.n_rows, 4);
  CHECK_EQ(a.n_cols, 4);
  CHECK_EQ(a.nnz(), std::int64_t{8});
  check_array(a.row_ptr, std::vector<int>{0, 2, 4, 7, 8});
  check_array(a.col, std::vector<int>{0, 2, 0, 1, 1, 2, 3, 3});
  check_array(a.val, std::vector<double>{10, 2, 3, 9, 7, 8, 7, 5});
}

TEST(guide_matrix_times_vector_matches_hand_computation) {
  const CsrMatrix a = csr_from_coo(guide_coo());
  const double x[] = {1, 2, 3, 4};
  double y[4];
  spmv_reference(a, x, y);
  CHECK_EQ(y[0], 16.0);
  CHECK_EQ(y[1], 21.0);
  CHECK_EQ(y[2], 66.0);
  CHECK_EQ(y[3], 20.0);
}

TEST(random_matrix_matches_a_dense_product) {
  const int n = 200;
  std::mt19937_64 rng(12345);
  std::uniform_real_distribution<double> value(-1.0, 1.0);
  std::uniform_real_distribution<double> coin(0.0, 1.0);

  std::vector<double> dense(static_cast<std::size_t>(n) * n, 0.0);
  Coo coo;
  coo.n_rows = n;
  coo.n_cols = n;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      if (coin(rng) < 0.05) {
        const double v = value(rng);
        coo.add(i, j, v);
        dense[static_cast<std::size_t>(i) * n + j] = v;
      }
    }
  }

  std::vector<double> x(n);
  for (double& v : x) v = value(rng);
  std::vector<double> y(n);
  spmv_reference(csr_from_coo(coo), x.data(), y.data());

  for (int i = 0; i < n; ++i) {
    double expected = 0.0;
    for (int j = 0; j < n; ++j) expected += dense[static_cast<std::size_t>(i) * n + j] * x[j];
    CHECK_NEAR(y[i], expected, 1e-13);
  }
}

TEST(empty_row_gives_zero) {
  Coo coo;
  coo.n_rows = 3;
  coo.n_cols = 3;
  coo.add(0, 0, 2);
  coo.add(2, 2, 3);
  const double x[] = {1, 1, 1};
  double y[] = {9, 9, 9};
  spmv_reference(csr_from_coo(coo), x, y);
  CHECK_EQ(y[0], 2.0);
  CHECK_EQ(y[1], 0.0);
  CHECK_EQ(y[2], 3.0);
}

TEST(matrix_with_no_entries_works) {
  Coo coo;
  coo.n_rows = 2;
  coo.n_cols = 2;
  const CsrMatrix a = csr_from_coo(coo);
  CHECK_EQ(a.nnz(), std::int64_t{0});
  const double x[] = {1, 1};
  double y[] = {9, 9};
  spmv_reference(a, x, y);
  CHECK_EQ(y[0], 0.0);
  CHECK_EQ(y[1], 0.0);
}

TEST(unsorted_input_columns_come_out_sorted) {
  Coo coo;
  coo.n_rows = 2;
  coo.n_cols = 4;
  coo.add(1, 3, 4);
  coo.add(0, 2, 2);
  coo.add(1, 0, 3);
  coo.add(0, 1, 1);
  const CsrMatrix a = csr_from_coo(coo);
  check_array(a.col, std::vector<int>{1, 2, 0, 3});
  check_array(a.val, std::vector<double>{1, 2, 3, 4});
}

TEST(duplicate_entries_are_summed) {
  Coo coo;
  coo.n_rows = 1;
  coo.n_cols = 2;
  coo.add(0, 1, 1.5);
  coo.add(0, 0, 1);
  coo.add(0, 1, 2.5);
  const CsrMatrix a = csr_from_coo(coo);
  CHECK_EQ(a.nnz(), std::int64_t{2});
  CHECK_EQ(a.val.size(), std::size_t{2});
  CHECK_EQ(a.val[1], 4.0);
}

TEST(explicit_stored_zero_is_kept) {
  Coo coo;
  coo.n_rows = 1;
  coo.n_cols = 2;
  coo.add(0, 0, 0.0);
  coo.add(0, 1, 1.0);
  CHECK_EQ(csr_from_coo(coo).nnz(), std::int64_t{2});
}

TEST(out_of_range_indices_are_rejected) {
  Coo bad_row;
  bad_row.n_rows = 2;
  bad_row.n_cols = 2;
  bad_row.add(2, 0, 1);
  CHECK_THROWS_WITH(csr_from_coo(bad_row), "row 2");

  Coo bad_col;
  bad_col.n_rows = 2;
  bad_col.n_cols = 2;
  bad_col.add(0, -1, 1);
  CHECK_THROWS_WITH(csr_from_coo(bad_col), "column -1");
}

TEST(mismatched_coo_arrays_are_rejected) {
  Coo coo;
  coo.n_rows = 1;
  coo.n_cols = 1;
  coo.row.push_back(0);
  CHECK_THROWS_WITH(csr_from_coo(coo), "same length");
}

TEST(clone_is_an_independent_copy) {
  const CsrMatrix a = csr_from_coo(guide_coo());
  CsrMatrix b = clone(a);
  b.val[0] = -1.0;
  CHECK_EQ(a.val[0], 10.0);
  CHECK_EQ(b.nnz(), a.nnz());
  CHECK_EQ(b.col[7], a.col[7]);
  CHECK_EQ(b.row_ptr[4], a.row_ptr[4]);
}

TEST(narrowing_keeps_content_and_moves_the_big_arrays) {
  CsrMatrix a = csr_from_coo(guide_coo());
  const Index* const col = a.col.data();
  const double* const val = a.val.data();
  CHECK(fits_row_ptr32(a));
  const CsrMatrix32 narrow = narrow_row_ptr(std::move(a));
  CHECK(narrow.col.data() == col);
  CHECK(narrow.val.data() == val);
  CHECK_EQ(narrow.nnz(), std::int64_t{8});
  check_array(narrow.row_ptr, std::vector<int>{0, 2, 4, 7, 8});
}

TEST(narrowing_refuses_more_than_2_to_the_31_nonzeros) {
  // Only row_ptr is inspected, so a fake end pointer stands in for a huge matrix.
  CsrMatrix a;
  a.n_rows = 1;
  a.n_cols = 1;
  a.row_ptr = AlignedArray<std::int64_t>(2);
  a.row_ptr[0] = 0;
  a.row_ptr[1] = std::int64_t{1} << 31;
  CHECK(!fits_row_ptr32(a));
  CHECK_THROWS_WITH(narrow_row_ptr(std::move(a)), "2^31");
}

TEST(symmetry_defect_separates_symmetric_from_nonsymmetric) {
  Coo sym;
  sym.n_rows = 3;
  sym.n_cols = 3;
  sym.add(0, 0, 4);
  sym.add(0, 1, -1);
  sym.add(1, 0, -1);
  sym.add(1, 1, 4);
  sym.add(1, 2, -2);
  sym.add(2, 1, -2);
  sym.add(2, 2, 4);
  CHECK(symmetry_defect(csr_from_coo(sym)) < 1e-12);
  CHECK(symmetry_defect(csr_from_coo(guide_coo())) > 1e-3);
}

TEST(symmetry_defect_rejects_a_rectangular_matrix) {
  Coo coo;
  coo.n_rows = 1;
  coo.n_cols = 2;
  CHECK_THROWS_WITH(symmetry_defect(csr_from_coo(coo)), "square");
}

TEST(true_residual_matches_hand_computation) {
  const CsrMatrix a = csr_from_coo(guide_coo());
  const double x[] = {1, 2, 3, 4};
  const double exact_b[] = {16, 21, 66, 20};
  CHECK_EQ(true_relative_residual(a, exact_b, x), 0.0);
  const double off_by_one[] = {17, 21, 66, 20};
  CHECK_NEAR(true_relative_residual(a, off_by_one, x), 1.0 / std::sqrt(5486.0), 1e-15);
}

TEST(true_residual_is_zero_for_zero_rhs_and_zero_solution) {
  const CsrMatrix a = csr_from_coo(guide_coo());
  const double zeros[] = {0, 0, 0, 0};
  CHECK_EQ(true_relative_residual(a, zeros, zeros), 0.0);
}

TEST(true_residual_survives_extreme_magnitudes) {
  Coo coo;
  coo.n_rows = 2;
  coo.n_cols = 2;
  coo.add(0, 0, 1);
  coo.add(1, 1, 1);
  const CsrMatrix identity = csr_from_coo(coo);
  const double b[] = {3e-200, 4e-200};
  const double x[] = {3e-200, 3e-200};
  // Squaring these underflows, so a naive norm would report 0/0.
  CHECK_NEAR(true_relative_residual(identity, b, x), 0.2, 1e-14);
}

TEST(true_residual_reports_nan_for_a_nan_solution) {
  const CsrMatrix a = csr_from_coo(guide_coo());
  const double x[] = {1, std::nan(""), 3, 4};
  const double b[] = {16, 21, 66, 20};
  CHECK(std::isnan(true_relative_residual(a, b, x)));
}

TEST_MAIN()
