#include <cmath>
#include <vector>

#include "ssolve/generators.hpp"
#include "ssolve/verify.hpp"
#include "test_util.hpp"

using namespace ssolve;

namespace {

const Stencil kAll[] = {Stencil::Poisson2D, Stencil::Poisson3D, Stencil::Point27};

std::int64_t expected_nnz(Stencil s, std::int64_t N) {
  switch (s) {
    case Stencil::Poisson2D:
      return 5 * N * N - 4 * N;
    case Stencil::Poisson3D:
      return 7 * N * N * N - 6 * N * N;
    case Stencil::Point27:
      return (3 * N - 2) * (3 * N - 2) * (3 * N - 2);
  }
  return -1;
}

double expected_diagonal(Stencil s) {
  switch (s) {
    case Stencil::Poisson2D:
      return 4.0;
    case Stencil::Poisson3D:
      return 6.0;
    case Stencil::Point27:
      return 26.0;
  }
  return 0.0;
}

}  // namespace

TEST(nnz_matches_the_closed_forms) {
  for (Stencil s : kAll) {
    for (Index N : {1, 2, 3, 10}) {
      const CsrMatrix a = make_stencil(s, N);
      CHECK_EQ(a.nnz(), expected_nnz(s, N));
      CHECK_EQ(stencil_nnz(s, N), expected_nnz(s, N));
    }
  }
}

TEST(poisson2d_center_row_has_the_five_point_stencil) {
  const CsrMatrix a = make_stencil(Stencil::Poisson2D, 5);
  const std::int64_t begin = a.row_ptr[12];
  CHECK_EQ(a.row_ptr[13] - begin, std::int64_t{5});
  const Index columns[] = {7, 11, 12, 13, 17};
  const double values[] = {-1, -1, 4, -1, -1};
  for (int k = 0; k < 5; ++k) {
    CHECK_EQ(a.col[begin + k], columns[k]);
    CHECK_EQ(a.val[begin + k], values[k]);
  }
}

TEST(poisson3d_center_row_reaches_all_six_neighbours) {
  const Index N = 3;
  const CsrMatrix a = make_stencil(Stencil::Poisson3D, N);
  const Index center = 13;  // (1, 1, 1)
  const std::int64_t begin = a.row_ptr[center];
  const Index columns[] = {4, 10, 12, 13, 14, 16, 22};
  CHECK_EQ(a.row_ptr[center + 1] - begin, std::int64_t{7});
  for (int k = 0; k < 7; ++k) CHECK_EQ(a.col[begin + k], columns[k]);
}

TEST(all_stencils_are_symmetric_sorted_and_diagonally_dominant) {
  for (Stencil s : kAll) {
    const CsrMatrix a = make_stencil(s, 6);
    CHECK(symmetry_defect(a) < 1e-13);
    for (Index i = 0; i < a.n_rows; ++i) {
      double diagonal = 0.0, off_diagonal = 0.0;
      for (std::int64_t k = a.row_ptr[i]; k < a.row_ptr[i + 1]; ++k) {
        if (k > a.row_ptr[i]) CHECK(a.col[k - 1] < a.col[k]);
        if (a.col[k] == i) {
          diagonal = a.val[k];
        } else {
          off_diagonal += std::fabs(a.val[k]);
          CHECK_EQ(a.val[k], -1.0);
        }
      }
      CHECK_EQ(diagonal, expected_diagonal(s));
      CHECK(diagonal >= off_diagonal);
    }
  }
}

TEST(size_one_grid_is_a_single_diagonal_entry) {
  const CsrMatrix a = make_stencil(Stencil::Poisson3D, 1);
  CHECK_EQ(a.n_rows, 1);
  CHECK_EQ(a.nnz(), std::int64_t{1});
  CHECK_EQ(a.val[0], 6.0);
}

TEST(invalid_grid_sizes_throw) {
  CHECK_THROWS_WITH(make_stencil(Stencil::Poisson2D, 0), "positive");
  CHECK_THROWS_WITH(make_stencil(Stencil::Poisson3D, -3), "positive");
  CHECK_THROWS_WITH(make_stencil(Stencil::Poisson3D, 2000), "too large");
  CHECK_THROWS_WITH(make_stencil(Stencil::Poisson2D, 50000), "too large");
}

TEST(target_bytes_picks_the_smallest_sufficient_grid) {
  for (Stencil s : kAll) {
    for (std::int64_t bytes : {std::int64_t{1000}, std::int64_t{84000000}, std::int64_t{1} << 30}) {
      const Index N = n_for_target_bytes(s, bytes);
      CHECK(12 * stencil_nnz(s, N) >= bytes);
      if (N > 1) CHECK(12 * stencil_nnz(s, N - 1) < bytes);
    }
  }
}

TEST(target_bytes_of_zero_gives_the_smallest_grid) { CHECK_EQ(n_for_target_bytes(Stencil::Poisson3D, 0), 1); }

TEST(unit_solution_rhs_equals_row_sums) {
  const CsrMatrix a = make_stencil(Stencil::Poisson2D, 3);
  const AlignedArray<double> b = rhs_for_unit_solution(a);
  CHECK_EQ(b[0], 2.0);  // corner: 4 - 2
  CHECK_EQ(b[1], 1.0);  // edge: 4 - 3
  CHECK_EQ(b[4], 0.0);  // center: 4 - 4
}

TEST_MAIN()
