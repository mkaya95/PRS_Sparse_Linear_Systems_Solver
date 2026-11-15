// Reverse Cuthill-McKee against SciPy's answers, and the symmetric permutation built from an order.
#include <cstdint>
#include <utility>
#include <vector>

#include "ssolve/generators.hpp"
#include "ssolve/reorder.hpp"
#include "ssolve/verify.hpp"
#include "test_util.hpp"

using namespace ssolve;

namespace {

// A symmetric matrix with a full diagonal and the given edges. Entry (i, j) is 1 + i + j off the
// diagonal and 100 + i on it, so every entry says where it came from.
CsrMatrix graph(Index n, const std::vector<std::pair<Index, Index>>& edges) {
  Coo coo;
  coo.n_rows = n;
  coo.n_cols = n;
  for (Index i = 0; i < n; ++i) coo.add(i, i, 100.0 + i);
  for (const auto& [a, b] : edges) {
    coo.add(a, b, 1.0 + a + b);
    coo.add(b, a, 1.0 + a + b);
  }
  return csr_from_coo(coo);
}

void check_order(const std::vector<Index>& order, const std::vector<Index>& expected) {
  CHECK_EQ(order.size(), expected.size());
  for (std::size_t k = 0; k < expected.size() && k < order.size(); ++k) CHECK_EQ(order[k], expected[k]);
}

}  // namespace

// The expected orders are scipy.sparse.csgraph.reverse_cuthill_mckee(A, symmetric_mode=True), SciPy 1.17.1.

TEST(rcm_matches_scipy_on_a_small_graph) {
  const CsrMatrix a = graph(8, {{0, 4}, {0, 6}, {1, 2}, {1, 5}, {2, 7}, {3, 6}, {4, 5}, {5, 7}});
  check_order(reverse_cuthill_mckee(a), {2, 7, 1, 5, 4, 0, 6, 3});
}

TEST(rcm_matches_scipy_on_a_disconnected_graph) {
  const CsrMatrix a = graph(6, {{0, 3}, {1, 4}, {3, 4}});
  check_order(reverse_cuthill_mckee(a), {1, 4, 3, 0, 5, 2});
}

TEST(rcm_matches_scipy_where_a_level_is_sorted_by_degree) {
  const CsrMatrix a = graph(7, {{0, 1}, {0, 2}, {0, 3}, {3, 4}, {4, 5}, {2, 5}, {1, 6}, {2, 6}});
  check_order(reverse_cuthill_mckee(a), {4, 5, 3, 2, 0, 6, 1});
}

TEST(rcm_matches_scipy_on_a_grid) {
  const CsrMatrix a = make_stencil(Stencil::Poisson2D, 4);
  check_order(reverse_cuthill_mckee(a), {15, 14, 11, 13, 10, 7, 12, 9, 6, 3, 8, 5, 2, 4, 1, 0});
}

TEST(rcm_of_an_empty_matrix_is_empty) {
  CHECK_EQ(reverse_cuthill_mckee(graph(0, {})).size(), std::size_t{0});
}

TEST(permuting_moves_every_entry_and_keeps_rows_sorted) {
  const CsrMatrix a = graph(8, {{0, 4}, {0, 6}, {1, 2}, {1, 5}, {2, 7}, {3, 6}, {4, 5}, {5, 7}});
  const std::vector<Index> order = reverse_cuthill_mckee(a);
  const CsrMatrix p = permute_symmetric(a, order);
  CHECK_EQ(p.nnz(), a.nnz());
  for (Index k = 0; k < p.n_rows; ++k) {
    for (std::int64_t at = p.row_ptr[k]; at < p.row_ptr[k + 1]; ++at) {
      if (at > p.row_ptr[k]) CHECK(p.col[at] > p.col[at - 1]);
      const Index i = order[static_cast<std::size_t>(k)], j = order[static_cast<std::size_t>(p.col[at])];
      CHECK_EQ(p.val[at], i == j ? 100.0 + i : 1.0 + i + j);
    }
  }
}

TEST(a_product_with_the_permuted_matrix_is_the_permuted_product) {
  const CsrMatrix a = make_stencil(Stencil::Point27, 5);
  const std::size_t n = static_cast<std::size_t>(a.n_rows);
  const std::vector<Index> order = reverse_cuthill_mckee(a);
  const CsrMatrix p = permute_symmetric(a, order);
  std::vector<double> x(n), y(n), px(n), py(n), back(n);
  for (std::size_t i = 0; i < n; ++i) x[i] = 0.5 + static_cast<double>(i % 13);
  spmv_reference(a, x.data(), y.data());
  to_new_order(order, x.data(), px.data());
  spmv_reference(p, px.data(), py.data());
  to_original_order(order, py.data(), back.data());
  for (std::size_t i = 0; i < n; ++i) CHECK_NEAR(back[i], y[i], 1e-12);
}

TEST(rcm_narrows_a_scattered_matrix) {
  // A path graph whose nodes are numbered far apart: bandwidth 1 once reordered.
  const Index n = 101;
  std::vector<std::pair<Index, Index>> edges;
  for (Index i = 0; i + 1 < n; ++i) edges.push_back({(i * 37) % n, ((i + 1) * 37) % n});
  const CsrMatrix a = graph(n, edges);
  CHECK(bandwidth(a) > 30);
  CHECK_EQ(bandwidth(permute_symmetric(a, reverse_cuthill_mckee(a))), Index{1});
}

TEST(a_bad_order_is_refused) {
  const CsrMatrix a = graph(3, {{0, 1}});
  CHECK_THROWS_WITH(permute_symmetric(a, {0, 1}), "every row once");
  CHECK_THROWS_WITH(permute_symmetric(a, {0, 1, 1}), "every row once");
  CHECK_THROWS_WITH(permute_symmetric(a, {0, 1, 3}), "every row once");
}

TEST_MAIN()
