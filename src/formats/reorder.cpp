#include "ssolve/reorder.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <stdexcept>

namespace ssolve {

namespace {

// SciPy's _node_degrees: the row length, plus one more if the row holds its diagonal.
std::vector<std::int64_t> node_degrees(const Index* ind, const std::int64_t* ptr, Index num_rows) {
  std::vector<std::int64_t> degree(static_cast<std::size_t>(num_rows), 0);
  for (Index ii = 0; ii < num_rows; ii++) {
    degree[ii] = ptr[ii + 1] - ptr[ii];
    for (std::int64_t jj = ptr[ii]; jj < ptr[ii + 1]; jj++) {
      if (ind[jj] == ii) {
        // add one if the diagonal is in row ii
        degree[ii] += 1;
        break;
      }
    }
  }
  return degree;
}

}  // namespace

// SciPy's _reverse_cuthill_mckee, with its names. NumPy's argsort becomes a stable sort, so rows of
// equal degree keep their order on every machine.
std::vector<Index> reverse_cuthill_mckee(const CsrMatrix& a) {
  if (a.n_rows != a.n_cols) throw std::runtime_error("reverse Cuthill-McKee needs a square matrix");
  const Index num_rows = a.n_rows;
  const Index* const ind = a.col.data();
  const std::int64_t* const ptr = a.row_ptr.data();
  const std::size_t un = static_cast<std::size_t>(num_rows);

  std::int64_t N = 0, N_old, level_start, level_end, temp;
  std::int64_t level_len;
  std::vector<Index> order(un, 0);
  const std::vector<std::int64_t> degree = node_degrees(ind, ptr, num_rows);
  std::vector<Index> inds(un);  // np.argsort(degree)
  std::iota(inds.begin(), inds.end(), Index{0});
  std::stable_sort(inds.begin(), inds.end(), [&](Index x, Index y) { return degree[x] < degree[y]; });
  std::vector<Index> rev_inds(un);  // np.argsort(inds)
  for (std::size_t k = 0; k < un; ++k) rev_inds[static_cast<std::size_t>(inds[k])] = static_cast<Index>(k);
  const std::int64_t max_degree = un == 0 ? 0 : *std::max_element(degree.begin(), degree.end());
  std::vector<std::int64_t> temp_degrees(static_cast<std::size_t>(max_degree), 0);
  Index i, j, seed, temp2;

  // loop over zz takes into account possible disconnected graph.
  for (Index zz = 0; zz < num_rows; zz++) {
    if (inds[zz] != -1) {  // Do BFS with seed=inds[zz]
      seed = inds[zz];
      order[N] = seed;
      N += 1;
      inds[rev_inds[seed]] = -1;
      level_start = N - 1;
      level_end = N;

      while (level_start < level_end) {
        for (std::int64_t ii = level_start; ii < level_end; ii++) {
          i = order[ii];
          N_old = N;

          // add unvisited neighbors
          for (std::int64_t jj = ptr[i]; jj < ptr[i + 1]; jj++) {
            // j is node number connected to i
            j = ind[jj];
            if (inds[rev_inds[j]] != -1) {
              inds[rev_inds[j]] = -1;
              order[N] = j;
              N += 1;
            }
          }

          // Add values to temp_degrees array for insertion sort
          level_len = 0;
          for (std::int64_t kk = N_old; kk < N; kk++) {
            temp_degrees[level_len] = degree[order[kk]];
            level_len += 1;
          }

          // Do insertion sort for nodes from lowest to highest degree
          for (std::int64_t kk = 1; kk < level_len; kk++) {
            temp = temp_degrees[kk];
            temp2 = order[N_old + kk];
            std::int64_t ll = kk;
            while ((ll > 0) && (temp < temp_degrees[ll - 1])) {
              temp_degrees[ll] = temp_degrees[ll - 1];
              order[N_old + ll] = order[N_old + ll - 1];
              ll -= 1;
            }
            temp_degrees[ll] = temp;
            order[N_old + ll] = temp2;
          }
        }

        // set next level start and end ranges
        level_start = level_end;
        level_end = N;
      }
    }

    if (N == num_rows) break;
  }

  // return reversed order for RCM ordering
  std::reverse(order.begin(), order.end());
  return order;
}

CsrMatrix permute_symmetric(const CsrMatrix& a, const std::vector<Index>& order) {
  if (a.n_rows != a.n_cols) throw std::runtime_error("reordering needs a square matrix");
  const std::size_t n = static_cast<std::size_t>(a.n_rows);
  if (order.size() != n) throw std::runtime_error("the order must list every row once");
  std::vector<Index> new_of(n, -1);  // the new index of each old row
  for (std::size_t k = 0; k < n; ++k) {
    const Index old = order[k];
    if (old < 0 || static_cast<std::size_t>(old) >= n || new_of[static_cast<std::size_t>(old)] != -1) {
      throw std::runtime_error("the order must list every row once");
    }
    new_of[static_cast<std::size_t>(old)] = static_cast<Index>(k);
  }

  CsrMatrix p;
  p.n_rows = a.n_rows;
  p.n_cols = a.n_cols;
  p.row_ptr = AlignedArray<std::int64_t>(n + 1);
  p.col = AlignedArray<Index>(static_cast<std::size_t>(a.nnz()));
  p.val = AlignedArray<double>(static_cast<std::size_t>(a.nnz()));
  p.row_ptr[0] = 0;
  for (std::size_t k = 0; k < n; ++k) p.row_ptr[k + 1] = p.row_ptr[k] + a.row_ptr[order[k] + 1] - a.row_ptr[order[k]];

  std::vector<std::int64_t> by_column;  // one row's entries, to be sorted by their new column
  for (std::size_t k = 0; k < n; ++k) {
    const std::int64_t begin = a.row_ptr[order[k]], end = a.row_ptr[order[k] + 1];
    by_column.resize(static_cast<std::size_t>(end - begin));
    std::iota(by_column.begin(), by_column.end(), begin);
    std::sort(by_column.begin(), by_column.end(),
              [&](std::int64_t x, std::int64_t y) { return new_of[a.col[x]] < new_of[a.col[y]]; });
    std::int64_t at = p.row_ptr[k];
    for (const std::int64_t source : by_column) {
      p.col[at] = new_of[a.col[source]];
      p.val[at++] = a.val[source];
    }
  }
  return p;
}

void to_new_order(const std::vector<Index>& order, const double* from, double* to) {
  for (std::size_t k = 0; k < order.size(); ++k) to[k] = from[order[k]];
}

void to_original_order(const std::vector<Index>& order, const double* from, double* to) {
  for (std::size_t k = 0; k < order.size(); ++k) to[order[k]] = from[k];
}

Index bandwidth(const CsrMatrix& a) {
  Index widest = 0;
  for (Index i = 0; i < a.n_rows; ++i) {
    for (std::int64_t k = a.row_ptr[i]; k < a.row_ptr[i + 1]; ++k) widest = std::max(widest, std::abs(a.col[k] - i));
  }
  return widest;
}

}  // namespace ssolve
