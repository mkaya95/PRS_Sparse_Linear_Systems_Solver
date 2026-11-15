#include "ssolve/sell.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace ssolve {

SellMatrix sell_from_csr(const CsrMatrix& a, int chunk, Index sigma) {
  if (chunk != 4 && chunk != 8 && chunk != 16 && chunk != 32) {
    throw std::runtime_error("SELL chunk must be 4, 8, 16 or 32");
  }
  if (sigma < 1 || (sigma != 1 && sigma % chunk != 0)) {
    throw std::runtime_error("SELL sigma must be 1 or a multiple of the chunk");
  }
  const std::int64_t n = a.n_rows;
  const auto length = [&](Index row) { return a.row_ptr[row + 1] - a.row_ptr[row]; };

  // Sort inside each window of σ rows, longest first. A stable sort keeps equal rows in order,
  // so every machine builds the same layout.
  std::vector<Index> order(static_cast<std::size_t>(n));
  std::iota(order.begin(), order.end(), Index{0});
  for (std::int64_t start = 0; start < n; start += sigma) {
    const auto first = order.begin() + start, last = order.begin() + std::min<std::int64_t>(start + sigma, n);
    std::stable_sort(first, last, [&](Index x, Index y) { return length(x) > length(y); });
  }

  const std::int64_t chunks = (n + chunk - 1) / chunk;
  SellMatrix s;
  s.n_rows = a.n_rows;
  s.n_cols = a.n_cols;
  s.chunk = chunk;
  s.sigma = sigma;
  s.nnz = a.nnz();
  s.chunk_start = AlignedArray<std::int64_t>(static_cast<std::size_t>(chunks + 1));
  s.chunk_width = AlignedArray<Index>(static_cast<std::size_t>(chunks));
  s.row_of = AlignedArray<Index>(static_cast<std::size_t>(chunks * chunk));
  s.row_of.fill(-1);
  s.chunk_start[0] = 0;
  for (std::int64_t c = 0; c < chunks; ++c) {
    std::int64_t width = 0;
    for (std::int64_t slot = c * chunk; slot < std::min<std::int64_t>((c + 1) * chunk, n); ++slot) {
      s.row_of[slot] = order[static_cast<std::size_t>(slot)];
      width = std::max(width, length(s.row_of[slot]));
    }
    s.chunk_width[c] = static_cast<Index>(width);
    s.chunk_start[c + 1] = s.chunk_start[c] + width * chunk;
  }

  s.col = AlignedArray<Index>(static_cast<std::size_t>(s.stored()));
  s.val = AlignedArray<double>(static_cast<std::size_t>(s.stored()));
  for (std::int64_t c = 0; c < chunks; ++c) {
    for (int r = 0; r < chunk; ++r) {
      const Index row = s.row_of[c * chunk + r];
      const std::int64_t begin = row >= 0 ? a.row_ptr[row] : 0;
      const std::int64_t count = row >= 0 ? length(row) : 0;
      const Index pad_col = count > 0 ? a.col[begin + count - 1] : 0;
      for (Index j = 0; j < s.chunk_width[c]; ++j) {
        const std::int64_t k = s.chunk_start[c] + static_cast<std::int64_t>(j) * chunk + r;
        s.col[k] = j < count ? a.col[begin + j] : pad_col;
        s.val[k] = j < count ? a.val[begin + j] : 0.0;
      }
    }
  }
  return s;
}

}  // namespace ssolve
