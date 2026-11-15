#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "ssolve/generators.hpp"
#include "ssolve/sell.hpp"
#include "ssolve/verify.hpp"
#include "test_util.hpp"

using namespace ssolve;

namespace {

const std::pair<int, Index> kSettings[] = {{4, 1}, {8, 1}, {8, 8}, {8, 256}, {32, 64}};

// Rows of length 0, 1, 7 and 2 mixed together, with one explicitly stored zero, which the reader keeps.
CsrMatrix mixed_matrix() {
  const std::vector<int> lengths = {0, 1, 7, 2, 3, 0, 5, 1, 2, 7, 4};
  Coo coo;
  coo.n_rows = 11;
  coo.n_cols = 11;
  for (Index i = 0; i < 11; ++i) {
    for (int k = 0; k < lengths[static_cast<std::size_t>(i)]; ++k) {
      const Index j = (i + 2 * k) % 11;
      coo.add(i, j, i == 3 && k == 1 ? 0.0 : 1.0 + 0.5 * i - 0.25 * k);
    }
  }
  return csr_from_coo(coo);
}

std::vector<CsrMatrix> test_matrices() {
  std::vector<CsrMatrix> all;
  all.push_back(make_stencil(Stencil::Point27, 4));
  all.push_back(mixed_matrix());
  return all;
}

// The obvious loop over chunks, widths and slots: one sum per slot, in storage order.
void sell_spmv_plain(const SellMatrix& s, const std::vector<double>& x, std::vector<double>& y) {
  const std::int64_t chunks = static_cast<std::int64_t>(s.chunk_width.size());
  for (std::int64_t c = 0; c < chunks; ++c) {
    for (int r = 0; r < s.chunk; ++r) {
      double sum = 0.0;
      for (Index j = 0; j < s.chunk_width[c]; ++j) {
        const std::int64_t k = s.chunk_start[c] + static_cast<std::int64_t>(j) * s.chunk + r;
        sum += s.val[k] * x[s.col[k]];
      }
      const Index row = s.row_of[c * s.chunk + r];
      if (row >= 0) y[row] = sum;
    }
  }
}

std::vector<double> ramp(std::size_t n) {
  std::vector<double> v(n);
  for (std::size_t i = 0; i < n; ++i) v[i] = 1.0 + 0.125 * static_cast<double>(i % 13) - 0.5 * static_cast<double>(i % 3);
  return v;
}

std::string setting(int chunk, Index sigma) { return "C=" + std::to_string(chunk) + " sigma=" + std::to_string(sigma); }

}  // namespace

TEST(every_entry_is_stored_once_in_its_row) {
  for (const CsrMatrix& a : test_matrices()) {
    for (const auto& [chunk, sigma] : kSettings) {
      const SellMatrix s = sell_from_csr(a, chunk, sigma);
      CHECK_EQ(s.nnz, a.nnz());
      CHECK_EQ(s.chunk, chunk);
      CHECK_EQ(s.sigma, sigma);
      std::vector<int> seen(static_cast<std::size_t>(a.n_rows), 0);
      const std::int64_t chunks = static_cast<std::int64_t>(s.chunk_width.size());
      CHECK_EQ(chunks, (static_cast<std::int64_t>(a.n_rows) + chunk - 1) / chunk);
      for (std::int64_t c = 0; c < chunks; ++c) {
        for (int r = 0; r < chunk; ++r) {
          const Index row = s.row_of[c * chunk + r];
          if (row < 0) continue;
          ++seen[static_cast<std::size_t>(row)];
          const std::int64_t begin = a.row_ptr[row], length = a.row_ptr[row + 1] - begin;
          CHECK(length <= s.chunk_width[c]);
          for (Index j = 0; j < s.chunk_width[c]; ++j) {
            const std::int64_t k = s.chunk_start[c] + static_cast<std::int64_t>(j) * chunk + r;
            if (j < length) {
              if (s.col[k] != a.col[begin + j] || s.val[k] != a.val[begin + j]) {
                test_util::fail(__FILE__, __LINE__, "row " + std::to_string(row) + " differs, " + setting(chunk, sigma));
              }
            } else {
              const Index pad_col = length > 0 ? a.col[begin + length - 1] : 0;
              if (s.val[k] != 0.0 || s.col[k] != pad_col) {
                test_util::fail(__FILE__, __LINE__, "bad padding in row " + std::to_string(row) + ", " + setting(chunk, sigma));
              }
            }
          }
        }
      }
      for (const int count : seen) CHECK_EQ(count, 1);
      CHECK_EQ(s.stored(), s.chunk_start[chunks]);
    }
  }
}

TEST(spmv_equals_the_reference_bit_for_bit) {
  for (const CsrMatrix& a : test_matrices()) {
    const std::size_t n = static_cast<std::size_t>(a.n_rows);
    const std::vector<double> x = ramp(n);
    std::vector<double> expected(n), y(n, -7.0);
    spmv_reference(a, x.data(), expected.data());
    for (const auto& [chunk, sigma] : kSettings) {
      sell_spmv_plain(sell_from_csr(a, chunk, sigma), x, y);
      for (std::size_t i = 0; i < n; ++i) {
        if (y[i] != expected[i]) test_util::fail(__FILE__, __LINE__, "row " + std::to_string(i) + ", " + setting(chunk, sigma));
      }
    }
  }
}

TEST(rows_are_sorted_longest_first_inside_each_window) {
  Coo coo;
  coo.n_rows = 16;
  coo.n_cols = 16;
  for (Index i = 0; i < 16; ++i) {
    for (Index k = 0; k <= i % 5; ++k) coo.add(i, (i + k) % 16, 1.0);
  }
  const CsrMatrix a = csr_from_coo(coo);
  const SellMatrix s = sell_from_csr(a, 4, 8);
  for (Index window = 0; window < 2; ++window) {
    std::int64_t previous = 1 << 30;
    for (Index slot = window * 8; slot < window * 8 + 8; ++slot) {
      const Index row = s.row_of[slot];
      CHECK(row >= window * 8 && row < window * 8 + 8);
      const std::int64_t length = a.row_ptr[row + 1] - a.row_ptr[row];
      CHECK(length <= previous);
      previous = length;
    }
  }
}

TEST(no_sorting_with_sigma_one_keeps_row_order) {
  const CsrMatrix a = mixed_matrix();
  const SellMatrix s = sell_from_csr(a, 4, 1);
  for (Index k = 0; k < a.n_rows; ++k) CHECK_EQ(s.row_of[k], k);
}

TEST(the_last_chunk_is_padded_past_n) {
  Coo coo;
  coo.n_rows = 5;
  coo.n_cols = 5;
  for (Index i = 0; i < 5; ++i) coo.add(i, i, 2.0);
  const SellMatrix s = sell_from_csr(csr_from_coo(coo), 4, 4);
  CHECK_EQ(s.chunk_width.size(), std::size_t{2});
  CHECK_EQ(s.row_of.size(), std::size_t{8});
  for (std::size_t k = 5; k < 8; ++k) CHECK_EQ(s.row_of[k], Index{-1});
}

TEST(bad_settings_are_refused) {
  const CsrMatrix a = mixed_matrix();
  CHECK_THROWS_WITH(sell_from_csr(a, 6, 1), "chunk must be 4, 8, 16 or 32");
  CHECK_THROWS_WITH(sell_from_csr(a, 8, 12), "multiple of the chunk");
  CHECK_THROWS_WITH(sell_from_csr(a, 8, 0), "multiple of the chunk");
}

TEST_MAIN()
