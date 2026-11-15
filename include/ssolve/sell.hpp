// SELL-C-σ storage (Kreutzer et al. 2014, docs/papers/01_Kreutzer2014_SELL-C-sigma.pdf).
//
// Rows are sorted by length, longest first, inside each window of σ rows, then cut into chunks of C
// rows. Each chunk is as wide as its longest row and stored column by column, so the C rows of a chunk
// can be processed together by vector instructions. Shorter rows are padded with zeros.
#pragma once

#include <cstdint>

#include "ssolve/aligned_array.hpp"
#include "ssolve/matrix.hpp"

namespace ssolve {

constexpr int kSellChunk = 8;      // C: fits AVX2 (two registers) and AVX-512 (one) without tuning
constexpr Index kSellSigma = 256;  // σ: the paper's sorting window in its Table 2.2

struct SellMatrix {
  Index n_rows = 0;
  Index n_cols = 0;
  int chunk = kSellChunk;                  // C
  Index sigma = kSellSigma;                // σ
  std::int64_t nnz = 0;                    // real entries, without padding
  AlignedArray<std::int64_t> chunk_start;  // cs in the paper; n_chunks + 1 entries
  AlignedArray<Index> chunk_width;         // cl in the paper; the longest row in each chunk
  AlignedArray<Index> row_of;              // n_chunks * C slots: the original row, or -1 past n
  AlignedArray<Index> col;                 // chunk_start[n_chunks] entries
  AlignedArray<double> val;

  // Entries stored, padding included.
  std::int64_t stored() const { return chunk_start.size() == 0 ? 0 : chunk_start[chunk_start.size() - 1]; }
};

// Entry j of slot r in chunk c is at chunk_start[c] + j * C + r. Padding holds the value 0 at the row's
// last column, or at column 0 for an empty row. C must be 4, 8, 16 or 32, and σ 1 or a multiple of C.
SellMatrix sell_from_csr(const CsrMatrix& a, int chunk = kSellChunk, Index sigma = kSellSigma);

}  // namespace ssolve
